"""Run genomic representation benchmarks for True DNA checkpoints.

This runner intentionally separates two benchmark classes:

1. Built-in GenomicBenchmarks sequence-classification datasets, downloaded by
   the optional ``genomic-benchmarks`` package.
2. Manifest-provided harder datasets: splice, TFBS, chromatin/accessibility,
   expression, taxonomy/biotype, and variant-effect scoring.

The script does not bundle biological datasets. It defines the adapters and
metrics so the same checkpoint can be evaluated reproducibly once datasets are
available locally.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dna_model.checkpoint_loading import (  # noqa: E402
    config_from_checkpoint,
    load_state_dict_with_report,
    normalize_state_dict_keys,
)
from dna_model.config import (  # noqa: E402
    IGNORE_INDEX,
    MASK_TOKEN_ID,
    SPECIAL_TOKENS_MAX_ID,
)
from dna_model.model import DnaModel  # noqa: E402
from dna_model.tokenizer import (  # noqa: E402
    build_tokenizer,
    normalize_dna_sequence,
    normalize_tokenizer_type,
    read_fasta_records,
)

DEFAULT_GENOMIC_BENCHMARKS = [
    "human_nontata_promoters",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ocr_ensembl",
    "human_ensembl_regulatory",
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
]

BENCHMARK_PRESETS = {
    "genomic_benchmarks": DEFAULT_GENOMIC_BENCHMARKS,
    "motif_short_range": ["human_nontata_promoters"],
    "regulatory_short_range": [
        "human_nontata_promoters",
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "human_ocr_ensembl",
        "human_ensembl_regulatory",
    ],
    "taxonomy_biotype": ["demo_coding_vs_intergenomic_seqs", "demo_human_or_worm"],
    "all_builtin": DEFAULT_GENOMIC_BENCHMARKS,
}

TASK_CATEGORIES = {
    "splice": "short_range_motif",
    "splice_site": "short_range_motif",
    "promoter": "short_range_motif",
    "tss": "short_range_motif",
    "tfbs": "short_range_motif",
    "chipseq": "short_range_motif",
    "enhancer": "long_range_regulatory",
    "ocr": "long_range_regulatory",
    "atac": "long_range_regulatory",
    "chromatin": "long_range_regulatory",
    "accessibility": "long_range_regulatory",
    "regulatory": "long_range_regulatory",
    "expression": "long_range_expression",
    "rna": "long_range_expression",
    "biotype": "evolutionary_taxonomy",
    "taxonomy": "evolutionary_taxonomy",
    "taxon": "evolutionary_taxonomy",
    "species": "evolutionary_taxonomy",
    "variant": "variant_effect",
    "vep": "variant_effect",
    "clinvar": "variant_effect",
}

MANIFEST_SCHEMA_NOTE = {
    "classification": "splice/promoter/TFBS/ATAC-OCR/biotype/taxonomy/species tasks",
    "regression": "gene-expression or quantitative regulatory activity tasks",
    "variant_effect": "ref/alt sequence tables scored by pseudo-log-likelihood delta",
    "genomic_benchmarks": "optional built-in GenomicBenchmarks dataset names",
}

DNA_ALPHABET = "ACGT"


@dataclass
class SplitRecords:
    train: list[tuple[str, Any]]
    test: list[tuple[str, Any]]
    task: str


def _clean_sequence(seq: str) -> str:
    return normalize_dna_sequence(seq)


def _clean_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _infer_category(name: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    lowered = _clean_key(name)
    for token, category in TASK_CATEGORIES.items():
        if token in lowered:
            return category
    return "custom"


def _resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute() and base_dir is not None:
        out = Path(base_dir) / out
    return out


def _as_float_if_possible(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _binary_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower()
    positives = {"1", "true", "positive", "pathogenic", "deleterious", "disease", "case"}
    negatives = {"0", "false", "negative", "benign", "neutral", "control"}
    if text in positives:
        return 1
    if text in negatives:
        return 0
    return None


def _dict_rows_from_path(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    suffix = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".jsonl" or suffix.endswith(".jsonl.gz"):
        open_text = gzip.open if path.suffix.lower() == ".gz" else open
        with open_text(path, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} JSONL row is not an object")
                yield row
        return
    if path.suffix.lower() == ".json" or suffix.endswith(".json.gz"):
        open_text = gzip.open if path.suffix.lower() == ".gz" else open
        with open_text(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("records", payload.get("data"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} JSON must be a list or an object with records/data")
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"{path} JSON record is not an object")
            yield row
        return

    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} or suffix.endswith(".tsv.gz") else ","
    open_text = gzip.open if path.suffix.lower() == ".gz" else open
    with open_text(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        for row in reader:
            yield row


def _row_value(row: Mapping[str, Any], key: str, *, required: bool = True) -> Any:
    if key in row:
        return row[key]
    cleaned = {_clean_key(str(k)): v for k, v in row.items()}
    cleaned_key = _clean_key(key)
    if cleaned_key in cleaned:
        return cleaned[cleaned_key]
    if required:
        raise ValueError(f"Missing required column/key: {key}")
    return None


def _header_field(header: str, keys: set[str]) -> str | None:
    for part in header.replace("|", " ").replace(";", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if _clean_key(key) in keys:
            return value
    return None


def _coerce_label(label: Any, task: str) -> Any | None:
    if task == "regression":
        return _as_float_if_possible(label)
    if label is None:
        return None
    label_text = str(label).strip()
    return label_text if label_text else None


def _finalize_split_records(
    train: list[tuple[str, Any]],
    test: list[tuple[str, Any]],
    fallback: list[tuple[str, Any]],
    *,
    task: str,
    source: str | Path,
) -> SplitRecords:
    if fallback:
        cutoff = max(1, int(round(len(fallback) * 0.8)))
        train.extend(fallback[:cutoff])
        test.extend(fallback[cutoff:] or fallback[:1])
    if not train or not test:
        raise ValueError(f"{source} did not provide non-empty train and test splits")
    return SplitRecords(train=train, test=test, task=task)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _safe_corr(x: Sequence[float], y: Sequence[float], *, spearman: bool = False) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2 or np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return None
    if spearman:
        x_arr = _rankdata(x_arr)
        y_arr = _rankdata(y_arr)
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _read_tabular_records(
    path: str | Path,
    *,
    seq_col: str = "sequence",
    label_col: str = "label",
    split_col: str | None = "split",
    task: str = "classification",
    train_values: set[str] | None = None,
    test_values: set[str] | None = None,
    max_records: int = 0,
) -> SplitRecords:
    train_values = train_values or {"train", "training"}
    test_values = test_values or {"test", "valid", "validation", "val", "eval"}
    train: list[tuple[str, Any]] = []
    test: list[tuple[str, Any]] = []
    fallback: list[tuple[str, Any]] = []

    for row in _dict_rows_from_path(path):
        seq = _clean_sequence(str(_row_value(row, seq_col)))
        if not seq:
            continue
        label = _coerce_label(_row_value(row, label_col), task)
        if label is None:
            continue
        rec = (seq, label)
        split_value = _row_value(row, split_col, required=False) if split_col else None
        if split_value:
            split = str(split_value).strip().lower()
            if split in train_values:
                train.append(rec)
            elif split in test_values:
                test.append(rec)
            else:
                fallback.append(rec)
        else:
            fallback.append(rec)
        if max_records and len(train) + len(test) + len(fallback) >= max_records:
            break

    return _finalize_split_records(train, test, fallback, task=task, source=path)


def _read_fasta_manifest_records(
    entry: Mapping[str, Any],
    *,
    task: str,
    base_dir: str | Path | None,
    max_records: int = 0,
) -> SplitRecords:
    train_values = set(entry.get("train_values", ["train", "training"]))
    test_values = set(entry.get("test_values", ["test", "valid", "validation", "val", "eval"]))
    label_keys = {_clean_key(k) for k in entry.get("label_keys", ["label", "class", "species", "taxon", "biotype"])}
    split_keys = {_clean_key(k) for k in entry.get("split_keys", ["split", "partition", "set"])}
    records_spec = entry.get("files")
    if records_spec is None:
        records_spec = [
            {
                "path": entry["path"],
                "label": entry.get("label"),
                "split": entry.get("split"),
            }
        ]

    train: list[tuple[str, Any]] = []
    test: list[tuple[str, Any]] = []
    fallback: list[tuple[str, Any]] = []

    for file_spec in records_spec:
        path = _resolve_path(file_spec["path"], base_dir)
        default_label = file_spec.get("label", entry.get("label"))
        default_split = file_spec.get("split", entry.get("split"))
        for header, seq, _is_last in read_fasta_records(str(path)):
            cleaned = _clean_sequence(seq)
            if not cleaned:
                continue
            label = _coerce_label(
                _header_field(header, label_keys) or default_label or path.stem,
                task,
            )
            if label is None:
                continue
            rec = (cleaned, label)
            split = str(_header_field(header, split_keys) or default_split or "").strip().lower()
            if split in train_values:
                train.append(rec)
            elif split in test_values:
                test.append(rec)
            else:
                fallback.append(rec)
            if max_records and len(train) + len(test) + len(fallback) >= max_records:
                return _finalize_split_records(train, test, fallback, task=task, source=path)

    return _finalize_split_records(train, test, fallback, task=task, source=entry.get("path", "FASTA manifest"))


def _read_csv_records(*args: Any, **kwargs: Any) -> SplitRecords:
    """Backward-compatible alias for older tests/imports."""
    return _read_tabular_records(*args, **kwargs)


def _read_genomic_benchmarks_dataset(
    dataset: str,
    *,
    version: int = 0,
    root_dir: str | None = None,
    max_records: int = 0,
) -> SplitRecords:
    try:
        from genomic_benchmarks.loc2seq import download_dataset
    except ImportError as exc:
        raise RuntimeError(
            "GenomicBenchmarks support requires optional dependency "
            "`genomic-benchmarks`. Install with `pip install -r requirements-benchmarks.txt`."
        ) from exc

    kwargs: dict[str, Any] = {"version": version}
    if root_dir:
        kwargs["dest_path"] = root_dir
    downloaded = download_dataset(dataset, **kwargs)
    candidate_paths = []
    if downloaded:
        candidate_paths.append(Path(downloaded))
    if root_dir:
        candidate_paths.extend([Path(root_dir) / dataset, Path(root_dir)])
    candidate_paths.extend([Path("datasets") / dataset, Path.home() / ".genomic_benchmarks" / dataset])
    dataset_path = next((p for p in candidate_paths if p.exists()), None)
    if dataset_path is None:
        raise ValueError(
            f"Could not locate downloaded GenomicBenchmarks dataset {dataset!r}; "
            f"checked {[str(p) for p in candidate_paths]}"
        )

    def read_split(split: str) -> list[tuple[str, str]]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            raise ValueError(f"GenomicBenchmarks dataset has no {split!r} split: {split_dir}")
        records: list[tuple[str, str]] = []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            for seq_file in sorted(p for p in class_dir.iterdir() if p.is_file()):
                raw = seq_file.read_text(encoding="utf-8", errors="ignore")
                seq = _clean_sequence("".join(line.strip() for line in raw.splitlines() if not line.startswith(">")))
                if seq:
                    records.append((seq, label))
                    if max_records and len(records) >= max_records:
                        return records
        if not records:
            raise ValueError(f"No sequence files found under {split_dir}")
        return records

    return SplitRecords(train=read_split("train"), test=read_split("test"), task="classification")


def _gc_fraction(seq: str) -> float:
    seq = seq.upper()
    return float(sum(ch in "GC" for ch in seq)) / max(1, len(seq))


def _kmer_index(kmer: str) -> int:
    idx = 0
    for ch in kmer:
        idx = idx * 4 + DNA_ALPHABET.index(ch)
    return idx


def handcrafted_features(sequences: Sequence[str], *, k: int = 3) -> np.ndarray:
    width = 2 + (4**k)
    features = np.zeros((len(sequences), width), dtype=np.float32)
    for row, seq in enumerate(sequences):
        seq = _clean_sequence(seq)
        features[row, 0] = len(seq)
        features[row, 1] = _gc_fraction(seq)
        denom = max(1, len(seq) - k + 1)
        for i in range(max(0, len(seq) - k + 1)):
            kmer = seq[i : i + k]
            if set(kmer) <= set(DNA_ALPHABET):
                features[row, 2 + _kmer_index(kmer)] += 1.0 / denom
    return features


def load_checkpoint_model(
    checkpoint_path: str,
    tokenizer: Any,
    device: torch.device,
    *,
    random_init: bool = False,
) -> DnaModel:
    ckpt: dict[str, Any] = {}
    if checkpoint_path and not random_init:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg_dict = None
    if isinstance(ckpt, Mapping):
        cfg_dict = ckpt.get("model_config") or ckpt.get("config")
    if cfg_dict is None:
        cfg_dict = {"max_input_bases": tokenizer.max_length}
    if not isinstance(cfg_dict, Mapping):
        raise ValueError("Checkpoint does not contain a model configuration")
    cfg = config_from_checkpoint(cfg_dict, tokenizer)
    model = DnaModel(cfg)
    if checkpoint_path and not random_init:
        state = ckpt.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint does not contain model weights")
        load_state_dict_with_report(
            model,
            normalize_state_dict_keys(state),
            checkpoint_name=f"genomic benchmark checkpoint {checkpoint_path}",
            strict=True,
        )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def embed_sequences(
    model: DnaModel,
    tokenizer: Any,
    sequences: Sequence[str],
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    pooling: str = "mean",
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = [tokenizer.encode(seq, max_length=max_length, return_tensors=None) for seq in batch]
        input_ids = torch.tensor([enc["input_ids"] for enc in encoded], dtype=torch.long, device=device)
        attention_mask = torch.tensor([enc["attention_mask"] for enc in encoded], dtype=torch.long, device=device)
        kmer_ids = None
        if "kmer_ids" in encoded[0]:
            kmer_ids = torch.tensor([enc["kmer_ids"] for enc in encoded], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, kmer_ids=kmer_ids)
        hidden = out["last_hidden_state"]
        if pooling == "cls":
            pooled = hidden[:, 0]
        elif pooling == "last":
            lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=device), lengths]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)


def classification_probe(
    x_train: np.ndarray,
    y_train: Sequence[Any],
    x_test: np.ndarray,
    y_test: Sequence[Any],
    *,
    seed: int,
) -> dict[str, Any]:
    enc = LabelEncoder()
    ytr = enc.fit_transform(list(y_train))
    if len(enc.classes_) < 2:
        raise ValueError("Classification probe needs at least two classes in the training split")
    unseen = sorted(set(str(y) for y in y_test) - set(str(c) for c in enc.classes_))
    if unseen:
        raise ValueError(f"Test split contains labels not present in train split: {unseen}")
    yte = enc.transform(list(y_test))
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xte = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    out: dict[str, Any] = {
        "task": "classification",
        "num_train": int(len(ytr)),
        "num_test": int(len(yte)),
        "num_classes": int(len(enc.classes_)),
        "classes": [str(c) for c in enc.classes_],
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "mcc": float(matthews_corrcoef(yte, pred)),
    }
    if len(enc.classes_) == 2 and len(set(yte.tolist())) == 2:
        scores = clf.predict_proba(xte)[:, 1]
        out["auroc"] = float(roc_auc_score(yte, scores))
        out["auprc"] = float(average_precision_score(yte, scores))
    return out


def regression_probe(
    x_train: np.ndarray,
    y_train: Sequence[float],
    x_test: np.ndarray,
    y_test: Sequence[float],
    *,
    seed: int,
) -> dict[str, Any]:
    del seed
    ytr = np.asarray(y_train, dtype=np.float64)
    yte = np.asarray(y_test, dtype=np.float64)
    if len(ytr) < 2 or len(yte) < 2:
        raise ValueError("Regression probe needs at least two train and two test examples")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xte = scaler.transform(x_test)
    reg = Ridge(alpha=1.0)
    reg.fit(xtr, ytr)
    pred = reg.predict(xte)
    return {
        "task": "regression",
        "num_train": int(len(ytr)),
        "num_test": int(len(yte)),
        "r2": float(r2_score(yte, pred)),
        "pearson": _safe_corr(yte, pred),
        "spearman": _safe_corr(yte, pred, spearman=True),
    }


def run_probe_suite(
    records: SplitRecords,
    *,
    model: DnaModel | None,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
    pooling: str,
    seed: int,
    include_handcrafted: bool = True,
    category: str = "custom",
) -> dict[str, Any]:
    train_seq = [seq for seq, _ in records.train]
    test_seq = [seq for seq, _ in records.test]
    train_y = [label for _, label in records.train]
    test_y = [label for _, label in records.test]

    results: dict[str, Any] = {
        "task": records.task,
        "category": category,
        "num_train": len(records.train),
        "num_test": len(records.test),
        "max_length": max_length,
        "pooling": pooling,
        "model_probe": None,
        "handcrafted_baseline": None,
    }

    if model is not None:
        x_train = embed_sequences(
            model, tokenizer, train_seq, device, max_length=max_length, batch_size=batch_size, pooling=pooling
        )
        x_test = embed_sequences(
            model, tokenizer, test_seq, device, max_length=max_length, batch_size=batch_size, pooling=pooling
        )
        if records.task == "regression":
            results["model_probe"] = regression_probe(x_train, train_y, x_test, test_y, seed=seed)
        else:
            results["model_probe"] = classification_probe(x_train, train_y, x_test, test_y, seed=seed)

    if include_handcrafted:
        x_train_h = handcrafted_features(train_seq)
        x_test_h = handcrafted_features(test_seq)
        if records.task == "regression":
            results["handcrafted_baseline"] = regression_probe(x_train_h, train_y, x_test_h, test_y, seed=seed)
        else:
            results["handcrafted_baseline"] = classification_probe(x_train_h, train_y, x_test_h, test_y, seed=seed)
    return results


def _valid_token_positions(encoded: dict[str, list[int]]) -> list[int]:
    out: list[int] = []
    for i, (token_id, attn) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"], strict=True)):
        if attn and token_id > SPECIAL_TOKENS_MAX_ID:
            out.append(i)
    return out


@torch.no_grad()
def sequence_pseudo_nll(
    model: DnaModel,
    tokenizer: Any,
    sequence: str,
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    encoded = tokenizer.encode(sequence, max_length=max_length, return_tensors=None)
    positions = _valid_token_positions(encoded)
    if not positions:
        return {"nll": float("nan"), "n_tokens": 0}
    losses: list[float] = []
    for start in range(0, len(positions), batch_size):
        pos_batch = positions[start : start + batch_size]
        input_rows = []
        label_rows = []
        for pos in pos_batch:
            row = list(encoded["input_ids"])
            labels = [IGNORE_INDEX] * len(row)
            labels[pos] = row[pos]
            row[pos] = MASK_TOKEN_ID
            input_rows.append(row)
            label_rows.append(labels)
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        labels = torch.tensor(label_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor([encoded["attention_mask"] for _ in pos_batch], dtype=torch.long, device=device)
        kmer_ids = None
        if "kmer_ids" in encoded:
            kmer_ids = torch.tensor([encoded["kmer_ids"] for _ in pos_batch], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, kmer_ids=kmer_ids)
        logits = out["logits"]
        log_probs = torch.log_softmax(logits, dim=-1)
        for row_idx, pos in enumerate(pos_batch):
            token_id = labels[row_idx, pos]
            losses.append(float(-log_probs[row_idx, pos, token_id].item()))
    return {"nll": float(np.mean(losses)), "n_tokens": int(len(losses))}


def run_variant_effect_benchmark(
    variant_path: str | Path,
    *,
    model: DnaModel,
    tokenizer: Any,
    device: torch.device,
    ref_col: str = "ref_seq",
    alt_col: str = "alt_seq",
    label_col: str = "label",
    max_length: int,
    batch_size: int,
    max_records: int = 0,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[Any] = []
    rows = 0
    for row in _dict_rows_from_path(variant_path):
        ref = _clean_sequence(str(_row_value(row, ref_col)))
        alt = _clean_sequence(str(_row_value(row, alt_col)))
        if not ref or not alt:
            continue
        ref_nll = sequence_pseudo_nll(model, tokenizer, ref, device, max_length=max_length, batch_size=batch_size)
        alt_nll = sequence_pseudo_nll(model, tokenizer, alt, device, max_length=max_length, batch_size=batch_size)
        if ref_nll["n_tokens"] == 0 or alt_nll["n_tokens"] == 0:
            continue
        scores.append(float(alt_nll["nll"] - ref_nll["nll"]))
        labels.append(_row_value(row, label_col))
        rows += 1
        if max_records and rows >= max_records:
            break

    out: dict[str, Any] = {
        "task": "variant_effect",
        "category": "variant_effect",
        "score": "alt_pseudo_nll_minus_ref_pseudo_nll",
        "num_samples": len(scores),
    }
    binary = [_binary_label(label) for label in labels]
    if scores and all(label is not None for label in binary) and len(set(binary)) == 2:
        y = np.asarray(binary, dtype=np.int64)
        out["auroc"] = float(roc_auc_score(y, np.asarray(scores)))
        out["auprc"] = float(average_precision_score(y, np.asarray(scores)))
    else:
        numeric = [_as_float_if_possible(label) for label in labels]
        if scores and all(label is not None for label in numeric):
            out["pearson"] = _safe_corr(numeric, scores)
            out["spearman"] = _safe_corr(numeric, scores, spearman=True)
    return out


def _manifest_records(
    entry: dict[str, Any],
    *,
    max_records: int,
    base_dir: str | Path | None = None,
) -> SplitRecords:
    fmt = str(entry.get("format", "auto")).lower()
    task = entry.get("task", "classification")
    entry_path = str(entry.get("path", "")).lower()
    fasta_like = entry_path.endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz"))
    if fmt == "fasta" or (fmt == "auto" and ("files" in entry or fasta_like)):
        return _read_fasta_manifest_records(
            entry,
            task=task,
            base_dir=base_dir,
            max_records=int(entry.get("max_records", max_records or 0)),
        )

    return _read_tabular_records(
        _resolve_path(entry["path"], base_dir),
        seq_col=entry.get("seq_col", "sequence"),
        label_col=entry.get("label_col", "label"),
        split_col=entry.get("split_col", "split"),
        task=task,
        max_records=int(entry.get("max_records", max_records or 0)),
    )


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_results(results: dict[str, Any], out_json: str | None) -> None:
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark True DNA checkpoints on genomic tasks")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        type=normalize_tokenizer_type,
        choices=["bpe", "base"],
        default="bpe",
    )
    parser.add_argument("--tokenizer_path", default="dna_model/bpe_vocab/tokenizer.json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean", "cls", "last"], default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--max_records", type=int, default=0, help="Debug cap per dataset split/file")
    parser.add_argument("--skip_handcrafted_baseline", action="store_true")
    parser.add_argument("--include_random_model", action="store_true")
    parser.add_argument(
        "--allow_cpu", action="store_true", help="For parser tests only; real benchmark runs should use CUDA"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--genomic_benchmarks", nargs="*", help="Dataset names, or empty for default list")
    group.add_argument("--preset", choices=sorted(BENCHMARK_PRESETS), help="Built-in dataset preset")
    group.add_argument("--manifest", help="JSON manifest for harder custom datasets")

    parser.add_argument("--genomic_benchmarks_version", type=int, default=0)
    parser.add_argument("--genomic_benchmarks_root", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is required for genomic benchmarks; no CUDA device is available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(
        args.tokenizer_type,
        vocab_path=args.tokenizer_path,
        max_length=args.max_length,
    )
    model = load_checkpoint_model(args.checkpoint, tokenizer, device)
    random_model = (
        load_checkpoint_model(args.checkpoint, tokenizer, device, random_init=True)
        if args.include_random_model
        else None
    )

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "max_length": args.max_length,
        "pooling": args.pooling,
        "supported_manifest_sections": MANIFEST_SCHEMA_NOTE,
        "benchmarks": {},
        "variant_effect": {},
    }

    datasets: list[str] = []
    if args.preset:
        datasets = list(BENCHMARK_PRESETS[args.preset])
    elif args.genomic_benchmarks is not None:
        datasets = list(args.genomic_benchmarks or DEFAULT_GENOMIC_BENCHMARKS)

    for dataset in datasets:
        category = _infer_category(dataset)
        records = _read_genomic_benchmarks_dataset(
            dataset,
            version=args.genomic_benchmarks_version,
            root_dir=args.genomic_benchmarks_root,
            max_records=args.max_records,
        )
        entry = run_probe_suite(
            records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pooling=args.pooling,
            seed=args.seed,
            include_handcrafted=not args.skip_handcrafted_baseline,
            category=category,
        )
        if random_model is not None:
            entry["random_model_probe"] = run_probe_suite(
                records,
                model=random_model,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling=args.pooling,
                seed=args.seed,
                include_handcrafted=False,
                category=category,
            )["model_probe"]
        results["benchmarks"][dataset] = entry

    if args.manifest:
        manifest = _load_manifest(args.manifest)
        manifest_base = Path(args.manifest).resolve().parent
        for dataset_entry in manifest.get("genomic_benchmarks", []):
            if isinstance(dataset_entry, str):
                dataset_name = dataset_entry
                dataset_version = args.genomic_benchmarks_version
                dataset_max_records = args.max_records
                category = _infer_category(dataset_name)
            else:
                dataset_name = dataset_entry["name"]
                dataset_version = int(dataset_entry.get("version", args.genomic_benchmarks_version))
                dataset_max_records = int(dataset_entry.get("max_records", args.max_records or 0))
                category = _infer_category(dataset_name, dataset_entry.get("category"))
            records = _read_genomic_benchmarks_dataset(
                dataset_name,
                version=dataset_version,
                root_dir=args.genomic_benchmarks_root,
                max_records=dataset_max_records,
            )
            entry_results = run_probe_suite(
                records,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling=args.pooling,
                seed=args.seed,
                include_handcrafted=not args.skip_handcrafted_baseline,
                category=category,
            )
            if random_model is not None:
                entry_results["random_model_probe"] = run_probe_suite(
                    records,
                    model=random_model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    pooling=args.pooling,
                    seed=args.seed,
                    include_handcrafted=False,
                    category=category,
                )["model_probe"]
            results["benchmarks"][dataset_name] = entry_results

        for entry in manifest.get("classification", []):
            category = _infer_category(entry["name"], entry.get("category"))
            records = _manifest_records(
                {**entry, "task": "classification"}, max_records=args.max_records, base_dir=manifest_base
            )
            entry_results = run_probe_suite(
                records,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling=args.pooling,
                seed=args.seed,
                include_handcrafted=not args.skip_handcrafted_baseline,
                category=category,
            )
            if random_model is not None:
                entry_results["random_model_probe"] = run_probe_suite(
                    records,
                    model=random_model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    pooling=args.pooling,
                    seed=args.seed,
                    include_handcrafted=False,
                    category=category,
                )["model_probe"]
            results["benchmarks"][entry["name"]] = entry_results
        for entry in manifest.get("regression", []):
            category = _infer_category(entry["name"], entry.get("category"))
            records = _manifest_records(
                {**entry, "task": "regression"}, max_records=args.max_records, base_dir=manifest_base
            )
            entry_results = run_probe_suite(
                records,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                pooling=args.pooling,
                seed=args.seed,
                include_handcrafted=not args.skip_handcrafted_baseline,
                category=category,
            )
            if random_model is not None:
                entry_results["random_model_probe"] = run_probe_suite(
                    records,
                    model=random_model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    pooling=args.pooling,
                    seed=args.seed,
                    include_handcrafted=False,
                    category=category,
                )["model_probe"]
            results["benchmarks"][entry["name"]] = entry_results
        for entry in manifest.get("variant_effect", []):
            results["variant_effect"][entry["name"]] = run_variant_effect_benchmark(
                _resolve_path(entry["path"], manifest_base),
                model=model,
                tokenizer=tokenizer,
                device=device,
                ref_col=entry.get("ref_col", "ref_seq"),
                alt_col=entry.get("alt_col", "alt_seq"),
                label_col=entry.get("label_col", "label"),
                max_length=args.max_length,
                batch_size=args.batch_size,
                max_records=int(entry.get("max_records", args.max_records or 0)),
            )

    _write_results(results, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

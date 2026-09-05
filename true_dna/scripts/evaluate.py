"""Evaluate a trained DNA masked-language model checkpoint."""

from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dna_model.checkpoint_loading import (
    config_from_checkpoint,
    load_state_dict_with_report,
    normalize_state_dict_keys,
)
from dna_model.dataset import DatasetImpl, collate, fast_reverse_complement
from dna_model.loaders import RoundRobinLoader, load_split_manifest, scan_fasta_lengths
from dna_model.model import DnaModel
from dna_model.sequence_baselines import load_baseline_payload
from dna_model.tokenizer import build_tokenizer, normalize_tokenizer_type, read_fasta_records
from dna_model.utils import evaluate, masked_mean_pool

LOGGER = logging.getLogger(__name__)
EVAL_SAMPLE_SCHEMA = "true-dna-eval-windows-v1"


def _load_checkpoint(path: Path, tokenizer, device: torch.device) -> DnaModel:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Safe checkpoint loading requires a PyTorch version with weights_only support") from exc

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    config_values = checkpoint.get("model_config") or checkpoint.get("config")
    if not isinstance(config_values, Mapping):
        raise ValueError("Checkpoint does not contain a model configuration")
    config_values = dict(config_values)
    run_config = checkpoint.get("config")
    if "tokenizer_type" not in config_values and isinstance(run_config, Mapping):
        saved_mode = run_config.get("tokenizer_type")
        if saved_mode is not None:
            config_values["tokenizer_type"] = normalize_tokenizer_type(saved_mode)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain model weights")

    model = DnaModel(config_from_checkpoint(config_values, tokenizer))
    load_state_dict_with_report(
        model,
        normalize_state_dict_keys(state_dict),
        checkpoint_name=f"evaluation checkpoint {path}",
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def _expand_fasta_patterns(patterns: Sequence[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return sorted({os.path.abspath(path) for path in files if os.path.isfile(path)})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_groups(
    split_manifest: str | os.PathLike[str] | None,
    fasta_files: Sequence[str],
) -> dict[str, dict[int, str]]:
    """Load record-to-group mappings for assembly/family-stratified sampling."""
    result = {os.path.realpath(path): {} for path in fasta_files}
    if split_manifest is None:
        return result
    manifest_path = Path(split_manifest).expanduser().resolve()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
            raw_path = Path(str(row["fasta"])).expanduser()
            resolved = (raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path).resolve()
            key = os.path.realpath(resolved)
            record_index = int(row["record_index"])
            group_id = str(row.get("group_id") or f"record:{record_index}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid split manifest row {line_number}: {exc}") from exc
        if key in result:
            result[key][record_index] = group_id
    return result


def _stable_rank(seed: int, *parts: object) -> str:
    return hashlib.sha256("|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")).hexdigest()


def _stratified_window_indices(
    dataset: DatasetImpl,
    *,
    group_by_record: Mapping[int, str],
    target: int,
    seed: int,
    domain_name: str,
) -> list[int]:
    """Select windows evenly across held-out groups, then records and coordinates."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for dataset_index in range(len(dataset)):
        meta = dataset.window_metadata(dataset_index)
        group = group_by_record.get(meta["record_index"], f"record:{meta['record_index']}")
        grouped[group].append(dataset_index)
    for group, indices in grouped.items():
        indices.sort(
            key=lambda index: _stable_rank(
                seed,
                domain_name,
                group,
                dataset.window_metadata(index)["record_index"],
                dataset.window_metadata(index)["start"],
            )
        )
    group_order = sorted(grouped, key=lambda group: _stable_rank(seed, domain_name, group))
    selected: list[int] = []
    cursors = Counter()
    while len(selected) < min(target, len(dataset)):
        added = False
        for group in group_order:
            cursor = cursors[group]
            if cursor >= len(grouped[group]):
                continue
            selected.append(grouped[group][cursor])
            cursors[group] += 1
            added = True
            if len(selected) >= min(target, len(dataset)):
                break
        if not added:
            break
    return selected


def _read_eval_sample_manifest(path: Path) -> tuple[dict, list[dict]]:
    metadata = None
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        if line.startswith("#"):
            if metadata is None:
                metadata = json.loads(line[1:].strip())
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evaluation sample manifest row {line_number}: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != EVAL_SAMPLE_SCHEMA:
        raise ValueError(f"Unsupported evaluation sample manifest: {path}")
    return metadata, rows


def _write_eval_sample_manifest(path: Path, metadata: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _build_eval_loader(
    fasta_files: Sequence[str],
    tokenizer,
    *,
    max_length: int,
    stride: int,
    min_length: int,
    batch_size: int,
    workers: int,
    split_manifest: str | os.PathLike[str] | None = None,
    max_batches: int = 0,
    sample_manifest: Path | None = None,
    sample_seed: int = 43,
) -> tuple[RoundRobinLoader, dict]:
    datasets: list[tuple[str, str, DatasetImpl]] = []
    assignments = load_split_manifest(split_manifest, fasta_files) if split_manifest is not None else None
    groups = _split_groups(split_manifest, fasta_files)
    min_chunk_length = max(int(min_length), max_length // 4, 64)
    for fasta_path in fasta_files:
        if assignments is not None:
            if fasta_path.lower().endswith(".gz"):
                raise ValueError("--split-manifest requires uncompressed FASTA paths for random-access evaluation")
            file_key = os.path.realpath(fasta_path)
            seq_offsets = scan_fasta_lengths(fasta_path)
            valid_indices = {index for index, _start, _end, length in seq_offsets if length >= min_chunk_length}
            assigned = assignments[file_key]
            assigned_indices = assigned["train"] | assigned["eval"]
            unknown = assigned_indices.difference({index for index, _start, _end, _length in seq_offsets})
            unassigned = valid_indices.difference(assigned_indices)
            if unknown:
                raise ValueError(f"Split manifest references missing records in {fasta_path}: {sorted(unknown)[:5]}")
            if unassigned:
                raise ValueError(f"Split manifest leaves qualifying records unassigned in {fasta_path}")
            eval_indices = sorted(assigned["eval"].intersection(valid_indices))
            source = fasta_path
            lazy_load = True
            dataset_kwargs = {"indices": eval_indices, "precomputed_offsets": seq_offsets}
        elif fasta_path.lower().endswith(".gz"):
            sequences = [
                sequence
                for _header, sequence, _is_last in read_fasta_records(fasta_path)
                if len(sequence) >= min_length
            ]
            source = sequences
            lazy_load = False
            dataset_kwargs = {}
        else:
            source = fasta_path
            lazy_load = True
            dataset_kwargs = {}

        dataset = DatasetImpl(
            tokenizer,
            source,
            max_length=max_length,
            stride=stride,
            min_chunk_length=min_chunk_length,
            lazy_load=lazy_load,
            return_offsets=True,
            **dataset_kwargs,
        )
        if len(dataset) == 0:
            LOGGER.warning("Skipping %s because it has no qualifying windows", fasta_path)
            continue
        datasets.append((fasta_path, os.path.basename(fasta_path), dataset))

    if not datasets:
        raise ValueError("No qualifying evaluation windows were found")

    expected_metadata = {
        "schema": EVAL_SAMPLE_SCHEMA,
        "max_length": int(max_length),
        "stride": int(stride),
        "min_length": int(min_length),
        "batch_size": int(batch_size),
        "max_batches": int(max_batches),
        "seed": int(sample_seed),
        "split_manifest_sha256": _sha256(Path(split_manifest).resolve()) if split_manifest is not None else None,
        "fasta_sha256": {os.path.realpath(path): _sha256(Path(path).resolve()) for path in sorted(fasta_files)},
    }
    rows_by_domain: dict[str, list[dict]] = defaultdict(list)
    if sample_manifest is not None and sample_manifest.is_file():
        saved_metadata, saved_rows = _read_eval_sample_manifest(sample_manifest)
        mismatches = {
            key: (saved_metadata.get(key), value)
            for key, value in expected_metadata.items()
            if saved_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Evaluation sample manifest protocol mismatch: {mismatches}")
        for row in saved_rows:
            rows_by_domain[str(row["domain_name"])].append(row)
    else:
        domain_count = len(datasets)
        quotient, remainder = divmod(max(0, int(max_batches)), domain_count)
        all_rows: list[dict] = []
        for domain_index, (fasta_path, domain_name, dataset) in enumerate(datasets):
            target_batches = quotient + (1 if domain_index < remainder else 0)
            target_windows = len(dataset) if max_batches <= 0 else target_batches * batch_size
            selected = _stratified_window_indices(
                dataset,
                group_by_record=groups[os.path.realpath(fasta_path)],
                target=target_windows,
                seed=sample_seed,
                domain_name=domain_name,
            )
            for order, dataset_index in enumerate(selected):
                meta = dataset.window_metadata(dataset_index)
                row = {
                    "domain_name": domain_name,
                    "record_index": meta["record_index"],
                    "start": meta["start"],
                    "end": meta["end"],
                    "group_id": groups[os.path.realpath(fasta_path)].get(
                        meta["record_index"], f"record:{meta['record_index']}"
                    ),
                    "order": order,
                }
                rows_by_domain[domain_name].append(row)
                all_rows.append(row)
        if sample_manifest is not None:
            _write_eval_sample_manifest(sample_manifest, expected_metadata, all_rows)

    loaders = []
    names = []
    group_counts: dict[str, int] = {}
    window_counts: dict[str, int] = {}
    for _fasta_path, domain_name, dataset in datasets:
        lookup = {
            (meta["record_index"], meta["start"], meta["end"]): index
            for index in range(len(dataset))
            for meta in [dataset.window_metadata(index)]
        }
        selected_indices = []
        for row in sorted(rows_by_domain[domain_name], key=lambda item: int(item["order"])):
            key = (int(row["record_index"]), int(row["start"]), int(row["end"]))
            if key not in lookup:
                raise ValueError(f"Evaluation sample window is unavailable in {domain_name}: {key}")
            selected_indices.append(lookup[key])
        if not selected_indices:
            raise ValueError(f"Evaluation sample contains no windows for {domain_name}")
        sampled_dataset = Subset(dataset, selected_indices)
        loaders.append(
            DataLoader(
                sampled_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=functools.partial(collate, max_length=max_length),
                persistent_workers=workers > 0,
                prefetch_factor=2 if workers > 0 else None,
            )
        )
        names.append(domain_name)
        window_counts[domain_name] = len(selected_indices)
        group_counts[domain_name] = len({str(row["group_id"]) for row in rows_by_domain[domain_name]})

    sample_info = {
        **expected_metadata,
        "path": str(sample_manifest.resolve()) if sample_manifest is not None else None,
        "sha256": _sha256(sample_manifest) if sample_manifest is not None and sample_manifest.is_file() else None,
        "windows_by_domain": window_counts,
        "groups_by_domain": group_counts,
    }
    return RoundRobinLoader(loaders, names=names), sample_info


def _sample_sequences(
    fasta_files: Sequence[str],
    *,
    sample_size: int,
    min_length: int,
) -> list[str]:
    """Reservoir-sample qualifying records without loading the corpus into RAM."""

    if sample_size <= 0:
        return []
    rng = random.Random(42)
    samples: list[str] = []
    seen = 0
    for fasta_path in fasta_files:
        for _header, sequence, _is_last in read_fasta_records(fasta_path):
            if len(sequence) < min_length:
                continue
            seen += 1
            if len(samples) < sample_size:
                samples.append(sequence)
                continue
            replacement = rng.randrange(seen)
            if replacement < sample_size:
                samples[replacement] = sequence
    return samples


@torch.inference_mode()
def rc_embedding_agreement(
    model: DnaModel,
    tokenizer,
    sequences: Sequence[str],
    device: torch.device,
    *,
    max_length: int,
) -> dict[str, float | int]:
    """Compare mean-pooled embeddings for sequences and reverse complements."""

    cosine_values = []
    normalized_l2_values = []
    for sequence in sequences:
        sequence = sequence[:max_length]
        encoded_pairs = [
            tokenizer.encode(sequence, max_length=max_length),
            tokenizer.encode(fast_reverse_complement(sequence), max_length=max_length),
        ]
        pooled = []
        for encoded in encoded_pairs:
            input_ids = encoded["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
            kmer_ids = encoded.get("kmer_ids")
            if kmer_ids is not None:
                kmer_ids = kmer_ids.unsqueeze(0).to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kmer_ids=kmer_ids,
            )
            hidden = output["last_hidden_state"]
            pooled.append(masked_mean_pool(hidden, attention_mask))

        cosine_values.append(F.cosine_similarity(pooled[0], pooled[1]).item())
        normalized_l2_values.append(
            torch.linalg.vector_norm(pooled[0] - pooled[1]).item() / math.sqrt(pooled[0].shape[-1])
        )

    if not cosine_values:
        return {"num_sequences": 0, "mean_cosine_similarity": float("nan"), "mean_normalized_l2": float("nan")}
    return {
        "num_sequences": len(cosine_values),
        "mean_cosine_similarity": sum(cosine_values) / len(cosine_values),
        "min_cosine_similarity": min(cosine_values),
        "max_cosine_similarity": max(cosine_values),
        "mean_normalized_l2": sum(normalized_l2_values) / len(normalized_l2_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate masked-token metrics and reverse-complement embedding agreement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint path")
    parser.add_argument("--fasta", required=True, nargs="+", help="FASTA paths or glob patterns")
    parser.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        type=normalize_tokenizer_type,
        default="bpe",
        choices=["bpe", "base"],
        help="Checkpoint tokenization granularity (single_base is accepted as a legacy alias for base)",
    )
    parser.add_argument("--tokenizer-path", "--tokenizer_path", default="dna_model/bpe_vocab/tokenizer.json")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional explicit record-level split manifest; evaluates only its held-out records.",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--max-length", type=int, default=None, help="Evaluation window size; defaults to checkpoint config"
    )
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to max length")
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=500, help="Masked-token batches; 0 evaluates all batches")
    parser.add_argument(
        "--eval-sample-manifest",
        type=Path,
        default=None,
        help="Reusable domain/group-stratified raw-window manifest; created if absent",
    )
    parser.add_argument("--eval-sample-seed", type=int, default=43)
    parser.add_argument(
        "--mask-coordinate-system",
        choices=["base", "token"],
        default="base",
        help="Use identical raw-base corruption spans across tokenizer conditions",
    )
    parser.add_argument("--mask-seed", type=int, default=43)
    parser.add_argument(
        "--baseline-model-json",
        type=Path,
        default=None,
        help="Training-split unigram/first-order baseline model from build_sequence_baselines.py",
    )
    parser.add_argument("--rc-check-seqs", "--rc_check_seqs", type=int, default=100)
    parser.add_argument("--rc-max-length", type=int, default=512)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-masked-targets",
        action="store_true",
        help="Replace every selected MLM target with [MASK] instead of the 80/10/10 corruption policy.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for evaluation; no CUDA device is available")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size <= 0 or args.workers < 0 or args.min_length <= 0:
        raise SystemExit("batch size and minimum length must be positive; workers must be non-negative")
    if args.max_batches < 0:
        raise SystemExit("max batches must be non-negative")
    if args.rc_check_seqs < 0 or args.rc_max_length <= 0:
        raise SystemExit("RC sample count must be non-negative and RC max length must be positive")
    if args.baseline_model_json is not None and not args.baseline_model_json.is_file():
        raise SystemExit(f"Baseline model does not exist: {args.baseline_model_json}")

    fasta_files = _expand_fasta_patterns(args.fasta)
    if not fasta_files:
        raise SystemExit(f"No FASTA files matched: {args.fasta}")

    device = torch.device("cuda")

    tokenizer = build_tokenizer(args.tokenizer_type, vocab_path=args.tokenizer_path)

    model = _load_checkpoint(args.checkpoint, tokenizer, device)
    model_config = model.config
    max_length = int(args.max_length or model_config.max_input_bases)
    stride = int(args.stride or max_length)
    if max_length <= 0 or stride <= 0:
        raise SystemExit("max length and stride must be positive")

    sample_manifest = args.eval_sample_manifest
    if sample_manifest is None:
        sample_root = args.output_json.parent if args.output_json is not None else args.checkpoint.parent
        sample_manifest = sample_root / (
            f"eval_windows_ctx{max_length}_stride{stride}_b{args.batch_size}"
            f"_n{args.max_batches}_seed{args.eval_sample_seed}.jsonl"
        )
    eval_loader, sample_info = _build_eval_loader(
        fasta_files,
        tokenizer,
        max_length=max_length,
        stride=stride,
        min_length=args.min_length,
        batch_size=args.batch_size,
        workers=args.workers,
        split_manifest=args.split_manifest,
        max_batches=args.max_batches,
        sample_manifest=sample_manifest,
        sample_seed=args.eval_sample_seed,
    )
    baseline_payload = None
    if args.baseline_model_json is not None:
        baseline_payload = load_baseline_payload(
            args.baseline_model_json,
            fasta_paths=fasta_files,
            split_manifest=args.split_manifest,
        )
    masked_metrics = evaluate(
        model,
        eval_loader,
        device,
        args.amp,
        epoch=0,
        tok=tokenizer,
        max_batches=args.max_batches,
        strict_masked_targets=args.strict_masked_targets,
        mask_coordinate_system=args.mask_coordinate_system,
        mask_seed=args.mask_seed,
        baseline_payload=baseline_payload,
    )

    rc_sequences = _sample_sequences(
        fasta_files,
        sample_size=args.rc_check_seqs,
        min_length=args.min_length,
    )
    rc_metrics = rc_embedding_agreement(
        model,
        tokenizer,
        rc_sequences,
        device,
        max_length=args.rc_max_length,
    )

    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "fasta_files": len(fasta_files),
        "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest is not None else None,
        "strict_masked_targets": bool(args.strict_masked_targets),
        "evaluation_sample": sample_info,
        "mask_coordinate_system": args.mask_coordinate_system,
        "mask_seed": args.mask_seed,
        "baseline_model_json": (
            str(args.baseline_model_json.resolve()) if args.baseline_model_json is not None else None
        ),
        "masked_token": masked_metrics,
        "reverse_complement": rc_metrics,
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

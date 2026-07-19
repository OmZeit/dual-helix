import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

from .dataset import fast_reverse_complement


def dinucleotide_shuffle(seq: str, rng: random.Random) -> str:
    """Randomize a sequence while preserving its exact dinucleotide counts."""

    if len(seq) < 2:
        return seq

    outgoing = defaultdict(list)
    for left, right in zip(seq, seq[1:], strict=False):
        outgoing[left].append(right)
    for destinations in outgoing.values():
        rng.shuffle(destinations)

    stack = [seq[0]]
    trail = []
    while stack:
        current = stack[-1]
        if outgoing[current]:
            stack.append(outgoing[current].pop())
        else:
            trail.append(stack.pop())

    shuffled = "".join(reversed(trail))
    if len(shuffled) != len(seq):
        raise RuntimeError("Failed to construct a dinucleotide-preserving shuffle")
    return shuffled


def gc_matched_random(seq: str, rng: random.Random) -> str:
    gc = sum(ch in "GCgc" for ch in seq) / max(1, len(seq))
    weights = [(1.0 - gc) / 2.0, gc / 2.0, gc / 2.0, (1.0 - gc) / 2.0]
    return "".join(rng.choices(["A", "C", "G", "T"], weights=weights, k=len(seq)))


def bpe_rc_diagnostics(tokenizer, sequences: list[str], max_length: int = 1024) -> dict[str, float]:
    length_diffs = []
    token_match_rates = []
    kmer_match_rates = []

    for seq in sequences:
        seq = seq[:max_length]
        enc = tokenizer.encode(seq, max_length=max_length, return_tensors=None)
        rc_enc = tokenizer.encode(fast_reverse_complement(seq), max_length=max_length, return_tensors=None)
        ids = [x for x, m in zip(enc["input_ids"], enc["attention_mask"], strict=True) if m]
        rc_ids = [x for x, m in zip(rc_enc["input_ids"], rc_enc["attention_mask"], strict=True) if m]
        n = min(len(ids), len(rc_ids))
        length_diffs.append(abs(len(ids) - len(rc_ids)))
        token_match_rates.append(sum(a == b for a, b in zip(ids[:n], reversed(rc_ids[:n]), strict=True)) / max(1, n))

        if "kmer_ids" in enc and "kmer_ids" in rc_enc:
            kmers = [x for x, m in zip(enc["kmer_ids"], enc["attention_mask"], strict=True) if m]
            rc_kmers = [x for x, m in zip(rc_enc["kmer_ids"], rc_enc["attention_mask"], strict=True) if m]
            nk = min(len(kmers), len(rc_kmers))
            kmer_match_rates.append(
                sum(a == b for a, b in zip(kmers[:nk], reversed(rc_kmers[:nk]), strict=True)) / max(1, nk)
            )

    return {
        "mean_token_length_abs_diff": float(np.mean(length_diffs)) if length_diffs else 0.0,
        "max_token_length_abs_diff": float(np.max(length_diffs)) if length_diffs else 0.0,
        "mean_reversed_token_id_match": float(np.mean(token_match_rates)) if token_match_rates else 0.0,
        "mean_reversed_kmer_id_match": float(np.mean(kmer_match_rates)) if kmer_match_rates else 0.0,
        "num_sequences": len(sequences),
    }


@torch.no_grad()
def sequence_embeddings(model, tokenizer, sequences: list[str], device, max_length: int = 1024, batch_size: int = 8):
    vectors = []
    model.eval()

    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
        batch = sequences[i : i + batch_size]
        encoded = [tokenizer.encode(seq, max_length=max_length, return_tensors=None) for seq in batch]
        input_ids = torch.tensor([enc["input_ids"] for enc in encoded], dtype=torch.long, device=device)
        attention_mask = torch.tensor([enc["attention_mask"] for enc in encoded], dtype=torch.long, device=device)
        kmer_ids = None
        if "kmer_ids" in encoded[0]:
            kmer_ids = torch.tensor([enc["kmer_ids"] for enc in encoded], dtype=torch.long, device=device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, kmer_ids=kmer_ids)
        hidden = out["last_hidden_state"]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vectors.append(pooled.float().cpu().numpy())

    return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32)


def run_linear_probe_benchmark(
    model,
    tokenizer,
    records: list[tuple[str, str]],
    device,
    max_length: int = 1024,
    batch_size: int = 8,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    sequences = [seq for seq, _label in records]
    labels = [label for _seq, label in records]
    vectors = sequence_embeddings(model, tokenizer, sequences, device, max_length=max_length, batch_size=batch_size)

    y = LabelEncoder().fit_transform(labels)
    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        vectors, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    results = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "num_classes": int(len(np.unique(y))),
        "num_samples": int(len(records)),
    }
    if results["num_classes"] == 2:
        scores = clf.predict_proba(x_test)[:, 1]
        results["auroc"] = float(roc_auc_score(y_test, scores))
        results["auprc"] = float(average_precision_score(y_test, scores))
    return results


def build_negative_control_records(
    positives: list[str],
    *,
    mode: str = "dinuc",
    seed: int = 42,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    records = [(seq, "real") for seq in positives]
    for seq in positives:
        if mode == "gc":
            neg = gc_matched_random(seq, rng)
        else:
            neg = dinucleotide_shuffle(seq, rng)
        records.append((neg, mode))
    return records

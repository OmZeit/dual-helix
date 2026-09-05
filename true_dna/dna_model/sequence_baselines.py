"""Transparent nucleotide baselines evaluated on the exact masked base coordinates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

ALPHABET = "ACGT"
SCHEMA = "true-dna-sequence-baselines-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class NucleotideCounter:
    """Count unigrams and left-to-right first-order transitions."""

    def __init__(self) -> None:
        self.unigrams: Counter[str] = Counter()
        self.transitions: dict[str, Counter[str]] = {base: Counter() for base in ALPHABET}
        self.records = 0

    def update(self, sequence: str) -> None:
        self.records += 1
        previous: str | None = None
        for base in sequence.upper():
            if base not in ALPHABET:
                previous = None
                continue
            self.unigrams[base] += 1
            if previous is not None:
                self.transitions[previous][base] += 1
            previous = base

    def probabilities(self, *, alpha: float) -> dict[str, Any]:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        unigram_total = sum(self.unigrams.values()) + alpha * len(ALPHABET)
        unigram = {base: (self.unigrams[base] + alpha) / unigram_total for base in ALPHABET}
        transition = {}
        for previous in ALPHABET:
            total = sum(self.transitions[previous].values()) + alpha * len(ALPHABET)
            transition[previous] = {base: (self.transitions[previous][base] + alpha) / total for base in ALPHABET}
        return {
            "records": self.records,
            "bases": sum(self.unigrams.values()),
            "unigram": unigram,
            "first_order": transition,
        }


def build_baseline_payload(
    sequences_by_domain: dict[str, Iterable[str]],
    *,
    alpha: float,
    split_manifest: str | Path | None = None,
    fasta_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    global_counter = NucleotideCounter()
    domain_models: dict[str, dict[str, Any]] = {}
    for domain in sorted(sequences_by_domain):
        counter = NucleotideCounter()
        for sequence in sequences_by_domain[domain]:
            counter.update(sequence)
            global_counter.update(sequence)
        domain_models[domain] = counter.probabilities(alpha=alpha)
    return {
        "schema": SCHEMA,
        "alphabet": ALPHABET,
        "alpha": float(alpha),
        "split_manifest": str(Path(split_manifest).resolve()) if split_manifest is not None else None,
        "split_manifest_sha256": sha256_file(split_manifest) if split_manifest is not None else None,
        "fastas": {str(Path(path).resolve()): {"sha256": sha256_file(path)} for path in sorted(map(Path, fasta_paths))},
        "global": global_counter.probabilities(alpha=alpha),
        "domains": domain_models,
    }


def validate_baseline_sources(
    payload: dict[str, Any],
    *,
    fasta_paths: Iterable[str | Path],
    split_manifest: str | Path | None,
) -> None:
    """Reject a cached baseline when any corpus or split input has changed."""
    if split_manifest is not None:
        expected_split_hash = sha256_file(split_manifest)
        if payload.get("split_manifest_sha256") != expected_split_hash:
            raise ValueError("Baseline split-manifest hash mismatch")

    saved_fastas = payload.get("fastas")
    if not isinstance(saved_fastas, dict):
        raise ValueError("Baseline model does not record FASTA content hashes")
    expected_paths = {str(Path(path).resolve()) for path in fasta_paths}
    if set(saved_fastas) != expected_paths:
        raise ValueError("Baseline FASTA source set mismatch")
    for fasta_path in sorted(expected_paths):
        record = saved_fastas.get(fasta_path)
        if not isinstance(record, dict) or record.get("sha256") != sha256_file(fasta_path):
            raise ValueError(f"Baseline FASTA content hash mismatch: {fasta_path}")


def load_baseline_payload(
    path: str | Path,
    *,
    fasta_paths: Iterable[str | Path] | None = None,
    split_manifest: str | Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("alphabet") != ALPHABET:
        raise ValueError(f"Unsupported nucleotide baseline model: {path}")
    if fasta_paths is not None:
        validate_baseline_sources(
            payload,
            fasta_paths=fasta_paths,
            split_manifest=split_manifest,
        )
    return payload


class MaskedBaselineAccumulator:
    """Score uniform, unigram, and first-order models on selected MLM bases."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload
        self.supported_bases = 0
        self.nll = Counter()

    @staticmethod
    def _probability(model: dict[str, Any], target: str, previous: str | None, *, order: int) -> float:
        if order == 1 and previous in ALPHABET:
            return float(model["first_order"][previous][target])
        return float(model["unigram"][target])

    def update(
        self,
        *,
        input_ids: torch.Tensor,
        token_offsets: torch.Tensor | None,
        mask_bool: torch.Tensor,
        tokenizer,
        domain_names: str | list[str] | tuple[str, ...] | None,
    ) -> None:
        if token_offsets is None:
            return
        ids_cpu = input_ids.detach().cpu()
        offsets_cpu = token_offsets.detach().cpu()
        mask_cpu = mask_bool.detach().cpu()
        if isinstance(domain_names, str) or domain_names is None:
            domains = [domain_names] * input_ids.size(0)
        else:
            domains = list(domain_names)
            if len(domains) != input_ids.size(0):
                domains = [domains[0] if domains else None] * input_ids.size(0)

        for row in range(input_ids.size(0)):
            max_end = int(offsets_cpu[row, :, 1].max().item())
            if max_end <= 0:
                continue
            sequence = [""] * max_end
            base_mask = [False] * max_end
            for position in range(input_ids.size(1)):
                start, end = (int(value) for value in offsets_cpu[row, position].tolist())
                if end <= start or start < 0 or end > max_end:
                    continue
                token_sequence = tokenizer.token_sequence(int(ids_cpu[row, position].item()))
                if len(token_sequence) != end - start:
                    continue
                sequence[start:end] = token_sequence
                if bool(mask_cpu[row, position].item()):
                    for base_position in range(start, end):
                        base_mask[base_position] = True

            global_model = self.payload.get("global") if self.payload is not None else None
            domain_model = None
            if self.payload is not None and domains[row] is not None:
                domain_model = self.payload.get("domains", {}).get(str(domains[row]))
            for position, selected in enumerate(base_mask):
                target = sequence[position]
                if not selected or target not in ALPHABET:
                    continue
                self.supported_bases += 1
                self.nll["uniform"] += -math.log(1.0 / len(ALPHABET))
                previous = sequence[position - 1] if position > 0 and not base_mask[position - 1] else None
                if global_model is not None:
                    self.nll["global_unigram"] += -math.log(self._probability(global_model, target, None, order=0))
                    self.nll["global_first_order"] += -math.log(
                        self._probability(global_model, target, previous, order=1)
                    )
                if domain_model is not None:
                    self.nll["domain_unigram"] += -math.log(self._probability(domain_model, target, None, order=0))
                    self.nll["domain_first_order"] += -math.log(
                        self._probability(domain_model, target, previous, order=1)
                    )

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"supported_masked_bases": self.supported_bases}
        if self.supported_bases == 0:
            return result
        result["models"] = {
            name: {
                "nats_per_base": total / self.supported_bases,
                "bits_per_base": total / self.supported_bases / math.log(2.0),
            }
            for name, total in sorted(self.nll.items())
        }
        return result

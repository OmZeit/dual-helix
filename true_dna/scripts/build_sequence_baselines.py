#!/usr/bin/env python3
"""Fit transparent nucleotide baselines on training-assigned FASTA records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dna_model.loaders import load_split_manifest
from dna_model.sequence_baselines import ALPHABET, SCHEMA, NucleotideCounter, sha256_file
from dna_model.tokenizer import read_fasta_records


def build_payload(fasta_files: list[Path], split_manifest: Path, *, alpha: float) -> dict:
    assignments = load_split_manifest(split_manifest, [str(path) for path in fasta_files])
    global_counter = NucleotideCounter()
    domains = {}
    for fasta in fasta_files:
        counter = NucleotideCounter()
        train_indices = assignments[str(fasta.resolve())]["train"]
        for record_index, (_header, sequence, _is_last) in enumerate(read_fasta_records(str(fasta))):
            if record_index not in train_indices:
                continue
            counter.update(sequence)
            global_counter.update(sequence)
        domains[fasta.name] = counter.probabilities(alpha=alpha)
    return {
        "schema": SCHEMA,
        "alphabet": ALPHABET,
        "alpha": float(alpha),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest),
        "fastas": {
            str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in fasta_files
        },
        "global": global_counter.probabilities(alpha=alpha),
        "domains": domains,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", nargs="+", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fasta_files = sorted({path.expanduser().resolve() for path in args.fasta})
    if not fasta_files or any(not path.is_file() for path in fasta_files):
        raise SystemExit("All --fasta paths must exist")
    split_manifest = args.split_manifest.expanduser().resolve()
    if not split_manifest.is_file() or args.alpha <= 0:
        raise SystemExit("The split manifest must exist and alpha must be positive")
    payload = build_payload(fasta_files, split_manifest, alpha=args.alpha)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + os.linesep, encoding="utf-8")
    print(json.dumps({"output": str(args.output_json.resolve()), "global": payload["global"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

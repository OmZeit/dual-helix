#!/usr/bin/env python3
"""Create a reproducible, group-aware train/eval JSONL split manifest."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _expand_fastas(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if matches:
            files.extend(matches)
        elif Path(pattern).is_file():
            files.append(Path(pattern))
    return sorted({path.resolve() for path in files})


def _records(path: Path) -> Iterable[tuple[int, str, int]]:
    index = -1
    header: str | None = None
    sequence_length = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield index, header, sequence_length
                index += 1
                header = line[1:].strip()
                sequence_length = 0
            elif header is not None:
                sequence_length += len(line.strip())
    if header is not None:
        yield index, header, sequence_length


def _group_id(
    *,
    fasta: Path,
    record_index: int,
    header: str,
    group_by: str,
    header_pattern: re.Pattern[str] | None,
) -> str:
    if group_by == "record":
        return f"{fasta.name}:record:{record_index}"
    if group_by == "assembly":
        match = re.search(r"(?:^|[|\s])assembly=([^|\s]+)", header)
        if match is None:
            raise ValueError(
                f"{fasta} record {record_index} has no assembly=<accession> header tag. "
                "Use newly fetched data, or choose --group-by record only for smoke testing."
            )
        return f"assembly:{match.group(1)}"
    if header_pattern is None:
        raise ValueError("--header-regex is required with --group-by header-regex")
    match = header_pattern.search(header)
    if match is None:
        raise ValueError(f"Header regex did not match {fasta} record {record_index}: {header!r}")
    return f"header:{match.group(1) if match.groups() else match.group(0)}"


def build_manifest(
    fasta_files: list[Path],
    output: Path,
    *,
    group_by: str,
    holdout_fraction: float,
    seed: int,
    header_regex: str | None = None,
    balance_by: str = "bases",
) -> dict[str, int | float | str]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if not fasta_files:
        raise ValueError("No FASTA files matched")
    if balance_by not in {"bases", "records"}:
        raise ValueError("balance_by must be 'bases' or 'records'")

    header_pattern = re.compile(header_regex) if header_regex else None
    logical_fasta_ids = {
        fasta: os.path.relpath(fasta.resolve(), output.parent.resolve()).replace("\\", "/") for fasta in fasta_files
    }
    groups: dict[str, list[tuple[Path, int, int]]] = defaultdict(list)
    for fasta in fasta_files:
        for record_index, header, sequence_length in _records(fasta):
            group = _group_id(
                fasta=fasta,
                record_index=record_index,
                header=header,
                group_by=group_by,
                header_pattern=header_pattern,
            )
            groups[group].append((fasta, record_index, sequence_length))

    if len(groups) < 2:
        raise ValueError("At least two groups are required to make a train/eval split")

    groups_by_fasta: dict[Path, set[str]] = defaultdict(set)
    records_by_fasta: dict[Path, int] = defaultdict(int)
    bases_by_fasta: dict[Path, int] = defaultdict(int)
    fastas_by_group: dict[str, set[Path]] = defaultdict(set)
    records_by_group_and_fasta: dict[tuple[str, Path], int] = defaultdict(int)
    bases_by_group_and_fasta: dict[tuple[str, Path], int] = defaultdict(int)
    for group, rows in groups.items():
        for fasta, _record_index, sequence_length in rows:
            groups_by_fasta[fasta].add(group)
            fastas_by_group[group].add(fasta)
            records_by_fasta[fasta] += 1
            records_by_group_and_fasta[(group, fasta)] += 1
            bases_by_fasta[fasta] += sequence_length
            bases_by_group_and_fasta[(group, fasta)] += sequence_length

    for fasta in fasta_files:
        group_count = len(groups_by_fasta[fasta])
        if group_count < 2:
            raise ValueError(
                f"{fasta} has only {group_count} distinct {group_by} group(s). "
                "A record-disjoint split requires at least two groups in every FASTA."
            )

    def ordered_for_fasta(fasta: Path) -> list[str]:
        """Stable pseudo-random group ordering scoped to one FASTA."""
        return sorted(
            groups_by_fasta[fasta],
            key=lambda group: hashlib.sha256(f"{seed}|{logical_fasta_ids[fasta]}|{group}".encode()).hexdigest(),
        )

    eval_groups: set[str] = set()

    def can_mark_eval(group: str) -> bool:
        """Keep at least one train assembly in every FASTA sharing this group."""
        candidate_eval_groups = eval_groups | {group}
        return all(
            any(candidate not in candidate_eval_groups for candidate in groups_by_fasta[affected_fasta])
            for affected_fasta in fastas_by_group[group]
        )

    # The original implementation targeted the corpus globally. That can
    # leave an entire domain without evaluation data when one domain dominates
    # its record or nucleotide mass. Allocate each FASTA its own deterministic
    # target while retaining assembly-level disjointness across the manifest.
    for fasta in sorted(fasta_files):
        metric_by_group = bases_by_group_and_fasta if balance_by == "bases" else records_by_group_and_fasta
        total_metric = bases_by_fasta[fasta] if balance_by == "bases" else records_by_fasta[fasta]
        target_eval = max(1, round(total_metric * holdout_fraction))
        selected_for_fasta = sum(
            metric_by_group[(group, fasta)] for group in groups_by_fasta[fasta] if group in eval_groups
        )
        for group in ordered_for_fasta(fasta):
            if selected_for_fasta >= target_eval:
                break
            if group in eval_groups:
                continue
            if not can_mark_eval(group):
                continue
            eval_groups.add(group)
            selected_for_fasta += metric_by_group[(group, fasta)]

        if selected_for_fasta == 0:
            raise ValueError(
                f"Could not assign an evaluation {group_by} group to {fasta} without emptying another FASTA's train split."
            )

    per_fasta: dict[str, dict[str, int | float]] = {}
    for fasta in sorted(fasta_files):
        fasta_groups = groups_by_fasta[fasta]
        eval_group_count = sum(group in eval_groups for group in fasta_groups)
        train_group_count = len(fasta_groups) - eval_group_count
        if eval_group_count == 0 or train_group_count == 0:
            raise ValueError(f"Split must leave both train and eval {group_by} groups for {fasta}")
        eval_records = sum(records_by_group_and_fasta[(group, fasta)] for group in fasta_groups if group in eval_groups)
        eval_bases = sum(bases_by_group_and_fasta[(group, fasta)] for group in fasta_groups if group in eval_groups)
        per_fasta[str(fasta)] = {
            "records": records_by_fasta[fasta],
            "groups": len(fasta_groups),
            "eval_groups": eval_group_count,
            "train_groups": train_group_count,
            "eval_records": eval_records,
            "train_records": records_by_fasta[fasta] - eval_records,
            "bases": bases_by_fasta[fasta],
            "eval_bases": eval_bases,
            "train_bases": bases_by_fasta[fasta] - eval_bases,
            "eval_record_fraction": eval_records / records_by_fasta[fasta],
            "eval_base_fraction": eval_bases / bases_by_fasta[fasta],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", encoding="utf-8") as handle:
        metadata = {
            "schema": "true-dna-split-v2",
            "group_by": group_by,
            "holdout_fraction": holdout_fraction,
            "seed": seed,
            "stratified_per_fasta": True,
            "balance_by": balance_by,
        }
        handle.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        for group in sorted(groups):
            split = "eval" if group in eval_groups else "train"
            for fasta, record_index, _sequence_length in groups[group]:
                row = {
                    "fasta": os.path.relpath(fasta.resolve(), output.parent.resolve()),
                    "record_index": record_index,
                    "split": split,
                    "group_id": group,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    selected_records = sum(len(groups[group]) for group in eval_groups)
    selected_bases = sum(sequence_length for group in eval_groups for _fasta, _record, sequence_length in groups[group])
    summary: dict[str, Any] = {
        "schema": "true-dna-split-v2",
        "path": str(output),
        "sha256": digest,
        "group_by": group_by,
        "groups": len(groups),
        "eval_groups": len(eval_groups),
        "records": rows_written,
        "eval_records": selected_records,
        "bases": sum(bases_by_fasta.values()),
        "eval_bases": selected_bases,
        "balance_by": balance_by,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        "per_fasta": per_fasta,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible group-aware FASTA split manifest")
    parser.add_argument("--fasta", nargs="+", required=True, help="FASTA paths or glob patterns")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL split manifest")
    parser.add_argument("--holdout-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--group-by", choices=["assembly", "header-regex", "record"], default="assembly")
    parser.add_argument("--header-regex", default=None, help="Regex with a capture group used by header-regex grouping")
    parser.add_argument(
        "--balance-by",
        choices=["bases", "records"],
        default="bases",
        help="Target holdout mass by nucleotide bases; records is retained for legacy reproduction",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_manifest(
        _expand_fastas(args.fasta),
        args.output.resolve(),
        group_by=args.group_by,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        header_regex=args.header_regex,
        balance_by=args.balance_by,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

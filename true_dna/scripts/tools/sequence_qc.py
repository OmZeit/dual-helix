#!/usr/bin/env python3
"""Filter FASTA records and report sequence-quality statistics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import TextIO

VALID_BASES = frozenset("ACGTN")
RC_TABLE = str.maketrans("ACGTN", "TGCAN")


@dataclass
class Stats:
    """Quality-control counters for one input file."""

    filename: str
    total_seqs: int = 0
    total_bases: int = 0
    removed_length: int = 0
    removed_n: int = 0
    removed_entropy: int = 0
    removed_dupe: int = 0
    removed_ambiguous: int = 0
    final_seqs: int = 0
    final_bases: int = 0
    gc_percent: float = 0.0
    n_percent: float = 0.0
    entropy: float = 0.0

    @property
    def removed_total(self) -> int:
        return self.removed_length + self.removed_n + self.removed_entropy + self.removed_dupe + self.removed_ambiguous

    @property
    def percent_removed(self) -> float:
        if self.total_seqs == 0:
            return 0.0
        return self.removed_total / self.total_seqs * 100.0


def open_maybe_gzip(path: str | Path, mode: str = "rt") -> TextIO:
    """Open a plain-text or gzip-compressed FASTA file."""

    path = str(path)
    if "b" in mode:
        raise ValueError("open_maybe_gzip only supports text modes")
    if path.lower().endswith(".gz"):
        return gzip.open(path, mode=mode, encoding="utf-8", errors="replace")
    return open(path, mode=mode, encoding="utf-8", errors="replace")


def fasta_iter(path: str | Path):
    """Yield header and sequence pairs without hiding invalid bases."""

    header: str | None = None
    seq_parts: list[str] = []

    with open_maybe_gzip(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
                continue
            if header is None:
                raise ValueError(f"{path}: sequence data before the first FASTA header at line {line_number}")
            seq_parts.append(line)

    if header is not None:
        yield header, "".join(seq_parts)


def shannon_entropy(seq: str) -> float:
    """Return per-symbol Shannon entropy in bits."""

    if not seq:
        return 0.0
    counts = Counter(seq)
    length = len(seq)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def canonical_seq_hash(seq: str) -> bytes:
    """Return a strand-invariant digest for exact deduplication."""

    reverse_complement = seq.translate(RC_TABLE)[::-1]
    canonical = min(seq, reverse_complement)
    return hashlib.sha256(canonical.encode("ascii")).digest()


def _filtered_name(path: str | Path) -> str:
    name = Path(path).name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return f"filtered_{name}"


def process_file(
    fasta_path: str | Path,
    output_dir: str | Path,
    min_length: int,
    n_threshold: float,
    min_entropy: float,
    trim_ns: bool,
    dedupe: bool,
    seen_seqs: set[bytes] | None,
) -> Stats:
    """Filter one FASTA file and write accepted records."""

    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / _filtered_name(fasta_path)
    stats = Stats(filename=fasta_path.name)
    total_gc = 0
    total_n = 0

    with filtered_path.open("w", encoding="utf-8", newline="\n") as output:
        for header, raw_seq in fasta_iter(fasta_path):
            stats.total_seqs += 1
            stats.total_bases += len(raw_seq)
            seq = raw_seq.upper()
            if trim_ns:
                seq = seq.strip("N")

            if len(seq) < min_length:
                stats.removed_length += 1
                continue

            invalid = set(seq) - VALID_BASES
            if invalid:
                stats.removed_ambiguous += 1
                continue

            n_count = seq.count("N")
            n_percent = n_count / len(seq) * 100.0 if seq else 0.0
            if n_percent > n_threshold:
                stats.removed_n += 1
                continue

            entropy = shannon_entropy(seq)
            if entropy < min_entropy:
                stats.removed_entropy += 1
                continue

            if dedupe:
                if seen_seqs is None:
                    raise ValueError("seen_seqs is required when dedupe=True")
                digest = canonical_seq_hash(seq)
                if digest in seen_seqs:
                    stats.removed_dupe += 1
                    continue
                seen_seqs.add(digest)

            stats.final_seqs += 1
            stats.final_bases += len(seq)
            stats.entropy += entropy
            total_gc += seq.count("G") + seq.count("C")
            total_n += n_count
            output.write(f">{header}\n{seq}\n")

    if stats.final_bases:
        stats.gc_percent = total_gc / stats.final_bases * 100.0
        stats.n_percent = total_n / stats.final_bases * 100.0
    if stats.final_seqs:
        stats.entropy /= stats.final_seqs
    return stats


def _write_csv(path: Path, stats: list[Stats]) -> None:
    columns = [
        "filename",
        "total_seqs",
        "final_seqs",
        "removed_total",
        "percent_removed",
        "removed_length",
        "removed_n",
        "removed_entropy",
        "removed_dupe",
        "removed_ambiguous",
        "final_bases",
        "gc_percent",
        "n_percent",
        "avg_entropy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in stats:
            writer.writerow(
                {
                    "filename": item.filename,
                    "total_seqs": item.total_seqs,
                    "final_seqs": item.final_seqs,
                    "removed_total": item.removed_total,
                    "percent_removed": f"{item.percent_removed:.2f}",
                    "removed_length": item.removed_length,
                    "removed_n": item.removed_n,
                    "removed_entropy": item.removed_entropy,
                    "removed_dupe": item.removed_dupe,
                    "removed_ambiguous": item.removed_ambiguous,
                    "final_bases": item.final_bases,
                    "gc_percent": f"{item.gc_percent:.2f}",
                    "n_percent": f"{item.n_percent:.2f}",
                    "avg_entropy": f"{item.entropy:.3f}",
                }
            )


def _summary(stats: list[Stats]) -> dict[str, object]:
    total_sequences = sum(item.total_seqs for item in stats)
    kept_sequences = sum(item.final_seqs for item in stats)
    removed = sum(item.removed_total for item in stats)
    return {
        "total_files": len(stats),
        "total_sequences_read": total_sequences,
        "total_sequences_kept": kept_sequences,
        "total_removed": removed,
        "percent_removed": removed / total_sequences * 100.0 if total_sequences else 0.0,
        "removals_by_reason": {
            "length": sum(item.removed_length for item in stats),
            "ambiguous_bases": sum(item.removed_ambiguous for item in stats),
            "n_percent": sum(item.removed_n for item in stats),
            "entropy": sum(item.removed_entropy for item in stats),
            "duplicate": sum(item.removed_dupe for item in stats),
        },
        "per_file_stats": [asdict(item) for item in stats],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter FASTA records and write quality-control reports.")
    parser.add_argument("--fasta", nargs="+", required=True, help="Input FASTA paths or glob patterns.")
    parser.add_argument("--output-dir", "--output_dir", default="qc_results")
    parser.add_argument("--min-length", "--min_length", type=int, default=100)
    parser.add_argument(
        "--n-threshold",
        "--n_threshold",
        type=float,
        default=5.0,
        help="Maximum N content as a percentage from 0 to 100.",
    )
    parser.add_argument("--min-entropy", "--min_entropy", type=float, default=1.5)
    parser.add_argument("--trim-ns", "--trim_ns", action="store_true")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate exact forward/reverse-complement records.")
    parser.add_argument("--csv", action="store_true", help="Write qc_summary.csv.")
    parser.add_argument("--json", action="store_true", help="Write summary.json.")
    args = parser.parse_args(argv)

    if args.min_length < 0:
        parser.error("--min-length must be non-negative")
    if not 0.0 <= args.n_threshold <= 100.0:
        parser.error("--n-threshold must be between 0 and 100")
    if args.min_entropy < 0.0:
        parser.error("--min-entropy must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted({match for pattern in args.fasta for match in glob(pattern)})
    if not files:
        print(f"No FASTA files matched: {', '.join(args.fasta)}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_sequences: set[bytes] | None = set() if args.dedupe else None
    all_stats: list[Stats] = []
    failures = 0

    for path in files:
        try:
            all_stats.append(
                process_file(
                    path,
                    output_dir,
                    args.min_length,
                    args.n_threshold,
                    args.min_entropy,
                    args.trim_ns,
                    args.dedupe,
                    seen_sequences,
                )
            )
        except (OSError, UnicodeError, ValueError) as exc:
            failures += 1
            print(f"Failed to process {path}: {exc}", file=sys.stderr)

    summary = _summary(all_stats)
    print(
        "Processed {total_files} files: {total_sequences_kept}/{total_sequences_read} "
        "sequences retained ({percent_removed:.2f}% removed).".format(**summary)
    )

    if args.csv:
        _write_csv(output_dir / "qc_summary.csv", all_stats)
    if args.json:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

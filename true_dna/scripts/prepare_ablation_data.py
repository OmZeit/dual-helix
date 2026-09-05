#!/usr/bin/env python3
"""Prepare a provenance-aware corpus split for controlled ablations.

With ``--download-refseq`` this delegates acquisition to the repository's
RefSeq pipeline, then creates an assembly-held-out split.  Existing FASTAs
can be supplied directly instead.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

from build_split_manifest import build_manifest

PROJECT_DIR = Path(__file__).resolve().parents[1]


def ncbi_environment(api_key: str | None, email: str | None) -> dict[str, str]:
    """Return child environment without putting credentials in command logs."""
    env = os.environ.copy()
    if api_key:
        # NCBI Datasets CLI discovers this variable automatically. Passing a
        # --api-key argument through multiple subprocesses would expose it in
        # process listings and the pipeline's printed command history.
        env["NCBI_API_KEY"] = api_key
    if email:
        env["NCBI_TAXONOMY_EMAIL"] = email
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data and a split manifest for controlled ablations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", nargs="+", help="Existing FASTA paths or glob patterns")
    source.add_argument(
        "--download-refseq", action="store_true", help="Build a fresh RefSeq corpus with the repository pipeline"
    )
    parser.add_argument("--output", type=Path, default=Path("data/domain_fastas/split_manifest.jsonl"))
    parser.add_argument("--target-mb-per-domain", type=float, default=64.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument("--assembly-source", choices=["all", "RefSeq", "GenBank"], default="RefSeq")
    parser.add_argument("--exclude-plasmids", action="store_true")
    parser.add_argument(
        "--api-key",
        default=os.getenv("NCBI_API_KEY") or os.getenv("NCBI_DATASETS_API_KEY"),
        help="NCBI API key; prefer exporting NCBI_API_KEY rather than placing a key in shell history.",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("NCBI_TAXONOMY_EMAIL"),
        help="Contact email for NCBI taxonomy requests (or set NCBI_TAXONOMY_EMAIL).",
    )
    parser.add_argument("--group-by", choices=["assembly", "header-regex", "record"], default="assembly")
    parser.add_argument("--header-regex", default=None)
    return parser.parse_args()


def _expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        files.extend(matches if matches else [Path(pattern)])
    return sorted({path.resolve() for path in files if path.is_file()})


def main() -> int:
    args = parse_args()
    if args.download_refseq:
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "prepare_refseq_pretrain_data.py"),
            "--run-download",
            "--write-manifest",
            "--target-mb-per-domain",
            str(args.target_mb_per_domain),
            "--assembly-source",
            args.assembly_source,
        ]
        if args.exclude_plasmids:
            command.append("--exclude-plasmids")
        if args.rebuild_tokenizer:
            command.append("--rebuild-tokenizer")
        subprocess.run(command, cwd=PROJECT_DIR, env=ncbi_environment(args.api_key, args.email), check=True)
        fastas = _expand([str(PROJECT_DIR / "data" / "domain_fastas" / "*.fa")])
    else:
        fastas = _expand(args.fasta)

    if not fastas:
        raise SystemExit("No FASTA files were available after preparation")

    output = args.output if args.output.is_absolute() else PROJECT_DIR / args.output
    summary = build_manifest(
        fastas,
        output.resolve(),
        group_by=args.group_by,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        header_regex=args.header_regex,
    )
    print(f"Prepared split manifest: {summary['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

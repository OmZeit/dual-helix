"""Orchestrate genome downloads, quality control, and FASTA concatenation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[Warning] tqdm not installed. Install with: pip install tqdm")

REPO_ROOT = Path(__file__).resolve().parent.parent
FETCH = REPO_ROOT / "scripts" / "fetch_genomes.py"
SEQ_QC = REPO_ROOT / "scripts" / "tools" / "sequence_qc.py"
PRINT_QC = REPO_ROOT / "scripts" / "tools" / "print_qc_summary.py"

DOMAIN_FASTAS_DIR = REPO_ROOT / "data" / "domain_fastas"

# Domain-specific QC thresholds
DOMAIN_THRESHOLDS = {
    "bacteria": {
        "min_entropy": 1.3,
        "max_removed_pct": 40.0,  # Allow up to 40% removal
    },
    "archaea": {
        "min_entropy": 1.4,
        "max_removed_pct": 40.0,
    },
    "eukaryotes": {
        "min_entropy": 1.5,  # Higher due to repeat regions
        "max_removed_pct": 50.0,  # More lenient due to repetitive DNA
    },
}


def run(
    cmd: list[str],
    cwd: Path | None = None,
    desc: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command, stream output, fail fast with helpful message."""
    if desc and HAS_TQDM:
        print(f"[Running] {desc}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"\n[error] Command failed ({proc.returncode}):\n  {' '.join(cmd)}")


def ensure_empty_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def validate_csv_header(csv_path: Path) -> None:
    """Validate CSV has correct headers."""
    try:
        first = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        first = ""
    if "species" not in first or "target" not in first:
        print(f"[warn] {csv_path.name} header should look like: 'species,target_mb' (got: {first!r})")


def validate_qc_results(domain_dir: Path, domain: str) -> tuple[bool, str]:
    """
    Validate QC results and return (success, message).
    """
    summary_path = domain_dir / "qc_results" / "summary.json"
    if not summary_path.exists():
        return False, f"QC summary not found at {summary_path}"

    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        return False, f"Failed to read QC summary: {e}"

    # Domain-specific thresholds
    thresholds = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["bacteria"])

    # Totals (support multiple key names)
    total_seqs = summary.get("total_sequences_read") or summary.get("total_sequences") or summary.get("total") or 0
    total_removed = summary.get("total_removed") or summary.get("removed_total") or summary.get("removed") or 0
    removed_pct = summary.get("percent_removed")
    if removed_pct is None and total_seqs:
        removed_pct = 100.0 * float(total_removed) / float(total_seqs)
    removed_pct = float(removed_pct or 0.0)

    if total_seqs == 0:
        return False, "No sequences were processed"

    if removed_pct > thresholds["max_removed_pct"]:
        return False, (
            f"Too many sequences removed: {removed_pct:.1f}% (threshold: {thresholds['max_removed_pct']:.1f}%)"
        )

    # Flag files where filtering removes more than 90% of the sequence.
    failed_files = []

    # Prefer a dict of {filename: info}; fall back to list of dicts
    per_file = summary.get("files") or summary.get("per_file_stats") or {}
    if isinstance(per_file, dict):
        items = per_file.items()
    elif isinstance(per_file, list):
        items = []
        for item in per_file:
            if isinstance(item, dict):
                fname = item.get("filename") or item.get("file") or item.get("name")
                if fname:
                    items.append((fname, item))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                items.append((item[0], item[1]))
    else:
        items = []

    for fname, info in items:
        if not fname or not isinstance(info, dict):
            continue
        file_total = info.get("total_seqs") or info.get("total_sequences") or info.get("total") or 0
        file_removed = info.get("removed_total") or info.get("removed_sequences") or info.get("removed") or 0
        if file_total > 0:
            file_removed_pct = 100.0 * float(file_removed) / float(file_total)
            if file_removed_pct > 90.0:
                failed_files.append(fname)

    if failed_files:
        return False, f"Files with >90% removal: {', '.join(failed_files)}"

    return True, f"QC passed: {removed_pct:.1f}% removed"


def qc_domain_dir(domain_dir: Path, domain: str, n_threshold: float) -> bool:
    """Run quality control for all FASTA files in a domain directory."""
    raws = sorted(list(domain_dir.glob("*.fa")) + list(domain_dir.glob("*.fasta")))
    if not raws:
        print(f"[warn] no raw FASTA files found in {domain_dir}")
        return False

    # Clean accidental double-filtered files from previous runs
    for bad in domain_dir.glob("qc_results/filtered_filtered_*.fa"):
        try:
            bad.unlink()
        except Exception:
            pass

    # Get domain-specific threshold
    thresholds = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["bacteria"])
    min_entropy = thresholds["min_entropy"]

    print(f"[qc] Using min_entropy={min_entropy} for {domain}")

    cmd = [
        sys.executable,
        str(SEQ_QC),
        "--n_threshold",
        str(n_threshold),
        "--min_entropy",
        str(min_entropy),
        "--fasta",
        *[str(p) for p in raws],
        "--json",
        "--csv",
    ]

    try:
        run(cmd, cwd=domain_dir, desc=f"QC for {domain}")
    except SystemExit:
        print(f"[error] QC failed for {domain_dir}")
        return False

    # Validate QC results
    success, message = validate_qc_results(domain_dir, domain)
    if success:
        print(f"[qc] PASS: {message}")
    else:
        print(f"[qc] FAILED: {message}")
        return False

    # Print per-domain QC summary
    try:
        run([sys.executable, str(PRINT_QC)], cwd=domain_dir)
    except Exception:
        pass  # Non-critical

    return True


def concat_filtered(domain_dir: Path, out_fa: Path) -> bool:
    """
    Concatenate filtered_*.fa into a single domain FASTA.

    Returns True if successful.
    """
    ensure_dir(out_fa.parent)
    filtered = sorted(domain_dir.glob("qc_results/filtered_*.fa"))
    if not filtered:
        print(f"[warn] no filtered FASTAs in {domain_dir}/qc_results; skip building {out_fa.name}")
        return False

    total_size = 0
    with out_fa.open("wb") as w:
        if HAS_TQDM:
            filtered_iter = tqdm(filtered, desc=f"Concatenating {out_fa.name}", unit="file")
        else:
            filtered_iter = filtered

        for f in filtered_iter:
            data = f.read_bytes()
            w.write(data)
            total_size += len(data)

    print(f"[ok] wrote {out_fa} ({len(filtered)} files, {total_size / 1024 / 1024:.1f} MB)")
    return True


def verify_output_quality(out_fa: Path, domain: str) -> tuple[bool, str]:
    """Verify that the concatenated FASTA is suitable for training."""
    if not out_fa.exists():
        return False, "Output file does not exist"

    # Check file size
    size_mb = out_fa.stat().st_size / 1024 / 1024
    if size_mb < 10:
        return False, f"Output file suspiciously small: {size_mb:.1f} MB"

    # Quick scan: count sequences and check for 'N's
    seq_count = 0
    n_count = 0
    total_bases = 0

    try:
        with open(out_fa) as f:
            for line in f:
                if line.startswith(">"):
                    seq_count += 1
                else:
                    bases = line.strip()
                    total_bases += len(bases)
                    n_count += bases.count("N") + bases.count("n")

                # Sample first 1000 sequences for speed
                if seq_count >= 1000:
                    break
    except Exception as e:
        return False, f"Failed to read output file: {e}"

    if seq_count == 0:
        return False, "No sequences found in output file"

    n_pct = (n_count / total_bases * 100.0) if total_bases > 0 else 0.0
    if n_pct > 1.0:
        return False, f"High N content: {n_pct:.2f}% (sampled)"

    return True, f"{seq_count}+ sequences, {size_mb:.1f} MB, {n_pct:.3f}% N's"


def pipeline_for_csv(
    csv_path: Path,
    out_root: Path,
    fresh: bool,
    levels: str,
    min_length: int,
    max_n_frac: float,
    max_assemblies: int,
    overshoot_mb: int,
    parallel_downloads: int,
    refseq_only: bool,
    fuzzy: bool,
    api_key: str | None,
    email: str | None,
    fail_on_qc_error: bool,
) -> bool:
    """Run the download and validation pipeline for one CSV manifest."""
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        return False

    validate_csv_header(csv_path)

    domain = csv_path.stem  # e.g., "bacteria", "archaea", "eukaryotes"
    domain_dir = out_root / domain

    if fresh:
        print(f"[reset] wiping {domain_dir} and starting fresh...")
        ensure_empty_dir(domain_dir)
    else:
        ensure_dir(domain_dir)

    # Build fetch command
    cmd = [
        sys.executable,
        str(FETCH),
        "--config",
        str(csv_path),
        "--levels",
        levels,
        "--min-length",
        str(min_length),
        "--max-n-frac",
        str(max_n_frac),
        "--append",
        "--overshoot-mb",
        str(overshoot_mb),
        "--max-assemblies",
        str(max_assemblies),
        "--parallel-downloads",
        str(parallel_downloads),
        "--run-qc",
        "--qc-n-threshold",
        "0.0",
        "--results-dir",
        str(domain_dir),
    ]

    if refseq_only:
        cmd.append("--refseq-only")
    if fuzzy:
        cmd.append("--fuzzy")
    child_env = os.environ.copy()
    if api_key:
        child_env["NCBI_DATASETS_API_KEY"] = api_key
    if email:
        child_env["NCBI_TAXONOMY_EMAIL"] = email

    print(f"\n{'=' * 60}")
    print(f"Processing: {csv_path.name} -> {domain_dir}")
    print(f"{'=' * 60}")

    try:
        run(cmd, cwd=REPO_ROOT, desc=f"Fetching {domain}", env=child_env)
    except SystemExit as e:
        print(f"[error] Fetch failed for {domain}: {e}")
        if fail_on_qc_error:
            raise
        return False

    # QC with validation
    print(f"\n[qc] Running quality control for {domain}...")
    qc_success = qc_domain_dir(domain_dir, domain, max_n_frac * 100.0)
    if not qc_success:
        print(f"[error] QC validation failed for {domain}")
        if fail_on_qc_error:
            raise SystemExit(f"QC validation failed for {domain}")
        return False

    # Concatenate
    print(f"\n[concat] Building domain FASTA: {domain}.fa")
    out_fa = DOMAIN_FASTAS_DIR / f"{domain}.fa"
    concat_success = concat_filtered(domain_dir, out_fa)
    if not concat_success:
        print(f"[error] Concatenation failed for {domain}")
        if fail_on_qc_error:
            raise SystemExit(f"Concatenation failed for {domain}")
        return False

    # Final verification
    print("\n[verify] Checking output quality...")
    verify_success, verify_msg = verify_output_quality(out_fa, domain)
    if verify_success:
        print(f"[verify] {verify_msg}")
    else:
        print(f"[verify] FAILED: {verify_msg}")
        if fail_on_qc_error:
            raise SystemExit(f"Output quality check failed for {domain}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Successfully completed: {domain}")
    print(f"{'=' * 60}\n")
    return True


def main():
    ap = argparse.ArgumentParser(description="Genome fetch/QC/concat pipeline with validation")
    ap.add_argument(
        "--csv", nargs="+", required=True, help="One or more CSV files (e.g., bacteria.csv archaea.csv eukaryotes.csv)"
    )
    ap.add_argument("--out-root", default="data", help="Root output directory (default: data)")
    ap.add_argument("--fresh", action="store_true", help="Start from scratch: deletes data/<domain> before running")
    ap.add_argument("--levels", default="complete,chromosome,scaffold,contig", help="Assembly levels in priority order")
    ap.add_argument("--min-length", type=int, default=1000, help="Minimum sequence length to keep (default: 1000)")
    ap.add_argument("--max-n-frac", type=float, default=0.0, help="Max fraction of Ns allowed (default: 0.0 = strict)")
    ap.add_argument(
        "--max-assemblies",
        type=int,
        default=20000,
        help="Upper bound on assemblies to consider per species (default: 20000)",
    )
    ap.add_argument(
        "--overshoot-mb", type=int, default=10, help="Extra Mb allowed past each species target (default: 10)"
    )
    ap.add_argument("--parallel-downloads", type=int, default=4, help="Parallel assembly downloads per species")
    ap.add_argument("--refseq-only", action="store_true", help="Restrict to RefSeq assemblies only")
    ap.add_argument("--no-fuzzy", dest="fuzzy", action="store_false", help="Disable fuzzy taxonomy matching")
    ap.add_argument(
        "--api-key",
        default=None,
        help="NCBI API key (prefer the NCBI_DATASETS_API_KEY environment variable to avoid shell history)",
    )
    ap.add_argument("--email", default=None, help="Contact email for NCBI taxonomy")
    ap.add_argument(
        "--fail-on-qc-error", action="store_true", help="Exit immediately if QC validation fails (default: continue)"
    )
    args = ap.parse_args()

    out_root = (REPO_ROOT / args.out_root).resolve()
    DOMAIN_FASTAS_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve API/email fallbacks from env if not given
    api_key = args.api_key or os.getenv("NCBI_DATASETS_API_KEY")
    email = args.email or os.getenv("NCBI_TAXONOMY_EMAIL")

    # Basic checks for required helper scripts
    for p in (FETCH, SEQ_QC, PRINT_QC):
        if not p.exists():
            raise SystemExit(f"[error] required script missing: {p}")

    # Process all CSVs with progress tracking
    csv_paths = [REPO_ROOT / csv for csv in args.csv]
    results = {}

    if HAS_TQDM:
        csv_iter = tqdm(csv_paths, desc="Processing domains", unit="domain")
    else:
        csv_iter = csv_paths
        print(f"\nProcessing {len(csv_paths)} domain(s)...")

    for csv_path in csv_iter:
        csv_path = csv_path.resolve()
        domain = csv_path.stem

        success = pipeline_for_csv(
            csv_path=csv_path,
            out_root=out_root,
            fresh=args.fresh,
            levels=args.levels,
            min_length=args.min_length,
            max_n_frac=args.max_n_frac,
            max_assemblies=args.max_assemblies,
            overshoot_mb=args.overshoot_mb,
            parallel_downloads=args.parallel_downloads,
            refseq_only=args.refseq_only,
            fuzzy=args.fuzzy,
            api_key=api_key,
            email=email,
            fail_on_qc_error=args.fail_on_qc_error,
        )

        results[domain] = success

    # Final summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for domain, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {domain:15s} {status}")

    print(f"\nCompleted: {success_count}/{total_count} domains")
    print(f"Output directory: {DOMAIN_FASTAS_DIR}")
    print(f"{'=' * 60}")

    if success_count < total_count:
        print("\n[Warning] Some domains failed. Review errors above.")
        if args.fail_on_qc_error:
            sys.exit(1)


if __name__ == "__main__":
    main()

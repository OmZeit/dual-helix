#!/usr/bin/env python3
"""Prepare a fresh balanced RefSeq-based corpus for full pretraining.

This script intentionally keeps destructive actions behind an explicit flag.
When enabled, deletion is restricted to generated data artifacts under the repo
root so source files and experiment checkpoints are left alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_FASTAS = REPO_ROOT / "data" / "domain_fastas"
DEFAULT_DOMAINS = ("bacteria", "archaea", "eukaryotes")
TOKENIZER_JSON = REPO_ROOT / "dna_model" / "bpe_vocab" / "tokenizer.json"


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("[run]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to operate outside repo: {resolved}")
    return resolved


def _safe_unlink(path: Path) -> None:
    resolved = _inside_repo(path)
    if resolved.exists() and resolved.is_file():
        print(f"[delete] {resolved.relative_to(REPO_ROOT)}")
        resolved.unlink()


def _safe_rmtree(path: Path) -> None:
    resolved = _inside_repo(path)
    if resolved.exists() and resolved.is_dir():
        print(f"[delete] {resolved.relative_to(REPO_ROOT)}/")
        shutil.rmtree(resolved)


def hard_delete_generated_data() -> None:
    """Delete generated corpus/tokenizer artifacts, not source or experiments."""
    DOMAIN_FASTAS.mkdir(parents=True, exist_ok=True)

    patterns = [
        "*.fa",
        "*.fasta",
        "*.index.json",
        "*_input_ids.npy",
        "*_kmer_ids.npy",
        "*_offsets.json",
    ]
    for pattern in patterns:
        for path in DOMAIN_FASTAS.glob(pattern):
            _safe_unlink(path)

    for domain in DEFAULT_DOMAINS:
        _safe_rmtree(REPO_ROOT / "data" / domain)

    _safe_unlink(TOKENIZER_JSON)


def load_species(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("species JSON must be a list of {name, domain, target_mb}")

    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        tax_id = str(item.get("tax_id", "")).strip()
        domain = str(item.get("domain", "")).strip()
        query = tax_id or name
        if not query or domain not in DEFAULT_DOMAINS:
            continue
        if "(" in name and ")" in name and not tax_id:
            print(
                f"[warn] species entry uses a parenthetical free-text name without tax_id: {name!r}. "
                "Pin this entry to an NCBI TaxID for reproducible downloads."
            )
        rows.append(
            {
                "name": query,
                "display_name": name or query,
                "tax_id": tax_id or None,
                "domain": domain,
                "target_mb": float(item.get("target_mb", 100.0)),
            }
        )
    if not rows:
        raise ValueError(f"No usable species entries found in {path}")
    return rows


def write_scaled_domain_csvs(
    species_rows: list[dict[str, Any]],
    out_dir: Path,
    target_mb_per_domain: float,
) -> tuple[list[Path], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: list[Path] = []
    summary: dict[str, Any] = {
        "target_mb_per_domain": float(target_mb_per_domain),
        "domains": {},
    }

    for domain in DEFAULT_DOMAINS:
        rows = [r for r in species_rows if r["domain"] == domain]
        if not rows:
            continue

        original_total = sum(float(r["target_mb"]) for r in rows)
        scale = float(target_mb_per_domain) / max(original_total, 1e-9)
        out_path = out_dir / f"{domain}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tax_id", "species", "target_mb"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "tax_id": row["tax_id"] or "",
                        "species": row["display_name"],
                        "target_mb": f"{float(row['target_mb']) * scale:.6f}",
                    }
                )
        csv_paths.append(out_path)
        try:
            csv_rel = str(out_path.relative_to(REPO_ROOT))
        except ValueError:
            csv_rel = str(out_path)
        summary["domains"][domain] = {
            "species": len(rows),
            "original_target_mb": original_total,
            "scaled_target_mb": float(target_mb_per_domain),
            "scale_factor": scale,
            "csv": csv_rel,
        }
        print(f"[csv] {domain}: {len(rows)} species, {original_total:.1f} MB -> {target_mb_per_domain:.1f} MB")

    summary_path = out_dir / "scaling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return csv_paths, summary


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def strict_fasta_stats(path: Path) -> dict[str, Any]:
    allowed = {ord("A"), ord("C"), ord("G"), ord("T")}
    seqs = 0
    bases = 0
    gc = 0
    with open(path, "rb") as f:
        for line_no, line in enumerate(f, start=1):
            if line.startswith(b">"):
                seqs += 1
                continue
            seq = line.strip()
            if not seq:
                continue
            bad = set(seq) - allowed
            if bad:
                chars = ", ".join(chr(b) for b in sorted(bad))
                raise ValueError(f"{path} has invalid sequence bytes at line {line_no}: {chars}")
            bases += len(seq)
            gc += seq.count(b"G") + seq.count(b"C")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sequences": seqs,
        "bases": bases,
        "mb": bases / 1_000_000.0,
        "gc_percent": (100.0 * gc / bases) if bases else 0.0,
        "sha256": sha256_file(path),
    }


def collect_source_manifests() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for domain in DEFAULT_DOMAINS:
        domain_dir = REPO_ROOT / "data" / domain
        entries: list[dict[str, Any]] = []
        for manifest_path in sorted(domain_dir.glob("*.manifest.json")):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as exc:
                entries.append(
                    {
                        "path": str(manifest_path.relative_to(REPO_ROOT)),
                        "error": str(exc),
                    }
                )
                continue
            entries.append(
                {
                    "path": str(manifest_path.relative_to(REPO_ROOT)),
                    "scientific_name": manifest.get("scientific_name"),
                    "tax_id": manifest.get("tax_id"),
                    "written_mb": manifest.get("written_mb"),
                    "num_contigs": manifest.get("num_contigs"),
                    "duplicates_found": manifest.get("duplicates_found"),
                    "assemblies_used": manifest.get("assemblies_used", []),
                    "skipped_assemblies": manifest.get("skipped_assemblies", []),
                    "assembly_selection": manifest.get("assembly_selection", {}),
                }
            )
        out[domain] = entries
    return out


def write_dataset_manifest(
    csv_summary: dict[str, Any],
    out_path: Path,
    selection_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fasta_stats = {}
    for domain in DEFAULT_DOMAINS:
        fasta_path = DOMAIN_FASTAS / f"{domain}.fa"
        if fasta_path.exists():
            fasta_stats[domain] = strict_fasta_stats(fasta_path)

    tokenizer = None
    if TOKENIZER_JSON.exists():
        tokenizer = {
            "path": str(TOKENIZER_JSON.relative_to(REPO_ROOT)),
            "bytes": TOKENIZER_JSON.stat().st_size,
            "sha256": sha256_file(TOKENIZER_JSON),
            "source_fastas": [
                str((DOMAIN_FASTAS / f"{domain}.fa").relative_to(REPO_ROOT))
                for domain in DEFAULT_DOMAINS
                if (DOMAIN_FASTAS / f"{domain}.fa").exists()
            ],
        }

    manifest = {
        "schema_version": 1,
        "source": "NCBI Datasets / RefSeq preferred",
        "selection_policy": selection_policy or {},
        "domain_fastas": fasta_stats,
        "source_manifests": collect_source_manifests(),
        "scaling": csv_summary,
        "tokenizer": tokenizer,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[manifest] wrote {out_path.relative_to(REPO_ROOT)}")
    write_dataset_card(manifest, out_path.with_name("dataset_card.md"))
    return manifest


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def write_dataset_card(manifest: dict[str, Any], out_path: Path) -> None:
    """Write a compact human-readable card beside the machine manifest."""
    policy = manifest.get("selection_policy") or {}
    fastas = manifest.get("domain_fastas") or {}
    source_manifests = manifest.get("source_manifests") or {}

    lines = [
        "# True DNA Dataset Card",
        "",
        "## Provenance",
        "",
        "- Source: NCBI Datasets / RefSeq preferred",
        "- Machine manifest: data/domain_fastas/dataset_manifest.json",
        "- Dependency file: true_dna/requirements.txt",
        "- Homology screening dependency: sourmash>=4.9.4 (optional at runtime; near-dedup is skipped with a warning if unavailable)",
        "",
        "## Selection Policy",
        "",
        f"- Assembly source: {policy.get('assembly_source', 'RefSeq')}",
        f"- Assembly version: {policy.get('assembly_version', 'latest')}",
        f"- Exclude atypical: {_fmt_bool(policy.get('exclude_atypical', True))}",
        f"- Exclude multi-isolate: {_fmt_bool(policy.get('exclude_multi_isolate', True))}",
        f"- MAG policy: {policy.get('mag', 'exclude')}",
        f"- Allow RefSeq-excluded assemblies: {_fmt_bool(policy.get('allow_excluded_from_refseq', False))}",
        f"- CheckM completeness min: {policy.get('min_checkm_completeness', 90.0)}",
        f"- CheckM contamination max: {policy.get('max_checkm_contamination', 5.0)}",
        f"- FCS policy: {policy.get('fcs_policy', 'warn')}",
        f"- Require FCS record: {_fmt_bool(policy.get('require_fcs', False))}",
        f"- Near-dedup enabled: {_fmt_bool(policy.get('near_dedup', True))}",
        f"- Near-dedup threshold: {policy.get('near_dedup_threshold', 0.98)}",
        f"- Accession lock in: {policy.get('accession_lock_in') or 'none'}",
        f"- Accession lock out: {policy.get('accession_lock_out') or 'none'}",
        "",
        "## Domain FASTAs",
        "",
    ]

    for domain, stats in sorted(fastas.items()):
        lines.extend(
            [
                f"### {domain}",
                "",
                f"- Path: {stats.get('path')}",
                f"- Bases: {stats.get('bases')}",
                f"- MB: {stats.get('mb')}",
                f"- Sequences: {stats.get('sequences')}",
                f"- GC percent: {stats.get('gc_percent')}",
                f"- SHA256: {stats.get('sha256')}",
                f"- Contig sidecar: {str(Path(stats.get('path', '')).with_suffix('.contigs.jsonl')) if stats.get('path') else 'unknown'}",
                "",
            ]
        )

    lines.extend(["## Source Manifests", ""])
    for domain, entries in sorted(source_manifests.items()):
        assemblies = sum(len(e.get("assemblies_used", [])) for e in entries if isinstance(e, dict))
        skipped = sum(len(e.get("skipped_assemblies", [])) for e in entries if isinstance(e, dict))
        lines.append(
            f"- {domain}: {len(entries)} species manifests, {assemblies} selected assemblies, {skipped} skipped assemblies"
        )

    lines.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Exact duplicate removal uses reverse-complement-canonical sequence hashes.",
            "- Near-duplicate screening uses assembly-level sourmash max-containment when sourmash is installed.",
            "- Train/eval leakage control should use data/domain_fastas/split_manifest.jsonl with scripts/train.py --split-manifest.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] wrote {out_path.relative_to(REPO_ROOT)}")


def verify_domain_sizes(manifest: dict[str, Any], target_mb: float, tolerance_frac: float) -> None:
    low = target_mb * (1.0 - tolerance_frac)
    high = target_mb * (1.0 + tolerance_frac)
    for domain in DEFAULT_DOMAINS:
        stats = manifest.get("domain_fastas", {}).get(domain)
        if not stats:
            raise ValueError(f"Missing final FASTA for {domain}")
        mb = float(stats["mb"])
        if mb < low or mb > high:
            raise ValueError(f"{domain}.fa size {mb:.1f} MB is outside expected range {low:.1f}-{high:.1f} MB")
        print(f"[verify] {domain}: {mb:.1f} MB, {stats['sequences']:,} seqs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fresh balanced RefSeq pretraining data")
    parser.add_argument("--species-json", default="configs/species.json")
    parser.add_argument("--metadata-dir", default="data/metadata/refseq_balanced")
    parser.add_argument("--target-mb-per-domain", type=float, default=4096.0)
    parser.add_argument("--hard-delete-generated-data", action="store_true")
    parser.add_argument("--run-download", action="store_true")
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--manifest-path", default="data/domain_fastas/dataset_manifest.json")
    parser.add_argument("--refseq-only", action="store_true")
    parser.add_argument("--assembly-source", choices=["all", "RefSeq", "GenBank"], default="RefSeq")
    parser.add_argument("--assembly-version", choices=["latest", "all"], default="latest")
    parser.add_argument("--exclude-atypical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-multi-isolate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mag", choices=["all", "only", "exclude"], default="exclude")
    parser.add_argument("--allow-excluded-from-refseq", action="store_true")
    parser.add_argument("--min-checkm-completeness", type=float, default=90.0)
    parser.add_argument("--max-checkm-contamination", type=float, default=5.0)
    parser.add_argument("--max-contigs", type=int, default=0)
    parser.add_argument("--max-scaffolds", type=int, default=0)
    parser.add_argument("--accession-lock-in", default=None)
    parser.add_argument("--accession-lock-out", default="data/domain_fastas/selected_accessions.lock.json")
    parser.add_argument("--allow-missing-lock-accessions", action="store_true")
    parser.add_argument("--near-dedup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--near-dedup-threshold", type=float, default=0.98)
    parser.add_argument("--near-dedup-ksize", type=int, default=31)
    parser.add_argument("--near-dedup-scaled", type=int, default=1000)
    parser.add_argument("--fcs-summary-refseq", default=None)
    parser.add_argument("--fcs-summary-genbank", default=None)
    parser.add_argument("--fcs-policy", choices=["off", "warn", "reject"], default=None)
    parser.add_argument("--require-fcs", action="store_true")
    parser.add_argument("--parallel-downloads", type=int, default=4)
    parser.add_argument("--max-assemblies", type=int, default=20000)
    parser.add_argument("--overshoot-mb", type=int, default=10)
    parser.add_argument("--min-length", type=int, default=1000)
    parser.add_argument("--max-n-frac", type=float, default=0.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--email", default=None)
    parser.add_argument("--size-tolerance-frac", type=float, default=0.20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    species_json = (REPO_ROOT / args.species_json).resolve()
    metadata_dir = (REPO_ROOT / args.metadata_dir).resolve()
    manifest_path = (REPO_ROOT / args.manifest_path).resolve()

    if args.run_download and shutil.which("datasets") is None:
        raise SystemExit("NCBI datasets CLI is required for --run-download but was not found on PATH.")

    if args.hard_delete_generated_data:
        hard_delete_generated_data()

    rows = load_species(species_json)
    csv_paths, csv_summary = write_scaled_domain_csvs(
        rows,
        metadata_dir,
        target_mb_per_domain=args.target_mb_per_domain,
    )

    if args.run_download:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_data.py"),
            "--csv",
            *[str(path) for path in csv_paths],
            "--fresh",
            "--fail-on-qc-error",
            "--parallel-downloads",
            str(args.parallel_downloads),
            "--max-assemblies",
            str(args.max_assemblies),
            "--overshoot-mb",
            str(args.overshoot_mb),
            "--min-length",
            str(args.min_length),
            "--max-n-frac",
            str(args.max_n_frac),
            "--assembly-source",
            args.assembly_source,
            "--assembly-version",
            args.assembly_version,
            "--mag",
            args.mag,
            "--min-checkm-completeness",
            str(args.min_checkm_completeness),
            "--max-checkm-contamination",
            str(args.max_checkm_contamination),
            "--max-contigs",
            str(args.max_contigs),
            "--max-scaffolds",
            str(args.max_scaffolds),
            "--near-dedup-threshold",
            str(args.near_dedup_threshold),
            "--near-dedup-ksize",
            str(args.near_dedup_ksize),
            "--near-dedup-scaled",
            str(args.near_dedup_scaled),
        ]
        if args.refseq_only:
            cmd.append("--refseq-only")
        if not args.near_dedup:
            cmd.append("--no-near-dedup")
        if args.allow_excluded_from_refseq:
            cmd.append("--allow-excluded-from-refseq")
        if not args.exclude_atypical:
            cmd.append("--no-exclude-atypical")
        if not args.exclude_multi_isolate:
            cmd.append("--no-exclude-multi-isolate")
        if args.accession_lock_in:
            cmd.extend(["--accession-lock-in", args.accession_lock_in])
        if args.accession_lock_out:
            cmd.extend(["--accession-lock-out", args.accession_lock_out])
        if args.allow_missing_lock_accessions:
            cmd.append("--allow-missing-lock-accessions")
        if args.fcs_summary_refseq:
            cmd.extend(["--fcs-summary-refseq", args.fcs_summary_refseq])
        if args.fcs_summary_genbank:
            cmd.extend(["--fcs-summary-genbank", args.fcs_summary_genbank])
        if args.fcs_policy:
            cmd.extend(["--fcs-policy", args.fcs_policy])
        if args.require_fcs:
            cmd.append("--require-fcs")
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        if args.email:
            cmd.extend(["--email", args.email])
        _run(cmd)

    if args.rebuild_tokenizer:
        fastas = [str(DOMAIN_FASTAS / f"{domain}.fa") for domain in DEFAULT_DOMAINS]
        _run([sys.executable, str(REPO_ROOT / "scripts" / "train_bpe.py"), "--fasta", *fastas])

    manifest = {}
    if args.write_manifest or args.verify:
        manifest = write_dataset_manifest(
            csv_summary,
            manifest_path,
            selection_policy={
                "assembly_source": args.assembly_source,
                "assembly_version": args.assembly_version,
                "exclude_atypical": bool(args.exclude_atypical),
                "exclude_multi_isolate": bool(args.exclude_multi_isolate),
                "mag": args.mag,
                "allow_excluded_from_refseq": bool(args.allow_excluded_from_refseq),
                "min_checkm_completeness": float(args.min_checkm_completeness),
                "max_checkm_contamination": float(args.max_checkm_contamination),
                "max_contigs": int(args.max_contigs),
                "max_scaffolds": int(args.max_scaffolds),
                "near_dedup": bool(args.near_dedup),
                "near_dedup_threshold": float(args.near_dedup_threshold),
                "near_dedup_ksize": int(args.near_dedup_ksize),
                "near_dedup_scaled": int(args.near_dedup_scaled),
                "fcs_summary_refseq": args.fcs_summary_refseq,
                "fcs_summary_genbank": args.fcs_summary_genbank,
                "fcs_policy": args.fcs_policy
                or ("reject" if (args.fcs_summary_refseq or args.fcs_summary_genbank) else "warn"),
                "require_fcs": bool(args.require_fcs),
                "accession_lock_in": args.accession_lock_in,
                "accession_lock_out": args.accession_lock_out,
                "refseq_only_legacy_flag": bool(args.refseq_only),
                "max_n_frac": float(args.max_n_frac),
                "min_length": int(args.min_length),
            },
        )

    if args.verify:
        verify_domain_sizes(manifest, args.target_mb_per_domain, args.size_tolerance_frac)
        print("[verify] strict A/C/G/T checks passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

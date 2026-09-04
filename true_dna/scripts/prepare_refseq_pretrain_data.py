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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_taxonomic_manifest import build_manifest as build_taxonomic_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_FASTAS = REPO_ROOT / "data" / "domain_fastas"
DEFAULT_DOMAINS = ("bacteria", "archaea", "eukaryotes")
TOKENIZER_JSON = REPO_ROOT / "dna_model" / "bpe_vocab" / "tokenizer.json"
EXAMPLE_CREDENTIAL_VALUES = {"paste-your-ncbi-key-here", "your-email@example.com"}


def _sanitize_cmd_for_logging(cmd: list[str]) -> str:
    """Mask sensitive argument values from logged command lines."""
    sanitized: list[str] = []
    redact_next = False
    for arg in cmd:
        if redact_next:
            sanitized.append("******")
            redact_next = False
        elif arg.lower() in ("--api-key", "--token", "--password", "--secret"):
            sanitized.append(arg)
            redact_next = True
        else:
            sanitized.append(arg)
    return " ".join(sanitized)


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    sanitized_cmd = _sanitize_cmd_for_logging(cmd)
    print("[run]", sanitized_cmd, flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed with exit code {proc.returncode}: {sanitized_cmd}")


def clean_ncbi_credential(value: str | None, env_name: str) -> str | None:
    """Do not forward copied documentation placeholders to NCBI services."""
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.casefold() in EXAMPLE_CREDENTIAL_VALUES:
        if os.getenv(env_name, "").strip().casefold() in EXAMPLE_CREDENTIAL_VALUES:
            os.environ.pop(env_name, None)
        return None
    return cleaned


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

    for domain in DEFAULT_DOMAINS:
        fasta = DOMAIN_FASTAS / f"{domain}.fa"
        managed_paths = [
            fasta,
            Path(f"{fasta}.index.json"),
            fasta.with_suffix(".contigs.jsonl"),
        ]
        for tokenizer_infix in ("", "_base"):
            managed_paths.extend(
                Path(f"{fasta}{tokenizer_infix}{suffix}")
                for suffix in ("_input_ids.npy", "_kmer_ids.npy", "_offsets.json", "_tokenizer.json")
            )
        for path in managed_paths:
            _safe_unlink(path)

    # These describe generated corpus contents and splits, not the pinned
    # taxonomic selection.  Removing them prevents a fresh run from carrying
    # forward stale provenance or a split that names old records.
    for filename in ("dataset_manifest.json", "dataset_card.md", "split_manifest.jsonl"):
        _safe_unlink(DOMAIN_FASTAS / filename)


def load_species(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("species") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("species JSON must be a list or an object with a 'species' list")

    rows: list[dict[str, Any]] = []
    for item in items:
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
                "target_mb": float(item.get("target_mb", item.get("weight", 1.0))),
                "family_tax_id": str(item.get("family_tax_id", "")) or None,
                "family_name": str(item.get("family_name", "")) or None,
                "selection_reason": str(item.get("selection_reason", "")) or None,
                "preferred_refseq_accession": str(item.get("preferred_refseq_accession", "")) or None,
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
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "tax_id",
                    "species",
                    "target_mb",
                    "family_tax_id",
                    "family_name",
                    "selection_reason",
                    "preferred_refseq_accession",
                    "domain",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "tax_id": row["tax_id"] or "",
                        "species": row["display_name"],
                        "target_mb": f"{float(row['target_mb']) * scale:.6f}",
                        "family_tax_id": row["family_tax_id"] or "",
                        "family_name": row["family_name"] or "",
                        "selection_reason": row["selection_reason"] or "",
                        "preferred_refseq_accession": row["preferred_refseq_accession"] or "",
                        "domain": domain,
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
        "- Exact reverse-complement deduplication and sequence-level quality control are applied by this pipeline.",
        "",
        "## Selection Policy",
        "",
        f"- Assembly source: {policy.get('assembly_source', 'RefSeq')}",
        f"- Minimum contig length: {policy.get('min_length', 1000)}",
        f"- Maximum N fraction before QC: {policy.get('max_n_fraction', 0.0)}",
        f"- Exclude plasmids: {_fmt_bool(policy.get('exclude_plasmids', False))}",
        f"- Exact reverse-complement deduplication: {_fmt_bool(policy.get('exact_reverse_complement_deduplication', True))}",
        f"- Entropy QC: {policy.get('entropy_qc', 'domain-specific')}",
        f"- GC policy: {policy.get('gc_filtering', 'measured and reported')}",
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
            "- Near-homology screening, CheckM, and FCS controls are not performed by this pipeline and must not be claimed from this dataset card.",
            "- Create an assembly-held-out split with scripts/prepare_ablation_data.py or scripts/build_split_manifest.py before controlled evaluation.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] wrote {out_path.relative_to(REPO_ROOT)}")


def verify_domain_sizes(
    manifest: dict[str, Any], target_mb: float, tolerance_frac: float, allowed_underfilled_domains: set[str]
) -> None:
    low = target_mb * (1.0 - tolerance_frac)
    high = target_mb * (1.0 + tolerance_frac)
    for domain in DEFAULT_DOMAINS:
        stats = manifest.get("domain_fastas", {}).get(domain)
        if not stats:
            raise ValueError(f"Missing final FASTA for {domain}")
        mb = float(stats["mb"])
        if mb <= 0:
            raise ValueError(f"{domain}.fa is empty")
        if mb < low and domain in allowed_underfilled_domains:
            print(
                f"[verify] {domain}: {mb:.1f} MB, {stats['sequences']:,} seqs "
                f"(documented underfilled domain; target range would be {low:.1f}-{high:.1f} MB)"
            )
            continue
        if mb < low or mb > high:
            raise ValueError(f"{domain}.fa size {mb:.1f} MB is outside expected range {low:.1f}-{high:.1f} MB")
        print(f"[verify] {domain}: {mb:.1f} MB, {stats['sequences']:,} seqs")


def build_download_command(
    args: argparse.Namespace, csv_paths: list[Path], storage_budget_mb_per_domain: float
) -> list[str]:
    """Build the command for filters the downloader actually implements."""
    command = [
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
        "--domain-budget-mb",
        str(storage_budget_mb_per_domain),
        "--min-length",
        str(args.min_length),
        "--max-n-frac",
        str(args.max_n_frac),
        "--assembly-source",
        args.assembly_source,
        "--scratch-dir",
        str(args.scratch_dir),
    ]
    if args.exclude_plasmids:
        command.append("--exclude-plasmids")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fresh balanced RefSeq pretraining data")
    parser.add_argument("--species-json", default="configs/species_taxonomically_diverse_v1.json")
    parser.add_argument(
        "--selection-manifest",
        default=None,
        help="Previously NCBI-validated selection manifest. Reuses its pinned TaxIDs without another taxonomy lookup.",
    )
    parser.add_argument("--metadata-dir", default="data/metadata/refseq_balanced")
    parser.add_argument("--target-mb-per-domain", type=float, default=None)
    parser.add_argument("--target-gb-per-domain", type=float, default=None)
    parser.add_argument("--hard-delete-generated-data", action="store_true")
    parser.add_argument("--run-download", action="store_true")
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--manifest-path", default="data/domain_fastas/dataset_manifest.json")
    parser.add_argument("--refseq-only", action="store_true")
    parser.add_argument("--assembly-source", choices=["all", "RefSeq", "GenBank"], default="RefSeq")
    parser.add_argument(
        "--exclude-plasmids", action="store_true", help="Discard contigs labelled as plasmids before QC"
    )
    parser.add_argument(
        "--parallel-downloads",
        type=int,
        default=4,
        help="Bounded concurrent package downloads; filtering and writes remain deterministic (default: 4).",
    )
    parser.add_argument("--max-assemblies", type=int, default=1000)
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(os.getenv("TRUE_DNA_NCBI_SCRATCH", Path.home() / ".cache" / "true_dna_ncbi")),
        help="Temporary NCBI package storage; native WSL storage is recommended.",
    )
    parser.add_argument("--overshoot-mb", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=1000)
    parser.add_argument("--max-n-frac", type=float, default=0.0)
    parser.add_argument(
        "--api-key",
        default=os.getenv("NCBI_API_KEY") or os.getenv("NCBI_DATASETS_API_KEY"),
        help="NCBI API key; prefer exporting NCBI_API_KEY rather than placing a key in shell history.",
    )
    parser.add_argument("--email", default=os.getenv("NCBI_TAXONOMY_EMAIL"))
    parser.add_argument("--size-tolerance-frac", type=float, default=0.15)
    parser.add_argument(
        "--allow-underfilled-domain",
        choices=DEFAULT_DOMAINS,
        action="append",
        default=[],
        help="Document and permit a nonempty domain below the lower size tolerance; repeat for each approved domain.",
    )
    parser.add_argument(
        "--storage-budget-tolerance-frac",
        type=float,
        default=None,
        help="Extra collection headroom above the desired domain target. Defaults to --size-tolerance-frac.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.api_key = clean_ncbi_credential(args.api_key, "NCBI_API_KEY")
    args.email = clean_ncbi_credential(args.email, "NCBI_TAXONOMY_EMAIL")
    args.scratch_dir = args.scratch_dir.expanduser().resolve()
    if args.target_mb_per_domain is not None and args.target_gb_per_domain is not None:
        raise SystemExit("Use only one of --target-mb-per-domain or --target-gb-per-domain")
    if args.target_mb_per_domain is not None:
        target_mb_per_domain = float(args.target_mb_per_domain)
    elif args.target_gb_per_domain is not None:
        target_mb_per_domain = float(args.target_gb_per_domain) * 1_000.0
    else:
        target_mb_per_domain = 5_000.0
    if target_mb_per_domain <= 0:
        raise SystemExit("The per-domain target must be positive")
    if not 0.0 <= args.size_tolerance_frac < 1.0:
        raise SystemExit("--size-tolerance-frac must be in [0, 1)")
    if args.storage_budget_tolerance_frac is None:
        storage_budget_tolerance_frac = float(args.size_tolerance_frac)
    else:
        storage_budget_tolerance_frac = float(args.storage_budget_tolerance_frac)
    if not 0.0 <= storage_budget_tolerance_frac < 1.0:
        raise SystemExit("--storage-budget-tolerance-frac must be in [0, 1)")
    storage_budget_mb_per_domain = target_mb_per_domain * (1.0 + storage_budget_tolerance_frac)
    if args.refseq_only and args.assembly_source == "GenBank":
        raise SystemExit("--refseq-only conflicts with --assembly-source GenBank")
    if args.refseq_only:
        args.assembly_source = "RefSeq"
    species_json = (REPO_ROOT / args.species_json).resolve()
    metadata_dir = (REPO_ROOT / args.metadata_dir).resolve()
    manifest_path = (REPO_ROOT / args.manifest_path).resolve()

    if args.run_download and shutil.which("datasets") is None:
        raise SystemExit("NCBI datasets CLI is required for --run-download but was not found on PATH.")

    if args.hard_delete_generated_data:
        hard_delete_generated_data()

    if args.selection_manifest:
        selection_path = (REPO_ROOT / args.selection_manifest).resolve()
        if not selection_path.is_file():
            raise SystemExit(f"Selection manifest not found: {selection_path}")
        with selection_path.open(encoding="utf-8") as handle:
            selection = json.load(handle)
        if not selection.get("selection_policy", {}).get("tax_id_pinned") or not isinstance(
            selection.get("species"), list
        ):
            raise SystemExit(f"Selection manifest is not a TaxID-pinned manifest: {selection_path}")
    else:
        selection_path = metadata_dir / "taxonomic_selection_manifest.json"
        selection = build_taxonomic_manifest(
            species_json,
            selection_path,
            api_key=args.api_key,
            email=args.email,
        )
    rows = load_species(selection_path)
    csv_paths, csv_summary = write_scaled_domain_csvs(
        rows,
        metadata_dir,
        target_mb_per_domain=target_mb_per_domain,
    )

    if args.run_download:
        cmd = build_download_command(args, csv_paths, storage_budget_mb_per_domain)
        child_env = os.environ.copy()
        if args.api_key:
            child_env["NCBI_API_KEY"] = args.api_key
        if args.email:
            child_env["NCBI_TAXONOMY_EMAIL"] = args.email
        _run(cmd, env=child_env)

    if args.rebuild_tokenizer:
        fastas = [str(DOMAIN_FASTAS / f"{domain}.fa") for domain in DEFAULT_DOMAINS]
        _run([sys.executable, str(REPO_ROOT / "scripts" / "train_bpe.py"), "--fasta", *fastas])

    manifest = {}
    if args.write_manifest or args.verify:
        allowed_underfilled_domains = sorted(set(args.allow_underfilled_domain))
        manifest = write_dataset_manifest(
            csv_summary,
            manifest_path,
            selection_policy={
                "assembly_source": args.assembly_source,
                "min_length": int(args.min_length),
                "max_n_fraction": float(args.max_n_frac),
                "exclude_plasmids": bool(args.exclude_plasmids),
                "exact_reverse_complement_deduplication": True,
                "entropy_qc": "domain-specific threshold (1.3 bacteria, 1.4 archaea, 1.5 eukaryotes)",
                "gc_filtering": "not applied; GC is measured and recorded per domain",
                "refseq_only_legacy_flag": bool(args.refseq_only),
                "domain_storage_budget_mb": target_mb_per_domain,
                "domain_size_tolerance_fraction": float(args.size_tolerance_frac),
                "domain_hard_storage_ceiling_mb": storage_budget_mb_per_domain,
                "domain_storage_budget_tolerance_fraction": storage_budget_tolerance_frac,
                "allowed_underfilled_domains": allowed_underfilled_domains,
                "taxonomic_selection_manifest": str(selection_path.relative_to(REPO_ROOT)),
                "taxonomic_selection_config_sha256": selection.get("selection_config_sha256"),
                "max_n_frac": float(args.max_n_frac),
            },
        )

    if args.verify:
        verify_domain_sizes(
            manifest,
            target_mb_per_domain,
            args.size_tolerance_frac,
            set(args.allow_underfilled_domain),
        )
        print("[verify] strict A/C/G/T checks passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

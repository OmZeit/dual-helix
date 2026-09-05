#!/usr/bin/env python3
"""Audit a fixed corpus split and create a taxonomically held-out split.

This tool is deliberately checkpoint-agnostic. It proves properties of the
corpus and split without modifying FASTA files or prior training artifacts:

* streams every record to find exact forward/reverse-complement duplicates
  that cross the supplied train/evaluation assignment;
* checks that every record is assigned exactly once;
* maps assembly accessions to the TaxID-pinned corpus manifest; and
* writes a deterministic family-held-out split for a *future* training run.

The family-held-out split must not be used to claim generalization for a
checkpoint that was already trained on the existing assembly-held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "true-dna-integrity-audit-v1"
ASSEMBLY_RE = re.compile(r"(?:^|\|)assembly=([^|]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sequence_hash(sequence: str) -> bytes:
    normalized = sequence.upper()
    reverse_complement = normalized.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return hashlib.sha256(min(normalized, reverse_complement).encode("ascii")).digest()


def fasta_records(path: Path) -> Iterator[tuple[int, str, str]]:
    """Yield zero-based record index, header, and sequence without loading a FASTA."""
    header: str | None = None
    chunks: list[str] = []
    record_index = -1
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield record_index, header, "".join(chunks)
                record_index += 1
                header = line[1:]
                chunks = []
            elif header is not None:
                chunks.append(line)
    if header is not None:
        yield record_index, header, "".join(chunks)


def expand_fastas(values: Sequence[str]) -> list[Path]:
    files = sorted({Path(value).expanduser().resolve() for value in values})
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FASTA files not found: {', '.join(missing)}")
    if not files:
        raise ValueError("At least one FASTA file is required")
    return files


def load_assignments(split_manifest: Path, fastas: Sequence[Path]) -> dict[Path, dict[int, str]]:
    requested = {path.resolve() for path in fastas}
    assignments: dict[Path, dict[int, str]] = {path: {} for path in requested}
    with split_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                raw_path = Path(str(row["fasta"])).expanduser()
                path = (raw_path if raw_path.is_absolute() else split_manifest.parent / raw_path).resolve()
                record_index = int(row["record_index"])
                split = str(row["split"]).lower()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid split-manifest row {line_number}: {exc}") from exc
            if path not in requested:
                continue
            if record_index < 0 or split not in {"train", "eval"}:
                raise ValueError(f"Invalid split-manifest row {line_number}")
            if record_index in assignments[path]:
                raise ValueError(f"Duplicate split assignment for {path} record {record_index}")
            assignments[path][record_index] = split
    missing = [str(path) for path, rows in assignments.items() if not rows]
    if missing:
        raise ValueError(f"Split manifest has no assignments for: {', '.join(missing)}")
    return assignments


def load_assembly_taxonomy(selection_manifest: Path, dataset_manifest: Path) -> dict[str, dict[str, str]]:
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    taxa = {str(item["tax_id"]): item for item in selection.get("species", [])}
    if not taxa:
        raise ValueError(f"No selected species found in {selection_manifest}")

    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, str]] = {}
    for domain, source_manifests in (dataset.get("source_manifests") or {}).items():
        for source in source_manifests or []:
            tax_id = str(source.get("tax_id") or "")
            taxon = taxa.get(tax_id)
            if taxon is None:
                raise ValueError(
                    f"Dataset assembly provenance references TaxID {tax_id!r} absent from selection manifest"
                )
            for assembly in source.get("assemblies_used") or []:
                accession = str(assembly.get("accession") or "")
                if not accession:
                    continue
                value = {
                    "domain": str(taxon.get("domain") or domain),
                    "tax_id": tax_id,
                    "family_tax_id": str(taxon.get("family_tax_id") or ""),
                    "family_name": str(taxon.get("family_name") or ""),
                    "scientific_name": str(taxon.get("scientific_name") or ""),
                }
                if not value["family_tax_id"]:
                    raise ValueError(f"TaxID {tax_id} has no family TaxID in {selection_manifest}")
                previous = mapping.get(accession)
                if previous is not None and previous != value:
                    raise ValueError(f"Assembly {accession} maps to conflicting taxa in dataset provenance")
                mapping[accession] = value
    if not mapping:
        raise ValueError(f"No assembly provenance found in {dataset_manifest}")
    return mapping


def assembly_from_header(header: str, *, fasta: Path, record_index: int) -> str:
    match = ASSEMBLY_RE.search(header)
    if match is None:
        raise ValueError(f"Missing assembly= accession in {fasta} record {record_index}")
    return match.group(1)


def deterministic_eval_families(families: Sequence[str], *, domain: str, fraction: float, seed: int) -> set[str]:
    unique = sorted(set(families))
    if len(unique) < 2:
        raise ValueError(f"{domain} needs at least two distinct families for a held-out split")
    count = min(len(unique) - 1, max(1, round(len(unique) * fraction)))
    ranked = sorted(
        unique,
        key=lambda family: hashlib.sha256(f"{seed}|{domain}|{family}".encode()).hexdigest(),
    )
    return set(ranked[:count])


def build_audit(
    *,
    fastas: Sequence[Path],
    split_manifest: Path,
    selection_manifest: Path,
    dataset_manifest: Path,
    output_dir: Path,
    family_holdout_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 < family_holdout_fraction < 1.0:
        raise ValueError("family_holdout_fraction must be in (0, 1)")
    assignments = load_assignments(split_manifest, fastas)
    assembly_taxonomy = load_assembly_taxonomy(selection_manifest, dataset_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    hashes: dict[bytes, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    cross_split_examples: list[dict[str, Any]] = []
    per_fasta: dict[str, dict[str, int]] = {}
    unmatched_assemblies: list[str] = []

    for fasta in fastas:
        records = 0
        split_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        for record_index, header, sequence in fasta_records(fasta):
            split = assignments[fasta].get(record_index)
            if split is None:
                raise ValueError(f"Split manifest leaves {fasta} record {record_index} unassigned")
            assembly = assembly_from_header(header, fasta=fasta, record_index=record_index)
            taxon = assembly_taxonomy.get(assembly)
            if taxon is None:
                unmatched_assemblies.append(assembly)
                continue
            if taxon["domain"] != fasta.stem:
                raise ValueError(f"Assembly {assembly} is mapped to {taxon['domain']}, but occurs in {fasta.name}")
            record = {
                "fasta": fasta,
                "record_index": record_index,
                "split": split,
                "assembly": assembly,
                **taxon,
            }
            all_records.append(record)
            records += 1
            split_counts[split] += 1
            family_counts[taxon["family_tax_id"]] += 1

            digest = canonical_sequence_hash(sequence)
            first = hashes.get(digest)
            if first is None:
                hashes[digest] = record
            else:
                key = "same_split" if first["split"] == split else "cross_split"
                duplicate_counts[key] += 1
                if key == "cross_split" and len(cross_split_examples) < 25:
                    cross_split_examples.append(
                        {
                            "sha256": digest.hex(),
                            "first": {
                                "fasta": first["fasta"].name,
                                "record_index": first["record_index"],
                                "split": first["split"],
                                "assembly": first["assembly"],
                            },
                            "duplicate": {
                                "fasta": fasta.name,
                                "record_index": record_index,
                                "split": split,
                                "assembly": assembly,
                            },
                        }
                    )
        expected = len(assignments[fasta])
        if records != expected:
            raise ValueError(
                f"Assembly provenance did not cover all {fasta.name} records: matched {records}, expected {expected}. "
                f"Examples: {sorted(set(unmatched_assemblies))[:5]}"
            )
        per_fasta[str(fasta)] = {
            "records": records,
            "train_records": split_counts["train"],
            "eval_records": split_counts["eval"],
            "families": len(family_counts),
        }

    if unmatched_assemblies:
        raise ValueError(f"Unmatched assembly accessions: {sorted(set(unmatched_assemblies))[:5]}")

    family_split_path = output_dir / "family_heldout_split.jsonl"
    family_summary: dict[str, dict[str, Any]] = {}
    family_eval_by_domain: dict[str, set[str]] = {}
    for domain in sorted({record["domain"] for record in all_records}):
        families = [record["family_tax_id"] for record in all_records if record["domain"] == domain]
        selected = deterministic_eval_families(
            families,
            domain=domain,
            fraction=family_holdout_fraction,
            seed=seed,
        )
        family_eval_by_domain[domain] = selected
        family_names = {
            record["family_tax_id"]: record["family_name"] for record in all_records if record["domain"] == domain
        }
        family_summary[domain] = {
            "families": len(set(families)),
            "eval_families": len(selected),
            "train_families": len(set(families) - selected),
            "eval_family_tax_ids": sorted(selected),
            "eval_family_names": [family_names[family] for family in sorted(selected)],
        }

    metadata = {
        "schema": "true-dna-split-v1",
        "group_by": "family",
        "holdout_fraction": family_holdout_fraction,
        "seed": seed,
        "selection_manifest_sha256": sha256_file(selection_manifest),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "warning": (
            "This split is for future training only. Existing checkpoints trained on the assembly split "
            "already saw records from these held-out families."
        ),
    }
    with family_split_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# " + json.dumps(metadata, sort_keys=True) + "\n")
        for record in sorted(all_records, key=lambda item: (str(item["fasta"]), int(item["record_index"]))):
            split = "eval" if record["family_tax_id"] in family_eval_by_domain[record["domain"]] else "train"
            row = {
                "fasta": os.path.relpath(record["fasta"], family_split_path.parent),
                "record_index": record["record_index"],
                "split": split,
                "group_id": f"family:{record['family_tax_id']}",
                "tax_id": record["tax_id"],
                "family_tax_id": record["family_tax_id"],
                "family_name": record["family_name"],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema": SCHEMA,
        "source_split_manifest": str(split_manifest),
        "source_split_manifest_sha256": sha256_file(split_manifest),
        "fastas": {str(path): {"sha256": sha256_file(path)} for path in fastas},
        "records": len(all_records),
        "unique_canonical_sequences": len(hashes),
        "exact_duplicates": {
            "same_split_record_pairs": duplicate_counts["same_split"],
            "cross_split_record_pairs": duplicate_counts["cross_split"],
            "cross_split_examples": cross_split_examples,
        },
        "per_fasta": per_fasta,
        "family_heldout_split": {
            "path": str(family_split_path),
            "sha256": sha256_file(family_split_path),
            "summary": family_summary,
            "not_valid_for_existing_checkpoints": True,
        },
    }
    report_path = output_dir / "integrity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corpus split integrity and build a family-held-out split")
    parser.add_argument("--fasta", nargs="+", required=True, help="Final domain FASTA paths")
    parser.add_argument("--split-manifest", type=Path, required=True, help="Existing assembly-held-out manifest")
    parser.add_argument("--selection-manifest", type=Path, required=True, help="TaxID/family selection manifest")
    parser.add_argument("--dataset-manifest", type=Path, required=True, help="Dataset provenance manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit(
        fastas=expand_fastas(args.fasta),
        split_manifest=args.split_manifest.expanduser().resolve(),
        selection_manifest=args.selection_manifest.expanduser().resolve(),
        dataset_manifest=args.dataset_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        family_holdout_fraction=args.family_holdout_fraction,
        seed=args.seed,
    )
    duplicates = report["exact_duplicates"]
    print(f"Wrote {Path(report['family_heldout_split']['path']).parent / 'integrity_report.json'}")
    print(
        "Exact reverse-complement duplicate pairs: "
        f"same split={duplicates['same_split_record_pairs']}, cross split={duplicates['cross_split_record_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

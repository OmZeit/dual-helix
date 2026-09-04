#!/usr/bin/env python3
"""Validate a TaxID-pinned corpus selection against NCBI Taxonomy.

The input is a small, reviewed set of candidate species.  NCBI is the source
of truth for the scientific name, superkingdom, and family.  A manifest is
written only when each selected candidate belongs to the configured domain and
no two candidates in that domain belong to the same family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import defusedxml.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
EUTILS_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DOMAIN_BY_SUPERKINGDOM = {
    "bacteria": "bacteria",
    "archaea": "archaea",
    "eukaryota": "eukaryotes",
}
TAXONOMY_RECORD_CACHE: dict[str, dict[str, str]] = {}
EXAMPLE_CREDENTIAL_VALUES = {"paste-your-ncbi-key-here", "your-email@example.com"}


def optional_credential(value: str | None) -> str | None:
    """Treat copied documentation placeholders as unset credentials."""
    cleaned = str(value or "").strip()
    return None if not cleaned or cleaned.casefold() in EXAMPLE_CREDENTIAL_VALUES else cleaned


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_tax_id(value: object) -> str:
    tax_id = str(value or "").strip()
    if not tax_id.isdecimal():
        raise ValueError(f"TaxID must be a positive integer, got {value!r}")
    return tax_id


def _preferred_refseq_accession(value: object) -> str | None:
    """Validate an optional audited RefSeq accession pinned by the selection config."""
    accession = str(value or "").strip()
    if not accession:
        return None
    if not re.fullmatch(r"GCF_\d+\.\d+", accession):
        raise ValueError(
            "preferred_refseq_accession must be a versioned RefSeq assembly accession "
            f"(for example GCF_000146045.2), got {accession!r}"
        )
    return accession


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load a versioned candidate config. The final manifest pins its TaxIDs."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("species") if isinstance(raw, Mapping) else raw
    if not isinstance(items, list):
        raise ValueError("Selection config must be a list or an object containing a 'species' list")

    candidates: list[dict[str, Any]] = []
    for priority, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Selection item {priority} must be an object")
        domain = str(item.get("domain") or "").strip().lower()
        if domain not in set(DOMAIN_BY_SUPERKINGDOM.values()):
            raise ValueError(f"Selection item {priority} has unsupported domain {domain!r}")
        configured_tax_id = str(item.get("tax_id") or "").strip()
        if configured_tax_id:
            configured_tax_id = _as_tax_id(configured_tax_id)
        configured_name = str(item.get("scientific_name") or item.get("name") or "").strip()
        if not configured_tax_id and not configured_name:
            raise ValueError(f"Selection item {priority} needs a tax_id or scientific_name")
        reason = str(item.get("selection_reason") or "one representative species from its NCBI family").strip()
        preferred_refseq_accession = _preferred_refseq_accession(item.get("preferred_refseq_accession"))
        candidates.append(
            {
                "tax_id": configured_tax_id or None,
                "configured_name": configured_name,
                "domain": domain,
                "weight": float(item.get("weight", item.get("target_mb", 1.0))),
                "selection_reason": reason,
                "selection_priority": priority,
                "preferred_refseq_accession": preferred_refseq_accession,
            }
        )

    if not candidates:
        raise ValueError("Selection config has no candidates")
    if any(candidate["weight"] <= 0 for candidate in candidates):
        raise ValueError("Every selection weight must be positive")
    return candidates


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _safe_http_status(exc: requests.RequestException) -> int | None:
    """Extract a status code without formatting a credential-bearing request URL."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


def _taxonomy_request(ids: list[str], api_key: str | None, email: str | None) -> str:
    params = {"db": "taxonomy", "id": ",".join(ids), "retmode": "xml", "tool": "true_dna_corpus_builder"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    last_status: int | None = None
    for attempt in range(4):
        try:
            response = requests.get(EUTILS_EFETCH, params=params, timeout=45)
            if response.status_code == 429 and attempt < 3:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_status = _safe_http_status(exc)
            if attempt < 3:
                time.sleep(2**attempt)
    status = f" (HTTP {last_status})" if last_status is not None else ""
    raise RuntimeError(f"NCBI taxonomy request failed for TaxID(s) {','.join(ids)}{status}") from None


def _taxonomy_search(scientific_name: str, api_key: str | None, email: str | None) -> list[str]:
    """Find exact scientific-name candidates using the public NCBI taxonomy API."""
    params = {
        "db": "taxonomy",
        "term": f'"{scientific_name}"[Scientific Name]',
        "retmode": "json",
        "retmax": "20",
        "tool": "true_dna_corpus_builder",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    url = EUTILS_EFETCH.rsplit("/", 1)[0] + "/esearch.fcgi"
    last_status: int | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=45)
            if response.status_code == 429 and attempt < 3:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return [str(value) for value in response.json().get("esearchresult", {}).get("idlist", [])]
        except requests.RequestException as exc:
            last_status = _safe_http_status(exc)
            if attempt < 3:
                time.sleep(2**attempt)
    status = f" (HTTP {last_status})" if last_status is not None else ""
    raise RuntimeError(f"NCBI taxonomy search failed for {scientific_name!r}{status}") from None


def _taxon_from_xml(node: ET.Element) -> dict[str, str]:
    lineage = []
    lineage_node = node.find("LineageEx")
    if lineage_node is not None:
        lineage.extend(lineage_node.findall("Taxon"))
    lineage.append(node)

    ranks: dict[str, dict[str, str]] = {}
    for taxon in lineage:
        rank = (taxon.findtext("Rank") or "").strip().lower()
        if rank:
            ranks[rank] = {
                "tax_id": (taxon.findtext("TaxId") or "").strip(),
                "scientific_name": (taxon.findtext("ScientificName") or "").strip(),
            }
    family = ranks.get("family")
    # NCBI's current taxonomy calls the three top-level cellular groups
    # "domain"; older records and APIs call the same rank "superkingdom".
    superkingdom = ranks.get("superkingdom") or ranks.get("domain")
    return {
        "tax_id": (node.findtext("TaxId") or "").strip(),
        "scientific_name": (node.findtext("ScientificName") or "").strip(),
        "rank": (node.findtext("Rank") or "").strip(),
        "parent_tax_id": (node.findtext("ParentTaxId") or "").strip(),
        "family_tax_id": family["tax_id"] if family else "",
        "family_name": family["scientific_name"] if family else "",
        "superkingdom_tax_id": superkingdom["tax_id"] if superkingdom else "",
        "superkingdom_name": superkingdom["scientific_name"] if superkingdom else "",
    }


def fetch_taxonomy_records(
    tax_ids: list[str], api_key: str | None = None, email: str | None = None
) -> dict[str, dict[str, str]]:
    """Fetch NCBI taxonomy records in stable, rate-limited batches."""
    requested = list(dict.fromkeys(tax_ids))
    records = {tax_id: TAXONOMY_RECORD_CACHE[tax_id] for tax_id in requested if tax_id in TAXONOMY_RECORD_CACHE}
    for ids in _chunks([tax_id for tax_id in requested if tax_id not in records], 50):
        root = ET.fromstring(_taxonomy_request(ids, api_key, email))
        for node in root.findall("Taxon"):
            record = _taxon_from_xml(node)
            if record["tax_id"]:
                records[record["tax_id"]] = record
                TAXONOMY_RECORD_CACHE[record["tax_id"]] = record
        # Remain below NCBI's no-key request limit even across batch requests.
        if not api_key:
            time.sleep(0.34)
    missing = sorted(set(requested) - set(records))
    if missing:
        raise ValueError(f"NCBI returned no taxonomy record for TaxID(s): {', '.join(missing)}")
    return records


def nearest_species_ancestor(tax_id: str, api_key: str | None, email: str | None) -> dict[str, str] | None:
    """Follow NCBI parent IDs until finding the enclosing species record."""
    current = tax_id
    visited: set[str] = set()
    for _ in range(20):
        if not current or current in {"0", "1"} or current in visited:
            return None
        visited.add(current)
        record = fetch_taxonomy_records([current], api_key, email)[current]
        if record["rank"].casefold() == "species":
            return record
        current = record.get("parent_tax_id", "")
    return None


def resolve_candidate_tax_ids(
    candidates: list[dict[str, Any]], api_key: str | None, email: str | None
) -> list[dict[str, Any]]:
    """Resolve only unpinned candidates, then pin their IDs in the manifest.

    Configured TaxIDs are intentionally fetched in a later batch and validated
    against rank, domain, family, and current NCBI scientific name.  This is
    both reproducible and far quicker than a separate name-search request for
    each of hundreds of already-pinned candidates.
    """
    configured_ids = [str(candidate["tax_id"]) for candidate in candidates if candidate.get("tax_id")]
    configured_records = fetch_taxonomy_records(configured_ids, api_key, email) if configured_ids else {}
    resolved: list[dict[str, Any]] = []
    for candidate in candidates:
        expected_domain = candidate["domain"]
        configured_tax_id = str(candidate.get("tax_id") or "")
        record = configured_records.get(configured_tax_id)
        if record:
            actual_domain = DOMAIN_BY_SUPERKINGDOM.get(record["superkingdom_name"].casefold())
            if (
                record["rank"].casefold() == "species"
                and actual_domain == expected_domain
                and record.get("family_tax_id")
            ):
                resolved.append(candidate)
                continue
            if actual_domain == expected_domain and record["rank"].casefold() in {
                "strain",
                "subspecies",
                "isolate",
                "no rank",
            }:
                species_ancestor = nearest_species_ancestor(configured_tax_id, api_key, email)
                if species_ancestor:
                    ancestor_domain = DOMAIN_BY_SUPERKINGDOM.get(species_ancestor["superkingdom_name"].casefold())
                    if ancestor_domain == expected_domain and species_ancestor.get("family_tax_id"):
                        resolved.append(
                            {
                                **candidate,
                                "tax_id": species_ancestor["tax_id"],
                                "configured_tax_id": configured_tax_id,
                                "tax_id_resolution": "nearest NCBI species ancestor",
                            }
                        )
                        continue
        name = candidate["configured_name"]
        ids = _taxonomy_search(name, api_key, email)
        records = fetch_taxonomy_records(ids, api_key, email) if ids else {}
        exact = [
            record
            for record in records.values()
            if (
                record["scientific_name"].casefold() == name.casefold()
                and record["rank"].casefold() == "species"
                and DOMAIN_BY_SUPERKINGDOM.get(record["superkingdom_name"].casefold()) == expected_domain
                and record.get("family_tax_id")
            )
        ]
        if len(exact) != 1:
            found = ", ".join(
                f"{record['scientific_name']} (TaxID {record['tax_id']}, rank={record['rank']})"
                for record in records.values()
            )
            raise ValueError(f"Expected one exact NCBI species match for {name!r}; found: {found or 'none'}")
        record = exact[0]
        resolved.append(
            {
                **candidate,
                "tax_id": record["tax_id"],
                "configured_tax_id": configured_tax_id or None,
                "tax_id_resolution": "NCBI exact scientific-name lookup",
            }
        )
        # Keep the public service within the no-key rate limit.
        if not api_key:
            time.sleep(0.34)
    resolved_ids = [str(candidate["tax_id"]) for candidate in resolved]
    if len(set(resolved_ids)) != len(resolved_ids):
        duplicate_ids = sorted({tax_id for tax_id in resolved_ids if resolved_ids.count(tax_id) > 1})
        detail = "; ".join(
            f"TaxID {tax_id}: "
            + ", ".join(candidate["configured_name"] for candidate in resolved if candidate["tax_id"] == tax_id)
            for tax_id in duplicate_ids
        )
        raise ValueError(f"Two configured candidates resolve to the same NCBI TaxID: {detail}")
    return resolved


def validate_selection(
    candidates: list[dict[str, Any]], taxonomy_records: Mapping[str, Mapping[str, str]]
) -> list[dict[str, Any]]:
    """Attach NCBI lineage metadata and enforce one family per domain."""
    selected: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_families: dict[tuple[str, str], dict[str, Any]] = {}

    for candidate in candidates:
        taxon = taxonomy_records[candidate["tax_id"]]
        if str(taxon.get("rank") or "").casefold() != "species":
            errors.append(
                f"TaxID {candidate['tax_id']} ({taxon.get('scientific_name')}) has rank "
                f"{taxon.get('rank')!r}, not species."
            )
            continue
        superkingdom = str(taxon.get("superkingdom_name") or "").strip().lower()
        actual_domain = DOMAIN_BY_SUPERKINGDOM.get(superkingdom)
        if actual_domain != candidate["domain"]:
            errors.append(
                f"TaxID {candidate['tax_id']} ({taxon.get('scientific_name')}) is in "
                f"{taxon.get('superkingdom_name') or 'an unknown superkingdom'}, not {candidate['domain']}"
            )
            continue

        family_tax_id = str(taxon.get("family_tax_id") or "")
        family_name = str(taxon.get("family_name") or "")
        if not family_tax_id or not family_name:
            errors.append(
                f"TaxID {candidate['tax_id']} ({taxon.get('scientific_name')}) has no NCBI family lineage; "
                "choose a species with an assigned family."
            )
            continue
        family_key = (actual_domain, family_tax_id)
        if family_key in seen_families:
            previous = seen_families[family_key]
            errors.append(
                f"Duplicate family in {actual_domain}: {family_name} (TaxID {family_tax_id}) is represented by "
                f"both {previous['scientific_name']} (TaxID {previous['tax_id']}) and "
                f"{taxon.get('scientific_name')} (TaxID {candidate['tax_id']})."
            )
            continue

        item = {
            **candidate,
            "scientific_name": str(taxon.get("scientific_name") or candidate["configured_name"]),
            "taxonomic_rank": str(taxon.get("rank") or ""),
            "family_tax_id": family_tax_id,
            "family_name": family_name,
            "superkingdom_tax_id": str(taxon.get("superkingdom_tax_id") or ""),
            "superkingdom_name": str(taxon.get("superkingdom_name") or ""),
            "taxonomy_url": f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={candidate['tax_id']}",
        }
        selected.append(item)
        seen_families[family_key] = item
    if errors:
        raise ValueError("Taxonomic selection rejected:\n- " + "\n- ".join(errors))
    return selected


def build_manifest(
    config_path: Path,
    output_path: Path,
    *,
    api_key: str | None = None,
    email: str | None = None,
    taxonomy_fetcher: Callable[[list[str], str | None, str | None], dict[str, dict[str, str]]] = fetch_taxonomy_records,
) -> dict[str, Any]:
    candidates = resolve_candidate_tax_ids(load_candidates(config_path), api_key, email)
    records = taxonomy_fetcher([candidate["tax_id"] for candidate in candidates], api_key, email)
    selected = validate_selection(candidates, records)
    counts = {
        domain: sum(1 for item in selected if item["domain"] == domain)
        for domain in sorted(DOMAIN_BY_SUPERKINGDOM.values())
    }
    manifest = {
        "schema_version": 1,
        "selection_policy": {
            "source": "NCBI Taxonomy E-utilities",
            "tax_id_pinned": True,
            "configured_scientific_names_reverified_against_ncbi": True,
            "one_representative_per_family_per_domain": True,
            "selection_order": "configuration order",
            "taxonomy_query": "E-utilities efetch db=taxonomy",
        },
        "selection_config": str(config_path),
        "selection_config_sha256": _config_sha256(config_path),
        "domain_counts": counts,
        "species": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a family-diverse, TaxID-pinned NCBI corpus selection")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "species_taxonomically_diverse_v1.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "metadata" / "taxonomic_selection_v1.json")
    parser.add_argument("--api-key", default=os.getenv("NCBI_API_KEY"))
    parser.add_argument("--email", default=os.getenv("NCBI_TAXONOMY_EMAIL"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.api_key = optional_credential(args.api_key)
    args.email = optional_credential(args.email)
    config = args.config if args.config.is_absolute() else (REPO_ROOT / args.config)
    output = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    manifest = build_manifest(config.resolve(), output.resolve(), api_key=args.api_key, email=args.email)
    print(f"Wrote {output}: {len(manifest['species'])} TaxID-pinned species")
    for domain, count in manifest["domain_counts"].items():
        print(f"  {domain}: {count} distinct NCBI families")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

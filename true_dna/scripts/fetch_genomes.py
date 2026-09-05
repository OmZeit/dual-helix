#!/usr/bin/env python3
"""Fetch and quality-filter genomic FASTA data from NCBI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import requests
from rapidfuzz import fuzz, process

try:
    from dna_model.tokenizer import reverse_complement
except ImportError:

    def reverse_complement(seq: str) -> str:
        mapping = str.maketrans("ACGTNacgtn", "TGCANtgcan")
        return seq.translate(mapping)[::-1]


try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

WRITE_LOCK = Lock()
EXAMPLE_CREDENTIAL_VALUES = {"paste-your-ncbi-key-here", "your-email@example.com"}


def clean_ncbi_credential(value: str | None, env_name: str) -> str | None:
    """Ignore copied documentation placeholders and remove them from children."""
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.casefold() in EXAMPLE_CREDENTIAL_VALUES:
        if os.getenv(env_name, "").strip().casefold() in EXAMPLE_CREDENTIAL_VALUES:
            os.environ.pop(env_name, None)
        return None
    return cleaned


# Subprocess execution


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr) as text."""
    try:
        p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=False)
        return p.returncode, p.stdout, p.stderr
    except OSError as exc:
        return 1, "", f"[exec:error] {exc}"


# Taxonomy resolution

_TAXONOMY_CACHE = {}


def datasets_taxonomy_candidates(query: str) -> list[dict]:
    """Ask 'datasets' CLI for taxonomy summaries (with caching)."""
    if query in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[query]

    if shutil.which("datasets") is None:
        return []

    code, out, err = _run(["datasets", "summary", "taxonomy", "taxon", query, "--as-json-lines"])
    if code != 0 or not out:
        return []

    candidates: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        tax_id = obj.get("taxonomy", {}).get("tax_id") or obj.get("taxId") or obj.get("tax_id")
        sci = (
            obj.get("taxonomy", {}).get("organism", {}).get("scientific_name")
            or obj.get("taxonomy", {}).get("scientific_name")
            or obj.get("organism", {}).get("scientific_name")
            or obj.get("scientific_name")
        )
        rank = obj.get("taxonomy", {}).get("rank") or obj.get("rank")
        if tax_id and sci:
            candidates.append({"tax_id": str(tax_id), "scientific_name": sci, "rank": rank})

    _TAXONOMY_CACHE[query] = candidates
    return candidates


def eutils_taxonomy_candidates(
    query: str, api_key: str | None = None, retmax: int = 100, tool: str = "dna_lm_builder", email: str | None = None
) -> list[dict]:
    """Use NCBI Entrez E-utilities to find taxonomy IDs (with caching)."""
    cache_key = f"{query}:{api_key}:{email}"
    if cache_key in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[cache_key]

    params = {
        "db": "taxonomy",
        "term": f"{query}[All Names]",
        "retmode": "json",
        "retmax": str(retmax),
        "tool": tool,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    try:
        r = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
    except Exception as e:
        print(f"[warn] eutils esearch failed: {e}", file=sys.stderr)
        return []

    id_str = ",".join(ids)
    sum_params = {"db": "taxonomy", "id": id_str, "retmode": "json", "tool": tool}
    if email:
        sum_params["email"] = email
    if api_key:
        sum_params["api_key"] = api_key

    max_retries = 3
    for attempt in range(max_retries):
        try:
            r2 = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=sum_params, timeout=30)
            r2.raise_for_status()
            res = r2.json().get("result", {})
            break
        except requests.exceptions.HTTPError as e:
            if r2.status_code == 429 and attempt < max_retries - 1:
                sleep_time = 2**attempt
                print(f"[warn] eutils esummary 429 Too Many Requests, retrying in {sleep_time}s...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print(f"[warn] eutils esummary failed: {e}", file=sys.stderr)
                return []
        except Exception as e:
            print(f"[warn] eutils esummary failed: {e}", file=sys.stderr)
            return []

    uids = res.get("uids", [])
    out: list[dict] = []
    for uid in uids:
        rec = res.get(uid, {})
        sci = rec.get("scientificname") or rec.get("title")
        rank = rec.get("rank")
        bag = set([sci]) if sci else set()
        for key in ("othernames", "commonnames", "synonym"):
            for n in rec.get(key) or []:
                if isinstance(n, str):
                    bag.add(n)
        if sci:
            out.append({"tax_id": uid, "scientific_name": sci, "rank": rank, "name_bag": list(bag)})

    _TAXONOMY_CACHE[cache_key] = out
    return out


def fuzzy_pick_taxon(
    query: str,
    min_score: int = 85,
    prefer_rank: str | None = "species",
    api_key: str | None = None,
    email: str | None = None,
) -> tuple[str, str, float]:
    """Resolve a free-text name to (tax_id, canonical_name, score)."""
    if re.fullmatch(r"\d+", query.strip()):
        return query.strip(), query.strip(), 100.0

    cands = datasets_taxonomy_candidates(query)
    pool: list[dict] = []
    for c in cands:
        bag = [c["scientific_name"]]
        pool.append(
            {"tax_id": c["tax_id"], "scientific_name": c["scientific_name"], "rank": c.get("rank"), "name_bag": bag}
        )

    if not pool:
        pool = eutils_taxonomy_candidates(query, api_key=api_key, email=email)

    if not pool:
        raise RuntimeError(f"No taxonomy candidates found for '{query}'.")

    def score_candidate(q, cand):
        choices = cand.get("name_bag") or [cand.get("scientific_name", "")]
        match, score, _ = process.extractOne(q, choices, scorer=fuzz.WRatio)
        if prefer_rank and (cand.get("rank") or "").lower() == prefer_rank.lower():
            score += 2.0
        return float(score), (match or "")

    best = None
    best_score = -1.0
    best_match_name = ""
    for cand in pool:
        s, matched = score_candidate(query, cand)
        if s > best_score:
            best = cand
            best_score = s
            best_match_name = matched

    if best is None or best_score < min_score:
        top = sorted(
            [
                (
                    c,
                    process.extractOne(
                        query, (c.get("name_bag") or [c.get("scientific_name", "")]), scorer=fuzz.WRatio
                    )[1],
                )
                for c in pool
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        msg = " | ".join([f"{c['scientific_name']} (TaxID {c['tax_id']}, score {sc:.1f})" for c, sc in top])
        raise RuntimeError(
            f"Lowest acceptable fuzzy score not met for '{query}' "
            f"(best {best_score:.1f} < threshold {min_score}). Top hits: {msg}"
        )

    canonical = best.get("scientific_name") or best_match_name
    return str(best["tax_id"]), canonical, float(best_score)


def resolve_species_to_taxid(species_name: str, args) -> tuple[str, str]:
    """Returns (tax_id, canonical_scientific_name)."""
    if re.fullmatch(r"\d+", species_name.strip()):
        return species_name.strip(), species_name.strip()

    time.sleep(0.5)

    if getattr(args, "fuzzy", False):
        taxid, canon, score = fuzzy_pick_taxon(
            species_name,
            min_score=getattr(args, "fuzzy_min_score", 85),
            api_key=getattr(args, "taxonomy_api_key", None),
            email=getattr(args, "taxonomy_email", None),
        )
        print(f"[taxonomy] '{species_name}': TaxID {taxid} ({canon}), score={score:.1f}")
        return taxid, canon

    cands = datasets_taxonomy_candidates(species_name)
    if not cands:
        raise RuntimeError(f"No taxonomy match for '{species_name}'. Try --fuzzy.")
    species = [c for c in cands if (c.get("rank") or "").lower() == "species"]
    picked = species[0] if species else cands[0]
    print(f"[taxonomy] '{species_name}': TaxID {picked['tax_id']} ({picked['scientific_name']})")
    return picked["tax_id"], picked["scientific_name"]


# Dataset and FASTA processing


def have_datasets_cli() -> bool:
    return shutil.which("datasets") is not None


def sanitize_name(name: str) -> str:
    """Sanitize a species name or TaxID into a safe filename."""
    if re.fullmatch(r"\d+", name.strip()):
        return name.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return re.sub(r"_+", "_", safe).strip("_")


def parse_assembly_summary_record(obj: Mapping[str, object], assembly_source: str) -> dict | None:
    """Normalize one current NCBI Datasets assembly-summary record."""
    if assembly_source not in {"all", "RefSeq", "GenBank"}:
        raise ValueError(f"Unsupported assembly source: {assembly_source!r}")

    assembly_info = obj.get("assembly_info", {}) or obj.get("assembly", {}) or {}
    assembly_stats = obj.get("assembly_stats", {}) or {}
    if not isinstance(assembly_info, Mapping) or not isinstance(assembly_stats, Mapping):
        return None
    organism = obj.get("organism", {}) or {}
    if not isinstance(organism, Mapping):
        organism = {}

    acc = obj.get("accession") or assembly_info.get("accession") or obj.get("assembly_accession")
    level = obj.get("assembly_level") or assembly_info.get("assembly_level")
    name = (
        assembly_info.get("assembly_name")
        or assembly_info.get("display_name")
        or obj.get("assembly_name")
        or organism.get("organism_name")
    )
    refcat = obj.get("refseq_category") or assembly_info.get("refseq_category")
    size = (
        obj.get("total_length")
        or assembly_info.get("sequence_length")
        or assembly_stats.get("total_sequence_length")
        or obj.get("sequence_length")
        or 0
    )
    try:
        size = int(size)
    except (TypeError, ValueError):
        size = 0

    if not acc:
        return None
    accession = str(acc)
    is_refseq = accession.startswith("GCF_") or str(obj.get("source_database") or "").endswith("REFSEQ") or bool(refcat)
    is_genbank = accession.startswith("GCA_")
    if assembly_source == "RefSeq" and not is_refseq:
        return None
    if assembly_source == "GenBank" and not is_genbank:
        return None

    return {
        "accession": accession,
        "assembly_level": level,
        "assembly_name": name,
        "refseq_category": refcat,
        "total_length": size,
        "source": "RefSeq" if is_refseq else "GenBank" if is_genbank else "unknown",
    }


def parse_assembly_summary_lines(output: str, assembly_source: str) -> list[dict]:
    """Parse line-delimited JSON emitted by `datasets summary genome`."""
    records: list[dict] = []
    for line in (output or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, Mapping):
            continue
        record = parse_assembly_summary_record(obj, assembly_source)
        if record is not None:
            records.append(record)
    return records


def list_taxon_assemblies(tax_id: str, levels_csv: str, assembly_source: str, max_rows: int) -> list[dict]:
    """Use 'datasets summary genome taxon' to enumerate assemblies."""
    if not have_datasets_cli():
        raise SystemExit("NCBI 'datasets' CLI not found on PATH.")
    if assembly_source not in {"all", "RefSeq", "GenBank"}:
        raise ValueError(f"Unsupported assembly source: {assembly_source!r}")

    cmd = [
        "datasets",
        "summary",
        "genome",
        "taxon",
        str(tax_id),
        "--as-json-lines",
        "--limit",
        str(max_rows),
        "--assembly-level",
        levels_csv,
    ]

    code, out, err = _run(cmd)

    if code != 0:
        if "no genome data is currently available" in err:
            raise RuntimeError(f"No genome data available for TaxID {tax_id}")
        raise RuntimeError(f"'datasets summary' failed for TaxID {tax_id}: {err.strip()}")

    return parse_assembly_summary_lines(out, assembly_source)


def get_assembly_by_accession(accession: str, assembly_source: str) -> dict:
    """Resolve an explicit, versioned assembly accession through NCBI Datasets."""
    if not have_datasets_cli():
        raise SystemExit("NCBI 'datasets' CLI not found on PATH.")
    if not re.fullmatch(r"GC[AF]_\d+\.\d+", accession):
        raise ValueError(f"Invalid versioned assembly accession: {accession!r}")

    code, out, err = _run(["datasets", "summary", "genome", "accession", accession, "--as-json-lines"])
    if code != 0:
        raise RuntimeError(f"'datasets summary genome accession' failed for {accession}: {err.strip()}")

    records = parse_assembly_summary_lines(out, assembly_source)
    for record in records:
        if record["accession"] == accession:
            return record
    if assembly_source == "GenBank" and accession.startswith("GCF_"):
        raise RuntimeError(f"Pinned RefSeq accession {accession} conflicts with --assembly-source GenBank")
    raise RuntimeError(f"NCBI returned no {assembly_source} assembly record for pinned accession {accession}")


def sort_assemblies(assemblies: list[dict], level_priority: list[str]) -> list[dict]:
    """Sort assemblies by priority (level, refseq status, size)."""
    prio = {lvl.lower(): i for i, lvl in enumerate(level_priority)}

    def key(a: dict):
        lvl = (a.get("assembly_level") or "").lower()
        return (
            prio.get(lvl, len(prio) + 1),
            0 if (a.get("refseq_category") or "").lower() else 1,
            -int(a.get("total_length") or 0),
            str(a.get("accession") or ""),
        )

    return sorted(assemblies, key=key)


def is_assembly_cached(accession: str, dest_root: Path) -> bool:
    """Check if assembly already downloaded."""
    acc_dir = dest_root / accession
    data_dir = acc_dir / "ncbi_dataset" / "data"
    return data_dir.exists() and any(data_dir.glob("**/*.fna"))


def download_and_prepare(accession: str, dest_root: Path, retries: int = 2) -> Path:
    """Download, extract, and rehydrate one assembly package."""
    dest_root.mkdir(parents=True, exist_ok=True)
    acc_dir = dest_root / accession

    if is_assembly_cached(accession, dest_root):
        return acc_dir

    acc_dir.mkdir(parents=True, exist_ok=True)
    zip_path = acc_dir / f"{accession}.zip"

    last_err = None
    for _attempt in range(1, retries + 1):
        code, out, err = _run(
            [
                "datasets",
                "download",
                "genome",
                "accession",
                accession,
                "--include",
                "genome",
                "--dehydrated",
                "--filename",
                str(zip_path),
            ]
        )
        if code != 0:
            last_err = f"datasets download failed for {accession}: {err}"
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Reject archive members that escape the extraction directory.
                for member in zf.infolist():
                    member_path = os.path.join(acc_dir, member.filename)
                    abs_target = os.path.abspath(member_path)
                    abs_root = os.path.abspath(acc_dir)
                    if os.path.commonpath([abs_root, abs_target]) != abs_root:
                        raise RuntimeError(f"Zip Slip detected: {member.filename} tries to escape {acc_dir}")
                    zf.extract(member, acc_dir)
        except Exception as e:
            last_err = f"bad zip for {accession}: {e}"
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        code2, out2, err2 = _run(["datasets", "rehydrate", "--directory", str(acc_dir)])
        if code2 != 0:
            last_err = f"rehydrate failed for {accession}: {err2}"
            try:
                shutil.rmtree(acc_dir, ignore_errors=True)
            except Exception:
                pass
            continue

        return acc_dir

    raise RuntimeError(last_err or f"failed to download {accession}")


def download_assembly_worker(assembly: dict, dest_root: Path) -> tuple[dict, Path | None, str | None]:
    """Download one package without changing corpus state.

    Filtering and FASTA appends deliberately happen in the main thread.  That
    makes a parallel fetch schedule produce the same output as a serial run.
    """
    try:
        return assembly, download_and_prepare(assembly["accession"], dest_root), None
    except Exception as exc:
        return assembly, None, str(exc)


def discard_download_package(package_dir: Path | None, keep_downloads: bool) -> None:
    """Remove a consumed NCBI package unless the caller explicitly keeps it for debugging."""
    if keep_downloads or package_dir is None or not package_dir.exists():
        return
    parent = package_dir.parent
    shutil.rmtree(package_dir, ignore_errors=True)
    try:
        parent.rmdir()  # Remove the now-empty per-species cache directory too.
    except OSError:
        pass


def ordered_prefetched_downloads(
    assemblies: list[dict], dest_root: Path, workers: int, keep_downloads: bool
) -> Iterable[tuple[dict, Path | None, str | None]]:
    """Yield packages in accession-selection order while downloading a small window in parallel."""
    if workers <= 1:
        for assembly in assemblies:
            yield download_assembly_worker(assembly, dest_root)
        return

    pending: dict[int, object] = {}
    next_index = 0
    executor = ThreadPoolExecutor(max_workers=workers)

    def submit(index: int) -> None:
        pending[index] = executor.submit(download_assembly_worker, assemblies[index], dest_root)

    try:
        while next_index < min(workers, len(assemblies)):
            submit(next_index)
            next_index += 1

        for expected_index in range(len(assemblies)):
            future = pending.pop(expected_index)
            assembly, package_dir, error = future.result()
            if next_index < len(assemblies):
                submit(next_index)
                next_index += 1
            yield assembly, package_dir, error
    finally:
        # A quota may be met before the prefetch window is consumed.  Cancel
        # queued work, wait for the bounded in-flight window, then remove any
        # unconsumed temporary packages.
        for future in pending.values():
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if not keep_downloads:
            for future in pending.values():
                if future.cancelled() or not future.done():
                    continue
                try:
                    _assembly, package_dir, _error = future.result()
                except Exception:
                    continue
                discard_download_package(package_dir, keep_downloads)


def fasta_iter(path: Path) -> Iterable[tuple[str, str]]:
    """Yield (header, seq) tuples from a fasta/fastn file."""
    header = None
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line.strip()[1:]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks)


def contig_passes_filters(header: str, seq: str, min_length: int, max_n_frac: float, exclude_plasmids: bool) -> bool:
    if exclude_plasmids and re.search(r"plasmid", header, flags=re.I):
        return False
    if len(seq) < min_length or len(seq) == 0:
        return False

    if max_n_frac < 0:
        return True

    # A zero threshold means "emit no N bases", not "discard every source
    # contig that contains an N".  ``iter_filtered_contigs`` below splits at
    # every N and only emits A/C/G/T chunks, so accepting the source contig in
    # this case preserves the strict output guarantee.  This matters for
    # otherwise high-quality complete genomes that have a handful of gap
    # symbols (for example, seven Ns in an approximately 1.8 Mb chromosome).
    if max_n_frac == 0:
        return True

    n_frac = seq.count("N") / float(len(seq))
    if n_frac > max_n_frac:
        return False
    return True


# Split, rather than discard, a contig at every ambiguity/gap symbol.  Every
# emitted sequence is therefore strictly A/C/G/T even when a public assembly
# uses IUPAC ambiguity letters in an otherwise high-quality chromosome.
STRICT_SPLIT_RE = re.compile(r"[^ACGT]+")
MAX_LINEAR_CONTIG_CHUNK = 10_000_000
# A FASTA header plus the configured minimum sequence must fit before another
# assembly is considered.  Without this guard a nearly-full byte budget can
# scan every remaining assembly trying to fill an impossible final few bytes.
MIN_FASTA_RECORD_HEADROOM = 4_096


def iter_filtered_contigs(
    pkg_dir: Path, min_length: int, max_n_frac: float, exclude_plasmids: bool
) -> Iterable[tuple[str, str]]:
    """Iterate all .fna/.fa/.fasta under ncbi_dataset/data/* and yield filtered contigs.
    Splits contigs by any non-ACGT character to ensure strictly clean outputs.

    Very large linear chromosomes are emitted as coordinate-labeled chunks so
    target-size sampling does not skip a whole chromosome merely because the
    remaining quota is smaller than that chromosome.
    """
    data_dir = pkg_dir / "ncbi_dataset" / "data"
    if not data_dir.exists():
        return
    for ext in ("*.fna", "*.fa", "*.fasta"):
        for fna in data_dir.rglob(ext):
            for h, s in fasta_iter(fna):
                s = s.upper()
                if not contig_passes_filters(h, s, min_length, max_n_frac, exclude_plasmids):
                    continue
                clean_chunks = STRICT_SPLIT_RE.split(s)
                for i, chunk in enumerate(clean_chunks):
                    if len(chunk) < min_length:
                        continue

                    # Append chunk index to header if it was split.
                    chunk_header = f"{h}_chunk_{i}" if len(clean_chunks) > 1 else h
                    if len(chunk) <= MAX_LINEAR_CONTIG_CHUNK:
                        yield chunk_header, chunk
                        continue

                    for start in range(0, len(chunk), MAX_LINEAR_CONTIG_CHUNK):
                        end = min(start + MAX_LINEAR_CONTIG_CHUNK, len(chunk))
                        if end - start >= min_length:
                            yield f"{chunk_header}_{start}_{end}", chunk[start:end]


def canonical_sequence_hash(sequence: str) -> bytes:
    """Return a reverse-complement-invariant digest for a DNA sequence."""

    rc_sequence = reverse_complement(sequence)
    canonical = min(sequence, rc_sequence)
    return hashlib.sha256(canonical.encode("ascii")).digest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without retaining an assembly in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assembly_file_inventory(pkg_dir: Path, accession: str) -> list[dict[str, object]]:
    """Record the exact public genome files selected from one NCBI package."""
    data_dir = pkg_dir / "ncbi_dataset" / "data"
    files = sorted(path for pattern in ("*.fna", "*.fa", "*.fasta") for path in data_dir.rglob(pattern))
    return [
        {
            "path": str(path.relative_to(pkg_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "source_url": f"https://www.ncbi.nlm.nih.gov/datasets/genome/{accession}/",
            "source": "NCBI Datasets genome package",
        }
        for path in files
    ]


def _fragment_to_fit(
    header: str,
    sequence: str,
    available_record_bytes: int,
    *,
    min_length: int,
    seed: int,
) -> tuple[str, str] | None:
    """Return a deterministic interval whose complete FASTA record fits the cap."""
    full_record_bytes = len(header.encode("utf-8")) + len(sequence) + 3
    if full_record_bytes <= available_record_bytes:
        return header, sequence

    # Reserve enough room for the coordinate label before choosing an interval.
    label_prefix = f"{header}|sampled="
    max_sequence_bytes = available_record_bytes - len(label_prefix.encode("utf-8")) - 3 - 30
    if max_sequence_bytes < min_length:
        return None
    take = min(len(sequence), max_sequence_bytes)
    if take < len(sequence):
        span = len(sequence) - take + 1
        offset = int.from_bytes(hashlib.sha256(f"{seed}|{header}|{len(sequence)}".encode()).digest()[:8], "big") % span
        end = offset + take
        sampled_header = f"{header}|sampled={offset}:{end}"
        return sampled_header, sequence[offset:end]
    return header, sequence


def append_contigs_to_fasta_with_dedup(
    out_fa: Path,
    contigs: list[tuple[str, str]],
    seen_hashes: set[bytes],
    *,
    current_bases: list[int] | None = None,
    max_total_bases: int | None = None,
    current_bytes: list[int] | None = None,
    max_total_bytes: int | None = None,
    fragment_seed: int = 43,
    min_length: int = 1000,
) -> tuple[int, int, int]:
    """Append contigs while deduplicating and respecting base and byte caps."""
    bases = 0
    dupes = 0
    written = 0

    with WRITE_LOCK:
        existing_bases = current_bases[0] if current_bases is not None else 0
        existing_bytes = current_bytes[0] if current_bytes is not None else 0
        with open(out_fa, "a", encoding="utf-8") as w:
            for h, s in contigs:
                if max_total_bytes is not None:
                    fitted = _fragment_to_fit(
                        h,
                        s,
                        max_total_bytes - existing_bytes,
                        min_length=min_length,
                        seed=fragment_seed,
                    )
                    if fitted is None:
                        continue
                    h, s = fitted

                seq_hash = canonical_sequence_hash(s)

                if seq_hash in seen_hashes:
                    dupes += 1
                    continue
                if max_total_bases is not None and existing_bases + len(s) > max_total_bases:
                    continue

                record_bytes = len(h.encode("utf-8")) + len(s) + 3
                if max_total_bytes is not None and existing_bytes + record_bytes > max_total_bytes:
                    continue

                seen_hashes.add(seq_hash)
                w.write(f">{h}\n")
                w.write(s + "\n")
                bases += len(s)
                existing_bases += len(s)
                existing_bytes += record_bytes
                written += 1

        if current_bases is not None:
            current_bases[0] = existing_bases
        if current_bytes is not None:
            current_bytes[0] = existing_bytes

    return bases, dupes, written


def measure_existing_bases(out_fasta: Path) -> int:
    """Count bases already present (non-header characters)."""
    if not out_fasta.exists():
        return 0
    n = 0
    with open(out_fasta, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line and not line.startswith(">"):
                n += len(line.strip())
    return n


def run_qc_if_requested(path: Path, args):
    if not getattr(args, "run_qc", False):
        return

    qc_script = Path(__file__).resolve().parent / "tools" / "sequence_qc.py"
    if not qc_script.is_file():
        raise FileNotFoundError(f"Quality-control script not found: {qc_script}")

    out_dir = path.parent / "qc_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_cmd = [
        sys.executable,
        str(qc_script),
        "--fasta",
        str(path),
        "--min_length",
        str(getattr(args, "qc_min_length", 1000)),
        "--n_threshold",
        str(getattr(args, "qc_n_threshold", 0.05)),
        "--csv",
        "--json",
        "--output_dir",
        str(out_dir),
    ]

    code, _out, err = _run(qc_cmd)
    if code != 0:
        raise RuntimeError(f"QC failed on {path.name}: {err.strip()}")
    print(f"[qc] {path.name}: QC complete")


# Parallel assembly processing


def process_assembly_worker(args_tuple):
    """
    Worker function for parallel assembly processing.

    Returns: (bases_added, dupes_found, contigs_used, accession_info) or None on failure
    """
    (
        asm,
        per_species_root,
        out_fasta,
        seen_hashes,
        min_length,
        max_n_frac,
        exclude_plasmids,
        overshoot_bases,
        current_bases,
        target_bases,
        keep_downloads,
    ) = args_tuple

    acc = asm["accession"]

    try:
        # Check if we've already met target
        with WRITE_LOCK:
            if current_bases[0] >= target_bases:
                return None

        pkg_dir = download_and_prepare(acc, per_species_root)

        batch: list[tuple[str, str]] = []
        batch_bytes = 0

        bases_added = 0
        dupes = 0
        contigs_used = 0

        for h, s in iter_filtered_contigs(pkg_dir, min_length, max_n_frac, exclude_plasmids):
            with WRITE_LOCK:
                remaining = target_bases - current_bases[0]
                if remaining <= 0:
                    break

            if len(s) > remaining and (len(s) - remaining) > overshoot_bases:
                continue

            # Preserve assembly provenance in the final combined FASTA so a
            # later group-aware split cannot leak related contigs.
            batch.append((f"assembly={acc}|{h}", s))
            batch_bytes += len(s)

            # Flush when batch reaches ~32MB to prevent massive OOM from 256 Eukaryotic chromosomes
            if batch_bytes >= 32 * 1024 * 1024:
                b_add, d_found, written_count = append_contigs_to_fasta_with_dedup(
                    out_fasta,
                    batch,
                    seen_hashes,
                    current_bases=current_bases,
                    max_total_bases=target_bases + overshoot_bases,
                )

                bases_added += b_add
                dupes += d_found
                contigs_used += written_count

                batch = []
                batch_bytes = 0

                with WRITE_LOCK:
                    if current_bases[0] >= target_bases + overshoot_bases:
                        break

        # Process remaining batch
        if batch:
            b_add, d_found, written_count = append_contigs_to_fasta_with_dedup(
                out_fasta,
                batch,
                seen_hashes,
                current_bases=current_bases,
                max_total_bases=target_bases + overshoot_bases,
            )

            bases_added += b_add
            dupes += d_found
            contigs_used += written_count

        if bases_added > 0 or contigs_used > 0:
            accession_info = {
                "accession": acc,
                "assembly_level": asm.get("assembly_level"),
                "refseq_category": asm.get("refseq_category"),
                "assembly_name": asm.get("assembly_name"),
            }
            return (bases_added, dupes, contigs_used, accession_info)

        return None

    except Exception as e:
        print(f"[warn] skip {acc}: {e}", file=sys.stderr)
        return None
    finally:
        # Extracted assembly packages are temporary working data.
        if not keep_downloads and "pkg_dir" in locals() and pkg_dir is not None and pkg_dir.exists():
            import shutil

            shutil.rmtree(pkg_dir, ignore_errors=True)


# Command-line interface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch, filter, and deduplicate genomes from NCBI")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--species", nargs="+", help="One or more species/taxa names or IDs")
    g.add_argument("--config", type=str, help="YAML or CSV config")

    p.add_argument("--target-mb", type=float, default=100.0)
    p.add_argument("--targets-mb", type=float, nargs="+")

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.getenv("TRUE_DNA_NCBI_SCRATCH", "data/raw_downloads")),
        help="Temporary NCBI package directory; use native WSL storage for best throughput.",
    )
    p.add_argument("--results-dir", type=Path, default=Path("data/balanced"))
    p.add_argument("--keep-downloads", action="store_true", help="Retain extracted NCBI assembly packages")

    p.add_argument("--min-length", type=int, default=1000)
    p.add_argument(
        "--max-n-frac",
        type=float,
        default=0.05,
        help="Maximum source-contig N fraction when positive. At 0, split at Ns and emit only zero-N A/C/G/T chunks.",
    )
    p.add_argument("--exclude-plasmids", action="store_true")

    p.add_argument("--refseq-only", action="store_true")
    p.add_argument(
        "--assembly-source",
        choices=["all", "RefSeq", "GenBank"],
        default="all",
        help="Restrict selected accessions by their GCF_ (RefSeq) or GCA_ (GenBank) prefix.",
    )
    p.add_argument("--levels", type=str, default="complete,chromosome,scaffold")
    p.add_argument("--max-assemblies", type=int, default=1_000)

    p.add_argument("--run-qc", action="store_true")
    p.add_argument("--qc-min-length", type=int, default=1000)
    p.add_argument(
        "--qc-n-threshold",
        type=float,
        default=5.0,
        help="Maximum N content accepted by QC, as a percentage from 0 to 100",
    )

    p.add_argument("--fuzzy", action="store_true")
    p.add_argument("--fuzzy-min-score", type=int, default=85)
    p.add_argument(
        "--api-key",
        "--taxonomy-api-key",
        dest="api_key",
        type=str,
        default=os.getenv("NCBI_API_KEY") or os.getenv("NCBI_DATASETS_API_KEY"),
        help="NCBI API key; prefer the NCBI_API_KEY environment variable",
    )
    p.add_argument(
        "--taxonomy-email",
        type=str,
        default=os.getenv("NCBI_TAXONOMY_EMAIL"),
        help="NCBI contact email; defaults to NCBI_TAXONOMY_EMAIL",
    )

    p.add_argument("--append", action="store_true")
    p.add_argument("--overshoot-mb", type=float, default=0.0)
    p.add_argument(
        "--domain-budget-mb",
        type=float,
        default=None,
        help="Hard byte budget shared by every species in this config/domain. Enables deterministic quota redistribution.",
    )
    p.add_argument("--sampling-seed", type=int, default=43, help="Seed for deterministic final-record truncation")

    p.add_argument(
        "--parallel-downloads",
        type=int,
        default=4,
        help="Bounded parallel package downloads; filtering/writing remains deterministic (default: 4)",
    )

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _config_item(item: Mapping[str, object], default_target_mb: float = 100.0) -> dict[str, object] | None:
    def optional_text(value: object) -> str | None:
        text = str(value or "").strip()
        return None if not text or text.casefold() in {"none", "null"} else text

    tax_id = str(item.get("tax_id") or "").strip()
    name = str(
        item.get("species") or item.get("scientific_name") or item.get("name") or item.get("taxon") or ""
    ).strip()
    if tax_id and not tax_id.isdecimal():
        raise ValueError(f"Invalid tax_id {tax_id!r}")
    if not tax_id and not name:
        return None
    target_mb = float(item.get("target_mb") or item.get("weight") or default_target_mb)
    return {
        "query": tax_id or name,
        "tax_id": tax_id or None,
        "scientific_name": name or None,
        "target_mb": target_mb,
        "family_tax_id": str(item.get("family_tax_id") or "") or None,
        "family_name": str(item.get("family_name") or "") or None,
        "selection_reason": str(item.get("selection_reason") or "") or None,
        "preferred_refseq_accession": optional_text(item.get("preferred_refseq_accession")),
        "domain": str(item.get("domain") or "") or None,
    }


def load_config(path: Path) -> list[dict[str, object]]:
    """Return TaxID-aware selection records from YAML or CSV."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit("PyYAML is required to read YAML configuration files") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"YAML config must contain a mapping: {path}")
        species_items = data.get("species", [])
        if not isinstance(species_items, list):
            raise ValueError("YAML 'species' must be a list")
        items: list[dict[str, object]] = []
        for item in species_items:
            if not isinstance(item, Mapping):
                raise ValueError("Each YAML species entry must be a mapping")
            record = _config_item(item, float(data.get("target_mb", 100)))
            if record:
                items.append(record)
        return items

    rows: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            record = _config_item(r)
            if record:
                rows.append(record)
    return rows


def get_processed_taxids(results_dir: Path) -> set[str]:
    """Build a set of all TaxIDs that have already been processed."""
    processed_tax_ids = set()
    for manifest_path in results_dir.glob("*.manifest.json"):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            tax_id = data.get("tax_id")
            if tax_id:
                processed_tax_ids.add(str(tax_id))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            print(f"[warn] Ignoring invalid manifest {manifest_path}: {exc}", file=sys.stderr)
    return processed_tax_ids


def fasta_storage_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def domain_storage_bytes(results_dir: Path) -> int:
    return sum(fasta_storage_bytes(path) for path in results_dir.glob("*.fa"))


def write_domain_download_manifest(results_dir: Path, budget_bytes: int | None, sampling_seed: int) -> None:
    """Summarize the exact raw files before the domain-level QC/concatenation step."""
    species = []
    for manifest_path in sorted(results_dir.glob("*.manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        output = Path(str(manifest.get("output_fasta") or ""))
        species.append(
            {
                "manifest": manifest_path.name,
                "tax_id": manifest.get("tax_id"),
                "scientific_name": manifest.get("scientific_name"),
                "family_tax_id": manifest.get("family_tax_id"),
                "family_name": manifest.get("family_name"),
                "selection_reason": manifest.get("selection_reason"),
                "fasta_bytes": fasta_storage_bytes(output),
                "fasta_sha256": sha256_file(output) if output.exists() else None,
                "assemblies_used": manifest.get("assemblies_used", []),
            }
        )
    total_bytes = sum(int(item["fasta_bytes"]) for item in species)
    payload = {
        "schema_version": 1,
        "budget_bytes": budget_bytes,
        "written_bytes": total_bytes,
        "within_budget": budget_bytes is None or total_bytes <= budget_bytes,
        "sampling_seed": sampling_seed,
        "species": species,
    }
    (results_dir / "domain_download_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def reached_target(written_bases: int, current_bytes: list[int], target_bases: int, byte_cap: int | None) -> bool:
    return written_bases >= target_bases or (
        byte_cap is not None and current_bytes[0] >= byte_cap - MIN_FASTA_RECORD_HEADROOM
    )


def main() -> int:
    args = parse_args()
    args.api_key = clean_ncbi_credential(args.api_key, "NCBI_API_KEY")
    args.taxonomy_email = clean_ncbi_credential(args.taxonomy_email, "NCBI_TAXONOMY_EMAIL")
    if not 0.0 <= args.max_n_frac <= 1.0:
        raise SystemExit("--max-n-frac must be between 0 and 1")
    if not 0.0 <= args.qc_n_threshold <= 100.0:
        raise SystemExit("--qc-n-threshold must be between 0 and 100")
    if args.parallel_downloads <= 0:
        raise SystemExit("--parallel-downloads must be positive")
    if args.min_length <= 0 or args.qc_min_length <= 0:
        raise SystemExit("--min-length and --qc-min-length must be positive")
    if args.max_assemblies <= 0:
        raise SystemExit("--max-assemblies must be positive")
    if args.overshoot_mb < 0:
        raise SystemExit("--overshoot-mb must be non-negative")
    if args.domain_budget_mb is not None and args.domain_budget_mb <= 0:
        raise SystemExit("--domain-budget-mb must be positive")
    if not 0 <= args.fuzzy_min_score <= 100:
        raise SystemExit("--fuzzy-min-score must be between 0 and 100")
    if args.refseq_only and args.assembly_source == "GenBank":
        raise SystemExit("--refseq-only conflicts with --assembly-source GenBank")
    if args.refseq_only:
        args.assembly_source = "RefSeq"
    if args.api_key:
        # The current NCBI Datasets CLI automatically reads this environment
        # variable for both summary and download requests. It also lets our
        # E-utilities taxonomy fallback use the same key.
        os.environ["NCBI_API_KEY"] = args.api_key
        args.taxonomy_api_key = args.api_key
    else:
        args.taxonomy_api_key = None
    print(f"[env] Python: {sys.version.split()[0]}  cwd: {os.getcwd()}")
    print(f"[env] datasets CLI: {shutil.which('datasets') or 'NOT FOUND'}")
    print(f"[env] Parallel downloads: {args.parallel_downloads} workers")

    if not have_datasets_cli():
        print("[error] NCBI 'datasets' CLI not found.", file=sys.stderr)
        sys.exit(1)

    if args.species:
        targets = args.targets_mb or [args.target_mb] * len(args.species)
        if len(targets) != len(args.species):
            raise SystemExit("--targets-mb must contain one value per --species entry")
        spec_items = [
            {
                "query": name,
                "tax_id": name if str(name).strip().isdecimal() else None,
                "scientific_name": None,
                "target_mb": float(target),
                "family_tax_id": None,
                "family_name": None,
                "selection_reason": None,
                "preferred_refseq_accession": None,
                "domain": None,
            }
            for name, target in zip(args.species, targets, strict=True)
        ]
    else:
        spec_items = load_config(Path(args.config))
    if not spec_items:
        raise SystemExit("No species targets were provided")
    if any(float(item["target_mb"]) <= 0 for item in spec_items):
        raise SystemExit("All target sizes must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    level_pref = [x.strip() for x in args.levels.split(",") if x.strip()]

    processed_tax_ids = set()
    if args.append:
        processed_tax_ids = get_processed_taxids(args.results_dir)
        if processed_tax_ids:
            print(f"[append] Found {len(processed_tax_ids)} completed. Will skip.")

    domain_budget_bytes = (
        int(round(float(args.domain_budget_mb) * 1_000_000)) if args.domain_budget_mb is not None else None
    )
    if domain_budget_bytes is not None:
        existing_domain_bytes = domain_storage_bytes(args.results_dir)
        if existing_domain_bytes > domain_budget_bytes:
            raise SystemExit(
                f"Existing files use {existing_domain_bytes / 1e6:.1f} MB, exceeding the "
                f"{args.domain_budget_mb:.1f} MB domain budget. Use a fresh output directory."
            )
        print(
            f"[budget] shared domain cap: {args.domain_budget_mb:.1f} MB; "
            f"existing: {existing_domain_bytes / 1e6:.1f} MB"
        )

    if HAS_TQDM:
        species_iter = tqdm(spec_items, desc="Processing species", unit="species")
    else:
        species_iter = spec_items

    failures = 0
    for item_index, item in enumerate(species_iter):
        raw_name = str(item["query"])
        configured_tax_id = str(item["tax_id"] or "")
        configured_name = str(item["scientific_name"] or "")
        configured_target_mb = float(item["target_mb"])
        preferred_refseq_accession = str(item.get("preferred_refseq_accession") or "")
        try:
            if domain_budget_bytes is not None and domain_storage_bytes(args.results_dir) >= domain_budget_bytes:
                print("[budget] shared domain cap reached; remaining species are not downloaded")
                break
            try:
                if configured_tax_id:
                    tax_id = configured_tax_id
                    canonical = configured_name or configured_tax_id
                else:
                    tax_id, canonical = resolve_species_to_taxid(raw_name, args)
            except Exception as e:
                print(f"[taxonomy:error] Failed to resolve '{raw_name}': {e}")
                failures += 1
                continue

            safe_name = sanitize_name(canonical)
            per_species_root = args.out_dir / f"{safe_name}__TaxID_{tax_id}"
            out_fasta = args.results_dir / f"{safe_name}.fa"
            manifest_path = args.results_dir / f"{safe_name}.manifest.json"

            remaining_weight = sum(float(rest["target_mb"]) for rest in spec_items[item_index:])
            current_domain_bytes = domain_storage_bytes(args.results_dir)
            if domain_budget_bytes is not None:
                domain_remaining_bytes = max(domain_budget_bytes - current_domain_bytes, 0)
                target_storage_bytes = int(round(domain_remaining_bytes * configured_target_mb / remaining_weight))
                target_mb = target_storage_bytes / 1_000_000.0
            else:
                target_storage_bytes = None
                target_mb = configured_target_mb
            target_bases = int(round(target_mb * 1_000_000))
            overshoot_bases = int(round(float(args.overshoot_mb) * 1_000_000))

            pinned_assembly = None
            if preferred_refseq_accession:
                pinned_assembly = get_assembly_by_accession(preferred_refseq_accession, args.assembly_source)
                print(f"[pin] {canonical}: using audited RefSeq accession {preferred_refseq_accession}")

            if args.dry_run:
                asms = list_taxon_assemblies(tax_id, args.levels, args.assembly_source, 10)
                if pinned_assembly is not None:
                    asms = [pinned_assembly] + [asm for asm in asms if asm["accession"] != pinned_assembly["accession"]]
                print(f"[dry-run] {canonical}: showing up to 5 assemblies:")
                for a in asms[:5]:
                    print(f"  - {a['accession']} | {a.get('assembly_level')} | {a.get('total_length')} bp")
                continue

            # Append logic
            written_bases = 0
            current_bytes = [fasta_storage_bytes(out_fasta) if args.append else 0]
            byte_cap = target_storage_bytes
            processed_accessions = set()
            seen_hashes = set()
            duplicates_found = 0

            if args.append and out_fasta.exists():
                already = measure_existing_bases(out_fasta)
                if tax_id in processed_tax_ids and reached_target(already, current_bytes, target_bases, byte_cap):
                    print(f"[append] {out_fasta.name} already has {already / 1e6:.2f} Mb; nothing to do")
                    processed_tax_ids.add(tax_id)
                    continue
                else:
                    print(f"[append] {out_fasta.name} already has {already / 1e6:.2f} Mb; topping up")
                    written_bases = already

                    print("[dedup] Indexing existing sequences...")
                    for _, seq in fasta_iter(out_fasta):
                        seen_hashes.add(canonical_sequence_hash(seq.upper()))
                    print(f"[dedup] Loaded {len(seen_hashes)} hashes")

                if manifest_path.exists():
                    try:
                        manifest_data = json.loads(manifest_path.read_text())
                        for asm in manifest_data.get("assemblies_used", []):
                            if asm.get("accession"):
                                processed_accessions.add(asm["accession"])
                        duplicates_found = manifest_data.get("duplicates_found", 0)
                    except Exception as e:
                        print(f"[warn] Could not read manifest: {e}")
            else:
                if out_fasta.exists():
                    out_fasta.unlink()
                if manifest_path.exists():
                    manifest_path.unlink()

            print(f"[plan] {canonical} (TaxID {tax_id}): target about {target_mb:.1f} MB")

            assemblies = list_taxon_assemblies(tax_id, args.levels, args.assembly_source, args.max_assemblies)
            if pinned_assembly is not None:
                # NCBI sometimes indexes a reference below a strain TaxID, so
                # the parent species query can be empty.  Keep the explicit,
                # audited accession first, then retain any non-duplicate
                # assemblies returned for the selected parent species.
                assemblies = [pinned_assembly] + [
                    asm for asm in assemblies if asm["accession"] != pinned_assembly["accession"]
                ]

            if not assemblies:
                print(f"[warn] No assemblies found for {canonical}")
                failures += 1
                continue

            if pinned_assembly is not None:
                assemblies = [pinned_assembly] + sort_assemblies(assemblies[1:], level_pref)
            else:
                assemblies = sort_assemblies(assemblies, level_pref)

            original_assembly_count = len(assemblies)
            if processed_accessions:
                assemblies = [asm for asm in assemblies if asm["accession"] not in processed_accessions]
                print(f"[plan] {len(assemblies)} new assemblies (of {original_assembly_count})")

            if not assemblies:
                if args.append:
                    print("[append] All assemblies already processed")
                processed_tax_ids.add(tax_id)
                continue

            chosen_accessions: list[dict] = []
            used_contigs = 0

            current_bases = [written_bases]
            if args.parallel_downloads > 1:
                print(
                    f"[parallel] Prefetching up to {args.parallel_downloads} assemblies; writing in deterministic order"
                )
            prefetch_iter = ordered_prefetched_downloads(
                assemblies,
                per_species_root,
                args.parallel_downloads,
                args.keep_downloads,
            )
            download_iter = prefetch_iter
            if HAS_TQDM:
                download_iter = tqdm(
                    download_iter, total=len(assemblies), desc=f"Fetching {canonical}", unit="asm", leave=False
                )

            for asm, pkg_dir, download_error in download_iter:
                if reached_target(written_bases, current_bytes, target_bases, byte_cap):
                    # The item just yielded may be an unconsumed prefetched
                    # package.  Delete it before closing the iterator, whose
                    # finalizer handles the remaining pending window.
                    discard_download_package(pkg_dir, args.keep_downloads)
                    prefetch_iter.close()
                    break
                acc = asm["accession"]
                if acc in processed_accessions:
                    discard_download_package(pkg_dir, args.keep_downloads)
                    continue
                if download_error or pkg_dir is None:
                    print(f"[warn] skip {acc}: {download_error or 'download did not yield a package'}")
                    continue

                batch: list[tuple[str, str]] = []
                batch_bytes = 0
                assembly_bases_added = 0
                for h, s in iter_filtered_contigs(pkg_dir, args.min_length, args.max_n_frac, args.exclude_plasmids):
                    if reached_target(written_bases, current_bytes, target_bases, byte_cap):
                        break
                    batch.append((f"assembly={acc}|{h}", s))
                    batch_bytes += len(s)

                    # Flush based on memory footprint instead of item count.
                    if batch_bytes >= 32 * 1024 * 1024:
                        bases_added, dupes, written_count = append_contigs_to_fasta_with_dedup(
                            out_fasta,
                            batch,
                            seen_hashes,
                            current_bases=current_bases,
                            max_total_bases=target_bases + overshoot_bases,
                            current_bytes=current_bytes,
                            max_total_bytes=byte_cap,
                            fragment_seed=args.sampling_seed,
                            min_length=args.min_length,
                        )
                        written_bases = current_bases[0]
                        assembly_bases_added += bases_added
                        duplicates_found += dupes
                        used_contigs += written_count
                        batch = []
                        batch_bytes = 0
                        if reached_target(written_bases, current_bytes, target_bases + overshoot_bases, byte_cap):
                            break

                if batch and not reached_target(written_bases, current_bytes, target_bases + overshoot_bases, byte_cap):
                    bases_added, dupes, written_count = append_contigs_to_fasta_with_dedup(
                        out_fasta,
                        batch,
                        seen_hashes,
                        current_bases=current_bases,
                        max_total_bases=target_bases + overshoot_bases,
                        current_bytes=current_bytes,
                        max_total_bytes=byte_cap,
                        fragment_seed=args.sampling_seed,
                        min_length=args.min_length,
                    )
                    written_bases = current_bases[0]
                    assembly_bases_added += bases_added
                    duplicates_found += dupes
                    used_contigs += written_count

                if assembly_bases_added > 0:
                    processed_accessions.add(acc)
                    chosen_accessions.append(
                        {
                            "accession": acc,
                            "assembly_level": asm.get("assembly_level"),
                            "refseq_category": asm.get("refseq_category"),
                            "assembly_name": asm.get("assembly_name"),
                            "assembly_total_bases": asm.get("total_length"),
                            "source": asm.get("source"),
                            "source_url": f"https://www.ncbi.nlm.nih.gov/datasets/genome/{acc}/",
                            "genome_files": assembly_file_inventory(pkg_dir, acc),
                        }
                    )

                discard_download_package(pkg_dir, args.keep_downloads)
            prefetch_iter.close()

            if written_bases == 0:
                print(f"[error] Wrote 0 bases for {canonical}")
                failures += 1
                continue

            # Build manifest
            all_assemblies_used = []
            if args.append and manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text())
                    all_assemblies_used = manifest_data.get("assemblies_used", [])
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"[warn] Could not read existing manifest: {exc}", file=sys.stderr)

            all_assemblies_used.extend(chosen_accessions)

            manifest = {
                "scientific_name": canonical,
                "tax_id": tax_id,
                "domain": item.get("domain"),
                "family_tax_id": item.get("family_tax_id"),
                "family_name": item.get("family_name"),
                "selection_reason": item.get("selection_reason"),
                "preferred_refseq_accession": item.get("preferred_refseq_accession"),
                "target_mb": float(target_mb),
                "storage_budget_bytes": byte_cap,
                "written_bases": int(written_bases),
                "written_mb": float(written_bases) / 1_000_000.0,
                "output_bytes": fasta_storage_bytes(out_fasta),
                "output_sha256": sha256_file(out_fasta),
                "num_contigs": int(used_contigs),
                "duplicates_found": int(duplicates_found),
                "unique_sequences": len(seen_hashes),
                "output_fasta": str(out_fasta),
                "assemblies_used": all_assemblies_used,
                "params": {
                    "min_length": args.min_length,
                    "max_n_frac": args.max_n_frac,
                    "exclude_plasmids": bool(args.exclude_plasmids),
                    "levels": args.levels,
                    "assembly_source": args.assembly_source,
                    "refseq_only": bool(args.assembly_source == "RefSeq"),
                    "append": bool(args.append),
                    "overshoot_mb": float(args.overshoot_mb),
                    "domain_budget_mb": args.domain_budget_mb,
                    "sampling_seed": args.sampling_seed,
                    "deduplicated": True,
                    "parallel_downloads": args.parallel_downloads,
                    "keep_downloads": bool(args.keep_downloads),
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))

            dedup_pct = (
                (duplicates_found / (used_contigs + duplicates_found) * 100.0)
                if (used_contigs + duplicates_found) > 0
                else 0.0
            )
            print(f"[done] {canonical}: wrote {manifest['written_mb']:.2f} Mb to {out_fasta.name}")
            print(f"       Unique: {used_contigs}, Duplicates: {duplicates_found} ({dedup_pct:.1f}%)")

            run_qc_if_requested(out_fasta, args)
            processed_tax_ids.add(tax_id)

        except Exception as exc:
            failures += 1
            print(f"[error] Failed processing {raw_name}: {exc}")
            if not args.keep_downloads and isinstance(locals().get("pkg_dir"), Path) and pkg_dir.exists():
                shutil.rmtree(pkg_dir, ignore_errors=True)
            import traceback

            traceback.print_exc()

    if domain_budget_bytes is not None:
        write_domain_download_manifest(args.results_dir, domain_budget_bytes, args.sampling_seed)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

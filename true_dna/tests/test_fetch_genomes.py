from pathlib import Path

from fetch_genomes import (
    append_contigs_to_fasta_with_dedup,
    canonical_sequence_hash,
    discard_download_package,
    get_assembly_by_accession,
    iter_filtered_contigs,
    list_taxon_assemblies,
    load_config,
    reached_target,
)


def _write_package(root: Path, records: list[tuple[str, str]]) -> Path:
    data_dir = root / "ncbi_dataset" / "data" / "assembly"
    data_dir.mkdir(parents=True)
    (data_dir / "genome.fna").write_text(
        "".join(f">{header}\n{sequence}\n" for header, sequence in records),
        encoding="utf-8",
    )
    return root


def test_filtered_contigs_normalize_case_and_apply_n_fraction(tmp_path):
    package = _write_package(
        tmp_path / "package",
        [
            ("accepted", "acgtnnacgt"),
            ("ambiguous", "ACGTRACGT"),
            ("plasmid sequence", "ACGTACGT"),
        ],
    )

    accepted = list(
        iter_filtered_contigs(
            package,
            min_length=4,
            max_n_frac=0.25,
            exclude_plasmids=True,
        )
    )
    rejected_for_ns = list(
        iter_filtered_contigs(
            package,
            min_length=4,
            max_n_frac=0.10,
            exclude_plasmids=True,
        )
    )
    zero_n_output = list(
        iter_filtered_contigs(
            package,
            min_length=4,
            max_n_frac=0.0,
            exclude_plasmids=True,
        )
    )

    assert [sequence for _header, sequence in accepted] == ["ACGT", "ACGT", "ACGT", "ACGT"]
    assert [sequence for _header, sequence in rejected_for_ns] == ["ACGT", "ACGT"]
    assert [sequence for _header, sequence in zero_n_output] == ["ACGT", "ACGT", "ACGT", "ACGT"]
    assert all("N" not in sequence for _header, sequence in zero_n_output)


def test_fasta_append_deduplicates_reverse_complements_and_honors_cap(tmp_path):
    output = tmp_path / "sequences.fa"
    seen: set[bytes] = set()
    current_bases = [0]

    bases, duplicates, written = append_contigs_to_fasta_with_dedup(
        output,
        [("forward", "AACCGG"), ("reverse", "CCGGTT")],
        seen,
        current_bases=current_bases,
        max_total_bases=10,
    )
    extra_bases, extra_duplicates, extra_written = append_contigs_to_fasta_with_dedup(
        output,
        [("fits", "ACCC"), ("over-cap", "GGGG")],
        seen,
        current_bases=current_bases,
        max_total_bases=10,
    )

    assert canonical_sequence_hash("AACCGG") == canonical_sequence_hash("CCGGTT")
    assert (bases, duplicates, written) == (6, 1, 1)
    assert (extra_bases, extra_duplicates, extra_written) == (4, 0, 1)
    assert current_bases == [10]
    assert output.read_text(encoding="utf-8") == ">forward\nAACCGG\n>fits\nACCC\n"


def test_fasta_append_honors_a_byte_cap_by_deterministically_sampling_a_contig(tmp_path):
    output = tmp_path / "sequences.fa"
    seen: set[bytes] = set()
    current_bases = [0]
    current_bytes = [0]

    _bases, _duplicates, written = append_contigs_to_fasta_with_dedup(
        output,
        [("assembly=GCF_1|chromosome", "ACGT" * 100)],
        seen,
        current_bases=current_bases,
        current_bytes=current_bytes,
        max_total_bytes=120,
        min_length=20,
        fragment_seed=43,
    )

    assert written == 1
    assert output.stat().st_size == current_bytes[0]
    assert current_bytes[0] <= 120
    assert "|sampled=" in output.read_text(encoding="utf-8")


def test_near_full_budget_is_considered_complete_when_no_valid_record_can_fit():
    assert reached_target(999_500, [999_500], 1_000_000, 1_000_000)


def test_consumed_download_package_is_discarded_unless_explicitly_retained(tmp_path):
    package = tmp_path / "species" / "GCF_1"
    package.mkdir(parents=True)
    (package / "genome.fna").write_text(">contig\nACGT\n", encoding="utf-8")

    discard_download_package(package, keep_downloads=False)

    assert not package.exists()
    assert not package.parent.exists()


def test_current_datasets_json_layout_is_parsed(monkeypatch):
    response = {
        "accession": "GCF_000000001.1",
        "source_database": "SOURCE_DATABASE_REFSEQ",
        "organism": {"organism_name": "Example species"},
        "assembly_info": {"assembly_level": "Complete Genome", "assembly_name": "Example assembly"},
        "assembly_stats": {"total_sequence_length": "12345"},
    }

    monkeypatch.setattr("fetch_genomes.have_datasets_cli", lambda: True)
    monkeypatch.setattr("fetch_genomes._run", lambda _cmd: (0, __import__("json").dumps(response), ""))

    assemblies = list_taxon_assemblies("1", "complete", "RefSeq", 1)

    assert assemblies == [
        {
            "accession": "GCF_000000001.1",
            "assembly_level": "Complete Genome",
            "assembly_name": "Example assembly",
            "refseq_category": None,
            "total_length": 12345,
            "source": "RefSeq",
        }
    ]


def test_explicit_refseq_accession_is_resolved_when_taxon_listing_is_empty(monkeypatch):
    response = {
        "accession": "GCF_000146045.2",
        "source_database": "SOURCE_DATABASE_REFSEQ",
        "organism": {"organism_name": "Saccharomyces cerevisiae S288C"},
        "assembly_info": {"assembly_level": "Complete Genome", "assembly_name": "R64"},
        "assembly_stats": {"total_sequence_length": "12071326"},
    }

    monkeypatch.setattr("fetch_genomes.have_datasets_cli", lambda: True)
    monkeypatch.setattr("fetch_genomes._run", lambda _cmd: (0, __import__("json").dumps(response), ""))

    assembly = get_assembly_by_accession("GCF_000146045.2", "RefSeq")

    assert assembly["accession"] == "GCF_000146045.2"
    assert assembly["source"] == "RefSeq"
    assert assembly["total_length"] == 12071326


def test_csv_blank_optional_refseq_accession_is_not_interpreted_as_a_pin(tmp_path):
    config = tmp_path / "eukaryotes.csv"
    config.write_text(
        "tax_id,species,target_mb,preferred_refseq_accession\n9606,Homo sapiens,10,\n",
        encoding="utf-8",
    )

    records = load_config(config)

    assert records[0]["preferred_refseq_accession"] is None

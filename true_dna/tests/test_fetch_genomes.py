from pathlib import Path

from fetch_genomes import (
    append_contigs_to_fasta_with_dedup,
    canonical_sequence_hash,
    iter_filtered_contigs,
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

    assert [sequence for _header, sequence in accepted] == ["ACGT", "ACGT"]
    assert rejected_for_ns == []


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

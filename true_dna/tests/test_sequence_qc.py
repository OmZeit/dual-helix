from __future__ import annotations

from tools.sequence_qc import canonical_seq_hash, process_file


def test_qc_reports_invalid_symbols_instead_of_dropping_the_line(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">valid\nacgt\n>invalid\nACGR\n", encoding="utf-8")

    stats = process_file(
        fasta,
        tmp_path / "output",
        min_length=1,
        n_threshold=100.0,
        min_entropy=0.0,
        trim_ns=False,
        dedupe=False,
        seen_seqs=None,
    )

    assert stats.total_seqs == 2
    assert stats.final_seqs == 1
    assert stats.removed_ambiguous == 1
    assert (tmp_path / "output" / "filtered_input.fa").read_text(encoding="utf-8") == ">valid\nACGT\n"


def test_qc_applies_n_threshold_as_a_percentage(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">accepted\nACGTN\n>rejected\nANNNN\n", encoding="utf-8")

    stats = process_file(
        fasta,
        tmp_path / "output",
        min_length=1,
        n_threshold=20.0,
        min_entropy=0.0,
        trim_ns=False,
        dedupe=False,
        seen_seqs=None,
    )

    assert stats.final_seqs == 1
    assert stats.removed_n == 1


def test_qc_deduplication_is_reverse_complement_invariant():
    assert canonical_seq_hash("AACG") == canonical_seq_hash("CGTT")

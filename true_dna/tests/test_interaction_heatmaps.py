from __future__ import annotations

import numpy as np
import pytest
from visualize_interactions import (
    compressed_base_spans,
    compressed_position_summary,
    mask_attention_positions,
    mask_edge_bins,
    parse_index_selection,
    project_attention_to_base_bins,
    write_top_pairs_csv,
)


def test_attention_capture_matches_sdpa_without_dropout():
    torch = pytest.importorskip("torch")
    from dna_model.backbones.common import TransformerEncoderLayerRoPE

    torch.manual_seed(7)
    layer = TransformerEncoderLayerRoPE(
        embed_dim=8,
        num_heads=2,
        dim_feedforward=16,
        dropout=0.0,
        num_kv_heads=2,
    )
    layer.eval()
    x = torch.randn(2, 5, 8)

    sdpa_out = layer(x)[0]
    captured = layer(x, capture_attention=True, attention_heads={1})

    assert torch.allclose(captured[0], sdpa_out, atol=1e-6)
    assert captured[3].shape == (2, 1, 5, 5)
    assert torch.allclose(captured[3].sum(dim=-1), torch.ones(2, 1, 5), atol=1e-6)


def test_parse_index_selection_accepts_all_lists_and_single_override():
    assert parse_index_selection("all", max_count=4, name="layer") is None
    assert parse_index_selection("2,0,2", max_count=4, name="layer") == [0, 2]
    assert parse_index_selection("all", max_count=4, name="layer", single=3) == [3]

    with pytest.raises(ValueError, match="out of range"):
        parse_index_selection("4", max_count=4, name="layer")


def test_tokenizer_offsets_are_opt_in_and_padding_aligned():
    pytest.importorskip("torch")
    from dna_model.tokenizer import DnaTokenizer

    tokenizer = DnaTokenizer(max_length=12)

    without_offsets = tokenizer.encode("ACGTACGT", max_length=12, return_tensors=None)
    with_offsets = tokenizer.encode("ACGTACGT", max_length=12, return_tensors=None, return_offsets=True)

    assert "offsets" not in without_offsets
    assert len(with_offsets["offsets"]) == len(with_offsets["input_ids"]) == 12
    assert len(with_offsets["offsets"]) == len(with_offsets["attention_mask"])
    assert all(len(pair) == 2 for pair in with_offsets["offsets"])


def test_base_aware_projection_uses_bpe_offsets_and_row_normalizes():
    offsets = [(0, 2), (2, 4), (4, 6), (6, 8)]
    attention_mask = [1, 1, 1, 1]
    spans = compressed_base_spans(
        offsets,
        attention_mask,
        compressed_length=2,
        compression_stride=2,
    )
    attention = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    matrix, bin_edges = project_attention_to_base_bins(attention, spans, sequence_length=8, max_bins=4)

    assert spans == [(0, 4), (4, 8)]
    assert matrix.shape == (4, 4)
    assert np.allclose(bin_edges, np.asarray([0, 2, 4, 6, 8], dtype=np.float64))
    assert matrix[0, 2] > 0
    assert matrix[0, 3] > 0
    assert np.allclose(matrix.sum(axis=1), np.ones(4))


def test_special_attention_sink_positions_can_be_masked_before_projection():
    offsets = [(0, 0), (0, 2), (2, 4), (0, 0)]
    attention_mask = [1, 1, 1, 1]
    input_ids = [2, 10, 11, 3]
    summary = compressed_position_summary(
        offsets,
        attention_mask,
        input_ids,
        special_token_ids=[0, 1, 2, 3, 4],
        compressed_length=2,
        compression_stride=2,
    )
    attention = np.asarray(
        [
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    masked = mask_attention_positions(
        attention,
        summary["special_only_positions"] + summary["mixed_special_positions"],
    )

    assert summary["spans"] == [(0, 2), (2, 4)]
    assert summary["mixed_special_positions"] == [0, 1]
    assert np.allclose(masked, np.zeros_like(masked))


def test_edge_bin_mask_hides_edges_and_top_pair_inputs_ignore_nan(tmp_path):
    matrix = np.ones((5, 5), dtype=np.float32)
    masked = mask_edge_bins(matrix, 1)
    csv_path = tmp_path / "top_pairs.csv"
    write_top_pairs_csv(csv_path, masked, np.arange(6, dtype=np.float64), top_k=3)

    assert np.isnan(masked[0]).all()
    assert np.isnan(masked[-1]).all()
    assert np.isnan(masked[:, 0]).all()
    assert np.isnan(masked[:, -1]).all()
    assert np.isfinite(masked[2, 2])
    assert ",0," not in csv_path.read_text(encoding="utf-8")

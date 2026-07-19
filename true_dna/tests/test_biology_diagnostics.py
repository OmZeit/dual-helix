from __future__ import annotations

import random
from collections import Counter

import torch
from dna_model.backbones.dual_helix import WindowAttention
from dna_model.benchmarks import (
    bpe_rc_diagnostics,
    build_negative_control_records,
    dinucleotide_shuffle,
)
from dna_model.config import DnaConfig


class TinyTokenizer:
    def encode(self, seq, max_length=None, return_tensors=None):
        ids = [ord(ch) % 17 for ch in seq[:max_length]]
        rc_len = max_length or len(seq)
        ids = ids + [0] * (rc_len - len(ids))
        mask = [1 if i < len(seq[:rc_len]) else 0 for i in range(rc_len)]
        return {"input_ids": ids, "attention_mask": mask, "kmer_ids": ids}


def test_bpe_rc_diagnostics_returns_rates():
    out = bpe_rc_diagnostics(TinyTokenizer(), ["ACGTACGT"], max_length=8)

    assert out["num_sequences"] == 1
    assert "mean_token_length_abs_diff" in out
    assert "mean_reversed_token_id_match" in out


def test_negative_controls_keep_labels_balanced():
    records = build_negative_control_records(["ACGTACGT", "GGCCAATT"], mode="dinuc")
    labels = [label for _seq, label in records]

    assert labels.count("real") == 2
    assert labels.count("dinuc") == 2


def test_dinucleotide_shuffle_preserves_overlapping_pair_counts():
    sequence = "AACGTCGATGCAACGT"

    shuffled = dinucleotide_shuffle(sequence, random.Random(7))

    assert len(shuffled) == len(sequence)
    assert Counter(zip(shuffled, shuffled[1:], strict=False)) == Counter(zip(sequence, sequence[1:], strict=False))


def test_shifted_window_attention_does_not_wrap_edge_signal():
    cfg = DnaConfig(
        hidden_size=8,
        num_attention_heads=2,
        microscope_window_size=4,
        vocab_size=16,
    )
    attn = WindowAttention(cfg, shift_size=2)
    with torch.no_grad():
        attn.qkv.weight.zero_()
        attn.o_proj.weight.copy_(torch.eye(8))
        attn.qkv.weight[16:24, :] = torch.eye(8)

    x = torch.zeros(1, 8, 8)
    x[0, -1, :] = 100.0
    out = attn(x)

    assert torch.allclose(out[0, 0], torch.zeros(8), atol=1e-6)

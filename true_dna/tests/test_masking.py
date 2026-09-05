from __future__ import annotations

import pytest
import torch
from dna_model import masking
from dna_model.config import IGNORE_INDEX


def test_motif_weight_dict_accepts_int_and_string_keys():
    input_ids = torch.tensor([[5, 6]], dtype=torch.long)

    masked, labels = masking.bpe_masking(
        input_ids,
        vocab_size=10,
        mask_prob=1.0,
        special_tokens_max_id=4,
        motif_importance_weights={"5": 0.0, 6: 1.0},
    )

    assert labels.tolist() == [[IGNORE_INDEX, 6]]
    assert masked.tolist() == [[5, 4]]


def test_random_replacement_samples_vocab_ids_not_batch_pool(monkeypatch):
    input_ids = torch.tensor([[5, 5, 5]], dtype=torch.long)

    rand_calls = 0

    def fake_rand(shape, device=None, generator=None):
        nonlocal rand_calls
        rand_calls += 1
        fill = 0.0 if rand_calls == 1 else 0.85
        return torch.full(shape, fill, device=device)

    def fake_randint(low, high, size, dtype=None, device=None, generator=None):
        return torch.ones(size, dtype=dtype, device=device)

    monkeypatch.setattr(masking.torch, "rand", fake_rand)
    monkeypatch.setattr(masking.torch, "randint", fake_randint)

    x_masked, _ = masking.bpe_masking(
        input_ids,
        vocab_size=8,
        mask_prob=1.0,
        special_tokens_max_id=4,
        replacement_strategy="bert_80_10_10",
    )

    assert x_masked.tolist() == [[6, 6, 6]]


def test_gpu_span_mask_honors_mean_span_length():
    input_ids = torch.arange(5, 17, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    generator = torch.Generator().manual_seed(7)

    _, labels, _, mask_bool = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=32,
        mask_fraction=0.25,
        mean_span_length=3,
        min_frac=0.25,
        max_frac=0.25,
        generator=generator,
    )

    masked_positions = torch.nonzero(mask_bool[0], as_tuple=False).flatten().tolist()
    assert labels.ne(IGNORE_INDEX).sum().item() == 3
    assert masked_positions == list(range(masked_positions[0], masked_positions[0] + 3))


def test_gpu_span_mask_excludes_special_and_padding_tokens():
    input_ids = torch.tensor([[2, 5, 6, 3, 0, 7]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 1]], dtype=torch.long)

    _, labels, _, mask_bool = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=16,
        mask_fraction=1.0,
        mean_span_length=8,
        min_frac=1.0,
        max_frac=1.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert mask_bool.tolist() == [[False, True, True, False, False, True]]
    assert labels.tolist() == [[IGNORE_INDEX, 5, 6, IGNORE_INDEX, IGNORE_INDEX, 7]]


def test_gpu_span_mask_defaults_to_hiding_every_selected_target():
    input_ids = torch.tensor([[2, 5, 6, 3, 0, 7]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 1]], dtype=torch.long)

    masked, labels, _, mask_bool = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=16,
        mask_fraction=1.0,
        mean_span_length=8,
        min_frac=1.0,
        max_frac=1.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert mask_bool.tolist() == [[False, True, True, False, False, True]]
    assert labels.tolist() == [[IGNORE_INDEX, 5, 6, IGNORE_INDEX, IGNORE_INDEX, 7]]
    assert masked.tolist() == [[2, 4, 4, 3, 0, 4]]


def test_gpu_span_mask_retains_legacy_80_10_10_control(monkeypatch):
    input_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    monkeypatch.setattr(
        masking.torch,
        "rand",
        lambda shape, device=None, generator=None: torch.full(shape, 0.95, device=device),
    )
    masked, labels, _, mask_bool = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=10,
        mask_fraction=1.0,
        mean_span_length=3,
        replacement_strategy="bert_80_10_10",
        generator=torch.Generator().manual_seed(1),
    )

    assert mask_bool.all()
    assert labels.tolist() == input_ids.tolist()
    assert masked.tolist() == input_ids.tolist()


def test_gpu_span_mask_does_not_clamp_explicit_fifteen_percent_request():
    input_ids = torch.arange(5, 105, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    _, labels, _, _ = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=128,
        mask_fraction=0.15,
        mean_span_length=3,
        generator=torch.Generator().manual_seed(11),
    )

    assert labels.ne(IGNORE_INDEX).sum().item() == 15


def test_gpu_span_mask_default_selects_fifteen_percent():
    input_ids = torch.arange(5, 105, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    _, labels, _, _ = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=128,
        generator=torch.Generator().manual_seed(11),
    )

    assert labels.ne(IGNORE_INDEX).sum().item() == 15


def test_gpu_span_mask_respects_requested_fraction_bounds():
    input_ids = torch.arange(5, 105, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    _, labels, _, _ = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=128,
        mask_fraction=0.30,
        mean_span_length=5,
        min_frac=0.30,
        max_frac=0.30,
        generator=torch.Generator().manual_seed(11),
    )

    assert labels.ne(IGNORE_INDEX).sum().item() == 30


def test_base_coordinate_masking_maps_raw_spans_to_token_boundaries():
    input_ids = torch.tensor([[10, 11]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    offsets = torch.tensor([[[0, 2], [2, 4]]], dtype=torch.long)

    _, labels, _, mask_bool = masking.gpu_span_mask(
        input_ids,
        attention_mask,
        vocab_size=12,
        mask_fraction=0.25,
        mean_span_length=1,
        min_frac=0.0,
        max_frac=1.0,
        generator=torch.Generator().manual_seed(7),
        token_offsets=offsets,
        coordinate_system="base",
        replacement_strategy="mask",
    )

    assert mask_bool.sum().item() == 1
    assert labels[mask_bool].item() in {10, 11}


def test_base_coordinate_masking_requires_offsets():
    with pytest.raises(ValueError, match="token_offsets"):
        masking.gpu_span_mask(
            torch.tensor([[5, 6]]),
            torch.ones(1, 2, dtype=torch.long),
            vocab_size=10,
            coordinate_system="base",
        )

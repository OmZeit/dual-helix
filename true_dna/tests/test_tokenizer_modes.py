from __future__ import annotations

import math

import pytest
import torch
from dna_model import utils
from dna_model.checkpoint_loading import config_from_checkpoint
from dna_model.tokenizer import (
    BaseTokenizer,
    build_tokenizer,
    normalize_tokenizer_type,
    pretokenized_paths,
)


def test_tokenizer_mode_aliases_are_canonicalized():
    assert normalize_tokenizer_type("bpe") == "bpe"
    assert normalize_tokenizer_type("base") == "base"
    assert normalize_tokenizer_type("single_base") == "base"
    assert normalize_tokenizer_type("base-level") == "base"
    with pytest.raises(ValueError, match="Unknown tokenizer type"):
        normalize_tokenizer_type("codon")


def test_base_tokenizer_applies_special_tokens_offsets_and_padding():
    tokenizer = build_tokenizer("base", max_length=8, return_kmer_ids=False)
    encoded = tokenizer.encode("ACGT", return_tensors=None, return_offsets=True)

    assert isinstance(tokenizer, BaseTokenizer)
    assert tokenizer.tokenizer_type == "base"
    assert encoded["input_ids"] == [
        tokenizer.cls_token_id,
        tokenizer.char_to_id["A"],
        tokenizer.char_to_id["C"],
        tokenizer.char_to_id["G"],
        tokenizer.char_to_id["T"],
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.pad_token_id,
    ]
    assert encoded["offsets"][:6] == [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (0, 0)]
    assert tokenizer.decode(encoded["input_ids"]) == "ACGT"


def test_pretokenized_cache_paths_do_not_collide():
    bpe = pretokenized_paths("records.fa", "bpe")
    base = pretokenized_paths("records.fa", "base")

    assert bpe["input_ids"] == "records.fa_input_ids.npy"
    assert base["input_ids"] == "records.fa_base_input_ids.npy"
    assert set(bpe.values()).isdisjoint(base.values())


class _BpeTokenizerStub:
    tokenizer_type = "bpe"
    vocab_size = 12
    pad_token_id = 0
    unk_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    mask_token_id = 4
    kmer_size = 3
    k_mer_size = 3
    kmer_vocab_size = 65
    kmer_pad_id = 0

    @staticmethod
    def token_base_length(token_id: int) -> int:
        return {10: 2, 11: 3}.get(token_id, 0)


class _FixedPredictions(torch.nn.Module):
    def __init__(self, predictions: list[int], vocab_size: int):
        super().__init__()
        logits = torch.zeros(1, len(predictions), vocab_size)
        for position, token_id in enumerate(predictions):
            logits[0, position, token_id] = 10.0
        self.register_buffer("fixed_logits", logits)

    def forward(self, input_ids, **_kwargs):
        return {"logits": self.fixed_logits.expand(input_ids.shape[0], -1, -1)}


def _keep_all_targets(input_ids, attention_mask, kmer_ids=None, **_kwargs):
    labels = input_ids.clone()
    return input_ids, labels, kmer_ids, attention_mask.bool()


def test_evaluate_reports_base_accuracy_only_for_base_mode(monkeypatch):
    monkeypatch.setattr(utils, "gpu_span_mask", _keep_all_targets)
    tokenizer = BaseTokenizer(return_kmer_ids=False)
    batch = {
        "input_ids": torch.tensor([[tokenizer.char_to_id["A"], tokenizer.char_to_id["C"]]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    model = _FixedPredictions([tokenizer.char_to_id["A"], tokenizer.char_to_id["G"]], tokenizer.vocab_size)

    metrics = utils.evaluate(model, [batch], torch.device("cpu"), False, 0, tokenizer, max_batches=0)

    assert metrics["metric_granularity"] == "base"
    assert metrics["base_accuracy"] == pytest.approx(0.5)
    assert metrics["accuracy_A"] == pytest.approx(1.0)
    assert metrics["accuracy_C"] == pytest.approx(0.0)
    assert metrics["support_A"] == 1
    assert metrics["support_C"] == 1
    assert metrics["support_G"] == 0
    assert "accuracy_G" not in metrics
    assert "token_accuracy" not in metrics
    assert metrics["masked_bases"] == metrics["masked_tokens"] == 2
    assert metrics["bits_per_base"] == pytest.approx(metrics["base_loss"] / math.log(2.0))


def test_evaluate_reports_bpe_token_and_base_normalized_metrics(monkeypatch):
    monkeypatch.setattr(utils, "gpu_span_mask", _keep_all_targets)
    tokenizer = _BpeTokenizerStub()
    batch = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    model = _FixedPredictions([10, 10], tokenizer.vocab_size)

    metrics = utils.evaluate(model, [batch], torch.device("cpu"), False, 0, tokenizer, max_batches=0)

    assert metrics["metric_granularity"] == "bpe_token"
    assert metrics["token_accuracy"] == pytest.approx(0.5)
    assert "base_accuracy" not in metrics
    assert "accuracy_A" not in metrics
    assert metrics["masked_tokens"] == 2
    assert metrics["masked_bases"] == 5
    assert metrics["base_normalized_loss"] == pytest.approx(metrics["token_loss"] * 2 / 5)
    assert metrics["bits_per_base"] == pytest.approx(metrics["base_normalized_loss"] / math.log(2.0))


def test_checkpoint_config_rejects_tokenizer_mode_mismatch():
    with pytest.raises(ValueError, match="Tokenizer mismatch for tokenizer_type"):
        config_from_checkpoint({"tokenizer_type": "bpe"}, BaseTokenizer())

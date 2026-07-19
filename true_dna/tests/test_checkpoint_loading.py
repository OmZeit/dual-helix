from __future__ import annotations

import logging

import pytest
import torch
from dna_model.checkpoint_loading import (
    config_from_checkpoint,
    load_state_dict_with_report,
    normalize_state_dict_keys,
)
from torch import nn


def _model() -> nn.Linear:
    return nn.Linear(2, 1)


class _TokenizerStub:
    vocab_size = 32
    pad_token_id = 0
    unk_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    mask_token_id = 4
    kmer_size = 3
    kmer_vocab_size = 65


def test_load_state_dict_with_report_accepts_matching_state(caplog):
    model = _model()
    state = model.state_dict()

    with caplog.at_level(logging.WARNING):
        load_state_dict_with_report(model, state, checkpoint_name="matching")

    assert not caplog.records


def test_load_state_dict_with_report_logs_missing_and_unexpected_keys(caplog):
    model = _model()
    state = dict(model.state_dict())
    state.pop("bias")
    state["extra.weight"] = torch.ones(1)

    with caplog.at_level(logging.WARNING):
        load_state_dict_with_report(model, state, checkpoint_name="drifted")

    messages = [record.getMessage() for record in caplog.records]
    assert any("drifted missing 1 state_dict keys: bias" in message for message in messages)
    assert any("drifted has 1 unexpected state_dict keys: extra.weight" in message for message in messages)


def test_load_state_dict_with_report_strict_env_fails_on_key_drift(monkeypatch):
    model = _model()
    state = dict(model.state_dict())
    state.pop("bias")
    monkeypatch.setenv("DNA_STRICT_CHECKPOINT_LOAD", "1")

    with pytest.raises(RuntimeError, match="drifted state_dict key drift detected"):
        load_state_dict_with_report(model, state, checkpoint_name="drifted")


def test_load_state_dict_with_report_strict_argument_fails_on_key_drift():
    model = _model()
    state = dict(model.state_dict())
    state["extra.weight"] = torch.ones(1)

    with pytest.raises(RuntimeError, match="strict-call state_dict key drift detected"):
        load_state_dict_with_report(model, state, checkpoint_name="strict-call", strict=True)


def test_config_from_checkpoint_reconstructs_training_aliases():
    config = config_from_checkpoint(
        {
            "max_length": 2048,
            "bridge_interval": 2,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        },
        _TokenizerStub(),
    )

    assert config.max_input_bases == 2048
    assert config.dual_helix_bridge_interval == 2
    assert config.num_key_value_heads == 4
    assert config.vocab_size == 32


def test_config_from_checkpoint_rejects_tokenizer_mismatch():
    with pytest.raises(ValueError, match="Tokenizer mismatch for vocab_size"):
        config_from_checkpoint({"vocab_size": 64}, _TokenizerStub())


def test_normalize_state_dict_keys_removes_only_leading_wrapper_prefixes():
    state = {
        "module._orig_mod.layer.module.weight": torch.ones(1),
        "module._orig_mod.layer.bias": torch.zeros(1),
    }

    normalized = normalize_state_dict_keys(state)

    assert set(normalized) == {"layer.module.weight", "layer.bias"}

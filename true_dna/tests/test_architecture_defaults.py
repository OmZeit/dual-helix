from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from dna_model.backbones.common import MoEFFN
from dna_model.config import DnaConfig


def test_dna_config_defaults_to_dual_helix():
    cfg = DnaConfig()

    assert cfg.backbone == "dual_helix"


def test_invalid_backbone_errors_without_transformer_fallback(caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError, match="backbone"):
        DnaConfig(backbone="transformer")

    assert "not falling back" in caplog.text


def test_training_and_pipeline_defaults_pin_dual_helix():
    root = Path(__file__).parents[1]
    train_text = (root / "scripts" / "train.py").read_text(encoding="utf-8")
    wrapper_text = (root / "scripts" / "run_training_and_benchmark.sh").read_text(encoding="utf-8")

    assert 'default="dual_helix"' in train_text
    assert "--backbone dual_helix" in wrapper_text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_attention_heads": 8, "num_key_value_heads": 3}, "num_key_value_heads"),
        ({"dual_helix_bridge_interval": 0}, "bridge intervals"),
        ({"moe_num_experts": 2, "moe_top_k": 3}, "moe_top_k"),
        ({"frameshift_prob": 1.1}, "frameshift_prob"),
        ({"expert_drop_prob": -0.1}, "expert_drop_prob"),
    ],
)
def test_dna_config_rejects_invalid_runtime_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DnaConfig(**kwargs)


def test_dna_config_normalizes_legacy_fshm_layer_string():
    cfg = DnaConfig(high_level_layers=4, fshm_layers="1,3")

    assert cfg.fshm_layers == [1, 3]


def test_moe_supports_top_k_routing_and_backpropagation():
    layer = MoEFFN(
        embed_dim=8,
        num_experts=4,
        top_k=2,
        dropout=0.0,
        expert_drop_prob=0.0,
    )
    inputs = torch.randn(2, 5, 8, requires_grad=True)

    output, auxiliary_loss, router_entropy = layer(inputs)
    (output.square().mean() + auxiliary_loss).backward()

    assert output.shape == inputs.shape
    assert torch.isfinite(auxiliary_loss)
    assert torch.isfinite(router_entropy)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_moe_top_one_task_loss_trains_router():
    layer = MoEFFN(
        embed_dim=8,
        num_experts=4,
        top_k=1,
        dropout=0.0,
        expert_drop_prob=0.0,
    )
    output, _auxiliary_loss, _router_entropy = layer(torch.randn(2, 5, 8))

    output.square().mean().backward()

    assert layer.router.weight.grad is not None
    assert torch.count_nonzero(layer.router.weight.grad).item() > 0

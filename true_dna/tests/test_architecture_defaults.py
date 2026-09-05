from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from dna_model.backbones.common import BiMamba, MoEFFN
from dna_model.backbones.dual_helix import MicroscopeBlock
from dna_model.config import DnaConfig
from dna_model.model import DnaModel


def test_dna_config_defaults_to_dual_helix():
    cfg = DnaConfig()

    assert cfg.backbone == "dual_helix"


def test_reference_backbones_are_explicitly_supported():
    assert DnaConfig(backbone="bimamba").backbone == "bimamba"
    assert DnaConfig(backbone="transformer").backbone == "transformer"


def test_invalid_backbone_errors_without_fallback(caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError, match="backbone"):
        DnaConfig(backbone="unsupported")

    assert "not falling back" in caplog.text


def test_transformer_reference_backbone_runs_on_cpu():
    config = DnaConfig(
        backbone="transformer",
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        high_level_layers=1,
        use_kmer_mix=False,
    )
    model = DnaModel(config)
    output = model(
        input_ids=torch.tensor([[2, 5, 6, 3]], dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    assert output["logits"].shape == (1, 4, 16)
    assert output["last_hidden_state"].shape == (1, 4, 8)


class _CausalCumulativeMixer(torch.nn.Module):
    def forward(self, values):
        return values.cumsum(dim=1)


def test_bimamba_valid_outputs_are_invariant_to_extra_right_padding():
    mixer = BiMamba.__new__(BiMamba)
    torch.nn.Module.__init__(mixer)
    mixer.fwd = _CausalCumulativeMixer()
    mixer.bwd = _CausalCumulativeMixer()
    mixer.out_proj = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        mixer.out_proj.weight.fill_(1.0)

    unpadded = torch.tensor([[[1.0], [2.0], [3.0]]])
    padded = torch.tensor([[[1.0], [2.0], [3.0], [99.0], [99.0]]])
    unpadded_mask = torch.zeros(1, 3, dtype=torch.bool)
    padded_mask = torch.tensor([[False, False, False, True, True]])

    unpadded_output = mixer(unpadded, padding_mask=unpadded_mask)
    padded_output = mixer(padded, padding_mask=padded_mask)

    assert torch.equal(padded_output[:, :3], unpadded_output)
    assert torch.count_nonzero(padded_output[:, 3:]) == 0


def test_dual_helix_microscope_threads_padding_mask_to_bimamba():
    mixer = BiMamba.__new__(BiMamba)
    torch.nn.Module.__init__(mixer)
    mixer.fwd = _CausalCumulativeMixer()
    mixer.bwd = _CausalCumulativeMixer()
    mixer.out_proj = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        mixer.out_proj.weight.fill_(1.0)

    block = MicroscopeBlock.__new__(MicroscopeBlock)
    torch.nn.Module.__init__(block)
    block.mixer = mixer
    block.norm = torch.nn.Identity()
    block.drop_path = torch.nn.Identity()

    unpadded = torch.tensor([[[1.0], [2.0], [3.0]]])
    padded = torch.tensor([[[1.0], [2.0], [3.0], [99.0], [99.0]]])
    unpadded_mask = torch.zeros(1, 3, dtype=torch.bool)
    padded_mask = torch.tensor([[False, False, False, True, True]])

    unpadded_output = block(unpadded, padding_mask=unpadded_mask)
    padded_output = block(padded, padding_mask=padded_mask)

    assert torch.equal(padded_output[:, :3], unpadded_output)
    assert torch.count_nonzero(padded_output[:, 3:]) == 0


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

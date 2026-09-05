from __future__ import annotations

from pathlib import Path


def test_training_wandb_logs_use_custom_optimizer_step_axis():
    text = (Path(__file__).parents[1] / "scripts" / "train.py").read_text(encoding="utf-8")

    assert 'wandb.define_metric("global_step")' in text
    assert 'wandb.define_metric("train/*", step_metric="global_step")' in text
    assert 'wandb.define_metric("eval/*", step_metric="global_step")' in text
    assert 'wandb.define_metric("gates/*", step_metric="global_step")' in text
    assert 'wandb.define_metric("resource/*", step_metric="global_step")' in text
    assert "resource/cuda_peak_allocated_mib" in text
    assert "resource/cuda_peak_reserved_mib" in text
    assert "wandb.log(log_dict)" in text
    assert "wandb.log(eval_log_payload)" in text
    assert "wandb.log(log_dict, step=global_step)" not in text
    assert "wandb.log(eval_log_payload, step=global_step)" not in text


def test_gate_hook_buffers_metrics_instead_of_partial_wandb_logs():
    text = (Path(__file__).parents[1] / "scripts" / "train.py").read_text(encoding="utf-8")

    assert "_GATE_STATS" in text
    assert "_drain_gate_stats()" in text
    assert "commit=False" not in text

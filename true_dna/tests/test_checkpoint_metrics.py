from __future__ import annotations

from dna_model.checkpoint_manager import _extract_display_loss


def test_extract_display_loss_prefers_nested_eval_metric():
    info = {
        "best_loss": 9.0,
        "metrics": {
            "eval": {"loss": 1.25},
            "best": {"eval_loss": 1.5},
            "eval_loss": 2.0,
        },
    }

    assert _extract_display_loss(info) == 1.25


def test_extract_display_loss_supports_flat_legacy_metric():
    info = {"best_loss": 9.0, "metrics": {"eval_loss": 2.5}}

    assert _extract_display_loss(info) == 2.5


def test_extract_display_loss_falls_back_to_checkpoint_best_loss():
    info = {"best_loss": 3.5, "metrics": {}}

    assert _extract_display_loss(info) == 3.5

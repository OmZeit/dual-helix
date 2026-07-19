"""Checkpoint loading helpers."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

LOGGER = logging.getLogger(__name__)

_STRICT_CHECKPOINT_ENV = "DNA_STRICT_CHECKPOINT_LOAD"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def config_from_checkpoint(values: Mapping[str, Any], tokenizer: Any):
    """Reconstruct a :class:`DnaConfig` from saved training arguments."""

    from .config import DnaConfig

    valid_keys = set(inspect.signature(DnaConfig.__init__).parameters) - {"self"}
    config_values = {key: value for key, value in values.items() if key in valid_keys}

    aliases = {
        "max_length": "max_input_bases",
        "bridge_interval": "dual_helix_bridge_interval",
    }
    for source, target in aliases.items():
        if target not in config_values and source in values:
            config_values[target] = values[source]

    tokenizer_fields = {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "bpe"),
        "tokenizer_vocab_sha256": getattr(tokenizer, "vocab_sha256", None),
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "kmer_size": tokenizer.kmer_size,
        "kmer_vocab_size": tokenizer.kmer_vocab_size,
    }
    for name, actual_value in tokenizer_fields.items():
        saved_value = config_values.get(name)
        if name == "tokenizer_type" and saved_value is not None:
            saved_value = {"single_base": "base", "base_level": "base"}.get(str(saved_value), str(saved_value))
        if (
            saved_value is not None
            and name in {"tokenizer_type", "tokenizer_vocab_sha256"}
            and saved_value != actual_value
        ):
            raise ValueError(f"Tokenizer mismatch for {name}: checkpoint={saved_value}, tokenizer={actual_value}")
        if (
            saved_value is not None
            and name not in {"tokenizer_type", "tokenizer_vocab_sha256"}
            and int(saved_value) != actual_value
        ):
            raise ValueError(f"Tokenizer mismatch for {name}: checkpoint={saved_value}, tokenizer={actual_value}")
        config_values[name] = actual_value

    if config_values.get("use_kmer_mix", True):
        config_values.setdefault("kmer_vocab_sizes", [tokenizer.kmer_vocab_size])
    return DnaConfig(**config_values)


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove leading prefixes introduced by compilation or distributed wrappers."""

    normalized = dict(state_dict)
    prefixes = ("_orig_mod.", "module.")
    changed = True
    while normalized and changed:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in normalized):
                normalized = {key[len(prefix) :]: value for key, value in normalized.items()}
                changed = True
                break
    return normalized


def _strict_checkpoint_load_enabled() -> bool:
    return os.environ.get(_STRICT_CHECKPOINT_ENV, "").strip().lower() in _TRUE_VALUES


def _format_key_sample(keys: Sequence[str], *, limit: int = 20) -> str:
    shown = list(keys[:limit])
    suffix = "" if len(keys) <= limit else f", ... (+{len(keys) - limit} more)"
    return ", ".join(shown) + suffix


def load_state_dict_with_report(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_name: str = "checkpoint",
    logger: logging.Logger | None = None,
    strict: bool | None = None,
) -> None:
    """Load a state dict and report non-strict checkpoint key drift.

    Args:
        module: Target PyTorch module.
        state_dict: State dictionary to load into ``module``.
        checkpoint_name: Human-readable checkpoint source for log messages.
        logger: Optional logger; defaults to this module's logger.
        strict: If ``True``, fail on missing or unexpected keys. If ``None``,
            ``DNA_STRICT_CHECKPOINT_LOAD=1`` enables the same failure mode.
    """
    active_logger = logger or LOGGER
    fail_on_key_drift = _strict_checkpoint_load_enabled() if strict is None else strict
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    if missing_keys:
        active_logger.warning(
            "%s missing %d state_dict keys: %s",
            checkpoint_name,
            len(missing_keys),
            _format_key_sample(missing_keys),
        )
    if unexpected_keys:
        active_logger.warning(
            "%s has %d unexpected state_dict keys: %s",
            checkpoint_name,
            len(unexpected_keys),
            _format_key_sample(unexpected_keys),
        )

    if fail_on_key_drift and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"{checkpoint_name} state_dict key drift detected: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )

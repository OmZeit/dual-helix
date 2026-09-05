import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import argparse
import glob
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from dna_model.benchmark_logging import (
    build_eval_history_record,
    build_eval_log_payload,
    log_eval_to_tensorboard,
)
from dna_model.checkpoint_loading import (
    config_from_checkpoint,
    load_state_dict_with_report,
    normalize_state_dict_keys,
)
from dna_model.config import DEFAULT_MASK_FRACTION, DEFAULT_MEAN_SPAN_LENGTH, IGNORE_INDEX, DnaConfig
from dna_model.loaders import build_loaders_balanced
from dna_model.losses import dna_target_cross_entropy
from dna_model.masking import gpu_span_mask
from dna_model.model import DnaModel
from dna_model.tokenizer import build_tokenizer, normalize_tokenizer_type
from dna_model.utils import EMA, evaluate, get_scheduler, masked_mean_pool, parameter_groups
from dna_model.weighted_loader import build_loaders_weighted
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from dna_model.checkpoint_manager import CheckpointManager

    _HAS_CKPT_MANAGER = True
except Exception:
    _HAS_CKPT_MANAGER = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[Warning] tqdm not installed. Install with: pip install tqdm")


_GATE_STATS: dict[str, float] = {}


# Experiment logging


def _configure_wandb_metrics() -> None:
    """Configure W&B metric grouping around optimizer global_step."""
    if wandb.run is None:
        return
    try:
        wandb.define_metric("global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")
        wandb.define_metric("gates/*", step_metric="global_step")
        # W&B's automatic system monitor is not guaranteed to emit GPU memory
        # samples on every platform.  These are explicit PyTorch allocator
        # measurements, which makes memory comparable across ablation runs.
        wandb.define_metric("resource/*", step_metric="global_step")
    except Exception as e:
        logging.warning("Could not configure W&B metric axes: %s", e)


def _drain_gate_stats() -> dict[str, float]:
    stats = dict(_GATE_STATS)
    _GATE_STATS.clear()
    return stats


def _to_hist_array(t: torch.Tensor, cap: int = 1_000_000) -> np.ndarray:
    """Helper to convert a tensor to a downsampled numpy array for W&B histograms."""
    x = t.detach()
    if x.is_cuda:
        x = x.cpu()
    x = x.to(torch.float32).reshape(-1)
    if x.numel() > cap:
        idx = torch.randint(0, x.numel(), (cap,))
        x = x[idx]
    return x.numpy()


def make_gate_hook(name):
    def hook(module, input, output):
        # output is the gate value (0 to 1) after Sigmoid
        gate = output.detach()
        _GATE_STATS[f"gates/{name}_mean"] = gate.mean().item()
        _GATE_STATS[f"gates/{name}_std"] = gate.std(unbiased=False).item()
        _GATE_STATS[f"gates/{name}_sat"] = (gate > 0.9).float().mean().item()

    return hook


def _metric_value(value):
    """Return a JSON/checkpoint-safe scalar or nested metric value."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _metric_value(value.detach().float().item())
        return None
    if isinstance(value, np.generic):
        return _metric_value(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): cleaned for k, v in value.items() if (cleaned := _metric_value(v)) is not None}
    if isinstance(value, (list, tuple)):
        cleaned = [_metric_value(v) for v in value]
        return [v for v in cleaned if v is not None]
    return None


def _cuda_memory_metrics(device) -> dict[str, int]:
    if getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _wandb_cuda_memory_metrics(device) -> dict[str, float]:
    """Return allocator memory measurements in MiB for W&B time-series logs.

    ``max_*`` values are peaks since the start of the current invocation (or
    since a resumed invocation began), not a sampled approximation.  They are
    intentionally logged under ``resource/`` rather than ``system/`` so they
    cannot be confused with W&B's optional system-monitor metrics.
    """

    raw = _cuda_memory_metrics(device)
    if not raw:
        return {}
    mib = float(1024**2)
    return {
        "resource/cuda_allocated_mib": raw["memory_allocated_bytes"] / mib,
        "resource/cuda_reserved_mib": raw["memory_reserved_bytes"] / mib,
        "resource/cuda_peak_allocated_mib": raw["max_memory_allocated_bytes"] / mib,
        "resource/cuda_peak_reserved_mib": raw["max_memory_reserved_bytes"] / mib,
    }


def write_run_summary(
    run_dir: str | Path,
    *,
    global_step: int,
    total_params: int,
    final_eval,
    device,
) -> None:
    """Atomically persist final metrics needed by benchmark manifests."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "global_step": int(global_step),
        "parameters": int(total_params),
        "final_eval": _metric_value(final_eval),
        "cuda_memory": _cuda_memory_metrics(device),
    }
    temporary = output_dir / "run_summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_dir / "run_summary.json")


# Reproducibility and performance


def seed_all(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_enable_tf32(args, device):
    """Enable TF32 for faster training on Ampere+ GPUs."""
    if getattr(device, "type", str(device)) == "cuda" and args.tf32:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                print("[OK] TF32 enabled (Ampere+ GPU detected).")
            else:
                print(f"[WARN] TF32 requested but GPU compute capability {cap[0]}.{cap[1]} < 8.0")
        else:
            print("[WARN] TF32 requested but CUDA not available.")
    else:
        print("TF32 disabled.")


# FASTA utilities


def list_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns and sort intelligently."""
    paths = []
    for p in patterns:
        expanded = glob.glob(p, recursive=True)
        if not expanded:
            if os.path.exists(p):
                paths.append(p)
        else:
            paths.extend(expanded)

    def key_fn(p):
        base = os.path.basename(p)
        if base.startswith("filtered_genome"):
            try:
                n = int(base.replace("filtered_genome", "").split(".")[0])
            except Exception:
                n = 10**9
            return (0, n)
        return (1, base)

    return sorted(set(paths), key=key_fn)


# Training helpers


class SpanBoundaryHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_boundary_labels(mask_bool, attention_mask):
    boundary_targets = torch.zeros_like(mask_bool, dtype=torch.float)
    boundary_targets[:, 1:] = (mask_bool[:, 1:] != mask_bool[:, :-1]).float()
    boundary_targets[:, :-1] = torch.max(boundary_targets[:, :-1], (mask_bool[:, 1:] != mask_bool[:, :-1]).float())
    boundary_targets[attention_mask == 0] = -1.0  # padding
    return boundary_targets


def compute_span_boundary_loss(boundary_head, features, boundary_labels, lambda_weight):
    logits = boundary_head(features).squeeze(-1)

    valid = boundary_labels >= 0.0
    if not valid.any():
        return features.new_tensor(0.0)

    valid_logits = logits[valid]
    valid_targets = boundary_labels[valid]

    # Dynamic pos_weight to boost sparse boundary edge signals
    num_pos = valid_targets.sum()
    num_neg = valid.sum() - num_pos
    pos_weight = (num_neg / num_pos.clamp_min(1.0)).clamp_max(100.0)

    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=pos_weight, reduction="mean")
    return loss * lambda_weight


def step_mask_and_forward(
    model,
    batch,
    device,
    *,
    mask_fraction: float,
    mean_span_length: float,
    accelerator: Accelerator | None = None,
    class_weights=None,
    tokenizer=None,
    generator=None,
    amp=False,
    autocast_dtype=None,
    loss_lambda_boundary=None,
    boundary_head=None,
    label_smoothing=0.0,
    teacher_model=None,
    kd_temp=2.0,
    alpha_kd=0.5,
    mask_coordinate_system="token",
    mask_replacement_strategy="mask",
):
    """Run one masked-language-modeling forward pass."""
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    kmer_ids = batch.get("kmer_ids")
    if kmer_ids is not None:
        kmer_ids = kmer_ids.to(device, non_blocking=True)
    token_offsets = batch.get("offsets")
    if token_offsets is not None:
        token_offsets = token_offsets.to(device, non_blocking=True)

    vocab_size = getattr(tokenizer, "vocab_size", 4096)

    x_masked, labels, kmer_ids_masked, mask_bool = gpu_span_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        kmer_ids=kmer_ids,
        kmer_k=getattr(tokenizer, "k_mer_size", None),
        kmer_pad_id=getattr(tokenizer, "kmer_pad_id", None),
        vocab_size=vocab_size,
        mask_fraction=mask_fraction,
        mean_span_length=mean_span_length,
        generator=generator,
        replacement_strategy=mask_replacement_strategy,
        token_offsets=token_offsets,
        coordinate_system=mask_coordinate_system,
    )

    context = (
        accelerator.autocast()
        if accelerator
        else torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp)
    )

    # Move optional reverse-complement inputs with the primary batch.
    input_ids_rc_dev = batch["input_ids_rc"].to(device, non_blocking=True) if "input_ids_rc" in batch else None
    attention_mask_rc_dev = (
        batch["attention_mask_rc"].to(device, non_blocking=True) if "attention_mask_rc" in batch else None
    )
    kmer_ids_rc_dev = batch["kmer_ids_rc"].to(device, non_blocking=True) if "kmer_ids_rc" in batch else None

    with context:
        out = model(
            input_ids=x_masked,
            attention_mask=attention_mask,
            labels=None,
            kmer_ids=kmer_ids_masked,
            input_ids_rc=input_ids_rc_dev,
            attention_mask_rc=attention_mask_rc_dev,
            kmer_ids_rc=kmer_ids_rc_dev,
        )
        logits = out["logits"]

        masked = labels != IGNORE_INDEX
        if masked.any():
            valid_target_ids = (
                tokenizer.valid_target_token_ids() if hasattr(tokenizer, "valid_target_token_ids") else None
            )
            loss = dna_target_cross_entropy(
                logits[masked].view(-1, logits.size(-1)),
                labels[masked].view(-1),
                label_smoothing=label_smoothing,
                valid_target_ids=valid_target_ids,
                class_weights=class_weights,
            )
        else:
            loss = logits.sum() * 0.0

        # Knowledge distillation
        if teacher_model is not None and masked.any():
            with torch.no_grad():
                teacher_out = teacher_model(
                    input_ids=x_masked,
                    attention_mask=attention_mask,
                    labels=None,
                    kmer_ids=kmer_ids_masked,
                )
                teacher_logits = teacher_out["logits"]

            # Only compute KD loss on the masked tokens
            s_logits = logits[masked].view(-1, logits.size(-1))
            t_logits = teacher_logits[masked].view(-1, teacher_logits.size(-1))

            # Soften probabilities
            distillation_loss = F.kl_div(
                F.log_softmax(s_logits / kd_temp, dim=-1), F.softmax(t_logits / kd_temp, dim=-1), reduction="batchmean"
            ) * (kd_temp**2)

            loss = (1.0 - alpha_kd) * loss + alpha_kd * distillation_loss

        boundary_loss = None
        if loss_lambda_boundary and loss_lambda_boundary > 0.0 and boundary_head is not None:
            features = out.get("last_hidden_state")
            if features is None:
                raise RuntimeError("Boundary loss requires last_hidden_state model output")
            boundary_labels = compute_boundary_labels(mask_bool, attention_mask)
            boundary_loss = compute_span_boundary_loss(boundary_head, features, boundary_labels, loss_lambda_boundary)
            loss = loss + boundary_loss

    return (
        loss,
        logits,
        labels,
        mask_bool,
        boundary_loss,
        out.get("moe_aux_loss"),
        out.get("fshm_router_aux_loss"),
        out.get("router_entropy"),
    )


class MaskingCurriculum:
    """Mask-fraction curriculum."""

    def __init__(
        self,
        total_steps: int,
        warmup_frac: float = 0.4,
        mask_start: float = 0.15,
        mask_end: float = 0.30,
        mean_span_length: float = DEFAULT_MEAN_SPAN_LENGTH,
    ):
        self.warmup_steps = int(total_steps * warmup_frac)
        self.mask_start = mask_start
        self.mask_end = mask_end
        self.mean_span_length = float(mean_span_length)

        print("\n[Masking Curriculum] Enabled (Cosine, mask-fraction only):")
        print(f"  Steps: 0 -> {self.warmup_steps} (then hold at {mask_end:.2f})")
        print(f"  Mask fraction: {mask_start:.2f} -> {mask_end:.2f}")
        print(f"  Mean span length: {self.mean_span_length:.2f} bases")

    def get_params(self, current_step: int):
        """Return the scheduled mask fraction and configured span length."""
        if current_step >= self.warmup_steps:
            return self.mask_end, self.mean_span_length

        p_linear = float(current_step) / float(max(1, self.warmup_steps))
        p_cosine = 0.5 * (1.0 - math.cos(math.pi * p_linear))
        mask_val = self.mask_start + (self.mask_end - self.mask_start) * p_cosine
        return mask_val, self.mean_span_length


def compute_class_weights_from_loader(
    train_loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    sample_batches: int = 200,
    mask_fraction: float = 0.20,
    mean_span_length: float = 3.0,
) -> torch.Tensor:
    """Estimate class frequencies."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    seen = 0
    print("[Class Weights] Estimating class distribution from training loader...")

    iterator = iter(train_loader)
    for _ in range(sample_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        input_ids = batch["input_ids"]

        _, labels, _, _ = gpu_span_mask(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"],
            vocab_size=num_classes,
            mask_fraction=mask_fraction,
            mean_span_length=mean_span_length,
            generator=None,
        )

        masked = labels != IGNORE_INDEX
        if masked.any():
            vals = labels[masked].view(-1)
            valid = vals < num_classes
            if valid.any():
                counts += torch.bincount(vals[valid], minlength=num_classes).cpu()
                seen += int(valid.sum().item())

    if seen == 0:
        print("[Class Weights] Warning: No valid tokens found. Using uniform.")
        return torch.ones(num_classes, device=device)

    freq = counts.float().clamp_min(1)
    weights = freq.mean() / freq
    print(f"[Class Weights] {weights.numpy()} (mean={weights.mean().item():.3f})")
    return weights.to(device)


def scale_learning_rate(optimizer, scheduler, factor):
    """Scale learning rate AFTER checkpoint resume."""
    if factor <= 0 or factor > 1.0:
        raise ValueError(f"factor must be in (0, 1], got {factor}")

    old_lrs = []
    for pg in optimizer.param_groups:
        old_lrs.append(pg["lr"])
        pg["lr"] *= factor

    if hasattr(scheduler, "base_lrs"):
        scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    if hasattr(scheduler, "max_lrs"):
        scheduler.max_lrs = [lr * factor for lr in scheduler.max_lrs]

    print("\n" + "=" * 60)
    print("LR SCALING FOR FINE-TUNING")
    print("=" * 60)
    print(f"Scale factor: {factor:.3f}x")
    for i, (old_lr, pg) in enumerate(zip(old_lrs, optimizer.param_groups, strict=True)):
        print(f"  Group {i}: {old_lr:.6e} -> {pg['lr']:.6e}")
    print("=" * 60 + "\n")


def parse_eval_schedule(schedule_str: str) -> list[tuple[int, int]]:
    """Parse a comma-separated list of start-step and interval pairs."""

    if not schedule_str:
        return []

    schedule = []
    try:
        for part in schedule_str.split(","):
            if not part.strip():
                continue
            step_text, interval_text = part.split(":", maxsplit=1)
            start_step = int(step_text)
            interval = int(interval_text)
            if start_step < 0 or interval <= 0:
                raise ValueError("start steps must be non-negative and intervals must be positive")
            schedule.append((start_step, interval))

        schedule.sort(key=lambda x: x[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid evaluation schedule {schedule_str!r}: {exc}") from exc

    return schedule


# Command-line interface


def parse_args():
    p = argparse.ArgumentParser("DNA Language Model Trainer", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    data_grp = p.add_argument_group("Data")
    data_grp.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        type=normalize_tokenizer_type,
        default="bpe",
        choices=["bpe", "base"],
        help="Primary tokenization granularity (single_base is accepted as a legacy alias for base)",
    )
    data_grp.add_argument(
        "--tokenizer-path",
        "--tokenizer_path",
        default="dna_model/bpe_vocab/tokenizer.json",
        help="BPE tokenizer.json path (ignored for base tokenization)",
    )
    data_grp.add_argument("--fasta", nargs="+", required=True, help="FASTA files or glob patterns")
    data_grp.add_argument(
        "--split-manifest",
        "--split_manifest",
        default=None,
        help="Optional JSONL train/eval assignment manifest created by scripts/build_split_manifest.py",
    )
    data_grp.add_argument("--save_dir", type=str, default="experiments/exp1", help="Directory for logs and checkpoints")
    data_grp.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    data_grp.add_argument("--stride", type=int, default=0, help="Stride for chunking (0 = max_length)")
    data_grp.add_argument("--train_split_ratio", type=float, default=0.99, help="Fraction of data for training")
    data_grp.add_argument("--max_chunks_per_file", type=int, default=0, help="Cap chunks per file (0 = no cap)")
    data_grp.add_argument("--cap_by_file", type=str, default=None, help="JSON file for per-file caps")
    data_grp.add_argument(
        "--weights", type=str, default=None, dest="weights_file", help="JSON {basename: sampling_weight}"
    )
    data_grp.add_argument("--stratify_gc", type=str, default=None, help="Path to GC metadata JSON")
    data_grp.add_argument("--use_smart_batching", action="store_true", help="Enable smart batching (group by length)")

    # Augmentation
    aug_grp = p.add_argument_group("Augmentation")
    aug_grp.add_argument(
        "--use_reverse_prob", type=float, default=0.5, help="Probability of reverse complement augmentation"
    )

    model_grp = p.add_argument_group("Model")
    model_grp.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    model_grp.add_argument("--high_level_layers", type=int, default=12, help="Number of layers")
    model_grp.add_argument("--num_attention_heads", type=int, default=12, help="Number of heads")
    model_grp.add_argument(
        "--num_key_value_heads",
        type=int,
        default=None,
        help="Number of KV heads (GQA). Defaults to num_attention_heads if None.",
    )
    model_grp.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    model_grp.add_argument("--use_conv_ffn", action="store_true", help="Use ConvFFN (SwiGLU)")
    model_grp.add_argument("--conv_kernel_size", type=int, default=7, help="ConvFFN kernel size")
    model_grp.add_argument("--drop_path_rate", type=float, default=0.0, help="Stochastic depth rate")
    model_grp.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    model_grp.add_argument(
        "--k_mer_sizes",
        type=str,
        default="3",
        help="Legacy comma-separated k-mer sizes; the model uses only the first value as one aligned side channel.",
    )
    model_grp.add_argument("--low_level_kernel", type=int, default=6, help="Conv kernel size")
    model_grp.add_argument("--low_level_stride", type=int, default=2, help="Stride for downsampling")
    model_grp.add_argument(
        "--fshm_layers", type=str, default="", help="Comma-separated layer indices for FSHM blocks (e.g. '2,5,8')"
    )
    model_grp.add_argument(
        "--backbone",
        type=str,
        default="dual_helix",
        choices=["dual_helix", "bimamba", "transformer", "legacy"],
        help="Backbone architecture. bimamba and transformer are full-resolution controlled-ablation references.",
    )
    model_grp.add_argument(
        "--use-kmer-mix",
        "--use_kmer_mix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the aligned k-mer embedding side channel.",
    )
    model_grp.add_argument(
        "--no-kmer-mix",
        "--no_kmer_mix",
        action="store_false",
        dest="use_kmer_mix",
        help="Compatibility alias for --no-use-kmer-mix.",
    )
    model_grp.add_argument("--use_moe", action="store_true", help="Use Mixture-of-Experts in Telescope stream")
    model_grp.add_argument("--moe_num_experts", type=int, default=8, help="Number of MoE experts")
    model_grp.add_argument("--moe_top_k", type=int, default=1, help="Top-K routing for MoE")
    model_grp.add_argument("--expert_drop_prob", type=float, default=0.1, help="Stochastic expert dropout probability")
    model_grp.add_argument("--bridge_interval", type=int, default=1, help="Interval for DualHelix bridge attention")
    model_grp.add_argument("--disable_bridge", action="store_true", help="Disable bridge attention (Ablation test)")
    model_grp.add_argument(
        "--use_conv_stem", action="store_true", help="Use the lightweight convolutional stem compatibility path"
    )
    model_grp.add_argument(
        "--microscope_window_size", type=int, default=128, help="Window size for Microscope (Stream A) attention"
    )
    model_grp.add_argument("--ssm_d_state", type=int, default=16, help="SSM state dimension (d_state)")
    model_grp.add_argument("--ssm_d_conv", type=int, default=4, help="SSM convolution kernel size (d_conv)")
    model_grp.add_argument("--ssm_expand", type=int, default=2, help="SSM expansion factor")
    model_grp.add_argument("--fshm_slow_stride", type=int, default=4, help="Stride for FSHM slow stream")
    model_grp.add_argument(
        "--no_mamba2", action="store_false", dest="use_mamba2", help="Disable Mamba-2 and use Mamba-1 instead"
    )
    model_grp.set_defaults(use_mamba2=True)
    model_grp.add_argument(
        "--mamba_version",
        type=str,
        default=None,
        choices=["mamba1", "mamba2", "mamba3"],
        help="Mamba mixer version; overrides --no_mamba2 when set",
    )
    model_grp.add_argument("--mamba2_head_dim", type=int, default=64, help="Mamba-2 head dimension")
    model_grp.add_argument("--mamba2_ngroups", type=int, default=1, help="Mamba-2 number of groups")
    model_grp.add_argument("--moe_beta_z", type=float, default=0.001, help="MoE Z-loss coefficient")
    model_grp.add_argument("--expert_drop_warmup_steps", type=int, default=2000, help="Expert dropout warmup duration")
    model_grp.add_argument(
        "--use_dual_strand", action="store_true", help="Enable dual-strand forward/RC processing in DualHelix backbone"
    )
    model_grp.add_argument(
        "--cross_strand_gate_interval", type=int, default=4, help="Apply CrossStrandGate every N backbone layers"
    )
    model_grp.add_argument(
        "--rc_equivariant_tying",
        action="store_true",
        help="Enable exact strand-swap equivariant parameter tying in dual-strand modules",
    )
    model_grp.add_argument(
        "--frameshift_prob", type=float, default=0.05, help="Probability of single-base pre-BPE indel perturbation"
    )

    # Training
    train_grp = p.add_argument_group("Training")
    train_grp.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    train_grp.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_grp.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_grp.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    train_grp.add_argument("--warmup_steps", type=int, default=1200, help="Warmup steps")
    train_grp.add_argument(
        "--lr_schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="LR schedule"
    )
    train_grp.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    train_grp.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    train_grp.add_argument("--amp", action="store_true", help="Enable AMP (Automatic Mixed Precision)")
    train_grp.add_argument("--tf32", action="store_true", help="Enable TF32 on Ampere+")
    train_grp.add_argument("--compile", action="store_true", help="Use torch.compile")
    train_grp.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode; reduce-overhead reduces Python/kernel-launch overhead for stable shapes",
    )
    train_grp.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average")
    train_grp.add_argument("--use_class_weights", action="store_true", help="Use inverse frequency class weights")
    train_grp.add_argument("--seed", type=int, default=42, help="Random seed for model, loader, and masking RNGs")

    train_grp.add_argument(
        "--lambda_rc_consist", type=float, default=0.1, help="Weight for asymmetric RC consistency loss"
    )
    train_grp.add_argument(
        "--rc_loss_prob", type=float, default=0.3, help="Stochastic prob for RC consistency (0.0-1.0)"
    )
    train_grp.add_argument("--auto_resume", action="store_true", help="Auto resume from latest")

    train_grp.add_argument("--reset_scheduler", action="store_true", help="Reset LR scheduler/optimizer on resume")
    train_grp.add_argument("--use_curriculum", action="store_true", help="Enable masking curriculum")
    train_grp.add_argument(
        "--mask_fraction",
        type=float,
        default=DEFAULT_MASK_FRACTION,
        help="Selected-token/base fraction when the masking curriculum is disabled",
    )
    train_grp.add_argument("--mask_start", type=float, default=0.15, help="Starting mask fraction for curriculum")
    train_grp.add_argument("--mask_end", type=float, default=0.35, help="Ending mask fraction for curriculum")
    train_grp.add_argument(
        "--mean_span_length",
        type=float,
        default=DEFAULT_MEAN_SPAN_LENGTH,
        help="Mean contiguous corruption-span length in the selected coordinate system",
    )
    train_grp.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="DNA-target-only label smoothing; the controlled research protocol uses 0.0",
    )
    train_grp.add_argument(
        "--mask-coordinate-system",
        choices=["token", "base"],
        default="token",
        help="Sample corruption spans in tokenizer positions or original nucleotide coordinates",
    )
    train_grp.add_argument(
        "--mask-replacement-strategy",
        choices=["mask", "bert_80_10_10"],
        default="mask",
        help="Replace every selected target with [MASK], or reproduce BERT's legacy 80/10/10 policy",
    )
    train_grp.add_argument(
        "--lambda_boundary", type=float, default=0.0, help="Weight for boundary prediction auxiliary loss"
    )
    train_grp.add_argument(
        "--lambda_moe_aux",
        type=float,
        default=0.01,
        help="Weight for MoE FFN auxiliary loss (Switch Transformer default: 0.01)",
    )
    train_grp.add_argument(
        "--lambda_fshm_router_aux",
        type=float,
        default=0.001,
        help="Weight for FSHM biological-router auxiliary loss (typically ~0.001)",
    )

    # Checkpointing
    ckpt_grp = p.add_argument_group("Checkpointing")
    ckpt_grp.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint, or 'best', or 'latest'")
    ckpt_grp.add_argument("--save_every", type=int, default=2000, help="Save checkpoint every N global steps")
    ckpt_grp.add_argument("--resume_lr_scale", type=float, default=None, help="Scale LR after resume")
    ckpt_grp.add_argument(
        "--fast_resume",
        action="store_true",
        help="Skip data playback when resuming (faster, slightly different data order)",
    )

    # Logging
    log_grp = p.add_argument_group("Logging")
    log_grp.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    log_grp.add_argument(
        "--eval_interval", type=int, default=1000, help="Evaluate every N steps (default if no schedule)"
    )
    log_grp.add_argument(
        "--eval_schedule",
        type=str,
        default=None,
        help="Dynamic eval schedule 'step:interval,step:interval'. Overrides eval_interval.",
    )
    log_grp.add_argument(
        "--eval_max_batches", type=int, default=500, help="Maximum eval batches per training eval (0 = full eval set)"
    )
    log_grp.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    log_grp.add_argument("--wandb-name", default=None, help="Optional explicit W&B run name")
    log_grp.add_argument("--wandb-group", default=None, help="Optional explicit W&B run group")
    log_grp.add_argument("--wandb-job-type", default=None, help="Optional explicit W&B job type")
    log_grp.add_argument(
        "--wandb-tags",
        default="",
        help="Comma-separated W&B tags. Empty entries are ignored.",
    )

    # Knowledge Distillation
    kd_grp = p.add_argument_group("Knowledge Distillation")
    kd_grp.add_argument("--teacher_model", type=str, default=None, help="Path to teacher checkpoint for KD")
    kd_grp.add_argument("--temperature", type=float, default=2.0, help="Temperature for KD softmax")
    kd_grp.add_argument("--alpha_kd", type=float, default=0.5, help="Weight of KD loss vs CE loss")

    # System
    sys_grp = p.add_argument_group("System")
    sys_grp.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    sys_grp.add_argument("--pin_memory", action="store_true", help="Pin CPU memory for faster transfer")

    # Weighted Sampling
    weight_grp = p.add_argument_group("Weighted Sampling")
    weight_grp.add_argument("--steps_per_epoch", type=int, default=0, help="Force steps per epoch (0=auto)")

    # Debugging
    p.add_argument("--cpu", action="store_true", help="Unsupported: exits because training requires CUDA")
    p.add_argument(
        "--print_params_only",
        action="store_true",
        help="Only build model config, calculate parameter count, print it, and immediately exit",
    )

    return p.parse_args()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _normalize_map_for_files(raw_map, files, *, default=None, cast_fn=None, name="weights"):
    cast_fn = cast_fn or (lambda x: x)
    out = {}
    star_default = raw_map.get("*", default)
    for f in files:
        base = os.path.basename(f)
        if f in raw_map:
            val = raw_map[f]
        elif base in raw_map:
            val = raw_map[base]
        else:
            val = star_default
        if val is None:
            raise ValueError(f"No {name} provided for file '{f}' (basename '{base}'), and no '*' default set.")
        out[base] = cast_fn(val)
    return out


def _parse_k_mer_sizes(value: str) -> list[int]:
    sizes = [int(x) for x in str(value).split(",") if x.strip()]
    if not sizes:
        raise ValueError("--k_mer_sizes must include at least one positive integer")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"--k_mer_sizes must be positive, got {sizes}")
    return sizes


def _parse_fshm_layers(value: str) -> list[int]:
    try:
        return [int(item) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--fshm_layers must be a comma-separated list of integers") from exc


def _validate_training_args(args: argparse.Namespace) -> None:
    positive = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "grad_accum": args.grad_accum,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "temperature": args.temperature,
        "mean_span_length": args.mean_span_length,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")

    non_negative = {
        "stride": args.stride,
        "workers": args.workers,
        "warmup_steps": args.warmup_steps,
        "save_every": args.save_every,
        "steps_per_epoch": args.steps_per_epoch,
        "max_chunks_per_file": args.max_chunks_per_file,
        "eval_max_batches": args.eval_max_batches,
        "weight_decay": args.weight_decay,
        "lambda_rc_consist": args.lambda_rc_consist,
        "lambda_boundary": args.lambda_boundary,
        "lambda_moe_aux": args.lambda_moe_aux,
        "lambda_fshm_router_aux": args.lambda_fshm_router_aux,
    }
    for name, value in non_negative.items():
        if value < 0:
            raise ValueError(f"--{name} must be non-negative")

    probabilities = {
        "use_reverse_prob": args.use_reverse_prob,
        "frameshift_prob": args.frameshift_prob,
        "rc_loss_prob": args.rc_loss_prob,
        "mask_fraction": args.mask_fraction,
        "mask_start": args.mask_start,
        "mask_end": args.mask_end,
        "alpha_kd": args.alpha_kd,
    }
    for name, value in probabilities.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be between 0 and 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0, 1)")
    if not 0.0 < args.train_split_ratio < 1.0:
        raise ValueError("--train_split_ratio must be between 0 and 1")
    if args.resume_lr_scale is not None and args.resume_lr_scale <= 0:
        raise ValueError("--resume_lr_scale must be positive")


def _build_model_config(args: argparse.Namespace, tokenizer) -> DnaConfig:
    return DnaConfig(
        tokenizer_type=tokenizer.tokenizer_type,
        tokenizer_vocab_sha256=tokenizer.vocab_sha256,
        kmer_vocab_sizes=[tokenizer.kmer_vocab_size],
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        unk_token_id=tokenizer.unk_token_id,
        use_kmer_mix=args.use_kmer_mix,
        kmer_size=tokenizer.kmer_size,
        kmer_vocab_size=tokenizer.kmer_vocab_size,
        hidden_size=args.hidden_size,
        high_level_layers=args.high_level_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_input_bases=args.max_length,
        dropout=args.dropout,
        use_conv_ffn=args.use_conv_ffn,
        conv_kernel_size=args.conv_kernel_size,
        drop_path_rate=args.drop_path_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        low_level_stride=args.low_level_stride,
        low_level_kernel=args.low_level_kernel,
        fshm_layers=_parse_fshm_layers(args.fshm_layers),
        backbone=args.backbone,
        dual_helix_bridge_interval=args.bridge_interval,
        disable_bridge=args.disable_bridge,
        use_conv_stem=args.use_conv_stem,
        use_dual_strand=args.use_dual_strand,
        cross_strand_gate_interval=args.cross_strand_gate_interval,
        rc_equivariant_tying=args.rc_equivariant_tying,
        use_moe=args.use_moe,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        expert_drop_prob=args.expert_drop_prob,
        microscope_window_size=args.microscope_window_size,
        ssm_d_state=args.ssm_d_state,
        ssm_d_conv=args.ssm_d_conv,
        ssm_expand=args.ssm_expand,
        fshm_slow_stride=args.fshm_slow_stride,
        use_mamba2=args.use_mamba2,
        mamba_version=args.mamba_version,
        mamba2_head_dim=args.mamba2_head_dim,
        mamba2_ngroups=args.mamba2_ngroups,
        moe_beta_z=args.moe_beta_z,
        expert_drop_warmup_steps=args.expert_drop_warmup_steps,
        frameshift_prob=args.frameshift_prob,
    )


def main():
    # Setup and initialization
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    _validate_training_args(args)
    if args.auto_resume:
        if args.resume_from:
            raise SystemExit("--auto_resume and --resume_from cannot be used together")
        latest_checkpoint = Path(args.save_dir) / "checkpoint_latest.pt"
        if latest_checkpoint.is_file():
            args.resume_from = str(latest_checkpoint)
            print(f"[Resume] Auto-resuming from {latest_checkpoint}")
        else:
            print(f"[Resume] No latest checkpoint found in {args.save_dir}; starting a new run")
    if args.mamba_version is None:
        args.mamba_version = "mamba2" if args.use_mamba2 else "mamba1"
    else:
        args.use_mamba2 = args.mamba_version == "mamba2"
    k_mer_sizes = _parse_k_mer_sizes(args.k_mer_sizes)
    seed_all(args.seed)

    if args.print_params_only:
        tok = build_tokenizer(
            args.tokenizer_type,
            vocab_path=args.tokenizer_path,
            k_mer_sizes=k_mer_sizes,
            max_length=args.max_length,
        )
        cfg = _build_model_config(args, tok)
        model = DnaModel(cfg)
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}")
        sys.exit(0)

    if args.cpu:
        raise SystemExit("--cpu is not supported for this training path; CUDA is required.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training; no CUDA device is available.")

    # Long contexts require Flash or memory-efficient scaled dot-product attention.
    print("[Memory] Disabling 'math' SDPA kernel to force Flash/MemEfficient attention.", flush=True)
    torch.backends.cuda.enable_math_sdp(False)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16" if args.amp else "no", cpu=False
    )

    writer = None
    if not args.no_wandb and accelerator.is_main_process:
        wandb_kwargs = {"project": "dna-language-model", "config": vars(args)}
        if args.wandb_name:
            wandb_kwargs["name"] = args.wandb_name
        if args.wandb_group:
            wandb_kwargs["group"] = args.wandb_group
        if args.wandb_job_type:
            wandb_kwargs["job_type"] = args.wandb_job_type
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        if tags:
            wandb_kwargs["tags"] = tags
        wandb.init(**wandb_kwargs)
        _configure_wandb_metrics()

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        # Preserve the original CLI arguments alongside the resolved config.
        with open(os.path.join(args.save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        writer = SummaryWriter(log_dir=args.save_dir)
        print(f"[OK] TensorBoard logging to {args.save_dir}", flush=True)

    # Device is handled by Accelerator
    device = accelerator.device
    print(f"[OK] Using device: {device} (Accelerator)", flush=True)
    maybe_enable_tf32(args, device)

    # Tokenizer
    tok = build_tokenizer(
        args.tokenizer_type,
        vocab_path=args.tokenizer_path,
        k_mer_sizes=k_mer_sizes,
        max_length=args.max_length,
    )
    print(f"[OK] Tokenizer initialized ({args.tokenizer_type}). Vocab size: {tok.vocab_size}", flush=True)

    # Data loading
    files = list_paths(args.fasta)
    if not files:
        print(f"[ERROR] No files found matching: {args.fasta}")
        sys.exit(1)
    print(f"[OK] Found {len(files)} FASTA files.", flush=True)

    stride = args.stride if args.stride > 0 else args.max_length

    cap_by_file = None
    if args.cap_by_file:
        raw_caps = _load_json(args.cap_by_file)
        cap_by_file = _normalize_map_for_files(raw_caps, files, default=None, cast_fn=int, name="cap")
        print(f"[OK] Loaded per-file caps for {len(cap_by_file)} files.")

    if args.weights_file:
        raw_weights = _load_json(args.weights_file)
        weights = _normalize_map_for_files(raw_weights, files, default=1.0, cast_fn=float, name="weights")
        print(f"[OK] Loaded sampling weights for {len(weights)} files.")

        train_loader, eval_loader = build_loaders_weighted(
            files=files,
            weights=weights,
            tokenizer=tok,
            max_length=args.max_length,
            stride=stride,
            train_split_ratio=args.train_split_ratio,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            use_reverse_prob=args.use_reverse_prob,
            provide_rc=(args.lambda_rc_consist > 0),
            frameshift_prob=args.frameshift_prob,
            max_chunks_per_file=args.max_chunks_per_file if args.max_chunks_per_file > 0 else None,
            cap_by_file=cap_by_file,
            steps_per_epoch=(args.steps_per_epoch * max(1, args.grad_accum) if args.steps_per_epoch > 0 else None),
            pin_memory=args.pin_memory,
            gc_metadata_file=args.stratify_gc,
            stratify_gc=bool(args.stratify_gc),
            split_manifest=args.split_manifest,
            seed=args.seed,
            return_offsets=args.mask_coordinate_system == "base",
        )
    else:
        train_loader, eval_loader = build_loaders_balanced(
            files=files,
            tokenizer=tok,
            max_length=args.max_length,
            stride=stride,
            train_split_ratio=args.train_split_ratio,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            use_reverse_prob=args.use_reverse_prob,
            provide_rc=(args.lambda_rc_consist > 0),
            frameshift_prob=args.frameshift_prob,
            max_chunks_per_file=args.max_chunks_per_file if args.max_chunks_per_file > 0 else None,
            cap_by_file=cap_by_file,
            pin_memory=args.pin_memory,
            use_smart_batching=args.use_smart_batching,
            split_manifest=args.split_manifest,
            seed=args.seed,
            return_offsets=args.mask_coordinate_system == "base",
        )

    print(f"[OK] Data loaders ready. Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}", flush=True)

    # Model
    cfg = _build_model_config(args, tok)

    model = DnaModel(cfg)
    # model.to(device) # Handled by Accelerator
    if args.use_moe and args.expert_drop_warmup_steps > 0:
        for m in model.modules():
            if m.__class__.__name__ == "MoEFFN":
                m.expert_drop_prob = 0.0
    model_config = vars(cfg).copy()
    run_config = vars(args).copy()
    run_config.update(model_config)
    if accelerator.is_main_process:
        with open(os.path.join(args.save_dir, "config.resolved.json"), "w") as f:
            json.dump(run_config, f, indent=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created. Total params: {total_params / 1e6:.2f}M", flush=True)

    if args.compile:
        print(f"[INFO] Compiling model with torch.compile(mode={args.compile_mode})...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"[WARNING] torch.compile failed: {e}")
            print("[WARNING] Proceeding without compilation (eager mode).")

    # The aligned k-mer side channel is derived directly from sequence offsets,
    # so neither primary-token mode needs an RC lookup permutation.
    print(f"[OK] {tok.tokenizer_type} tokenization: no k-mer RC lookup tables required")

    # Knowledge-distillation teacher
    teacher_model = None
    if args.teacher_model:
        if not os.path.exists(args.teacher_model):
            print(f"[ERROR] KD: Teacher checkpoint not found: {args.teacher_model}")
            sys.exit(1)

        print(f"\n[KD] Initializing Teacher Model from {args.teacher_model}...")
        try:
            ckpt = torch.load(args.teacher_model, map_location=device, weights_only=True)
            t_cfg_dict = ckpt.get("model_config") or ckpt.get("config", {})
            if not t_cfg_dict:
                print("[WARNING] KD: Teacher checkpoint has no config dict. Falling back to Student config.")
                t_cfg = cfg
            else:
                t_cfg = config_from_checkpoint(t_cfg_dict, tok)

            teacher_model = DnaModel(t_cfg)

            t_state = normalize_state_dict_keys(ckpt["model"])

            load_state_dict_with_report(
                teacher_model,
                t_state,
                checkpoint_name=f"teacher checkpoint {args.teacher_model}",
                strict=True,
            )

            # Freeze teacher
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

            teacher_model.to(device)
            print(
                f"[OK] KD: Teacher Model loaded successfully. Using alpha_kd={args.alpha_kd}, temp={args.temperature}"
            )
        except Exception as e:
            print(f"[ERROR] KD: Failed to load teacher model: {e}")
            sys.exit(1)

    # Register the optional fusion-gate telemetry hook.
    if hasattr(model, "fusion_gate"):
        print("[Telemetry] Registering fusion gate hook")
        model.fusion_gate.register_forward_hook(make_gate_hook("fusion"))

    # Optimizer and scheduler
    use_fused = device.type == "cuda"

    boundary_head = None
    if getattr(args, "lambda_boundary", 0.0) > 0.0:
        bhead_dim = model.config.hidden_size
        boundary_head = SpanBoundaryHead(bhead_dim).to(device)
        boundary_head_params = sum(p.numel() for p in boundary_head.parameters())
        print(f"[OK] SpanBoundaryHead initialized. Input dim: {bhead_dim}, Params: {boundary_head_params:,}")

    all_param_groups = parameter_groups(model, args.weight_decay)
    if boundary_head is not None:
        all_param_groups.append(
            {
                "params": list(boundary_head.parameters()),
                "weight_decay": 0.0,
                "lr": args.lr * 2.0,
            }
        )

    optimizer = torch.optim.AdamW(all_param_groups, lr=args.lr, fused=use_fused)

    loader_optimizer_steps = len(train_loader) // max(1, args.grad_accum)
    if args.steps_per_epoch > 0:
        steps_per_epoch = int(args.steps_per_epoch)
        print(
            f"[OK] Forced steps/epoch: {steps_per_epoch} optimizer steps "
            f"({steps_per_epoch * max(1, args.grad_accum)} raw batches)",
            flush=True,
        )
    else:
        steps_per_epoch = loader_optimizer_steps
    num_train_steps = args.epochs * steps_per_epoch

    scheduler = get_scheduler(
        optimizer,
        args.warmup_steps,
        num_train_steps,
        args.lr_schedule,
    )

    print(
        f"\n[OK] Training: {args.epochs} epochs x {steps_per_epoch} steps/epoch = {num_train_steps} total steps",
        flush=True,
    )
    print(f"[OK] Warmup: {args.warmup_steps} steps", flush=True)

    # AMP
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp)) # Handled by Accelerator
    if args.amp:
        print(f"[OK] AMP enabled with dtype={autocast_dtype}")

    if boundary_head is not None:
        model, boundary_head, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, boundary_head, optimizer, train_loader, eval_loader, scheduler
        )
    else:
        model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, scheduler
        )

    ema = EMA(accelerator.unwrap_model(model), decay=0.9999) if args.use_ema else None
    if ema:
        print("[OK] Exponential Moving Average enabled (decay=0.9999)")

    # Class weights and curriculum
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights_from_loader(
            train_loader,
            device,
            num_classes=tok.vocab_size,
            sample_batches=200,
            mask_fraction=0.20,
            mean_span_length=3.0,
        )

    curriculum = None
    if args.use_curriculum:
        curriculum = MaskingCurriculum(
            total_steps=num_train_steps,
            warmup_frac=0.4,
            mask_start=args.mask_start,
            mask_end=args.mask_end,
            mean_span_length=args.mean_span_length,
        )

    # Resume from checkpoint
    best_eval_loss = float("inf")
    global_step = 0
    start_epoch = 0
    last_eval_loss = None
    last_eval_acc = None
    last_eval_ece = None
    last_eval_metrics = None
    last_eval_step = None
    current_grad_norm = 0.0
    old_config = {}

    def _append_eval_history(run_dir, rec: dict):
        try:
            hist = os.path.join(run_dir, "eval_history.jsonl")
            with open(hist, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _build_checkpoint_metrics(
        *,
        train_loss,
        lr_curr,
        grad_norm_val,
        tokens_per_sec,
        mask_fraction,
        span_length,
        mask_bool,
        input_ids,
        aux_losses,
        eval_metrics,
        eval_step,
        eval_ran_now,
        domain_counts_snapshot,
    ):
        masked_tokens = int(mask_bool.sum().item()) if mask_bool is not None else None
        batch_tokens = int(input_ids.numel()) if input_ids is not None else None
        masked_bases = None
        batch_bases = None
        if input_ids is not None:
            base_lengths = torch.tensor(
                [tok.token_base_length(token_id) for token_id in range(tok.vocab_size)],
                dtype=torch.long,
                device=input_ids.device,
            )
            batch_bases = int(base_lengths[input_ids].sum().item())
            if mask_bool is not None:
                masked_bases = int(base_lengths[input_ids[mask_bool]].sum().item())
        train_metrics = {
            "loss": _metric_value(train_loss),
            "lr": _metric_value(lr_curr),
            "lr_groups": _metric_value([pg.get("lr") for pg in optimizer.param_groups]),
            "grad_norm_unclipped": _metric_value(grad_norm_val),
            "grad_clip_ratio": _metric_value(
                float(grad_norm_val) / float(args.grad_clip) if args.grad_clip > 0 else 0.0
            ),
            "tokens_per_sec": _metric_value(tokens_per_sec),
            "mask_fraction_requested": _metric_value(mask_fraction),
            "mean_span_length_requested": _metric_value(span_length),
            "masked_tokens": _metric_value(masked_tokens),
            "batch_tokens": _metric_value(batch_tokens),
            "masked_fraction_actual": _metric_value(
                float(masked_tokens) / float(batch_tokens) if masked_tokens is not None and batch_tokens else None
            ),
            "masked_bases": _metric_value(masked_bases),
            "batch_bases": _metric_value(batch_bases),
            "masked_base_fraction_actual": _metric_value(
                float(masked_bases) / float(batch_bases) if masked_bases is not None and batch_bases else None
            ),
            "mask_coordinate_system": args.mask_coordinate_system,
            "mask_replacement_strategy": args.mask_replacement_strategy,
            "eval_replacement_strategy": "mask",
        }
        eval_clean = _metric_value(eval_metrics or {})
        metrics = {
            "schema_version": 1,
            "checkpoint": {
                "global_step": int(global_step),
                "epoch": int(epoch),
                "epoch_step": int(epoch_step),
                "steps_per_epoch": int(steps_per_epoch),
                "timestamp": time.time(),
            },
            "train": train_metrics,
            "aux": _metric_value(aux_losses),
            "eval": eval_clean,
            "eval_from_step": _metric_value(eval_step),
            "eval_is_fresh": bool(eval_step == global_step) if eval_step is not None else False,
            "eval_ran_during_checkpoint": bool(eval_ran_now),
            "best": {"eval_loss": _metric_value(best_eval_loss)},
            "runtime": {
                "amp_enabled": bool(args.amp),
                "tf32_enabled": bool(args.tf32),
                "compiled": bool(args.compile),
                "device": str(device),
                "cuda_memory": _cuda_memory_metrics(device),
            },
            "data": {
                "domain_fraction_since_last_log": _metric_value(domain_counts_snapshot),
            },
        }

        # Flat aliases keep existing checkpoint summary/backfill tools useful.
        if isinstance(eval_clean, dict) and eval_clean:
            metrics["eval_loss"] = _metric_value(eval_clean.get("loss"))
            metrics["eval_acc"] = _metric_value(eval_clean.get("acc", eval_clean.get("accuracy")))
            metrics["ece"] = _metric_value(eval_clean.get("ece"))
            metrics["from_step"] = _metric_value(eval_step)
        metrics["best_loss_so_far"] = _metric_value(best_eval_loss)
        return _metric_value(metrics)

    def load_checkpoint(path: str, model, optimizer, scheduler, ema, device):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"[Resume] Loading checkpoint from {path}")
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception as exc:
            print(f"[Resume] Error loading checkpoint: {exc}")
            raise

        state_dict = normalize_state_dict_keys(ckpt["model"])

        target_model = accelerator.unwrap_model(model)
        model_keys = list(target_model.state_dict().keys())
        if any(k.startswith("_orig_mod.") for k in model_keys):
            print("[Resume] Detected compiled model. Adding '_orig_mod.' prefix to checkpoint keys...")
            state_dict = {f"_orig_mod.{key}": value for key, value in state_dict.items()}

        load_state_dict_with_report(
            target_model,
            state_dict,
            checkpoint_name=f"resume checkpoint {path}",
            strict=True,
        )
        if optimizer and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            if args.reset_scheduler:
                print("[Resume] reset_scheduler=True. Skipping optimizer load.")
            else:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                except Exception as e:
                    print(f"[Resume] Warning: Could not load optimizer state: {e}")

        if scheduler and ckpt.get("scheduler") is not None:
            if args.reset_scheduler:
                print("[Resume] reset_scheduler=True. Skipping scheduler load. Starting fresh LR curve.")
            else:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as e:
                    print(f"[Resume] Warning: Could not load scheduler state: {e}")
        if ema and ckpt.get("ema_shadow") is not None:
            ema.load_state_dict(ckpt["ema_shadow"])
        se = int(ckpt.get("epoch", 0))
        gs = int(ckpt.get("global_step", 0))
        bl = float(ckpt.get("best_loss", float("inf")))
        old_cfg = ckpt.get("config", {})
        print(f"[Resume] Loaded from epoch {se}, step {gs}")
        return se, gs, bl, old_cfg

    if args.resume_from:
        resume_arg = str(args.resume_from).strip()
        resume_path = resume_arg

        # Resolve a run name relative to the conventional runs directory.
        if not os.path.exists(resume_path):
            candidate = os.path.join("runs", resume_path)
            if os.path.exists(candidate):
                resume_path = candidate

        # A run directory resumes from its latest checkpoint.
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoint_latest.pt")

        if _HAS_CKPT_MANAGER and (resume_arg in ("best", "latest") or resume_arg.startswith("epoch:")):
            mgr = CheckpointManager(Path(args.save_dir))
            if resume_arg == "best":
                p = mgr.find_best_checkpoint()
            elif resume_arg == "latest":
                p = mgr.find_latest_checkpoint()
            else:
                ep = int(resume_arg.split(":")[1])
                p = mgr.find_checkpoint_by_epoch(ep)

            if p:
                resume_path = str(p)
                print(f"[Resume] Resolved '{resume_arg}' -> {resume_path}")
            else:
                resume_path = os.path.join(args.save_dir, f"checkpoint_{resume_arg}.pt")

        try:
            start_epoch, global_step, best_eval_loss, old_config = load_checkpoint(
                resume_path, model, optimizer, scheduler, ema, device
            )

            if boundary_head is not None:
                ckpt = torch.load(resume_path, map_location=device, weights_only=True)
                if ckpt.get("boundary_head") is not None:
                    try:
                        accelerator.unwrap_model(boundary_head).load_state_dict(ckpt["boundary_head"])
                        print("[Resume] Restored boundary_head weights.")
                    except Exception as e:
                        print(f"[Resume] Warning: Could not restore boundary_head: {e}")

            if steps_per_epoch > 0:
                derived_epoch = int(global_step) // int(steps_per_epoch)
                if derived_epoch > start_epoch:
                    start_epoch = derived_epoch
                resume_epoch_offset = int(global_step) % int(steps_per_epoch)
            else:
                resume_epoch_offset = 0

            if args.resume_lr_scale:
                scale_learning_rate(optimizer, scheduler, args.resume_lr_scale)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            sys.exit(1)
    else:
        resume_epoch_offset = 0

    # Do not let tokenizer/model construction or checkpoint loading inflate the
    # training-memory comparison.  Checkpoint metadata and W&B both receive
    # allocator peaks measured from this point onward.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    first_epoch_resume = args.resume_from is not None
    domain_counts = defaultdict(int)

    # Eval Schedule
    eval_schedule = parse_eval_schedule(args.eval_schedule) if args.eval_schedule else []
    current_eval_interval = args.eval_interval

    for epoch in range(start_epoch, args.epochs):
        model.train()

        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

        epoch_step = 0
        if first_epoch_resume:
            epoch_step = resume_epoch_offset

        # Keep masking reproducible while allowing independent seeded runs.
        g = torch.Generator(device=device).manual_seed(args.seed + 1234 + epoch)

        print(f"\n==== Epoch {epoch + 1}/{args.epochs} ====", flush=True)

        optimizer.zero_grad(set_to_none=True)
        if HAS_TQDM:
            pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}", unit="step", initial=epoch_step)
        else:
            pbar = None

        t_last_log = time.time()
        tokens_since_log = 0

        batches_to_skip = 0
        if first_epoch_resume and resume_epoch_offset > 0:
            # Use the old grad_accum to calculate how many data batches were consumed
            old_grad_accum = old_config.get("grad_accum", args.grad_accum)
            batches_to_skip = resume_epoch_offset * max(1, old_grad_accum)

            if args.fast_resume:
                if hasattr(train_loader, "fast_forward"):
                    train_loader.fast_forward(batches_to_skip)
                    if HAS_TQDM:
                        pbar.set_description(f"Epoch {epoch + 1} (Fast-forwarded {batches_to_skip} steps)")
                    batches_to_skip = 0
                elif hasattr(getattr(train_loader, "batch_sampler", None), "set_skip"):
                    train_loader.batch_sampler.set_skip(batches_to_skip)
                    batches_to_skip = 0
                elif hasattr(train_loader, "loaders"):
                    samplers = [getattr(loader, "batch_sampler", None) for loader in train_loader.loaders]
                    if samplers and all(hasattr(sampler, "set_skip") for sampler in samplers):
                        base_skip, remainder = divmod(batches_to_skip, len(samplers))
                        for index, sampler in enumerate(samplers):
                            sampler.set_skip(base_skip + (index < remainder))
                        batches_to_skip = 0
                    else:
                        print("[Fast Resume] Samplers cannot seek; replaying skipped batches.")
                else:
                    print("[Fast Resume] Loader cannot seek; replaying skipped batches.")
            else:
                if HAS_TQDM:
                    pbar.set_description(f"Epoch {epoch + 1} (Skipping {batches_to_skip} steps...)")

        first_epoch_resume = False

        for ib, batch in enumerate(train_loader, start=1):
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            if isinstance(batch, dict) and "domain_name" in batch:
                dnames = batch["domain_name"]
                if isinstance(dnames, list):
                    for dn in dnames:
                        domain_counts[dn] += 1
                else:
                    domain_counts[dnames] += 1

            # Curriculum (Masking)
            frac_mask, span_len = args.mask_fraction, args.mean_span_length
            if curriculum:
                frac_mask, span_len = curriculum.get_params(global_step)

            # RC consistent scheduling
            current_lambda_rc = args.lambda_rc_consist
            if args.warmup_steps > 0 and global_step < args.warmup_steps:
                rc_progress = float(global_step) / float(args.warmup_steps)
                current_lambda_rc = args.lambda_rc_consist * rc_progress

            input_ids_gpu = batch["input_ids"].to(device, non_blocking=True)
            tokens_since_log += input_ids_gpu.numel()

            # Forward
            (
                loss,
                logits,
                labels,
                mask_bool,
                boundary_loss,
                moe_aux_loss,
                fshm_router_aux_loss,
                router_entropy,
            ) = step_mask_and_forward(
                model,
                batch,
                device,
                mask_fraction=frac_mask,
                mean_span_length=span_len,
                accelerator=accelerator,  # Pass accelerator
                class_weights=class_weights,
                tokenizer=tok,
                amp=args.amp,
                autocast_dtype=autocast_dtype,
                generator=g,
                loss_lambda_boundary=args.lambda_boundary,
                boundary_head=boundary_head,
                label_smoothing=args.label_smoothing,
                teacher_model=teacher_model,
                kd_temp=args.temperature,
                alpha_kd=args.alpha_kd,
                mask_coordinate_system=args.mask_coordinate_system,
                mask_replacement_strategy=args.mask_replacement_strategy,
            )

            # Separate aux losses to avoid implicit coupling between:
            # (1) Telescope MoE FFN aux and (2) FSHM biological-router aux.
            if moe_aux_loss is not None:
                loss = loss + args.lambda_moe_aux * moe_aux_loss
            if fshm_router_aux_loss is not None:
                loss = loss + args.lambda_fshm_router_aux * fshm_router_aux_loss

            rc_loss = None
            if (
                current_lambda_rc > 0.0
                and "input_ids_rc" in batch
                and torch.rand((), device=device).item() < args.rc_loss_prob
            ):
                rc_attention_mask = batch["attention_mask_rc"].to(device, non_blocking=True)
                forward_attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with accelerator.autocast():
                    with torch.no_grad():
                        rc_kmer_ids = (
                            batch["kmer_ids_rc"].to(device, non_blocking=True) if "kmer_ids_rc" in batch else None
                        )
                        rc_out = model(
                            input_ids=batch["input_ids_rc"].to(device, non_blocking=True),
                            attention_mask=rc_attention_mask,
                            kmer_ids=rc_kmer_ids,
                        )
                        embed_rc = (
                            masked_mean_pool(
                                rc_out["last_hidden_state"],
                                rc_attention_mask,
                                # A detached view can still alias a CUDA-Graph output.
                                # The next model invocation in this same optimizer step
                                # would then overwrite it before the RC loss consumes it.
                            )
                            .detach()
                            .clone()
                        )

                    fwd_clean = model(
                        input_ids=batch["input_ids"].to(device, non_blocking=True),
                        attention_mask=forward_attention_mask,
                        kmer_ids=(batch["kmer_ids"].to(device, non_blocking=True) if "kmer_ids" in batch else None),
                    )
                    embed_fwd = masked_mean_pool(
                        fwd_clean["last_hidden_state"],
                        forward_attention_mask,
                    )

                rc_loss = F.mse_loss(embed_fwd, embed_rc)
                loss = loss + current_lambda_rc * rc_loss

            loss = loss / args.grad_accum

            # Backward
            # scaler.scale(loss).backward()
            accelerator.backward(loss)

            if ib % args.grad_accum == 0:
                # scaler.unscale_(optimizer)
                if accelerator.sync_gradients:
                    gn = accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    current_grad_norm = gn.item() if gn is not None else 0.0
                else:
                    current_grad_norm = 0.0

                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if ema:
                    ema.update(accelerator.unwrap_model(model))

                global_step += 1
                epoch_step += 1

                # Expert-dropout warmup schedule
                if args.use_moe and args.expert_drop_warmup_steps > 0:
                    # Linearly scale from 0.0 to target_prob over first N steps
                    current_drop_prob = args.expert_drop_prob * min(
                        1.0, float(global_step) / args.expert_drop_warmup_steps
                    )

                    # Update all MoE layers directly (unwrap for accelerator)
                    if global_step % 10 == 0 or global_step < 100:
                        unwrapped = accelerator.unwrap_model(model)
                        for m in unwrapped.modules():
                            if m.__class__.__name__ == "MoEFFN":
                                m.expert_drop_prob = current_drop_prob
                if HAS_TQDM:
                    pbar.update(1)

                # Logging
                if global_step % args.log_interval == 0:
                    lr_curr = optimizer.param_groups[0]["lr"]

                    t_now = time.time()
                    dt = t_now - t_last_log
                    tps = tokens_since_log / dt if dt > 0 else 0.0

                    grad_norm_val = current_grad_norm
                    clip_ratio = grad_norm_val / args.grad_clip if args.grad_clip > 0 else 0.0

                    log_dict = {
                        "train/loss": loss.item() * args.grad_accum,
                        "train/lr": lr_curr,
                        "train/grad_norm_unclipped": grad_norm_val,
                        "train/grad_clip_ratio": clip_ratio,
                        "train/tokens_per_sec": tps,
                        "train/mask_fraction": frac_mask,
                        "train/span_length": span_len,
                        "global_step": global_step,
                        "epoch": epoch + 1,
                    }

                    if boundary_loss is not None:
                        log_dict["train/boundary_loss"] = boundary_loss.item()

                    if moe_aux_loss is not None:
                        log_dict["train/moe_aux_loss_raw"] = moe_aux_loss.item()
                        log_dict["train/moe_aux_loss_weighted"] = (args.lambda_moe_aux * moe_aux_loss).item()
                    if fshm_router_aux_loss is not None:
                        log_dict["train/fshm_router_aux_loss_raw"] = fshm_router_aux_loss.item()
                        log_dict["train/fshm_router_aux_loss_weighted"] = (
                            args.lambda_fshm_router_aux * fshm_router_aux_loss
                        ).item()

                    if rc_loss is not None:
                        log_dict["train/rc_loss"] = rc_loss.item()
                    # Reset throughput counters
                    t_last_log = time.time()
                    tokens_since_log = 0

                    total_domains = sum(domain_counts.values())
                    if total_domains > 0:
                        for dname, count in domain_counts.items():
                            log_dict[f"train/domain_{dname}"] = count / total_domains
                        domain_counts.clear()

                    log_dict.update(_drain_gate_stats())
                    log_dict.update(_wandb_cuda_memory_metrics(device))

                    if accelerator.is_main_process and not args.no_wandb:
                        wandb.log(log_dict)

                    if HAS_TQDM:
                        pbar.set_postfix(
                            loss=f"{loss.item() * args.grad_accum:.4f}", lr=f"{lr_curr:.2e}", tps=f"{tps:.0f}"
                        )
                    else:
                        print(
                            f"Step {global_step} | Loss: {loss.item() * args.grad_accum:.4f} | LR: {lr_curr:.2e} | TPS: {tps:.0f}"
                        )

                # Evaluation and checkpointing. Checkpoint steps force an eval first
                # so every saved checkpoint carries fresh validation metrics.
                checkpoint_step = args.save_every > 0 and global_step % args.save_every == 0
                checkpoint_due = accelerator.is_main_process and checkpoint_step

                # Determine current eval interval
                if eval_schedule:
                    # Find the last schedule entry where start_step <= global_step
                    # Default to args.eval_interval if before first schedule step
                    new_interval = args.eval_interval
                    for start_step, interval in eval_schedule:
                        if global_step >= start_step:
                            new_interval = interval
                        else:
                            break
                    current_eval_interval = new_interval

                eval_due = (global_step % current_eval_interval == 0) or checkpoint_step
                eval_ran_now = False
                if eval_due:
                    print(f"\n[Eval] Running evaluation at step {global_step}...")

                    if ema:
                        ema.apply_shadow(accelerator.unwrap_model(model))

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    # Disable RC loss during eval to save memory
                    eval_metrics = evaluate(
                        model,
                        eval_loader,
                        device,
                        args.amp,
                        epoch,
                        tok,
                        kmer_rc_tables=None,  # No RC tables during eval.
                        lambda_rc=0.0,
                        max_batches=args.eval_max_batches,
                        strict_masked_targets=True,
                        mask_coordinate_system=args.mask_coordinate_system,
                        mask_seed=args.seed,
                    )

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    val_loss = eval_metrics["loss"]
                    val_ppl = eval_metrics["ppl"]
                    val_acc = eval_metrics["acc"]

                    accuracy_label = "Base Acc" if tok.tokenizer_type == "base" else "BPE Token Acc"
                    print(
                        f"[Eval] Token Loss: {val_loss:.4f} | Token PPL: {val_ppl:.2f} | "
                        f"{accuracy_label}: {val_acc:.2%}"
                    )
                    if tok.tokenizer_type == "bpe" and "bits_per_base" in eval_metrics:
                        print(
                            f"[Eval] Bits/Base: {eval_metrics['bits_per_base']:.4f} | "
                            f"Base-normalized PPL: {eval_metrics['base_perplexity']:.2f}"
                        )

                    if tok.tokenizer_type == "base":
                        for base in ["A", "C", "G", "T"]:
                            accuracy = eval_metrics.get(f"accuracy_{base}")
                            support = eval_metrics[f"support_{base}"]
                            accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A"
                            print(f"[Eval] {base} Acc: {accuracy_text} (n={support})", end=" | ")
                        print("")  # Newline

                    last_eval_loss = val_loss
                    last_eval_acc = val_acc
                    last_eval_ece = eval_metrics.get("ece")
                    last_eval_metrics = eval_metrics
                    last_eval_step = global_step
                    eval_ran_now = True

                    rec = build_eval_history_record(
                        step=global_step,
                        epoch=epoch,
                        eval_metrics=eval_metrics,
                    )
                    if accelerator.is_main_process:
                        _append_eval_history(args.save_dir, rec)
                        eval_log_payload = build_eval_log_payload(rec, eval_metrics)
                        eval_log_payload.update(_wandb_cuda_memory_metrics(device))
                        log_eval_to_tensorboard(writer, eval_log_payload, global_step)
                        if not args.no_wandb:
                            wandb.log(eval_log_payload)

                    if val_loss < best_eval_loss:
                        best_eval_loss = val_loss
                        if accelerator.is_main_process:
                            print(f"[Eval] New best loss: {best_eval_loss:.4f}")
                            ckpt_path = os.path.join(args.save_dir, "checkpoint_best.pt")
                            aux_losses = {
                                "boundary_loss": boundary_loss,
                                "moe_aux_loss_raw": moe_aux_loss,
                                "moe_aux_loss_weighted": (
                                    args.lambda_moe_aux * moe_aux_loss if moe_aux_loss is not None else None
                                ),
                                "fshm_router_aux_loss_raw": fshm_router_aux_loss,
                                "fshm_router_aux_loss_weighted": (
                                    args.lambda_fshm_router_aux * fshm_router_aux_loss
                                    if fshm_router_aux_loss is not None
                                    else None
                                ),
                                "rc_loss": rc_loss,
                            }
                            best_metrics = _build_checkpoint_metrics(
                                train_loss=loss.item() * args.grad_accum,
                                lr_curr=optimizer.param_groups[0]["lr"],
                                grad_norm_val=current_grad_norm,
                                tokens_per_sec=(tokens_since_log / max(time.time() - t_last_log, 1e-9)),
                                mask_fraction=frac_mask,
                                span_length=span_len,
                                mask_bool=mask_bool,
                                input_ids=input_ids_gpu,
                                aux_losses=aux_losses,
                                eval_metrics=last_eval_metrics,
                                eval_step=last_eval_step,
                                eval_ran_now=True,
                                domain_counts_snapshot={},
                            )

                            if ema:
                                deploy_dict = {
                                    "model": ema.state_dict(),
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "best_loss": best_eval_loss,
                                    "config": run_config,
                                    "model_config": model_config,
                                    "eval_metrics": eval_metrics,
                                    "metrics": best_metrics,
                                }

                                if boundary_head is not None:
                                    deploy_dict["boundary_head"] = accelerator.unwrap_model(boundary_head).state_dict()

                                if _HAS_CKPT_MANAGER:
                                    try:
                                        mgr = CheckpointManager(Path(args.save_dir))
                                        mgr.save_checkpoint(deploy_dict, Path(ckpt_path), is_latest=False)
                                        print(f"[Checkpoint] Saved EMA-only best to {ckpt_path}")
                                    except Exception as e:
                                        print(f"[Checkpoint] Failed to save best: {e}")
                                        torch.save(deploy_dict, ckpt_path)
                                else:
                                    torch.save(deploy_dict, ckpt_path)
                                    print(f"[Checkpoint] Saved EMA-only best to {ckpt_path}")
                            else:
                                save_dict = {
                                    "model": accelerator.unwrap_model(model).state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "best_loss": best_eval_loss,
                                    "config": run_config,
                                    "model_config": model_config,
                                    "eval_metrics": eval_metrics,
                                    "metrics": best_metrics,
                                }

                                if boundary_head is not None:
                                    save_dict["boundary_head"] = accelerator.unwrap_model(boundary_head).state_dict()

                                if _HAS_CKPT_MANAGER:
                                    try:
                                        mgr = CheckpointManager(Path(args.save_dir))
                                        mgr.save_checkpoint(save_dict, Path(ckpt_path), is_latest=True)
                                    except Exception as e:
                                        print(f"[Checkpoint] Failed to save best: {e}")
                                        torch.save(save_dict, ckpt_path)
                                else:
                                    torch.save(save_dict, ckpt_path)

                    if ema:
                        ema.restore(accelerator.unwrap_model(model))

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    model.train()

                # Checkpointing
                if checkpoint_due:
                    ckpt_path = os.path.join(args.save_dir, f"checkpoint_step{global_step}.pt")
                    domain_total = sum(domain_counts.values())
                    domain_fraction = (
                        {dname: count / domain_total for dname, count in domain_counts.items()}
                        if domain_total > 0
                        else {}
                    )
                    aux_losses = {
                        "boundary_loss": boundary_loss,
                        "moe_aux_loss_raw": moe_aux_loss,
                        "moe_aux_loss_weighted": (
                            args.lambda_moe_aux * moe_aux_loss if moe_aux_loss is not None else None
                        ),
                        "fshm_router_aux_loss_raw": fshm_router_aux_loss,
                        "fshm_router_aux_loss_weighted": (
                            args.lambda_fshm_router_aux * fshm_router_aux_loss
                            if fshm_router_aux_loss is not None
                            else None
                        ),
                        "rc_loss": rc_loss,
                    }
                    checkpoint_metrics = _build_checkpoint_metrics(
                        train_loss=loss.item() * args.grad_accum,
                        lr_curr=optimizer.param_groups[0]["lr"],
                        grad_norm_val=current_grad_norm,
                        tokens_per_sec=tokens_since_log / max(time.time() - t_last_log, 1e-9),
                        mask_fraction=frac_mask,
                        span_length=span_len,
                        mask_bool=mask_bool,
                        input_ids=input_ids_gpu,
                        aux_losses=aux_losses,
                        eval_metrics=last_eval_metrics,
                        eval_step=last_eval_step,
                        eval_ran_now=eval_ran_now,
                        domain_counts_snapshot=domain_fraction,
                    )
                    save_dict = {
                        "model": accelerator.unwrap_model(model).state_dict(),  # Unwrap model
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_loss": best_eval_loss,
                        "config": run_config,
                        "model_config": model_config,
                        "eval_metrics": last_eval_metrics
                        or {
                            "loss": last_eval_loss,
                            "acc": last_eval_acc,
                            "ece": last_eval_ece,
                        },
                        "metrics": checkpoint_metrics,
                    }
                    if ema:
                        save_dict["ema_shadow"] = ema.state_dict()

                    if boundary_head is not None:
                        save_dict["boundary_head"] = accelerator.unwrap_model(boundary_head).state_dict()

                    if _HAS_CKPT_MANAGER:
                        try:
                            mgr = CheckpointManager(Path(args.save_dir))
                            mgr.save_checkpoint(save_dict, Path(ckpt_path), is_latest=False)
                            print(f"\n[Checkpoint] Saved archival to {ckpt_path}")

                            latest_path = os.path.join(args.save_dir, "checkpoint_latest.pt")
                            mgr.save_checkpoint(save_dict, Path(latest_path), is_latest=True)
                            print(f"[Checkpoint] Saved latest to {latest_path}")

                            mgr.clean_old_checkpoints(keep_n=5)
                        except Exception as e:
                            print(f"[Checkpoint] Save/Prune failed: {e}")
                    else:
                        torch.save(save_dict, ckpt_path)
                        print(f"\n[Checkpoint] Saved to {ckpt_path}")

                if epoch_step >= steps_per_epoch:
                    break

        if HAS_TQDM:
            pbar.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    if accelerator.is_main_process:
        write_run_summary(
            args.save_dir,
            global_step=global_step,
            total_params=total_params,
            final_eval=last_eval_metrics,
            device=device,
        )
    if accelerator.is_main_process and writer is not None:
        writer.close()
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one CUDA forward/backward step for each controlled-ablation backbone.

This intentionally uses synthetic token IDs and no FASTA files. It verifies
CUDA kernels, model shape, autocast, gradient checkpointing, and the memory
envelope selected by the RTX 5080 or A100 profile before a data-backed run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch  # noqa: E402
from dna_model.config import DnaConfig  # noqa: E402
from dna_model.model import DnaModel  # noqa: E402
from run_controlled_ablation import HARDWARE_PROFILES  # noqa: E402


def build_config(
    profile: dict[str, int], backbone: str, context_length: int, *, gradient_checkpointing: bool
) -> DnaConfig:
    return DnaConfig(
        backbone=backbone,
        tokenizer_type="base",
        vocab_size=10,
        max_input_bases=context_length,
        hidden_size=profile["hidden_size"],
        high_level_layers=profile["high_level_layers"],
        num_attention_heads=profile["num_attention_heads"],
        dropout=0.0,
        gradient_checkpointing=gradient_checkpointing,
        use_kmer_mix=False,
        mamba_version="mamba2",
        mamba2_head_dim=64,
        mamba2_ngroups=1,
        ssm_d_state=16,
        ssm_d_conv=4,
        ssm_expand=2,
        dual_helix_bridge_interval=4,
        use_moe=False,
    )


def smoke_one(
    profile: dict[str, int],
    backbone: str,
    context_length: int,
    seed: int,
    *,
    gradient_checkpointing: bool,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = DnaModel(
        build_config(
            profile,
            backbone,
            context_length,
            gradient_checkpointing=gradient_checkpointing,
        )
    ).to(device)
    model.train()
    input_ids = torch.randint(5, 10, (profile["batch_size"], context_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    def run_step(
        current_model: DnaModel,
        current_input_ids: torch.Tensor,
        current_attention_mask: torch.Tensor,
        current_labels: torch.Tensor,
    ) -> tuple[float, dict[str, object], torch.Tensor]:
        current_model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = current_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                labels=current_labels,
            )
            loss = output["loss"]
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"{backbone} produced a non-finite loss")
        loss.backward()
        torch.cuda.synchronize(device)
        return time.perf_counter() - start, output, loss

    warmup_seconds: list[float] = []
    output = None
    loss = None
    for _ in range(warmup_steps):
        elapsed, output, loss = run_step(model, input_ids, attention_mask, labels)
        warmup_seconds.append(elapsed)

    torch.cuda.reset_peak_memory_stats(device)
    timed_seconds: list[float] = []
    for _ in range(timed_steps):
        elapsed, output, loss = run_step(model, input_ids, attention_mask, labels)
        timed_seconds.append(elapsed)

    result = {
        "backbone": backbone,
        "context_length": context_length,
        "batch_size": profile["batch_size"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(loss.detach().cpu()),
        "gradient_checkpointing": gradient_checkpointing,
        "warmup_step_seconds": [round(value, 4) for value in warmup_seconds],
        "step_seconds": round(sum(timed_seconds) / len(timed_seconds), 4),
        "step_seconds_per_iteration": [round(value, 4) for value in timed_seconds],
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 2),
    }
    del output, loss, labels, attention_mask, input_ids, model
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA forward/backward smoke test for controlled ablation backbones")
    parser.add_argument("--profile", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=["dual_helix", "bimamba", "transformer"],
        default=["dual_helix", "bimamba", "transformer"],
    )
    parser.add_argument(
        "--context-length", type=int, default=None, help="Override the profile context for a quicker test"
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override activation checkpointing (default: use the selected ablation profile).",
    )
    parser.add_argument("--warmup-steps", type=int, default=1, help="Untimed steps before measurement (default: 1).")
    parser.add_argument("--timed-steps", type=int, default=3, help="Timed steps per backbone (default: 3).")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test")
    profile = HARDWARE_PROFILES[args.profile]
    gradient_checkpointing = (
        bool(profile.get("gradient_checkpointing", True))
        if args.gradient_checkpointing is None
        else bool(args.gradient_checkpointing)
    )
    context_length = int(args.context_length or profile["max_length"])
    if context_length <= 0:
        raise SystemExit("--context-length must be positive")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    if args.timed_steps <= 0:
        raise SystemExit("--timed-steps must be positive")

    results: list[dict[str, float | int | str]] = []
    failures: list[dict[str, str]] = []
    for backbone in args.backbones:
        try:
            results.append(
                smoke_one(
                    profile,
                    backbone,
                    context_length,
                    args.seed,
                    gradient_checkpointing=gradient_checkpointing,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                )
            )
        except Exception as exc:
            failures.append({"backbone": backbone, "error": str(exc)})
            if not args.continue_on_error:
                break

    report = {
        "profile": args.profile,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

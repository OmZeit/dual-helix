#!/usr/bin/env python3
"""Run reproducible Dual Helix, BiMamba, and Transformer ablations.

Run from any directory.  The launcher resolves the project from its own path,
requires a provenance-aware split for scientific runs, and stores commands,
preflight parameter counts, and final evaluation records in one manifest.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = PROJECT_DIR.parent
TRAIN_SCRIPT = PROJECT_DIR / "scripts" / "train.py"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "experiments" / "controlled_architecture_ablation_v2"

WANDB_PROJECT = "dna-language-model"
WANDB_GROUP_PREFIX = "controlled-architecture-ablation-v2"

# train.py intentionally uses underscore spellings for most arguments. Keep
# the few hyphenated compatibility flags explicit rather than mechanically
# converting every configuration key.
TRAINER_FLAG_OVERRIDES = {
    "tokenizer_type": "--tokenizer-type",
    "tokenizer_path": "--tokenizer-path",
    "split_manifest": "--split-manifest",
    "wandb_name": "--wandb-name",
    "wandb_group": "--wandb-group",
    "wandb_job_type": "--wandb-job-type",
    "wandb_tags": "--wandb-tags",
    "mask_coordinate_system": "--mask-coordinate-system",
    "mask_replacement_strategy": "--mask-replacement-strategy",
}


# Both profiles use the same effective global batch (16) so their results are
# comparable within a hardware class. The RTX 5080 profile uses a one-sample
# device batch with sixteen accumulation passes and activation checkpointing to
# leave a conservative memory margin on a 16 GiB device.
# Validate a preflight before increasing sequence length or per-device batch
# size further.
HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx5080_16gb": {
        "max_length": 1024,
        "stride": 512,
        "batch_size": 1,
        "grad_accum": 16,
        "hidden_size": 384,
        "high_level_layers": 6,
        "num_attention_heads": 6,
        "workers": 8,
        "gradient_checkpointing": True,
        "steps_per_epoch": 2000,
        "eval_max_batches": 64,
    },
    "a100_80gb": {
        "max_length": 4096,
        "stride": 2048,
        "batch_size": 4,
        "grad_accum": 4,
        "hidden_size": 512,
        "high_level_layers": 8,
        "num_attention_heads": 8,
        "workers": 8,
        "steps_per_epoch": 2000,
        "eval_max_batches": 128,
    },
}


COMMON_CONFIG: dict[str, Any] = {
    "tokenizer_type": "base",
    "tokenizer_path": "dna_model/bpe_vocab/tokenizer.json",
    "train_split_ratio": 0.99,
    "max_chunks_per_file": 0,
    "k_mer_sizes": "3",
    "use_kmer_mix": False,
    "dropout": 0.0,
    "drop_path_rate": 0.0,
    "conv_kernel_size": 7,
    "low_level_kernel": 6,
    "low_level_stride": 2,
    "cross_strand_gate_interval": 4,
    "mamba_version": "mamba2",
    "mamba2_head_dim": 64,
    "mamba2_ngroups": 1,
    "ssm_d_conv": 4,
    "ssm_d_state": 16,
    "ssm_expand": 2,
    "frameshift_prob": 0.05,
    "mask_start": 0.15,
    "mask_end": 0.35,
    "mask_fraction": 0.15,
    "mean_span_length": 3.0,
    "lambda_rc_consist": 0.20,
    "rc_loss_prob": 0.30,
    "use_reverse_prob": 0.50,
    "label_smoothing": 0.0,
    "mask_coordinate_system": "base",
    "mask_replacement_strategy": "mask",
    "use_curriculum": True,
    "use_dual_strand": False,
    "rc_equivariant_tying": False,
    "epochs": 1,
    "lr": 5e-4,
    "weight_decay": 0.05,
    "warmup_steps": 200,
    "lr_schedule": "cosine",
    "grad_clip": 1.0,
    "eval_interval": 250,
    "save_every": 500,
    "log_interval": 100,
    "amp": True,
    "tf32": True,
    "pin_memory": True,
    "gradient_checkpointing": True,
    "compile": False,
    "auto_resume": False,
    "use_moe": False,
}


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "01-dual-bridge4",
        "backbone": "dual_helix",
        "test_variable": "architecture_baseline",
        "overrides": {"bridge_interval": 4},
    },
    {
        "name": "02-bimamba",
        "backbone": "bimamba",
        "test_variable": "architecture_baseline",
        "overrides": {},
    },
    {
        "name": "03-transformer",
        "backbone": "transformer",
        "test_variable": "architecture_baseline",
        "overrides": {},
    },
    {
        "name": "04-dual-no-bridge",
        "backbone": "dual_helix",
        "test_variable": "bridge_disabled",
        "overrides": {"bridge_interval": 4, "disable_bridge": True},
    },
    {
        "name": "05-dual-bridge1",
        "backbone": "dual_helix",
        "test_variable": "bridge_interval",
        "overrides": {"bridge_interval": 1},
    },
    {
        "name": "06-transformer-bpe",
        "backbone": "transformer",
        "test_variable": "tokenizer_ablation",
        "comparison": "03-transformer-base-vs-06-transformer-bpe",
        "overrides": {"tokenizer_type": "bpe"},
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(REPOSITORY_DIR), *arguments], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_versions() -> dict[str, str | None]:
    versions = {
        "python": sys.version.split()[0],
        "torch": installed_version("torch"),
        "mamba_ssm": installed_version("mamba-ssm"),
        "wandb": installed_version("wandb"),
    }
    try:
        import torch

        versions["cuda_runtime"] = torch.version.cuda
        versions["cuda_available"] = str(torch.cuda.is_available())
        versions["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        versions.update({"cuda_runtime": None, "cuda_available": None, "device_name": None})
    return versions


def expand_fastas(patterns: list[str]) -> list[str]:
    files: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.update(str(Path(path).resolve()) for path in matches if Path(path).is_file())
        elif Path(pattern).is_file():
            files.add(str(Path(pattern).resolve()))
    return sorted(files)


def seed_output_root(output_root: Path, seed: int) -> Path:
    """Return the isolated evidence directory for one random seed."""
    return output_root / f"seed-{int(seed)}"


def ensure_empty_output_dir(path: Path) -> None:
    """Refuse to mix or overwrite evidence from an earlier run."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing nonempty seed output directory: {path}. "
            "Archive it or select a different --output-root before starting a new run."
        )


def build_config(args: argparse.Namespace, experiment: dict[str, Any], output_root: Path) -> dict[str, Any]:
    profile = HARDWARE_PROFILES[args.profile]
    config = {
        **COMMON_CONFIG,
        **profile,
        "fasta": args.fasta,
        "split_manifest": args.split_manifest,
        "seed": args.seed,
        "backbone": experiment["backbone"],
        **experiment["overrides"],
        "save_dir": str(output_root / experiment["name"]),
        "wandb_name": f"{experiment['name']}-seed-{args.seed}",
        "wandb_group": f"{WANDB_GROUP_PREFIX}-{args.profile}",
        "wandb_job_type": experiment["test_variable"],
        "wandb_tags": ",".join(
            [
                "controlled-ablation",
                f"profile-{args.profile}",
                f"backbone-{experiment['backbone']}",
                f"tokenizer-{experiment['overrides'].get('tokenizer_type', COMMON_CONFIG['tokenizer_type'])}",
                f"seed-{args.seed}",
            ]
        ),
    }
    if args.steps is not None:
        config["steps_per_epoch"] = args.steps
    if args.no_wandb:
        config["no_wandb"] = True
        for key in ("wandb_name", "wandb_group", "wandb_job_type", "wandb_tags"):
            config.pop(key, None)
    return config


def describe_tokenizer(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the tokenizer metadata that defines a run's output space."""
    from dna_model.tokenizer import build_tokenizer

    tokenizer_type = str(config["tokenizer_type"])
    try:
        tokenizer = build_tokenizer(
            tokenizer_type,
            vocab_path=str(config["tokenizer_path"]),
            k_mer_sizes=[int(item) for item in str(config["k_mer_sizes"]).split(",") if item],
            max_length=int(config["max_length"]),
        )
    except Exception as exc:
        if tokenizer_type == "bpe":
            raise RuntimeError(
                "The BPE tokenizer control requires dna_model/bpe_vocab/tokenizer.json. "
                "Build it from the verified corpus with scripts/train_bpe.py before running this control."
            ) from exc
        raise
    return {
        "tokenizer_type": tokenizer.tokenizer_type,
        "vocab_size": int(tokenizer.vocab_size),
        "vocab_sha256": getattr(tokenizer, "vocab_sha256", None),
        "kmer_size": int(tokenizer.kmer_size),
        "kmer_vocab_size": int(tokenizer.kmer_vocab_size),
    }


def config_to_command(config: dict[str, Any], *, print_params_only: bool = False) -> list[str]:
    command = [sys.executable, str(TRAIN_SCRIPT)]
    for key, value in config.items():
        if value is None:
            continue
        flag = TRAINER_FLAG_OVERRIDES.get(key, "--" + key)
        if isinstance(value, bool):
            if key == "use_kmer_mix" and not value:
                command.append("--no-kmer-mix")
            elif value:
                command.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
            continue
        command.extend([flag, str(value)])
    if print_params_only:
        command.extend(["--print_params_only", "--no_wandb"])
    return command


def run_command(command: list[str], env: dict[str, str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_DIR,
        env=env,
        text=True,
        capture_output=capture,
        check=False,
    )


def final_eval_metrics(run_dir: Path) -> dict[str, Any] | None:
    history_path = run_dir / "eval_history.jsonl"
    if not history_path.is_file():
        return None
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[-1] if records else None


def preflight_experiments(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every selected condition whose exact parameter count must be recorded."""
    return list(experiments)


def parse_parameter_count_millions(stdout: str) -> float:
    """Extract train.py's final ``--print_params_only`` value."""
    for line in reversed(stdout.splitlines()):
        value = line.strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    raise ValueError("Parameter preflight did not emit a numeric parameter count")


def final_run_evidence(run_dir: Path) -> dict[str, Any]:
    """Collect the canonical end-of-run evaluation and resource evidence."""
    summary_path = run_dir / "run_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            summary = loaded
    return {
        "final_eval": final_eval_metrics(run_dir) or summary.get("final_eval"),
        "parameters": summary.get("parameters"),
        "cuda_memory": summary.get("cuda_memory", {}),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def validate_inputs(args: argparse.Namespace) -> None:
    if args.prepare_data:
        command = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "prepare_ablation_data.py"),
            "--download-refseq",
            "--target-mb-per-domain",
            str(args.prepare_target_mb_per_domain),
            "--output",
            args.split_manifest,
            "--seed",
            str(args.seed),
        ]
        if args.rebuild_tokenizer:
            command.append("--rebuild-tokenizer")
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        args.fasta = ["data/domain_fastas/*.fa"]

    files = expand_fastas(args.fasta or [])
    if not files and not args.dry_run:
        raise FileNotFoundError(
            "No FASTA files matched. Run scripts/prepare_ablation_data.py --download-refseq "
            "for a fresh corpus, or pass --fasta with existing files."
        )
    if files:
        args.fasta = files
    split_path = Path(args.split_manifest)
    if not split_path.is_absolute():
        split_path = PROJECT_DIR / split_path
    args.split_manifest = str(split_path.resolve())
    if not args.allow_implicit_split and not split_path.is_file() and not args.dry_run:
        raise FileNotFoundError(
            f"Split manifest not found: {split_path}. Create it with scripts/prepare_ablation_data.py."
        )
    if args.allow_implicit_split:
        args.split_manifest = None


def prepare_inputs(args: argparse.Namespace) -> None:
    """Resolve data-backed run inputs, preserving no-data parameter preflight."""
    if args.preflight_only:
        if args.prepare_data:
            raise SystemExit(
                "--preflight-only cannot be combined with --prepare-data; preflight deliberately reads no data."
            )
        # train.py keeps --fasta required even though --print-params-only exits
        # before opening it. Use a clear sentinel rather than requiring users to
        # invent a FASTA or a split manifest just to inspect model sizes.
        if not args.fasta:
            args.fasta = ["__parameter_preflight_no_data__"]
        args.split_manifest = None
        return

    if not args.prepare_data and not args.fasta:
        args.fasta = ["data/domain_fastas/*.fa"]
    validate_inputs(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled three-backbone DNA architecture ablation")
    parser.add_argument("--profile", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument("--fasta", nargs="+", default=None, help="FASTA paths or glob patterns")
    parser.add_argument("--split-manifest", default="data/domain_fastas/split_manifest.jsonl")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=None, help="Override profile steps per epoch")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=["dual_helix", "bimamba", "transformer"],
        default=None,
        help="Limit execution to selected backbones; Dual Helix retains its bridge controls unless --only-baselines is used.",
    )
    parser.add_argument(
        "--only-baselines", action="store_true", help="Run Dual Bridge-4, BiMamba, and Transformer only"
    )
    parser.add_argument(
        "--include-tokenizer-ablation",
        action="store_true",
        help="Also run the matched Transformer BPE control alongside the base-token architecture baselines.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--allow-implicit-split", action="store_true", help="Use record-ratio split; smoke testing only"
    )
    parser.add_argument(
        "--prepare-data", action="store_true", help="Download a RefSeq corpus then create an assembly split"
    )
    parser.add_argument("--prepare-target-mb-per-domain", type=float, default=64.0)
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"Training entry point not found: {TRAIN_SCRIPT}")
    if args.prepare_data and args.fasta:
        raise SystemExit("--prepare-data and --fasta cannot be used together")
    prepare_inputs(args)

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "Working tree is dirty. Commit/stash changes or pass --allow-dirty for an explicit local run."
        )

    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_DIR / args.output_root
    output_root = output_root.resolve()
    run_root = seed_output_root(output_root, args.seed)
    ensure_empty_output_dir(run_root)
    tokenizer_controls = [
        experiment for experiment in EXPERIMENTS if experiment["test_variable"] == "tokenizer_ablation"
    ]
    architecture_experiments = [
        experiment for experiment in EXPERIMENTS if experiment["test_variable"] != "tokenizer_ablation"
    ]
    experiments = architecture_experiments[:3] if args.only_baselines else architecture_experiments
    if args.include_tokenizer_ablation:
        experiments.extend(tokenizer_controls)
    if args.backbones:
        experiments = [experiment for experiment in experiments if experiment["backbone"] in set(args.backbones)]
    if not experiments:
        raise SystemExit("No experiments selected")
    manifest_path = run_root / "ablation_manifest.json"
    tokenizer_specs = {
        experiment["name"]: describe_tokenizer(build_config(args, experiment, run_root)) for experiment in experiments
    }
    manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "profile": args.profile,
        "seed": args.seed,
        "profile_config": HARDWARE_PROFILES[args.profile],
        "dependencies": dependency_versions(),
        "fasta": args.fasta,
        "split_manifest": args.split_manifest,
        "primary_objective": {"name": "eval/bits_per_base", "direction": "minimize"},
        "secondary_objective": {"name": "eval/loss", "direction": "minimize"},
        "tokenizer_policy": {
            "architecture_baselines": {
                "tokenizer_type": "base",
                "vocab_size": 10,
                "comparison_note": "All three architecture baselines use the same single-base tokenization and output vocabulary.",
            },
            "tokenizer_control": {
                "enabled": bool(args.include_tokenizer_ablation),
                "comparison": "matched Transformer base versus BPE",
                "interpretation": "BPE changes vocabulary size and parameter count by design; compare its base-normalized metrics separately from architecture effects.",
            },
        },
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_PROJECT"] = WANDB_PROJECT

    # Bridge controls change instantiated modules, so every selected condition
    # gets its own exact parameter count before expensive training starts.
    for preflight_experiment in preflight_experiments(experiments):
        backbone = preflight_experiment["backbone"]
        config = build_config(args, preflight_experiment, run_root / "preflight")
        command = config_to_command(config, print_params_only=True)
        record: dict[str, Any] = {
            "name": preflight_experiment["name"],
            "backbone": backbone,
            "tokenizer": tokenizer_specs[preflight_experiment["name"]],
            "command": command,
            "status": "pending",
        }
        manifest.setdefault("preflight", []).append(record)
        if args.dry_run:
            record["status"] = "dry_run"
        else:
            result = run_command(command, env, capture=True)
            record.update({"return_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            record["status"] = "passed" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                write_manifest(manifest_path, manifest)
                raise RuntimeError(f"Preflight failed for {backbone}; see {manifest_path}")
            record["parameters_m"] = parse_parameter_count_millions(result.stdout)
        write_manifest(manifest_path, manifest)

    if args.preflight_only:
        print(f"Preflight complete: {manifest_path}")
        return 0

    for experiment in experiments:
        if git("rev-parse", "HEAD") != commit:
            raise RuntimeError("Git revision changed during the ablation")
        config = build_config(args, experiment, run_root)
        command = config_to_command(config)
        run_dir = Path(config["save_dir"])
        record = {
            "name": experiment["name"],
            "backbone": experiment["backbone"],
            "test_variable": experiment["test_variable"],
            "comparison": experiment.get("comparison"),
            "tokenizer": tokenizer_specs[experiment["name"]],
            "config": config,
            "command": command,
            "status": "pending",
        }
        manifest["runs"].append(record)
        write_manifest(manifest_path, manifest)
        print("\n" + "=" * 80)
        print(f"{experiment['name']} ({experiment['backbone']})")
        print(shlex.join(command))

        if args.dry_run:
            record["status"] = "dry_run"
            write_manifest(manifest_path, manifest)
            continue

        record["status"] = "running"
        record["started_at"] = utc_now()
        write_manifest(manifest_path, manifest)
        start = time.monotonic()
        result = run_command(command, env)
        record["return_code"] = result.returncode
        record["duration_seconds"] = round(time.monotonic() - start, 3)
        record["finished_at"] = utc_now()
        record["status"] = "finished" if result.returncode == 0 else "failed"
        record.update(final_run_evidence(run_dir))
        write_manifest(manifest_path, manifest)
        if result.returncode != 0 and not args.continue_on_error:
            raise RuntimeError(f"{experiment['name']} failed; see {manifest_path}")

    print(f"Ablation complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

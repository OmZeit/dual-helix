#!/usr/bin/env python3
"""Plan and run reproducible follow-up studies for the True DNA ablation.

The existing controlled-ablation runner is deliberately focused on one
architecture suite.  This script schedules the next research questions without
silently mixing them together:

* ``tokenizer``: matched Transformer base-vs-BPE replication;
* ``tokenizer_capacity``: total-parameter-matched tokenizer sensitivity study;
* ``masking``: 80/10/10 control versus 100% mask span-corruption rates;
* ``architecture``: capacity-matched base-token architecture study;
* ``factorial``: architecture x tokenizer interaction study;
* ``family``: fresh training against a deterministic family-held-out split;
* ``context``: selected architecture/context-length scaling curve;
* ``robustness``: strict MLM plus reverse-complement checks for completed runs;
* ``downstream``: independent genomic benchmarks supplied by the user.

It defaults to a *plan* so invoking it cannot accidentally consume days of
GPU time.  Pass ``--run`` after inspecting ``suite_manifest.json``.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_controlled_ablation import (  # noqa: E402
    COMMON_CONFIG,
    HARDWARE_PROFILES,
    config_to_command,
)

STAGES = (
    "tokenizer",
    "tokenizer_capacity",
    "masking",
    "architecture",
    "factorial",
    "family",
    "context",
    "robustness",
    "downstream",
)
BACKBONES = ("dual_helix", "bimamba", "transformer")
MIB = 1024**2
SENSITIVE_SOURCE_PATTERNS = (
    ".env",
    ".env.*",
    "*credential*",
    "*secret*",
    "*api_key*",
    "*api-key*",
    "*.key",
    "*.pem",
)


@dataclass(frozen=True)
class StudyJob:
    """A single fresh training condition in the research matrix."""

    name: str
    study: str
    backbone: str
    tokenizer_type: str
    seed: int
    steps: int
    max_length: int
    split_manifest: str
    capacity_matched: bool
    comparison_scope: str
    mask_replacement_strategy: str
    mask_fraction: float | None
    run_dir: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_input_fingerprints(args: argparse.Namespace, jobs: Iterable[StudyJob]) -> dict[str, dict[str, Any]]:
    """Hash every file that can affect training or strict evaluation."""
    paths = {
        Path(args.assembly_split),
        Path(args.tokenizer_path),
        Path(args.selection_manifest),
        Path(args.dataset_manifest),
        *(Path(path) for path in args.fasta_files),
        *(Path(job.split_manifest) for job in jobs),
    }
    if args.downstream_manifest is not None:
        paths.add(Path(args.downstream_manifest))
    fingerprints: dict[str, dict[str, Any]] = {}
    for raw_path in sorted(paths, key=lambda path: str(path).lower()):
        path = raw_path.expanduser().resolve()
        record: dict[str, Any] = {"exists": path.is_file()}
        if path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
        fingerprints[str(path)] = record
    return fingerprints


def _job_protocol_fingerprint(
    args: argparse.Namespace,
    job: StudyJob,
    config: dict[str, Any],
    *,
    source_snapshot_sha256: str,
    inputs: dict[str, dict[str, Any]],
    capacity: dict[str, dict[str, float | int]],
) -> str:
    capacity_selection = capacity.get(f"{job.backbone}:{job.tokenizer_type}") if job.capacity_matched else None
    payload = {
        "schema": "true-dna-job-protocol-fingerprint-v1",
        "job": asdict(job),
        "config": config,
        "source_snapshot_sha256": source_snapshot_sha256,
        "inputs": inputs,
        "capacity_selection": capacity_selection,
        "strict_evaluation": {
            "eval_batch_size": args.eval_batch_size,
            "eval_max_batches": args.eval_max_batches,
            "eval_sample_seed": args.eval_sample_seed,
            "mask_coordinate_system": args.mask_coordinate_system,
            "mask_seed": args.mask_seed,
            "baseline_alpha": args.baseline_alpha,
            "rc_check_seqs": args.rc_check_seqs,
            "strict_learning_curve": args.strict_learning_curve,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path.resolve())


def _source_snapshot(protocol_dir: Path) -> dict[str, Any]:
    """Archive the exact dirty source tree used to plan or execute a suite."""
    allowed_suffixes = {".py", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}
    excluded_parts = {".git", ".venv", "__pycache__", "data", "experiments", "build"}
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_DIR,
        text=True,
        capture_output=True,
        check=True,
    )
    files = []
    for raw_path in result.stdout.split("\0"):
        if not raw_path:
            continue
        relative = Path(raw_path)
        path = (REPOSITORY_DIR / relative).resolve()
        name = relative.name.lower()
        credential_shaped = name != ".env.example" and any(
            fnmatch.fnmatch(name, pattern) for pattern in SENSITIVE_SOURCE_PATTERNS
        )
        if (
            not path.is_file()
            or path.suffix.lower() not in allowed_suffixes
            or any(part in excluded_parts for part in relative.parts)
            or "bpe_vocab" in relative.parts
            or credential_shaped
        ):
            continue
        files.append(path)
    files.sort(key=lambda path: str(path).lower())

    source_dir = protocol_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    temporary_archive = source_dir / "source_snapshot-building.zip"
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(REPOSITORY_DIR.resolve()).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    archive_digest = _sha256(temporary_archive)
    archive_path = source_dir / f"source_snapshot-{archive_digest[:16]}.zip"
    if archive_path.is_file():
        temporary_archive.unlink()
    else:
        temporary_archive.replace(archive_path)

    def git_output(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_DIR), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git_output("status", "--porcelain=v1")
    snapshot = {
        "schema": "true-dna-source-snapshot-v1",
        "created_at": _utc_now(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": status.splitlines() if status else [],
        "git_dirty": bool(status),
        "archive": str(archive_path.resolve()),
        "archive_sha256": archive_digest,
        "files": {
            path.relative_to(REPOSITORY_DIR.resolve()).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
    }
    snapshot_path = source_dir / f"source_snapshot-{archive_digest[:16]}.json"
    _write_json(snapshot_path, snapshot)
    snapshot["manifest"] = str(snapshot_path.resolve())
    snapshot["manifest_sha256"] = _sha256(snapshot_path)
    return snapshot


def _expand_fastas(patterns: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.update(Path(item).resolve() for item in matches if Path(item).is_file())
        else:
            candidate = Path(pattern).expanduser()
            if candidate.is_file():
                files.add(candidate.resolve())
    return sorted(files)


def _slug(backbone: str) -> str:
    return backbone.replace("_helix", "").replace("_", "-")


def _checkpoint_for(run_dir: Path) -> Path | None:
    for name in ("checkpoint_best.pt", "checkpoint_latest.pt"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    candidates = sorted(run_dir.glob("checkpoint_step*.pt"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _parameter_count_m(command: list[str]) -> float:
    """Build a model once and return the trainer's exact parameter count."""

    result = subprocess.run(command, cwd=PROJECT_DIR, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Parameter preflight failed:\n{detail}")
    values = re.findall(r"^\s*(\d+(?:\.\d+)?)\s*$", result.stdout, flags=re.MULTILINE)
    if not values:
        raise RuntimeError(f"Could not read parameter count from preflight output:\n{result.stdout}")
    return float(values[-1])


def _checkpoint_memory(checkpoint: Path) -> dict[str, float | int] | None:
    """Read allocator peaks captured inside a checkpoint without trusting pickle."""

    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        raw = payload.get("metrics", {}).get("runtime", {}).get("cuda_memory", {})
        if not isinstance(raw, dict) or not raw:
            return None
        return {
            "allocated_mib": round(float(raw["memory_allocated_bytes"]) / MIB, 3),
            "reserved_mib": round(float(raw["memory_reserved_bytes"]) / MIB, 3),
            "peak_allocated_mib": round(float(raw["max_memory_allocated_bytes"]) / MIB, 3),
            "peak_reserved_mib": round(float(raw["max_memory_reserved_bytes"]) / MIB, 3),
        }
    except Exception as exc:  # A failed summary must never invalidate a completed run.
        return {"unavailable": str(exc)}


def _base_config(args: argparse.Namespace, job: StudyJob) -> dict[str, Any]:
    profile = dict(HARDWARE_PROFILES[args.profile])
    profile["max_length"] = int(job.max_length)
    profile["stride"] = max(1, int(job.max_length) // 2)
    batch_label = f"mb{profile['batch_size']}-ga{profile['grad_accum']}"
    compile_enabled = bool(args.compile_transformer and job.backbone == "transformer")
    compile_label = f"compiled-{args.compile_mode}" if compile_enabled else "eager"
    eval_interval = min(int(args.eval_interval), int(job.steps))
    save_every = min(int(args.save_every), int(job.steps))
    # The cosine scheduler reserves 10% of total steps for cooldown.  Keep the
    # normal 200-step warmup for research runs, but make tiny smoke schedules
    # valid instead of failing before the first optimizer step.
    scheduler_cooldown = int(job.steps * 0.1)
    warmup_steps = min(int(COMMON_CONFIG["warmup_steps"]), max(0, int(job.steps) - scheduler_cooldown))
    run_dir = Path(job.run_dir)
    config = {
        **COMMON_CONFIG,
        **profile,
        "tokenizer_type": job.tokenizer_type,
        "tokenizer_path": str(args.tokenizer_path),
        "fasta": [str(path) for path in args.fasta_files],
        "split_manifest": job.split_manifest,
        "seed": job.seed,
        "backbone": job.backbone,
        "bridge_interval": 4,
        "steps_per_epoch": job.steps,
        "warmup_steps": warmup_steps,
        "eval_interval": eval_interval,
        "save_every": save_every,
        "eval_max_batches": args.eval_max_batches,
        "save_dir": str(run_dir),
        "wandb_name": f"{job.name}-{batch_label}-{compile_label}",
        "wandb_group": f"true-dna-research-v2-{job.study}-{args.profile}",
        "wandb_job_type": job.study,
        "wandb_tags": ",".join(
            (
                "research-suite-v2",
                f"study-{job.study}",
                f"backbone-{job.backbone}",
                f"tokenizer-{job.tokenizer_type}",
                f"context-{job.max_length}",
                f"seed-{job.seed}",
                f"replacement-{job.mask_replacement_strategy}",
                batch_label,
                compile_label,
            )
        ),
        "no_wandb": bool(args.no_wandb),
        "compile": compile_enabled,
        "compile_mode": args.compile_mode,
        "label_smoothing": float(args.label_smoothing),
        "mask_coordinate_system": args.mask_coordinate_system,
        "mask_replacement_strategy": job.mask_replacement_strategy,
    }
    if job.mask_fraction is not None:
        config.update(
            {
                "use_curriculum": False,
                "mask_fraction": float(job.mask_fraction),
                "mask_start": float(job.mask_fraction),
                "mask_end": float(job.mask_fraction),
            }
        )
    return config


def _make_job(
    args: argparse.Namespace,
    *,
    study: str,
    backbone: str,
    tokenizer_type: str,
    seed: int,
    steps: int,
    max_length: int,
    split_manifest: Path,
    capacity_matched: bool = False,
    comparison_scope: str = "fixed_budget_system",
    mask_replacement_strategy: str | None = None,
    mask_fraction: float | None = None,
) -> StudyJob:
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if tokenizer_type not in {"base", "bpe"}:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    replacement_strategy = mask_replacement_strategy or args.mask_replacement_strategy
    if replacement_strategy not in {"mask", "bert_80_10_10"}:
        raise ValueError(f"Unsupported mask replacement strategy: {replacement_strategy}")
    if mask_fraction is not None and not 0.0 <= float(mask_fraction) <= 1.0:
        raise ValueError(f"mask_fraction must be in [0, 1], got {mask_fraction}")
    split_name = split_manifest.stem.replace("_manifest", "")
    condition = f"{_slug(backbone)}-{tokenizer_type}-ctx{max_length}-seed{seed}"
    if study == "masking":
        fraction_label = int(round(float(mask_fraction) * 100)) if mask_fraction is not None else "curriculum"
        condition = f"{condition}-{replacement_strategy}-mask{fraction_label}"
    name = f"{study}-{condition}"
    run_dir = args.output_root / study / split_name / condition
    return StudyJob(
        name=name,
        study=study,
        backbone=backbone,
        tokenizer_type=tokenizer_type,
        seed=int(seed),
        steps=int(steps),
        max_length=int(max_length),
        split_manifest=str(split_manifest.resolve()),
        capacity_matched=bool(capacity_matched),
        comparison_scope=str(comparison_scope),
        mask_replacement_strategy=replacement_strategy,
        mask_fraction=None if mask_fraction is None else float(mask_fraction),
        run_dir=str(run_dir.resolve()),
    )


def _build_jobs(args: argparse.Namespace, stages: list[str], family_split: Path) -> list[StudyJob]:
    jobs: list[StudyJob] = []
    assembly_split = args.assembly_split.resolve()
    default_context = int(HARDWARE_PROFILES[args.profile]["max_length"])

    if "tokenizer" in stages:
        for seed in args.seeds:
            for tokenizer_type in ("base", "bpe"):
                jobs.append(
                    _make_job(
                        args,
                        study="tokenizer",
                        backbone="transformer",
                        tokenizer_type=tokenizer_type,
                        seed=seed,
                        steps=args.tokenizer_steps,
                        max_length=default_context,
                        split_manifest=assembly_split,
                        comparison_scope="same_backbone_end_to_end_tokenizer_system",
                    )
                )

    if "tokenizer_capacity" in stages:
        for seed in args.seeds:
            for tokenizer_type in ("base", "bpe"):
                jobs.append(
                    _make_job(
                        args,
                        study="tokenizer_capacity",
                        backbone="transformer",
                        tokenizer_type=tokenizer_type,
                        seed=seed,
                        steps=args.tokenizer_steps,
                        max_length=default_context,
                        split_manifest=assembly_split,
                        capacity_matched=True,
                        comparison_scope="total_parameter_matched_tokenizer_sensitivity",
                    )
                )

    if "masking" in stages:
        masking_conditions = (
            ("bert_80_10_10", 0.15),
            ("mask", 0.15),
            ("mask", 0.25),
            ("mask", 0.35),
        )
        for seed in args.masking_seeds:
            for replacement_strategy, mask_fraction in masking_conditions:
                jobs.append(
                    _make_job(
                        args,
                        study="masking",
                        backbone=args.masking_backbone,
                        tokenizer_type=args.masking_tokenizer,
                        seed=seed,
                        steps=args.masking_steps,
                        max_length=default_context,
                        split_manifest=assembly_split,
                        comparison_scope="fixed_budget_mask_replacement_and_rate_pilot",
                        mask_replacement_strategy=replacement_strategy,
                        mask_fraction=mask_fraction,
                    )
                )

    if "architecture" in stages:
        for seed in args.seeds:
            for backbone in args.architecture_backbones:
                jobs.append(
                    _make_job(
                        args,
                        study="architecture",
                        backbone=backbone,
                        tokenizer_type="base",
                        seed=seed,
                        steps=args.architecture_steps,
                        max_length=default_context,
                        split_manifest=assembly_split,
                        capacity_matched=True,
                    )
                )

    if "factorial" in stages:
        for seed in args.seeds:
            for backbone in args.factorial_backbones:
                for tokenizer_type in ("base", "bpe"):
                    jobs.append(
                        _make_job(
                            args,
                            study="factorial",
                            backbone=backbone,
                            tokenizer_type=tokenizer_type,
                            seed=seed,
                            steps=args.factorial_steps,
                            max_length=default_context,
                            split_manifest=assembly_split,
                        )
                    )

    if "family" in stages:
        for seed in args.seeds:
            for backbone in args.family_backbones:
                jobs.append(
                    _make_job(
                        args,
                        study="family",
                        backbone=backbone,
                        tokenizer_type="base",
                        seed=seed,
                        steps=args.family_steps,
                        max_length=default_context,
                        split_manifest=family_split,
                        capacity_matched=True,
                    )
                )

    if "context" in stages:
        for seed in args.seeds:
            for backbone in args.context_backbones:
                for context in args.contexts:
                    jobs.append(
                        _make_job(
                            args,
                            study="context",
                            backbone=backbone,
                            tokenizer_type=args.context_tokenizer,
                            seed=seed,
                            steps=args.context_steps,
                            max_length=context,
                            split_manifest=assembly_split,
                        )
                    )
    return jobs


def _capacity_settings(args: argparse.Namespace, jobs: Iterable[StudyJob]) -> dict[str, dict[str, float | int]]:
    """Choose the closest width per backbone/tokenizer output space."""

    selected: dict[str, dict[str, float | int]] = {}
    representatives: dict[str, StudyJob] = {}
    for job in jobs:
        key = f"{job.backbone}:{job.tokenizer_type}"
        if job.capacity_matched and key not in representatives:
            representatives[key] = job

    for key, job in representatives.items():
        candidates: list[dict[str, float | int]] = []
        for hidden_size in args.capacity_hidden_sizes:
            heads = hidden_size // 64
            if heads <= 0:
                continue
            probe = _base_config(args, job)
            probe.update(
                {
                    "hidden_size": hidden_size,
                    "num_attention_heads": heads,
                    "save_dir": str(
                        args.output_root / "parameter_preflight" / key.replace(":", "-") / str(hidden_size)
                    ),
                    "wandb_name": f"parameter-preflight-{key.replace(':', '-')}-{hidden_size}",
                    "no_wandb": True,
                }
            )
            count = _parameter_count_m(config_to_command(probe, print_params_only=True))
            candidates.append({"hidden_size": hidden_size, "num_attention_heads": heads, "parameters_m": count})
        if not candidates:
            raise RuntimeError(f"No valid parameter candidates for {key}")
        best = min(candidates, key=lambda item: abs(float(item["parameters_m"]) - args.capacity_target_m))
        selected[key] = {
            **best,
            "target_parameters_m": float(args.capacity_target_m),
            "absolute_error_m": abs(float(best["parameters_m"]) - args.capacity_target_m),
            "candidates": candidates,
        }
    return selected


def _apply_capacity_settings(
    config: dict[str, Any], job: StudyJob, settings: dict[str, dict[str, float | int]]
) -> None:
    if job.capacity_matched:
        selected = settings[f"{job.backbone}:{job.tokenizer_type}"]
        config["hidden_size"] = int(selected["hidden_size"])
        config["num_attention_heads"] = int(selected["num_attention_heads"])


def _resolved_config(
    args: argparse.Namespace,
    job: StudyJob,
    capacity: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    """Build the exact config shown in the plan and later used for execution."""
    config = _base_config(args, job)
    _apply_capacity_settings(config, job, capacity)
    return config


def _split_protocol_name(split_manifest: str | Path) -> str:
    path = Path(split_manifest)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def _eval_sample_path(args: argparse.Namespace, job: StudyJob) -> Path:
    return (
        args.output_root
        / "protocol"
        / "evaluation_samples"
        / (
            f"{_split_protocol_name(job.split_manifest)}-ctx{job.max_length}"
            f"-stride{max(1, job.max_length // 2)}-b{args.eval_batch_size}"
            f"-n{args.eval_max_batches}-seed{args.eval_sample_seed}.jsonl"
        )
    )


def _baseline_model_path(args: argparse.Namespace, split_manifest: str | Path) -> Path:
    return args.output_root / "protocol" / "baselines" / f"{_split_protocol_name(split_manifest)}.json"


def _strict_eval_command(
    args: argparse.Namespace,
    job: StudyJob,
    checkpoint: Path,
    output: Path,
    *,
    rc_check_seqs: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate.py"),
        "--checkpoint",
        str(checkpoint),
        "--fasta",
        *(str(path) for path in args.fasta_files),
        "--tokenizer-type",
        job.tokenizer_type,
        "--tokenizer-path",
        str(args.tokenizer_path),
        "--split-manifest",
        job.split_manifest,
        "--batch-size",
        str(args.eval_batch_size),
        "--workers",
        str(args.eval_workers),
        "--max-batches",
        str(args.eval_max_batches),
        "--max-length",
        str(job.max_length),
        "--stride",
        str(max(1, job.max_length // 2)),
        "--eval-sample-manifest",
        str(_eval_sample_path(args, job)),
        "--eval-sample-seed",
        str(args.eval_sample_seed),
        "--mask-coordinate-system",
        args.mask_coordinate_system,
        "--mask-seed",
        str(args.mask_seed),
        "--baseline-model-json",
        str(_baseline_model_path(args, job.split_manifest)),
        "--rc-check-seqs",
        str(args.rc_check_seqs if rc_check_seqs is None else rc_check_seqs),
        "--strict-masked-targets",
        "--output-json",
        str(output),
    ]
    return command


def _baseline_command(args: argparse.Namespace, split_manifest: str | Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "build_sequence_baselines.py"),
        "--fasta",
        *(str(path) for path in args.fasta_files),
        "--split-manifest",
        str(split_manifest),
        "--alpha",
        str(args.baseline_alpha),
        "--output-json",
        str(output),
    ]


def _budget_summary(args: argparse.Namespace, job: StudyJob) -> dict[str, Any]:
    profile = HARDWARE_PROFILES[args.profile]
    effective_batch = int(profile["batch_size"]) * int(profile["grad_accum"])
    estimated_raw_bases = int(job.steps) * effective_batch * int(job.max_length)
    classification = "pilot" if job.steps < args.conclusion_step_threshold else "conclusion_candidate"
    return {
        "classification": classification,
        "optimizer_steps": int(job.steps),
        "effective_batch_windows": effective_batch,
        "max_raw_bases_per_window": int(job.max_length),
        "estimated_raw_bases_presented": estimated_raw_bases,
        "warning": (
            "Fixed-budget early-training comparison; do not describe as full-corpus training."
            if classification == "pilot"
            else "Longer fixed-budget run; conclusions still require learning-curve and baseline checks."
        ),
    }


def _run_command(command: list[str]) -> None:
    print("[run]", shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {shlex.join(command)}")


def _run_strict_learning_curve(
    args: argparse.Namespace, job: StudyJob, run_dir: Path
) -> tuple[Path | None, Path | None]:
    if not args.strict_learning_curve:
        return None, None
    checkpoints = sorted(
        run_dir.glob("checkpoint_step*.pt"),
        key=lambda path: int(re.search(r"step(\d+)", path.stem).group(1)),
    )
    if not checkpoints:
        return None, None
    curve_dir = run_dir / "strict_learning_curve"
    rows = []
    for checkpoint in checkpoints:
        step_match = re.search(r"step(\d+)", checkpoint.stem)
        step = int(step_match.group(1))
        output = curve_dir / f"step{step}.json"
        if not output.is_file():
            _run_command(_strict_eval_command(args, job, checkpoint, output, rc_check_seqs=0))
        metrics = json.loads(output.read_text(encoding="utf-8"))
        masked = metrics.get("masked_token", {})
        rows.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "metrics": str(output),
                "bits_per_base": masked.get("bits_per_base"),
                "conditional_valid_bits_per_base": masked.get("conditional_valid_bits_per_base"),
                "accuracy": masked.get("acc"),
                "baselines": masked.get("baselines"),
            }
        )
    curve_path = run_dir / "strict_learning_curve.json"
    _write_json(
        curve_path,
        {
            "schema": "true-dna-strict-learning-curve-v1",
            "evaluation_sample": str(_eval_sample_path(args, job)),
            "mask_coordinate_system": args.mask_coordinate_system,
            "mask_seed": args.mask_seed,
            "runs": rows,
        },
    )
    finite_rows = [row for row in rows if isinstance(row.get("bits_per_base"), (int, float))]
    best_checkpoint = (
        Path(min(finite_rows, key=lambda row: float(row["bits_per_base"]))["checkpoint"]) if finite_rows else None
    )
    return curve_path, best_checkpoint


def _run_job(
    args: argparse.Namespace,
    job: StudyJob,
    capacity: dict[str, dict[str, float | int]],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> None:
    run_dir = Path(job.run_dir)
    result_path = run_dir / "research_result.json"
    record = next(item for item in manifest["jobs"] if item["name"] == job.name)
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        expected_fingerprint = record.get("protocol_fingerprint")
        if not expected_fingerprint or saved.get("protocol_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                f"Refusing stale completed result for {job.name}: protocol fingerprint mismatch. "
                "Use a new output root or archive the previous result."
            )
        record.update(saved)
        _write_json(manifest_path, manifest)
        print(f"[skip] completed: {job.name}")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        if args.retry_interrupted:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive = run_dir.with_name(f"{run_dir.name}.interrupted-{stamp}")
            if archive.exists():
                raise RuntimeError(f"Refusing to replace interrupted-run archive: {archive}")
            run_dir.rename(archive)
            record["previous_interrupted_artifacts"] = str(archive)
            print(f"[retry] preserved interrupted artifacts at: {archive}")
        if args.continue_on_error:
            if not run_dir.exists() or not any(run_dir.iterdir()):
                # --retry-interrupted moved the stale directory aside, so this
                # condition can now start fresh instead of being skipped.
                pass
            else:
                # A power loss, WSL restart, or forced terminal close cannot be
                # caught by the original parent process.  Preserve all partial
                # artifacts for diagnosis, mark the stale run explicitly, and
                # allow the independent remaining conditions to continue.
                record.update(
                    {
                        "status": "interrupted",
                        "error": "Pre-existing incomplete run directory; artifacts preserved and condition skipped.",
                        "interrupted_at": _utc_now(),
                    }
                )
                _write_json(manifest_path, manifest)
                print(f"[skip] interrupted run preserved: {job.name}")
                return
        elif run_dir.exists() and any(run_dir.iterdir()):
            # A power loss, WSL restart, or forced terminal close cannot be
            # caught by the original parent process.  Preserve all partial
            # artifacts for diagnosis, mark the stale run explicitly, and
            # allow the independent remaining conditions to continue.
            raise RuntimeError(
                f"Refusing to overwrite incomplete run directory {run_dir}. "
                "Resume it manually, move it before rerunning, pass --continue-on-error to skip it, "
                "or use --retry-interrupted to archive it and restart the condition."
            )

    config = _resolved_config(args, job, capacity)
    command = config_to_command(config)
    record["config"] = config
    record["command"] = command
    record["status"] = "running"
    _write_json(manifest_path, manifest)

    try:
        record["parameters_m"] = _parameter_count_m(config_to_command(config, print_params_only=True))
        _run_command(command)
        training_checkpoint = _checkpoint_for(run_dir)
        if training_checkpoint is None:
            raise RuntimeError(f"No checkpoint was created in {run_dir}")
        learning_curve, strict_curve_best = _run_strict_learning_curve(args, job, run_dir)
        checkpoint = strict_curve_best or training_checkpoint
        strict_path = run_dir / "strict_metrics.json"
        strict_command = _strict_eval_command(args, job, checkpoint, strict_path)
        _run_command(strict_command)
        record.update(
            {
                "status": "completed",
                "checkpoint": str(checkpoint),
                "training_selected_checkpoint": str(training_checkpoint),
                "strict_curve_selected_checkpoint": str(strict_curve_best) if strict_curve_best is not None else None,
                "strict_evaluation": str(strict_path),
                "checkpoint_cuda_memory_mib": _checkpoint_memory(checkpoint),
                "completed_at": _utc_now(),
                "strict_learning_curve": str(learning_curve) if learning_curve is not None else None,
            }
        )
        _write_json(result_path, record)
    except Exception as exc:
        record.update({"status": "failed", "error": str(exc), "failed_at": _utc_now()})
        raise
    finally:
        _write_json(manifest_path, manifest)


def _family_audit_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "audit_ablation_integrity.py"),
        "--fasta",
        *(str(path) for path in args.fasta_files),
        "--split-manifest",
        str(args.assembly_split),
        "--selection-manifest",
        str(args.selection_manifest),
        "--dataset-manifest",
        str(args.dataset_manifest),
        "--output-dir",
        str(output_dir),
        "--family-holdout-fraction",
        str(args.family_holdout_fraction),
        "--seed",
        str(args.family_seed),
    ]


def _discover_completed_jobs(root: Path) -> list[tuple[StudyJob, Path]]:
    discovered: list[tuple[StudyJob, Path]] = []
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        checkpoint = _checkpoint_for(run_dir)
        if checkpoint is None:
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            job = StudyJob(
                name=run_dir.name,
                study=run_dir.parents[2].name if len(run_dir.parents) >= 3 else "existing",
                backbone=str(config["backbone"]),
                tokenizer_type=str(config["tokenizer_type"]),
                seed=int(config["seed"]),
                steps=int(config.get("steps_per_epoch", 0)),
                max_length=int(config["max_length"]),
                split_manifest=str(config["split_manifest"]),
                capacity_matched=False,
                comparison_scope=str(config.get("comparison_scope", "discovered_existing_checkpoint")),
                mask_replacement_strategy=str(config.get("mask_replacement_strategy", "bert_80_10_10")),
                mask_fraction=(float(config["mask_fraction"]) if config.get("mask_fraction") is not None else None),
                run_dir=str(run_dir),
            )
            discovered.append((job, checkpoint))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            print(f"[warn] skipping incomplete config: {config_path}", file=sys.stderr)
    return discovered


def _run_completed_strict_checks(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    completed = _discover_completed_jobs(args.output_root)
    if not completed:
        raise RuntimeError("No completed suite checkpoints found for robustness checks")
    records = manifest.setdefault("robustness", [])
    for job, checkpoint in completed:
        baseline_path = _baseline_model_path(args, job.split_manifest)
        if not baseline_path.is_file():
            _run_command(_baseline_command(args, job.split_manifest, baseline_path))
        output = Path(job.run_dir) / "strict_rc_metrics.json"
        if output.is_file():
            print(f"[skip] existing strict/RC check: {checkpoint}")
            continue
        record = {"checkpoint": str(checkpoint), "output": str(output), "status": "running"}
        records.append(record)
        _write_json(manifest_path, manifest)
        try:
            _run_command(_strict_eval_command(args, job, checkpoint, output))
            record.update({"status": "completed", "completed_at": _utc_now()})
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
            raise
        finally:
            _write_json(manifest_path, manifest)


def _run_downstream_benchmarks(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    if not args.downstream_manifest and not args.downstream_preset:
        raise RuntimeError("--downstream-manifest or --downstream-preset is required for the downstream stage")
    completed = _discover_completed_jobs(args.output_root)
    if not completed:
        raise RuntimeError("No completed suite checkpoints found for downstream benchmarks")
    records = manifest.setdefault("downstream", [])
    for job, checkpoint in completed:
        output = Path(job.run_dir) / "genomic_benchmark.json"
        if output.is_file():
            print(f"[skip] existing downstream benchmark: {checkpoint}")
            continue
        command = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_genomic.py"),
            "--checkpoint",
            str(checkpoint),
            "--tokenizer-type",
            job.tokenizer_type,
            "--tokenizer_path",
            str(args.tokenizer_path),
            "--max_length",
            str(job.max_length),
            "--batch_size",
            str(args.eval_batch_size),
            "--seed",
            str(job.seed),
            "--out_json",
            str(output),
        ]
        if args.downstream_manifest:
            command.extend(["--manifest", str(args.downstream_manifest)])
        else:
            command.extend(["--preset", str(args.downstream_preset)])
        record = {"checkpoint": str(checkpoint), "output": str(output), "command": command, "status": "running"}
        records.append(record)
        _write_json(manifest_path, manifest)
        try:
            _run_command(command)
            record.update({"status": "completed", "completed_at": _utc_now()})
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
            raise
        finally:
            _write_json(manifest_path, manifest)


def _summary_stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    result: dict[str, float | int] = {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        result["sample_std"] = statistics.stdev(values)
    return result


def _write_aggregate_report(root: Path) -> Path:
    """Aggregate strict, cross-tokenizer metrics without choosing a winner."""

    groups: dict[tuple[str, str, str, int, str, str, float | None], list[dict[str, Any]]] = {}
    for result_path in root.rglob("research_result.json"):
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
            metrics_path = Path(record["strict_evaluation"])
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            masked = metrics.get("masked_token", {})
            key = (
                str(record["study"]),
                str(record["backbone"]),
                str(record["tokenizer_type"]),
                int(record["max_length"]),
                str(record["split_manifest"]),
                str(record.get("mask_replacement_strategy", "bert_80_10_10")),
                float(record["mask_fraction"]) if record.get("mask_fraction") is not None else None,
            )
            groups.setdefault(key, []).append(
                {
                    "name": record["name"],
                    "seed": record["seed"],
                    "checkpoint": record["checkpoint"],
                    "parameters_m": record.get("parameters_m"),
                    "bits_per_base": masked.get("bits_per_base"),
                    "conditional_valid_bits_per_base": masked.get("conditional_valid_bits_per_base"),
                    "accuracy": masked.get("acc"),
                    "baselines": masked.get("baselines"),
                    "budget": record.get("budget"),
                    "strict_learning_curve": record.get("strict_learning_curve"),
                    "peak_allocated_mib": (record.get("checkpoint_cuda_memory_mib") or {}).get("peak_allocated_mib"),
                    "peak_reserved_mib": (record.get("checkpoint_cuda_memory_mib") or {}).get("peak_reserved_mib"),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue

    summaries = []
    for key, rows in sorted(groups.items()):
        study, backbone, tokenizer, context, split, replacement_strategy, mask_fraction = key

        def numeric(name: str, current_rows: list[dict[str, Any]] = rows) -> list[float]:
            return [float(row[name]) for row in current_rows if isinstance(row.get(name), (int, float))]

        summaries.append(
            {
                "study": study,
                "backbone": backbone,
                "tokenizer_type": tokenizer,
                "max_length": context,
                "split_manifest": split,
                "mask_replacement_strategy": replacement_strategy,
                "mask_fraction": mask_fraction,
                "bits_per_base": _summary_stats(numeric("bits_per_base")),
                "conditional_valid_bits_per_base": _summary_stats(numeric("conditional_valid_bits_per_base")),
                "masked_accuracy": _summary_stats(numeric("accuracy")),
                "parameters_m": _summary_stats(numeric("parameters_m")),
                "peak_allocated_mib": _summary_stats(numeric("peak_allocated_mib")),
                "peak_reserved_mib": _summary_stats(numeric("peak_reserved_mib")),
                "runs": rows,
            }
        )
    output = root / "aggregate_report.json"
    _write_json(output, {"generated_at": _utc_now(), "groups": summaries})
    return output


def _normalise_stages(raw: list[str]) -> list[str]:
    stages = list(STAGES) if "all" in raw else list(raw)
    unknown = sorted(set(stages).difference(STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    return list(dict.fromkeys(stages))


def _validate_inputs(args: argparse.Namespace, stages: list[str]) -> None:
    args.fasta_files = _expand_fastas(args.fasta)
    if not args.fasta_files:
        raise FileNotFoundError(f"No FASTA files matched: {args.fasta}")
    for path in (*args.fasta_files, args.assembly_split):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required corpus artifact: {path}")
    requires_bpe = any(stage in stages for stage in ("tokenizer", "tokenizer_capacity", "factorial", "context"))
    requires_bpe = requires_bpe or ("masking" in stages and args.masking_tokenizer == "bpe")
    if requires_bpe and not args.tokenizer_path.is_file():
        raise FileNotFoundError(f"BPE tokenizer missing: {args.tokenizer_path}; run scripts/train_bpe.py first")
    if "family" in stages:
        for path in (args.selection_manifest, args.dataset_manifest):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Family-held-out split requires: {path}")
    if args.run and "downstream" in stages and not (args.downstream_manifest or args.downstream_preset):
        raise ValueError(
            "The downstream stage requires --downstream-manifest or --downstream-preset before GPU work starts"
        )
    if (
        args.tokenizer_steps <= 0
        or args.masking_steps <= 0
        or args.architecture_steps <= 0
        or args.factorial_steps <= 0
    ):
        raise ValueError("All step budgets must be positive")
    if args.eval_workers < 0:
        raise ValueError("--eval-workers must be non-negative")
    if args.eval_max_batches <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Research-suite evaluation batches and batch size must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.baseline_alpha <= 0 or args.conclusion_step_threshold <= 0:
        raise ValueError("Baseline alpha and conclusion step threshold must be positive")
    if any(context <= 0 for context in args.contexts):
        raise ValueError("All context lengths must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run staged, isolated True DNA research studies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument("--stages", nargs="+", choices=(*STAGES, "all"), default=["tokenizer"])
    parser.add_argument(
        "--run", action="store_true", help="Execute the planned commands; otherwise only write a manifest"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B; useful for a local smoke run")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--retry-interrupted",
        action="store_true",
        help="Archive incomplete run directories and restart those conditions fresh instead of skipping them",
    )
    parser.add_argument("--output-root", type=Path, default=Path("experiments/research_suite_v2"))
    parser.add_argument(
        "--fasta",
        nargs="+",
        default=["data/domain_fastas/archaea.fa", "data/domain_fastas/bacteria.fa", "data/domain_fastas/eukaryotes.fa"],
    )
    parser.add_argument("--assembly-split", type=Path, default=Path("data/domain_fastas/split_manifest.jsonl"))
    parser.add_argument("--selection-manifest", type=Path, default=Path("data/metadata/taxonomic_selection_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("data/domain_fastas/dataset_manifest.json"))
    parser.add_argument("--tokenizer-path", type=Path, default=Path("dna_model/bpe_vocab/tokenizer.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])

    parser.add_argument("--tokenizer-steps", type=int, default=5_000)
    parser.add_argument("--masking-steps", type=int, default=5_000)
    parser.add_argument("--architecture-steps", type=int, default=10_000)
    parser.add_argument("--factorial-steps", type=int, default=5_000)
    parser.add_argument("--family-steps", type=int, default=20_000)
    parser.add_argument("--context-steps", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-max-batches", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-sample-seed", type=int, default=43)
    parser.add_argument("--mask-seed", type=int, default=43)
    parser.add_argument(
        "--mask-coordinate-system",
        choices=["base", "token"],
        default="base",
        help="The v2 tokenizer protocol samples corruption spans in raw base coordinates",
    )
    parser.add_argument(
        "--mask-replacement-strategy",
        choices=["mask", "bert_80_10_10"],
        default="mask",
        help="Primary v2 training replacement policy; masking-stage jobs override this per condition",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Primary comparisons disable smoothing; nonzero values smooth over DNA-bearing targets only",
    )
    parser.add_argument("--baseline-alpha", type=float, default=1.0)
    parser.add_argument(
        "--strict-learning-curve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strictly evaluate archival checkpoints on the shared evaluation sample",
    )
    parser.add_argument(
        "--conclusion-step-threshold",
        type=int,
        default=20_000,
        help="Runs below this fixed budget are explicitly labeled pilot results",
    )
    parser.add_argument(
        "--compile-transformer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile Transformer jobs with torch.compile; BiMamba/Dual-Helix remain eager by default",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode for Transformer jobs; default avoids CUDA-Graph output reuse in the RC objective",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        help="Standalone strict-evaluation workers; 0 avoids WSL worker-shutdown hangs",
    )
    parser.add_argument("--rc-check-seqs", type=int, default=100)

    parser.add_argument("--architecture-backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--masking-seeds", nargs="+", type=int, default=[43])
    parser.add_argument("--masking-backbone", choices=BACKBONES, default="transformer")
    parser.add_argument("--masking-tokenizer", choices=["base", "bpe"], default="base")
    parser.add_argument("--factorial-backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--family-backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--context-backbones", nargs="+", choices=BACKBONES, default=["transformer"])
    parser.add_argument("--context-tokenizer", choices=["base", "bpe"], default="base")
    parser.add_argument("--contexts", nargs="+", type=int, default=None)

    parser.add_argument("--capacity-target-m", type=float, default=14.2)
    parser.add_argument("--capacity-hidden-sizes", nargs="+", type=int, default=[128, 192, 256, 320, 384, 448, 512])
    parser.add_argument("--family-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--family-seed", type=int, default=43)
    parser.add_argument("--rebuild-family-split", action="store_true")
    parser.add_argument("--downstream-manifest", type=Path, default=None)
    parser.add_argument("--downstream-preset", default=None, help="A preset accepted by scripts/benchmark_genomic.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_DIR)
    args.output_root = args.output_root.expanduser().resolve()
    for attribute in ("assembly_split", "selection_manifest", "dataset_manifest", "tokenizer_path"):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    if args.downstream_manifest is not None:
        args.downstream_manifest = args.downstream_manifest.expanduser().resolve()
    stages = _normalise_stages(args.stages)
    if args.contexts is None:
        args.contexts = [512, 1024] if args.profile == "rtx5080_16gb" else [1024, 2048, 4096]
    _validate_inputs(args, stages)

    protocol_dir = args.output_root / "protocol"
    family_output = protocol_dir / "family_holdout"
    family_split = family_output / "family_heldout_split.jsonl"
    jobs = _build_jobs(args, stages, family_split)
    capacity: dict[str, dict[str, float | int]] = {}
    if any(job.capacity_matched for job in jobs):
        # Capacity matching is part of planning: the manifest users inspect
        # must contain the same concrete widths and commands that --run uses.
        capacity = _capacity_settings(args, jobs)
    manifest_path = args.output_root / "suite_manifest.json"
    source_snapshot = _source_snapshot(protocol_dir)
    baseline_models = [
        {
            "split_manifest": split,
            "output": str(_baseline_model_path(args, split)),
            "command": _baseline_command(args, split, _baseline_model_path(args, split)),
            "status": "planned",
        }
        for split in sorted({job.split_manifest for job in jobs})
    ]
    inputs = _protocol_input_fingerprints(args, jobs)
    job_records = []
    for job in jobs:
        config = _resolved_config(args, job, capacity)
        job_records.append(
            {
                **asdict(job),
                "status": "planned",
                "budget": _budget_summary(args, job),
                "source_snapshot_sha256": source_snapshot["archive_sha256"],
                "source_snapshot_archive": source_snapshot["archive"],
                "evaluation_sample": str(_eval_sample_path(args, job)),
                "baseline_model": str(_baseline_model_path(args, job.split_manifest)),
                "config": config,
                "command": config_to_command(config),
                "protocol_fingerprint": _job_protocol_fingerprint(
                    args,
                    job,
                    config,
                    source_snapshot_sha256=source_snapshot["archive_sha256"],
                    inputs=inputs,
                    capacity=capacity,
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "protocol": "true-dna-research-v2",
        "created_at": _utc_now(),
        "mode": "run" if args.run else "plan",
        "profile": args.profile,
        "stages": stages,
        "seeds": args.seeds,
        "inputs": inputs,
        "source_snapshot": source_snapshot,
        "capacity_matching": capacity,
        "evaluation_protocol": {
            "sampling": "deterministic_domain_and_group_stratified_raw_windows",
            "eval_sample_seed": args.eval_sample_seed,
            "mask_coordinate_system": args.mask_coordinate_system,
            "mask_seed": args.mask_seed,
            "training_replacement_default": args.mask_replacement_strategy,
            "strict_masked_targets": True,
            "label_smoothing": args.label_smoothing,
            "baselines": ["uniform", "global_unigram", "global_first_order", "domain_unigram", "domain_first_order"],
            "masking_pilot_conditions": [
                {"replacement": "bert_80_10_10", "mask_fraction": 0.15},
                {"replacement": "mask", "mask_fraction": 0.15},
                {"replacement": "mask", "mask_fraction": 0.25},
                {"replacement": "mask", "mask_fraction": 0.35},
            ],
        },
        "baseline_models": baseline_models,
        "family_split": str(family_split),
        "jobs": job_records,
    }
    if "family" in stages:
        manifest["family_audit_command"] = _family_audit_command(args, family_output)
    _write_json(manifest_path, manifest)
    print(f"Wrote research suite manifest: {manifest_path}")
    print(f"Planned {len(jobs)} fresh training runs across stages: {', '.join(stages)}")
    if not args.run:
        print("Plan only. Inspect the manifest, then re-run with --run to start GPU work.")
        return 0

    if "family" in stages and (args.rebuild_family_split or not family_split.is_file()):
        _run_command(_family_audit_command(args, family_output))
        if not family_split.is_file():
            raise RuntimeError(f"Family audit did not produce expected split: {family_split}")
        inputs = _protocol_input_fingerprints(args, jobs)
        manifest["inputs"] = inputs
        for job, record in zip(jobs, manifest["jobs"], strict=True):
            record["protocol_fingerprint"] = _job_protocol_fingerprint(
                args,
                job,
                record["config"],
                source_snapshot_sha256=source_snapshot["archive_sha256"],
                inputs=inputs,
                capacity=capacity,
            )
        _write_json(manifest_path, manifest)

    for baseline_record in manifest["baseline_models"]:
        output = Path(baseline_record["output"])
        try:
            if not output.is_file():
                _run_command(list(baseline_record["command"]))
            baseline_record.update(
                {
                    "status": "completed",
                    "sha256": _sha256(output),
                    "completed_at": _utc_now(),
                }
            )
        except Exception as exc:
            baseline_record.update({"status": "failed", "error": str(exc), "failed_at": _utc_now()})
            _write_json(manifest_path, manifest)
            raise
    _write_json(manifest_path, manifest)

    for job in jobs:
        try:
            _run_job(args, job, capacity, manifest, manifest_path)
        except Exception as exc:
            print(f"[error] {job.name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1

    try:
        if "robustness" in stages:
            _run_completed_strict_checks(args, manifest, manifest_path)
        if "downstream" in stages:
            _run_downstream_benchmarks(args, manifest, manifest_path)
    except Exception as exc:
        print(f"[error] follow-up evaluation: {exc}", file=sys.stderr)
        return 1

    manifest["completed_at"] = _utc_now()
    manifest["aggregate_report"] = str(_write_aggregate_report(args.output_root))
    _write_json(manifest_path, manifest)
    print(f"Research suite complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark and evaluation logging helpers.

The training loop and post-training benchmark runners both use these helpers to
keep local JSON/JSONL records as the source of truth while optionally mirroring
scalar summaries to W&B.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]


def _json_safe_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar value, or ``None`` when it is not scalar-like."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe_scalar(value.item())
        except Exception:
            return None
    return None


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON record to a JSONL file.

    Args:
        path: Output JSONL path.
        record: JSON-serializable mapping to append.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def build_eval_history_record(
    *,
    step: int,
    epoch: int,
    eval_metrics: Mapping[str, Any],
    timestamp: float | None = None,
) -> JsonDict:
    """Build the canonical lightweight eval-history record.

    Args:
        step: Global optimizer step.
        epoch: Zero-based epoch index.
        eval_metrics: Output from ``dna_model.utils.evaluate``.
        timestamp: Optional UNIX timestamp. Defaults to current time.

    Returns:
        A JSON-safe record suitable for ``eval_history.jsonl``.
    """
    loss = _json_safe_scalar(eval_metrics.get("loss"))
    acc = _json_safe_scalar(eval_metrics.get("acc", eval_metrics.get("accuracy")))
    record = {
        "step": int(step),
        "epoch": int(epoch),
        "loss": loss,
        "eval_loss": loss,
        "ppl": _json_safe_scalar(eval_metrics.get("ppl")),
        "acc": acc,
        "eval_acc": acc,
        "ece": _json_safe_scalar(eval_metrics.get("ece")),
        "evaluated_batches": _json_safe_scalar(eval_metrics.get("evaluated_batches")),
        "eval_max_batches": _json_safe_scalar(eval_metrics.get("max_batches")),
        "timestamp": float(time.time() if timestamp is None else timestamp),
    }
    for key in ("tokenizer_type", "metric_granularity", "mask_coordinate_system", "mask_replacement"):
        if eval_metrics.get(key) is not None:
            record[key] = eval_metrics[key]
    for key in (
        "base_loss",
        "base_normalized_loss",
        "bits_per_base",
        "base_perplexity",
        "masked_base_fraction",
        "mean_valid_target_probability_mass",
        "conditional_valid_token_loss",
        "conditional_valid_bits_per_base",
        "conditional_acgt_accuracy",
        "conditional_acgt_correct",
        "conditional_acgt_targets",
        "special_token_argmax_rate",
        "special_token_argmax_count",
    ):
        if eval_metrics.get(key) is not None:
            record[key] = _json_safe_scalar(eval_metrics.get(key))
    return record


def build_eval_log_payload(record: Mapping[str, Any], eval_metrics: Mapping[str, Any]) -> JsonDict:
    """Build scalar eval metrics for W&B/TensorBoard.

    Args:
        record: Canonical eval-history record from ``build_eval_history_record``.
        eval_metrics: Full evaluator metrics, including optional per-base metrics.

    Returns:
        Flat scalar mapping using stable ``eval/...`` metric names.
    """
    payload: JsonDict = {
        "global_step": record.get("step"),
        "epoch": record.get("epoch"),
        "eval/loss": record.get("eval_loss"),
        "eval/ppl": record.get("ppl"),
        "eval/acc": record.get("eval_acc"),
        "eval/ece": record.get("ece"),
        "eval/evaluated_batches": record.get("evaluated_batches"),
        "eval/max_batches": record.get("eval_max_batches"),
    }
    for key in (
        "base_accuracy",
        "token_accuracy",
        "base_loss",
        "token_loss",
        "base_normalized_loss",
        "base_perplexity",
        "token_perplexity",
        "bits_per_base",
        "base_ece",
        "token_ece",
        "masked_bases",
        "masked_tokens",
        "masked_base_fraction",
        "mean_valid_target_probability_mass",
        "conditional_valid_token_loss",
        "conditional_valid_bits_per_base",
        "conditional_acgt_accuracy",
        "conditional_acgt_correct",
        "conditional_acgt_targets",
        "special_token_argmax_rate",
        "special_token_argmax_count",
        "accuracy_A",
        "accuracy_C",
        "accuracy_G",
        "accuracy_T",
        "support_A",
        "support_C",
        "support_G",
        "support_T",
    ):
        if key in eval_metrics:
            payload[f"eval/{key}"] = _json_safe_scalar(eval_metrics.get(key))
    baselines = eval_metrics.get("baselines")
    if isinstance(baselines, Mapping):
        payload.update(flatten_numeric_metrics(baselines, prefix="eval/baselines"))
    return {k: v for k, v in payload.items() if _json_safe_scalar(v) is not None}


def log_eval_to_tensorboard(writer: Any, payload: Mapping[str, Any], step: int) -> None:
    """Write scalar eval metrics to a TensorBoard writer."""
    if writer is None:
        return
    for key, value in payload.items():
        if key in {"global_step", "epoch"}:
            continue
        scalar = _json_safe_scalar(value)
        if isinstance(scalar, (int, float)):
            writer.add_scalar(key, scalar, int(step))
    try:
        writer.flush()
    except Exception:
        pass


def _sanitize_key_part(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9_.@+/-]+", "_", text)
    text = re.sub(r"/+", "/", text).strip("/")
    return text or "unnamed"


def flatten_numeric_metrics(
    payload: Mapping[str, Any],
    *,
    prefix: str = "benchmark",
    max_depth: int = 8,
) -> dict[str, float | int]:
    """Flatten nested numeric benchmark metrics into slash-delimited keys."""
    out: dict[str, float | int] = {}

    def visit(value: Any, parts: list[str], depth: int) -> None:
        if depth > max_depth:
            return
        scalar = _json_safe_scalar(value)
        if isinstance(scalar, bool):
            return
        if isinstance(scalar, int):
            out["/".join(parts)] = scalar
            return
        if isinstance(scalar, float):
            if math.isfinite(scalar):
                out["/".join(parts)] = scalar
            return
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                if child_key in {"checkpoint", "supported_manifest_sections"}:
                    continue
                visit(child_value, [*parts, _sanitize_key_part(child_key)], depth + 1)

    visit(payload, [_sanitize_key_part(prefix)], 0)
    return out


def infer_checkpoint_step(checkpoint: str | Path, metadata: Mapping[str, Any] | None = None) -> int | None:
    """Infer a checkpoint step from metadata or a checkpoint filename."""
    if metadata is not None:
        step = _json_safe_scalar(metadata.get("global_step"))
        if isinstance(step, int):
            return step
    name = Path(checkpoint).name
    match = re.search(r"checkpoint_step_?(\d+)\.pt$", name)
    if match:
        return int(match.group(1))
    return None


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object from disk."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_checkpoint_meta(checkpoint: str | Path) -> JsonDict:
    meta_path = Path(str(checkpoint) + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return read_json(meta_path)
    except Exception:
        return {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def build_benchmark_record(
    *,
    run_dir: str | Path,
    checkpoint: str | Path,
    result_json: str | Path,
    label: str,
    benchmark_type: str,
    results: Mapping[str, Any],
    step: int | None = None,
    timestamp: float | None = None,
) -> JsonDict:
    """Build a canonical benchmark-history record."""
    checkpoint_path = Path(checkpoint)
    result_path = Path(result_json)
    meta = _load_checkpoint_meta(checkpoint_path)
    inferred_step = step if step is not None else infer_checkpoint_step(checkpoint_path, meta)
    metrics = flatten_numeric_metrics(results, prefix=f"benchmark/{benchmark_type}/{label}")
    return {
        "schema_version": 1,
        "benchmark_type": str(benchmark_type),
        "label": str(label),
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "result_json": str(result_path),
        "global_step": inferred_step,
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "run_dir": str(Path(run_dir)),
        "metrics": metrics,
    }


def update_checkpoint_benchmark_meta(checkpoint: str | Path, record: Mapping[str, Any]) -> None:
    """Attach a compact benchmark summary to a checkpoint sidecar metadata file."""
    checkpoint_path = Path(checkpoint)
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    meta: MutableMapping[str, Any] = _load_checkpoint_meta(checkpoint_path)
    if not meta:
        meta = {
            "epoch": 0,
            "global_step": record.get("global_step"),
            "best_loss": float("inf"),
        }

    summary = {
        "benchmark_type": record.get("benchmark_type"),
        "label": record.get("label"),
        "result_json": record.get("result_json"),
        "global_step": record.get("global_step"),
        "timestamp": record.get("timestamp"),
        "metrics": dict(record.get("metrics") or {}),
    }

    existing = [
        item
        for item in meta.get("benchmarks", [])
        if not (item.get("benchmark_type") == summary["benchmark_type"] and item.get("label") == summary["label"])
    ]
    existing.append(summary)
    meta["benchmarks"] = existing

    metrics = meta.setdefault("metrics", {})
    if isinstance(metrics, MutableMapping):
        bench_metrics = metrics.setdefault("benchmarks", {})
        if isinstance(bench_metrics, MutableMapping):
            bench_metrics[f"{summary['benchmark_type']}/{summary['label']}"] = summary["metrics"]

    _atomic_write_json(meta_path, meta)


def log_benchmark_record_to_wandb(record: Mapping[str, Any], wandb_module: Any) -> None:
    """Log one benchmark record to an active W&B run."""
    if wandb_module is None or getattr(wandb_module, "run", None) is None:
        return
    try:
        wandb_module.define_metric("benchmark/global_step")
        wandb_module.define_metric("benchmark/*", step_metric="benchmark/global_step")
    except Exception:
        pass
    metrics = dict(record.get("metrics") or {})
    step = record.get("global_step")
    payload: JsonDict = {
        **metrics,
        "benchmark/global_step": step,
        "benchmark/timestamp": record.get("timestamp"),
    }
    try:
        table = wandb_module.Table(columns=["benchmark_type", "label", "checkpoint", "metric", "value"])
        for key, value in sorted(metrics.items()):
            table.add_data(
                record.get("benchmark_type"),
                record.get("label"),
                record.get("checkpoint_name"),
                key,
                value,
            )
        payload["benchmark/table"] = table
    except Exception:
        pass
    try:
        wandb_module.log(payload)
    except Exception:
        pass


def record_benchmark_result(
    *,
    run_dir: str | Path,
    checkpoint: str | Path,
    result_json: str | Path,
    label: str,
    benchmark_type: str,
    step: int | None = None,
    wandb_module: Any | None = None,
) -> JsonDict:
    """Append benchmark history, enrich checkpoint sidecar, and optionally log W&B."""
    results = read_json(result_json)
    record = build_benchmark_record(
        run_dir=run_dir,
        checkpoint=checkpoint,
        result_json=result_json,
        label=label,
        benchmark_type=benchmark_type,
        results=results,
        step=step,
    )
    append_jsonl(Path(run_dir) / "benchmark_history.jsonl", record)
    update_checkpoint_benchmark_meta(checkpoint, record)
    log_benchmark_record_to_wandb(record, wandb_module)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a benchmark result in local history and checkpoint metadata.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--benchmark_type", default="genomic")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="dna-language-model")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_id", default=None)
    args = parser.parse_args(argv)

    wandb_module = None
    if args.wandb:
        import wandb

        wandb_module = wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None,
            config={
                "run_dir": args.run_dir,
                "benchmark_type": args.benchmark_type,
            },
        )

    record = record_benchmark_result(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        result_json=args.result_json,
        label=args.label,
        benchmark_type=args.benchmark_type,
        step=args.step,
        wandb_module=wandb_module,
    )

    if wandb_module is not None:
        wandb_module.finish()

    print(json.dumps({"recorded": True, "label": record["label"], "metrics": len(record["metrics"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

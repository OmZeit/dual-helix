#!/usr/bin/env python3
"""Validate a short training preflight run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_history(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_meta(run_dir: Path) -> dict:
    candidates = [
        run_dir / "checkpoint_latest.pt.meta.json",
        run_dir / "checkpoint_best.pt.meta.json",
    ]
    candidates.extend(sorted(run_dir.glob("checkpoint_step*.pt.meta.json")))
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No checkpoint metadata found in {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check preflight loss, metadata, and cache policy")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-steps-per-epoch", type=int, required=True)
    parser.add_argument("--min-evals", type=int, default=2)
    parser.add_argument("--forbid-npy-cache", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "eval_history.jsonl"
    if not history_path.exists():
        raise SystemExit(f"Missing eval history: {history_path}")

    rows = _read_history(history_path)
    if len(rows) < args.min_evals:
        raise SystemExit(f"Expected at least {args.min_evals} evals, found {len(rows)}")

    first_loss = float(rows[0]["eval_loss"])
    last_loss = float(rows[-1]["eval_loss"])
    if not math.isfinite(first_loss) or not math.isfinite(last_loss):
        raise SystemExit("Eval loss is not finite")
    if last_loss > first_loss:
        raise SystemExit(f"Preflight loss did not improve: {first_loss:.4f} -> {last_loss:.4f}")
    for row in rows:
        evaluated = row.get("evaluated_batches")
        capped_at = row.get("eval_max_batches")
        if evaluated is None:
            continue
        evaluated_i = int(evaluated)
        if evaluated_i <= 0:
            raise SystemExit("Preflight eval recorded zero batches")
        if capped_at is not None and int(capped_at) > 0 and evaluated_i > int(capped_at):
            raise SystemExit(f"Eval exceeded cap: evaluated_batches={evaluated_i}, eval_max_batches={int(capped_at)}")

    meta = _load_meta(run_dir)
    metrics = meta.get("metrics", {})
    ckpt = metrics.get("checkpoint", {})
    actual_steps = int(ckpt.get("steps_per_epoch", -1))
    if actual_steps != args.expected_steps_per_epoch:
        raise SystemExit(f"steps_per_epoch mismatch: expected {args.expected_steps_per_epoch}, got {actual_steps}")

    if args.forbid_npy_cache:
        log_path = run_dir / "training.log"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if "Found pre-tokenized arrays. Using NpyDataset." in text:
                raise SystemExit("Preflight used stale NPY cache despite raw-sequence augmentations")

    print(f"[preflight] ok: loss {first_loss:.4f} -> {last_loss:.4f}, steps_per_epoch={actual_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

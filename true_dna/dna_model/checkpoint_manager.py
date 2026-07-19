#!/usr/bin/env python3
"""Inspect, select, average, and clean training checkpoints.

Features:
- Checkpoint loading by best metric, latest step, or explicit path
- Checkpoint averaging
- Checkpoint inspection and comparison
- Automatic backup and cleanup
- Resume training with validation

Usage:
    # List available checkpoints
    python checkpoint_manager.py --list --run-dir runs/exp1

    # Inspect a checkpoint
    python checkpoint_manager.py --inspect runs/exp1/checkpoint_best.pt

    # Compare checkpoints
    python checkpoint_manager.py --compare runs/exp1/checkpoint_best.pt runs/exp1/checkpoint_latest.pt

    # Average multiple checkpoints
    python checkpoint_manager.py --average runs/exp1/checkpoint_*.pt --output averaged_model.pt

    # Clean old checkpoints (keep best + latest + last N)
    python checkpoint_manager.py --clean runs/exp1 --keep 5
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch


def _extract_display_loss(info: dict) -> float | None:
    """
    Returns a float loss for ranking, or None if not available.
    Priority:
      1) sidecar metrics.eval_loss
      2) sidecar metrics.best_loss_so_far
      3) checkpoint best_loss (if finite)
    """
    m = info.get("metrics") or {}
    nested_eval = m.get("eval") if isinstance(m, dict) else None
    v = None
    if isinstance(nested_eval, dict):
        v = nested_eval.get("loss")
    if v is None and isinstance(m, dict):
        v = m.get("eval_loss")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    nested_best = m.get("best") if isinstance(m, dict) else None
    v = nested_best.get("eval_loss") if isinstance(nested_best, dict) else None
    if v is None and isinstance(m, dict):
        v = m.get("best_loss_so_far")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    v = info.get("best_loss")
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _read_eval_history(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "eval_history.jsonl")
    if not os.path.exists(path):
        return []
    recs: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    recs.sort(key=lambda r: r.get("step", -1))
    return recs


class CheckpointManager:
    """Manage training checkpoints with advanced features."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.checkpoints_cache = None

    def list_checkpoints(self, sort_by: str = "step") -> list[dict]:
        """
        List all checkpoints with metadata.

        Args:
            sort_by: 'step', 'epoch', 'time', 'loss'

        Returns:
            List of checkpoint info dicts
        """
        if not self.run_dir.exists():
            return []

        checkpoints: list[dict] = []

        # Find all checkpoint files
        for ckpt_path in self.run_dir.glob("checkpoint_*.pt"):
            try:
                info = self._get_checkpoint_info(ckpt_path)
                checkpoints.append(info)
            except Exception as e:
                print(f"Warning: Could not load {ckpt_path.name}: {e}")

        # Sort by requested key
        if sort_by == "step":
            checkpoints.sort(key=lambda x: x.get("global_step", 0))
        elif sort_by == "epoch":
            checkpoints.sort(key=lambda x: x.get("epoch", 0))
        elif sort_by == "time":
            checkpoints.sort(key=lambda x: x.get("modified_time", 0))
        elif sort_by == "loss":
            checkpoints.sort(
                key=lambda i: (
                    float("inf") if _extract_display_loss(i) is None else _extract_display_loss(i),
                    i.get("global_step", 0),
                )
            )

        return checkpoints

    def save_checkpoint(self, save_dict: dict, path: Path, is_latest: bool = False):
        """
        Save checkpoint with optional optimizer stripping.

        Args:
            save_dict: Dictionary containing model state, optimizer, etc.
            path: Path to save the checkpoint to.
            is_latest: If True, keeps optimizer/scheduler states.
                       If False (archival), strips them to save space.
        """
        # Create a shallow copy to avoid modifying the original dict in memory
        # if we are going to strip keys.
        final_dict = save_dict.copy()

        if not is_latest:
            # Strip optimizer and scheduler for archival checkpoints
            final_dict.pop("optimizer", None)
            final_dict.pop("scheduler", None)
            # We might want to keep EMA shadow if it exists, as it's part of the model weights essentially
            # But optimizer is the big one (2x params for Adam)

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        torch.save(final_dict, tmp_path)
        os.replace(tmp_path, path)

        # Save sidecar metadata
        try:
            meta = {
                "epoch": final_dict.get("epoch"),
                "global_step": final_dict.get("global_step"),
                "best_loss": final_dict.get("best_loss"),
                "modified_time": datetime.now().timestamp(),
                "modified_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "has_optimizer": is_latest,
            }
            # Add metrics if available in save_dict (custom convention)
            if "metrics" in final_dict:
                meta["metrics"] = final_dict["metrics"]

            meta_path = Path(str(path) + ".meta.json")
            tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
            with open(tmp_meta, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp_meta, meta_path)
        except Exception as e:
            print(f"Warning: Failed to save metadata for {path}: {e}")

    def _get_checkpoint_info(self, ckpt_path: Path) -> dict:
        """Extract metadata from a checkpoint without loading full model."""
        stat = ckpt_path.stat()

        # Prefer sidecar metadata JSON if present (fast, safe)
        meta_path = ckpt_path.with_name(ckpt_path.name + ".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as mf:
                    meta = json.load(mf)

                # Build base info from sidecar
                info: dict = {
                    "path": str(ckpt_path),
                    "name": ckpt_path.name,
                    "size_mb": meta.get("size_mb", stat.st_size / (1024 * 1024)),
                    "modified_time": meta.get("modified_time", stat.st_mtime),
                    "modified_date": meta.get(
                        "modified_date", datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    "epoch": meta.get("epoch", 0),
                    "global_step": meta.get("global_step", 0),
                    "best_loss": meta.get("best_loss", float("inf")),
                    "metrics": meta.get("metrics", {}),
                    "steps_per_epoch": meta.get("steps_per_epoch"),
                    "epoch_estimate": meta.get("epoch_estimate"),
                }

                # If metrics are missing or lack an eval_loss, try to backfill from eval_history.jsonl
                metrics = info.get("metrics") or {}
                eval_loss = metrics.get("eval_loss") if isinstance(metrics, dict) else None
                if eval_loss is None:
                    try:
                        hist = _read_eval_history(str(ckpt_path.parent))
                        if hist:
                            # pick latest eval at or before this checkpoint's step
                            step = int(info.get("global_step") or 0)
                            best = None
                            for r in hist:
                                if isinstance(r.get("step"), int) and r["step"] <= step:
                                    best = r
                                else:
                                    break
                            if best:
                                info["metrics"] = {
                                    "eval_loss": best.get("eval_loss", best.get("loss")),
                                    "eval_acc": best.get("eval_acc", best.get("acc")),
                                    "ece": best.get("ece"),
                                    "from_step": best.get("step"),
                                }
                    except Exception:
                        pass

                display_loss = _extract_display_loss(info)
                info["display_loss"] = display_loss

                return info
            except Exception as e:
                print(f"Warning: Failed to read meta file {meta_path.name}: {e}")

        # Fallback: Load checkpoint metadata only. Try weights_only when available
        try:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except TypeError as e:
                raise RuntimeError("Safe checkpoint inspection requires PyTorch with weights_only=True support.") from e
        except Exception as e:
            # If loading fails, return minimal info and don't crash
            print(f"Warning: Failed to load checkpoint {ckpt_path.name}: {e}")
            return {
                "path": str(ckpt_path),
                "name": ckpt_path.name,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified_time": stat.st_mtime,
                "modified_date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": 0,
                "global_step": 0,
                "best_loss": float("inf"),
                "load_failed": True,
            }

        info = {
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
            "modified_date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": ckpt.get("epoch", 0),
            "global_step": ckpt.get("global_step", 0),
            "best_loss": ckpt.get("best_loss", float("inf")),
        }

        # Extract config info if available
        if "config" in ckpt:
            config = ckpt["config"]
            if isinstance(config, dict):
                info["hidden_size"] = config.get("hidden_size", "N/A")
                info["num_layers"] = config.get("high_level_layers", "N/A")
            else:
                info["hidden_size"] = getattr(config, "hidden_size", "N/A")
                info["num_layers"] = getattr(config, "high_level_layers", "N/A")

        # Align with display_loss concept even for fallback loads
        info["display_loss"] = _extract_display_loss(info)

        return info

    def find_best_checkpoint(self) -> Path | None:
        """Find the checkpoint with the lowest available loss metric."""
        infos = self.list_checkpoints()  # no specific sort required
        valid = [i for i in infos if _extract_display_loss(i) is not None]
        best_info = min(valid, key=lambda i: _extract_display_loss(i)) if valid else None

        # Retain an explicit checkpoint_best.pt only when its metric is at least as good.
        ckpt_best = self.run_dir / "checkpoint_best.pt"
        if ckpt_best.exists():
            try:
                info_best_file = self._get_checkpoint_info(ckpt_best)
                file_loss = _extract_display_loss(info_best_file)
                if file_loss is not None:
                    if best_info is None or file_loss <= _extract_display_loss(best_info):
                        return ckpt_best
            except Exception:
                # ignore read errors and fall through
                pass

        return Path(best_info["path"]) if best_info else None

    def find_latest_checkpoint(self) -> Path | None:
        """Find most recent checkpoint by step."""
        checkpoints = self.list_checkpoints(sort_by="step")

        if not checkpoints:
            return None

        # Try explicit latest checkpoint first
        latest_ckpt = self.run_dir / "checkpoint_latest.pt"
        if latest_ckpt.exists():
            return latest_ckpt

        # Otherwise return checkpoint with highest step
        return Path(checkpoints[-1]["path"])

    def find_checkpoint_by_epoch(self, epoch: int) -> Path | None:
        """Find checkpoint for specific epoch."""
        for ckpt in self.list_checkpoints():
            if ckpt.get("epoch") == epoch:
                return Path(ckpt["path"])
        return None

    def inspect_checkpoint(self, ckpt_path: Path, verbose: bool = False) -> dict:
        """
        Detailed inspection of a checkpoint.

        Returns comprehensive information including parameter counts,
        optimizer state, and training history.
        """
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Inspecting: {ckpt_path.name}")
        print("=" * 60)

        # Try to load checkpoint metadata safely
        try:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except TypeError as e:
                raise RuntimeError("Safe checkpoint inspection requires PyTorch with weights_only=True support.") from e
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt_path.name}: {e}")
            return {
                "path": str(ckpt_path),
                "size_mb": ckpt_path.stat().st_size / (1024 * 1024),
                "keys": [],
                "load_failed": True,
            }

        info = {
            "path": str(ckpt_path),
            "size_mb": ckpt_path.stat().st_size / (1024 * 1024),
            "keys": list(ckpt.keys()),
        }

        # Basic training info
        print("\nTraining state:")
        print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  Global Step: {ckpt.get('global_step', 'N/A')}")
        best_loss_val = ckpt.get("best_loss", None)
        # Render infinite or missing best_loss as 'N/A' for clarity
        if best_loss_val is None or (
            isinstance(best_loss_val, (int, float)) and not math.isfinite(float(best_loss_val))
        ):
            print("  Best Loss: N/A")
        else:
            try:
                print(f"  Best Loss: {float(best_loss_val):.6f}")
            except Exception:
                print(f"  Best Loss: {best_loss_val}")

        info["epoch"] = ckpt.get("epoch", 0)
        info["global_step"] = ckpt.get("global_step", 0)
        info["best_loss"] = ckpt.get("best_loss", float("inf"))

        # Model parameters
        if "model" in ckpt:
            model_state = ckpt["model"]
            total_params = 0
            model_bytes = 0
            for v in model_state.values():
                if isinstance(v, torch.Tensor):
                    total_params += int(v.numel())
                    try:
                        model_bytes += int(v.numel() * v.element_size())
                    except Exception:
                        pass
                else:
                    try:
                        total_params += int(len(v))
                    except Exception:
                        pass

            trainable_params = total_params  # cannot infer requires_grad from state_dict

            print("\nModel:")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Model Size: {model_bytes / (1024**2):.2f} MB")
            print(f"  Number of Layers: {len(model_state)}")

            info["total_params"] = total_params
            info["num_layers"] = len(model_state)
            info["trainable_params"] = trainable_params

            if verbose:
                print("\n  Layer Details:")
                for name, param in list(model_state.items())[:10]:  # First 10 layers
                    if isinstance(param, torch.Tensor):
                        shape = tuple(param.shape)
                        nparams = param.numel()
                    else:
                        try:
                            shape = (len(param),)
                        except Exception:
                            shape = ("?",)
                        try:
                            nparams = len(param)
                        except Exception:
                            nparams = 0
                    print(f"    {name}: {shape} ({nparams:,} params)")
                if len(model_state) > 10:
                    print(f"    ... and {len(model_state) - 10} more layers")

        # Optimizer state
        if "optimizer" in ckpt:
            opt_state = ckpt["optimizer"]
            print("\nOptimizer:")
            print(f"  State dict keys: {list(opt_state.keys())}")
            if "param_groups" in opt_state:
                for i, pg in enumerate(opt_state["param_groups"]):
                    lr = pg.get("lr")
                    lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)
                    print(f"  Group {i}: LR={lr_str}, Weight Decay={pg.get('weight_decay', 'N/A')}")

        # Scheduler state
        if "scheduler" in ckpt and ckpt["scheduler"]:
            print("\nScheduler: present")

        # EMA state
        if "ema_shadow" in ckpt and ckpt["ema_shadow"]:
            try:
                ema_params = sum(p.numel() for p in ckpt["ema_shadow"].values() if isinstance(p, torch.Tensor))
            except Exception:
                ema_params = 0
            print(f"\nEMA: present ({ema_params:,} parameters)")
            info["has_ema"] = True
        else:
            info["has_ema"] = False

        # Config
        if "config" in ckpt:
            config = ckpt["config"]
            print("\nConfiguration:")
            if isinstance(config, dict):
                for key, value in config.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {config}")
            info["config"] = config if isinstance(config, dict) else str(config)

        print("\n" + "=" * 60)
        return info

    def compare_checkpoints(self, ckpt_paths: list[Path]) -> None:
        """Compare multiple checkpoints side by side."""
        if len(ckpt_paths) < 2:
            print("Need at least 2 checkpoints to compare")
            return

        print("Checkpoint Comparison")
        print("=" * 80)

        infos = []
        for path in ckpt_paths:
            try:
                info = self._get_checkpoint_info(path)
                infos.append(info)
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not infos:
            return

        headers = ["Metric"] + [f"Ckpt {i + 1}" for i in range(len(infos))]

        print(f"\n{'Metric':<20} " + " ".join([f"{h:>15}" for h in headers[1:]]))
        print("-" * 80)

        # Compare key metrics (use display_loss instead of raw best_loss)
        metrics = ["name", "epoch", "global_step", "display_loss", "size_mb"]

        for metric in metrics:
            values = []
            for info in infos:
                v = info.get(metric, "N/A")
                if metric == "display_loss":
                    loss = _extract_display_loss(info)
                    values.append("N/A" if loss is None else f"{loss:.6f}")
                elif metric in ("epoch", "global_step"):
                    values.append(str(v if v is not None else "N/A"))
                elif metric == "size_mb":
                    try:
                        values.append(f"{float(v):.2f}")
                    except Exception:
                        values.append(str(v))
                else:
                    values.append(str(v))

            # Highlight best value for loss (lower is better)
            if metric == "display_loss":
                try:
                    float_values = [float(v) if v != "N/A" else float("inf") for v in values]
                    best_idx = float_values.index(min(float_values))
                    values = [f"* {v}" if i == best_idx else f"  {v}" for i, v in enumerate(values)]
                except Exception:
                    pass
            # Highlight best value for step (higher is better)
            elif metric == "global_step":
                try:
                    int_values = [int(v) if v != "N/A" else 0 for v in values]
                    best_idx = int_values.index(max(int_values))
                    values = [f"* {v}" if i == best_idx else f"  {v}" for i, v in enumerate(values)]
                except Exception:
                    pass

            print(f"{metric:<20} " + " ".join([f"{v:>15}" for v in values]))

        print("=" * 80)
        print("* indicates best value")

    def average_checkpoints(
        self, ckpt_paths: list[Path], output_path: Path, weights: list[float] | None = None
    ) -> Path:
        """
        Average multiple checkpoints for better generalization.

        This technique (SWA - Stochastic Weight Averaging) often improves
        model performance.

        Args:
            ckpt_paths: List of checkpoint paths to average
            output_path: Where to save averaged checkpoint
            weights: Optional weights for weighted averaging (must sum to 1)

        Returns:
            Path to saved averaged checkpoint
        """
        if len(ckpt_paths) < 2:
            raise ValueError("Need at least 2 checkpoints to average")

        if weights is None:
            weights = [1.0 / len(ckpt_paths)] * len(ckpt_paths)

        if len(weights) != len(ckpt_paths):
            raise ValueError("Number of weights must match number of checkpoints")

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        print(f"Averaging {len(ckpt_paths)} checkpoints...")
        print(f"Weights: {weights}")

        # Load and accumulate in a memory-conscious way on CPU using float64 accumulator
        def _safe_load(path: Path):
            try:
                try:
                    return torch.load(path, map_location="cpu", weights_only=True)
                except TypeError as e:
                    raise RuntimeError(
                        "Safe checkpoint averaging requires PyTorch with weights_only=True support."
                    ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint {path}: {e}") from e

        base_ckpt = _safe_load(ckpt_paths[0])
        averaged_state: dict[str, torch.Tensor] = {}

        # Initialize averaged state as float64 tensors to reduce numeric error
        for key, param in base_ckpt["model"].items():
            if isinstance(param, torch.Tensor):
                averaged_state[key] = param.to(dtype=torch.float64, device="cpu") * float(weights[0])
            else:
                # Non-tensor entries: keep last-seen
                averaged_state[key] = param

        # Add remaining checkpoints
        for i, (path, weight) in enumerate(zip(ckpt_paths[1:], weights[1:], strict=True), 1):
            print(f"  Loading checkpoint {i + 1}/{len(ckpt_paths)}: {path.name}")
            ckpt = _safe_load(path)

            for key, param in ckpt["model"].items():
                if isinstance(param, torch.Tensor):
                    if key not in averaged_state or not isinstance(averaged_state[key], torch.Tensor):
                        averaged_state[key] = param.to(dtype=torch.float64, device="cpu") * float(weight)
                    else:
                        averaged_state[key] += param.to(dtype=torch.float64, device="cpu") * float(weight)
                else:
                    averaged_state[key] = param

        # Optionally cast averaged parameters back to float32 to save space
        out_model: dict[str, torch.Tensor] = {}
        for k, v in averaged_state.items():
            if isinstance(v, torch.Tensor):
                try:
                    out_model[k] = v.to(dtype=torch.float32)
                except Exception:
                    out_model[k] = v
            else:
                out_model[k] = v

        output_ckpt = {
            "model": out_model,
            "epoch": base_ckpt.get("epoch", 0),
            "global_step": base_ckpt.get("global_step", 0),
            "config": base_ckpt.get("config"),
            "averaged_from": [str(p) for p in ckpt_paths],
            "averaging_weights": weights,
        }

        # Save averaged checkpoint
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_ckpt, output_path)

        print(f"Saved averaged checkpoint to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / (1024**2):.2f} MB")

        return output_path

    def clean_old_checkpoints(
        self, keep_n: int = 5, keep_best: bool = True, keep_latest: bool = True, dry_run: bool = False
    ) -> list[Path]:
        """
        Clean up old checkpoints to save disk space.

        Args:
            keep_n: Number of most recent checkpoints to keep
            keep_best: Always keep checkpoint_best.pt
            keep_latest: Always keep checkpoint_latest.pt
            dry_run: If True, only show what would be deleted

        Returns:
            List of deleted (or would-be-deleted) paths
        """
        checkpoints = self.list_checkpoints(sort_by="step")

        protected = set()

        # Protect best and latest
        if keep_best:
            best = self.run_dir / "checkpoint_best.pt"
            if best.exists():
                protected.add(str(best))

        if keep_latest:
            latest = self.run_dir / "checkpoint_latest.pt"
            if latest.exists():
                protected.add(str(latest))

        # Protect last N checkpoints
        if keep_n > 0:
            for ckpt in checkpoints[-keep_n:]:
                protected.add(ckpt["path"])

        # Find checkpoints to delete
        to_delete: list[Path] = []
        for ckpt in checkpoints:
            if ckpt["path"] not in protected:
                to_delete.append(Path(ckpt["path"]))

        if not to_delete:
            print("No checkpoints to clean up.")
            return []

        print(f"{'[DRY RUN] ' if dry_run else ''}Cleaning up {len(to_delete)} checkpoint(s):")

        total_size = 0.0
        for path in to_delete:
            size_mb = path.stat().st_size / (1024**2)
            total_size += size_mb
            print(f"  {'Would delete' if dry_run else 'Deleting'}: {path.name} ({size_mb:.2f} MB)")

            if not dry_run:
                # delete sidecar if present
                meta_path = Path(str(path) + ".meta.json")
                try:
                    if meta_path.exists():
                        meta_path.unlink()
                except Exception:
                    pass
                path.unlink()

        print(f"{'Would free' if dry_run else 'Freed'}: {total_size:.2f} MB")

        if dry_run:
            print("\nRun without --dry-run to delete these files.")

        return to_delete

    def export_checkpoint_summary(self, output_file: Path | None = None) -> dict:
        """Export summary of all checkpoints to JSON."""
        checkpoints = self.list_checkpoints()

        # Use the improved best/latest finders
        best_path = self.find_best_checkpoint()
        latest_path = self.find_latest_checkpoint()

        summary = {
            "run_dir": str(self.run_dir),
            "total_checkpoints": len(checkpoints),
            "total_size_mb": float(sum(c.get("size_mb", 0.0) for c in checkpoints)),
            "checkpoints": checkpoints,
            "best_checkpoint": str(best_path) if best_path else None,
            "latest_checkpoint": str(latest_path) if latest_path else None,
            "exported_at": datetime.now().isoformat(),
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Exported summary to: {output_file}")

        return summary


def load_checkpoint_smart(run_dir: Path, mode: str = "best", device: str = "cpu") -> tuple[Path, dict]:
    """
    Smart checkpoint loading with multiple strategies.

    Args:
        run_dir: Directory containing checkpoints
        mode: 'best', 'latest', 'epoch:N', or explicit path
        device: Device to load checkpoint on

    Returns:
        (checkpoint_path, checkpoint_dict)
    """
    manager = CheckpointManager(run_dir)

    if mode == "best":
        ckpt_path = manager.find_best_checkpoint()
        if not ckpt_path:
            raise FileNotFoundError("No best checkpoint found")
        print(f"Loading best checkpoint: {ckpt_path.name}")

    elif mode == "latest":
        ckpt_path = manager.find_latest_checkpoint()
        if not ckpt_path:
            raise FileNotFoundError("No latest checkpoint found")
        print(f"Loading latest checkpoint: {ckpt_path.name}")

    elif mode.startswith("epoch:"):
        epoch = int(mode.split(":")[1])
        ckpt_path = manager.find_checkpoint_by_epoch(epoch)
        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")
        print(f"Loading checkpoint from epoch {epoch}: {ckpt_path.name}")

    else:
        # Treat as explicit path
        ckpt_path = Path(mode)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading checkpoint: {ckpt_path.name}")

    # Load checkpoint safely (try weights_only when available)
    try:
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError as e:
            raise RuntimeError("Safe checkpoint loading requires PyTorch with weights_only=True support.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}") from e

    # Print info
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Step: {ckpt.get('global_step', 'N/A')}")

    # Prefer display loss if available from sidecar
    try:
        info = manager._get_checkpoint_info(ckpt_path)
        dloss = _extract_display_loss(info)
        if dloss is not None:
            print(f"  Loss: {dloss:.6f}")
        else:
            best_loss_val = ckpt.get("best_loss", None)
            if isinstance(best_loss_val, (int, float)) and math.isfinite(float(best_loss_val)):
                print(f"  Loss: {best_loss_val:.6f}")
            else:
                print("  Loss: N/A")
    except Exception:
        best_loss_val = ckpt.get("best_loss", None)
        if isinstance(best_loss_val, (int, float)) and math.isfinite(float(best_loss_val)):
            print(f"  Loss: {best_loss_val:.6f}")
        else:
            print("  Loss: N/A")

    return ckpt_path, ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint Management Utility")
    parser.add_argument("--run-dir", type=Path, default=Path("runs"), help="Training run directory")

    # Commands
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--inspect", type=Path, help="Inspect a checkpoint")
    parser.add_argument("--compare", nargs="+", type=Path, help="Compare checkpoints")
    parser.add_argument("--average", nargs="+", type=Path, help="Average checkpoints")
    parser.add_argument("--output", type=Path, help="Output path for averaged checkpoint")
    parser.add_argument("--clean", action="store_true", help="Clean old checkpoints")
    parser.add_argument("--export", type=Path, help="Export summary to JSON")

    # Options
    parser.add_argument("--keep", type=int, default=5, help="Number of checkpoints to keep when cleaning")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--sort", choices=["step", "epoch", "time", "loss"], default="step", help="Sort order for listing"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Find run directory
    if args.run_dir.is_dir():
        run_dir = args.run_dir
    else:
        # Try to find latest run
        runs_dir = Path("runs")
        if runs_dir.exists():
            run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
            if run_dirs:
                run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
                print(f"Using latest run: {run_dir.name}\n")
            else:
                print("No run directories found.")
                return 1
        else:
            print(f"Run directory not found: {args.run_dir}")
            return 1

    manager = CheckpointManager(run_dir)

    # Execute commands
    if args.list:
        checkpoints = manager.list_checkpoints(sort_by=args.sort)
        if not checkpoints:
            print("No checkpoints found.")
            return 0

        print(f"Found {len(checkpoints)} checkpoint(s) in {run_dir}:")
        print("=" * 100)
        print(f"{'Name':<30} {'Epoch':>6} {'Step':>8} {'Loss':>12} {'Size (MB)':>10} {'Modified':<20}")
        print("-" * 100)

        for ckpt in checkpoints:
            # Prefer informative epoch (use epoch_estimate if epoch is 0)
            epoch_val = ckpt.get("epoch")
            if (epoch_val is None or int(epoch_val) == 0) and ckpt.get("global_step") and ckpt.get("steps_per_epoch"):
                try:
                    epoch_val = int(ckpt.get("global_step")) // max(1, int(ckpt.get("steps_per_epoch")))
                except Exception:
                    epoch_val = ckpt.get("epoch", 0)

            # Choose best available loss: eval_loss -> best_loss_so_far -> stored best_loss
            loss = _extract_display_loss(ckpt)
            loss_str = "N/A" if loss is None else f"{loss:.6f}"

            print(
                f"{ckpt['name']:<30} {str(epoch_val):>6} {ckpt['global_step']:>8} {loss_str:>12} {ckpt['size_mb']:>10.2f} {ckpt['modified_date']:<20}"
            )

        print("=" * 100)

        # Show best (by displayed loss) and latest
        best_path = manager.find_best_checkpoint()
        latest = manager.find_latest_checkpoint()

        if best_path:
            best_info = manager._get_checkpoint_info(best_path)
            best_loss = _extract_display_loss(best_info)
            from_step = (best_info.get("metrics") or {}).get("from_step")
            extra = f" (loss {best_loss:.4f}" + (f", from step {from_step})" if from_step else ")")
            print(f"\nBest checkpoint:   {best_path.name}{extra}")
        else:
            print("\nBest checkpoint:   N/A")

        print(f"Latest checkpoint: {latest.name if latest else 'N/A'}")

    elif args.inspect:
        manager.inspect_checkpoint(args.inspect, verbose=args.verbose)

    elif args.compare:
        manager.compare_checkpoints(args.compare)

    elif args.average:
        if not args.output:
            args.output = run_dir / "checkpoint_averaged.pt"
        manager.average_checkpoints(args.average, args.output)

    elif args.clean:
        manager.clean_old_checkpoints(keep_n=args.keep, dry_run=args.dry_run)

    elif args.export:
        manager.export_checkpoint_summary(args.export)

    else:
        # Default: show summary
        checkpoints = manager.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
            return 0

        print(f"Checkpoint Summary for: {run_dir.name}")
        print(f"Total: {len(checkpoints)} checkpoints")
        print(f"Total Size: {sum(c.get('size_mb', 0.0) for c in checkpoints):.2f} MB")
        print("\nUse --list to see all checkpoints")
        print("Use --inspect <path> to inspect a specific checkpoint")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

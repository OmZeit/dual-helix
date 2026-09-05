import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


class WeightedDataLoader:
    """Sample batches from multiple loaders according to configured weights."""

    def __init__(
        self,
        loaders: Sequence[DataLoader],
        weights: dict[str, float],
        names: list[str] | None = None,
        steps_per_epoch: int | None = None,
        seed: int | float = 42,
    ):
        self._all_loaders = list(loaders)
        self._all_names = names or [f"file_{i}" for i in range(len(loaders))]
        if len(self._all_names) != len(self._all_loaders):
            raise ValueError("names must contain one entry per loader")
        self._base_seed = int(seed)
        self.rng = np.random.default_rng(self._base_seed)
        self._current_epoch = 0

        # Filter out truly empty loaders first
        loaders_f, names_f = [], []
        for loader, name in zip(self._all_loaders, self._all_names, strict=True):
            if len(loader) > 0:
                loaders_f.append(loader)
                names_f.append(name)
            else:
                LOGGER.warning("[WeightedDataLoader] Skipping empty loader for %s", name)

        if not loaders_f:
            raise ValueError("All data loaders are empty!")

        self.loaders = loaders_f
        self.names = names_f

        # Keep only weights for remaining names, then re-normalize
        missing = [name for name in self.names if name not in weights and "*" not in weights]
        if missing:
            raise ValueError(f"No weight specified for loaders: {missing}")
        default_weight = weights.get("*")
        w = np.array(
            [float(weights.get(name, default_weight)) for name in self.names],
            dtype=np.float64,
        )
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative")
        if not np.any(w > 0):
            raise ValueError("At least one weight must be > 0")
        self.probs = w / w.sum()  # sum to 1.0

        LOGGER.info("[WeightedDataLoader] Normalized sampling probabilities:")
        for name, prob in zip(self.names, self.probs, strict=True):
            LOGGER.info("  %s: %.3f", name, prob)

        # Epoch length
        if steps_per_epoch is not None and steps_per_epoch > 0:
            self.total_len = int(steps_per_epoch)
        else:
            # Use longest loader
            self.total_len = max(len(ld) for ld in self.loaders)

        self._iters = None
        self._step = 0
        self._skip_on_iter = 0

    def set_epoch(self, epoch: int):
        """Reset deterministic sampling state for an epoch."""
        self._current_epoch = int(epoch)
        self.rng = np.random.default_rng(self._base_seed + self._current_epoch)
        for loader in self.loaders:
            ds = getattr(loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
        # Reset iterators for determinism
        if self._iters is not None:
            self._iters = [iter(loader) for loader in self.loaders]
        LOGGER.info("[WeightedDataLoader] Epoch %s set, RNG reseeded", epoch)

    def __len__(self):
        return self.total_len

    def __iter__(self):
        self._iters = [iter(loader) for loader in self.loaders]
        self._step = int(self._skip_on_iter)
        self._skip_on_iter = 0
        return self

    def __next__(self):
        if self._step >= self.total_len:
            raise StopIteration

        # Sample which loader to draw from
        idx = int(self.rng.choice(len(self.loaders), p=self.probs))

        attempts = 0
        max_attempts = len(self.loaders)

        while attempts < max_attempts:
            try:
                batch = next(self._iters[idx])
                if isinstance(batch, dict):
                    batch["domain_name"] = self.names[idx]
                self._step += 1
                return batch
            except StopIteration:
                # Restart this iterator and try again
                try:
                    self._iters[idx] = iter(self.loaders[idx])
                    batch = next(self._iters[idx])
                    if isinstance(batch, dict):
                        batch["domain_name"] = self.names[idx]
                    self._step += 1
                    return batch
                except StopIteration:
                    # This loader is truly empty, try another
                    attempts += 1
                    idx = (idx + 1) % len(self.loaders)
                    continue

        raise StopIteration

    def fast_forward(self, steps: int):
        """Advance sampling state without reading skipped batches."""
        if steps <= 0:
            return

        LOGGER.info("[WeightedDataLoader] Fast-forwarding %s steps (skipping data IO)...", steps)

        _ = self.rng.choice(len(self.loaders), size=steps, p=self.probs)

        self._skip_on_iter += steps
        self._step += steps
        LOGGER.info("[WeightedDataLoader] Fast-forward complete. New step: %s", self._step)


def build_loaders_weighted(
    files: Sequence[str],
    weights: dict[str, float],
    tokenizer,
    *,
    max_length: int,
    stride: int,
    train_split_ratio: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool = True,
    use_reverse_prob: float = 0.0,
    provide_rc: bool = False,
    frameshift_prob: float = 0.0,
    max_chunks_per_file: int | None = None,
    cap_by_file: dict | None = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    steps_per_epoch: int | None = None,
    gc_metadata_file: str | None = None,
    stratify_gc: bool = False,
    split_manifest: str | None = None,
    seed: int = 42,
    return_offsets: bool = False,
):
    """Build record-disjoint loaders with weighted training-file sampling."""
    from .loaders import build_loaders_balanced

    # GC Stratification Logic
    if stratify_gc and gc_metadata_file:
        LOGGER.info("[WeightedLoader] Loading GC metadata from %s...", gc_metadata_file)
        try:
            with open(gc_metadata_file) as f:
                gc_meta = json.load(f)

            # 1. Collect GC values for input files
            file_gcs = {}
            for f in files:
                name = os.path.basename(f)
                if name in gc_meta:
                    file_gcs[name] = gc_meta[name]["gc_percent"]
                else:
                    LOGGER.warning("[WeightedLoader] No GC data for %s, assuming 50%%", name)
                    file_gcs[name] = 50.0

            # 2. Bin files
            # Bins: 0-10, 10-20, ... 90-100
            bins = defaultdict(list)
            for name, gc in file_gcs.items():
                bin_idx = int(gc // 10) * 10
                bins[bin_idx].append(name)

            # 3. Calculate weights
            # Target: Uniform sampling across bins -> Weight = 1 / (count_in_bin)
            # This makes the total probability mass of each bin equal.
            # Then we divide by number of files in bin to distribute mass equally within bin.
            # Weight ~ 1 / N_files_in_bin

            weights = {}
            for names_in_bin in bins.values():
                count = len(names_in_bin)
                if count > 0:
                    w = 1.0 / count
                    for name in names_in_bin:
                        weights[name] = w

            LOGGER.info("[WeightedLoader] Calculated GC-stratified weights for %s files.", len(weights))
            LOGGER.info("  Bins found: %s", sorted(bins.keys()))

        except Exception as e:
            LOGGER.error("[WeightedLoader] Failed to load GC metadata: %s", e)
            if not weights:
                raise

    # Validate weights before building loaders
    basenames = [os.path.basename(f) for f in files]
    missing_weights = [b for b in basenames if b not in weights and "*" not in weights]
    if missing_weights:
        LOGGER.warning("[WeightedLoader] No weights specified for: %s", missing_weights)
        LOGGER.warning("[WeightedLoader] Using default weight from '*' if available")

    # Build balanced loaders
    train_rr, eval_loader = build_loaders_balanced(
        files=files,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        train_split_ratio=train_split_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        use_reverse_prob=use_reverse_prob,
        provide_rc=provide_rc,
        frameshift_prob=frameshift_prob,
        max_chunks_per_file=max_chunks_per_file,
        cap_by_file=cap_by_file,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        split_manifest=split_manifest,
        seed=seed,
        return_offsets=return_offsets,
    )

    # Wrap in weighted loader if possible
    if hasattr(train_rr, "loaders") and hasattr(train_rr, "names"):
        weighted_train = WeightedDataLoader(
            loaders=train_rr.loaders,
            weights=weights,
            names=train_rr.names,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
        )
        return weighted_train, eval_loader

    LOGGER.warning("[WeightedLoader] train_loader is not a RoundRobinLoader; returning original loaders.")
    return train_rr, eval_loader

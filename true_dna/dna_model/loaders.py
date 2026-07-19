import functools
import json
import logging
import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import PAD_TOKEN_ID
from .tokenizer import DnaTokenizer, pretokenized_paths, tokenizer_metadata

LOGGER = logging.getLogger(__name__)


def _seed_worker(_worker_id):
    wseed = torch.initial_seed() % 2**32
    np.random.seed(wseed)
    random.seed(wseed)


LOADER_GENERATOR = torch.Generator()
LOADER_GENERATOR.manual_seed(42)


class RoundRobinLoader:
    """Yield every batch from the configured loaders in round-robin order."""

    def __init__(self, loaders: list[DataLoader], names: list[str] = None, steps_per_loader: list[int] = None):
        if not loaders:
            raise ValueError("loaders must not be empty")
        if names is not None and len(names) != len(loaders):
            raise ValueError("names must contain one entry per loader")
        self.loaders = loaders
        self.names = names or [f"Loader_{i}" for i in range(len(loaders))]
        self.steps_per_loader = steps_per_loader
        self.total_len = sum(len(loader) for loader in loaders)
        self.iters = [iter(loader) for loader in loaders]
        self._active = [True] * len(loaders)
        self.ptr = 0

    def set_epoch(self, epoch: int):
        """Set epoch for all underlying datasets."""
        for loader in self.loaders:
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(epoch)

    def __len__(self):
        return self.total_len

    def __iter__(self):
        self.iters = [iter(loader) for loader in self.loaders]
        self._active = [True] * len(self.loaders)
        self.ptr = 0
        return self

    def __next__(self):
        while any(self._active):
            i = self.ptr
            self.ptr = (self.ptr + 1) % len(self.iters)
            if not self._active[i]:
                continue

            try:
                return next(self.iters[i])
            except StopIteration:
                self._active[i] = False

        raise StopIteration


class SmartBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that groups sequences by length to minimize padding.
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.skip_batches = 0

    def set_skip(self, n: int):
        self.skip_batches = n

    def __iter__(self):
        # Get lengths
        if hasattr(self.data_source, "get_lengths"):
            lengths = self.data_source.get_lengths()
        else:
            # Fallback: iterate (slow)
            LOGGER.warning("[SmartBatchSampler] data_source has no get_lengths, iterating...")
            lengths = [len(self.data_source[i]["input_ids"]) for i in range(len(self.data_source))]

        # Sort indices by length
        indices = torch.argsort(torch.tensor(lengths))

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size].tolist()
            batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            # Reseed RNG for determinism across resumes if needed,
            # but usually we rely on external seeding.
            # Assuming 'random' state is restored or consistent.
            random.shuffle(batches)

        if self.skip_batches > 0:
            LOGGER.info("[SmartBatchSampler] Fast-skipping %s batches.", self.skip_batches)
            batches = batches[self.skip_batches :]
            self.skip_batches = 0

        # Yield indices
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def count_chunks_exact(seq_or_len, max_length: int, stride: int, min_chunk_length: int) -> int:
    """
    Count exactly how many chunks will be generated from a sequence.
    Can accept either a sequence string or its length.
    """
    if isinstance(seq_or_len, str):
        seq_len = len(seq_or_len)
    else:
        seq_len = int(seq_or_len)

    if seq_len < min_chunk_length:
        return 0
    if seq_len <= max_length:
        return 1

    count = 0
    for s in range(0, seq_len - min_chunk_length + 1, max(1, stride)):
        chunk_len = min(max_length, seq_len - s)
        if chunk_len >= min_chunk_length:
            count += 1
        if s + max_length >= seq_len:
            break
    return count


def scan_fasta_lengths(path: str) -> list[tuple[int, int, int, int]]:
    """
    Scan FASTA file and return list of (index, start_offset, end_offset, length).
    Uses caching to avoid re-scanning large files.
    """
    # Cache path: fasta_path + ".index.json"
    cache_path = path + ".index.json"

    # Check cache
    if os.path.exists(cache_path):
        try:
            # Check modification times
            if os.path.getmtime(cache_path) >= os.path.getmtime(path):
                import json

                with open(cache_path) as f:
                    LOGGER.info("[DataPrep] Loading cached offsets from %s", cache_path)
                    return json.load(f)
        except Exception as e:
            LOGGER.warning("[DataPrep] Cache load failed: %s. Re-scanning.", e)

    LOGGER.info("[DataPrep] Scanning %s for offsets (this may take a while)...", path)
    offsets = []
    idx = 0

    with open(path, "rb") as f:
        offset = 0
        current_seq_start = -1
        current_seq_len = 0

        for line in f:
            line_len = len(line)
            if line.startswith(b">"):
                if current_seq_start != -1:
                    # Previous sequence ends here
                    offsets.append((idx, current_seq_start, offset, current_seq_len))
                    idx += 1

                current_seq_start = offset + line_len
                current_seq_len = 0
            else:
                if current_seq_start != -1:
                    stripped = line.strip()
                    current_seq_len += len(stripped)

            offset += line_len

        if current_seq_start != -1:
            offsets.append((idx, current_seq_start, offset, current_seq_len))

    # Save cache
    try:
        import json

        with open(cache_path, "w") as f:
            json.dump(offsets, f)
        LOGGER.info("[DataPrep] Saved cache to %s", cache_path)
    except Exception as e:
        LOGGER.warning("[DataPrep] Failed to save cache: %s", e)

    return offsets


class NpyDataset(Dataset):
    """Load token-aligned windows from preprocessed uint16 arrays."""

    def __init__(
        self,
        fa_path,
        max_length,
        stride,
        min_chunk_length,
        indices=None,
        provide_rc=False,
        tokenizer=None,
    ):
        self.fa_path = fa_path
        self.file_path = fa_path
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.min_chunk_length = int(min_chunk_length)
        if self.max_length <= 0 or self.stride <= 0 or self.min_chunk_length <= 0:
            raise ValueError("max_length, stride, and min_chunk_length must be positive")
        if provide_rc:
            raise ValueError("Pre-tokenized arrays do not support reverse-complement pairs; use the FASTA loader")
        self.name_root = os.path.basename(fa_path)
        tokenizer_type = getattr(tokenizer, "tokenizer_type", "bpe")
        cache_paths = pretokenized_paths(fa_path, tokenizer_type)

        index_path = cache_paths["offsets"]
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Missing token-offset index: {index_path}. "
                "Raw FASTA .index.json offsets cannot index pre-tokenized arrays."
            )

        with open(index_path, encoding="utf-8") as f:
            raw_offsets = json.load(f)
        if not isinstance(raw_offsets, list):
            raise ValueError(f"Token-offset index must contain a list: {index_path}")

        self.all_seq_offsets = []
        previous_end = 0
        for position, entry in enumerate(raw_offsets):
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise ValueError(f"Invalid token-offset entry {position} in {index_path}: {entry!r}")
            record_index, token_start, token_end, token_count = map(int, entry)
            if record_index < 0 or token_start < 0 or token_end < token_start:
                raise ValueError(f"Invalid token-offset bounds at entry {position}: {entry!r}")
            if token_end - token_start != token_count:
                raise ValueError(f"Token count does not match bounds at entry {position}: {entry!r}")
            if token_start != previous_end:
                raise ValueError(f"Token-offset entries must be contiguous at entry {position}: {entry!r}")
            self.all_seq_offsets.append((record_index, token_start, token_end, token_count))
            previous_end = token_end
        if not self.all_seq_offsets:
            raise ValueError(f"Token-offset index contains no sequences: {index_path}")

        if indices is not None:
            try:
                self.seq_offsets = [self.all_seq_offsets[int(i)] for i in indices]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError("indices contains an invalid sequence index") from exc
        else:
            self.seq_offsets = self.all_seq_offsets

        self.input_ids_path = cache_paths["input_ids"]
        if not os.path.isfile(self.input_ids_path):
            raise FileNotFoundError(f"Missing pre-tokenized array: {self.input_ids_path}")

        self.kmer_ids_path = cache_paths["kmer_ids"]
        self.has_kmer_ids = os.path.exists(self.kmer_ids_path)
        metadata_path = cache_paths["metadata"]
        if tokenizer is not None and os.path.isfile(metadata_path):
            with open(metadata_path, encoding="utf-8") as handle:
                saved_metadata = json.load(handle)
            expected_metadata = tokenizer_metadata(tokenizer)
            if saved_metadata != expected_metadata:
                raise ValueError(
                    f"Pre-tokenized cache is incompatible with the active {tokenizer_type} tokenizer: {metadata_path}"
                )
        elif tokenizer is not None and tokenizer_type != "bpe":
            raise ValueError(f"Base-tokenized cache is missing tokenizer metadata: {metadata_path}")
        elif tokenizer is not None:
            LOGGER.warning(
                "[NpyDataset] %s has no tokenizer metadata; treating it as a legacy BPE cache",
                self.name_root,
            )
        expected_tokens = self.all_seq_offsets[-1][2] if self.all_seq_offsets else 0
        input_size = os.path.getsize(self.input_ids_path)
        if input_size % np.dtype(np.uint16).itemsize:
            raise ValueError(f"Invalid uint16 array size: {self.input_ids_path}")
        file_tokens = input_size // np.dtype(np.uint16).itemsize
        if file_tokens != expected_tokens:
            raise ValueError(
                f"Token index describes {expected_tokens} tokens, but {self.input_ids_path} contains {file_tokens}"
            )
        if self.has_kmer_ids and os.path.getsize(self.kmer_ids_path) != input_size:
            raise ValueError("Input-token and k-mer arrays must contain the same number of uint16 values")

        try:
            import psutil
        except ImportError:
            load_into_memory = False
        else:
            total_size = input_size * (2 if self.has_kmer_ids else 1)
            load_into_memory = psutil.virtual_memory().available > total_size * 2

        if load_into_memory:
            LOGGER.info("[NpyDataset] %s: loading token arrays into RAM", self.name_root)
            self.input_ids_data = np.fromfile(self.input_ids_path, dtype=np.uint16)
            if self.has_kmer_ids:
                self.kmer_ids_data = np.fromfile(self.kmer_ids_path, dtype=np.uint16)
        else:
            LOGGER.info("[NpyDataset] %s: memory-mapping token arrays", self.name_root)
            self.input_ids_data = np.memmap(
                self.input_ids_path,
                dtype=np.uint16,
                mode="r",
                shape=(file_tokens,),
            )
            if self.has_kmer_ids:
                self.kmer_ids_data = np.memmap(
                    self.kmer_ids_path,
                    dtype=np.uint16,
                    mode="r",
                    shape=(file_tokens,),
                )

        self.chunk_coords = []
        for i, (_, _, _, seq_len) in enumerate(self.seq_offsets):
            if seq_len < self.min_chunk_length:
                continue
            if seq_len <= self.max_length:
                self.chunk_coords.append((i, 0, seq_len))
            else:
                for s in range(0, seq_len - self.min_chunk_length + 1, max(1, self.stride)):
                    chunk_end = min(s + self.max_length, seq_len)
                    chunk_len = chunk_end - s
                    if chunk_len >= self.min_chunk_length:
                        self.chunk_coords.append((i, s, chunk_end))
                    if s + self.max_length >= seq_len:
                        break

    def __len__(self):
        return len(self.chunk_coords)

    def __getitem__(self, idx):
        seq_idx, start_local, end_local = self.chunk_coords[idx]
        _, token_start, _, _ = self.seq_offsets[seq_idx]
        global_start = token_start + start_local
        global_end = token_start + end_local

        input_ids_np = np.array(self.input_ids_data[global_start:global_end], copy=True).astype(np.int64)
        attention_mask_np = (input_ids_np != PAD_TOKEN_ID).astype(np.int64)

        item = {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_np, dtype=torch.long),
        }

        if self.has_kmer_ids:
            kmer_ids_np = np.array(self.kmer_ids_data[global_start:global_end], copy=True).astype(np.int64)
            item["kmer_ids"] = torch.tensor(kmer_ids_np, dtype=torch.long)

        item["domain_name"] = os.path.basename(self.file_path)

        return item


def build_loaders_balanced(
    files: Sequence[str],
    tokenizer: DnaTokenizer,
    *,
    max_length: int,
    stride: int,
    train_split_ratio: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    use_reverse_prob: float,
    provide_rc: bool = False,
    frameshift_prob: float = 0.0,
    max_chunks_per_file: int | None = None,
    cap_by_file: dict | None = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    use_smart_batching: bool = False,
):
    """Build record-disjoint training and evaluation loaders."""
    from .dataset import DatasetImpl, collate

    if not (0.0 < train_split_ratio < 1.0):
        raise ValueError(f"train_split_ratio must be in (0,1), got {train_split_ratio}")

    names = []
    train_loaders = []
    pooled_eval_seqs = []
    kept_total_windows = 0

    for path in files:
        if not os.path.exists(path):
            LOGGER.warning("[DataPrep] File not found: %s", path)
            continue

        # Scan file for lengths (fast, low memory)
        # Returns list of (index, start_offset, end_offset, length)
        seq_offsets = scan_fasta_lengths(path)
        if not seq_offsets:
            LOGGER.warning("[DataPrep] No sequences found in %s", path)
            continue

        # Filter by min_chunk_length
        min_chunk_length = max(64, max_length // 4)
        valid_items = [(index, length) for index, _, _, length in seq_offsets if length >= min_chunk_length]

        if not valid_items:
            LOGGER.warning("[DataPrep] No sequences >= %sbp in %s", min_chunk_length, path)
            continue

        # Split per file by exact chunk count
        # seq_windows = [((original_index, length), count) ...]
        seq_windows = [
            ((index, length), count_chunks_exact(length, max_length, stride, min_chunk_length))
            for index, length in valid_items
        ]
        seq_windows = [(item, w) for item, w in seq_windows if w > 0]

        if not seq_windows:
            LOGGER.warning("[DataPrep] No valid chunks >= %sbp in %s", min_chunk_length, path)
            continue

        total_windows = sum(w for _, w in seq_windows)

        # Split by cumulative window count with better boundary handling
        target_train_windows = int(total_windows * train_split_ratio)
        train_items = []  # List of (original_index, length)
        eval_items = []  # List of (original_index, length)
        cumulative = 0

        for item, wc in seq_windows:
            if cumulative + wc <= target_train_windows:
                # Entire sequence fits in training
                train_items.append(item)
                cumulative += wc
            elif cumulative < target_train_windows:
                # Sequence straddles boundary - assign to whichever is closer
                remaining_train = target_train_windows - cumulative
                if remaining_train > wc / 2:
                    # More than half needed for train
                    train_items.append(item)
                    cumulative += wc
                else:
                    # Send to eval
                    eval_items.append(item)
            else:
                # Already met training quota
                eval_items.append(item)

        # Preserve record-level separation between training and evaluation.
        if not train_items and valid_items:
            train_items = [valid_items[0]]
            eval_items = valid_items[1:] if len(valid_items) > 1 else []
        if not eval_items and len(train_items) > 1:
            eval_item = min(
                train_items,
                key=lambda item: count_chunks_exact(item[1], max_length, stride, min_chunk_length),
            )
            train_items.remove(eval_item)
            eval_items.append(eval_item)

        # Handle cap_by_file with warnings
        basename = os.path.basename(path)
        effective_cap = None

        if cap_by_file:
            if basename in cap_by_file:
                effective_cap = cap_by_file[basename]
            else:
                LOGGER.warning("[DataPrep] No cap specified for %s in cap_by_file", basename)
                if "*" in cap_by_file:
                    effective_cap = cap_by_file["*"]
                    LOGGER.info("[DataPrep] Using default cap: %s", effective_cap)
                else:
                    effective_cap = max_chunks_per_file
                    LOGGER.info("[DataPrep] Using max_chunks_per_file: %s", effective_cap)
        else:
            effective_cap = max_chunks_per_file

        file_kept_windows = 0

        if effective_cap and effective_cap > 0:
            # Recalculate windows for train_items
            train_windows = [
                ((index, length), count_chunks_exact(length, max_length, stride, min_chunk_length))
                for index, length in train_items
            ]
            train_windows = [(item, w) for item, w in train_windows if w > 0]
            train_windows.sort(key=lambda x: -x[1])  # Prioritize longer sequences

            kept = []
            running = 0
            for item, count in train_windows:
                if kept and running + count > effective_cap:
                    continue
                kept.append(item)
                running += count

            if not kept and train_items:
                kept = [train_items[0]]  # Keep at least one

            LOGGER.info(
                "[DataPrep] %s: windows %s -> %s (capped at %s)",
                basename,
                sum(count_chunks_exact(length, max_length, stride, min_chunk_length) for _, length in train_items),
                running,
                effective_cap,
            )
            train_items = kept
            file_kept_windows = running
        else:
            file_kept_windows = sum(
                count_chunks_exact(length, max_length, stride, min_chunk_length) for _, length in train_items
            )
            LOGGER.info(
                "[DataPrep] %s: train_seqs=%s, eval_seqs=%s, windows~%s",
                basename,
                len(train_items),
                len(eval_items),
                file_kept_windows,
            )

        kept_total_windows += file_kept_windows

        # Extract indices for DatasetImpl
        train_indices = [index for index, _ in train_items]

        npy_path = pretokenized_paths(path, getattr(tokenizer, "tokenizer_type", "bpe"))["input_ids"]

        ds_train = None
        raw_sequence_required = provide_rc or use_reverse_prob > 0.0 or frameshift_prob > 0.0
        if os.path.exists(npy_path) and not raw_sequence_required:
            LOGGER.info("[DataPrep] %s: Found pre-tokenized arrays. Using NpyDataset.", basename)
            try:
                ds_train = NpyDataset(
                    path,
                    max_length,
                    stride,
                    min_chunk_length,
                    indices=train_indices,
                    provide_rc=provide_rc,
                    tokenizer=tokenizer,
                )
            except Exception as e:
                LOGGER.warning("[DataPrep] Error initializing NpyDataset: %s. Falling back to standard loader.", e)
                ds_train = None
        elif os.path.exists(npy_path) and raw_sequence_required:
            reasons = []
            if provide_rc:
                reasons.append("RC pairs")
            if use_reverse_prob > 0.0:
                reasons.append("reverse-complement augmentation")
            if frameshift_prob > 0.0:
                reasons.append("pre-BPE frameshift augmentation")
            reason_text = ", ".join(reasons)
            LOGGER.info("[DataPrep] %s: .npy found, but %s requested; using FASTA loader.", basename, reason_text)

        if ds_train is None:
            try:
                import psutil

                mem = psutil.virtual_memory()
                file_size = os.path.getsize(path)
                # Threshold: If available RAM is > 1.5x file size, load into memory
                should_lazy = file_size * 1.5 > mem.available
                if not should_lazy:
                    LOGGER.info(
                        "[DataPrep] %s: %.1fMB fits in RAM (%.1fGB free). Loading into memory for speed.",
                        basename,
                        file_size / 1e6,
                        mem.available / 1e9,
                    )
                else:
                    LOGGER.info(
                        "[DataPrep] %s: %.1fMB too large for RAM. Using lazy loading.",
                        basename,
                        file_size / 1e6,
                    )
            except ImportError:
                should_lazy = True

            ds_train = DatasetImpl(
                tokenizer,
                path,
                max_length,
                stride,
                use_reverse_prob=use_reverse_prob,
                provide_rc=provide_rc,
                frameshift_prob=frameshift_prob,
                min_chunk_length=min_chunk_length,
                indices=train_indices,
                lazy_load=should_lazy,
                precomputed_offsets=seq_offsets,
            )

        if len(ds_train) == 0:
            LOGGER.warning("[DataPrep] No valid chunks from %s, skipping", path)
            continue

        batch_sampler = None
        shuffle = True
        if use_smart_batching:
            LOGGER.info("[DataPrep] Enabling Smart Batching for %s", basename)
            batch_sampler = SmartBatchSampler(ds_train, batch_size, shuffle=True)
            shuffle = False  # batch_sampler handles shuffling

        dl_train = DataLoader(
            ds_train,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            drop_last=drop_last
            if batch_sampler is None
            else False,  # batch_sampler handles drop_last logic internally or we accept partial
            batch_size=1
            if batch_sampler is not None
            else batch_size,  # batch_size is ignored if batch_sampler is provided
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=functools.partial(collate, max_length=max_length),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=LOADER_GENERATOR,
        )
        train_loaders.append(dl_train)
        names.append(basename)

        if eval_items:
            eval_indices_set = {index for index, _ in eval_items}
            LOGGER.info("[DataPrep] Loading %s eval sequences via random access...", len(eval_indices_set))

            with open(path, "rb") as f:
                for idx in eval_indices_set:
                    if idx < len(seq_offsets):
                        _, start_offset, end_offset, _ = seq_offsets[idx]
                        f.seek(start_offset)
                        raw_bytes = f.read(end_offset - start_offset)
                        seq = raw_bytes.replace(b"\n", b"").replace(b"\r", b"").decode("utf-8")
                        pooled_eval_seqs.append(seq)
                    else:
                        LOGGER.error("[DataPrep] Index %s out of bounds for %s", idx, path)

    if not pooled_eval_seqs:
        raise ValueError(
            "No record-disjoint evaluation data is available. "
            "Provide at least one FASTA file with two qualifying records."
        )

    # Trim eval set to match desired ratio
    r = float(train_split_ratio)
    desired_eval_windows = max(1, int(round(kept_total_windows * (1.0 - r) / r)))

    eval_accum = 0
    trimmed_eval = []

    random.Random(42).shuffle(pooled_eval_seqs)

    for s in pooled_eval_seqs:
        w = count_chunks_exact(len(s), max_length, stride, min_chunk_length)
        if w <= 0:
            continue
        if eval_accum >= desired_eval_windows:
            break
        trimmed_eval.append(s)
        eval_accum += w

    pooled_eval_seqs = trimmed_eval

    LOGGER.info(
        "[DataPrep] Adjusted eval windows to %s (target: %s) to match %.1f%% train split ratio "
        "(kept train windows: %s)",
        eval_accum,
        desired_eval_windows,
        r * 100.0,
        kept_total_windows,
    )

    # Build evaluation loader
    ds_eval = DatasetImpl(
        tokenizer,
        pooled_eval_seqs,
        max_length,
        stride,
        use_reverse_prob=0.0,
        frameshift_prob=0.0,
        min_chunk_length=min_chunk_length,
    )

    eval_loader = DataLoader(
        ds_eval,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    LOGGER.info(
        "[DataPrep] FINAL: windows~%s, files=%s, cap=%s, train_chunks=%s, eval_chunks=%s",
        kept_total_windows,
        len(names),
        max_chunks_per_file,
        sum(len(dl.dataset) for dl in train_loaders),
        len(ds_eval),
    )

    # Create round-robin loader
    if not train_loaders:
        raise ValueError("No training data available!")

    train_loader = RoundRobinLoader(train_loaders, names=names)

    # Final verification
    kept_train = sum(len(dl.dataset) for dl in train_loaders)
    kept_eval = len(ds_eval)
    tot = kept_train + kept_eval if (kept_train + kept_eval) > 0 else 1

    LOGGER.info(
        "[DataPrep] KEPT: train=%s (%.1f%%), eval=%s (%.1f%%), target=%.1f%%",
        kept_train,
        kept_train / tot * 100.0,
        kept_eval,
        kept_eval / tot * 100.0,
        train_split_ratio * 100.0,
    )

    return train_loader, eval_loader

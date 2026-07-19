import argparse
import glob
import json
import multiprocessing
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_model.tokenizer import (  # noqa: E402, I001
    build_tokenizer,
    normalize_dna_sequence,
    normalize_tokenizer_type,
    pretokenized_paths,
    read_fasta_records,
    tokenizer_metadata,
)


_cached_tokenizer = None


def init_worker(tokenizer_type, vocab_path):
    global _cached_tokenizer
    _cached_tokenizer = build_tokenizer(tokenizer_type, vocab_path=vocab_path)


def _tokenize_worker(args):
    global _cached_tokenizer
    idx, chunk_seq, is_last = args

    encoded = _cached_tokenizer.encode(
        chunk_seq,
        return_tensors=None,
        padding=False,
        truncation=False,
        return_kmer_ids=True,
        return_offsets=True,
        add_special_tokens=False,
    )
    ids = encoded["input_ids"]
    kmer_ids = encoded["kmer_ids"]
    ids_np = np.array(ids, dtype=np.uint16)
    kmer_np = np.array(kmer_ids, dtype=np.uint16)
    return idx, ids_np, kmer_np, len(ids_np), is_last


def preprocess_stream():
    parser = argparse.ArgumentParser(description="Pre-tokenize FASTA files into tokenizer-specific .npy memmaps.")
    parser.add_argument("--fasta", nargs="+", required=True, help="Glob pattern(s) or paths for FASTA files")
    parser.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        type=normalize_tokenizer_type,
        default="bpe",
        choices=["bpe", "base"],
        help="Primary tokenization granularity (single_base is accepted as a legacy alias for base)",
    )
    parser.add_argument(
        "--vocab_path", type=str, default="dna_model/bpe_vocab/tokenizer.json", help="Path to BPE tokenizer.json"
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, multiprocessing.cpu_count()), help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cache when its tokenizer metadata is missing or incompatible",
    )
    # --k_mer_sizes kept for CLI compatibility with old scripts but ignored.
    parser.add_argument(
        "--k_mer_sizes", type=str, default="3", help="[IGNORED] Legacy arg kept for script compatibility"
    )
    args = parser.parse_args()

    files = []
    for f_pattern in args.fasta:
        matched = glob.glob(f_pattern)
        if matched:
            files.extend(matched)
        elif os.path.exists(f_pattern):
            files.append(f_pattern)

    files = sorted(list(set(files)))  # Unique sorted list
    if not files:
        print(f"[Error] No files matched pattern: {args.fasta}")
        sys.exit(1)

    # Resolve vocab path relative to project root
    vocab_path = args.vocab_path
    if not os.path.isabs(vocab_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_path = os.path.join(project_root, vocab_path)

    if args.tokenizer_type == "bpe" and not os.path.exists(vocab_path):
        print(f"[Error] Tokenizer vocab not found: {vocab_path}")
        print("  Run: python scripts/train_bpe.py --fasta 'data/domain_fastas/*.fa'")
        sys.exit(1)

    print(f"Initializing {args.tokenizer_type} tokenizer...")
    tokenizer = build_tokenizer(args.tokenizer_type, vocab_path=vocab_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise SystemExit(f"Tokenizer vocabulary ({tokenizer.vocab_size}) does not fit in the uint16 cache format")
    expected_metadata = tokenizer_metadata(tokenizer)

    incompatible_caches = []
    for fasta_path in files:
        if not os.path.exists(fasta_path):
            print(f"[Warning] File not found: {fasta_path}")
            continue

        name_root = os.path.basename(fasta_path)

        cache_paths = pretokenized_paths(fasta_path, args.tokenizer_type)
        input_ids_path = cache_paths["input_ids"]
        kmer_ids_path = cache_paths["kmer_ids"]
        offsets_path = cache_paths["offsets"]
        metadata_path = cache_paths["metadata"]

        cache_complete = all(os.path.exists(path) for path in (offsets_path, input_ids_path, kmer_ids_path))
        existing_metadata = None
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as handle:
                    existing_metadata = json.load(handle)
            except (OSError, json.JSONDecodeError):
                existing_metadata = None
        if cache_complete and existing_metadata == expected_metadata:
            print(f"  [SKIP] {name_root}: compatible {args.tokenizer_type} cache already exists.")
            continue
        if cache_complete and not args.force:
            print(
                f"  [ERROR] {name_root}: existing {args.tokenizer_type} cache has missing or incompatible "
                "metadata; rerun with --force to replace it."
            )
            incompatible_caches.append(name_root)
            continue

        print(f"\nProcessing: {fasta_path} ({args.workers} workers)...")

        # FASTA byte size is a safe upper bound for both tokenization modes.
        total_len_upper = os.path.getsize(fasta_path)
        print(f"  Allocating upper-bound memmap: {total_len_upper:,} uint16 slots")

        mm_input = np.memmap(input_ids_path, dtype="uint16", mode="w+", shape=(total_len_upper,))
        mm_kmer = np.memmap(kmer_ids_path, dtype="uint16", mode="w+", shape=(total_len_upper,))

        offsets = []
        current_pos = 0
        flush_bytes = 0
        seq_start_pos = 0

        def string_generator(source_path=fasta_path):
            CHUNK_SIZE = 2_000_000
            seq_idx = -1
            last_header = None
            for header, chunk, is_last in read_fasta_records(source_path, chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                if header != last_header:
                    seq_idx += 1
                    last_header = header
                yield seq_idx, normalize_dna_sequence(chunk), is_last

        with multiprocessing.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.tokenizer_type, vocab_path),
        ) as pool:
            results = pool.imap(_tokenize_worker, string_generator(), chunksize=2)

            for idx, ids_np, kmer_np, length, is_last in tqdm(results, desc=f"  Tokenizing {name_root}", unit="chunk"):
                end_pos = current_pos + length

                if end_pos > total_len_upper:
                    # Safety: extend memmap if BPE ever produces more tokens than bytes
                    new_size = end_pos + 10_000_000
                    mm_input.flush()
                    mm_kmer.flush()
                    del mm_input
                    del mm_kmer
                    os.truncate(input_ids_path, new_size * 2)
                    os.truncate(kmer_ids_path, new_size * 2)
                    mm_input = np.memmap(input_ids_path, dtype="uint16", mode="r+", shape=(new_size,))
                    mm_kmer = np.memmap(kmer_ids_path, dtype="uint16", mode="r+", shape=(new_size,))
                    total_len_upper = new_size

                mm_input[current_pos:end_pos] = ids_np[:length]
                mm_kmer[current_pos:end_pos] = kmer_np[:length]
                current_pos = end_pos
                flush_bytes += length

                if is_last:
                    offsets.append((idx, seq_start_pos, current_pos, current_pos - seq_start_pos))
                    seq_start_pos = current_pos

                if flush_bytes > 50_000_000:
                    mm_input.flush()
                    mm_kmer.flush()
                    flush_bytes = 0

                del ids_np
                del kmer_np

        mm_input.flush()
        mm_kmer.flush()
        del mm_input
        del mm_kmer
        import gc

        gc.collect()

        # Truncate to actual written size (uint16 = 2 bytes per element)
        os.truncate(input_ids_path, current_pos * 2)
        os.truncate(kmer_ids_path, current_pos * 2)

        with open(offsets_path, "w") as f:
            json.dump(offsets, f)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(expected_metadata, f, indent=2, sort_keys=True)

        actual_mb = (current_pos * 2) / 1e6
        print(f"  3-mer side channel: {kmer_ids_path}")
        print(f"  Done: {current_pos:,} tokens ({actual_mb:.1f} MB): {input_ids_path}")
        print(f"  Offsets: {len(offsets)} sequences: {offsets_path}")
        print(f"  Tokenizer metadata: {metadata_path}")

    if incompatible_caches:
        raise SystemExit(
            f"Refused to replace {len(incompatible_caches)} incompatible cache(s) without --force: "
            + ", ".join(incompatible_caches)
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    preprocess_stream()

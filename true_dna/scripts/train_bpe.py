import itertools
import os
import sys
from collections import Counter
from collections.abc import Iterable

from tokenizers import Tokenizer, models
from tokenizers.processors import TemplateProcessing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dna_model.tokenizer import read_fasta_records

DNA_BASES = "ACGT"
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _iter_sampled_dna(fasta_files: list[str], limit_bases: int) -> Iterable[str]:
    total = 0
    for file in fasta_files:
        if total >= limit_bases:
            break
        for _header, seq, _is_last in read_fasta_records(file, chunk_size=1_000_000):
            clean_seq = "".join(ch for ch in seq.upper() if ch in "ACGTN")
            if not clean_seq:
                continue
            remaining = limit_bases - total
            if remaining <= 0:
                return
            clean_seq = clean_seq[:remaining]
            total += len(clean_seq)
            yield clean_seq
    print(f"BPE sample: {total:,} bases", flush=True)


def _count_kmers(fasta_files: list[str], limit_bases: int, k: int = 6) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seq in _iter_sampled_dna(fasta_files, limit_bases):
        for run in seq.split("N"):
            if len(run) < k:
                continue
            for start in range(0, len(run) - k + 1):
                counts[run[start : start + k]] += 1
    return counts


def _add_token(vocab: dict[str, int], token: str) -> None:
    if token not in vocab:
        vocab[token] = len(vocab)


def _build_dna_bpe(vocab_size: int, top_6mers: list[str]) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Build a deterministic DNA BPE vocab and merge list.

    The vocabulary contains special tokens, single A/C/G/T/N bases, all
    A/C/G/T 2-5-mers, and the most frequent 6-mers that fit the requested
    vocabulary size. This avoids HuggingFace trainer stalls while preserving a
    standard BPE tokenizer.json for runtime encoding.
    """
    min_vocab = len(SPECIAL_TOKENS) + 5
    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be at least {min_vocab}, got {vocab_size}")

    vocab: dict[str, int] = {}
    merges: list[tuple[str, str]] = []

    for token in SPECIAL_TOKENS:
        _add_token(vocab, token)
    for base in "ACGTN":
        _add_token(vocab, base)

    previous_level = list(DNA_BASES)
    for _length in range(2, 6):
        next_level: list[str] = []
        for prefix in previous_level:
            for base in DNA_BASES:
                token = prefix + base
                if len(vocab) >= vocab_size:
                    return vocab, merges
                _add_token(vocab, token)
                merges.append((prefix, base))
                next_level.append(token)
        previous_level = next_level

    capacity = vocab_size - len(vocab)
    if capacity <= 0:
        return vocab, merges

    selected_6mers: list[str] = []
    seen = set()
    for kmer in top_6mers:
        if kmer in seen or len(kmer) != 6 or any(ch not in DNA_BASES for ch in kmer):
            continue
        selected_6mers.append(kmer)
        seen.add(kmer)
        if len(selected_6mers) >= capacity:
            break

    if len(selected_6mers) < capacity:
        for chars in itertools.product(DNA_BASES, repeat=6):
            kmer = "".join(chars)
            if kmer in seen:
                continue
            selected_6mers.append(kmer)
            seen.add(kmer)
            if len(selected_6mers) >= capacity:
                break

    for token in selected_6mers:
        prefix = token[:-1]
        suffix = token[-1]
        if prefix not in vocab:
            continue
        _add_token(vocab, token)
        merges.append((prefix, suffix))
        if len(vocab) >= vocab_size:
            break

    return vocab, merges


def train_dna_bpe(fasta_files: list[str], vocab_size: int = 4096) -> None:
    print(f"Training deterministic DNA BPE Tokenizer (Vocab Size: {vocab_size})...", flush=True)

    sample_limit = int(os.environ.get("BPE_SAMPLE_BYTES", 100_000_000))
    kmer_counts = _count_kmers(fasta_files, sample_limit, k=6)
    top_6mers = [kmer for kmer, _count in sorted(kmer_counts.items(), key=lambda item: (-item[1], item[0]))]

    vocab, merges = _build_dna_bpe(vocab_size, top_6mers)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="[UNK]"))

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dna_model", "bpe_vocab")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(out_path)
    print(
        f"Tokenizer saved to {out_path} "
        f"(vocab={tokenizer.get_vocab_size()}, merges={len(merges)}, top_6mers={len(top_6mers)})",
        flush=True,
    )


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", nargs="+", required=True, help="FASTA files or glob patterns")
    parser.add_argument("--vocab_size", type=int, default=4096)
    args = parser.parse_args()

    files = []
    for p in args.fasta:
        files.extend(glob.glob(p))
    train_dna_bpe(files, args.vocab_size)

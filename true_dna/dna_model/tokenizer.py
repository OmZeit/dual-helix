from __future__ import annotations

import gzip
import hashlib
import os
from collections.abc import Iterable, Sequence

import torch
from tokenizers import Tokenizer

from .config import KMER_PAD_ID

_RC = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
        "N": "N",
        "n": "n",
    }
)

_KMER_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
_VALID_DNA = set("ACGTN")
TOKENIZER_TYPES = ("bpe", "base")


def normalize_tokenizer_type(value: str) -> str:
    """Return the canonical tokenizer name used in configs and metadata."""
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "bpe": "bpe",
        "base": "base",
        "base_level": "base",
        "single_base": "base",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(TOKENIZER_TYPES)
        raise ValueError(f"Unknown tokenizer type {value!r}; expected one of: {choices}") from exc


def pretokenized_paths(fasta_path: str, tokenizer_type: str) -> dict[str, str]:
    """Return non-colliding cache paths for a tokenizer mode.

    BPE retains the repository's legacy filenames. Base-level caches use an
    explicit ``_base`` infix so both representations can exist together.
    """
    mode = normalize_tokenizer_type(tokenizer_type)
    infix = "" if mode == "bpe" else "_base"
    return {
        "input_ids": f"{fasta_path}{infix}_input_ids.npy",
        "kmer_ids": f"{fasta_path}{infix}_kmer_ids.npy",
        "offsets": f"{fasta_path}{infix}_offsets.json",
        "metadata": f"{fasta_path}{infix}_tokenizer.json",
    }


def tokenizer_metadata(tokenizer: object) -> dict[str, object]:
    """Build the compatibility metadata stored beside pre-tokenized arrays."""
    return {
        "format_version": 1,
        "tokenizer_type": tokenizer.tokenizer_type,
        "vocab_size": int(tokenizer.vocab_size),
        "vocab_sha256": getattr(tokenizer, "vocab_sha256", None),
        "kmer_size": int(tokenizer.kmer_size),
        "kmer_vocab_size": int(tokenizer.kmer_vocab_size),
        "special_token_ids": {
            name: int(getattr(tokenizer, f"{name}_token_id")) for name in ("pad", "unk", "cls", "sep", "mask")
        },
    }


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(_RC)[::-1]


def normalize_dna_sequence(seq: str) -> str:
    """Uppercase a DNA string and map non-ACGTN symbols to N without changing length."""
    return "".join(ch if ch in _VALID_DNA else "N" for ch in seq.upper())


def _open_text(path: str):
    """Open text file, handling gzip compression automatically."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def read_fasta_records(path: str, chunk_size: int | None = None) -> Iterable[tuple[str, str, bool]]:
    """
    Minimal FASTA reader.

    Yields ``(header, sequence_or_chunk, is_last)``.  When ``chunk_size`` is
    provided, large records may be yielded in multiple chunks with ``is_last``
    marking the end of the original FASTA record.
    """
    hdr = None
    seq_parts: list[str] = []
    current_len = 0
    pending_chunk: str | None = None

    def flush_record():
        nonlocal pending_chunk, seq_parts, current_len
        if hdr is None:
            return
        if chunk_size:
            if pending_chunk is not None:
                if seq_parts:
                    yield hdr, pending_chunk, False
                    yield hdr, "".join(seq_parts), True
                else:
                    yield hdr, pending_chunk, True
            elif seq_parts:
                yield hdr, "".join(seq_parts), True
        elif seq_parts:
            yield hdr, "".join(seq_parts), True
        pending_chunk = None
        seq_parts = []
        current_len = 0

    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                yield from flush_record()
                hdr = line[1:].strip()
            else:
                seq_parts.append(line)
                current_len += len(line)

                if chunk_size and current_len >= chunk_size:
                    chunk = "".join(seq_parts)
                    if pending_chunk is not None:
                        yield hdr or "", pending_chunk, False
                    pending_chunk = chunk
                    seq_parts = []
                    current_len = 0

        yield from flush_record()


def kmer_to_id(kmer: str) -> int:
    """Map an A/C/G/T k-mer to 1..4**k. 0 is reserved for pad/unknown."""
    value = 0
    for ch in kmer.upper():
        base = _KMER_BASE_TO_ID.get(ch)
        if base is None:
            return KMER_PAD_ID
        value = value * 4 + base
    return value + 1


class DnaTokenizer:
    def __init__(
        self,
        vocab_path: str = "dna_model/bpe_vocab/tokenizer.json",
        max_length: int = 8192,
        k_mer_sizes: Sequence[int] | None = None,
        kmer_size: int = 3,
        return_kmer_ids: bool = True,
        **_: object,
    ):
        """
        Lightweight wrapper around the Hugging Face BPE tokenizer.

        BPE remains the primary token stream.  ``kmer_ids`` is an aligned side
        channel: one 3-mer ID per BPE token, derived from the token offset in
        the original sequence.  Special/pad/ambiguous tokens receive ID 0.
        """
        self.tokenizer_type = "bpe"
        self.max_length = int(max_length)
        if k_mer_sizes:
            self.k_mer_sizes = [int(k) for k in k_mer_sizes]
            self.kmer_size = self.k_mer_sizes[0]
        else:
            self.k_mer_sizes = [int(kmer_size)]
            self.kmer_size = int(kmer_size)
        self.k_mer_size = self.kmer_size  # legacy attribute name
        self.return_kmer_ids = bool(return_kmer_ids)
        self.kmer_pad_id = KMER_PAD_ID
        self.kmer_vocab_size = (4**self.kmer_size) + 1
        self.vocab_sizes = {int(k): (4 ** int(k)) + 1 for k in self.k_mer_sizes}

        if not os.path.exists(vocab_path) and not os.path.isabs(vocab_path):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            rel = vocab_path.replace("\\", "/")
            if rel.startswith("dna_model/"):
                rel = rel[len("dna_model/") :]
            alt_path = os.path.join(module_dir, rel)
            if os.path.exists(alt_path):
                vocab_path = alt_path

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"BPE tokenizer not found at {vocab_path}. Run scripts/train_bpe.py first.")

        self.vocab_path = os.path.abspath(vocab_path)
        with open(self.vocab_path, "rb") as handle:
            self.vocab_sha256 = hashlib.sha256(handle.read()).hexdigest()
        self.tokenizer = Tokenizer.from_file(self.vocab_path)

        required_specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.special_ids = {}
        for token in required_specials:
            tid = self.tokenizer.token_to_id(token)
            if tid is None:
                raise ValueError(f"Special token '{token}' missing from BPE vocab at {vocab_path}")
            self.special_ids[token] = tid

        self.pad_token_id = self.special_ids["[PAD]"]
        self.unk_token_id = self.special_ids["[UNK]"]
        self.cls_token_id = self.special_ids["[CLS]"]
        self.sep_token_id = self.special_ids["[SEP]"]
        self.mask_token_id = self.special_ids["[MASK]"]
        self.special_tokens_max_id = max(self.special_ids.values())
        self.vocab_size = self.tokenizer.get_vocab_size()

    def kmer_ids_from_offsets(
        self,
        sequence: str,
        offsets: Sequence[tuple[int, int]],
        *,
        target_length: int | None = None,
    ) -> list[int]:
        ids: list[int] = []
        seq_upper = sequence.upper()
        k = self.kmer_size
        for start, end in offsets:
            if end <= start:
                ids.append(self.kmer_pad_id)
                continue
            token_seq = seq_upper[start:end]
            if len(token_seq) < k:
                ids.append(self.kmer_pad_id)
                continue
            ids.append(kmer_to_id(token_seq[:k]))

        if target_length is not None:
            if len(ids) > target_length:
                ids = ids[:target_length]
            elif len(ids) < target_length:
                ids.extend([self.kmer_pad_id] * (target_length - len(ids)))
        return ids

    def encode(
        self,
        sequence: str,
        *,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        padding: bool = True,
        truncation: bool = True,
        return_kmer_ids: bool | None = None,
        return_offsets: bool = False,
        add_special_tokens: bool = True,
    ):
        """Encode a raw DNA sequence into BPE token IDs and optional metadata."""
        sequence = normalize_dna_sequence(sequence)
        effective_max = int(max_length or self.max_length)
        encoded = self.tokenizer.encode(sequence, add_special_tokens=add_special_tokens)

        input_ids = list(encoded.ids)
        attention_mask = list(encoded.attention_mask)
        offsets = list(encoded.offsets)
        include_kmers = self.return_kmer_ids if return_kmer_ids is None else bool(return_kmer_ids)

        if truncation and len(input_ids) > effective_max:
            input_ids = input_ids[:effective_max]
            attention_mask = attention_mask[:effective_max]
            offsets = offsets[:effective_max]
            if input_ids and input_ids[-1] != self.sep_token_id:
                input_ids[-1] = self.sep_token_id
                attention_mask[-1] = 1
                offsets[-1] = (0, 0)

        kmer_ids = None
        if include_kmers:
            kmer_ids = self.kmer_ids_from_offsets(sequence, offsets)

        if padding and len(input_ids) < effective_max:
            pad_len = effective_max - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            offsets.extend([(0, 0)] * pad_len)
            if kmer_ids is not None:
                kmer_ids.extend([self.kmer_pad_id] * pad_len)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if kmer_ids is not None:
            out["kmer_ids"] = kmer_ids
        if return_offsets:
            out["offsets"] = offsets

        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out

    def encode_with_kmers(self, sequence: str, max_length: int | None = None, **kwargs):
        """Backward-compatible alias for callers that request aligned k-mers."""
        kwargs.setdefault("return_tensors", None)
        kwargs.setdefault("return_kmer_ids", True)
        return self.encode(sequence, max_length=max_length, **kwargs)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode BPE token IDs back into a continuous DNA sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_base_length(self, token_id: int) -> int:
        """Return how many DNA bases a vocabulary token represents."""
        token = self.tokenizer.id_to_token(int(token_id))
        if token is None or token.startswith("["):
            return 0
        return len(token) if all(base in _VALID_DNA for base in token) else 0


class BaseTokenizer:
    def __init__(
        self,
        max_length: int = 8192,
        k_mer_sizes: Sequence[int] | None = None,
        kmer_size: int = 3,
        return_kmer_ids: bool = True,
        **_: object,
    ):
        """
        Single-base tokenizer mapping A/C/G/T/N to unique IDs.
        """
        self.tokenizer_type = "base"
        self.vocab_sha256 = None
        self.max_length = int(max_length)
        if k_mer_sizes:
            self.k_mer_sizes = [int(k) for k in k_mer_sizes]
            self.kmer_size = self.k_mer_sizes[0]
        else:
            self.k_mer_sizes = [int(kmer_size)]
            self.kmer_size = int(kmer_size)
        self.k_mer_size = self.kmer_size
        self.return_kmer_ids = bool(return_kmer_ids)
        self.kmer_pad_id = KMER_PAD_ID
        self.kmer_vocab_size = (4**self.kmer_size) + 1
        self.vocab_sizes = {int(k): (4 ** int(k)) + 1 for k in self.k_mer_sizes}

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.special_tokens_max_id = 4

        self.char_to_id = {"A": 5, "C": 6, "G": 7, "T": 8, "N": 9}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        for k, v in {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}.items():
            self.id_to_char[v] = k

        self.vocab_size = 10

    def kmer_ids_from_offsets(
        self,
        sequence: str,
        offsets: Sequence[tuple[int, int]],
        *,
        target_length: int | None = None,
    ) -> list[int]:
        ids: list[int] = []
        seq_upper = sequence.upper()
        k = self.kmer_size
        for start, end in offsets:
            if end <= start:
                ids.append(self.kmer_pad_id)
                continue
            token_seq = seq_upper[start : start + k]
            if len(token_seq) < k:
                ids.append(self.kmer_pad_id)
                continue
            ids.append(kmer_to_id(token_seq))

        if target_length is not None:
            if len(ids) > target_length:
                ids = ids[:target_length]
            elif len(ids) < target_length:
                ids.extend([self.kmer_pad_id] * (target_length - len(ids)))
        return ids

    def encode(
        self,
        sequence: str,
        *,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        padding: bool = True,
        truncation: bool = True,
        return_kmer_ids: bool | None = None,
        return_offsets: bool = False,
        add_special_tokens: bool = True,
    ):
        """Encode a raw DNA sequence into base token IDs and optional metadata."""
        sequence = normalize_dna_sequence(sequence)
        effective_max = int(max_length or self.max_length)

        input_ids = [self.char_to_id.get(ch, self.unk_token_id) for ch in sequence]
        attention_mask = [1] * len(input_ids)
        offsets = [(i, i + 1) for i in range(len(input_ids))]
        if add_special_tokens:
            input_ids = [self.cls_token_id, *input_ids, self.sep_token_id]
            attention_mask = [1, *attention_mask, 1]
            offsets = [(0, 0), *offsets, (0, 0)]

        include_kmers = self.return_kmer_ids if return_kmer_ids is None else bool(return_kmer_ids)

        if truncation and len(input_ids) > effective_max:
            input_ids = input_ids[:effective_max]
            attention_mask = attention_mask[:effective_max]
            offsets = offsets[:effective_max]
            if input_ids and input_ids[-1] != self.sep_token_id:
                input_ids[-1] = self.sep_token_id
                attention_mask[-1] = 1
                offsets[-1] = (0, 0)

        kmer_ids = None
        if include_kmers:
            kmer_ids = self.kmer_ids_from_offsets(sequence, offsets)

        if padding and len(input_ids) < effective_max:
            pad_len = effective_max - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            offsets.extend([(0, 0)] * pad_len)
            if kmer_ids is not None:
                kmer_ids.extend([self.kmer_pad_id] * pad_len)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if kmer_ids is not None:
            out["kmer_ids"] = kmer_ids
        if return_offsets:
            out["offsets"] = offsets

        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out

    def encode_with_kmers(self, sequence: str, max_length: int | None = None, **kwargs):
        """Backward-compatible alias for callers that request aligned k-mers."""
        kwargs.setdefault("return_tensors", None)
        kwargs.setdefault("return_kmer_ids", True)
        return self.encode(sequence, max_length=max_length, **kwargs)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode base token IDs back into a continuous DNA sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid <= self.special_tokens_max_id:
                continue
            chars.append(self.id_to_char.get(tid, ""))
        return "".join(chars)

    def token_base_length(self, token_id: int) -> int:
        """Every non-special base token represents exactly one base."""
        return 1 if int(token_id) in self.char_to_id.values() else 0


def build_tokenizer(
    tokenizer_type: str = "bpe",
    *,
    vocab_path: str = "dna_model/bpe_vocab/tokenizer.json",
    **kwargs: object,
) -> DnaTokenizer | BaseTokenizer:
    """Construct a tokenizer from a CLI/config-friendly mode string."""
    mode = normalize_tokenizer_type(tokenizer_type)
    if mode == "base":
        return BaseTokenizer(**kwargs)
    return DnaTokenizer(vocab_path=vocab_path, **kwargs)

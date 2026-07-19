"""Evaluate a trained DNA masked-language model checkpoint."""

from __future__ import annotations

import argparse
import functools
import glob
import json
import logging
import math
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dna_model.checkpoint_loading import (
    config_from_checkpoint,
    load_state_dict_with_report,
    normalize_state_dict_keys,
)
from dna_model.dataset import DatasetImpl, collate, fast_reverse_complement
from dna_model.loaders import RoundRobinLoader
from dna_model.model import DnaModel
from dna_model.tokenizer import build_tokenizer, normalize_tokenizer_type, read_fasta_records
from dna_model.utils import evaluate, masked_mean_pool

LOGGER = logging.getLogger(__name__)


def _load_checkpoint(path: Path, tokenizer, device: torch.device) -> DnaModel:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Safe checkpoint loading requires a PyTorch version with weights_only support") from exc

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    config_values = checkpoint.get("model_config") or checkpoint.get("config")
    if not isinstance(config_values, Mapping):
        raise ValueError("Checkpoint does not contain a model configuration")
    config_values = dict(config_values)
    run_config = checkpoint.get("config")
    if "tokenizer_type" not in config_values and isinstance(run_config, Mapping):
        saved_mode = run_config.get("tokenizer_type")
        if saved_mode is not None:
            config_values["tokenizer_type"] = normalize_tokenizer_type(saved_mode)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain model weights")

    model = DnaModel(config_from_checkpoint(config_values, tokenizer))
    load_state_dict_with_report(
        model,
        normalize_state_dict_keys(state_dict),
        checkpoint_name=f"evaluation checkpoint {path}",
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def _expand_fasta_patterns(patterns: Sequence[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return sorted({os.path.abspath(path) for path in files if os.path.isfile(path)})


def _build_eval_loader(
    fasta_files: Sequence[str],
    tokenizer,
    *,
    max_length: int,
    stride: int,
    min_length: int,
    batch_size: int,
    workers: int,
) -> RoundRobinLoader:
    loaders = []
    names = []
    for fasta_path in fasta_files:
        if fasta_path.lower().endswith(".gz"):
            sequences = [
                sequence
                for _header, sequence, _is_last in read_fasta_records(fasta_path)
                if len(sequence) >= min_length
            ]
            source = sequences
            lazy_load = False
        else:
            source = fasta_path
            lazy_load = True

        dataset = DatasetImpl(
            tokenizer,
            source,
            max_length=max_length,
            stride=stride,
            min_chunk_length=min_length,
            lazy_load=lazy_load,
        )
        if len(dataset) == 0:
            LOGGER.warning("Skipping %s because it has no qualifying windows", fasta_path)
            continue
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=functools.partial(collate, max_length=max_length),
                persistent_workers=workers > 0,
                prefetch_factor=2 if workers > 0 else None,
            )
        )
        names.append(os.path.basename(fasta_path))

    if not loaders:
        raise ValueError("No qualifying evaluation windows were found")
    return RoundRobinLoader(loaders, names=names)


def _sample_sequences(
    fasta_files: Sequence[str],
    *,
    sample_size: int,
    min_length: int,
) -> list[str]:
    """Reservoir-sample qualifying records without loading the corpus into RAM."""

    if sample_size <= 0:
        return []
    rng = random.Random(42)
    samples: list[str] = []
    seen = 0
    for fasta_path in fasta_files:
        for _header, sequence, _is_last in read_fasta_records(fasta_path):
            if len(sequence) < min_length:
                continue
            seen += 1
            if len(samples) < sample_size:
                samples.append(sequence)
                continue
            replacement = rng.randrange(seen)
            if replacement < sample_size:
                samples[replacement] = sequence
    return samples


@torch.inference_mode()
def rc_embedding_agreement(
    model: DnaModel,
    tokenizer,
    sequences: Sequence[str],
    device: torch.device,
    *,
    max_length: int,
) -> dict[str, float | int]:
    """Compare mean-pooled embeddings for sequences and reverse complements."""

    cosine_values = []
    normalized_l2_values = []
    for sequence in sequences:
        sequence = sequence[:max_length]
        encoded_pairs = [
            tokenizer.encode(sequence, max_length=max_length),
            tokenizer.encode(fast_reverse_complement(sequence), max_length=max_length),
        ]
        pooled = []
        for encoded in encoded_pairs:
            input_ids = encoded["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
            kmer_ids = encoded.get("kmer_ids")
            if kmer_ids is not None:
                kmer_ids = kmer_ids.unsqueeze(0).to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                kmer_ids=kmer_ids,
            )
            hidden = output["last_hidden_state"]
            pooled.append(masked_mean_pool(hidden, attention_mask))

        cosine_values.append(F.cosine_similarity(pooled[0], pooled[1]).item())
        normalized_l2_values.append(
            torch.linalg.vector_norm(pooled[0] - pooled[1]).item() / math.sqrt(pooled[0].shape[-1])
        )

    if not cosine_values:
        return {"num_sequences": 0, "mean_cosine_similarity": float("nan"), "mean_normalized_l2": float("nan")}
    return {
        "num_sequences": len(cosine_values),
        "mean_cosine_similarity": sum(cosine_values) / len(cosine_values),
        "min_cosine_similarity": min(cosine_values),
        "max_cosine_similarity": max(cosine_values),
        "mean_normalized_l2": sum(normalized_l2_values) / len(normalized_l2_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate masked-token metrics and reverse-complement embedding agreement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint path")
    parser.add_argument("--fasta", required=True, nargs="+", help="FASTA paths or glob patterns")
    parser.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        type=normalize_tokenizer_type,
        default="bpe",
        choices=["bpe", "base"],
        help="Checkpoint tokenization granularity (single_base is accepted as a legacy alias for base)",
    )
    parser.add_argument("--tokenizer-path", "--tokenizer_path", default="dna_model/bpe_vocab/tokenizer.json")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--max-length", type=int, default=None, help="Evaluation window size; defaults to checkpoint config"
    )
    parser.add_argument("--stride", type=int, default=None, help="Window stride; defaults to max length")
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=500, help="Masked-token batches; 0 evaluates all batches")
    parser.add_argument("--rc-check-seqs", "--rc_check_seqs", type=int, default=100)
    parser.add_argument("--rc-max-length", type=int, default=512)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for evaluation; no CUDA device is available")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size <= 0 or args.workers < 0 or args.min_length <= 0:
        raise SystemExit("batch size and minimum length must be positive; workers must be non-negative")
    if args.rc_check_seqs < 0 or args.rc_max_length <= 0:
        raise SystemExit("RC sample count must be non-negative and RC max length must be positive")

    fasta_files = _expand_fasta_patterns(args.fasta)
    if not fasta_files:
        raise SystemExit(f"No FASTA files matched: {args.fasta}")

    device = torch.device("cuda")

    tokenizer = build_tokenizer(args.tokenizer_type, vocab_path=args.tokenizer_path)

    model = _load_checkpoint(args.checkpoint, tokenizer, device)
    model_config = model.config
    max_length = int(args.max_length or model_config.max_input_bases)
    stride = int(args.stride or max_length)
    if max_length <= 0 or stride <= 0:
        raise SystemExit("max length and stride must be positive")

    eval_loader = _build_eval_loader(
        fasta_files,
        tokenizer,
        max_length=max_length,
        stride=stride,
        min_length=args.min_length,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    masked_metrics = evaluate(
        model,
        eval_loader,
        device,
        args.amp,
        epoch=0,
        tok=tokenizer,
        max_batches=args.max_batches,
    )

    rc_sequences = _sample_sequences(
        fasta_files,
        sample_size=args.rc_check_seqs,
        min_length=args.min_length,
    )
    rc_metrics = rc_embedding_agreement(
        model,
        tokenizer,
        rc_sequences,
        device,
        max_length=args.rc_max_length,
    )

    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "fasta_files": len(fasta_files),
        "masked_token": masked_metrics,
        "reverse_complement": rc_metrics,
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import os
import sys
from collections.abc import Mapping

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dna_model.benchmarks import (  # noqa: E402
    bpe_rc_diagnostics,
    build_negative_control_records,
    run_linear_probe_benchmark,
)
from dna_model.checkpoint_loading import (  # noqa: E402
    config_from_checkpoint,
    load_state_dict_with_report,
    normalize_state_dict_keys,
)
from dna_model.model import DnaModel  # noqa: E402
from dna_model.tokenizer import build_tokenizer, normalize_tokenizer_type, read_fasta_records  # noqa: E402


def _clean_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _label_from_header(header: str, fallback: str) -> str:
    for part in header.replace("|", " ").split():
        if "=" in part:
            key, value = part.split("=", 1)
            if _clean_key(key) in {"label", "class", "species", "taxon", "task"}:
                return value
    return fallback


def load_labeled_fasta(path: str, default_label: str = None, max_records: int = 0) -> list[tuple[str, str]]:
    records = []
    fallback = default_label or os.path.splitext(os.path.basename(path))[0]
    for header, seq, _is_last in read_fasta_records(path):
        records.append((seq, _label_from_header(header, fallback)))
        if max_records and len(records) >= max_records:
            break
    return records


def load_model(checkpoint_path: str, tokenizer, device: torch.device) -> DnaModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg_dict = ckpt.get("model_config") or ckpt.get("config")
    if not isinstance(cfg_dict, Mapping):
        raise ValueError("Checkpoint does not contain a model configuration")
    cfg = config_from_checkpoint(cfg_dict, tokenizer)
    model = DnaModel(cfg)
    state = ckpt.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain model weights")
    load_state_dict_with_report(
        model,
        normalize_state_dict_keys(state),
        checkpoint_name=f"biology benchmark checkpoint {checkpoint_path}",
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Biological embedding benchmarks for DNA checkpoints")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fasta", action="append", required=True, help="Labeled FASTA. Repeat for multiple classes.")
    ap.add_argument("--label", action="append", default=None, help="Optional label for matching --fasta entry.")
    ap.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        type=normalize_tokenizer_type,
        choices=["bpe", "base"],
        default="bpe",
    )
    ap.add_argument("--tokenizer_path", default="dna_model/bpe_vocab/tokenizer.json")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_records_per_file", type=int, default=0)
    ap.add_argument("--negative_control", choices=["none", "dinuc", "gc"], default="dinuc")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for biology benchmarking; no CUDA device is available.")
    device = torch.device("cuda")
    tokenizer = build_tokenizer(
        args.tokenizer_type,
        vocab_path=args.tokenizer_path,
        max_length=args.max_length,
    )
    model = load_model(args.checkpoint, tokenizer, device)

    labels = args.label or []
    records = []
    for i, path in enumerate(args.fasta):
        label = labels[i] if i < len(labels) else None
        records.extend(load_labeled_fasta(path, default_label=label, max_records=args.max_records_per_file))

    if args.negative_control != "none" and len({label for _seq, label in records}) == 1:
        records = build_negative_control_records([seq for seq, _label in records], mode=args.negative_control)

    rc_diag = bpe_rc_diagnostics(tokenizer, [seq for seq, _label in records[: min(100, len(records))]], args.max_length)
    probe = run_linear_probe_benchmark(
        model,
        tokenizer,
        records,
        device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    results = {
        "rc_tokenization": rc_diag,
        "linear_probe": probe,
    }
    print(json.dumps(results, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

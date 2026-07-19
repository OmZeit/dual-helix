"""Manual DualHelix attention heatmaps for FASTA sequences.

This script is intentionally separate from training. It captures Telescope
attention only when launched directly, then projects compressed token attention
into base-coordinate bins using tokenizer offsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_index_selection(
    spec: str,
    *,
    max_count: int,
    name: str,
    single: int | None = None,
) -> list[int] | None:
    """Parse ``all`` or comma-separated zero-based indices."""
    if single is not None:
        values = [single]
    elif spec.strip().lower() == "all":
        return None
    else:
        values = [int(part.strip()) for part in spec.split(",") if part.strip()]

    bad = [value for value in values if value < 0 or value >= max_count]
    if bad:
        raise ValueError(f"{name} index out of range for max_count={max_count}: {bad}")
    return sorted(set(values))


def compressed_base_spans(
    offsets: Sequence[Sequence[int]],
    attention_mask: Sequence[int],
    *,
    compressed_length: int,
    compression_stride: int = 4,
) -> list[tuple[int, int]]:
    """Map compressed Telescope positions back to raw base spans.

    Args:
        offsets: Token offsets, one ``(start, end)`` pair per token.
        attention_mask: Token mask with ``1`` for valid tokens and ``0`` for padding.
        compressed_length: Number of positions in the Telescope attention matrix.
        compression_stride: Token stride used by DualHelix ``PositionAwareCompress``.

    Returns:
        Base-coordinate spans for each compressed position. Empty/special-only
        compressed positions are returned as ``(0, 0)``.
    """
    spans: list[tuple[int, int]] = []
    for compressed_idx in range(compressed_length):
        token_start = compressed_idx * compression_stride
        token_end = min(token_start + compression_stride, len(offsets))
        starts: list[int] = []
        ends: list[int] = []
        for token_idx in range(token_start, token_end):
            if token_idx >= len(attention_mask) or not attention_mask[token_idx]:
                continue
            start, end = int(offsets[token_idx][0]), int(offsets[token_idx][1])
            if end > start:
                starts.append(start)
                ends.append(end)
        spans.append((min(starts), max(ends)) if starts else (0, 0))
    return spans


def compressed_position_summary(
    offsets: Sequence[Sequence[int]],
    attention_mask: Sequence[int],
    input_ids: Sequence[int],
    *,
    special_token_ids: Sequence[int],
    compressed_length: int,
    compression_stride: int = 4,
) -> dict[str, Any]:
    """Summarize which compressed positions represent DNA bases or token sinks."""
    special_ids = {int(token_id) for token_id in special_token_ids}
    spans: list[tuple[int, int]] = []
    special_only_positions: list[int] = []
    mixed_special_positions: list[int] = []
    padding_positions: list[int] = []

    for compressed_idx in range(compressed_length):
        token_start = compressed_idx * compression_stride
        token_end = min(token_start + compression_stride, len(offsets))
        starts: list[int] = []
        ends: list[int] = []
        valid_tokens = 0
        special_or_empty_tokens = 0

        for token_idx in range(token_start, token_end):
            if token_idx >= len(attention_mask) or not attention_mask[token_idx]:
                continue

            valid_tokens += 1
            token_id = int(input_ids[token_idx]) if token_idx < len(input_ids) else -1
            start, end = int(offsets[token_idx][0]), int(offsets[token_idx][1])
            has_base_span = end > start

            if token_id in special_ids or not has_base_span:
                special_or_empty_tokens += 1
            if has_base_span:
                starts.append(start)
                ends.append(end)

        spans.append((min(starts), max(ends)) if starts else (0, 0))
        if valid_tokens == 0:
            padding_positions.append(compressed_idx)
        elif not starts:
            special_only_positions.append(compressed_idx)
        elif special_or_empty_tokens:
            mixed_special_positions.append(compressed_idx)

    return {
        "spans": spans,
        "special_only_positions": special_only_positions,
        "mixed_special_positions": mixed_special_positions,
        "padding_positions": padding_positions,
    }


def mask_attention_positions(attention: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """Remove selected compressed positions as attention sources and targets."""
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(f"attention must be square, got shape={attention.shape}")

    length = attention.shape[0]
    valid_positions = sorted({int(pos) for pos in positions if 0 <= int(pos) < length})
    if not valid_positions:
        return attention.astype(np.float32, copy=True)

    masked = attention.astype(np.float32, copy=True)
    masked[valid_positions, :] = 0.0
    masked[:, valid_positions] = 0.0
    row_sums = masked.sum(axis=1, keepdims=True)
    return np.divide(masked, row_sums, out=np.zeros_like(masked), where=row_sums > 0)


def base_projection_matrix(
    spans: Sequence[tuple[int, int]],
    *,
    sequence_length: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a compressed-position to base-bin overlap projection matrix."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    bin_edges = np.linspace(0, sequence_length, num_bins + 1, dtype=np.float64)
    projection = np.zeros((len(spans), num_bins), dtype=np.float32)
    for span_idx, (raw_start, raw_end) in enumerate(spans):
        start = max(0.0, min(float(sequence_length), float(raw_start)))
        end = max(0.0, min(float(sequence_length), float(raw_end)))
        if end <= start:
            continue
        span_len = end - start
        for bin_idx in range(num_bins):
            overlap = max(0.0, min(end, bin_edges[bin_idx + 1]) - max(start, bin_edges[bin_idx]))
            if overlap > 0:
                projection[span_idx, bin_idx] = float(overlap / span_len)
    return projection, bin_edges


def project_attention_to_base_bins(
    attention: np.ndarray,
    spans: Sequence[tuple[int, int]],
    *,
    sequence_length: int,
    max_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project compressed attention into uniform base-coordinate bins."""
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError(f"attention must be square, got shape={attention.shape}")

    length = min(attention.shape[0], len(spans))
    if length == 0:
        raise ValueError("attention matrix has no positions")
    attention = attention[:length, :length].astype(np.float32, copy=False)
    spans = list(spans[:length])

    num_bins = min(max_bins, max(1, sequence_length))
    projection, bin_edges = base_projection_matrix(spans, sequence_length=sequence_length, num_bins=num_bins)
    binned = projection.T @ attention @ projection
    row_sums = binned.sum(axis=1, keepdims=True)
    binned = np.divide(binned, row_sums, out=np.zeros_like(binned), where=row_sums > 0)
    return binned.astype(np.float32, copy=False), bin_edges


def mask_edge_bins(matrix: np.ndarray, edge_bins: int) -> np.ndarray:
    """Hide leading and trailing base bins in the visible/reporting matrix."""
    if edge_bins <= 0:
        return matrix.astype(np.float32, copy=True)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={matrix.shape}")

    rows, cols = matrix.shape
    masked = matrix.astype(np.float32, copy=True)
    row_edge = min(edge_bins, rows // 2)
    col_edge = min(edge_bins, cols // 2)
    if row_edge:
        masked[:row_edge, :] = np.nan
        masked[-row_edge:, :] = np.nan
    if col_edge:
        masked[:, :col_edge] = np.nan
        masked[:, -col_edge:] = np.nan
    return masked


def aggregate_attention(attention_maps: Sequence[dict[str, Any]]) -> np.ndarray:
    """Average captured attention over selected layers and heads."""
    if not attention_maps:
        raise ValueError("no attention maps were captured")

    total: Any | None = None
    count = 0
    for record in attention_maps:
        weights = record["weights"]
        if weights.ndim != 4:
            raise ValueError(f"expected [batch, heads, length, length] weights, got {tuple(weights.shape)}")
        if weights.size(0) != 1:
            raise ValueError("visualize_interactions processes one sequence at a time")
        if weights.size(1) == 0:
            raise ValueError("selected heads did not match any captured attention heads")
        matrix = weights[0].mean(dim=0)
        total = matrix if total is None else total + matrix
        count += 1

    if total is None or count == 0:
        raise ValueError("no attention maps were captured")
    return (total / count).numpy()


def load_checkpoint_model(checkpoint_path: str, tokenizer: Any, device: Any) -> Any:
    """Load a True DNA checkpoint for manual visualization."""
    import torch
    from dna_model.checkpoint_loading import (
        config_from_checkpoint,
        load_state_dict_with_report,
        normalize_state_dict_keys,
    )
    from dna_model.model import DnaModel

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg_dict = None
    if isinstance(ckpt, Mapping):
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
        checkpoint_name=f"interaction visualization checkpoint {checkpoint_path}",
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def select_device(device_arg: str) -> Any:
    """Resolve a CLI device argument."""
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_stem(header: str, index: int) -> str:
    """Create a filesystem-safe output stem for one FASTA record."""
    label = header.split()[0] if header else f"sequence_{index}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    return f"{index:04d}_{label[:80] or 'sequence'}"


def write_top_pairs_csv(
    path: Path,
    matrix: np.ndarray,
    bin_edges: np.ndarray,
    *,
    top_k: int,
) -> None:
    """Write the highest-scoring base-bin pairs for quick inspection."""
    flat = matrix.reshape(-1)
    valid_indices = np.flatnonzero(np.isfinite(flat))
    if valid_indices.size == 0 or top_k <= 0:
        return
    top_k = min(top_k, valid_indices.size)
    valid_scores = flat[valid_indices]
    local_top = np.argpartition(valid_scores, -top_k)[-top_k:]
    top_indices = valid_indices[local_top]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "source_bin",
                "source_start",
                "source_end",
                "target_bin",
                "target_start",
                "target_end",
                "score",
            ]
        )
        width = matrix.shape[1]
        for rank, flat_idx in enumerate(top_indices, start=1):
            source_bin = int(flat_idx // width)
            target_bin = int(flat_idx % width)
            writer.writerow(
                [
                    rank,
                    source_bin,
                    float(bin_edges[source_bin]),
                    float(bin_edges[source_bin + 1]),
                    target_bin,
                    float(bin_edges[target_bin]),
                    float(bin_edges[target_bin + 1]),
                    float(flat[flat_idx]),
                ]
            )


def _plot_matrix_and_norm(
    matrix: np.ndarray,
    *,
    color_scale: Literal["linear", "percentile", "log"],
    vmax_percentile: float,
) -> tuple[Any, Any, str]:
    """Prepare a matrix and matplotlib normalization for stable heatmap rendering."""
    from matplotlib import colors

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.ma.masked_invalid(matrix), None, "row-normalized attention"

    if color_scale == "linear":
        return np.ma.masked_invalid(matrix), None, "row-normalized attention"

    vmax_percentile = max(0.0, min(100.0, float(vmax_percentile)))
    if color_scale == "percentile":
        vmax = float(np.percentile(finite, vmax_percentile))
        if vmax <= 0.0:
            vmax = float(np.max(finite))
        norm = colors.Normalize(vmin=0.0, vmax=max(vmax, np.finfo(np.float32).eps), clip=True)
        return np.ma.masked_invalid(matrix), norm, f"row-normalized attention (vmax p{vmax_percentile:g})"

    positive = finite[finite > 0.0]
    if positive.size == 0:
        return np.ma.masked_invalid(matrix), None, "row-normalized attention"
    vmax = float(np.percentile(positive, vmax_percentile))
    vmin = float(np.min(positive))
    if vmax <= vmin:
        vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = vmin * 1.01
    norm = colors.LogNorm(vmin=max(vmin, np.finfo(np.float32).tiny), vmax=vmax, clip=True)
    masked = np.ma.masked_where(~np.isfinite(matrix) | (matrix <= 0.0), matrix)
    return masked, norm, f"row-normalized attention (log, vmax p{vmax_percentile:g})"


def save_heatmap_png(
    path: Path,
    matrix: np.ndarray,
    *,
    title: str,
    color_scale: Literal["linear", "percentile", "log"],
    vmax_percentile: float,
    hidden_edge_bins: int,
    annotation: str | None = None,
) -> None:
    """Save a binned region-to-region attention heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_matrix, norm, colorbar_label = _plot_matrix_and_norm(
        matrix,
        color_scale=color_scale,
        vmax_percentile=vmax_percentile,
    )

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(plot_matrix, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    display_title = title[:120]
    if annotation:
        display_title = f"{display_title}\n{annotation[:160]}"
    ax.set_title(display_title)
    ax.set_xlabel("Target base bin")
    ax.set_ylabel("Source base bin")
    if hidden_edge_bins > 0:
        rows, cols = matrix.shape
        row_edge = min(hidden_edge_bins, rows // 2)
        col_edge = min(hidden_edge_bins, cols // 2)
        for x_pos in (col_edge - 0.5, cols - col_edge - 0.5):
            if 0.0 < x_pos < cols - 1:
                ax.axvline(x_pos, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
        for y_pos in (row_edge - 0.5, rows - row_edge - 0.5):
            if 0.0 < y_pos < rows - 1:
                ax.axhline(y_pos, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def visualize_sequence(
    *,
    model: Any,
    tokenizer: Any,
    header: str,
    sequence: str,
    device: Any,
    max_length: int,
    max_heatmap_bins: int,
    layers: list[int] | None,
    heads: list[int] | None,
    out_dir: Path,
    index: int,
    top_k_pairs: int,
    color_scale: Literal["linear", "percentile", "log"],
    vmax_percentile: float,
    mask_special_positions: bool,
    mask_mixed_special_positions: bool,
    hide_edge_bins: int,
) -> dict[str, Any]:
    """Capture and save one sequence's region-to-region attention heatmap."""
    import torch

    encoded = tokenizer.encode(sequence, max_length=max_length, return_tensors=None, return_offsets=True)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=device)
    kmer_ids = None
    if "kmer_ids" in encoded:
        kmer_ids = torch.tensor([encoded["kmer_ids"]], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kmer_ids=kmer_ids,
            capture_attention=True,
            attention_layers=layers,
            attention_heads=heads,
        )
    compressed_attention_raw = aggregate_attention(outputs["attention_maps"])
    special_token_ids = [
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    ]
    position_summary = compressed_position_summary(
        encoded["offsets"],
        encoded["attention_mask"],
        encoded["input_ids"],
        special_token_ids=special_token_ids,
        compressed_length=compressed_attention_raw.shape[0],
        compression_stride=4,
    )
    spans = position_summary["spans"]

    masked_positions: list[int] = []
    if mask_special_positions:
        masked_positions.extend(position_summary["special_only_positions"])
        masked_positions.extend(position_summary["padding_positions"])
    if mask_mixed_special_positions:
        masked_positions.extend(position_summary["mixed_special_positions"])
    masked_positions = sorted(set(masked_positions))
    compressed_attention = mask_attention_positions(compressed_attention_raw, masked_positions)

    raw_sequence_length = len(sequence)
    encoded_base_length = max(
        (
            int(end)
            for (start, end), keep in zip(encoded["offsets"], encoded["attention_mask"], strict=False)
            if keep and int(end) > int(start)
        ),
        default=raw_sequence_length,
    )
    binned_attention, bin_edges = project_attention_to_base_bins(
        compressed_attention,
        spans,
        sequence_length=encoded_base_length,
        max_bins=max_heatmap_bins,
    )
    visible_attention = mask_edge_bins(binned_attention, hide_edge_bins)

    stem = safe_stem(header, index)
    png_path = out_dir / f"{stem}.png"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}_top_pairs.csv"

    annotation_parts = []
    if masked_positions:
        annotation_parts.append(f"masked compressed sinks={len(masked_positions)}")
    if position_summary["mixed_special_positions"] and not mask_mixed_special_positions:
        annotation_parts.append(
            f"mixed special/base compressed positions={len(position_summary['mixed_special_positions'])}"
        )
    if hide_edge_bins:
        annotation_parts.append(f"hidden edge bins={hide_edge_bins}")
    annotation = "; ".join(annotation_parts) if annotation_parts else None

    save_heatmap_png(
        png_path,
        visible_attention,
        title=header or stem,
        color_scale=color_scale,
        vmax_percentile=vmax_percentile,
        hidden_edge_bins=hide_edge_bins,
        annotation=annotation,
    )
    np.savez_compressed(
        npz_path,
        attention_bins=binned_attention,
        visible_attention_bins=visible_attention,
        bin_edges=bin_edges,
        compressed_spans=np.asarray(spans, dtype=np.int64),
        compressed_attention=compressed_attention,
        compressed_attention_raw=compressed_attention_raw,
    )
    write_top_pairs_csv(csv_path, visible_attention, bin_edges, top_k=top_k_pairs)

    metadata = {
        "header": header,
        "raw_sequence_length": raw_sequence_length,
        "encoded_base_length": encoded_base_length,
        "max_length": max_length,
        "token_count": int(sum(encoded["attention_mask"])),
        "compressed_attention_shape": list(compressed_attention.shape),
        "binned_attention_shape": list(binned_attention.shape),
        "layers": layers if layers is not None else "all",
        "heads": heads if heads is not None else "all",
        "aggregation": "mean_over_selected_layers_and_heads",
        "projection": "base_span_overlap_then_row_normalized",
        "compression_stride_tokens": 4,
        "attention_sink_handling": {
            "mask_special_positions": mask_special_positions,
            "mask_mixed_special_positions": mask_mixed_special_positions,
            "special_only_positions": position_summary["special_only_positions"],
            "mixed_special_positions": position_summary["mixed_special_positions"],
            "padding_positions": position_summary["padding_positions"],
            "masked_positions": masked_positions,
        },
        "display": {
            "color_scale": color_scale,
            "vmax_percentile": vmax_percentile,
            "hidden_edge_bins": hide_edge_bins,
            "hidden_values_are_nan_in_visible_attention_bins": hide_edge_bins > 0,
        },
        "png": str(png_path),
        "npz": str(npz_path),
        "top_pairs_csv": str(csv_path),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize manual DualHelix region-to-region attention heatmaps")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--fasta", required=True, help="Input FASTA path")
    parser.add_argument("--out_dir", default="experiments/interaction_heatmaps", help="Output directory")
    parser.add_argument(
        "--tokenizer-type",
        "--tokenizer_type",
        choices=["bpe", "base", "single_base"],
        default="bpe",
        help="Primary tokenization granularity",
    )
    parser.add_argument("--tokenizer_path", default="dna_model/bpe_vocab/tokenizer.json")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of sampled sequences")
    parser.add_argument("--sample_every", type=int, default=1, help="Use every Nth FASTA record")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum primary-token length")
    parser.add_argument("--max_heatmap_bins", type=int, default=256, help="Maximum base-coordinate bins per axis")
    parser.add_argument("--layers", default="all", help="all or comma-separated zero-based layer indices")
    parser.add_argument("--layer", type=int, default=None, help="Single zero-based layer; overrides --layers")
    parser.add_argument("--heads", default="all", help="all or comma-separated zero-based head indices")
    parser.add_argument("--head", type=int, default=None, help="Single zero-based head; overrides --heads")
    parser.add_argument("--top_k_pairs", type=int, default=25, help="Number of high-scoring bin pairs to write")
    parser.add_argument(
        "--color_scale",
        choices=["linear", "percentile", "log"],
        default="log",
        help="PNG color scaling; NPZ always stores unclipped values",
    )
    parser.add_argument(
        "--vmax_percentile",
        type=float,
        default=99.0,
        help="Upper percentile for percentile/log PNG color scaling",
    )
    parser.add_argument(
        "--mask_special_positions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask compressed positions that contain only special/pad/empty-offset tokens",
    )
    parser.add_argument(
        "--mask_mixed_special_positions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask compressed positions that mix DNA-token spans with special tokens",
    )
    parser.add_argument(
        "--hide_edge_bins",
        type=int,
        default=1,
        help="Hide N leading/trailing base bins in PNG and top-pairs CSV",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_size <= 0:
        raise SystemExit("--sample_size must be positive")
    if args.sample_every <= 0:
        raise SystemExit("--sample_every must be positive")
    if args.max_heatmap_bins <= 0:
        raise SystemExit("--max_heatmap_bins must be positive")
    if args.hide_edge_bins < 0:
        raise SystemExit("--hide_edge_bins must be non-negative")
    if not 0.0 < args.vmax_percentile <= 100.0:
        raise SystemExit("--vmax_percentile must be in (0, 100]")

    device = select_device(args.device)
    from dna_model.tokenizer import build_tokenizer, read_fasta_records

    tokenizer = build_tokenizer(
        args.tokenizer_type,
        vocab_path=args.tokenizer_path,
        max_length=args.max_length,
    )
    model = load_checkpoint_model(args.checkpoint, tokenizer, device)
    if model.config.backbone != "dual_helix":
        raise SystemExit("visualize_interactions currently supports only the dual_helix backbone")

    layers = parse_index_selection(
        args.layers,
        max_count=model.config.high_level_layers,
        name="layer",
        single=args.layer,
    )
    heads = parse_index_selection(
        args.heads,
        max_count=model.config.num_attention_heads,
        name="head",
        single=args.head,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    seen = 0
    for record_idx, (header, sequence, _is_last) in enumerate(read_fasta_records(args.fasta)):
        if record_idx % args.sample_every != 0:
            continue
        seen += 1
        metadata = visualize_sequence(
            model=model,
            tokenizer=tokenizer,
            header=header,
            sequence=sequence,
            device=device,
            max_length=args.max_length,
            max_heatmap_bins=args.max_heatmap_bins,
            layers=layers,
            heads=heads,
            out_dir=out_dir,
            index=record_idx,
            top_k_pairs=args.top_k_pairs,
            color_scale=args.color_scale,
            vmax_percentile=args.vmax_percentile,
            mask_special_positions=args.mask_special_positions,
            mask_mixed_special_positions=args.mask_mixed_special_positions,
            hide_edge_bins=args.hide_edge_bins,
        )
        results.append(metadata)
        print(f"[{seen}/{args.sample_size}] wrote {metadata['png']}", flush=True)
        if seen >= args.sample_size:
            break

    manifest = {
        "checkpoint": args.checkpoint,
        "fasta": args.fasta,
        "sample_size": args.sample_size,
        "sample_every": args.sample_every,
        "layers": layers if layers is not None else "all",
        "heads": heads if heads is not None else "all",
        "color_scale": args.color_scale,
        "vmax_percentile": args.vmax_percentile,
        "mask_special_positions": args.mask_special_positions,
        "mask_mixed_special_positions": args.mask_mixed_special_positions,
        "hide_edge_bins": args.hide_edge_bins,
        "records": results,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

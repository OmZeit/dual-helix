import torch

from .config import (
    CLS_TOKEN_ID,
    DEFAULT_MASK_FRACTION,
    DEFAULT_MEAN_SPAN_LENGTH,
    IGNORE_INDEX,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    SPECIAL_TOKENS_MAX_ID,
    UNK_TOKEN_ID,
)


def bpe_masking(
    input_ids,
    vocab_size,
    mask_prob=0.15,
    mask_token_id=4,
    special_tokens_max_id=4,
    generator: torch.Generator | None = None,
    motif_importance_weights: dict[int, float] | torch.Tensor | None = None,
    replacement_strategy: str = "mask",
):
    """Apply independent-token masking to BPE token IDs."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2 [B, L], got shape={tuple(input_ids.shape)}")
    if not (0.0 <= float(mask_prob) <= 1.0):
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    if special_tokens_max_id >= 0 and vocab_size <= special_tokens_max_id:
        raise ValueError(f"vocab_size ({vocab_size}) must be > special_tokens_max_id ({special_tokens_max_id})")
    if not (0 <= int(mask_token_id) < int(vocab_size)):
        raise ValueError(f"mask_token_id ({mask_token_id}) must be in [0, {vocab_size})")
    if replacement_strategy == "bert":
        replacement_strategy = "bert_80_10_10"
    if replacement_strategy not in {"mask", "bert_80_10_10"}:
        raise ValueError("replacement_strategy must be 'mask' or 'bert_80_10_10'")

    x_masked = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.long)
    probability_matrix = torch.full(input_ids.shape, float(mask_prob), device=input_ids.device, dtype=torch.float)

    if motif_importance_weights is not None:
        if isinstance(motif_importance_weights, dict):
            motif_importance_weights = {int(k): v for k, v in motif_importance_weights.items()}
            weight_tensor = torch.ones(int(vocab_size), dtype=torch.float, device=input_ids.device)
            valid_ids = [k for k in motif_importance_weights if 0 <= k < int(vocab_size)]
            if valid_ids:
                idx_t = torch.tensor(valid_ids, dtype=torch.long, device=input_ids.device)
                val_t = torch.tensor(
                    [float(motif_importance_weights[k]) for k in valid_ids],
                    dtype=torch.float,
                    device=input_ids.device,
                )
                weight_tensor.index_put_((idx_t,), val_t)
        else:
            weight_tensor = motif_importance_weights.to(device=input_ids.device, dtype=torch.float)
            if weight_tensor.ndim != 1 or weight_tensor.shape[0] != int(vocab_size):
                raise ValueError(
                    f"motif_importance_weights tensor must have shape [vocab_size] == [{vocab_size}], "
                    f"got {tuple(weight_tensor.shape)}"
                )

        safe_ids = input_ids.clamp(0, int(vocab_size) - 1)
        probability_matrix = (probability_matrix * weight_tensor[safe_ids]).clamp(min=0.0, max=1.0)

    if special_tokens_max_id >= 0:
        special_tokens_mask = input_ids <= special_tokens_max_id
    else:
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < probability_matrix
    labels[masked_indices] = input_ids[masked_indices]

    if replacement_strategy == "mask":
        x_masked[masked_indices] = mask_token_id
    else:
        replacement_draw = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
        indices_replaced = masked_indices & (replacement_draw < 0.8)
        x_masked[indices_replaced] = mask_token_id

        indices_random = masked_indices & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
        if indices_random.any():
            valid_token_ids = torch.arange(0, int(vocab_size), dtype=input_ids.dtype, device=input_ids.device)
            if special_tokens_max_id >= 0:
                valid_token_ids = valid_token_ids[valid_token_ids > int(special_tokens_max_id)]
            for token_id in (PAD_TOKEN_ID, UNK_TOKEN_ID, CLS_TOKEN_ID, SEP_TOKEN_ID, int(mask_token_id)):
                valid_token_ids = valid_token_ids[valid_token_ids != int(token_id)]

            if valid_token_ids.numel() == 0:
                x_masked[indices_random] = mask_token_id
            else:
                random_idx = torch.randint(
                    0,
                    int(valid_token_ids.numel()),
                    input_ids.shape,
                    dtype=torch.long,
                    device=input_ids.device,
                    generator=generator,
                )
                random_words = valid_token_ids[random_idx]
                x_masked[indices_random] = random_words[indices_random]

    return x_masked, labels


def _span_mask_bool(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_fraction: float,
    mean_span_length: float,
    special_tokens_max_id: int,
    vocab_size: int,
    generator: torch.Generator | None,
    motif_importance_weights: dict[int, float] | torch.Tensor | None,
) -> torch.Tensor:
    """Build a boolean mask made of contiguous token spans."""
    eligible = attention_mask.bool()
    if special_tokens_max_id >= 0:
        eligible = eligible & (input_ids > int(special_tokens_max_id))

    weights_by_token = None
    if motif_importance_weights is not None:
        if isinstance(motif_importance_weights, dict):
            weights_by_token = torch.ones(int(vocab_size), dtype=torch.float, device=input_ids.device)
            for key, val in motif_importance_weights.items():
                key_i = int(key)
                if 0 <= key_i < int(vocab_size):
                    weights_by_token[key_i] = max(0.0, float(val))
        else:
            weights_by_token = motif_importance_weights.to(device=input_ids.device, dtype=torch.float)
            if weights_by_token.ndim != 1 or weights_by_token.shape[0] != int(vocab_size):
                raise ValueError(
                    f"motif_importance_weights tensor must have shape [vocab_size] == [{vocab_size}], "
                    f"got {tuple(weights_by_token.shape)}"
                )

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    span_len = max(1, int(round(float(mean_span_length))))

    for row in range(input_ids.size(0)):
        row_eligible = eligible[row]
        valid_count = int(row_eligible.sum().item())
        if valid_count == 0:
            continue

        target = int(round(valid_count * float(mask_fraction)))
        target = max(1 if mask_fraction > 0.0 else 0, min(valid_count, target))
        if target == 0:
            continue

        attempts = 0
        max_attempts = max(valid_count * 4, target * 8, 16)
        while int(mask[row].sum().item()) < target and attempts < max_attempts:
            attempts += 1
            candidates = row_eligible & ~mask[row]
            if not candidates.any():
                break

            candidate_positions = torch.nonzero(candidates, as_tuple=False).flatten()
            if weights_by_token is not None:
                token_ids = input_ids[row, candidate_positions].clamp(0, int(vocab_size) - 1)
                probs = weights_by_token[token_ids].clamp_min(0.0)
                if probs.sum() > 0:
                    pick = torch.multinomial(probs, 1, generator=generator).item()
                    start = int(candidate_positions[pick].item())
                else:
                    pick = torch.randint(
                        0,
                        int(candidate_positions.numel()),
                        (1,),
                        device=input_ids.device,
                        generator=generator,
                    ).item()
                    start = int(candidate_positions[pick].item())
            else:
                pick = torch.randint(
                    0,
                    int(candidate_positions.numel()),
                    (1,),
                    device=input_ids.device,
                    generator=generator,
                ).item()
                start = int(candidate_positions[pick].item())

            added = 0
            pos = start
            while (
                pos < input_ids.size(1)
                and added < span_len
                and int(mask[row].sum().item()) < target
                and bool(row_eligible[pos].item())
            ):
                if not bool(mask[row, pos].item()):
                    mask[row, pos] = True
                    added += 1
                pos += 1

        if int(mask[row].sum().item()) < target:
            remaining = torch.nonzero(row_eligible & ~mask[row], as_tuple=False).flatten()
            need = min(target - int(mask[row].sum().item()), int(remaining.numel()))
            if need > 0:
                order = torch.randperm(int(remaining.numel()), device=input_ids.device, generator=generator)
                mask[row, remaining[order[:need]]] = True

    return mask


def _base_coordinate_mask_bool(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_offsets: torch.Tensor,
    *,
    mask_fraction: float,
    mean_span_length: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Select raw-base spans and map every overlapping lexical token to a target.

    Sampling happens on the original nucleotide coordinates rather than on
    tokenizer positions.  Given the same raw windows and generator seed, base
    and BPE tokenizers therefore receive the same intended corruption spans.
    A BPE token that straddles a span boundary is masked in full; callers should
    report the resulting token-covered base count as well as the requested
    base-coordinate fraction.
    """
    if token_offsets.ndim != 3 or token_offsets.shape[:2] != input_ids.shape or token_offsets.shape[-1] != 2:
        raise ValueError(
            "token_offsets must have shape [B, L, 2] matching input_ids; "
            f"got {tuple(token_offsets.shape)} for {tuple(input_ids.shape)}"
        )

    offsets = token_offsets.to(device=input_ids.device, dtype=torch.long)
    starts = offsets[..., 0]
    ends = offsets[..., 1]
    lexical = attention_mask.bool() & (input_ids > int(SPECIAL_TOKENS_MAX_ID)) & (ends > starts)
    max_base_length = int(torch.where(lexical, ends, torch.zeros_like(ends)).max().item())
    if max_base_length <= 0:
        return torch.zeros_like(input_ids, dtype=torch.bool)

    base_positions = torch.arange(max_base_length, device=input_ids.device).view(1, 1, -1)
    token_covers_base = (
        lexical.unsqueeze(-1) & (base_positions >= starts.unsqueeze(-1)) & (base_positions < ends.unsqueeze(-1))
    )
    base_attention = token_covers_base.any(dim=1).long()

    synthetic_ids = torch.full_like(base_attention, int(SPECIAL_TOKENS_MAX_ID) + 1)
    base_mask = _span_mask_bool(
        synthetic_ids,
        base_attention,
        mask_fraction=mask_fraction,
        mean_span_length=mean_span_length,
        special_tokens_max_id=-1,
        vocab_size=int(SPECIAL_TOKENS_MAX_ID) + 2,
        generator=generator,
        motif_importance_weights=None,
    )

    return (token_covers_base & base_mask.unsqueeze(1)).any(dim=-1)


def gpu_span_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    kmer_ids: torch.Tensor | None = None,
    kmer_k: int | None = None,
    kmer_pad_id: int | None = None,
    vocab_size: int = 4096,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
    mean_span_length: float = DEFAULT_MEAN_SPAN_LENGTH,
    generator: torch.Generator | None = None,
    min_frac: float | None = None,
    max_frac: float | None = None,
    motif_importance_weights: dict[int, float] | torch.Tensor | None = None,
    replacement_strategy: str = "mask",
    token_offsets: torch.Tensor | None = None,
    coordinate_system: str = "token",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Apply span masking to BPE token IDs.

    ``bpe_masking`` remains available for independent-token RoBERTa masking.
    This compatibility API honors ``mean_span_length`` and returns the same
    tuple shape used by training and eval utilities. ``replacement_strategy``
    defaults to ``"mask"``, replacing every selected target with ``[MASK]``.
    ``"bert_80_10_10"`` retains the legacy 80% mask, 10% random, and 10%
    unchanged policy as an explicit experimental control. ``"bert"`` remains
    accepted as a backward-compatible alias.
    """
    if kmer_ids is not None:
        if kmer_k is None or kmer_pad_id is None:
            raise ValueError("kmer_k and kmer_pad_id must be provided if kmer_ids is not None")

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} must match input_ids shape {tuple(input_ids.shape)}"
        )
    if replacement_strategy == "bert":
        replacement_strategy = "bert_80_10_10"
    if replacement_strategy not in {"bert_80_10_10", "mask"}:
        raise ValueError("replacement_strategy must be 'mask' or 'bert_80_10_10'")
    if coordinate_system not in {"token", "base"}:
        raise ValueError("coordinate_system must be 'token' or 'base'")
    if coordinate_system == "base" and token_offsets is None:
        raise ValueError("token_offsets are required for base-coordinate masking")

    # An explicit requested fraction must not be silently changed. Bounds are
    # optional caller controls, used by evaluation protocols that deliberately
    # constrain a schedule.
    lo = 0.0 if min_frac is None else float(min_frac)
    hi = 1.0 if max_frac is None else float(max_frac)
    if lo > hi:
        lo, hi = hi, lo

    bounded_mask_fraction = float(mask_fraction)
    bounded_mask_fraction = max(lo, min(hi, bounded_mask_fraction))
    bounded_mask_fraction = max(0.0, min(1.0, bounded_mask_fraction))

    if vocab_size <= MASK_TOKEN_ID:
        raise ValueError(f"vocab_size ({vocab_size}) must be > MASK_TOKEN_ID ({MASK_TOKEN_ID})")

    if coordinate_system == "base":
        if motif_importance_weights is not None:
            raise ValueError("motif importance weights are not supported with base-coordinate masking")
        mask_bool = _base_coordinate_mask_bool(
            input_ids,
            attention_mask,
            token_offsets,
            mask_fraction=bounded_mask_fraction,
            mean_span_length=mean_span_length,
            generator=generator,
        )
    else:
        mask_bool = _span_mask_bool(
            input_ids,
            attention_mask,
            mask_fraction=bounded_mask_fraction,
            mean_span_length=mean_span_length,
            special_tokens_max_id=SPECIAL_TOKENS_MAX_ID,
            vocab_size=int(vocab_size),
            generator=generator,
            motif_importance_weights=motif_importance_weights,
        )

    x_masked = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX, dtype=torch.long)
    labels[mask_bool] = input_ids[mask_bool]

    if replacement_strategy == "mask":
        x_masked[mask_bool] = MASK_TOKEN_ID
    else:
        replacement_draw = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
        indices_replaced = mask_bool & (replacement_draw < 0.8)
        x_masked[indices_replaced] = MASK_TOKEN_ID

        indices_random = mask_bool & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
        if indices_random.any():
            valid_token_ids = torch.arange(0, int(vocab_size), dtype=input_ids.dtype, device=input_ids.device)
            valid_token_ids = valid_token_ids[valid_token_ids > int(SPECIAL_TOKENS_MAX_ID)]
            for token_id in (PAD_TOKEN_ID, UNK_TOKEN_ID, CLS_TOKEN_ID, SEP_TOKEN_ID, MASK_TOKEN_ID):
                valid_token_ids = valid_token_ids[valid_token_ids != int(token_id)]

            if valid_token_ids.numel() == 0:
                x_masked[indices_random] = MASK_TOKEN_ID
            else:
                random_idx = torch.randint(
                    0,
                    int(valid_token_ids.numel()),
                    input_ids.shape,
                    dtype=torch.long,
                    device=input_ids.device,
                    generator=generator,
                )
                random_words = valid_token_ids[random_idx]
                x_masked[indices_random] = random_words[indices_random]

    labels = labels.masked_fill(attention_mask == 0, IGNORE_INDEX)
    mask_bool = labels != IGNORE_INDEX

    if kmer_ids is not None:
        kmer_ids = kmer_ids.clone()
        for offset in range(kmer_k):
            shifted_mask = mask_bool
            if offset > 0:
                shifted_mask = torch.cat(
                    [
                        mask_bool[:, offset:],
                        torch.zeros((mask_bool.size(0), offset), dtype=torch.bool, device=mask_bool.device),
                    ],
                    dim=1,
                )
            kmer_ids[shifted_mask] = kmer_pad_id

    return x_masked, labels, kmer_ids, mask_bool

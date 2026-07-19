import torch

from .config import (
    CLS_TOKEN_ID,
    DEFAULT_MASK_FRACTION,
    DEFAULT_MEAN_SPAN_LENGTH,
    IGNORE_INDEX,
    MASK_TOKEN_ID,
    MAX_MASK_FRACTION,
    MIN_MASK_FRACTION,
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
):
    """Apply RoBERTa-style masking to BPE token IDs."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2 [B, L], got shape={tuple(input_ids.shape)}")
    if not (0.0 <= float(mask_prob) <= 1.0):
        raise ValueError(f"mask_prob must be in [0, 1], got {mask_prob}")
    if special_tokens_max_id >= 0 and vocab_size <= special_tokens_max_id:
        raise ValueError(f"vocab_size ({vocab_size}) must be > special_tokens_max_id ({special_tokens_max_id})")
    if not (0 <= int(mask_token_id) < int(vocab_size)):
        raise ValueError(f"mask_token_id ({mask_token_id}) must be in [0, {vocab_size})")

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Apply span masking to BPE token IDs.

    ``bpe_masking`` remains available for independent-token RoBERTa masking.
    This compatibility API honors ``mean_span_length`` and returns the same
    tuple shape used by training and eval utilities.
    """
    if kmer_ids is not None:
        if kmer_k is None or kmer_pad_id is None:
            raise ValueError("kmer_k and kmer_pad_id must be provided if kmer_ids is not None")

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} must match input_ids shape {tuple(input_ids.shape)}"
        )

    lo = MIN_MASK_FRACTION if min_frac is None else float(min_frac)
    hi = MAX_MASK_FRACTION if max_frac is None else float(max_frac)
    if lo > hi:
        lo, hi = hi, lo

    bounded_mask_fraction = float(mask_fraction)
    bounded_mask_fraction = max(lo, min(hi, bounded_mask_fraction))
    bounded_mask_fraction = max(0.0, min(1.0, bounded_mask_fraction))

    if vocab_size <= MASK_TOKEN_ID:
        raise ValueError(f"vocab_size ({vocab_size}) must be > MASK_TOKEN_ID ({MASK_TOKEN_ID})")

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

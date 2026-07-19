import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .config import IGNORE_INDEX
from .masking import gpu_span_mask


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states while excluding padding positions."""

    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden_states must be [B, L, H] and attention_mask must be [B, L]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states and attention mask must share batch and sequence dimensions")
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def get_cosine_schedule_with_warmup_and_cooldown(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cooldown_steps: int = 0,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5,
):
    """Create a linear-warmup, cosine-decay schedule with optional cooldown."""

    if num_warmup_steps < 0 or num_training_steps <= 0 or num_cooldown_steps < 0:
        raise ValueError("scheduler step counts must be non-negative and training steps must be positive")
    if num_warmup_steps + num_cooldown_steps > num_training_steps:
        raise ValueError("warmup and cooldown cannot exceed total training steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        decay_steps = max(1, num_training_steps - num_warmup_steps)

        def cosine_ratio(step: int) -> float:
            progress = (step - num_warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        if num_cooldown_steps > 0:
            cooldown_start = num_training_steps - num_cooldown_steps
            if current_step >= cooldown_start:
                cooldown_progress = (current_step - cooldown_start) / num_cooldown_steps
                cooldown_progress = min(max(cooldown_progress, 0.0), 1.0)
                start_ratio = cosine_ratio(cooldown_start)
                return start_ratio + (min_lr_ratio - start_ratio) * cooldown_progress

        return max(min_lr_ratio, cosine_ratio(current_step))

    return LambdaLR(optimizer, lr_lambda)


def get_one_cycle_schedule(
    optimizer,
    max_lr: float,
    num_training_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
):
    """Create a one-cycle learning-rate schedule."""

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=num_training_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )


class EMA:
    """Maintain exponential moving averages of trainable parameters."""

    def __init__(self, model, decay=0.999, warmup_steps=2000):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    @staticmethod
    def _canonical_name(name: str) -> str:
        prefixes = ("module.", "_orig_mod.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
                    break
        return name

    def state_dict(self):
        """Return averaged parameters for checkpoint serialization."""

        return dict(self.shadow)

    def load_state_dict(self, state_dict):
        """Load averaged parameters while tolerating wrapper-only prefixes."""

        source_by_name = {self._canonical_name(name): value for name, value in state_dict.items()}
        target_by_name = {self._canonical_name(name): name for name in self.shadow}
        if source_by_name.keys() != target_by_name.keys():
            missing = sorted(target_by_name.keys() - source_by_name.keys())
            unexpected = sorted(source_by_name.keys() - target_by_name.keys())
            raise RuntimeError(
                f"EMA checkpoint parameter mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

        self.shadow = {
            target_name: source_by_name[canonical_name].detach().clone().to(self.shadow[target_name].device)
            for canonical_name, target_name in target_by_name.items()
        }

    def get_decay(self):
        """Compute decay with warmup."""
        if self.num_updates < self.warmup_steps:
            # Linear warmup
            return self.decay * (self.num_updates / self.warmup_steps)
        return self.decay

    def update(self, model):
        """Update shadow weights with current model weights."""
        self.num_updates += 1
        decay = self.get_decay()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    if self.shadow[name].device != param.device:
                        self.shadow[name] = self.shadow[name].to(param.device)
                    self.shadow[name].mul_(decay).add_(param.detach(), alpha=1 - decay)

    def apply_shadow(self, model):
        """Apply shadow weights to model (for evaluation)."""
        self.backup = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights from backup."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.copy_(self.backup[name])
        self.backup = {}


@torch.no_grad()
def evaluate(
    model,
    eval_loader,
    device,
    amp_enabled,
    epoch,
    tok,
    kmer_rc_tables=None,
    lambda_rc=0.0,
    max_batches: int | None = 500,
):
    """Evaluate masked-language modeling at the tokenizer's true granularity.

    Base tokenization reports base accuracy and per-base class accuracy. BPE
    reports exact-token accuracy plus loss normalized by the number of DNA
    bases represented by the masked target tokens; it does not mislabel the
    subset of one-character BPE tokens as base-level accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_target_bases = 0
    tokenizer_type = getattr(tok, "tokenizer_type", "bpe")
    if tokenizer_type not in {"base", "bpe"}:
        raise ValueError(f"Unsupported tokenizer type for evaluation: {tokenizer_type!r}")

    # Per-class accuracy is meaningful only when every target token is a base.
    base_names = ["A", "C", "G", "T"]
    base_token_ids = [tok.char_to_id.get(base) for base in base_names] if tokenizer_type == "base" else []
    per_class_correct = [0, 0, 0, 0]
    per_class_total = [0, 0, 0, 0]

    token_base_lengths = torch.tensor(
        [tok.token_base_length(token_id) for token_id in range(tok.vocab_size)],
        dtype=torch.long,
        device=device,
    )

    confidence_buckets = [0.0] * 10
    accuracy_buckets = [0.0] * 10
    counts_buckets = [0] * 10

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    eval_generator = torch.Generator(device=device).manual_seed(42 + epoch)

    max_steps = None if max_batches is None or max_batches <= 0 else int(max_batches)
    evaluated_batches = 0

    for batch_idx, batch in enumerate(eval_loader):
        if max_steps is not None and batch_idx >= max_steps:
            break
        evaluated_batches += 1

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        kmer_ids = batch.get("kmer_ids")
        if kmer_ids is not None:
            kmer_ids = kmer_ids.to(device, non_blocking=True)

        x_masked, labels, kmer_ids_masked, mask_bool = gpu_span_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kmer_ids=kmer_ids,
            kmer_k=getattr(tok, "k_mer_size", None),
            kmer_pad_id=getattr(tok, "kmer_pad_id", None),
            vocab_size=getattr(tok, "vocab_size", 4096),
            mask_fraction=0.20,
            mean_span_length=3.0,
            generator=eval_generator,
            min_frac=0.20,
            max_frac=0.25,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ):
            outputs = model(
                x_masked,
                attention_mask=attention_mask,
                kmer_ids=kmer_ids_masked,
            )
            logits = outputs.get("logits")
            if logits is None:
                raise RuntimeError("Model evaluation output does not contain logits")

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            valid_tokens = (labels != IGNORE_INDEX).sum()
            if valid_tokens.item() == 0:
                loss = torch.zeros((), device=loss.device)

        total_loss += float(loss.item())

        # Compute accuracy on masked tokens only
        pred = logits.argmax(dim=-1)
        masked = (labels != IGNORE_INDEX) & (attention_mask == 1)
        correct = (pred == labels) & masked

        total_correct += int(correct.sum().item())
        total_count += int(masked.sum().item())
        if masked.any():
            total_target_bases += int(token_base_lengths[labels[masked]].sum().item())

        # ECE Tracking
        if masked.any():
            m_logits = logits[masked]
            m_labels = labels[masked]

            probs = torch.softmax(m_logits, dim=-1)
            conf, pred_m = probs.max(dim=-1)
            correct_m = pred_m == m_labels

            b_idx = (conf * 10).long().clamp(max=9)
            for b_id in range(10):
                in_bucket = b_idx == b_id
                if in_bucket.any():
                    confidence_buckets[b_id] += conf[in_bucket].sum().item()
                    accuracy_buckets[b_id] += correct_m[in_bucket].sum().item()
                    counts_buckets[b_id] += in_bucket.sum().item()

        # Per-class accuracy for true single-base targets only.
        if tokenizer_type == "base":
            for i, token_id in enumerate(base_token_ids):
                class_mask = (labels == int(token_id)) & masked
                per_class_correct[i] += int((pred == int(token_id))[class_mask].sum().item())
                per_class_total[i] += int(class_mask.sum().item())

    if evaluated_batches == 0:
        raise RuntimeError("Evaluation loader produced no batches")
    if total_count == 0:
        raise RuntimeError("Evaluation produced no masked target tokens")

    token_loss = total_loss / total_count
    primary_accuracy = total_correct / total_count
    metrics = {
        "tokenizer_type": tokenizer_type,
        "metric_granularity": "base" if tokenizer_type == "base" else "bpe_token",
        "loss": token_loss,
        "token_loss": token_loss,
        "accuracy": primary_accuracy,
        "masked_tokens": total_count,
        "masked_bases": total_target_bases,
        "evaluated_batches": evaluated_batches,
        "max_batches": max_steps,
    }

    ece = 0.0
    if total_count > 0:
        for b_id in range(10):
            if counts_buckets[b_id] > 0:
                avg_conf = confidence_buckets[b_id] / counts_buckets[b_id]
                avg_acc = accuracy_buckets[b_id] / counts_buckets[b_id]
                ece += (counts_buckets[b_id] / total_count) * abs(avg_acc - avg_conf)
    metrics["ece"] = ece

    if tokenizer_type == "base":
        metrics["base_accuracy"] = primary_accuracy
        metrics["base_ece"] = ece
        metrics["base_loss"] = token_loss
        for c, name in enumerate(base_names):
            metrics[f"support_{name}"] = per_class_total[c]
            if per_class_total[c] > 0:
                metrics[f"accuracy_{name}"] = per_class_correct[c] / per_class_total[c]
    else:
        metrics["token_accuracy"] = primary_accuracy
        metrics["token_ece"] = ece
        if total_target_bases > 0:
            base_loss = total_loss / total_target_bases
            metrics["base_normalized_loss"] = base_loss
            metrics["bits_per_base"] = base_loss / math.log(2.0)
            try:
                metrics["base_perplexity"] = math.exp(base_loss)
            except OverflowError:
                metrics["base_perplexity"] = float("inf")

    metrics["acc"] = metrics["accuracy"]
    try:
        metrics["ppl"] = math.exp(token_loss)
    except OverflowError:
        metrics["ppl"] = float("inf")
    metrics["token_perplexity"] = metrics["ppl"]

    return metrics


def get_scheduler(optimizer, num_warmup_steps, num_training_steps, kind: str = "linear"):
    """Create a supported training scheduler."""
    if kind == "cosine":
        return get_cosine_schedule_with_warmup_and_cooldown(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            num_cooldown_steps=int(num_training_steps * 0.1),  # 10% cooldown
            min_lr_ratio=0.1,
        )
    if kind == "onecycle":
        base_lr = optimizer.param_groups[0]["lr"]
        return get_one_cycle_schedule(optimizer, base_lr, num_training_steps)
    if kind in {"linear", "constant"}:

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if kind == "constant":
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unsupported scheduler kind: {kind}")


def parameter_groups(model: nn.Module, weight_decay: float):
    """Separate parameters that use weight decay from biases and norms."""
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

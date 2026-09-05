import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

Mamba = None
Mamba2 = None
Mamba3 = None
_MAMBA_IMPORT_ERROR = None


def _load_mamba_class(mamba_version: str):
    """Import the requested Mamba class lazily.

    Importing all optional Mamba variants at module import time can hang on
    some local builds. In particular, Mamba3 should only be touched when the
    caller explicitly requests it.
    """
    global Mamba, Mamba2, Mamba3, _MAMBA_IMPORT_ERROR

    try:
        if mamba_version == "mamba1":
            if Mamba is None:
                from mamba_ssm import Mamba as _Mamba

                Mamba = _Mamba
            return Mamba
        if mamba_version == "mamba2":
            if Mamba2 is None:
                from mamba_ssm import Mamba2 as _Mamba2

                Mamba2 = _Mamba2
            return Mamba2
        if mamba_version == "mamba3":
            if Mamba3 is None:
                from mamba_ssm import Mamba3 as _Mamba3

                Mamba3 = _Mamba3
            return Mamba3
    except ImportError as exc:
        _MAMBA_IMPORT_ERROR = exc
        return None

    raise ValueError(f"Unsupported mamba_version={mamba_version!r}")


def can_use_mamba() -> bool:
    """Return whether CUDA-backed Mamba layers should be instantiated."""
    return torch.cuda.is_available()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # cos, sin: (L, D) -> need broadcasting to (B, H, L, D) or similar depending on x shape
    # x: (B, H, L, D) or (B, nH, L, hD)
    # sin, cos: (L, D)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, L, D)
        var = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed


class RMSNorm1d(nn.Module):
    """RMSNorm for (B, C, L) tensors (channel-first)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        # x: (B, C, L)
        var = torch.mean(x**2, dim=1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed


class DropPath(nn.Module):
    """Stochastic depth (DropPath) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class RoPECache(nn.Module):
    """
    Shared Rotary Positional Embedding Cache.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.cached_length = 0
        self.sin = None
        self.cos = None

    def forward(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if length <= self.cached_length and self.sin is not None and self.sin.device == device:
            return self.sin[:length], self.cos[:length]

        # Calculate RoPE
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        positions = torch.arange(length, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)

        sin = torch.zeros(length, self.head_dim, device=device)
        cos = torch.zeros(length, self.head_dim, device=device)

        sin[:, ::2] = torch.sin(freqs)
        cos[:, ::2] = torch.cos(freqs)
        sin[:, 1::2] = sin[:, ::2]
        cos[:, 1::2] = cos[:, ::2]

        self.cached_length = length
        self.sin = sin
        self.cos = cos

        return sin, cos


class SamePadConv1d(nn.Module):
    """Conv1d with asymmetric padding to support even kernel sizes while maintaining length."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_left, self.pad_right))
        return self.conv(x)


class ResBlock1d(nn.Module):
    """
    ResNet-style block for 1D sequence (B, C, L).
    Structure: Conv -> Norm -> SiLU -> Dropout -> Conv -> Norm -> SiLU -> Dropout
    """

    def __init__(self, channels, kernel_size, stride=1, dropout=0.0, drop_path=0.0):
        super().__init__()
        # Asymmetric padding for even kernel sizes to maintain length
        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=0, bias=False)
        self.norm1 = RMSNorm1d(channels)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=0, bias=False)
        self.norm2 = RMSNorm1d(channels)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Downsample path if stride > 1
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True),
                RMSNorm1d(channels),  # Optional: norm after pooling
            )

    def forward(self, x):
        residual = x

        # Asymmetric padding
        x_pad = F.pad(x, (self.pad_left, self.pad_right))
        out = self.conv1(x_pad)

        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = F.pad(out, (self.pad_left, self.pad_right))
        out = self.conv2(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + self.drop_path(out)


class BiMamba(nn.Module):
    """
    Bidirectional Mamba Wrapper.
    Runs Mamba forward and backward (on flipped sequence), concatenates, and projects back to d_model.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba2=False,
        headdim=64,
        ngroups=1,
        mamba_version=None,
    ):
        super().__init__()
        mamba_version = mamba_version or ("mamba2" if use_mamba2 else "mamba1")
        if not can_use_mamba():
            raise RuntimeError(
                "BiMamba requires CUDA-backed mamba_ssm/causal-conv1d kernels. "
                "No CUDA device is available, so the program cannot continue."
            )

        if mamba_version == "mamba3":
            mamba_cls = _load_mamba_class("mamba3")
            if mamba_cls is None:
                raise ImportError(
                    "Mamba3 requested but not found in mamba_ssm. Install the source build with "
                    "MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall "
                    "git+https://github.com/state-spaces/mamba.git --no-build-isolation"
                )
            self.fwd = mamba_cls(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                is_mimo=False,
                chunk_size=64,
                is_outproj_norm=False,
            )
            self.bwd = mamba_cls(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                is_mimo=False,
                chunk_size=64,
                is_outproj_norm=False,
            )
        elif mamba_version == "mamba2":
            mamba_cls = _load_mamba_class("mamba2")
            if mamba_cls is None:
                raise ImportError("Mamba2 requested but not found in mamba_ssm.") from _MAMBA_IMPORT_ERROR
            self.fwd = mamba_cls(
                d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, ngroups=ngroups
            )
            self.bwd = mamba_cls(
                d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, ngroups=ngroups
            )
        elif mamba_version == "mamba1":
            mamba_cls = _load_mamba_class("mamba1")
            if mamba_cls is None:
                raise ImportError("Mamba1 requested but not found in mamba_ssm.") from _MAMBA_IMPORT_ERROR
            self.fwd = mamba_cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.bwd = mamba_cls(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            raise ValueError(f"Unsupported mamba_version={mamba_version!r}")

        # Project 2*d_model -> d_model
        self.out_proj = nn.Linear(2 * d_model, d_model)

    @staticmethod
    def _reverse_valid_prefix(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Reverse each right-padded sample without moving padding before valid tokens."""
        if padding_mask.shape != x.shape[:2]:
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} does not match sequence shape {tuple(x.shape[:2])}"
            )
        padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
        valid_lengths = (~padding_mask).sum(dim=1, keepdim=True)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        reverse_positions = torch.where(positions < valid_lengths, valid_lengths - 1 - positions, positions)
        return x.gather(1, reverse_positions.unsqueeze(-1).expand_as(x))

    def forward(self, x, padding_mask: torch.Tensor | None = None):
        # x: (B, L, D)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x_fwd = self.fwd(x)

        # Backward pass: flip, run, flip back
        x_rev = x.flip([1]) if padding_mask is None else self._reverse_valid_prefix(x, padding_mask)
        x_bwd_raw = self.bwd(x_rev)  # (B, L, D) - in reversed coordinate space
        x_bwd = x_bwd_raw.flip([1]) if padding_mask is None else self._reverse_valid_prefix(x_bwd_raw, padding_mask)

        # Concat and project
        x_cat = torch.cat([x_fwd, x_bwd], dim=-1)
        output = self.out_proj(x_cat)
        return output if padding_mask is None else output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class MoEExpert(nn.Module):
    """Single SwiGLU expert with reduced width (H/2 instead of 4H)."""

    def __init__(self, embed_dim, expert_dim):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, expert_dim, bias=False)  # Gate
        self.w2 = nn.Linear(embed_dim, expert_dim, bias=False)  # Val
        self.w3 = nn.Linear(expert_dim, embed_dim, bias=False)  # Out

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MoEFFN(nn.Module):
    """Sparse top-k mixture-of-experts feed-forward network."""

    def __init__(
        self, embed_dim, num_experts=8, top_k=1, alpha_lb=0.01, beta_z=0.001, dropout=0.0, expert_drop_prob=0.1
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if not 0.0 <= expert_drop_prob <= 1.0:
            raise ValueError("expert_drop_prob must be between 0 and 1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha_lb = alpha_lb
        self.beta_z = beta_z
        self.expert_drop_prob = expert_drop_prob

        expert_dim = embed_dim // 2  # H/2 per expert

        # Router: Linear(H) -> num_experts logits
        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        # Avoid exact routing ties at initialization.
        self.router.is_router = True
        self.router.force_nonzero_init = True
        nn.init.xavier_uniform_(self.router.weight, gain=0.01)

        # Experts: each is a small SwiGLU block
        self.experts = nn.ModuleList([MoEExpert(embed_dim, expert_dim) for _ in range(num_experts)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
            aux_loss: scalar (load_balance_loss + z_loss)
            router_entropy: scalar (for logging)
        """
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (N, D) where N = B*L
        N = x_flat.shape[0]

        # 1. Router
        router_logits = self.router(x_flat)  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        top_val, top_idx = router_probs.topk(self.top_k, dim=-1)

        token_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(N, self.top_k).reshape(-1)
        expert_indices = top_idx.reshape(-1)
        routing_weights = top_val.reshape(-1)

        sorted_indices = torch.argsort(expert_indices, stable=True)
        sorted_token_indices = token_indices[sorted_indices]
        sorted_tokens = x_flat[sorted_token_indices]
        sorted_weights = routing_weights[sorted_indices]
        total_counts = torch.bincount(expert_indices, minlength=self.num_experts)

        output_flat = torch.zeros_like(sorted_tokens)
        offset = 0
        for expert_index in range(self.num_experts):
            count = total_counts[expert_index].item()
            if count == 0:
                continue
            if self.training and torch.rand(1, device=x.device).item() < self.expert_drop_prob:
                offset += count
                continue

            expert_input = sorted_tokens[offset : offset + count]
            output_flat[offset : offset + count] = self.experts[expert_index](expert_input)
            offset += count

        output_flat = output_flat * sorted_weights.unsqueeze(-1)
        combined = torch.zeros_like(x_flat)
        combined.index_add_(0, sorted_token_indices, output_flat)
        output = self.dropout(combined.view(B, L, D))

        # Switch Transformer load balancing: alpha * E * sum(f_e * P_e)
        #   f_e = fraction of tokens routed to expert e (routing distribution)
        #   P_e = mean router prob for expert e (probability distribution)
        # Routing counts keep this loss independent of stochastic expert dropout.
        f = total_counts.float() / (N * self.top_k)
        P = router_probs.mean(dim=0)  # (E,) mean router probability per expert
        load_balance_loss = self.alpha_lb * self.num_experts * (f * P).sum()

        # Z-loss: beta * mean(logsumexp(router_logits) ** 2)
        z_loss = self.beta_z * torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

        aux_loss = load_balance_loss + z_loss

        # Router entropy is logged as a routing diagnostic, not optimized as a loss.
        router_entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean()

        return output, aux_loss, router_entropy


class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout,
        activation="gelu",
        use_conv_ffn=False,
        conv_kernel_size=7,
        drop_path=0.0,
        num_kv_heads=None,
        use_moe=False,
        moe_num_experts=8,
        moe_top_k=1,
        moe_beta_z=0.001,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

        # Attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * self.num_kv_heads, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm1 = RMSNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # FFN: MoE or Dense SwiGLU
        self.use_moe = use_moe
        self.norm2 = RMSNorm(embed_dim)

        if use_moe:
            self.moe_ffn = MoEFFN(
                embed_dim, num_experts=moe_num_experts, top_k=moe_top_k, beta_z=moe_beta_z, dropout=dropout
            )
        else:
            # SwiGLU: 3 linear layers (gate, val, out)
            self.w1 = nn.Linear(embed_dim, dim_feedforward, bias=False)
            self.w2 = nn.Linear(embed_dim, dim_feedforward, bias=False)
            self.w3 = nn.Linear(dim_feedforward, embed_dim, bias=False)

            self.dropout2 = nn.Dropout(dropout)

            # ConvFFN (Optional — dense path only)
            self.use_conv_ffn = use_conv_ffn
            if use_conv_ffn:
                self.conv_ffn = nn.Conv1d(
                    dim_feedforward,
                    dim_feedforward,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    groups=dim_feedforward,
                    bias=False,
                )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _reshape_to_heads(self, x, n_heads):
        B, L, _ = x.shape
        x = x.view(B, L, n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def _apply_rope(self, x, sin, cos):
        return (x * cos.unsqueeze(0).unsqueeze(0)) + (self._rotate_half(x) * sin.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        src,
        src_key_padding_mask=None,
        sin=None,
        cos=None,
        *,
        capture_attention: bool = False,
        attention_heads: set[int] | None = None,
    ):
        # 1. Attention
        residual = src
        x = self.norm1(src)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._reshape_to_heads(q, self.num_heads)
        k = self._reshape_to_heads(k, self.num_kv_heads)
        v = self._reshape_to_heads(v, self.num_kv_heads)

        if sin is not None and cos is not None:
            q = self._apply_rope(q, sin, cos)
            k = self._apply_rope(k, sin, cos)

        # GQA: Repeat K/V heads if needed
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # SDPA
        attn_mask = None
        if src_key_padding_mask is not None:
            # The rest of the code uses True = ignore. PyTorch SDPA boolean
            # masks use True = participate, so invert here at the boundary.
            attn_mask = (~src_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)

        captured_attention = None
        if capture_attention:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, 0.0)

            if src_key_padding_mask is not None:
                query_keep = (~src_key_padding_mask.bool()).unsqueeze(1).unsqueeze(-1)
                attn = attn.masked_fill(~query_keep, 0.0)

            attn_for_output = attn
            if self.training and self.dropout1.p > 0.0:
                attn_for_output = F.dropout(attn_for_output, p=self.dropout1.p)
            x = torch.matmul(attn_for_output.to(v.dtype), v)

            if attention_heads is None:
                captured_attention = attn.detach().to(dtype=torch.float32, device="cpu")
            else:
                heads = sorted(head for head in attention_heads if 0 <= head < self.num_heads)
                captured_attention = attn[:, heads].detach().to(dtype=torch.float32, device="cpu")
        else:
            try:
                x = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=self.dropout1.p if self.training else 0.0, is_causal=False
                )
            except RuntimeError as e:
                LOGGER.error(
                    "[SDPA Error] q=%s, k=%s, mask=%s",
                    q.shape,
                    k.shape,
                    attn_mask.shape if attn_mask is not None else None,
                )
                raise e

        x = x.permute(0, 2, 1, 3).contiguous().flatten(2)
        x = self.out_proj(x)
        x = self.dropout1(x)

        src = residual + self.drop_path(x)

        # 2. FFN: MoE or Dense SwiGLU
        residual = src
        x = self.norm2(src)

        moe_aux_loss = None
        router_entropy = None

        if self.use_moe:
            x, moe_aux_loss, router_entropy = self.moe_ffn(x)
        else:
            gate = self.w1(x)
            val = self.w2(x)

            if hasattr(self, "use_conv_ffn") and self.use_conv_ffn:
                gate = gate.transpose(1, 2)
                gate = self.conv_ffn(gate)
                gate = gate.transpose(1, 2)

            x = torch.nn.functional.silu(gate) * val
            x = self.w3(x)
            x = self.dropout2(x)

        src = residual + self.drop_path(x)
        if capture_attention:
            return src, moe_aux_loss, router_entropy, captured_attention
        return src, moe_aux_loss, router_entropy

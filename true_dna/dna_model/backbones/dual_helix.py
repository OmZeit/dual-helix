from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DnaConfig
from .common import (
    BiMamba,
    DropPath,
    RMSNorm,
    RMSNorm1d,
    RoPECache,
    TransformerEncoderLayerRoPE,
    apply_rotary_pos_emb,
)


class PooledBridge(nn.Module):
    """
    Directional pooled bridge between Stream A (full-res) and Stream B (compressed).

    A to B: pool Stream A down to B's length, then add.
    B to A: upsample Stream B to A's length, then add through a gate.

    Complexity: O(N) — no quadratic attention.
    """

    def __init__(self, embed_dim, pool_factor=4, dropout=0.0):
        super().__init__()
        self.pool_factor = pool_factor

        # A-to-B path: pool and project
        self.norm_a = RMSNorm(embed_dim)
        self.proj_a2b = nn.Linear(embed_dim, embed_dim, bias=False)

        # B-to-A path: upsample and project through a gate
        self.norm_b = RMSNorm(embed_dim)
        self.proj_b2a = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim, bias=False), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)

    def a_to_b(self, x_a, x_b):
        """
        Pool Stream A down to B's length, project, and add to B.
        x_a: (B, L, D), full resolution
        x_b: (B, L/S, D) — compressed
        """
        x_a_norm = self.norm_a(x_a)

        # Adaptive avg pool to match B's length
        # Transpose to channel-first, pool, then restore sequence-first layout.
        L_b = x_b.shape[1]
        x_a_pooled = F.adaptive_avg_pool1d(x_a_norm.transpose(1, 2), output_size=L_b).transpose(1, 2)

        out = self.proj_a2b(x_a_pooled)
        out = self.dropout(out)
        return x_b + out

    def b_to_a(self, x_b, x_a):
        """
        Upsample Stream B to A's length and add it to A through a gate.
        x_b: (B, L/S, D), compressed
        x_a: (B, L, D)   — full resolution
        """
        x_b_norm = self.norm_b(x_b)
        L_a = x_a.shape[1]

        # Upsample B to match A's length
        x_b_up = F.interpolate(x_b_norm.transpose(1, 2), size=L_a, mode="nearest").transpose(1, 2)  # (B, L, D)

        x_b_proj = self.proj_b2a(x_b_up)

        # Learned gate: how much of B's context to inject into A
        gate_input = torch.cat([x_a, x_b_proj], dim=-1)  # (B, L, 2D)
        g = self.gate(gate_input)  # (B, L, D) sigmoid

        out = self.dropout(g * x_b_proj)
        return x_a + out


class WindowAttention(nn.Module):
    """Swin-style local attention with optional shifted windows."""

    def __init__(self, config: DnaConfig, shift_size: int = 0):
        super().__init__()
        self.H = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.H // self.num_heads
        self.window_size = config.microscope_window_size
        self.shift_size = shift_size

        self.qkv = nn.Linear(self.H, 3 * self.H, bias=False)
        self.o_proj = nn.Linear(self.H, self.H, bias=False)

        # Internal RoPE
        self.rope_cache = RoPECache(self.head_dim)

    def forward(self, x):
        # x: (B, L, H)
        B, L, H = x.shape
        W = self.window_size

        if self.shift_size > 0:
            shifted_x = F.pad(x[:, self.shift_size :, :], (0, 0, 0, self.shift_size))
        else:
            shifted_x = x

        # Pad to a whole number of windows.
        pad_r = (W - L % W) % W
        if pad_r > 0:
            shifted_x = F.pad(shifted_x, (0, 0, 0, pad_r))
            L_padded = L + pad_r
        else:
            L_padded = L

        # (B, L_pad, H) -> (B, L_pad/W, W, H)
        num_windows = L_padded // W
        x_windows = shifted_x.view(B, num_windows, W, H)

        # 3. Attention
        # Flatten to batch: (B*num_windows, W, H)
        x_windows = x_windows.view(-1, W, H)

        # QKV
        qkv = self.qkv(x_windows)  # (B*nW, W, 3H)
        qkv = qkv.reshape(-1, W, 3, self.num_heads, self.head_dim)

        q = qkv[:, :, 0].transpose(1, 2)  # (batch, head, W, D)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        # RoPE uses positions relative to each window; Stream B carries global context.
        sin, cos = self.rope_cache(W, x.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Scaled Dot Product Attention
        # (Batch, Head, W, W) -> Small dense attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # 4. Merge Windows
        out = out.transpose(1, 2).contiguous().view(-1, W, H)  # (B*nW, W, H)
        out = out.view(B, num_windows, W, H)
        out = out.view(B, L_padded, H)

        # Remove Padding
        if pad_r > 0:
            out = out[:, :L, :]

        # 5. Reverse Shift
        if self.shift_size > 0:
            out = torch.cat([x[:, : self.shift_size, :], out[:, : -self.shift_size, :]], dim=1)

        # 6. Projection
        out = self.o_proj(out)

        return out


class MicroscopeBlock(nn.Module):
    """
    Stream A (Microscope): Lightweight, full-length (L).
    Uses CUDA-backed BiMamba.
    """

    def __init__(self, config: DnaConfig, layer_idx: int = 0):
        super().__init__()
        H = config.hidden_size

        self.mixer = BiMamba(
            d_model=H,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            use_mamba2=config.use_mamba2,
            headdim=config.mamba2_head_dim,
            ngroups=config.mamba2_ngroups,
            mamba_version=config.mamba_version,
        )

        self.norm = RMSNorm(H)
        self.drop_path = DropPath(config.drop_path_rate)

    def forward(self, x, padding_mask: torch.Tensor | None = None):
        # x: (B, L, H)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        residual = x
        x_norm = self.norm(x)

        out = self.mixer(x_norm, padding_mask=padding_mask)

        output = residual + self.drop_path(out)
        return output if padding_mask is None else output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class TelescopeBlock(nn.Module):
    """
    Stream B (Telescope): Heavy Transformer, compressed length (L/4).
    Optionally uses MoE FFN instead of dense SwiGLU.
    """

    def __init__(self, config: DnaConfig):
        super().__init__()
        use_moe = getattr(config, "use_moe", False)
        moe_num_experts = getattr(config, "moe_num_experts", 8)
        moe_top_k = getattr(config, "moe_top_k", 1)

        self.layer = TransformerEncoderLayerRoPE(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dim_feedforward=4 * config.hidden_size,
            dropout=config.dropout,
            use_conv_ffn=config.use_conv_ffn,
            conv_kernel_size=config.conv_kernel_size,
            drop_path=config.drop_path_rate,
            num_kv_heads=config.num_key_value_heads,
            use_moe=use_moe,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_beta_z=getattr(config, "moe_beta_z", 0.001),
        )

    def forward(
        self,
        x,
        src_key_padding_mask=None,
        sin=None,
        cos=None,
        *,
        capture_attention: bool = False,
        attention_heads: set[int] | None = None,
    ):
        # Returns (src, moe_aux_loss, router_entropy)
        return self.layer(
            x,
            src_key_padding_mask=src_key_padding_mask,
            sin=sin,
            cos=cos,
            capture_attention=capture_attention,
            attention_heads=attention_heads,
        )


class PositionAwareCompress(nn.Module):
    """
    Position-aware compression for Telescope stream.

    Adds a learnable positional bias before strided convolution so compression
    can retain useful positional features.

    Args:
        hidden_size: Model hidden dimension (H)
        stride: Compression stride (default 2 for L -> L/2)
    """

    def __init__(self, hidden_size: int, stride: int = 2):
        super().__init__()
        self.stride = stride

        # Strided convolution for compression
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=stride, stride=stride, bias=False)

        # Learnable positional bias per channel
        self.pos_bias = nn.Parameter(torch.zeros(1, hidden_size, 1))

        # Post-conv normalization
        self.norm = RMSNorm1d(hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, H, L) in channel-first format
        Returns:
            Compressed tensor (B, H, L/stride)
        """
        # Create position-aware modulation
        # linspace from -1 to 1 provides smooth positional gradient
        L = x.size(-1)
        pos_scale = torch.linspace(-1, 1, L, device=x.device, dtype=x.dtype)

        # Add position-aware bias before compression
        x = x + self.pos_bias * pos_scale

        # Apply strided convolution
        x = self.conv(x)

        # Normalize and activate
        x = self.norm(x)
        x = F.silu(x)

        return x


class CrossStrandGate(nn.Module):
    """Fuse forward and reverse-complement hidden states with O(L) gates."""

    def __init__(self, hidden_size: int, dropout: float = 0.0, equivariant: bool = False):
        super().__init__()
        self.equivariant = bool(equivariant)
        if self.equivariant:
            # Exact strand-swap equivariance through parameter tying:
            # swapping (x_fwd, x_rc) swaps outputs exactly.
            self.norm_shared = RMSNorm(hidden_size)
            self.cross_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.gate_shared = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size, bias=False),
                nn.Sigmoid(),
            )
        else:
            # Forward strand: receive RC context
            self.norm_fwd = RMSNorm(hidden_size)
            self.rc_to_fwd = nn.Linear(hidden_size, hidden_size, bias=False)
            self.gate_fwd = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size, bias=False),
                nn.Sigmoid(),
            )
            # RC strand: receive forward context
            self.norm_rc = RMSNorm(hidden_size)
            self.fwd_to_rc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.gate_rc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size, bias=False),
                nn.Sigmoid(),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_fwd: torch.Tensor,
        x_rc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_fwd: (B, L_fwd, H) – forward strand hidden states
            x_rc:  (B, L_rc,  H) – RC strand hidden states
        Returns:
            Fused (x_fwd, x_rc) with cross-strand information injected.
        """
        L_fwd = x_fwd.size(1)
        L_rc = x_rc.size(1)

        if self.equivariant:
            if L_fwd != L_rc:
                raise ValueError(
                    "RC-equivariant CrossStrandGate requires equal forward/RC token lengths "
                    f"(got fwd={L_fwd}, rc={L_rc}). Either disable rc_equivariant_tying or "
                    "ensure tokenization/padding yields equal lengths."
                )
            xf_n = self.norm_shared(x_fwd)
            xr_n = self.norm_shared(x_rc)

            # Shared projection + shared gate on both directions (exact strand-swap equivariance).
            rc_proj = self.cross_proj(xr_n)
            fwd_proj = self.cross_proj(xf_n)
            g_fwd = self.gate_shared(torch.cat([xf_n, rc_proj], dim=-1))
            g_rc = self.gate_shared(torch.cat([xr_n, fwd_proj], dim=-1))

            x_fwd = x_fwd + self.dropout(g_fwd * rc_proj)
            x_rc = x_rc + self.dropout(g_rc * fwd_proj)
        else:
            xf_n = self.norm_fwd(x_fwd)
            xr_n = self.norm_rc(x_rc)

            # Align RC to forward length if BPE tokenisation produced different lengths
            if L_rc != L_fwd:
                xr_aligned = F.interpolate(
                    xr_n.transpose(1, 2), size=L_fwd, mode="linear", align_corners=False
                ).transpose(1, 2)
            else:
                xr_aligned = xr_n

            # Forward strand fuses RC context
            rc_proj = self.rc_to_fwd(xr_aligned)
            g_fwd = self.gate_fwd(torch.cat([xf_n, rc_proj], dim=-1))
            x_fwd = x_fwd + self.dropout(g_fwd * rc_proj)

            # RC strand fuses forward context
            if L_fwd != L_rc:
                xf_for_rc = F.interpolate(
                    xf_n.transpose(1, 2), size=L_rc, mode="linear", align_corners=False
                ).transpose(1, 2)
            else:
                xf_for_rc = xf_n
            fwd_proj = self.fwd_to_rc(xf_for_rc)
            g_rc = self.gate_rc(torch.cat([xr_n, fwd_proj], dim=-1))
            x_rc = x_rc + self.dropout(g_rc * fwd_proj)

        return x_fwd, x_rc


class DualHelixBackbone(nn.Module):
    """
    Dual-stream backbone with periodic bridging.

    Stream A: Full resolution (L), light BiMamba layers.
    Stream B: Compressed resolution (L/4), heavy Transformer layers. Attention
              is applied only to this shorter sequence.

    When ``config.use_dual_strand=True`` the backbone processes the forward
    strand and its reverse complement (RC) in parallel, fusing them via
    CrossStrandGates every ``config.cross_strand_gate_interval`` layers.
    """

    def __init__(self, config: DnaConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size
        self.bridge_interval = config.dual_helix_bridge_interval
        self.use_dual_strand = getattr(config, "use_dual_strand", False)
        self.cross_strand_gate_interval = getattr(config, "cross_strand_gate_interval", 4)
        self.rc_equivariant_tying = getattr(config, "rc_equivariant_tying", False)

        # Compression for Stream B (L/4) with positional precision preservation
        # Fourfold compression reduces 8k tokens to 2k before attention.
        self.compress_b = PositionAwareCompress(H, stride=4)
        # RC strand compression (only when use_dual_strand=True)
        if self.use_dual_strand:
            self.compress_b_rc = PositionAwareCompress(H, stride=4)
            if self.rc_equivariant_tying:
                # Keep distinct module instances while tying all learnable parameters.
                self.compress_b_rc.conv.weight = self.compress_b.conv.weight
                self.compress_b_rc.pos_bias = self.compress_b.pos_bias
                self.compress_b_rc.norm.weight = self.compress_b.norm.weight

        # Layers
        self.layers = nn.ModuleList()
        # RoPE cache for Stream B
        self.head_dim = H // config.num_attention_heads
        self.rope_cache = RoPECache(self.head_dim)

        for i in range(config.high_level_layers):
            micro = MicroscopeBlock(config, layer_idx=i)
            tele = TelescopeBlock(config)

            bridge = None
            if (not config.disable_bridge) and ((i + 1) % self.bridge_interval == 0):
                bridge = PooledBridge(H, pool_factor=4, dropout=config.dropout)

            # Cross-strand gate (dual-strand mode only)
            cross_gate = None
            if self.use_dual_strand and ((i + 1) % self.cross_strand_gate_interval == 0):
                cross_gate = CrossStrandGate(H, dropout=config.dropout, equivariant=self.rc_equivariant_tying)

            self.layers.append(
                nn.ModuleDict(
                    {
                        "micro": micro,
                        "tele": tele,
                        "bridge": bridge,
                        "cross_gate": cross_gate,
                    }
                )
            )

        self.final_norm = RMSNorm(H)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        x_rc: torch.Tensor | None = None,
        mask_rc: torch.Tensor | None = None,
        capture_attention: bool = False,
        attention_layers: set[int] | None = None,
        attention_heads: set[int] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[dict[str, Any]]]
    ):
        """
        Args:
            x:                   (B, L, H) forward strand embeddings
            src_key_padding_mask: (B, L) bool mask (True = ignore), optional
            x_rc:                (B, L_rc, H) RC strand embeddings, optional
            mask_rc:             (B, L_rc) bool mask for RC strand, optional
            capture_attention:    When true, manually captures Telescope
                                  attention weights for selected layers.
            attention_layers:     Optional zero-based layer indices to capture.
            attention_heads:      Optional zero-based head indices to capture.
        Returns:
            (output, moe_aux_loss, router_entropy) plus attention metadata when
            ``capture_attention`` is true.
        """
        # 1. Initialize Streams
        x_a = x  # Stream A (Microscope) full resolution

        # Stream B (Telescope) – Compressed
        x_b = self.compress_b(x.transpose(1, 2)).transpose(1, 2)

        # RC strand streams (dual-strand mode)
        x_a_rc = x_rc
        x_b_rc = None
        if self.use_dual_strand and x_rc is not None:
            x_b_rc = self.compress_b_rc(x_rc.transpose(1, 2)).transpose(1, 2)

        # Downsample Stream B mask
        mask_b = None
        if src_key_padding_mask is not None:
            L_b = x_b.size(1)
            mask_b = (
                F.interpolate(src_key_padding_mask.unsqueeze(1).float(), size=L_b, mode="nearest").squeeze(1).bool()
            )

        # RC Stream B mask
        mask_b_rc = None
        if self.use_dual_strand and mask_rc is not None and x_b_rc is not None:
            L_b_rc = x_b_rc.size(1)
            mask_b_rc = F.interpolate(mask_rc.unsqueeze(1).float(), size=L_b_rc, mode="nearest").squeeze(1).bool()

        # RoPE for Stream B
        sin, cos = self.rope_cache(x_b.shape[1], x_b.device)
        sin_rc, cos_rc = None, None
        if x_b_rc is not None:
            sin_rc, cos_rc = self.rope_cache(x_b_rc.shape[1], x_b_rc.device)

        # 2. Iterate
        moe_aux_losses: list[torch.Tensor] = []
        router_entropies: list[torch.Tensor] = []
        attention_records: list[dict[str, Any]] = []

        for layer_idx, layer_dict in enumerate(self.layers):
            micro = layer_dict["micro"]
            tele = layer_dict["tele"]
            bridge = layer_dict["bridge"]
            cross_gate = layer_dict["cross_gate"]
            capture_layer = capture_attention and (attention_layers is None or layer_idx in attention_layers)

            if self.config.gradient_checkpointing and self.training:

                def create_block_forward(m_micro, m_tele, m_bridge):
                    def block_forward(xa, xb, sin, cos, mask_a, mask_b):
                        xa = m_micro(xa, padding_mask=mask_a)
                        tele_result = m_tele(xb, src_key_padding_mask=mask_b, sin=sin, cos=cos)
                        aux, ent = None, None
                        if isinstance(tele_result, tuple):
                            xb = tele_result[0]
                            aux = tele_result[1]
                            ent = tele_result[2]
                        else:
                            xb = tele_result
                        if m_bridge is not None:
                            xb = m_bridge.a_to_b(xa, xb)
                            xa = m_bridge.b_to_a(xb, xa)
                            if mask_a is not None:
                                xa = xa.masked_fill(mask_a.unsqueeze(-1), 0.0)
                        return xa, xb, aux, ent

                    return block_forward

                result = torch.utils.checkpoint.checkpoint(
                    create_block_forward(micro, tele, bridge),
                    x_a,
                    x_b,
                    sin,
                    cos,
                    src_key_padding_mask,
                    mask_b,
                    use_reentrant=False,
                )
                x_a, x_b, aux, ent = result
                if aux is not None:
                    moe_aux_losses.append(aux)
                if ent is not None:
                    router_entropies.append(ent)
            else:
                x_a = micro(x_a, padding_mask=src_key_padding_mask)
                tele_result = tele(
                    x_b,
                    src_key_padding_mask=mask_b,
                    sin=sin,
                    cos=cos,
                    capture_attention=capture_layer,
                    attention_heads=attention_heads,
                )
                if isinstance(tele_result, tuple):
                    x_b = tele_result[0]
                    if tele_result[1] is not None:
                        moe_aux_losses.append(tele_result[1])
                    if tele_result[2] is not None:
                        router_entropies.append(tele_result[2])
                    if capture_layer and len(tele_result) > 3 and tele_result[3] is not None:
                        attention_records.append(
                            {
                                "layer": layer_idx,
                                "weights": tele_result[3],
                            }
                        )
                else:
                    x_b = tele_result
                if bridge is not None:
                    x_b = bridge.a_to_b(x_a, x_b)
                    x_a = bridge.b_to_a(x_b, x_a)
                    if src_key_padding_mask is not None:
                        x_a = x_a.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

            # RC strand processing (dual-strand mode)
            if self.use_dual_strand and x_a_rc is not None and x_b_rc is not None:
                x_a_rc = micro(x_a_rc, padding_mask=mask_rc)
                tele_rc_result = tele(x_b_rc, src_key_padding_mask=mask_b_rc, sin=sin_rc, cos=cos_rc)
                if isinstance(tele_rc_result, tuple):
                    x_b_rc = tele_rc_result[0]
                else:
                    x_b_rc = tele_rc_result
                if bridge is not None:
                    x_b_rc = bridge.a_to_b(x_a_rc, x_b_rc)
                    x_a_rc = bridge.b_to_a(x_b_rc, x_a_rc)
                    if mask_rc is not None:
                        x_a_rc = x_a_rc.masked_fill(mask_rc.unsqueeze(-1), 0.0)

            # Cross-strand gate fusion
            if cross_gate is not None and x_a_rc is not None:
                x_a, x_a_rc = cross_gate(x_a, x_a_rc)
                if src_key_padding_mask is not None:
                    x_a = x_a.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
                if mask_rc is not None:
                    x_a_rc = x_a_rc.masked_fill(mask_rc.unsqueeze(-1), 0.0)

        # 3. Finalize
        out = self.final_norm(x_a)
        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        moe_aux_loss = torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        mean_router_entropy = torch.stack(router_entropies).mean() if router_entropies else None

        if capture_attention:
            return out, moe_aux_loss, mean_router_entropy, attention_records
        return out, moe_aux_loss, mean_router_entropy

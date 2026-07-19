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
    SamePadConv1d,
    TransformerEncoderLayerRoPE,
)


class FSHMBlock(nn.Module):
    """Route features between full-resolution Mamba and compressed attention.

    Transformer attention is limited to the strided slow stream. A two-expert
    router combines it with the full-resolution bidirectional Mamba stream,
    with load-balancing and router z-loss regularization.
    """

    def __init__(self, config: DnaConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size
        stride = config.fshm_slow_stride

        self.norm = RMSNorm(H)

        # Expert 0 is the fast Mamba stream; expert 1 is bottleneck attention.
        self.bio_router = nn.Linear(H, 2, bias=False)
        self.bio_router.is_router = True
        nn.init.zeros_(self.bio_router.weight)
        self._router_alpha_lb = 0.01
        self._router_beta_z = float(getattr(config, "moe_beta_z", 0.001))

        # Fast stream (expert 0): full-resolution BiMamba
        self.fast_bimamba = BiMamba(
            d_model=H,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            use_mamba2=config.use_mamba2,
            headdim=config.mamba2_head_dim,
            ngroups=config.mamba2_ngroups,
            mamba_version=config.mamba_version,
        )

        # Slow stream: compressed bottleneck attention.
        # Compression: strided convolution reduces L to L/stride
        self.slow_compress = nn.Conv1d(H, H, kernel_size=stride, stride=stride)
        # Transformer attention applied exclusively on the compressed stream
        head_dim = H // config.num_attention_heads
        self.slow_attn = TransformerEncoderLayerRoPE(
            embed_dim=H,
            num_heads=config.num_attention_heads,
            dim_feedforward=2 * H,  # lean FFN inside the bottleneck
            dropout=config.dropout,
            drop_path=0.0,  # drop_path is managed at block level
            num_kv_heads=config.num_key_value_heads,
            use_moe=False,  # keep the bottleneck lean
        )
        self._slow_rope = RoPECache(head_dim)
        # Upsampling: artifact-free linear upsample + depth-wise conv
        self.slow_expand = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="linear", align_corners=False),
            SamePadConv1d(H, H, kernel_size=3, bias=False),
            RMSNorm1d(H),
            nn.SiLU(),
        )
        self.slow_norm = RMSNorm(H)

        # Fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, 1),
            nn.Sigmoid(),
        )

        self.out_linear = nn.Linear(H, H)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (B, L, H)
            router_aux_loss: scalar – load-balance + z-loss for the bio router
        """
        residual = x
        x = self.norm(x)
        B, L, H = x.shape

        # Biological-distance router
        router_logits = self.bio_router(x)  # (B, L, 2)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, L, 2)
        alpha_fast = router_probs[..., 0:1]  # (B, L, 1)
        alpha_slow = router_probs[..., 1:2]  # (B, L, 1)

        # Router auxiliary losses (load-balance + z-loss)
        # Use soft expected routing fraction to avoid per-step bincount overhead.
        rp_flat = router_probs.reshape(-1, 2)  # (N, 2)
        f = rp_flat.mean(0)
        P = rp_flat.mean(0)  # mean prob per expert
        lb_loss = self._router_alpha_lb * 2.0 * (f * P).sum()
        z_loss = self._router_beta_z * torch.mean(torch.logsumexp(router_logits.reshape(-1, 2), dim=-1) ** 2)
        router_aux_loss = lb_loss + z_loss

        # Fast stream: BiMamba at full resolution
        x_fast_out = self.fast_bimamba(x * alpha_fast)

        # Slow stream: bottleneck attention
        x_slow_t = (x * alpha_slow).transpose(1, 2)  # (B, H, L)
        x_slow_comp = self.slow_compress(x_slow_t)  # (B, H, L/stride)
        x_slow_comp = x_slow_comp.transpose(1, 2)  # (B, L/stride, H)
        L_s = x_slow_comp.size(1)
        sin, cos = self._slow_rope(L_s, x.device)
        slow_result = self.slow_attn(x_slow_comp, sin=sin, cos=cos)
        x_slow_processed = slow_result[0] if isinstance(slow_result, tuple) else slow_result
        x_slow_up = self.slow_expand(x_slow_processed.transpose(1, 2)).transpose(1, 2)
        if x_slow_up.size(1) != L:
            x_slow_up = F.interpolate(x_slow_up.transpose(1, 2), size=L, mode="linear", align_corners=False).transpose(
                1, 2
            )
        x_slow_up = self.slow_norm(x_slow_up)

        # Fusion
        fusion_gate = self.fusion_mlp(torch.cat([x_fast_out, x_slow_up], dim=-1))
        combined = x_fast_out + fusion_gate * x_slow_up

        out = self.out_linear(combined)
        out = self.dropout(out)
        return residual + self.drop_path(out), router_aux_loss


class BiMambaBlock(nn.Module):
    """Pre-normalized residual bidirectional Mamba block with a SwiGLU FFN."""

    def __init__(self, config: DnaConfig, drop_path_rate: float = 0.0):
        super().__init__()
        H = config.hidden_size
        ffn_dim = 4 * H

        self.norm_mixer = RMSNorm(H)
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

        # SwiGLU FFN
        self.norm_ffn = RMSNorm(H)
        self.w1 = nn.Linear(H, ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(H, ffn_dim, bias=False)  # value
        self.w3 = nn.Linear(ffn_dim, H, bias=False)  # output

        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None, None]:
        # kwargs accepted for call-site compatibility with TransformerEncoderLayerRoPE
        residual = x
        x_mix = self.mixer(self.norm_mixer(x))
        x = residual + self.drop_path(x_mix)

        residual = x
        x_n = self.norm_ffn(x)
        x_ffn = self.dropout(self.w3(F.silu(self.w1(x_n)) * self.w2(x_n)))
        x = residual + self.drop_path(x_ffn)

        # Return (output, moe_aux_loss=None, router_entropy=None) for API compatibility
        return x, None, None


class FSHMEncoder(nn.Module):
    """Interleave FSH-M bottlenecks with full-resolution BiMamba blocks."""

    def __init__(self, config: DnaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.high_level_layers)]

        for i in range(config.high_level_layers):
            if i in config.fshm_layers:
                # FSH-M block with a biological-distance router
                block = FSHMBlock(config)
                if hasattr(block, "drop_path") and isinstance(block.drop_path, DropPath):
                    block.drop_path.drop_prob = dpr[i]
                self.layers.append(block)
            else:
                # Full-resolution layers use linear-complexity BiMamba.
                self.layers.append(BiMambaBlock(config, drop_path_rate=dpr[i]))

        self.final_norm = RMSNorm(config.hidden_size)

    def forward(self, src, src_key_padding_mask=None):
        moe_aux_losses = []
        router_entropies = []

        for layer in self.layers:
            if isinstance(layer, FSHMBlock):
                if self.config.gradient_checkpointing and self.training:
                    result = torch.utils.checkpoint.checkpoint(layer, src, use_reentrant=False)
                else:
                    result = layer(src)
                # FSHMBlock returns (output, router_aux_loss)
                src = result[0] if isinstance(result, tuple) else result
                if isinstance(result, tuple) and result[1] is not None:
                    moe_aux_losses.append(result[1])
            else:
                # BiMambaBlock returns (output, None, None)
                if self.config.gradient_checkpointing and self.training:
                    result = torch.utils.checkpoint.checkpoint(layer, src, use_reentrant=False)
                else:
                    result = layer(src)
                if isinstance(result, tuple):
                    src = result[0]
                    if result[1] is not None:
                        moe_aux_losses.append(result[1])
                    if len(result) > 2 and result[2] is not None:
                        router_entropies.append(result[2])
                else:
                    src = result

        src = self.final_norm(src)

        moe_aux_loss = torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        mean_router_entropy = torch.stack(router_entropies).mean() if router_entropies else None

        return src, moe_aux_loss, mean_router_entropy

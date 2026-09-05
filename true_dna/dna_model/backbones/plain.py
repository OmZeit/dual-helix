"""Single-stream reference backbones used by controlled architecture ablations.

These modules deliberately keep the token sequence at full resolution.  They
are not aliases for the historical ``legacy`` backbone, which includes a
strided FSHM bottleneck and U-Net style fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DnaConfig
from .common import BiMamba, DropPath, RMSNorm, RoPECache, TransformerEncoderLayerRoPE


class _BiMambaOnlyBlock(nn.Module):
    """A full-resolution BiMamba mixer followed by a dense SwiGLU FFN."""

    def __init__(self, config: DnaConfig, drop_path_rate: float) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        ffn_size = 4 * hidden_size
        self.norm_mixer = RMSNorm(hidden_size)
        self.mixer = BiMamba(
            d_model=hidden_size,
            d_state=config.ssm_d_state,
            d_conv=config.ssm_d_conv,
            expand=config.ssm_expand,
            use_mamba2=config.use_mamba2,
            headdim=config.mamba2_head_dim,
            ngroups=config.mamba2_ngroups,
            mamba_version=config.mamba_version,
        )
        self.norm_ffn = RMSNorm(hidden_size)
        self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w2 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w3 = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        x = x + self.drop_path(self.mixer(self.norm_mixer(x), padding_mask=padding_mask))
        ffn_input = self.norm_ffn(x)
        ffn_output = self.w3(F.silu(self.w1(ffn_input)) * self.w2(ffn_input))
        output = x + self.drop_path(self.dropout(ffn_output))
        return output if padding_mask is None else output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class BiMambaBackbone(nn.Module):
    """Pure full-resolution bidirectional-Mamba reference backbone."""

    def __init__(self, config: DnaConfig) -> None:
        super().__init__()
        rates = torch.linspace(0, config.drop_path_rate, config.high_level_layers).tolist()
        self.layers = nn.ModuleList([_BiMambaOnlyBlock(config, float(rate)) for rate in rates])
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.final_norm = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None):
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:

                def layer_forward(hidden_states: torch.Tensor, block=layer) -> torch.Tensor:
                    return block(hidden_states, padding_mask=src_key_padding_mask)

                x = torch.utils.checkpoint.checkpoint(layer_forward, x, use_reentrant=False)
            else:
                x = layer(x, padding_mask=src_key_padding_mask)
        output = self.final_norm(x)
        if src_key_padding_mask is not None:
            output = output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return output, None, None


class TransformerBackbone(nn.Module):
    """Pure full-resolution RoPE Transformer reference backbone."""

    def __init__(self, config: DnaConfig) -> None:
        super().__init__()
        rates = torch.linspace(0, config.drop_path_rate, config.high_level_layers).tolist()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerRoPE(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dim_feedforward=4 * config.hidden_size,
                    dropout=config.dropout,
                    use_conv_ffn=config.use_conv_ffn,
                    conv_kernel_size=config.conv_kernel_size,
                    drop_path=float(rate),
                    num_kv_heads=config.num_key_value_heads,
                    use_moe=config.use_moe,
                    moe_num_experts=config.moe_num_experts,
                    moe_top_k=config.moe_top_k,
                    moe_beta_z=config.moe_beta_z,
                )
                for rate in rates
            ]
        )
        self.rope_cache = RoPECache(config.hidden_size // config.num_attention_heads)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.final_norm = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None):
        sin, cos = self.rope_cache(x.size(1), x.device)
        moe_losses: list[torch.Tensor] = []
        router_entropies: list[torch.Tensor] = []

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not layer.use_moe:

                def layer_forward(hidden_states: torch.Tensor, block=layer) -> torch.Tensor:
                    return block(
                        hidden_states,
                        src_key_padding_mask=src_key_padding_mask,
                        sin=sin,
                        cos=cos,
                    )[0]

                x = torch.utils.checkpoint.checkpoint(layer_forward, x, use_reentrant=False)
                continue

            result = layer(x, src_key_padding_mask=src_key_padding_mask, sin=sin, cos=cos)
            x = result[0]
            if result[1] is not None:
                moe_losses.append(result[1])
            if result[2] is not None:
                router_entropies.append(result[2])

        moe_aux_loss = torch.stack(moe_losses).mean() if moe_losses else None
        router_entropy = torch.stack(router_entropies).mean() if router_entropies else None
        return self.final_norm(x), moe_aux_loss, router_entropy

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import DualHelixBackbone, FSHMEncoder
from .backbones.common import ResBlock1d, RMSNorm, RMSNorm1d, SamePadConv1d
from .config import IGNORE_INDEX, KMER_PAD_ID, DnaConfig

LOGGER = logging.getLogger(__name__)


class DnaModel(nn.Module):
    """DNA masked-language model with a selectable primary tokenizer.

    Args:
        config: Model configuration. The primary token IDs have shape
            ``[batch, seq_len]``; optional ``kmer_ids`` provide an aligned
            side channel with the same sequence length.
    """

    def __init__(self, config: DnaConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size

        vocab_size = getattr(config, "vocab_size", 4096)
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=H,
            padding_idx=pad_token_id,
        )
        self.use_kmer_mix = bool(getattr(config, "use_kmer_mix", True))
        if self.use_kmer_mix:
            kmer_vocab_size = int(getattr(config, "kmer_vocab_size", 65))
            self.kmer_emb = nn.Embedding(
                num_embeddings=kmer_vocab_size,
                embedding_dim=H,
                padding_idx=KMER_PAD_ID,
            )
            self.kmer_gate = nn.Linear(2 * H, H, bias=False)
        else:
            self.kmer_emb = None
            self.kmer_gate = None

        self.stem_norm = RMSNorm(H)

        # Output projection (weight-tied to token_emb)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        if self.config.backbone == "dual_helix":
            LOGGER.info("[DnaModel] Initializing DualHelixBackbone")
            self.compress = nn.Identity()
            self.expand = nn.Identity()
            self.unet_fusion = nn.Identity()
            self.high_level_encoder = DualHelixBackbone(config)

        elif self.config.backbone == "legacy":
            # Legacy FSHMEncoder backbone
            S = self.config.low_level_stride
            K = self.config.low_level_kernel

            def make_down_block(stride):
                return ResBlock1d(
                    H, K, stride=stride, dropout=config.dropout * 0.5, drop_path=config.drop_path_rate * 0.1
                )

            if S == 1:
                self.compress = nn.Identity()
                self.expand = nn.Identity()
            elif S == 2:
                self.compress = make_down_block(stride=2)
                self.expand = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    SamePadConv1d(H, H, kernel_size=K, bias=False),
                    RMSNorm1d(H),
                    nn.SiLU(),
                )
            elif S == 4:
                self.compress = nn.Sequential(
                    make_down_block(stride=2),
                    make_down_block(stride=2),
                )
                self.expand = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    SamePadConv1d(H, H, kernel_size=K, bias=False),
                    RMSNorm1d(H),
                    nn.SiLU(),
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    SamePadConv1d(H, H, kernel_size=K, bias=False),
                    RMSNorm1d(H),
                    nn.SiLU(),
                )
            else:
                raise ValueError(f"Unsupported low_level_stride={S}")

            self.num_heads = config.num_attention_heads
            head_dim = H // self.num_heads
            if head_dim % 8 != 0:
                raise ValueError(
                    f"head_dim ({head_dim}) must be a multiple of 8 for Flash Attention efficiency. "
                    f"Check hidden_size ({H}) and num_attention_heads ({self.num_heads})."
                )

            self.high_level_encoder = FSHMEncoder(config)
            self.unet_fusion = nn.Sequential(
                nn.Linear(2 * H, H),
                RMSNorm(H),
                nn.SiLU(),
                nn.Dropout(config.dropout * 0.5),
            )
        else:
            LOGGER.error(
                "Unsupported backbone '%s'; refusing to fall back to legacy/Transformer.",
                self.config.backbone,
            )
            raise ValueError(f"Unsupported backbone {self.config.backbone!r}; expected 'dual_helix' or 'legacy'.")

        self.apply(self._init_weights)
        LOGGER.info("[Model] Weights initialized with Xavier/Kaiming")

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            if getattr(m, "is_router", False):
                if getattr(m, "force_nonzero_init", False):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                else:
                    nn.init.zeros_(m.weight)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _embed_sequence(self, input_ids: torch.Tensor, kmer_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Embed BPE token IDs and optional aligned 3-mer IDs.

        Args:
            input_ids: BPE token IDs with shape ``[batch, seq_len]``.
            kmer_ids: Optional aligned 3-mer IDs with shape ``[batch, seq_len]``
                or legacy ``[batch, seq_len, channels]``.

        Returns:
            Embedded hidden states with shape ``[batch, seq_len, hidden_size]``.
        """
        x = self.token_emb(input_ids)  # (B, L, H)
        if self.use_kmer_mix and kmer_ids is not None:
            if kmer_ids.ndim == 3:
                kmer_ids = kmer_ids[..., 0]
            kmer_ids = kmer_ids.to(device=input_ids.device, dtype=torch.long)
            kmer_ids = kmer_ids.clamp(0, self.kmer_emb.num_embeddings - 1)
            k = self.kmer_emb(kmer_ids)
            gate = torch.sigmoid(self.kmer_gate(torch.cat([x, k], dim=-1)))
            x = x + gate * k
        x = self.stem_norm(x)  # (B, L, H)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        kmer_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
        input_ids_rc: torch.Tensor | None = None,
        attention_mask_rc: torch.Tensor | None = None,
        kmer_ids_rc: torch.Tensor | None = None,
        capture_attention: bool = False,
        attention_layers: Sequence[int] | None = None,
        attention_heads: Sequence[int] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the masked-language-model forward pass.

        Args:
            input_ids: BPE token IDs with shape ``[batch, seq_len]``.
            kmer_ids: Optional aligned 3-mer side-channel IDs.
            attention_mask: Optional mask with ``1`` for valid tokens and ``0``
                for padding.
            labels: Optional BPE token targets with ``IGNORE_INDEX`` ignored.
            class_weights: Optional per-token-class loss weights.
            input_ids_rc: Optional reverse-complement BPE token IDs. These must
                already respect tokenizer boundaries; the model does not retokenize.
            attention_mask_rc: Optional reverse-complement attention mask.
            kmer_ids_rc: Optional reverse-complement aligned 3-mer IDs.
            capture_attention: Opt-in manual Telescope attention capture for
                visualization. Disabled during normal training/inference.
            attention_layers: Optional zero-based DualHelix layers to capture.
            attention_heads: Optional zero-based Telescope heads to capture.

        Returns:
            Dictionary containing ``logits`` with shape
            ``[batch, seq_len, vocab_size]`` plus optional losses and
            ``last_hidden_state`` with shape ``[batch, seq_len, hidden_size]``.
        """
        del kwargs
        if capture_attention and self.config.backbone != "dual_helix":
            raise ValueError("attention capture is only implemented for the dual_helix backbone")

        attention_layer_set = set(attention_layers) if attention_layers is not None else None
        attention_head_set = set(attention_heads) if attention_heads is not None else None
        _, L = input_ids.shape

        x = self._embed_sequence(input_ids, kmer_ids=kmer_ids)  # (B, L, H)

        # 2. Downsample
        x_skip = x
        x_down = self.compress(x.transpose(1, 2)).transpose(1, 2)  # (B, L', H)

        # Build attention mask for encoder
        mask_down = None
        if attention_mask is not None:
            stride = 1 if self.config.backbone == "dual_helix" else self.config.low_level_stride
            if stride > 1:
                target_len = x_down.size(1)
                mask_down = attention_mask[:, : target_len * stride : stride]
                if mask_down.size(1) < target_len:
                    pad = target_len - mask_down.size(1)
                    mask_down = F.pad(mask_down, (0, pad), value=0)
                elif mask_down.size(1) > target_len:
                    mask_down = mask_down[:, :target_len]
            else:
                mask_down = attention_mask
            mask_down = mask_down == 0  # invert: True = ignore

        # Prepare the optional reverse-complement stream.
        x_rc_emb = None
        mask_rc_inv = None
        use_dual = (
            getattr(self.config, "use_dual_strand", False)
            and input_ids_rc is not None
            and self.config.backbone == "dual_helix"
        )
        if use_dual:
            x_rc_emb = self._embed_sequence(input_ids_rc, kmer_ids=kmer_ids_rc)  # (B, L_rc, H)
            if attention_mask_rc is not None:
                mask_rc_inv = attention_mask_rc == 0

        # 4. Encoder
        if self.config.backbone == "dual_helix":
            enc_result = self.high_level_encoder(
                x_down,
                src_key_padding_mask=mask_down,
                x_rc=x_rc_emb,
                mask_rc=mask_rc_inv,
                capture_attention=capture_attention,
                attention_layers=attention_layer_set,
                attention_heads=attention_head_set,
            )
        else:
            enc_result = self.high_level_encoder(x_down, src_key_padding_mask=mask_down)

        attention_maps = None
        if isinstance(enc_result, tuple):
            x_enc = enc_result[0]
            aux_loss = enc_result[1]
            router_entropy = enc_result[2]
            if len(enc_result) > 3:
                attention_maps = enc_result[3]
        else:
            x_enc = enc_result
            aux_loss = None
            router_entropy = None

        # Separate aux-loss channels so training can weight them independently.
        # - legacy backbone (FSHMEncoder): aux_loss is FSHM biological-router aux
        # - dual_helix backbone (Telescope): aux_loss is MoE FFN aux
        fshm_router_aux_loss = None
        moe_aux_loss = None
        if aux_loss is not None:
            if self.config.backbone == "legacy":
                fshm_router_aux_loss = aux_loss
            else:
                moe_aux_loss = aux_loss

        # 5. Upsample
        x_up = self.expand(x_enc.transpose(1, 2)).transpose(1, 2)
        if x_up.size(1) != L:
            x_up = x_up[:, :L, :]

        # 6. U-Net skip (legacy backbone only)
        if self.config.backbone != "dual_helix":
            x_fused = torch.cat([x_up, x_skip], dim=-1)
            x_up = self.unet_fusion(x_fused)

        # 7. Weight-tied output projection
        logits = F.linear(x_up, self.token_emb.weight, self.output_bias)

        loss = None

        if labels is not None:
            loss_fct = (
                nn.CrossEntropyLoss(
                    ignore_index=IGNORE_INDEX,
                    weight=class_weights,
                    reduction="mean",
                )
                if class_weights is not None
                else nn.CrossEntropyLoss(
                    ignore_index=IGNORE_INDEX,
                    reduction="mean",
                )
            )
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        result = {
            "logits": logits,
            "loss": loss,
            "moe_aux_loss": moe_aux_loss,
            "fshm_router_aux_loss": fshm_router_aux_loss,
            "router_entropy": router_entropy,
            "last_hidden_state": x_up,
        }
        if capture_attention:
            result["attention_maps"] = attention_maps or []
        return result

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
        return n_params

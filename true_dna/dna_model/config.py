import logging

LOGGER = logging.getLogger(__name__)

# Token ids are aligned with dna_model/bpe_vocab/tokenizer.json.
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
MASK_TOKEN_ID = 4
SPECIAL_TOKENS_MAX_ID = 4
KMER_PAD_ID = 0
IGNORE_INDEX = -100

# Misc defaults (used by various scripts)
MUTA_PROB = 0.005
DEFAULT_MAX_LEN = 4096

# Base vocabulary (for tokenizer convenience; not the supervised target size)
BASES = ["A", "C", "G", "T", "N"]

# Masking constants
DEFAULT_MASK_FRACTION = 0.15
DEFAULT_MEAN_SPAN_LENGTH = 3.0
MIN_MASK_FRACTION = 0.20
MAX_MASK_FRACTION = 0.40


class DnaConfig:
    """Lightweight model config.

    The primary token stream can contain BPE tokens or one token per DNA base.
    A small aligned 3-mer side channel can be mixed into its embeddings when
    ``use_kmer_mix`` is enabled.
    """

    def __init__(
        self,
        *,
        max_input_bases: int = 4096,
        low_level_kernel: int = 8,
        low_level_stride: int = 4,
        high_level_layers: int = 8,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: int = None,
        tokenizer_type: str = "bpe",
        tokenizer_vocab_sha256: str | None = None,
        vocab_size: int = 4096,
        pad_token_id: int = PAD_TOKEN_ID,
        mask_token_id: int = MASK_TOKEN_ID,
        cls_token_id: int = CLS_TOKEN_ID,
        sep_token_id: int = SEP_TOKEN_ID,
        unk_token_id: int = UNK_TOKEN_ID,
        base_target_vocab_size: int = 4,
        kmer_vocab_sizes: list[int] = None,
        use_kmer_mix: bool = True,
        kmer_size: int = 3,
        kmer_vocab_size: int = 65,
        dropout: float = 0.1,
        use_conv_ffn: bool = False,
        conv_kernel_size: int = 7,
        drop_path_rate: float = 0.0,
        gradient_checkpointing: bool = False,
        # FSH-M v2.1 Params
        use_mgke: bool = True,
        use_conv_stem: bool = False,
        mgke_kmer_sizes: list[int] = None,
        fshm_layers: list[int] = None,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        fshm_slow_stride: int = 4,
        # Mamba
        mamba_version: str = None,
        use_mamba2: bool = True,
        mamba2_head_dim: int = 64,
        mamba2_ngroups: int = 1,
        # Dual Helix
        backbone: str = "dual_helix",
        dual_helix_bridge_interval: int = 4,
        disable_bridge: bool = False,
        microscope_window_size: int = 128,
        # MoE
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_beta_z: float = 0.001,
        expert_drop_prob: float = 0.1,
        expert_drop_warmup_steps: int = 2000,
        # Pre-BPE Frameshift Perturbation
        frameshift_prob: float = 0.05,
        # Cross-Strand Fusion / Dual-Strand Processing
        use_dual_strand: bool = False,
        cross_strand_gate_interval: int = 4,
        rc_equivariant_tying: bool = False,
    ):
        if max_input_bases <= 0:
            raise ValueError(f"max_input_bases must be positive, got {max_input_bases}")
        if low_level_kernel <= 0 or low_level_kernel % 2 != 0:
            raise ValueError(f"low_level_kernel must be positive and even, got {low_level_kernel}")
        if low_level_stride not in [1, 2, 4]:
            raise ValueError(f"low_level_stride must be 1, 2, or 4, got {low_level_stride}")
        if high_level_layers <= 0:
            raise ValueError(f"high_level_layers must be positive, got {high_level_layers}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_key_value_heads <= 0 or num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_key_value_heads must be positive and divide num_attention_heads; "
                f"got {num_key_value_heads} and {num_attention_heads}"
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        head_dim = hidden_size // num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE. Adjust hidden_size or num_attention_heads.")
        tokenizer_type = str(tokenizer_type).strip().lower()
        if tokenizer_type not in {"bpe", "base"}:
            raise ValueError(f"tokenizer_type must be 'bpe' or 'base', got {tokenizer_type!r}")
        if vocab_size <= SPECIAL_TOKENS_MAX_ID:
            raise ValueError(f"vocab_size must be > {SPECIAL_TOKENS_MAX_ID}, got {vocab_size}")
        special_ids = [pad_token_id, mask_token_id, cls_token_id, sep_token_id, unk_token_id]
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("special token IDs must be distinct")
        if any(token_id < 0 or token_id >= vocab_size for token_id in special_ids):
            raise ValueError("special token IDs must be within the configured vocabulary")
        if base_target_vocab_size < 4:
            raise ValueError(f"base_target_vocab_size should be at least 4, got {base_target_vocab_size}")

        if kmer_vocab_sizes is None:
            kmer_vocab_sizes = [kmer_vocab_size] if use_kmer_mix else []
        if not isinstance(kmer_vocab_sizes, list):
            raise ValueError(f"kmer_vocab_sizes must be a list, got {type(kmer_vocab_sizes)}")
        if any(int(size) <= 1 for size in kmer_vocab_sizes):
            raise ValueError("all kmer_vocab_sizes entries must be greater than 1")
        if kmer_size <= 0:
            raise ValueError(f"kmer_size must be positive, got {kmer_size}")
        if kmer_vocab_size <= 1:
            raise ValueError(f"kmer_vocab_size must be > 1, got {kmer_vocab_size}")

        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}")
        if any(value <= 0 for value in (ssm_d_state, ssm_d_conv, ssm_expand, fshm_slow_stride)):
            raise ValueError("SSM dimensions and fshm_slow_stride must be positive")
        if mamba2_head_dim <= 0 or mamba2_ngroups <= 0:
            raise ValueError("mamba2_head_dim and mamba2_ngroups must be positive")
        if dual_helix_bridge_interval <= 0 or cross_strand_gate_interval <= 0:
            raise ValueError("bridge intervals must be positive")
        if microscope_window_size <= 0:
            raise ValueError("microscope_window_size must be positive")
        if moe_num_experts <= 0 or not 1 <= moe_top_k <= moe_num_experts:
            raise ValueError("moe_top_k must be between 1 and moe_num_experts")
        if moe_beta_z < 0.0:
            raise ValueError("moe_beta_z must be non-negative")
        if not 0.0 <= expert_drop_prob <= 1.0:
            raise ValueError("expert_drop_prob must be between 0 and 1")
        if expert_drop_warmup_steps < 0:
            raise ValueError("expert_drop_warmup_steps must be non-negative")
        if not 0.0 <= frameshift_prob <= 1.0:
            raise ValueError("frameshift_prob must be between 0 and 1")

        if mgke_kmer_sizes is None:
            mgke_kmer_sizes = [3, 5, 7]
        if fshm_layers is None:
            fshm_layers = []
        elif isinstance(fshm_layers, str):
            try:
                fshm_layers = [int(value) for value in fshm_layers.split(",") if value.strip()]
            except ValueError as exc:
                raise ValueError("fshm_layers must be a list of indices or a comma-separated string") from exc
        if any(int(size) <= 0 for size in mgke_kmer_sizes):
            raise ValueError("mgke_kmer_sizes entries must be positive")
        if any(int(index) < 0 or int(index) >= high_level_layers for index in fshm_layers):
            raise ValueError("fshm_layers entries must reference configured layer indices")

        self.max_input_bases = int(max_input_bases)
        self.low_level_kernel = int(low_level_kernel)
        self.low_level_stride = int(low_level_stride)
        self.high_level_layers = int(high_level_layers)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.tokenizer_type = tokenizer_type
        self.tokenizer_vocab_sha256 = tokenizer_vocab_sha256
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.cls_token_id = int(cls_token_id)
        self.sep_token_id = int(sep_token_id)
        self.unk_token_id = int(unk_token_id)
        self.base_target_vocab_size = int(base_target_vocab_size)
        self.kmer_vocab_sizes = kmer_vocab_sizes
        self.use_kmer_mix = bool(use_kmer_mix)
        self.kmer_size = int(kmer_size)
        self.kmer_vocab_size = int(kmer_vocab_size)
        self.dropout = float(dropout)
        self.use_conv_ffn = bool(use_conv_ffn)
        self.conv_kernel_size = int(conv_kernel_size)
        self.drop_path_rate = float(drop_path_rate)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.use_mgke = bool(use_mgke)
        self.use_conv_stem = bool(use_conv_stem)
        self.mgke_kmer_sizes = mgke_kmer_sizes
        self.fshm_layers = fshm_layers
        self.ssm_d_state = int(ssm_d_state)
        self.ssm_d_conv = int(ssm_d_conv)
        self.ssm_expand = int(ssm_expand)
        self.fshm_slow_stride = int(fshm_slow_stride)
        self.mamba_version = str(mamba_version or ("mamba2" if use_mamba2 else "mamba1")).lower()
        if self.mamba_version not in {"mamba1", "mamba2", "mamba3"}:
            raise ValueError(f"mamba_version must be one of mamba1, mamba2, mamba3; got {self.mamba_version}")
        self.use_mamba2 = self.mamba_version == "mamba2"
        self.mamba2_head_dim = int(mamba2_head_dim)
        self.mamba2_ngroups = int(mamba2_ngroups)
        backbone_name = str(backbone).lower()
        if backbone_name not in {"dual_helix", "bimamba", "transformer", "legacy"}:
            LOGGER.error(
                "Unsupported backbone '%s'; not falling back to another architecture.",
                backbone,
            )
            raise ValueError(f"backbone must be one of dual_helix, bimamba, transformer, or legacy; got {backbone!r}")
        self.backbone = backbone_name
        self.dual_helix_bridge_interval = int(dual_helix_bridge_interval)
        self.disable_bridge = bool(disable_bridge)
        self.microscope_window_size = int(microscope_window_size)
        self.use_moe = bool(use_moe)
        self.moe_num_experts = int(moe_num_experts)
        self.moe_top_k = int(moe_top_k)
        self.moe_beta_z = float(moe_beta_z)
        self.expert_drop_prob = float(expert_drop_prob)
        self.expert_drop_warmup_steps = int(expert_drop_warmup_steps)
        self.frameshift_prob = float(frameshift_prob)
        self.use_dual_strand = bool(use_dual_strand)
        self.cross_strand_gate_interval = int(cross_strand_gate_interval)
        self.rc_equivariant_tying = bool(rc_equivariant_tying)

    def __repr__(self):
        return (
            f"DnaConfig(max_input_bases={self.max_input_bases}, "
            f"hidden_size={self.hidden_size}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"high_level_layers={self.high_level_layers}, "
            f"low_level_stride={self.low_level_stride}, "
            f"tokenizer_type={self.tokenizer_type!r}, "
            f"vocab_size={self.vocab_size}, "
            f"use_kmer_mix={self.use_kmer_mix}, "
            f"kmer_size={self.kmer_size}, "
            f"use_conv_ffn={self.use_conv_ffn}, "
            f"drop_path_rate={self.drop_path_rate}, "
            f"gradient_checkpointing={self.gradient_checkpointing}, "
            f"fshm_layers={self.fshm_layers}, "
            f"fshm_slow_stride={self.fshm_slow_stride}, "
            f"mamba_version={self.mamba_version}, "
            f"backbone={self.backbone})"
        )

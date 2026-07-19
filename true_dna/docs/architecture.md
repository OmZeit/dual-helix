# Architecture and training objective

This document describes the active implementation. It is a technical map of
the code, not a claim that a checkpoint has learned a particular biological
mechanism.

## Input representation

The canonical input is a genomic sequence normalized to `A`, `C`, `G`, `T`,
and `N`. The primary stream is selected with `--tokenizer-type bpe|base`.
`DnaTokenizer` wraps the checked-in Hugging Face Tokenizers BPE model at
`dna_model/bpe_vocab/tokenizer.json`, while `BaseTokenizer` emits one token per
base. Both vocabularies include `[PAD]`,
`[UNK]`, `[CLS]`, `[SEP]`, and `[MASK]` special tokens.

Alongside each primary token, the tokenizer can emit one aligned k-mer ID. The ID
is derived from the first k bases covered by the token's source-sequence
offset; special, padding, too-short, or ambiguous spans receive ID zero. The
current command-line training path uses the first value passed through
`--k_mer_sizes`, normally 3, as this side channel.

The model embeds primary and k-mer IDs separately. A learned sigmoid gate controls
the k-mer contribution before RMS normalization:

```text
sequence -> BPE/base IDs ------------------> token embedding --+
          source offsets -> aligned k-mer -> k-mer embedding ---+-> gated sum -> RMSNorm
```

The selected BPE or base stream remains the primary representation. A k-mer ID
does not replace its aligned primary token.

## Data loading and augmentation

Training accepts raw FASTA or pre-tokenized `uint16` arrays. Raw FASTA records
are divided into overlapping windows using `max_length` and `stride`.
Pre-tokenized data uses a token-offset index and can be memory-mapped when the
arrays do not fit comfortably in RAM.

Training and evaluation records are selected separately so windows from one
record are not assigned to both splits. This prevents direct record leakage;
datasets with related assemblies may still require accession-, species-, or
homology-aware splitting for a particular scientific question.

The loader supports per-file caps and round-robin or weighted sampling. Raw
FASTA mode also supports stochastic reverse-complement augmentation,
single-base frameshift perturbation, and paired forward/reverse-complement
inputs. Paired reverse-complement mode is unavailable for pre-tokenized arrays.

## DualHelix backbone

`dual_helix` is the default backbone. It maintains two representations at each
layer:

- The **Microscope stream** stays at full token resolution and applies a
  residual bidirectional Mamba mixer.
- The **Telescope stream** is compressed to approximately one quarter of the
  token length and applies a RoPE transformer block. Grouped-query attention
  and a mixture-of-experts feed-forward layer are configurable.
- Periodic **pooled bridges** exchange information in both directions. The
  full-resolution stream is adaptively pooled into the compressed stream; the
  compressed stream is upsampled and gated into the full-resolution stream.

```text
                         +-- Microscope: full length, bidirectional Mamba --+
embedded token states ---+                                                    +--> full-length states
                         +-- compress -> Telescope: RoPE transformer -------+
                                  ^       periodic pooled bridges       |
                                  +-------------------------------------+
```

When dual-strand processing is enabled and paired inputs are supplied, the
backbone runs corresponding forward and reverse-complement streams. Periodic
cross-strand gates exchange aligned information between them. Exact
strand-swap parameter tying is optional.

The `legacy` backbone remains available for compatible checkpoints and uses a
compressed FSHM encoder with an upsample-and-skip path. Backbone choice and all
shape-defining settings are stored in training checkpoints and must be
reconstructed when loading them.

## Output and objective

The final full-resolution hidden states are projected to the selected tokenizer's vocabulary logits
using the input token-embedding matrix as the projection weight, plus a learned
output bias. With labels, the primary loss is token cross-entropy at selected
masked positions; padding and unselected positions use the ignore index.

The masking implementation selects contiguous token spans and uses the common
80/10/10 replacement policy: most selected tokens become `[MASK]`, some become
random non-special vocabulary tokens, and the rest remain unchanged. The
training loop can vary the masked fraction with a curriculum.

Optional training terms include reverse-complement consistency, mixture-of-
experts routing losses, and the legacy FSHM router loss. Their coefficients are
independent command-line settings. A checkpoint should therefore be compared
only after confirming its saved configuration and enabled objectives.

`DnaModel.forward` returns vocabulary logits and full-resolution hidden states,
plus any active primary or auxiliary losses. Telescope attention capture is an
explicit diagnostic mode used by the interaction visualizer and is disabled
during ordinary training and inference.

## Runtime constraints

The active DualHelix implementation uses CUDA-backed `mamba-ssm` and
`causal-conv1d` kernels. Training and the supplied checkpoint evaluation tools
require CUDA. The RTX 5080 requirements file pins the project's CUDA 12.8
environment; other devices may need a different PyTorch wheel and locally
compatible kernel builds.

The repository includes a tokenizer and source code, but not a trained model or
a complete benchmark dataset. Benchmark and probe outputs should be interpreted
against data provenance, split design, handcrafted controls, and random-model
baselines. They do not by themselves establish biological causality, safety,
or clinical usefulness.

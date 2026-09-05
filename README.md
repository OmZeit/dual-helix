# True DNA

True DNA is a research codebase for self-supervised masked-language modeling
on genomic DNA. It combines a selectable BPE or single-base token stream with an aligned k-mer feature
channel and provides tools for collecting public genomes, training the model,
evaluating checkpoints, and probing learned representations.

## Project status

True DNA is experimental alpha-stage research software. The repository includes
training and evaluation code, but no pretrained model, benchmark claim, or
clinical validation. The latest revision on `main` is the only supported
version.

The active training and inference path requires Linux, an NVIDIA GPU, and
CUDA-compatible builds of PyTorch, `mamba-ssm`, and `causal-conv1d`. CPU-only
environments are suitable for linting and most unit tests, but the training,
evaluation, and benchmark entrypoints deliberately reject CPU execution.

## Repository layout

- `true_dna/dna_model/` contains tokenization, data loading, masking, model,
  checkpoint, and benchmark utilities.
- `true_dna/scripts/` contains command-line entrypoints for data preparation,
  training, evaluation, benchmarks, and diagnostics.
- `true_dna/configs/` contains example corpus and sampling configuration.
- `true_dna/tests/` contains the automated test suite.
- `true_dna/docs/architecture.md` describes the model and training objective.

## Installation

Python 3.10 through 3.12 is recommended. Create an isolated environment from
the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r true_dna/requirements.txt
```

For development, also install the test and lint dependencies:

```bash
python -m pip install -r true_dna/requirements-dev.txt
```

`true_dna/requirements_rtx5080.txt` records tested direct-dependency pins for
the project's CUDA 12.8 and RTX 5080 setup. Install it instead of the general
runtime requirements when reproducing that environment:

```bash
python -m pip install -r true_dna/requirements_rtx5080.txt
python -m pip install --no-build-isolation \
  -r true_dna/requirements_rtx5080-kernels.txt
```

Other GPU and CUDA combinations may require different PyTorch wheels and
locally compiled Mamba kernels. Confirm that a small model can be constructed
and run before starting a long job.

For the two target GPU classes, the repository also provides a setup helper.
Run this on Linux (or a Colab Linux runtime), not native Windows; it creates a
root-level `.venv`, installs the CUDA 12.8 dependencies and Mamba kernels, and
checks that PyTorch can see the GPU:

```bash
bash true_dna/scripts/setup_cuda_env.sh --rtx5080
# or: bash true_dna/scripts/setup_cuda_env.sh --a100
```

It requires Python 3.10--3.12, an NVIDIA driver, the CUDA toolkit (`nvcc`),
and a C++ compiler to already be present. Use `--python python3.12` if the
default `python3` is a different version.

## Local GPU workflow (Windows + WSL2)

Use the Windows checkout for editing if preferred, but run CUDA installation,
data collection, and training from a separate **native WSL filesystem copy**.
The native copy avoids the slow file metadata and write behavior under
`/mnt/<drive>` and has its own virtual environment. The Windows and native
copies do not automatically synchronize. Once it exists, open
`~/dual-helix-native` through VS Code's WSL integration (`code .` from that
directory) and make runtime changes there; the setup helper intentionally
refuses to overwrite a nonempty native workspace.

### One-time setup

From PowerShell, open your Ubuntu WSL distribution:

```powershell
wsl -d Ubuntu-24.04
```

Then, in the Ubuntu terminal, make the native copy and build its CUDA
environment. Adjust `/mnt/g/dual-helix` only if the Windows checkout is on a
different drive or path.

```bash
cd /mnt/g/dual-helix
bash true_dna/scripts/setup_native_wsl_workspace.sh --target "$HOME/dual-helix-native"

cd ~/dual-helix-native
bash true_dna/scripts/setup_cuda_env.sh --rtx5080 --python python3.12
```

For Colab or an 80 GB A100 machine, use `--a100` instead of `--rtx5080` in
the setup and launch commands. Confirm the installation before a long job:

```bash
cd ~/dual-helix-native/true_dna
source ../.venv/bin/activate
bash scripts/run_controlled_ablation.sh --rtx5080 --smoke
```

### Where local artifacts live

All paths below are relative to `~/dual-helix-native/true_dna` unless noted.

| Need | Location |
| --- | --- |
| Final clean training FASTAs | `data/domain_fastas/bacteria.fa`, `archaea.fa`, `eukaryotes.fa` |
| Corpus provenance and sizes | `data/domain_fastas/dataset_manifest.json` and `dataset_card.md` |
| Assembly-held-out split | `data/domain_fastas/split_manifest.jsonl` |
| Pinned taxa and families | `data/metadata/taxonomic_selection_v1.json` |
| Per-species FASTAs, QC, and provenance before cleanup | `data/bacteria/`, `data/archaea/`, `data/eukaryotes/` |
| Temporary NCBI packages | `~/.cache/true_dna_ncbi/` (outside the repository; safe to remove after a completed run) |
| Ablation commands, metrics, and run status | `experiments/controlled_architecture_ablation_v2/ablation_manifest.json` |
| Model checkpoints and logs | the experiment directory selected by the training command |

The final FASTAs are the quality-filtered, deduplicated files intended for
training. Do not train from temporary NCBI packages or the per-species
intermediates unless deliberately debugging the data pipeline.

### Launch choices

For a new, destructive corpus build followed automatically by the three
backbones, run:

```bash
cd ~/dual-helix-native/true_dna
source ../.venv/bin/activate
bash scripts/run_fresh_pretrain_and_ablation.sh --rtx5080
```

For an already verified corpus, preserve the existing FASTAs and start at
split creation/training instead:

```bash
cd ~/dual-helix-native/true_dna
source ../.venv/bin/activate
bash scripts/run_validated_corpus_ablation.sh --rtx5080
```

Both launchers use Weights & Biases by default and prompt for its API key
before long-running work. The fresh launcher also prompts for optional NCBI
credentials before it deletes or downloads data. Credentials are exported only
in the current shell; see `scripts/export_wandb_credentials.sh` and
`scripts/export_research_credentials.sh`. Add `--no-wandb` only when offline
logging is intentional. Use `--a100` on A100 hardware, and put trainer options
after `--`, for example `-- --steps 1000`.

The three architecture baselines use the same **single-base tokenizer** and
the same 10-token primary vocabulary (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`,
`[MASK]`, A, C, G, T, N). This makes their parameter and performance
differences attributable to the backbone. The launchers additionally build a
deterministic 4,096-token BPE vocabulary from the verified corpus and run a
matched Transformer BPE control. That control necessarily changes the
embedding/output vocabulary and parameter count, so compare it to the
single-base Transformer using base-normalized metrics such as
`eval/bits_per_base`, not as an architecture-only result. The ablation manifest
records tokenizer type, vocabulary size, and BPE vocabulary checksum per run.

## Docker

The CPU image provides a clean, portable test environment. It does not support
training:

```bash
docker build --file docker/Dockerfile.cpu --tag true-dna:test .
docker run --rm true-dna:test
```

The experimental GPU image reproduces the CUDA 12.8 and RTX 5080 dependency
stack. The host must run Linux with a CUDA 12.8-compatible NVIDIA driver and
the NVIDIA Container Toolkit. Docker packages the user-space CUDA libraries;
it does not replace the host driver.

```bash
docker build --file docker/Dockerfile.cuda128 --tag true-dna:cuda128 .
docker run --rm --gpus all true-dna:cuda128
```

Mount training data read-only and write experiment outputs to a separate host
directory. From the repository root, for example:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --mount type=bind,src="$(pwd)/data",dst=/workspace/true_dna/data,readonly \
  --mount type=bind,src="$(pwd)/experiments",dst=/workspace/true_dna/experiments \
  true-dna:cuda128 \
  python scripts/train.py \
    --fasta "data/balanced/*.fa" \
    --tokenizer-type bpe \
    --save_dir experiments/example_run \
    --no_wandb
```

Create the two host directories before running the command. The runtime user
flags let output files retain the invoking Linux user's ownership. Data,
checkpoints, credentials, and local environment files are excluded from the
build context. Pass optional service credentials at runtime rather than placing
them in an image or command committed to source control. The CPU image is
exercised by CI; the GPU image still requires a smoke test on the target NVIDIA
hardware.

The commands below assume the working directory is `true_dna`, which keeps
the default tokenizer and configuration paths valid:

```bash
cd true_dna
```

## Configuration and external services

No credential is required for local tests or training with existing FASTA
files. NCBI Datasets and NCBI E-utilities are optional data sources, and
Weights & Biases is optional experiment tracking. Environment variable names
and safe empty defaults are documented in [`.env.example`](.env.example).
The project does not automatically load `.env` files; export only the values
needed by your shell or job runner. Never commit populated credentials.

## Prepare data

Training accepts one or more FASTA files directly. To fetch a small RefSeq
sample, install the NCBI Datasets CLI and run:

```bash
python scripts/fetch_genomes.py \
  --species "Escherichia coli" \
  --target-mb 10 \
  --out-dir data/raw_downloads \
  --results-dir data/balanced \
  --refseq-only \
  --run-qc
```

The fetcher resolves the taxon, downloads assemblies, filters and deduplicates
contigs, writes a FASTA and manifest under `data/balanced/`, and optionally
runs sequence-level quality control. Use a numeric NCBI Taxonomy ID when a
stable, unambiguous taxon selection matters. Run
`python scripts/fetch_genomes.py --help` for CSV/YAML batch input, assembly
filters, append mode, and dry-run options.

For a controlled multi-domain RefSeq corpus, use the taxonomically validated
selection workflow. It resolves each candidate scientific name through NCBI
Taxonomy, records the resulting TaxID and family, rejects a repeated family
within a domain, and applies one hard storage budget to the *whole* domain
rather than to each species. The resulting selection and download manifests
record the NCBI TaxIDs, accessions, source URLs, file sizes, and SHA-256
checksums.

```bash
# Preview and validate the representative species; this does not download genomes.
python scripts/build_taxonomic_manifest.py \
  --config configs/species_taxonomically_diverse_v1.json \
  --output data/metadata/taxonomic_selection_v1.json

# Deliberately replace generated pilot data, then build toward 5 GB per domain.
# The final corpus target is 15 GB total. The collector has a 15% hard
# headroom (5.75 GB/domain) and --verify accepts 4.25--5.75 GB/domain.
python scripts/prepare_refseq_pretrain_data.py \
  --hard-delete-generated-data \
  --selection-manifest data/metadata/taxonomic_selection_v1.json \
  --run-download \
  --target-gb-per-domain 5 \
  --exclude-plasmids \
  --allow-underfilled-domain archaea \
  --write-manifest \
  --verify
```

The second command removes only repository-generated corpus artifacts
(`data/bacteria`, `data/archaea`, `data/eukaryotes`, `data/raw_downloads`,
generated domain FASTAs, and generated metadata); it never deletes source
code, experiments, or checkpoints. Expect temporary disk use above 15 GB
while the per-species FASTAs and QC outputs coexist. If NCBI has insufficient
eligible records for a domain, the verification step reports the shortfall
instead of silently exceeding the budget or substituting a related family.

For WSL, keep temporary NCBI packages on Linux's filesystem rather than under
`/mnt/<drive>`. The collection commands default to
`~/.cache/true_dna_ncbi`, use a bounded four-download prefetch window, and
write/filter assemblies in the fixed NCBI selection order. You can override
the scratch location (provided it is a native Linux path) and tune the bounded
window, for example `--scratch-dir "$HOME/.cache/true_dna_ncbi" --parallel-downloads 4`.
The output remains deterministic for the same NCBI responses and selection
manifest; only package transfer overlaps.

For the best sustained corpus-build and training throughput, create a separate
native-WSL copy of the repository (the Windows source checkout is left
untouched):

```bash
cd /mnt/g/dual-helix
bash true_dna/scripts/setup_native_wsl_workspace.sh --target "$HOME/dual-helix-native"
cd "$HOME/dual-helix-native"
bash true_dna/scripts/setup_cuda_env.sh --rtx5080 --python python3.12
```

Run data collection and training from the native copy after its environment is
ready. This avoids the high metadata/write overhead of `/mnt/g`; do not try to
share the virtual environment between the Windows-mounted and native copies.

Raw FASTA is the simplest input format. Large corpora can be pre-tokenized to
memory-mapped `uint16` arrays:

```bash
python scripts/preprocess_dna_stream.py \
  --fasta "data/balanced/*.fa" \
  --tokenizer-type bpe \
  --workers 4
```

Pass `--tokenizer-type base` to create one-token-per-base arrays instead. Base
and BPE caches use distinct filenames and carry tokenizer compatibility
metadata, so both can exist next to the same FASTA. Use the same
`--tokenizer-type` for preprocessing, training, and evaluation. The legacy
value `single_base` is accepted as an alias for `base`. Legacy BPE caches
without compatibility metadata remain readable; rerun preprocessing with
`--force` to replace and annotate one.

The loader uses the generated `_input_ids.npy`, `_kmer_ids.npy`, and
`_offsets.json` files when the files are complete and sequence-level
augmentation is disabled. Reverse-complement pairs, reverse-complement
augmentation, and pre-BPE frameshift augmentation require raw FASTA. To use
pre-tokenized arrays during training, set `--lambda_rc_consist 0`,
`--use_reverse_prob 0`, and `--frameshift_prob 0`.

## Train

Start with a small configuration and disable external experiment logging while
validating the environment:

```bash
python scripts/train.py \
  --fasta "data/balanced/*.fa" \
  --tokenizer-type bpe \
  --save_dir experiments/example_run \
  --max_length 1024 \
  --stride 512 \
  --batch_size 2 \
  --grad_accum 4 \
  --hidden_size 384 \
  --high_level_layers 4 \
  --num_attention_heads 6 \
  --steps_per_epoch 100 \
  --epochs 1 \
  --amp \
  --no_wandb
```

Training is CUDA-only. Increase sequence length, model size, batch size, and
worker count only after measuring memory use on the target system. Useful
production options include gradient checkpointing, EMA, curriculum masking,
weighted sampling, reverse-complement consistency, dual-strand processing,
and mixture-of-experts routing. Inspect the current option set with:

```bash
python scripts/train.py --help
```

Checkpoints are written to `--save_dir`; periodic checkpoints use names such
as `checkpoint_step2000.pt`, while `checkpoint_latest.pt` tracks the latest
saved state. Use `--auto_resume` or `--resume_from PATH` to resume a run.
Weights & Biases logging is enabled unless `--no_wandb` is supplied.

## Controlled architecture ablations

The controlled runner compares three explicitly different backbones: the
Dual Helix bridge model, a full-resolution BiMamba-only model, and a
full-resolution RoPE Transformer-only model. It also includes two Dual Helix
bridge controls (bridge disabled and bridge every layer). `legacy` is not used
as a BiMamba baseline because it contains a separate strided FSHM path.

First create a corpus and an assembly-held-out split. This downloads public
RefSeq data and may take substantial time and disk space; the 64 MB-per-domain
default is appropriate for an initial ablation smoke run, not a final study.
An NCBI API key is optional but increases the current Datasets CLI rate limit
from 5 to 10 requests per second. Export it for the current shell rather than
putting it in a command or committing it to `.env`:

```bash
export NCBI_API_KEY='your-ncbi-api-key'
export NCBI_TAXONOMY_EMAIL='you@example.org'
```

```bash
cd true_dna
python scripts/prepare_ablation_data.py \
  --download-refseq \
  --target-mb-per-domain 64
```

The download path selects RefSeq accessions, cleans sequence chunks, rejects
short or N-rich contigs, performs exact reverse-complement deduplication, runs
domain-specific entropy/N-content QC, records GC statistics, and retains
assembly provenance for the held-out split. It does not perform homology
screening, CheckM, or FCS contamination validation; add and document those
separately before making a final biological benchmark claim.

For an existing corpus produced by the current fetcher, create the split from
the assembly provenance tags in its FASTA headers:

```bash
python scripts/prepare_ablation_data.py \
  --fasta "data/domain_fastas/*.fa"
```

Use the hardware launcher on Linux with a CUDA-enabled Python environment.
The RTX 5080 profile keeps an effective global batch of 16 with one-sample
micro-batches, 16 accumulation passes, and gradient checkpointing; the A100
profile uses longer contexts. Run the parameter-only preflight before training.

```bash
bash scripts/run_controlled_ablation.sh --rtx5080 --smoke
bash scripts/run_controlled_ablation.sh --a100 --smoke
bash scripts/run_controlled_ablation.sh --rtx5080 --preflight-only --allow-dirty
bash scripts/run_controlled_ablation.sh --a100 --only-baselines --allow-dirty
bash scripts/run_controlled_ablation.sh --a100 --backbones transformer --allow-dirty
```

For a full fresh corpus followed immediately by the three primary backbones,
use the native-WSL-only launcher below. It deletes only generated corpus,
split, tokenizer, and generated corpus-metadata artifacts; it then rebuilds
the 5 GB-per-domain corpus, verifies it, makes an assembly-held-out split,
retains compact QC/provenance records, removes temporary packages and
per-species duplicates, and starts training. If any preparation step fails,
the script exits before training begins.

```bash
cd ~/dual-helix-native/true_dna
source ../.venv/bin/activate
source scripts/export_wandb_credentials.sh
source scripts/export_research_credentials.sh
bash scripts/run_fresh_pretrain_and_ablation.sh --rtx5080
```

The first helper interactively exports `WANDB_API_KEY`; the second exports
optional `NCBI_API_KEY` and `NCBI_TAXONOMY_EMAIL`. They affect the current
shell only and do not save credentials. The launcher also invokes these
helpers in that order before it deletes or downloads data, so sourcing them
manually is optional. Weights & Biases logging is enabled by default. Add
`--a100` on Colab/A100, `--no-wandb` to disable logging, or pass runner
arguments after `--`, for example `-- --steps 1000`.

If corpus collection completed but did not reach training, do not use the
fresh launcher again. Resume from the verified domain FASTAs without deleting
or downloading anything:

```bash
cd ~/dual-helix-native/true_dna
source ../.venv/bin/activate
bash scripts/run_validated_corpus_ablation.sh --rtx5080
```

This prompts for Weights & Biases before creating the held-out split, then
runs the three base-token architecture baselines and the matched Transformer
BPE control. Add `--no-wandb` only when offline logging is intentional.

The runner records commands, package/device versions, parameter preflights,
return codes, and final held-out evaluation metrics in
`experiments/controlled_architecture_ablation_v2/seed-<seed>/ablation_manifest.json`.
It requires a split manifest by default. `--allow-implicit-split` is available
only for smoke tests and must not be used for research claims. The generated
metrics include `eval/bits_per_base` for both BPE and base tokenization.
Use `--include-tokenizer-ablation` with `run_controlled_ablation.py` when
running the control directly; it is included automatically by both full
launchers.

### Post-hoc split integrity and strict masking audit

Legacy v1 runs used RoBERTa-style 80/10/10 corruption, so their online metric
includes selected positions whose original token remains visible. Protocol-v2
training defaults to replacing every selected target with `[MASK]`; the legacy
policy is retained as `--mask-replacement-strategy bert_80_10_10` only for a
controlled comparison. Before interpreting any completed run as an accuracy
claim, scan the frozen corpus and evaluate it with every selected target masked:

```bash
python scripts/audit_ablation_integrity.py \
  --fasta data/domain_fastas/archaea.fa data/domain_fastas/bacteria.fa data/domain_fastas/eukaryotes.fa \
  --split-manifest data/domain_fastas/split_manifest.jsonl \
  --selection-manifest data/metadata/taxonomic_selection_v1.json \
  --dataset-manifest data/domain_fastas/dataset_manifest.json \
  --output-dir experiments/controlled_architecture_ablation_v2/posthoc_integrity_audit

python scripts/evaluate.py \
  --checkpoint experiments/controlled_architecture_ablation_v2/01-dual-bridge4/checkpoint_latest.pt \
  --fasta data/domain_fastas/archaea.fa data/domain_fastas/bacteria.fa data/domain_fastas/eukaryotes.fa \
  --tokenizer-type base \
  --split-manifest data/domain_fastas/split_manifest.jsonl \
  --eval-sample-manifest experiments/protocol/eval-windows-ctx1024.jsonl \
  --mask-coordinate-system base --mask-seed 43 \
  --baseline-model-json experiments/protocol/sequence-baselines.json \
  --strict-masked-targets --batch-size 8 --max-batches 32 --rc-check-seqs 0 \
  --output-json experiments/controlled_architecture_ablation_v2/posthoc_integrity_audit/01-dual-bridge4-strict.json
```

The audit streams the complete corpus, detects exact sequence or
reverse-complement duplicates across the source split, and creates a
deterministic 20% family-held-out split. The latter is for a **new training
run only**: existing checkpoints have already trained on records from those
families and must not be evaluated against that new split as a generalization
claim. Protocol-v2 evaluation first selects a deterministic raw-window sample
balanced across input domains and held-out assembly/family groups. The selected
record IDs and base coordinates are written to `--eval-sample-manifest` and
reused unchanged for every tokenizer and checkpoint. This avoids evaluating
only the first windows of each domain.

### Staged follow-up research suite

`scripts/run_research_suite.sh` schedules the follow-up work without mixing
variables. It writes a reviewed command/configuration ledger before it starts
any GPU work. Protocol v2 disables label smoothing for the primary comparison,
uses 100% `[MASK]` replacement, samples contiguous spans in raw nucleotide
coordinates, and fits uniform, training-split
unigram, and first-order Markov baselines. Every completed condition receives
strict masked-only, domain/group-stratified evaluation and reverse-complement
diagnostics. Archival checkpoints are evaluated into a strict learning curve;
the strict curve, rather than the visible-token training metric, selects the
reported checkpoint. New training runs also log PyTorch allocator values to W&B under `resource/`:
current/peak allocated and reserved MiB. Those are exact allocator peaks for
the current training invocation, rather than optional sampled system metrics.

The RTX 5080 profile uses a micro-batch of 1, gradient accumulation of 16, and
gradient checkpointing (effective batch 16). Results from a differently batched
run must not be mixed with a newly started 1x16 replicate set. It uses eight
persistent data-loader workers on the detected 24-CPU WSL configuration.
Transformer suite jobs also use
`torch.compile(mode="default")`, which retains Inductor/Triton fusion without
CUDA-Graph output reuse during the multi-forward RC objective. Mamba-containing
backbones remain in eager mode unless separately benchmarked.

Start by creating the recommended three-seed base-vs-BPE Transformer plan:

```bash
cd ~/dual-helix-native/true_dna
source ../.venv/bin/activate
bash scripts/run_research_suite.sh --rtx5080
```

Inspect `experiments/research_suite_v2/suite_manifest.json`, then add `--run`
to execute those six 5,000-step pilot runs. Runs below 20,000 optimizer steps
are explicitly labeled as fixed-budget pilots and must not be described as
full-corpus training. W&B is on by default. A short local
validation may use `--no-wandb`; it should not be used for the actual study.

```bash
bash scripts/run_research_suite.sh --rtx5080 --run
```

If WSL, the terminal, or the host restarts during a condition, completed
conditions are detected automatically. To preserve the incomplete condition's
logs/configuration in a timestamped sibling directory and restart that one
condition fresh while continuing with the remaining jobs, run:

```bash
bash scripts/run_research_suite.sh --rtx5080 --run --continue-on-error --retry-interrupted
```

`--continue-on-error` alone preserves and skips incomplete conditions; use
`--retry-interrupted` only when you intend to rerun them from step zero.

The optional stages are intentionally separate:

```bash
# Capacity-matched base-token architecture study.
bash scripts/run_research_suite.sh --rtx5080 --stages architecture

# Total-parameter-matched tokenizer sensitivity study.
bash scripts/run_research_suite.sh --rtx5080 --stages tokenizer_capacity

# One-seed corruption pilot: legacy 15% 80/10/10 versus 15/25/35% mask-only.
bash scripts/run_research_suite.sh --rtx5080 --stages masking

# Architecture × tokenizer interaction study (three backbones × base/BPE).
bash scripts/run_research_suite.sh --rtx5080 --stages factorial

# Fresh family-held-out training. The audit/split is generated first.
bash scripts/run_research_suite.sh --rtx5080 --stages family

# Context-length curve for a selected final candidate.
bash scripts/run_research_suite.sh --rtx5080 --stages context \
  --context-backbones dual_helix --contexts 512 1024
```

The masking pilot disables the curriculum so each requested rate remains fixed.
Its four conditions share the same backbone, tokenizer, seed, data, span length,
and strict evaluation sample; only replacement policy or selected-base rate changes.

Add `--run` to any planned stage. `--stages all` is available but deliberately
expensive; do not use it until the earlier results justify the next stage. The
suite collects seed-wise results in each run directory and writes means and
sample standard deviations to `experiments/research_suite_v2/aggregate_report.json`.
The suite also stores a deterministic ZIP source snapshot plus per-file hashes
under `protocol/source/`, so dirty Windows/WSL source trees cannot silently
change between conditions.
For independent downstream tasks, provide `--downstream-manifest PATH` or a
`--downstream-preset` accepted by `scripts/benchmark_genomic.py`.

### Fast protocol-v2 training smoke test

After syncing the Windows source into the native WSL checkout, run:

```bash
cd ~/dual-helix-native/true_dna
bash scripts/run_protocol_v2_smoke_training.sh
```

The launcher uses a committed four-record synthetic FASTA, a 0.13M-parameter
Transformer, 128-base windows, five optimizer steps, mask-only 15% span
corruption, two evaluation batches, and no W&B. It writes to a new timestamped
directory under `experiments/protocol_v2_smoke/`. Override only the short step
count with `SMOKE_STEPS=N` (3 through 7).

Previous experiment directories can be moved into a dated archive while
retaining JSON/config/log artifacts and deleting only checkpoint payloads:

```bash
python scripts/archive_previous_runs.py \
  --experiments-root experiments \
  --archive-name previous_runs_YYYYMMDD \
  --delete-checkpoints
```

## Evaluate and benchmark

Evaluate masked-token loss, perplexity, accuracy, and a reverse-complement
consistency diagnostic with:

```bash
python scripts/evaluate.py \
  --checkpoint experiments/example_run/checkpoint_latest.pt \
  --fasta "data/balanced/*.fa" \
  --tokenizer-type bpe
```

For a base-level checkpoint, pass `--tokenizer-type base`. Base evaluation
reports `base_accuracy` and per-base A/C/G/T accuracy. BPE evaluation reports
exact `token_accuracy`, token perplexity, and base-normalized loss,
perplexity, and bits per base. Generic `accuracy`/`acc` and `ppl` remain as
backward-compatible aliases for the mode's primary accuracy and token
perplexity.

The biology benchmark compares checkpoint representations across labeled
FASTA inputs and supports dinucleotide- or GC-matched controls:

```bash
python scripts/benchmark_biology.py \
  --checkpoint experiments/example_run/checkpoint_latest.pt \
  --fasta data/balanced/Escherichia_coli.fa \
  --label bacteria \
  --negative_control dinuc \
  --out_json experiments/example_run/biology_benchmark.json
```

For frozen-representation classification and regression tasks, install the
optional benchmark requirements and use either a built-in preset or a custom
manifest:

```bash
python -m pip install -r requirements-benchmarks.txt
python scripts/benchmark_genomic.py \
  --checkpoint experiments/example_run/checkpoint_latest.pt \
  --preset all_builtin \
  --out_json experiments/example_run/genomic_benchmarks.json
```

See `docs/benchmark_manifest.example.json` for the custom task format.
Representation probes and diagnostic controls do not establish biological
mechanism or clinical utility; use held-out data, provenance-aware splits, and
appropriate task-specific baselines.

## Development

Run the repository checks from the project root:

```bash
git diff --check
python -m compileall -q true_dna
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m pip_audit -r true_dna/requirements.txt --progress-spinner off
docker build --file docker/Dockerfile.cpu --tag true-dna:test .
docker run --rm true-dna:test
```

CUDA smoke tests and full benchmark runs are intentionally separate from the
CPU-safe continuous-integration checks. Contribution standards and artifact
handling are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## Known limitations

- Training, checkpoint evaluation, and representation benchmarks require
  Linux, CUDA, and compatible Mamba kernels.
- The repository does not include pretrained weights or a ready-made training
  corpus.
- General requirements use compatible version ranges; the RTX 5080 file
  records the tested direct pins but is not a hash-locked transitive lockfile.
- Genomic benchmarks require task-specific provenance, leakage controls, and
  scientific review beyond the included utilities.

## Support and contributions

This project is maintained on a best-effort basis without a response-time or
commercial-support commitment. Focused bug reports and pull requests that
improve correctness, reproducibility, portability, or validation are welcome.
Feature requests may be considered as maintainer time permits. See
[CONTRIBUTING.md](CONTRIBUTING.md) for scope and submission requirements, and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.

## License

This project is released under the [MIT License](LICENSE).
External dependencies remain subject to their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

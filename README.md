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

For a controlled multi-domain RefSeq corpus, use
`scripts/prepare_refseq_pretrain_data.py` with `configs/species.json`. Review
its selection and deletion flags before use; generated-data deletion is only
performed when explicitly requested.

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

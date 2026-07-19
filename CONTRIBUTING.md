# Contributing

Contributions that improve correctness, reproducibility, portability, or
scientific validation are welcome. Keep changes focused and include enough
context for another contributor to reproduce the result.

## Scope

Bug fixes, tests, documentation, portability improvements, reproducibility
work, and carefully justified scientific validation are in scope. Clinical
claims, redistribution of controlled or identifiable genomic data, unrelated
platform integrations, hosted services, and bundled model or dataset artifacts
are outside the current project scope.

## Development setup

Use Python 3.10 through 3.12 in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r true_dna/requirements.txt
python -m pip install -r true_dna/requirements-dev.txt
```

The active model path depends on CUDA-backed Mamba kernels. Most unit tests and
all static checks should remain runnable without a GPU; place CUDA-only tests
behind an explicit availability check and document the hardware and dependency
versions used for performance results.

## Required checks

Run these commands from the repository root before submitting a change:

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

When a change affects CUDA execution, also run a focused smoke test on the
supported GPU environment. Changes to the CUDA image or pinned GPU requirements
must be built and smoke-tested on that environment. Data pipeline changes
should be exercised on a small fixture before processing a full corpus.

## Engineering guidelines

- Prefer portable paths derived from the repository, module, or user-supplied
  location. Do not commit paths tied to a particular workstation.
- Validate configuration and input data at the boundary. Fail with an
  actionable message instead of silently changing scientific behavior.
- Preserve record-level separation between training, validation, and test
  data. When related genomes or derived controls can leak across splits, use a
  stricter provenance-, accession-, species-, or homology-aware split.
- Keep seeded operations deterministic where practical. Record versions,
  accessions, selection filters, and random seeds needed to reproduce a
  dataset or benchmark.
- Add a regression test for each bug fix. Architecture and objective changes
  should include shape, gradient, and edge-case coverage as appropriate.
- Avoid broad exception handling around model loading, data validation, and
  metrics. A failed check should not be reported as a successful result.
- Keep public interfaces documented and avoid retaining dead compatibility
  paths without tests or a stated migration plan.

## Scientific claims

Report measurements with the dataset split, sample count, baseline, seed, and
hardware context needed to interpret them. Clearly distinguish a diagnostic,
correlation, or probe result from evidence of biological mechanism. Do not
present research outputs as clinical or diagnostic guidance.

For sequence data, confirm that its license and terms permit the intended use.
Do not commit controlled-access, identifiable, clinical, or other sensitive
genomic data.

## Generated artifacts

Keep large or generated files out of source control, including:

- downloaded FASTA and FASTQ data;
- pre-tokenized NumPy arrays and generated indexes;
- checkpoints and experiment directories;
- W&B runs, logs, plots, and benchmark outputs;
- virtual environments, caches, and compiled files.

The checked-in BPE tokenizer is an intentional small runtime asset. If it is
retrained, include the corpus provenance, tokenizer settings, and compatibility
impact in the change description.

## Change submissions

A submission should explain the problem, the chosen approach, and the checks
that passed. Call out any checks that could not be run, any checkpoint or data
format changes, and any expected change in memory, runtime, or numerical
behavior. Keep unrelated refactoring in a separate change whenever possible.

Open changes against `main` and allow the required CI checks to complete.
Maintainers may request smaller commits, additional tests, provenance details,
or changes to scientific claims before review. Do not include secrets, private
data, generated artifacts, or dependencies copied into the repository.

This project does not currently require a contributor license agreement or DCO
sign-off. By submitting a contribution, you represent that you have the right
to provide it under the repository's MIT License.

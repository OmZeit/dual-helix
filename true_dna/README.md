# True DNA project directory

The complete setup, data preparation, training, evaluation, and benchmark
instructions are in the [project README](../README.md).

Commands in that guide are run from this directory so relative paths such as
`dna_model/bpe_vocab/tokenizer.json`, `configs/species.json`, and `data/` work
without local path changes.

Quick references:

- [Architecture and training objective](docs/architecture.md)
- [Benchmark manifest example](docs/benchmark_manifest.example.json)
- `python scripts/train.py --help`
- `python scripts/fetch_genomes.py --help`
- `python scripts/benchmark_genomic.py --help`

from __future__ import annotations

from pathlib import Path


def test_training_wrapper_exposes_benchmark_and_wandb_flags():
    script = Path(__file__).parents[1] / "scripts" / "run_training_and_benchmark.sh"
    text = script.read_text(encoding="utf-8")

    assert "--wandb)" in text
    assert "--benchmark_milestones)" in text
    assert "--benchmark_every_checkpoint)" in text
    assert "--benchmark_manifest)" in text
    assert "--biology_fasta)" in text
    assert "BENCHMARK_EVERY_CHECKPOINT=false" in text
    assert 'BENCHMARK_MANIFEST=""' in text


def test_training_wrapper_records_benchmark_history_after_successful_benchmark():
    script = Path(__file__).parents[1] / "scripts" / "run_training_and_benchmark.sh"
    text = script.read_text(encoding="utf-8")

    assert "-m dna_model.benchmark_logging" in text
    assert "--benchmark_type genomic" in text
    assert "--benchmark_type biology" in text

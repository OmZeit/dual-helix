from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from dna_model.benchmark_logging import build_eval_history_record, build_eval_log_payload
from dna_model.losses import dna_target_cross_entropy
from dna_model.sequence_baselines import MaskedBaselineAccumulator, build_baseline_payload, load_baseline_payload
from dna_model.tokenizer import BaseTokenizer
from scripts.archive_previous_runs import archive_runs
from scripts.evaluate import _build_eval_loader, _stratified_window_indices
from scripts.run_research_suite import _build_jobs


class _WindowDatasetStub:
    def __init__(self) -> None:
        self.windows = [
            {"record_index": record, "start": start, "end": start + 10}
            for record in range(3)
            for start in range(0, 100, 10)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def window_metadata(self, index: int) -> dict[str, int]:
        return self.windows[index]


def test_eval_window_sampling_balances_heldout_groups_deterministically():
    dataset = _WindowDatasetStub()
    group_by_record = {0: "assembly:a", 1: "assembly:b", 2: "assembly:c"}

    first = _stratified_window_indices(
        dataset,
        group_by_record=group_by_record,
        target=6,
        seed=43,
        domain_name="bacteria.fa",
    )
    second = _stratified_window_indices(
        dataset,
        group_by_record=group_by_record,
        target=6,
        seed=43,
        domain_name="bacteria.fa",
    )

    assert first == second
    assert {dataset.window_metadata(index)["record_index"] for index in first} == {0, 1, 2}


def test_eval_sample_manifest_rejects_changed_fasta_content(tmp_path):
    fasta = tmp_path / "domain.fa"
    fasta.write_text(">record\n" + "A" * 80 + "\n", encoding="utf-8")
    sample_manifest = tmp_path / "eval_windows.jsonl"
    tokenizer = BaseTokenizer(return_kmer_ids=False)
    loader_args = {
        "max_length": 64,
        "stride": 64,
        "min_length": 1,
        "batch_size": 1,
        "workers": 0,
        "max_batches": 1,
        "sample_manifest": sample_manifest,
        "sample_seed": 43,
    }
    _build_eval_loader([str(fasta)], tokenizer, **loader_args)
    fasta.write_text(">record\n" + "C" * 80 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fasta_sha256"):
        _build_eval_loader([str(fasta)], tokenizer, **loader_args)


def test_baseline_model_rejects_changed_fasta_content(tmp_path):
    fasta = tmp_path / "domain.fa"
    fasta.write_text(">record\nAAAA\n", encoding="utf-8")
    split = tmp_path / "split.jsonl"
    split.write_text("{}\n", encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    payload = build_baseline_payload(
        {"domain.fa": ["AAAA"]},
        alpha=1.0,
        split_manifest=split,
        fasta_paths=[fasta],
    )
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    load_baseline_payload(baseline_path, fasta_paths=[fasta], split_manifest=split)
    fasta.write_text(">record\nCCCC\n", encoding="utf-8")

    with pytest.raises(ValueError, match="FASTA content hash mismatch"):
        load_baseline_payload(baseline_path, fasta_paths=[fasta], split_manifest=split)


def test_masked_baselines_score_exact_selected_base_coordinates():
    tokenizer = BaseTokenizer(return_kmer_ids=False)
    ids = torch.tensor([[tokenizer.char_to_id[base] for base in "ACGT"]], dtype=torch.long)
    offsets = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]], dtype=torch.long)
    mask = torch.tensor([[False, True, False, True]])
    payload = {
        "global": {
            "unigram": {base: 0.25 for base in "ACGT"},
            "first_order": {previous: {base: 0.25 for base in "ACGT"} for previous in "ACGT"},
        },
        "domains": {},
    }
    accumulator = MaskedBaselineAccumulator(payload)

    accumulator.update(
        input_ids=ids,
        token_offsets=offsets,
        mask_bool=mask,
        tokenizer=tokenizer,
        domain_names="test.fa",
    )
    summary = accumulator.summary()

    assert summary["supported_masked_bases"] == 2
    assert summary["models"]["uniform"]["bits_per_base"] == pytest.approx(2.0)
    assert summary["models"]["global_unigram"]["bits_per_base"] == pytest.approx(2.0)
    assert summary["models"]["global_first_order"]["bits_per_base"] == pytest.approx(2.0)


def test_dna_label_smoothing_assigns_no_target_mass_to_control_tokens():
    logits = torch.zeros(1, 10, requires_grad=True)
    target = torch.tensor([5])
    loss = dna_target_cross_entropy(
        logits,
        target,
        label_smoothing=0.05,
        valid_target_ids=[5, 6, 7, 8, 9],
    )
    loss.backward()

    assert loss.item() == pytest.approx(math.log(10.0))
    assert logits.grad[0, 0].item() == pytest.approx(0.1)
    assert logits.grad[0, 6].item() < logits.grad[0, 0].item()


def test_protocol_v2_metrics_reach_history_and_scalar_logging():
    metrics = {
        "loss": 1.0,
        "accuracy": 0.5,
        "mask_coordinate_system": "base",
        "masked_base_fraction": 0.15,
        "mean_valid_target_probability_mass": 0.9,
        "conditional_valid_token_loss": 0.8,
        "conditional_valid_bits_per_base": 1.2,
        "baselines": {
            "supported_masked_bases": 10,
            "models": {"uniform": {"bits_per_base": 2.0}},
        },
    }

    record = build_eval_history_record(step=7, epoch=0, eval_metrics=metrics)
    payload = build_eval_log_payload(record, metrics)

    assert record["mask_coordinate_system"] == "base"
    assert record["conditional_valid_bits_per_base"] == pytest.approx(1.2)
    assert payload["eval/masked_base_fraction"] == pytest.approx(0.15)
    assert payload["eval/baselines/models/uniform/bits_per_base"] == pytest.approx(2.0)


def test_masking_pilot_builds_unique_legacy_and_mask_only_conditions(tmp_path):
    args = Namespace(
        assembly_split=tmp_path / "split.jsonl",
        profile="rtx5080_16gb",
        masking_seeds=[43],
        masking_backbone="transformer",
        masking_tokenizer="base",
        masking_steps=5_000,
        mask_replacement_strategy="mask",
        output_root=tmp_path / "experiments",
    )

    jobs = _build_jobs(args, ["masking"], tmp_path / "family.jsonl")

    assert len(jobs) == 4
    assert len({job.name for job in jobs}) == 4
    assert {(job.mask_replacement_strategy, job.mask_fraction) for job in jobs} == {
        ("bert_80_10_10", 0.15),
        ("mask", 0.15),
        ("mask", 0.25),
        ("mask", 0.35),
    }


def test_run_archive_preserves_json_and_deletes_only_checkpoints(tmp_path):
    experiments = tmp_path / "experiments"
    run = experiments / "old_run"
    run.mkdir(parents=True)
    (run / "research_result.json").write_text('{"accuracy": 0.5}\n', encoding="utf-8")
    (run / "events.out.tfevents").write_bytes(b"log")
    (run / "checkpoint_latest.pt").write_bytes(b"checkpoint")

    result = archive_runs(
        experiments,
        archive_name="previous_runs_test",
        includes=["old_run"],
        delete_checkpoints=True,
    )
    archived = experiments / "archive" / "previous_runs_test" / "old_run"

    assert result["deleted_checkpoint_count"] == 1
    assert (archived / "research_result.json").is_file()
    assert (archived / "events.out.tfevents").is_file()
    assert not list(archived.rglob("*.pt"))


def test_protocol_v2_smoke_launcher_is_small_and_offline():
    script = (Path(__file__).parents[1] / "scripts" / "run_protocol_v2_smoke_training.sh").read_text(encoding="utf-8")

    assert 'STEPS="${SMOKE_STEPS:-5}"' in script
    assert "--hidden_size 64" in script
    assert "--high_level_layers 2" in script
    assert "--max_length 128" in script
    assert "--batch_size 2" in script
    assert "--mask-replacement-strategy mask" in script
    assert "--mask_fraction 0.15" in script
    assert "--no_wandb" in script
    assert (Path(__file__).parent / "fixtures" / "smoke_protocol_v2.fa").is_file()


def test_fresh_launcher_deletes_only_an_owned_scratch_directory():
    script = (Path(__file__).parents[1] / "scripts" / "run_fresh_pretrain_and_ablation.sh").read_text(encoding="utf-8")

    assert 'SCRATCH_MARKER="$SCRATCH_DIR/.true_dna_scratch"' in script
    assert 'if [[ ! -f "$SCRATCH_MARKER" ]]' in script

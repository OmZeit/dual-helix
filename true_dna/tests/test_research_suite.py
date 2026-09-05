import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from scripts import run_research_suite


def test_research_suite_defaults_to_a_safe_plan_and_isolates_studies(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_research_suite.py", "--profile", "rtx5080_16gb"])
    args = run_research_suite.parse_args()
    text = (Path(__file__).parents[1] / "scripts" / "run_research_suite.py").read_text(encoding="utf-8")

    assert args.stages == ["tokenizer"]
    assert args.run is False
    assert 'for tokenizer_type in ("base", "bpe")' in text
    assert "capacity_matched=True" in text
    assert '"--strict-masked-targets"' in text
    assert "scheduler_cooldown" in text
    assert "batch_label" in text
    assert "compile_transformer" in text
    assert "compile_mode" in text
    assert 'default="default"' in text
    assert 'default="base"' in text
    assert "default=0.0" in text
    assert '"tokenizer_capacity"' in text
    assert '"masking"' in text
    assert '("bert_80_10_10", 0.15)' in text
    assert '("mask", 0.15)' in text
    assert '("mask", 0.25)' in text
    assert '("mask", 0.35)' in text
    assert 'default="mask"' in text
    assert "default=[43]" in text
    assert '"--eval-sample-manifest"' in text
    assert '"--baseline-model-json"' in text
    assert "source_snapshot-" in text
    assert "strict_learning_curve" in text
    assert '"--eval-workers"' in text
    assert '"status": "interrupted"' in text
    assert '"--retry-interrupted"' in text
    assert '"resource/cuda_peak_allocated_mib"' not in text  # Logged by train.py, not duplicated here.


def test_research_suite_has_explicit_family_and_downstream_stages():
    text = (Path(__file__).parents[1] / "scripts" / "run_research_suite.py").read_text(encoding="utf-8")

    assert '"family"' in text
    assert '"--family-holdout-fraction"' in text
    assert '"downstream"' in text
    assert '"--downstream-manifest"' in text
    assert '"--downstream-preset"' in text
    assert "before GPU work starts" in text


def test_capacity_matched_plan_config_uses_preflight_selection(monkeypatch, tmp_path):
    job = run_research_suite.StudyJob(
        name="architecture-transformer-base",
        study="architecture",
        backbone="transformer",
        tokenizer_type="base",
        seed=43,
        steps=10,
        max_length=128,
        split_manifest=str(tmp_path / "split.jsonl"),
        capacity_matched=True,
        comparison_scope="total_parameter_matched",
        mask_replacement_strategy="mask",
        mask_fraction=None,
        run_dir=str(tmp_path / "run"),
    )
    monkeypatch.setattr(
        run_research_suite,
        "_base_config",
        lambda _args, _job: {"hidden_size": 384, "num_attention_heads": 6},
    )

    config = run_research_suite._resolved_config(
        object(),
        job,
        {
            "transformer:base": {
                "hidden_size": 320,
                "num_attention_heads": 5,
                "parameters_m": 14.1,
            }
        },
    )

    assert config["hidden_size"] == 320
    assert config["num_attention_heads"] == 5


def test_source_snapshot_excludes_ignored_and_credential_named_files(monkeypatch, tmp_path):
    project = tmp_path / "true_dna"
    project.mkdir()
    (tmp_path / ".gitignore").write_text("ignored_credentials.json\n", encoding="utf-8")
    (project / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ignored_credentials.json").write_text('{"api_key": "ignored"}\n', encoding="utf-8")
    (project / "service_credentials.json").write_text('{"api_key": "unignored"}\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(run_research_suite, "REPOSITORY_DIR", tmp_path)
    monkeypatch.setattr(run_research_suite, "PROJECT_DIR", project)

    snapshot = run_research_suite._source_snapshot(tmp_path / "protocol")

    with zipfile.ZipFile(snapshot["archive"]) as archive:
        names = set(archive.namelist())
    assert "true_dna/model.py" in names
    assert "ignored_credentials.json" not in names
    assert "true_dna/service_credentials.json" not in names


def test_completed_result_with_a_different_protocol_fingerprint_is_refused(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "research_result.json").write_text(
        '{"status": "completed", "protocol_fingerprint": "old"}\n',
        encoding="utf-8",
    )
    job = run_research_suite.StudyJob(
        name="architecture-transformer-base",
        study="architecture",
        backbone="transformer",
        tokenizer_type="base",
        seed=43,
        steps=10,
        max_length=128,
        split_manifest=str(tmp_path / "split.jsonl"),
        capacity_matched=True,
        comparison_scope="total_parameter_matched",
        mask_replacement_strategy="mask",
        mask_fraction=None,
        run_dir=str(run_dir),
    )
    manifest = {
        "jobs": [
            {
                "name": job.name,
                "protocol_fingerprint": "new",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="protocol fingerprint"):
        run_research_suite._run_job(
            object(),
            job,
            {},
            manifest,
            tmp_path / "suite_manifest.json",
        )

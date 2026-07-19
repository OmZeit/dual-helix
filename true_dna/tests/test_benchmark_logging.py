from __future__ import annotations

import json

from dna_model.benchmark_logging import (
    append_jsonl,
    build_eval_history_record,
    build_eval_log_payload,
    flatten_numeric_metrics,
    log_benchmark_record_to_wandb,
    record_benchmark_result,
)


def test_eval_history_and_log_payload_are_stable_without_wandb(tmp_path):
    metrics = {
        "tokenizer_type": "base",
        "metric_granularity": "base",
        "loss": 1.25,
        "base_loss": 1.25,
        "ppl": 3.49,
        "acc": 0.72,
        "base_accuracy": 0.72,
        "ece": 0.04,
        "evaluated_batches": 12,
        "max_batches": 16,
        "accuracy_A": 0.8,
        "accuracy_C": 0.7,
    }

    record = build_eval_history_record(step=100, epoch=2, eval_metrics=metrics, timestamp=123.0)
    assert record["eval_loss"] == 1.25
    assert record["eval_acc"] == 0.72
    assert record["tokenizer_type"] == "base"
    assert record["evaluated_batches"] == 12
    assert record["eval_max_batches"] == 16

    payload = build_eval_log_payload(record, metrics)
    assert payload["eval/loss"] == 1.25
    assert payload["eval/base_accuracy"] == 0.72
    assert payload["eval/accuracy_A"] == 0.8
    assert payload["global_step"] == 100

    history_path = tmp_path / "eval_history.jsonl"
    append_jsonl(history_path, record)
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [record]


def test_benchmark_record_appends_history_and_enriches_checkpoint_meta(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_step100.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    meta_path = run_dir / "checkpoint_step100.pt.meta.json"
    meta_path.write_text(
        json.dumps({"global_step": 100, "metrics": {"eval_loss": 2.0}}, indent=2),
        encoding="utf-8",
    )
    result_json = run_dir / "genomic_benchmark_step100.json"
    result_json.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "benchmarks": {
                    "demo_task": {
                        "model_probe": {"accuracy": 0.91, "macro_f1": 0.88},
                        "random_model_probe": {"accuracy": 0.51},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    record = record_benchmark_result(
        run_dir=run_dir,
        checkpoint=checkpoint,
        result_json=result_json,
        label="step100",
        benchmark_type="genomic",
    )

    assert record["global_step"] == 100
    assert record["metrics"]["benchmark/genomic/step100/benchmarks/demo_task/model_probe/accuracy"] == 0.91

    history_rows = [
        json.loads(line) for line in (run_dir / "benchmark_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history_rows[0]["label"] == "step100"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["metrics"]["eval_loss"] == 2.0
    assert meta["benchmarks"][0]["label"] == "step100"
    assert "genomic/step100" in meta["metrics"]["benchmarks"]


def test_flatten_numeric_metrics_ignores_non_scalars_and_checkpoint_text():
    flat = flatten_numeric_metrics(
        {
            "checkpoint": "checkpoint_best.pt",
            "benchmarks": {
                "task": {
                    "accuracy": 0.75,
                    "labels": ["a", "b"],
                    "nested": {"count": 4, "ok": True},
                }
            },
        },
        prefix="benchmark/genomic/best",
    )

    assert flat["benchmark/genomic/best/benchmarks/task/accuracy"] == 0.75
    assert flat["benchmark/genomic/best/benchmarks/task/nested/count"] == 4
    assert all("checkpoint" not in key for key in flat)
    assert all("ok" not in key for key in flat)


class _FakeWandbTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


class _FakeWandb:
    run = object()

    def __init__(self):
        self.metric_defs = []
        self.logs = []

    def define_metric(self, *args, **kwargs):
        self.metric_defs.append((args, kwargs))

    def Table(self, *, columns):
        return _FakeWandbTable(columns)

    def log(self, payload, step=None):
        self.logs.append((payload, step))


def test_benchmark_wandb_logs_define_global_step_axis():
    fake_wandb = _FakeWandb()
    record = {
        "benchmark_type": "genomic",
        "label": "step100",
        "checkpoint_name": "checkpoint_step100.pt",
        "checkpoint": "checkpoint_step100.pt",
        "global_step": 100,
        "timestamp": 123.0,
        "metrics": {
            "benchmark/genomic/step100/benchmarks/demo_task/accuracy": 0.91,
        },
    }

    log_benchmark_record_to_wandb(record, fake_wandb)

    assert (("benchmark/global_step",), {}) in fake_wandb.metric_defs
    assert (("benchmark/*",), {"step_metric": "benchmark/global_step"}) in fake_wandb.metric_defs
    assert fake_wandb.logs[0][1] is None
    assert fake_wandb.logs[0][0]["benchmark/global_step"] == 100

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import download_data
import prepare_ablation_data as prep
import prepare_refseq_pretrain_data as refseq_prep
import pytest
import run_controlled_ablation as runner
import train as trainer
from build_split_manifest import build_manifest
from dna_model.loaders import load_split_manifest


def test_split_manifest_resolves_paths_relative_to_its_location(tmp_path):
    fasta = tmp_path / "records.fa"
    fasta.write_text(
        ">assembly=GCF_000001|contig_a\n" + "A" * 80 + "\n>assembly=GCF_000002|contig_b\n" + "C" * 80 + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "splits" / "split_manifest.jsonl"

    summary = build_manifest(
        [fasta],
        manifest,
        group_by="assembly",
        holdout_fraction=0.5,
        seed=43,
    )
    assignments = load_split_manifest(manifest, [str(fasta)])

    assert summary["groups"] == 2
    assignment = assignments[str(fasta.resolve())]
    assert assignment["train"] | assignment["eval"] == {0, 1}
    assert assignment["train"]
    assert assignment["eval"]


def test_assembly_split_assigns_train_and_eval_records_in_every_fasta(tmp_path):
    fastas = []
    for domain in ("bacteria", "eukaryotes"):
        fasta = tmp_path / f"{domain}.fa"
        fasta.write_text(
            ">assembly=GCF_000001|contig_a\n" + "A" * 80 + "\n>assembly=GCF_000002|contig_b\n" + "C" * 80 + "\n",
            encoding="utf-8",
        )
        fastas.append(fasta)
    manifest = tmp_path / "split_manifest.jsonl"

    summary = build_manifest(fastas, manifest, group_by="assembly", holdout_fraction=0.01, seed=43)
    assignments = load_split_manifest(manifest, [str(fasta) for fasta in fastas])

    assert all(summary["per_fasta"][str(fasta.resolve())]["eval_records"] >= 1 for fasta in fastas)
    assert all(assignments[str(fasta.resolve())]["train"] for fasta in fastas)
    assert all(assignments[str(fasta.resolve())]["eval"] for fasta in fastas)


def test_split_manifest_reports_base_balanced_holdout_mass(tmp_path):
    fasta = tmp_path / "records.fa"
    fasta.write_text(
        ">assembly=GCF_000001|long\n" + "A" * 1000 + "\n"
        ">assembly=GCF_000002|short_a\n" + "C" * 100 + "\n"
        ">assembly=GCF_000003|short_b\n" + "G" * 100 + "\n",
        encoding="utf-8",
    )
    summary = build_manifest(
        [fasta],
        tmp_path / "split_v2.jsonl",
        group_by="assembly",
        holdout_fraction=0.10,
        seed=43,
        balance_by="bases",
    )

    per_fasta = summary["per_fasta"][str(fasta.resolve())]
    assert summary["schema"] == "true-dna-split-v2"
    assert summary["balance_by"] == "bases"
    assert per_fasta["bases"] == 1200
    assert per_fasta["eval_bases"] > 0
    assert per_fasta["train_bases"] > 0


def test_split_assignment_is_invariant_when_the_corpus_tree_moves(tmp_path):
    assignments = []
    fasta_text = "".join(f">assembly=GCF_{index:06d}|contig\n{'ACGT' * (index + 5)}\n" for index in range(16))
    for root_name in ("first_checkout", "moved_checkout"):
        root = tmp_path / root_name
        fasta = root / "data" / "records.fa"
        fasta.parent.mkdir(parents=True)
        fasta.write_text(fasta_text, encoding="utf-8")
        manifest = root / "manifests" / "split.jsonl"
        build_manifest(
            [fasta],
            manifest,
            group_by="assembly",
            holdout_fraction=0.25,
            seed=43,
        )
        assignments.append(
            {
                row["group_id"]: row["split"]
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
                for row in [json.loads(line)]
            }
        )

    assert assignments[0] == assignments[1]


def test_ablation_runner_has_distinct_memory_profiles_and_meaningful_bridge_runs():
    rtx_profile = runner.HARDWARE_PROFILES["rtx5080_16gb"]
    assert rtx_profile["batch_size"] == 1
    assert rtx_profile["grad_accum"] == 16
    assert rtx_profile["workers"] == 8
    assert rtx_profile["gradient_checkpointing"] is True
    assert runner.HARDWARE_PROFILES["a100_80gb"]["max_length"] > runner.HARDWARE_PROFILES["rtx5080_16gb"]["max_length"]

    bridge_runs = [run for run in runner.EXPERIMENTS if run["test_variable"] in {"bridge_disabled", "bridge_interval"}]
    assert bridge_runs
    assert all(run["backbone"] == "dual_helix" for run in bridge_runs)


def test_architecture_baselines_share_the_single_base_vocabulary_and_bpe_is_a_separate_control():
    architecture_baselines = [run for run in runner.EXPERIMENTS if run["test_variable"] == "architecture_baseline"]
    tokenizer_controls = [run for run in runner.EXPERIMENTS if run["test_variable"] == "tokenizer_ablation"]

    assert len(architecture_baselines) == 3
    assert runner.COMMON_CONFIG["tokenizer_type"] == "base"
    assert all(run["overrides"].get("tokenizer_type", "base") == "base" for run in architecture_baselines)
    assert tokenizer_controls == [
        {
            "name": "06-transformer-bpe",
            "backbone": "transformer",
            "test_variable": "tokenizer_ablation",
            "comparison": "03-transformer-base-vs-06-transformer-bpe",
            "overrides": {"tokenizer_type": "bpe"},
        }
    ]


def test_parameter_preflight_does_not_require_a_corpus_or_split_manifest():
    args = Namespace(
        preflight_only=True,
        prepare_data=False,
        fasta=None,
        split_manifest="data/domain_fastas/split_manifest.jsonl",
    )

    runner.prepare_inputs(args)

    assert args.fasta == ["__parameter_preflight_no_data__"]
    assert args.split_manifest is None


def test_runner_emits_the_trainer_flag_contract():
    command = runner.config_to_command(
        {
            "tokenizer_type": "base",
            "tokenizer_path": "dna_model/bpe_vocab/tokenizer.json",
            "split_manifest": "data/split.jsonl",
            "max_length": 1024,
            "high_level_layers": 6,
            "gradient_checkpointing": True,
            "use_kmer_mix": False,
            "mask_replacement_strategy": "mask",
            "wandb_name": "parameter-check",
        },
        print_params_only=True,
    )

    assert "--tokenizer-type" in command
    assert "--split-manifest" in command
    assert "--max_length" in command
    assert "--high_level_layers" in command
    assert "--gradient_checkpointing" in command
    assert "--no-kmer-mix" in command
    assert "--mask-replacement-strategy" in command
    assert command[command.index("--mask-replacement-strategy") + 1] == "mask"
    assert "--wandb-name" in command
    assert "--print_params_only" in command
    assert "--max-length" not in command
    assert "--print-params-only" not in command


def test_ablation_seed_has_an_isolated_output_root_and_wandb_identity(tmp_path):
    args = Namespace(
        profile="rtx5080_16gb",
        fasta=["data/domain_fastas/bacteria.fa"],
        split_manifest="data/domain_fastas/split_manifest.jsonl",
        seed=44,
        steps=None,
        no_wandb=False,
    )
    seed_root = runner.seed_output_root(tmp_path, args.seed)
    config = runner.build_config(args, runner.EXPERIMENTS[0], seed_root)

    assert seed_root == tmp_path / "seed-44"
    assert Path(config["save_dir"]).parent == seed_root
    assert config["wandb_name"].endswith("-seed-44")


def test_ablation_refuses_a_nonempty_seed_output_root(tmp_path):
    seed_root = tmp_path / "seed-43"
    seed_root.mkdir()
    (seed_root / "checkpoint_latest.pt").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="nonempty seed output directory"):
        runner.ensure_empty_output_dir(seed_root)


def test_parameter_preflight_keeps_every_selected_condition_and_parses_counts():
    selected = runner.preflight_experiments(runner.EXPERIMENTS)

    assert [experiment["name"] for experiment in selected] == [experiment["name"] for experiment in runner.EXPERIMENTS]
    assert runner.parse_parameter_count_millions("12.34\n") == pytest.approx(12.34)


def test_final_run_evidence_includes_eval_parameters_and_peak_memory(tmp_path):
    (tmp_path / "eval_history.jsonl").write_text(
        json.dumps({"step": 10, "bits_per_base": 1.5}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "parameters": 1234,
                "cuda_memory": {
                    "max_memory_allocated_bytes": 1024,
                    "max_memory_reserved_bytes": 2048,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.final_run_evidence(tmp_path)

    assert evidence["final_eval"]["bits_per_base"] == pytest.approx(1.5)
    assert evidence["parameters"] == 1234
    assert evidence["cuda_memory"]["max_memory_reserved_bytes"] == 2048


def test_training_run_summary_persists_parameters_and_runtime_evidence(tmp_path):
    trainer.write_run_summary(
        tmp_path,
        global_step=17,
        total_params=1234,
        final_eval={"loss": 0.75},
        device="cpu",
    )

    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["global_step"] == 17
    assert summary["parameters"] == 1234
    assert summary["final_eval"]["loss"] == pytest.approx(0.75)
    assert summary["cuda_memory"] == {}


def test_ablation_preparation_passes_ncbi_credentials_only_via_child_environment(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("NCBI_TAXONOMY_EMAIL", raising=False)

    environment = prep.ncbi_environment("test-key", "researcher@example.org")

    assert environment["NCBI_API_KEY"] == "test-key"
    assert environment["NCBI_TAXONOMY_EMAIL"] == "researcher@example.org"


def test_refseq_preparation_only_forwards_implemented_filter_contract(tmp_path):
    args = Namespace(
        parallel_downloads=4,
        max_assemblies=20,
        overshoot_mb=2,
        min_length=1000,
        max_n_frac=0.0,
        assembly_source="RefSeq",
        exclude_plasmids=True,
        scratch_dir=tmp_path / "scratch",
    )

    command = refseq_prep.build_download_command(
        args,
        [tmp_path / "bacteria.csv"],
        storage_budget_mb_per_domain=128.0,
    )

    assert "--assembly-source" in command
    assert command[command.index("--assembly-source") + 1] == "RefSeq"
    assert command[command.index("--domain-budget-mb") + 1] == "128.0"
    assert command[command.index("--scratch-dir") + 1] == str(tmp_path / "scratch")
    assert "--exclude-plasmids" in command
    assert "--min-checkm-completeness" not in command
    assert "--near-dedup-threshold" not in command


def test_hard_delete_removes_only_managed_domain_outputs(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    domain_fastas = repo_root / "data" / "domain_fastas"
    domain_fastas.mkdir(parents=True)
    managed = domain_fastas / "bacteria.fa"
    manual = domain_fastas / "manually_supplied.fa"
    managed.write_text(">managed\nACGT\n", encoding="utf-8")
    manual.write_text(">manual\nTGCA\n", encoding="utf-8")

    monkeypatch.setattr(refseq_prep, "REPO_ROOT", repo_root)
    monkeypatch.setattr(refseq_prep, "DOMAIN_FASTAS", domain_fastas)
    monkeypatch.setattr(refseq_prep, "TOKENIZER_JSON", repo_root / "dna_model" / "bpe_vocab" / "tokenizer.json")

    refseq_prep.hard_delete_generated_data()

    assert not managed.exists()
    assert manual.exists()


def test_fresh_download_directory_requires_pipeline_ownership(tmp_path):
    domain_dir = tmp_path / "bacteria"
    domain_dir.mkdir()
    manual = domain_dir / "manual.fa"
    manual.write_text(">manual\nACGT\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not owned by the True DNA data pipeline"):
        download_data.ensure_empty_dir(domain_dir)

    assert manual.is_file()


def test_fresh_download_directory_recreates_its_ownership_marker(tmp_path):
    domain_dir = tmp_path / "bacteria"
    domain_dir.mkdir()
    (domain_dir / download_data.GENERATED_DIR_MARKER).write_text("managed\n", encoding="utf-8")
    stale = domain_dir / "stale.fa"
    stale.write_text(">stale\nACGT\n", encoding="utf-8")

    download_data.ensure_empty_dir(domain_dir)

    assert not stale.exists()
    assert (domain_dir / download_data.GENERATED_DIR_MARKER).is_file()


def test_refseq_verification_can_document_a_nonempty_underfilled_domain():
    manifest = {
        "domain_fastas": {
            "bacteria": {"mb": 100.0, "sequences": 1},
            "archaea": {"mb": 12.0, "sequences": 1},
            "eukaryotes": {"mb": 100.0, "sequences": 1},
        }
    }

    refseq_prep.verify_domain_sizes(manifest, 100.0, 0.15, {"archaea"})


def test_cuda_smoke_script_uses_the_shared_hardware_profiles():
    script = Path(__file__).parents[1] / "scripts" / "smoke_backbones.py"
    text = script.read_text(encoding="utf-8")

    assert "from run_controlled_ablation import HARDWARE_PROFILES" in text
    assert "loss.backward()" in text
    assert "peak_allocated_mib" in text
    assert "vocab_size=10" in text

from __future__ import annotations

import json

import benchmark_genomic as bg  # noqa: E402
import numpy as np
import torch


def test_tabular_manifest_records_support_csv_tsv_and_jsonl(tmp_path):
    rows = [
        {"sequence": "ACGTACGT", "label": "pos", "split": "train"},
        {"sequence": "TGCATGCA", "label": "neg", "split": "train"},
        {"sequence": "AAAACCCC", "label": "pos", "split": "test"},
        {"sequence": "GGGGTTTT", "label": "neg", "split": "test"},
    ]

    csv_path = tmp_path / "task.csv"
    csv_path.write_text(
        "sequence,label,split\n" + "\n".join(f"{row['sequence']},{row['label']},{row['split']}" for row in rows) + "\n",
        encoding="utf-8",
    )
    csv_records = bg._manifest_records({"path": str(csv_path)}, max_records=0)
    assert len(csv_records.train) == 2
    assert len(csv_records.test) == 2

    tsv_path = tmp_path / "task.tsv"
    tsv_path.write_text(
        "sequence\tlabel\tsplit\n"
        + "\n".join(f"{row['sequence']}\t{row['label']}\t{row['split']}" for row in rows)
        + "\n",
        encoding="utf-8",
    )
    tsv_records = bg._manifest_records({"path": str(tsv_path), "format": "tsv"}, max_records=0)
    assert tsv_records.train == csv_records.train
    assert tsv_records.test == csv_records.test

    jsonl_path = tmp_path / "task.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    jsonl_records = bg._manifest_records({"path": str(jsonl_path), "format": "jsonl"}, max_records=0)
    assert jsonl_records.train == csv_records.train
    assert jsonl_records.test == csv_records.test


def test_fasta_manifest_records_support_class_file_splits(tmp_path):
    bacteria_train = tmp_path / "bacteria_train.fa"
    bacteria_test = tmp_path / "bacteria_test.fa"
    archaea_train = tmp_path / "archaea_train.fa"
    archaea_test = tmp_path / "archaea_test.fa"
    bacteria_train.write_text(">b1\nACGTACGT\n", encoding="utf-8")
    bacteria_test.write_text(">b2\nACGTACGA\n", encoding="utf-8")
    archaea_train.write_text(">a1\nTTTTCCCC\n", encoding="utf-8")
    archaea_test.write_text(">a2\nGGGGAAAA\n", encoding="utf-8")

    records = bg._manifest_records(
        {
            "name": "taxonomy_classification",
            "format": "fasta",
            "files": [
                {"path": "bacteria_train.fa", "label": "bacteria", "split": "train"},
                {"path": "bacteria_test.fa", "label": "bacteria", "split": "test"},
                {"path": "archaea_train.fa", "label": "archaea", "split": "train"},
                {"path": "archaea_test.fa", "label": "archaea", "split": "test"},
            ],
        },
        max_records=0,
        base_dir=tmp_path,
    )

    assert sorted(label for _seq, label in records.train) == ["archaea", "bacteria"]
    assert sorted(label for _seq, label in records.test) == ["archaea", "bacteria"]


def test_handcrafted_features_include_length_gc_and_kmers():
    features = bg.handcrafted_features(["ACGT", "GGGG"], k=2)

    assert features.shape == (2, 18)
    assert features[0, 0] == 4
    assert features[0, 1] == 0.5
    assert features[1, 1] == 1.0
    assert np.count_nonzero(features[:, 2:]) > 0


def test_probe_metrics_cover_classification_and_regression():
    x_train = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_train = ["neg", "neg", "pos", "pos"]
    x_test = np.asarray([[0.05], [1.05]], dtype=np.float32)
    y_test = ["neg", "pos"]

    cls = bg.classification_probe(x_train, y_train, x_test, y_test, seed=7)
    assert cls["accuracy"] == 1.0
    assert cls["macro_f1"] == 1.0
    assert "auroc" in cls

    reg = bg.regression_probe(
        np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        [0.0, 1.0, 2.0],
        np.asarray([[3.0], [4.0]], dtype=np.float32),
        [3.0, 4.0],
        seed=7,
    )
    assert np.isfinite(reg["r2"])
    assert reg["pearson"] is not None


class TinyVariantTokenizer:
    def encode(self, sequence, max_length=None, return_tensors=None):
        del return_tensors
        ids = [2] + [5 + (ord(ch) % 4) for ch in sequence] + [3]
        target = max_length or len(ids)
        ids = ids[:target] + [0] * max(0, target - len(ids))
        mask = [1 if i < min(len(sequence) + 2, target) else 0 for i in range(target)]
        return {"input_ids": ids, "attention_mask": mask}


class TinyVariantModel:
    def __call__(self, input_ids, attention_mask=None, kmer_ids=None):
        del attention_mask, kmer_ids
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 16, device=input_ids.device)
        return {"logits": logits}


def test_variant_effect_reader_scores_csv_and_labels(tmp_path):
    path = tmp_path / "variants.csv"
    path.write_text(
        "ref_seq,alt_seq,label\nACGT,ACGA,benign\nTTTT,TTTA,pathogenic\n",
        encoding="utf-8",
    )

    out = bg.run_variant_effect_benchmark(
        path,
        model=TinyVariantModel(),
        tokenizer=TinyVariantTokenizer(),
        device=torch.device("cpu"),
        max_length=8,
        batch_size=2,
    )

    assert out["num_samples"] == 2
    assert out["score"] == "alt_pseudo_nll_minus_ref_pseudo_nll"
    assert "auroc" in out


def test_category_inference_covers_harder_tasks():
    assert bg._infer_category("splice_site_prediction") == "short_range_motif"
    assert bg._infer_category("chromatin_accessibility_atac") == "long_range_regulatory"
    assert bg._infer_category("gene_expression_prediction") == "long_range_expression"
    assert bg._infer_category("variant_effect_pathogenicity") == "variant_effect"

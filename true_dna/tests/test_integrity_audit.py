from __future__ import annotations

import json

import audit_ablation_integrity as audit


def test_integrity_audit_finds_cross_split_duplicate_and_writes_family_split(tmp_path):
    fasta = tmp_path / "bacteria.fa"
    fasta.write_text(
        ">assembly=GCF_A|contig_a\nACGTACGT\n>assembly=GCF_B|contig_b\nACGTACGT\n",
        encoding="utf-8",
    )
    split_manifest = tmp_path / "assembly_split.jsonl"
    split_manifest.write_text(
        "# {}\n"
        + json.dumps({"fasta": "bacteria.fa", "record_index": 0, "split": "train"})
        + "\n"
        + json.dumps({"fasta": "bacteria.fa", "record_index": 1, "split": "eval"})
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "species": [
                    {"tax_id": "1", "domain": "bacteria", "family_tax_id": "11", "family_name": "Family A"},
                    {"tax_id": "2", "domain": "bacteria", "family_tax_id": "22", "family_name": "Family B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "source_manifests": {
                    "bacteria": [
                        {"tax_id": "1", "assemblies_used": [{"accession": "GCF_A"}]},
                        {"tax_id": "2", "assemblies_used": [{"accession": "GCF_B"}]},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    report = audit.build_audit(
        fastas=[fasta],
        split_manifest=split_manifest,
        selection_manifest=selection,
        dataset_manifest=dataset,
        output_dir=tmp_path / "audit",
        family_holdout_fraction=0.5,
        seed=43,
    )

    assert report["records"] == 2
    assert report["exact_duplicates"]["cross_split_record_pairs"] == 1
    assert report["family_heldout_split"]["summary"]["bacteria"]["eval_families"] == 1
    split_rows = [
        json.loads(line)
        for line in (tmp_path / "audit" / "family_heldout_split.jsonl").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert {row["split"] for row in split_rows} == {"train", "eval"}

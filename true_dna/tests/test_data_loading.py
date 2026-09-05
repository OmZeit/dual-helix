from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from dna_model.dataset import DatasetImpl
from dna_model.loaders import (
    NpyDataset,
    RoundRobinLoader,
    SmartBatchSampler,
    _requires_raw_sequences,
    build_loaders_balanced,
)
from dna_model.tokenizer import BaseTokenizer, pretokenized_paths, tokenizer_metadata
from torch.utils.data import DataLoader, TensorDataset


class _TokenizerStub:
    def encode(self, sequence, *, max_length=None, **_kwargs):
        values = [{"A": 5, "C": 6, "G": 7, "T": 8, "N": 9}[base] for base in sequence]
        if max_length is not None:
            values = values[:max_length]
        ids = torch.tensor(values, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "kmer_ids": torch.ones_like(ids),
        }


def _write_fasta(path, sequences):
    path.write_text(
        "".join(f">record_{index}\n{sequence}\n" for index, sequence in enumerate(sequences)),
        encoding="utf-8",
    )


def test_lazy_fasta_scan_uses_consistent_four_field_offsets(tmp_path):
    fasta = tmp_path / "records.fa"
    _write_fasta(fasta, ["ACGT", "TGCA"])

    dataset = DatasetImpl(
        _TokenizerStub(),
        fasta,
        max_length=4,
        stride=4,
        min_chunk_length=1,
        lazy_load=True,
    )

    assert all(len(offset) == 4 for offset in dataset.seq_offsets)
    assert dataset[0]["input_ids"].tolist() == [5, 6, 7, 8]
    assert dataset[1]["input_ids"].tolist() == [8, 7, 6, 5]


def test_dataset_rejects_missing_fasta_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="FASTA file not found"):
        DatasetImpl(_TokenizerStub(), tmp_path / "missing.fa", max_length=4, stride=4)


def test_npy_dataset_uses_stored_token_offsets_for_selected_records(tmp_path):
    fasta = tmp_path / "records.fa"
    _write_fasta(fasta, ["AAA", "CCCC"])
    np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint16).tofile(f"{fasta}_input_ids.npy")
    np.array([11, 12, 13, 14, 15, 16, 17], dtype=np.uint16).tofile(f"{fasta}_kmer_ids.npy")
    (tmp_path / "records.fa_offsets.json").write_text(
        json.dumps([[10, 0, 3, 3], [20, 3, 7, 4]]),
        encoding="utf-8",
    )

    dataset = NpyDataset(
        str(fasta),
        max_length=8,
        stride=8,
        min_chunk_length=1,
        indices=[1],
    )

    assert dataset[0]["input_ids"].tolist() == [4, 5, 6, 7]
    assert dataset[0]["kmer_ids"].tolist() == [14, 15, 16, 17]


def test_npy_dataset_never_uses_raw_fasta_offsets(tmp_path):
    fasta = tmp_path / "records.fa"
    _write_fasta(fasta, ["AAAA"])
    np.array([1, 2], dtype=np.uint16).tofile(f"{fasta}_input_ids.npy")
    (tmp_path / "records.fa.index.json").write_text(
        json.dumps([[0, 10, 20, 4]]),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Raw FASTA .index.json offsets cannot index"):
        NpyDataset(str(fasta), max_length=8, stride=8, min_chunk_length=1)


def test_npy_dataset_selects_and_validates_base_tokenizer_cache(tmp_path):
    fasta = tmp_path / "records.fa"
    _write_fasta(fasta, ["ACGT"])
    tokenizer = BaseTokenizer()
    paths = pretokenized_paths(str(fasta), "base")
    np.array([5, 6, 7, 8], dtype=np.uint16).tofile(paths["input_ids"])
    np.array([1, 2, 3, 4], dtype=np.uint16).tofile(paths["kmer_ids"])
    (tmp_path / "records.fa_base_offsets.json").write_text(json.dumps([[0, 0, 4, 4]]), encoding="utf-8")
    (tmp_path / "records.fa_base_tokenizer.json").write_text(
        json.dumps(tokenizer_metadata(tokenizer)),
        encoding="utf-8",
    )

    dataset = NpyDataset(
        str(fasta),
        max_length=8,
        stride=8,
        min_chunk_length=1,
        tokenizer=tokenizer,
    )

    assert dataset[0]["input_ids"].tolist() == [5, 6, 7, 8]


def test_npy_dataset_rejects_base_cache_without_metadata(tmp_path):
    fasta = tmp_path / "records.fa"
    _write_fasta(fasta, ["ACGT"])
    paths = pretokenized_paths(str(fasta), "base")
    np.array([5, 6, 7, 8], dtype=np.uint16).tofile(paths["input_ids"])
    np.array([1, 2, 3, 4], dtype=np.uint16).tofile(paths["kmer_ids"])
    (tmp_path / "records.fa_base_offsets.json").write_text(json.dumps([[0, 0, 4, 4]]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing tokenizer metadata"):
        NpyDataset(
            str(fasta),
            max_length=8,
            stride=8,
            min_chunk_length=1,
            tokenizer=BaseTokenizer(),
        )


def test_offsets_force_the_raw_sequence_loader_even_when_a_token_cache_exists():
    assert _requires_raw_sequences(
        provide_rc=False,
        use_reverse_prob=0.0,
        frameshift_prob=0.0,
        return_offsets=True,
    )


def test_balanced_loader_keeps_train_and_eval_records_disjoint(tmp_path):
    fasta = tmp_path / "records.fa"
    _write_fasta(fasta, ["A" * 64, "C" * 64])

    train_loader, eval_loader = build_loaders_balanced(
        [str(fasta)],
        _TokenizerStub(),
        max_length=64,
        stride=64,
        train_split_ratio=0.99,
        batch_size=1,
        num_workers=0,
        drop_last=False,
        use_reverse_prob=0.0,
        pin_memory=False,
    )

    train_tokens = set(next(iter(train_loader))["input_ids"].flatten().tolist())
    eval_tokens = set(next(iter(eval_loader))["input_ids"].flatten().tolist())
    assert train_tokens
    assert eval_tokens
    assert train_tokens.isdisjoint(eval_tokens)


def test_round_robin_loader_exhausts_each_loader_once():
    first = DataLoader(TensorDataset(torch.tensor([1])), batch_size=1)
    second = DataLoader(TensorDataset(torch.tensor([2, 3, 4])), batch_size=1)

    loader = RoundRobinLoader([first, second])
    values = [batch[0].item() for batch in loader]

    assert values == [1, 2, 3, 4]
    assert len(loader) == 4


def test_smart_batch_sampler_applies_resume_skip_once():
    dataset = DatasetImpl(
        _TokenizerStub(),
        ["A" * 64, "C" * 64, "G" * 64],
        max_length=64,
        stride=64,
        min_chunk_length=1,
    )
    sampler = SmartBatchSampler(dataset, batch_size=1, shuffle=False)
    sampler.set_skip(1)

    assert list(sampler) == [[1], [2]]
    assert list(sampler) == [[0], [1], [2]]

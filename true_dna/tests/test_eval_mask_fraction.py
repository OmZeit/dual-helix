from __future__ import annotations

import unittest

import torch
from dna_model.benchmark_logging import build_eval_history_record, build_eval_log_payload
from dna_model.tokenizer import BaseTokenizer
from dna_model.utils import evaluate


class _UniformPredictions(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> dict[str, torch.Tensor]:
        batch, length = input_ids.shape
        return {
            "logits": torch.zeros(
                batch,
                length,
                self.vocab_size,
                dtype=torch.float,
                device=input_ids.device,
            )
        }


class _SpecialTokenDominantPredictions(torch.nn.Module):
    def __init__(self, tokenizer: BaseTokenizer) -> None:
        super().__init__()
        self.vocab_size = int(tokenizer.vocab_size)
        self.mask_token_id = int(tokenizer.mask_token_id)
        self.a_token_id = int(tokenizer.char_to_id["A"])

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> dict[str, torch.Tensor]:
        batch, length = input_ids.shape
        logits = torch.zeros(
            batch,
            length,
            self.vocab_size,
            dtype=torch.float,
            device=input_ids.device,
        )
        logits[..., self.mask_token_id] = 10.0
        logits[..., self.a_token_id] = 9.0
        return {"logits": logits}


def _base_batch(tokenizer: BaseTokenizer, base_count: int = 100) -> dict[str, torch.Tensor]:
    input_ids = torch.full(
        (1, base_count),
        tokenizer.char_to_id["A"],
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "offsets": torch.tensor(
            [[[position, position + 1] for position in range(base_count)]],
            dtype=torch.long,
        ),
    }


class EvaluationMaskFractionTests(unittest.TestCase):
    def test_strict_evaluation_defaults_to_fifteen_percent_of_base_coordinates(self) -> None:
        tokenizer = BaseTokenizer(return_kmer_ids=False)
        batch = _base_batch(tokenizer)

        metrics = evaluate(
            _UniformPredictions(tokenizer.vocab_size),
            [batch],
            torch.device("cpu"),
            False,
            epoch=0,
            tok=tokenizer,
            max_batches=0,
            strict_masked_targets=True,
            mask_coordinate_system="base",
            mask_seed=43,
        )

        self.assertEqual(metrics["masked_bases"], 15)
        self.assertAlmostEqual(metrics["masked_base_fraction"], 0.15)

    def test_reports_acgt_restricted_accuracy_when_raw_argmax_is_special(self) -> None:
        tokenizer = BaseTokenizer(return_kmer_ids=False)

        metrics = evaluate(
            _SpecialTokenDominantPredictions(tokenizer),
            [_base_batch(tokenizer)],
            torch.device("cpu"),
            False,
            epoch=0,
            tok=tokenizer,
            max_batches=0,
            strict_masked_targets=True,
            mask_coordinate_system="base",
            mask_seed=43,
        )

        self.assertEqual(metrics["base_accuracy"], 0.0)
        self.assertEqual(metrics.get("conditional_acgt_accuracy"), 1.0)
        self.assertEqual(metrics.get("conditional_acgt_correct"), 15)
        self.assertEqual(metrics.get("conditional_acgt_targets"), 15)

    def test_reports_special_token_argmax_rate(self) -> None:
        tokenizer = BaseTokenizer(return_kmer_ids=False)

        metrics = evaluate(
            _SpecialTokenDominantPredictions(tokenizer),
            [_base_batch(tokenizer)],
            torch.device("cpu"),
            False,
            epoch=0,
            tok=tokenizer,
            max_batches=0,
            strict_masked_targets=True,
            mask_coordinate_system="base",
            mask_seed=43,
        )

        self.assertEqual(metrics.get("special_token_argmax_rate"), 1.0)
        self.assertEqual(metrics.get("special_token_argmax_count"), 15)

    def test_prediction_diagnostics_reach_history_and_scalar_logging(self) -> None:
        metrics = {
            "loss": 1.0,
            "accuracy": 0.0,
            "ppl": 2.0,
            "ece": 0.5,
            "conditional_acgt_accuracy": 0.25,
            "conditional_acgt_correct": 1,
            "conditional_acgt_targets": 4,
            "special_token_argmax_rate": 0.75,
            "special_token_argmax_count": 3,
        }

        record = build_eval_history_record(step=5, epoch=0, eval_metrics=metrics, timestamp=1.0)
        payload = build_eval_log_payload(record, metrics)

        self.assertEqual(record.get("conditional_acgt_accuracy"), 0.25)
        self.assertEqual(record.get("special_token_argmax_rate"), 0.75)
        self.assertEqual(payload.get("eval/conditional_acgt_accuracy"), 0.25)
        self.assertEqual(payload.get("eval/conditional_acgt_correct"), 1)
        self.assertEqual(payload.get("eval/conditional_acgt_targets"), 4)
        self.assertEqual(payload.get("eval/special_token_argmax_rate"), 0.75)
        self.assertEqual(payload.get("eval/special_token_argmax_count"), 3)


if __name__ == "__main__":
    unittest.main()

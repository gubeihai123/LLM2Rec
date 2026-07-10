import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "llm2rec"))

from precompute_cf_distill import main as precompute_main
from precompute_cf_distill import item_titles_checksum
from precompute_cf_distill import precompute_neighbors
from recdata.dataset import TrainSample
from run_unsupervised_SimCSE import DefaultCollator, SimCSETrainer


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.8, 0.2, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.8, 0.2],
                    ]
                )
            )

    def tokenize(self, texts):
        return {"input_ids": torch.tensor([int(text) for text in texts])}

    def forward(self, features):
        return self.embedding(features["input_ids"])


class CFDistillationTests(unittest.TestCase):
    def test_precompute_excludes_self_and_normalizes_probabilities(self):
        embeddings = torch.eye(4)
        neighbor_ids, teacher_probs, teacher_scores = precompute_neighbors(
            embeddings, top_k=2, teacher_tau=0.1, chunk_size=2, device="cpu"
        )
        self.assertEqual(neighbor_ids.shape, (4, 2))
        self.assertEqual(teacher_scores.shape, (4, 2))
        self.assertTrue(torch.all(neighbor_ids != torch.arange(4).unsqueeze(1)))
        self.assertTrue(torch.allclose(teacher_probs.sum(dim=-1), torch.ones(4)))

    def test_precompute_cli_writes_reliability_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding_path = os.path.join(directory, "teacher.pt")
            titles_path = os.path.join(directory, "item_titles.txt")
            sequence_path = os.path.join(directory, "sequences.txt")
            output_path = os.path.join(directory, "distill.pt")
            torch.save(torch.eye(4), embedding_path)
            with open(titles_path, "w", encoding="utf-8") as handle:
                handle.write("0\n1\n2\n3\n")
            with open(sequence_path, "w", encoding="utf-8") as handle:
                handle.write("0 0 1 1\n1 2 3\n")

            with patch.object(
                sys,
                "argv",
                [
                    "precompute_cf_distill.py",
                    "--teacher_item_emb_path",
                    embedding_path,
                    "--item_titles_path",
                    titles_path,
                    "--sequence_path",
                    sequence_path,
                    "--output_path",
                    output_path,
                    "--top_k",
                    "2",
                    "--teacher_tau",
                    "0.1",
                    "--use_reliability",
                    "true",
                    "--support_cap",
                    "2",
                ],
            ):
                precompute_main()

            data = torch.load(output_path, map_location="cpu")
            self.assertEqual(data["neighbor_ids"].shape, (4, 2))
            self.assertTrue(torch.allclose(data["teacher_probs"].sum(dim=-1), torch.ones(4)))
            self.assertTrue(torch.all((data["reliability"] >= 0) & (data["reliability"] <= 1)))
            self.assertTrue(data["meta"]["use_reliability"])
            self.assertEqual(data["meta"]["item_titles_sha256"], item_titles_checksum(["0", "1", "2", "3"]))

    def test_default_collator_preserves_original_tuple(self):
        model = ToyStudent()
        examples = [
            TrainSample(guid="2", texts=["2", "2"], label=1.0),
            TrainSample(guid="3", texts=["3", "3"], label=1.0),
        ]
        original_batch = DefaultCollator(model)(examples)
        self.assertIsInstance(original_batch, tuple)
        self.assertEqual(len(original_batch), 2)

        kd_batch = DefaultCollator(model, include_item_ids=True)(examples)
        self.assertEqual(kd_batch["item_ids"].tolist(), [2, 3])
        self.assertEqual(len(kd_batch["features"]), 2)

    def test_kd_loss_supports_uncalibrated_and_reliability_weights(self):
        model = ToyStudent()
        trainer = SimCSETrainer.__new__(SimCSETrainer)
        trainer.model = model
        trainer.cf_neighbor_ids_cpu = torch.tensor([[1, 2], [0, 2], [3, 0], [2, 1]])
        trainer.cf_teacher_probs_cpu = torch.tensor(
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]
        )
        trainer.cf_reliability_cpu = torch.tensor([0.25, 0.5, 0.75, 1.0])
        trainer.cf_item_texts = ["0", "1", "2", "3"]
        trainer.cf_distill_student_tau = 0.1
        trainer.cf_distill_top_k = None
        trainer.cf_distill_anchor_subsample = None

        anchor_ids = torch.tensor([0, 2])
        q_reps = model({"input_ids": anchor_ids})

        trainer.cf_distill_weight_mode = "none"
        uncalibrated_loss, uncalibrated_weight = trainer._compute_cf_kd_loss(
            q_reps, anchor_ids
        )
        self.assertTrue(torch.isfinite(uncalibrated_loss))
        self.assertGreater(uncalibrated_loss.item(), 0.0)
        self.assertEqual(uncalibrated_weight.item(), 1.0)

        trainer.cf_distill_weight_mode = "reliability"
        reliability_loss, reliability_weight = trainer._compute_cf_kd_loss(q_reps, anchor_ids)
        self.assertTrue(torch.isfinite(reliability_loss))
        self.assertAlmostEqual(reliability_weight.item(), 0.5)
        reliability_loss.backward()
        self.assertIsNotNone(model.embedding.weight.grad)

    def test_trainer_skips_kd_without_a_distillation_file(self):
        model = ToyStudent()
        trainer = SimCSETrainer.__new__(SimCSETrainer)
        trainer.model = model
        trainer.loss_function = lambda query, document, negative: ((query - document) ** 2).mean()
        trainer.cf_distill_enabled = False
        features = [model.tokenize(["0", "2"]), model.tokenize(["0", "2"])]
        loss = trainer.compute_loss(model, (features, torch.ones(2)))
        self.assertEqual(loss.item(), 0.0)

    def test_trainer_adds_and_logs_kd_loss(self):
        model = ToyStudent()
        trainer = SimCSETrainer.__new__(SimCSETrainer)
        trainer.model = model
        trainer.loss_function = lambda query, document, negative: ((query - document) ** 2).mean()
        trainer.cf_distill_enabled = True
        trainer.cf_distill_lambda = 0.05
        trainer.cf_neighbor_ids_cpu = torch.tensor([[1, 2], [0, 2], [3, 0], [2, 1]])
        trainer.cf_teacher_probs_cpu = torch.tensor(
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]]
        )
        trainer.cf_reliability_cpu = torch.tensor([0.25, 0.5, 0.75, 1.0])
        trainer.cf_item_texts = ["0", "1", "2", "3"]
        trainer.cf_distill_student_tau = 0.1
        trainer.cf_distill_top_k = None
        trainer.cf_distill_anchor_subsample = None
        trainer.cf_distill_weight_mode = "reliability"
        logged = {}
        trainer.log = logged.update
        features = [model.tokenize(["0", "2"]), model.tokenize(["0", "2"])]
        loss = trainer.compute_loss(
            model,
            {"features": features, "labels": torch.ones(2), "item_ids": torch.tensor([0, 2])},
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(
            set(logged),
            {"simcse_loss", "cf_kd_loss", "total_loss", "cf_weight_mean", "cf_distill_lambda"},
        )


if __name__ == "__main__":
    unittest.main()

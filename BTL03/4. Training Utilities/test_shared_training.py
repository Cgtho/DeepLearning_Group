"""Focused tests for the reusable BTL03 PyTorch training helpers."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shared_training import fit, run_epoch, seed_everything


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, *args):
        self.scalars.append(args)


class SharedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_seed_reproduces_torch_values(self):
        seed_everything(11)
        first = torch.rand(3)
        seed_everything(11)
        torch.testing.assert_close(first, torch.rand(3))

    def test_epoch_uses_sample_weighted_loss_and_evaluation_has_no_gradients(self):
        model = nn.Linear(2, 2)
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
        expected = nn.functional.cross_entropy(model(images), labels).item()

        self.assertAlmostEqual(metrics["loss"], expected, places=6)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(model.training)

    def test_fit_saves_and_restores_best_weights_and_stops_early(self):
        model = nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_loader = DataLoader(
            TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)),
            batch_size=2,
        )
        valid_losses = iter([1.0, 1.2, 1.3])
        train_epoch = 0
        writer = Writer()

        def fake_epoch(model, loader, criterion, device, optimizer=None):
            nonlocal train_epoch
            if optimizer is not None:
                train_epoch += 1
                with torch.no_grad():
                    model.weight.fill_(train_epoch)
                return {"loss": 0.5, "accuracy": 0.5}
            return {"loss": next(valid_losses), "accuracy": 0.75}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pth"
            with patch("shared_training.run_epoch", side_effect=fake_epoch):
                result = fit(
                    model,
                    train_loader,
                    train_loader,
                    optimizer,
                    nn.CrossEntropyLoss(),
                    "cpu",
                    epochs=8,
                    checkpoint_path=checkpoint,
                    patience=2,
                    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=0, factor=0.5
                    ),
                    writer=writer,
                )

            self.assertEqual(result["epochs_completed"], 3)
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(model.weight.item(), 1.0)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(len(writer.scalars), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)

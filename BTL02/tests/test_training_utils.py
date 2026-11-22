"""Offline regression checks. Synthetic tensors/CSVs are not FER2013 results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fer_data import training_loaders
from models import build_model
from training_utils import check_batch, fit, run_epoch


class Writer:
    def add_scalar(self, *args):
        pass


class TrainingUtilitiesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_models_forward_backward_and_reload(self):
        dataset = TensorDataset(torch.rand(2, 1, 48, 48), torch.tensor([0, 6]))
        for name in ("baseline", "cnn_mlp"):
            with self.subTest(model=name):
                result = check_batch(dataset, 2, torch.device("cpu"), model_name=name)
                self.assertEqual(result["output"], [2, 7])
                self.assertTrue(result["finite_gradients"])
                model = build_model(name).eval()
                other = build_model(name).eval()
                other.load_state_dict(model.state_dict())
                image = dataset.tensors[0]
                with torch.no_grad():
                    torch.testing.assert_close(model(image), other(image))

    def test_epoch_loss_is_weighted_and_validation_has_no_gradients(self):
        model = nn.Linear(2, 2)
        images = torch.tensor([[1., 0.], [0., 1.], [2., 1.]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        loss, accuracy = run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
        expected = nn.functional.cross_entropy(model(images), labels).item()
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertEqual(accuracy, (model(images).argmax(1) == labels).float().mean().item())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.training)

    def test_early_stop_lr_and_restore_best_weights(self):
        model = nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        values = iter([1., 1.2, 1.3])
        count = 0

        def epoch(model, loader, criterion, device, optimizer=None):
            nonlocal count
            if optimizer is not None:
                count += 1
                with torch.no_grad():
                    model.weight.fill_(count)
                return [1.0, 0.5]
            return [next(values), 0.5]

        with tempfile.TemporaryDirectory() as tmp, patch("training_utils.core.run_epoch", epoch):
            run = Path(tmp)
            summary = fit(model, None, None, optimizer, None, "cpu", 10, run, Writer(),
                          {"enabled": True, "patience": 2},
                          {"enabled": True, "patience": 0, "factor": 0.5})
            self.assertEqual(summary["epochs_completed"], 3)
            self.assertTrue(summary["stopped_early"])
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(model.weight.item(), 1.)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)
            history = pd.read_csv(run / "history.csv")
            np.testing.assert_allclose(history.learning_rate, [0.1, 0.1, 0.05])

    def test_nonfinite_validation_fails_without_checkpoint(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmp, patch("training_utils.core.run_epoch", return_value=[float("nan"), 0.]):
            with self.assertRaises(FloatingPointError):
                fit(model, None, None, optimizer, None, "cpu", 1, Path(tmp), Writer())
            self.assertFalse((Path(tmp) / "best.pth").exists())

    def test_augmentation_only_training_and_split_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            pixels = " ".join(str(i % 256) for i in range(2304))
            pd.DataFrame({"pixels": [pixels] * 70, "emotion": np.repeat(np.arange(7), 10)}).to_csv(data / "train.csv", index=False)
            train, valid = training_loaders(data, 8, 42, {"enabled": True})
            plain_train, plain_valid = training_loaders(data, 8, 42)
            self.assertIsNotNone(train.dataset.transform)
            self.assertIsNone(valid.dataset.transform)
            torch.testing.assert_close(valid.dataset.images, plain_valid.dataset.images)
            torch.testing.assert_close(train.dataset.labels, plain_train.dataset.labels)
            images, labels = next(iter(train))
            self.assertEqual(tuple(images.shape[1:]), (1, 48, 48))
            self.assertTrue(0 <= images.min() <= images.max() <= 1)

    def test_end_to_end_synthetic_train_evaluate(self):
        import train as training
        from evaluate import evaluate_checkpoint
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            (sandbox / "data").mkdir()
            rng = np.random.default_rng(42)
            frame = pd.DataFrame({
                "pixels": [" ".join(map(str, row)) for row in rng.integers(0, 256, (35, 2304))],
                "emotion": np.repeat(np.arange(7), 5),
            })
            frame.to_csv(sandbox / "data/train.csv", index=False)
            frame.iloc[::5].to_csv(sandbox / "data/test.csv", index=False)
            for name in ("baseline", "cnn_mlp"):
                config = yaml.safe_load((root / "configs" / ("baseline.yaml" if name == "baseline" else "cnn_mlp.yaml")).read_text(encoding="utf-8"))
                config.update(device="cpu", batch_size=7)
                path = sandbox / "config.yaml"
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with patch.object(training, "ROOT", sandbox):
                    run = training.run_experiment(path, smoke=True)
                summary = json.loads((run / "training.json").read_text())
                self.assertEqual(summary["epochs_completed"], 1)
                self.assertTrue(summary["smoke"])
                evaluation = evaluate_checkpoint(run, sandbox / "data", 7, torch.device("cpu"))
                self.assertEqual(evaluation["samples"], 7)
                self.assertEqual(len(evaluation["confusion_matrix"]), 7)
                self.assertTrue((run / "last_conv_filters.png").exists())
                self.assertTrue(any((sandbox / "runs").rglob("events.out.tfevents.*")))


if __name__ == "__main__":
    unittest.main(verbosity=2)

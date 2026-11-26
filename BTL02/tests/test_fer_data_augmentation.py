import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fer_data import training_loaders


class TrainingAugmentationTests(unittest.TestCase):
    def test_augmentation_changes_training_images_but_not_validation(self):
        image = np.arange(48 * 48, dtype=np.uint8).reshape(48, 48)
        frame = pd.DataFrame(
            {
                "emotion": [emotion for emotion in range(7) for _ in range(5)],
                "pixels": [" ".join(map(str, image.ravel())) for _ in range(35)],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            frame.to_csv(data_dir / "train.csv", index=False)

            plain_train, _ = training_loaders(data_dir, batch_size=4, seed=42)
            augmented_train, validation = training_loaders(
                data_dir,
                batch_size=4,
                seed=42,
                augmentation={"horizontal_flip_probability": 1.0},
            )

            original, _ = plain_train.dataset[0]
            flipped, _ = augmented_train.dataset[0]
            torch.testing.assert_close(flipped, torch.flip(original, dims=(-1,)))
            self.assertIsNone(validation.dataset.transform)


if __name__ == "__main__":
    unittest.main()

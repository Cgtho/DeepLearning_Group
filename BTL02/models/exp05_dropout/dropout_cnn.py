from torch import nn


class BaselineDropoutCNN(nn.Module):
    """Baseline CNN with dropout in the hidden fully connected layer."""

    def __init__(self, dropout_probability=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(128, 7),
        )

    def forward(self, images):
        return self.classifier(self.features(images))

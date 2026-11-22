from torch import nn


class CNNMLP(nn.Module):
    """Adapt the MNIST CNN+MLP example to FER2013: N x 1 x 48 x 48 -> N x 7 logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 6 * 6, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 7),
        )

    def forward(self, images):
        return self.classifier(self.features(images))

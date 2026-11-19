from torch import nn


class BaselineCNN(nn.Module):
    """CNN cơ sở cho ảnh FER2013 đã chuẩn hóa, kích thước (N, 1, 48, 48)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(4608, 128), nn.ReLU(), nn.Linear(128, 7),
        )

    def forward(self, images):
        return self.classifier(self.features(images))

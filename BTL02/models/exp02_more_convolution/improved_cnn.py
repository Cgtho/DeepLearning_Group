from torch import nn


class Exp02CNN(nn.Module):
    """Dựng CNN cho ảnh 48 × 48 từ các khối (số kênh ra, số lớp Conv).

    Mỗi Conv 3 × 3 đi kèm ReLU. Mỗi khối kết thúc bằng MaxPool 2 × 2.
    """

    def __init__(self, stages):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels, conv_count in stages:
            for _ in range(conv_count):
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(),
                ])
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        spatial_size = 48 // (2 ** len(stages))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * spatial_size ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class DeeperCNN(Exp02CNN):
    """Bốn lớp tích chập, hai lớp ở khối 128 kênh."""

    def __init__(self):
        super().__init__(((32, 1), (64, 1), (128, 2)))


class ImprovedCNNV2(Exp02CNN):
    """Sáu lớp tích chập qua ba khối 32, 64 và 128 kênh."""

    def __init__(self):
        super().__init__(((32, 2), (64, 2), (128, 2)))


class ImprovedCNNV3(Exp02CNN):
    """Tám lớp tích chập qua bốn khối 32, 64, 128 và 256 kênh."""

    def __init__(self):
        super().__init__(((32, 2), (64, 2), (128, 2), (256, 2)))

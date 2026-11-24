from torch import nn


class Exp03BatchNormCNN(nn.Module):
    """Kiến trúc CNN cho ảnh 48x48 tích hợp BatchNorm2d sau Conv và BatchNorm1d ở classifier."""

    def __init__(self, stages, use_classifier_bn=True):
        super().__init__()
        feature_layers = []
        in_channels = 1

        for out_channels, num_convs in stages:
            for _ in range(num_convs):
                feature_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ])
                in_channels = out_channels
            feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*feature_layers)

        spatial_dim = 48 // (2 ** len(stages))
        flattened_dim = in_channels * (spatial_dim ** 2)

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(flattened_dim, 128, bias=not use_classifier_bn),
        ]
        if use_classifier_bn:
            classifier_layers.append(nn.BatchNorm1d(128))
        classifier_layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        ])
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images):
        return self.classifier(self.features(images))


class BaselineBN(Exp03BatchNormCNN):
    """3 lớp Conv (32 -> 64 -> 128) + Batch Normalization."""

    def __init__(self):
        super().__init__(((32, 1), (64, 1), (128, 1)))


class DeeperCNNBN(Exp03BatchNormCNN):
    """4 lớp Conv (32 -> 64 -> 128 -> 128) + Batch Normalization."""

    def __init__(self):
        super().__init__(((32, 1), (64, 1), (128, 2)))


class ImprovedCNNV2BN(Exp03BatchNormCNN):
    """6 lớp Conv (32x2 -> 64x2 -> 128x2) + Batch Normalization."""

    def __init__(self):
        super().__init__(((32, 2), (64, 2), (128, 2)))


class ImprovedCNNV3BN(Exp03BatchNormCNN):
    """8 lớp Conv (32x2 -> 64x2 -> 128x2 -> 256x2) + Batch Normalization."""

    def __init__(self):
        super().__init__(((32, 2), (64, 2), (128, 2), (256, 2)))
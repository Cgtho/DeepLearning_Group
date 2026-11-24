from models.baseline_cnn import BaselineCNN
from models.exp02_more_convolution.improved_cnn import (
    DeeperCNN,
    ImprovedCNNV2,
    ImprovedCNNV3,
)

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "deeper_cnn": DeeperCNN,
    "improved_cnn_v2": ImprovedCNNV2,
    "improved_cnn_v3": ImprovedCNNV3,
}


def build_model(name="baseline"):
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model '{name}' không hợp lệ. Chọn một trong: {choices}.") from exc
    return model_class()

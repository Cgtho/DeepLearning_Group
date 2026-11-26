from models.baseline_cnn import BaselineCNN
from models.exp02_more_convolution.improved_cnn import (
    DeeperCNN,
    ImprovedCNNV2,
    ImprovedCNNV3,
)
from models.exp03_batch_normalization.batch_norm_cnn import (
    BaselineBN,
    DeeperCNNBN,
    ImprovedCNNV2BN,
    ImprovedCNNV3BN,
)
from models.exp05_dropout.dropout_cnn import BaselineDropoutCNN

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "deeper_cnn": DeeperCNN,
    "improved_cnn_v2": ImprovedCNNV2,
    "improved_cnn_v3": ImprovedCNNV3,
    "baseline_bn": BaselineBN,
    "deeper_cnn_bn": DeeperCNNBN,
    "improved_cnn_v2_bn": ImprovedCNNV2BN,
    "improved_cnn_v3_bn": ImprovedCNNV3BN,
    "baseline_dropout": BaselineDropoutCNN,
}


def build_model(name="baseline"):
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model '{name}' không tồn tại. Các lựa chọn hợp lệ: {valid}.") from exc
    return model_class()

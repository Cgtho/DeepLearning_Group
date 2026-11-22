from models.baseline_cnn import BaselineCNN
from models.cnn_mlp import CNNMLP

MODEL_REGISTRY = {"baseline": BaselineCNN, "cnn_mlp": CNNMLP}


def build_model(name="baseline"):
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model '{name}' không hợp lệ. Chọn một trong: {choices}.") from exc
    return model_class()

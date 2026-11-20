from models.baseline_cnn import BaselineCNN

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
}


def build_model(name="baseline"):
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Model '{name}' không hợp lệ. Chọn một trong: {choices}.") from exc
    return model_class()

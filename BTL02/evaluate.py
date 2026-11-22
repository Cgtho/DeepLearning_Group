import argparse
import json
import copy
from pathlib import Path

import yaml

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from fer_data import EMOTIONS, test_loader
from models import build_model
from training_utils import select_device
from visualize import plot_results


def evaluate_checkpoint(run, data_dir, batch_size, device, model_template=None):
    if model_template is None:
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        model = build_model(config.get("model", "baseline"))
    elif isinstance(model_template, torch.nn.Module):
        model = copy.deepcopy(model_template)
    elif callable(model_template):
        model = model_template()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model_template phải trả về một torch.nn.Module.")
    else:
        raise TypeError("model_template phải là torch.nn.Module hoặc hàm tạo model.")
    model = model.to(device)
    model.load_state_dict(torch.load(run / "best.pth", map_location=device, weights_only=True))
    model.eval()
    loader = test_loader(data_dir, batch_size)
    truth, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            truth.extend(labels.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    ids = list(range(len(EMOTIONS)))
    report = classification_report(truth, predicted, labels=ids, target_names=EMOTIONS,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=ids)
    result = {
        "split": "test.csv", "samples": len(truth), "class_order": EMOTIONS,
        "accuracy": float(np.mean(truth == predicted)), "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {name: report[name] for name in EMOTIONS},
        "confusion_matrix": matrix.tolist(),
    }
    plot_results(run, model, loader.dataset, truth, predicted, result)
    (run / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Test accuracy {result['accuracy']:.2%} | Macro F1 {result['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint đã chọn trên test.csv")
    parser.add_argument("run", type=Path, help="Thư mục kết quả chứa best.pth và config.yaml")
    args = parser.parse_args()
    run = args.run.resolve()
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    data_dir = root / config.get("data_dir", "data")
    evaluate_checkpoint(
        run,
        data_dir,
        config["batch_size"],
        select_device(config.get("device", "auto")),
    )

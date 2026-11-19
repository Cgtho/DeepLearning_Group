import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from fer_data import EMOTIONS, test_loader
from models.baseline_cnn import BaselineCNN
from visualize import plot_results


def evaluate_checkpoint(run, data_dir, batch_size, device):
    model = BaselineCNN().to(device)
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

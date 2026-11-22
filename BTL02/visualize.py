import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from fer_data import EMOTIONS


def save_figure(fig, run, name):
    fig.tight_layout()
    fig.savefig(run / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results(run, model, dataset, truth, predicted, result):
    history = pd.read_csv(run / "history.csv")
    best_epoch = history.loc[history.val_loss.idxmin(), "epoch"]
    for metric, title in [("loss", "Cross-entropy"), ("accuracy", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for split, name in [("train", "Training"), ("val", "Validation")]:
            ax.plot(history.epoch, history[f"{split}_{metric}"], label=name)
        ax.axvline(best_epoch, color="gray", ls=":", label=f"Checkpoint · epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=title)
        ax.legend()
        save_figure(fig, run, f"{metric}.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=EMOTIONS).plot(
        ax=ax, values_format="d", cmap="Blues", colorbar=False, xticks_rotation=35
    )
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế", title="Ma trận nhầm lẫn trên tập test")
    save_figure(fig, run, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.DataFrame(result["per_class"]).T[["precision", "recall", "f1-score"]].plot.bar(ax=ax, rot=0)
    ax.set(ylim=(0, 1), ylabel="Điểm", title="Kết quả theo biểu cảm trên tập test")
    save_figure(fig, run, "per_class.png")

    # Lấy mẫu tái lập trực tiếp theo chỉ số trong test.csv.
    rng = np.random.default_rng(42)
    for mask, title, filename in [(truth == predicted, "Dự đoán đúng", "correct.png"),
                                  (truth != predicted, "Dự đoán sai", "incorrect.png")]:
        indices = np.flatnonzero(mask)
        selected = rng.choice(indices, size=min(8, len(indices)), replace=False)
        fig, axes = plt.subplots(2, 4, figsize=(9, 5))
        fig.suptitle(title if len(selected) else f"{title} · 0 ảnh")
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, selected):
            image, _ = dataset[int(i)]
            ax.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Thật: {EMOTIONS[truth[i]]}\nĐoán: {EMOTIONS[predicted[i]]}", fontsize=9)
        save_figure(fig, run, filename)

    # First and last convolution, input-channel 0 slices; no channel averaging.
    from torch import nn
    convolutions = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
    for layer, filename, title in [
        (convolutions[0], "conv1_filters.png", "First convolution"),
        (convolutions[-1], "last_conv_filters.png", "Last convolution"),
    ]:
        weights = layer.weight.detach().cpu().numpy()[:, 0]
        limit = max(float(np.abs(weights).max()), 1e-12)
        columns = 8
        rows = (len(weights) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(10, rows * 1.1), squeeze=False)
        fig.suptitle(f"{title}: {len(weights)} kernels, input channel 0")
        for ax in axes.flat:
            ax.axis("off")
        for i, (ax, kernel) in enumerate(zip(axes.flat, weights)):
            ax.imshow(kernel, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(str(i + 1), fontsize=8)
        save_figure(fig, run, filename)

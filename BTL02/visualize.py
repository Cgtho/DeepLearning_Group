import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from fer_data import EMOTIONS


def display_saved_figures(run, width=680):
    from IPython.display import Image, display

    for path in sorted(run.glob("*.png")):
        if path.is_file():
            display(Image(filename=str(path), width=width))


def save_figure(fig, run, name):
    fig.tight_layout()
    fig.savefig(run / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results(run, model, dataset, truth, predicted, result, split="test"):
    history = pd.read_csv(run / "history.csv")
    best_epoch = history.loc[history.val_loss.idxmin(), "epoch"]
    for metric, title in [("loss", "Cross-entropy"), ("accuracy", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for phase, name in [("train", "Training"), ("val", "Validation")]:
            ax.plot(history.epoch, history[f"{phase}_{metric}"], label=name)
        ax.axvline(best_epoch, color="gray", ls=":", label=f"Checkpoint · epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=title)
        ax.legend()
        save_figure(fig, run, f"{metric}.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=EMOTIONS).plot(
        ax=ax, values_format="d", cmap="Blues", colorbar=False, xticks_rotation=35
    )
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế",
           title=f"Ma trận nhầm lẫn trên tập {split}")
    save_figure(fig, run, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.DataFrame(result["per_class"]).T[["precision", "recall", "f1-score"]].plot.bar(ax=ax, rot=0)
    ax.set(ylim=(0, 1), ylabel="Điểm", title=f"Kết quả theo biểu cảm trên tập {split}")
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

    # 32 kernel của lớp đầu chỉ có một kênh, dùng chung thang màu đối xứng.
    weights = model.features[0].weight.detach().cpu().numpy()[:, 0]
    limit = np.abs(weights).max()
    fig, axes = plt.subplots(4, 8, figsize=(9, 5))
    fig.suptitle("32 bộ lọc 3 × 3 của lớp tích chập đầu tiên")
    for i, (ax, kernel) in enumerate(zip(axes.flat, weights)):
        ax.imshow(kernel, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(str(i + 1), fontsize=8)
        ax.axis("off")
    save_figure(fig, run, "conv1_filters.png")

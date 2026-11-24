import argparse
import csv
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch import nn

from fer_data import training_loaders
from models import build_model

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/baseline.yaml"


def select_device(name):
    # Trên Windows, cuDNN của Conda cần cùng đường dẫn DLL với kernel Python.
    dll_dir = Path(sys.prefix) / "Library/bin"
    if os.name == "nt" and (dll_dir / "cudnn64_9.dll").exists():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def resolve_model(model=None, model_name="baseline"):
    if model is None:
        return build_model(model_name)
    if isinstance(model, nn.Module):
        return copy.deepcopy(model)
    if callable(model):
        resolved = model()
        if not isinstance(resolved, nn.Module):
            raise TypeError("model_factory phải trả về một torch.nn.Module.")
        return resolved
    raise TypeError("model phải là torch.nn.Module hoặc hàm tạo model.")


def check_batch(dataset, batch_size, device, model=None, model_name="baseline"):
    # Một mô hình riêng để phép thử cập nhật trọng số không ảnh hưởng thí nghiệm.
    images, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size)))
    assert images.shape[1:] == (1, 48, 48) and images.dtype == torch.float32
    assert 0 <= images.min() and images.max() <= 1
    assert labels.dtype == torch.long and labels.min() >= 0 and labels.max() <= 6
    model = resolve_model(model, model_name).to(device)
    logits = model(images.to(device))
    assert logits.shape == (len(labels), 7)
    loss = nn.functional.cross_entropy(logits, labels.to(device))
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    return {"input": list(images.shape), "output": list(logits.shape),
            "loss": loss.item(), "finite_gradients": True}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(2, device=device)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            totals[0] += loss.detach() * labels.size(0)
            totals[1] += (logits.argmax(1) == labels).sum()
    return (totals / len(loader.dataset)).tolist()


def run_experiment(config_path=DEFAULT_CONFIG, epochs=None, smoke=False, model=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if epochs is not None:
        cfg["epochs"] = epochs
    if smoke:
        cfg["epochs"] = 1
    if min(cfg["epochs"], cfg["batch_size"], cfg["learning_rate"]) <= 0:
        raise ValueError("Epochs, batch size và learning rate phải lớn hơn 0.")
    device = select_device(cfg["device"])
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data_dir = ROOT / cfg["data_dir"]
    train, valid = training_loaders(data_dir, cfg["batch_size"], seed)
    model_name = cfg.get("model", "baseline")
    preflight = check_batch(train.dataset, cfg["batch_size"], device, model, model_name)
    torch.manual_seed(seed)
    model = resolve_model(model, model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = stamp + ("_smoke" if smoke else "")
    result_dir = ROOT / cfg["output_dir"]
    if result_group := cfg.get("result_group"):
        result_dir = result_dir / result_group / model_name
    else:
        result_dir = result_dir / cfg["experiment"]
    run = result_dir / run_name
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu TensorBoard. Cài bằng: pip install tensorboard"
        ) from exc

    tensorboard_run = ROOT / cfg.get("tensorboard_dir", "runs") / cfg["experiment"] / run_name
    tensorboard_run.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_run))
    hparams = {
        "experiment": cfg["experiment"],
        "batch_size": cfg["batch_size"],
        "learning_rate": cfg["learning_rate"],
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "model": model_name,
    }
    writer.add_text(
        "experiment/config",
        yaml.safe_dump(cfg, sort_keys=False).replace("\n", "  \n"),
    )
    writer.add_graph(model, torch.zeros(1, 1, 48, 48, device=device))

    print(run, device, preflight, flush=True)
    best_loss = float("inf")
    started = perf_counter()
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, train_acc = run_epoch(model, train, criterion, device, optimizer)
            with torch.no_grad():
                val_loss, val_acc = run_epoch(model, valid, criterion, device)
            csv_writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            file.flush()

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Accuracy/validation", val_acc, epoch)

            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save(model.state_dict(), run / "best.tmp")
                (run / "best.tmp").replace(run / "best.pth")
            print(f"{epoch:02d}/{cfg['epochs']}  train {train_loss:.4f} {train_acc:.2%}"
                  f"  val {val_loss:.4f} {val_acc:.2%}", flush=True)
    summary = {
        "training_seconds": perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters()),
        "model": model_name,
        "train_size": len(train.dataset), "val_size": len(valid.dataset), "best_epoch": best_epoch,
        "best_val_loss": best_loss, "device": str(device), "torch": str(torch.__version__),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "preflight": preflight, "smoke": smoke,
    }
    (run / "training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics = {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "parameters": summary["parameters"],
    }
    writer.add_scalar("Loss/best_validation", best_loss, best_epoch)
    writer.add_hparams(hparams, metrics)
    writer.close()
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện CNN trên FER2013")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="Chạy một epoch, chỉ dùng train/validation")
    args = parser.parse_args()
    run_experiment(args.config, args.epochs, args.smoke)

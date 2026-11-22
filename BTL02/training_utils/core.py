import copy
import csv
import math
import os
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from models import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    if len(loader.dataset) == 0:
        raise ValueError("DataLoader rỗng.")
    return (totals / len(loader.dataset)).tolist()


def fit(model, train, valid, optimizer, criterion, device, epochs, run, writer,
        early_stopping=None, reduce_lr=None):
    """Select the best weights by raw val_loss; early stopping uses min_delta.

    Checkpoint format remains a plain state_dict for existing evaluate.py callers.
    Caller owns and closes the TensorBoard writer.
    """
    early = early_stopping or {}
    lr_cfg = reduce_lr or {}
    enabled = early.get("enabled", False)
    patience = early.get("patience", 5)
    min_delta = early.get("min_delta", 0.0)
    if patience < 1 or min_delta < 0:
        raise ValueError("Early stopping cần patience >= 1 và min_delta >= 0.")
    if epochs < 1:
        raise ValueError("epochs phải >= 1.")
    scheduler = None
    if lr_cfg.get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_cfg.get("factor", 0.1),
            patience=lr_cfg.get("patience", 2), min_lr=lr_cfg.get("min_lr", 1e-6),
        )
    best_loss = stop_best = float("inf")
    best_epoch = bad_epochs = 0
    stopped_early = False
    started = perf_counter()
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as file:
        history = csv.writer(file)
        history.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"])
        for epoch in range(1, epochs + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(model, train, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, valid, criterion, device)
            if not all(math.isfinite(v) for v in (train_loss, train_acc, val_loss, val_acc)):
                raise FloatingPointError(f"Loss/accuracy không hữu hạn tại epoch {epoch}.")
            history.writerow([epoch, train_loss, train_acc, val_loss, val_acc, learning_rate])
            file.flush()
            for tag, value in [("Loss/train", train_loss), ("Accuracy/train", train_acc),
                               ("Loss/validation", val_loss), ("Accuracy/validation", val_acc),
                               ("LearningRate", learning_rate)]:
                writer.add_scalar(tag, value, epoch)
            if val_loss < best_loss:
                best_loss, best_epoch, best_acc = val_loss, epoch, val_acc
                torch.save(model.state_dict(), run / "best.tmp")
                (run / "best.tmp").replace(run / "best.pth")
            if val_loss < stop_best - min_delta:
                stop_best, bad_epochs = val_loss, 0
            else:
                bad_epochs += 1
            if scheduler is not None:
                scheduler.step(val_loss)
            print(f"{epoch:02d}/{epochs} train {train_loss:.4f} {train_acc:.2%} "
                  f"val {val_loss:.4f} {val_acc:.2%} lr {learning_rate:g}", flush=True)
            if enabled and bad_epochs >= patience:
                stopped_early = True
                break
    model.load_state_dict(torch.load(run / "best.pth", map_location=device, weights_only=True))
    model.eval()
    return {"training_seconds": perf_counter() - started, "epochs_completed": epoch,
            "stopped_early": stopped_early, "best_epoch": best_epoch,
            "best_val_loss": best_loss, "best_val_accuracy": best_acc}

"""Reusable PyTorch training helpers for the BTL03 example notebooks."""
from __future__ import annotations

import math
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one training or evaluation epoch and return sample-weighted metrics."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss is not finite.")

            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            loss_sum += loss.detach().item() * count
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            samples += count

    if samples == 0:
        raise ValueError("DataLoader is empty.")
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    epochs: int,
    checkpoint_path: str | Path,
    patience: int = 5,
    min_delta: float = 0.0,
    scheduler: Any | None = None,
    writer: Any | None = None,
) -> dict[str, Any]:
    """Train with best-val-loss checkpointing, early stopping, and optional logging.

    A ReduceLROnPlateau scheduler is stepped on validation loss. The best
    checkpoint is restored before returning; the caller owns the writer.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if patience < 1 or min_delta < 0:
        raise ValueError("patience must be >= 1 and min_delta must be >= 0.")

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stopping_best = float("inf")
    best_epoch = 0
    bad_epochs = 0
    stopped_early = False
    history: list[dict[str, float | int]] = []
    started = perf_counter()

    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_metrics = run_epoch(model, valid_loader, criterion, device)
        train_loss = train_metrics["loss"]
        train_accuracy = train_metrics["accuracy"]
        valid_loss = valid_metrics["loss"]
        valid_accuracy = valid_metrics["accuracy"]
        if not all(
            math.isfinite(value)
            for value in (train_loss, train_accuracy, valid_loss, valid_accuracy)
        ):
            raise FloatingPointError(f"Metrics are not finite at epoch {epoch}.")

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": valid_loss,
            "val_accuracy": valid_accuracy,
            "learning_rate": learning_rate,
        }
        history.append(row)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            temporary = checkpoint.with_name(checkpoint.name + ".tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint)

        if valid_loss < stopping_best - min_delta:
            stopping_best = valid_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step(valid_loss)
        if writer is not None:
            for name, value in (
                ("Loss/train", train_loss),
                ("Accuracy/train", train_accuracy),
                ("Loss/validation", valid_loss),
                ("Accuracy/validation", valid_accuracy),
                ("LearningRate", learning_rate),
            ):
                writer.add_scalar(name, value, epoch)

        print(
            f"{epoch:02d}/{epochs} "
            f"train {train_loss:.4f} {train_accuracy:.2%} "
            f"val {valid_loss:.4f} {valid_accuracy:.2%} "
            f"lr {learning_rate:g}",
            flush=True,
        )
        if bad_epochs >= patience:
            stopped_early = True
            break

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_accuracy": float(history[best_epoch - 1]["val_accuracy"]),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "training_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint),
    }

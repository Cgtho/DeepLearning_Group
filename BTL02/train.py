import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn

from fer_data import training_loaders
# Re-export these names to preserve existing notebook imports from train.
from training_utils import select_device, seed_everything, resolve_model, check_batch, run_epoch, fit

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/baseline.yaml"


def run_experiment(config_path=DEFAULT_CONFIG, epochs=None, smoke=False, model=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if epochs is not None:
        cfg["epochs"] = epochs
    if smoke:
        cfg["epochs"] = 1
    if min(cfg["epochs"], cfg["batch_size"], cfg["learning_rate"]) <= 0:
        raise ValueError("Epochs, batch size và learning rate phải lớn hơn 0.")
    device = select_device(cfg["device"])
    seed_everything(cfg["seed"])
    train, valid = training_loaders(ROOT / cfg["data_dir"], cfg["batch_size"], cfg["seed"],
                                    augmentation=cfg.get("augmentation"))
    model_name = cfg.get("model", "baseline")
    preflight = check_batch(train.dataset, cfg["batch_size"], device, model, model_name)
    seed_everything(cfg["seed"])
    model = resolve_model(model, model_name).to(device)
    opt_name = cfg.get("optimizer", "adam").lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"],
                                    momentum=cfg.get("momentum", 0.0))
    else:
        raise ValueError("optimizer phải là adam hoặc sgd.")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("Thiếu TensorBoard. Cài bằng: pip install tensorboard") from exc
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ("_smoke" if smoke else "")
    run = ROOT / cfg["output_dir"] / cfg["experiment"] / stamp
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log_dir = ROOT / cfg.get("tensorboard_dir", "runs") / cfg["experiment"] / stamp
    print(run, device, preflight, flush=True)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        writer.add_text("experiment/config", yaml.safe_dump(cfg, sort_keys=False))
        writer.add_graph(model, torch.zeros(1, 1, 48, 48, device=device))
        summary = fit(model, train, valid, optimizer, nn.CrossEntropyLoss(), device,
                      cfg["epochs"], run, writer, cfg.get("early_stopping"), cfg.get("reduce_lr"))
        summary.update({
            "parameters": sum(p.numel() for p in model.parameters()), "model": model_name,
            "train_size": len(train.dataset), "val_size": len(valid.dataset),
            "device": str(device), "torch": str(torch.__version__),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "preflight": preflight, "smoke": smoke,
        })
        (run / "training.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        writer.add_scalar("Loss/best_validation", summary["best_val_loss"], summary["best_epoch"])
        writer.add_hparams({k: cfg[k] for k in ("experiment", "batch_size", "learning_rate", "seed", "epochs")},
                           {k: summary[k] for k in ("best_val_loss", "best_val_accuracy", "best_epoch", "parameters")})
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện CNN trên FER2013")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true", help="Một epoch trên toàn bộ train/validation")
    args = parser.parse_args()
    run_experiment(args.config, args.epochs, args.smoke)

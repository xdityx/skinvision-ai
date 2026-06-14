"""
phase1/scripts/train.py
Entry point for Phase 1 training.

Usage
-----
# Full training run (uses phase1/config/phase1.yaml by default):
    python -m phase1.scripts.train

# With explicit config and 1-epoch smoke test:
    python -m phase1.scripts.train --config phase1/config/phase1.yaml --epochs 1

# Resume from a checkpoint:
    python -m phase1.scripts.train --resume phase1/checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure project root is on PYTHONPATH when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.src.dataset import AcneDataset
from phase1.src.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    save_metrics_json,
    save_training_curves,
)
from phase1.src.model import get_model, predict
from phase1.src.trainer import Trainer
from phase1.src.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1.train")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 — Acne Severity Training")
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "phase1" / "config" / "phase1.yaml",
        help="Path to phase1.yaml",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config (useful for smoke tests)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size from config",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the best checkpoint on the test set",
    )
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataloaders(
    config: dict,
    project_root: Path,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.FloatTensor, list[str]]:
    """
    Build train / val / test DataLoaders from Phase-0 split CSVs.

    Returns
    -------
    train_loader, val_loader, test_loader, class_weights, class_names
    """
    dcfg = config["data"]
    tcfg = config["training"]
    image_size = dcfg["image_size"]

    splits_dir = project_root / dcfg["splits_dir"]
    severity_map: dict = {int(k): v for k, v in dcfg["severity_map"].items()}
    class_names = [severity_map[i] for i in sorted(severity_map)]

    train_ds = AcneDataset(
        splits_dir / "train.csv",
        transform=get_train_transforms(image_size),
    )
    val_ds = AcneDataset(
        splits_dir / "val.csv",
        transform=get_val_transforms(image_size),
    )
    test_ds = AcneDataset(
        splits_dir / "test.csv",
        transform=get_val_transforms(image_size),
    )

    class_weights = train_ds.class_weights()

    # Weighted sampler for training to compensate for class imbalance
    labels = train_ds.label_array()
    sample_weights = class_weights.numpy()[labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_ds),
        replacement=True,
    )

    num_workers = tcfg.get("num_workers", 0)
    pin_memory = tcfg.get("pin_memory", True) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,         # avoid tiny last batch breaking CORN loss
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tcfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        "Datasets: train=%d  val=%d  test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    logger.info("Class weights: %s", dict(zip(class_names, class_weights.tolist())))
    return train_loader, val_loader, test_loader, class_weights, class_names


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    seed = config["training"].get("seed", 42)
    seed_everything(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(
            "GPU: %s | VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — running on CPU (training will be slow).")

    # Data
    train_loader, val_loader, test_loader, class_weights, class_names = build_dataloaders(
        config, PROJECT_ROOT
    )

    # Model
    model = get_model(config)
    logger.info(
        "Model: %s | params=%.1fM",
        config["model"]["backbone"],
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # Optionally resume
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_weights=class_weights,
        class_names=class_names,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    log_dir = Path(config["training"]["log_dir"])
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])

    if not args.eval_only:
        # ── Training ──────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Starting training for %d epochs.", config["training"]["epochs"])
        logger.info("=" * 60)
        fit_result = trainer.fit()
        logger.info(
            "Training complete. Best val macro-F1 = %.4f at epoch %d.",
            fit_result["best_val_f1"],
            fit_result["best_epoch"],
        )

    # ── Test evaluation ────────────────────────────────────────────────────
    logger.info("Evaluating best model on test set…")
    trainer.load_best_checkpoint()
    test_metrics = trainer.evaluate(test_loader)

    logger.info("Test results:")
    logger.info("  Accuracy    : %.4f", test_metrics["accuracy"])
    logger.info("  Macro F1    : %.4f", test_metrics["macro_f1"])
    logger.info("  QWK         : %.4f", test_metrics["qwk"])
    for cls, acc in test_metrics["per_class_accuracy"].items():
        logger.info("  %s accuracy : %s", cls, f"{acc:.4f}" if acc is not None else "N/A")

    # Save outputs
    save_metrics_json(test_metrics, log_dir / "test_metrics.json")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        log_dir / "confusion_matrix.png",
        title="Test Set Confusion Matrix",
    )
    try:
        log_csv = log_dir / "training_log.csv"
        if log_csv.exists():
            save_training_curves(log_csv, log_dir / "training_curves.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save training curves: %s", exc)

    logger.info("Outputs saved to: %s", log_dir)
    logger.info("Best checkpoint: %s", checkpoint_dir / "best_model.pt")


if __name__ == "__main__":
    main()

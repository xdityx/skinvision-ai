"""
phase1/src/trainer.py
Training loop with early stopping, gradient accumulation, mixed precision,
and CSV logging.

Tuned for NVIDIA RTX 3050 4 GB:
  - AMP (FP16 autocast) to stay within VRAM budget
  - Gradient accumulation across micro-batches
  - Pin memory for fast CPU→GPU transfer
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from phase1.src.metrics import compute_metrics, plot_confusion_matrix, save_training_curves
from phase1.src.model import corn_loss, predict

logger = logging.getLogger(__name__)


class Trainer:
    """
    Full training engine for ordinal acne classification.

    Parameters
    ----------
    model : nn.Module
        EfficientNetB2Ordinal instance.
    train_loader : DataLoader
    val_loader : DataLoader
    config : dict
        The full phase1.yaml config dict.
    device : torch.device
    class_weights : torch.FloatTensor | None
        Per-class weights for CORN loss (shape [num_classes]).
    class_names : list[str]
        e.g. ["mild", "moderate", "severe"]
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
        class_weights: Optional[torch.FloatTensor] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.class_names = class_names or [str(i) for i in range(config["model"]["num_classes"])]

        tcfg = config["training"]
        self.num_classes: int = config["model"]["num_classes"]
        self.epochs: int = tcfg["epochs"]
        self.patience: int = tcfg["patience"]
        self.grad_accum: int = tcfg.get("grad_accum_steps", 1)
        self.label_smoothing: float = tcfg.get("label_smoothing", 0.0)
        self.use_amp: bool = tcfg.get("mixed_precision", True) and device.type == "cuda"

        # Checkpoint & log directories
        self.checkpoint_dir = Path(tcfg["checkpoint_dir"])
        self.log_dir = Path(tcfg["log_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path = self.log_dir / "training_log.csv"

        # Optimizer: AdamW with separate LR for backbone vs head
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": tcfg["learning_rate"] * 0.1},
                {"params": model.head.parameters(), "lr": tcfg["learning_rate"]},
            ],
            weight_decay=tcfg["weight_decay"],
        )

        # Cosine LR schedule with linear warmup
        total_steps = self.epochs * len(train_loader) // self.grad_accum
        warmup_steps = tcfg["warmup_epochs"] * len(train_loader) // self.grad_accum
        self.scheduler = _get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # State
        self.best_val_f1: float = -1.0
        self.no_improve_count: int = 0
        self.history: list[dict] = []

        logger.info(
            "Trainer initialised | device=%s | AMP=%s | grad_accum=%d | epochs=%d",
            device,
            self.use_amp,
            self.grad_accum,
            self.epochs,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Run the full training loop.

        Returns
        -------
        dict with 'best_val_f1', 'best_epoch', and 'history'.
        """
        best_epoch = 0
        _init_log_csv(self.log_csv_path)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            elapsed = time.time() - t0

            val_f1 = val_metrics["macro_f1"]
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
                "val_qwk": val_metrics["qwk"],
                "elapsed_s": round(elapsed, 1),
            }
            self.history.append(row)
            _append_log_csv(self.log_csv_path, row)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_f1=%.4f | val_qwk=%.4f | %.1fs",
                epoch,
                self.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_f1,
                val_metrics["qwk"],
                elapsed,
            )

            # Checkpoint + early stopping
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                best_epoch = epoch
                self.no_improve_count = 0
                self._save_checkpoint(epoch, val_f1, is_best=True)
                logger.info("  ↑ New best val macro-F1: %.4f — checkpoint saved.", val_f1)
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch,
                        self.patience,
                    )
                    break

        # Save training curves
        try:
            save_training_curves(self.log_csv_path, self.log_dir / "training_curves.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save training curves: %s", exc)

        return {
            "best_val_f1": self.best_val_f1,
            "best_epoch": best_epoch,
            "history": self.history,
        }

    def evaluate(self, loader: DataLoader) -> dict:
        """
        Evaluate the model on an arbitrary DataLoader.

        Returns
        -------
        dict with: loss, accuracy, macro_f1, weighted_f1, qwk,
                   per_class_accuracy, confusion_matrix
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = corn_loss(
                        logits,
                        labels,
                        self.num_classes,
                        weights=self.class_weights,
                        label_smoothing=0.0,    # no smoothing at eval
                    )

                total_loss += loss.item() * images.size(0)
                preds = predict(logits, self.num_classes)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / max(len(all_labels), 1)
        metrics = compute_metrics(
            all_labels, all_preds, self.num_classes, self.class_names
        )
        metrics["loss"] = avg_loss
        return metrics

    def load_best_checkpoint(self) -> None:
        """Load the best saved checkpoint into self.model."""
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint from %s (epoch %d, val_f1=%.4f)", path, ckpt["epoch"], ckpt["val_f1"])

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for step, (images, labels) in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = corn_loss(
                    logits,
                    labels,
                    self.num_classes,
                    weights=self.class_weights,
                    label_smoothing=self.label_smoothing,
                )
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Metrics (detached)
            with torch.no_grad():
                preds = predict(logits, self.num_classes)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item() * self.grad_accum * images.size(0)

            pbar.set_postfix(
                loss=f"{loss.item() * self.grad_accum:.4f}",
                acc=f"{correct / total:.3f}",
            )

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    def _save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False) -> None:
        fname = "best_model.pt" if is_best else f"epoch_{epoch:03d}.pt"
        path = self.checkpoint_dir / fname
        torch.save(
            {
                "epoch": epoch,
                "val_f1": val_f1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing LR schedule with a linear warm-up phase."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# CSV logging helpers
# ──────────────────────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    "epoch", "train_loss", "train_acc",
    "val_loss", "val_acc", "val_macro_f1", "val_qwk", "elapsed_s",
]


def _init_log_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_LOG_FIELDS).writeheader()


def _append_log_csv(path: Path, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_LOG_FIELDS).writerow(
            {k: row.get(k, "") for k in _LOG_FIELDS}
        )

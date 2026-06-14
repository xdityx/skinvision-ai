"""
phase1/src/metrics.py
Evaluation metrics for ordinal acne severity classification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


# ─── Core metric computation ──────────────────────────────────────────────────

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of int
    y_pred : array-like of int
    num_classes : int
    class_names : list[str] | None
        Human-readable names for each class (for per-class accuracy keys).

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1, qwk,
        per_class_accuracy, confusion_matrix (list of lists)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    # QWK: quadratic weighted kappa — the standard metric for ordinal tasks
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class accuracy = recall = TP / (TP + FN)
    per_class_acc = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            per_class_acc[class_names[c]] = None
        else:
            per_class_acc[class_names[c]] = float((y_pred[mask] == c).mean())

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
    }


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray | list,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """
    Save a colour-annotated confusion matrix as a PNG.

    Parameters
    ----------
    cm : 2-D array-like
    class_names : list[str]
    output_path : Path
    title : str

    Returns
    -------
    Path  (the saved file)
    """
    cm = np.asarray(cm, dtype=int)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )

    # Annotate each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_training_curves(
    log_csv_path: Path,
    output_path: Path,
) -> Path:
    """
    Read the CSV training log and save loss + macro-F1 curves.

    Expected CSV columns: epoch, train_loss, val_loss, train_acc, val_macro_f1.

    Returns
    -------
    Path  (the saved figure)
    """
    import pandas as pd  # noqa: PLC0415

    log_csv_path = Path(log_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_csv_path)
    epochs = df["epoch"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, df["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(epochs, df["val_loss"], label="Val Loss", color="tomato", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Macro-F1
    if "val_macro_f1" in df.columns:
        axes[1].plot(
            epochs,
            df["val_macro_f1"],
            label="Val Macro-F1",
            color="seagreen",
        )
    if "train_acc" in df.columns:
        axes[1].plot(
            epochs,
            df["train_acc"],
            label="Train Acc",
            color="royalblue",
            linestyle="--",
            alpha=0.7,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Macro-F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metrics_json(metrics: dict, output_path: Path) -> Path:
    """Serialise a metrics dict to JSON (handles numpy types)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(metrics, default=_convert))
    output_path.write_text(json.dumps(clean, indent=2))
    return output_path

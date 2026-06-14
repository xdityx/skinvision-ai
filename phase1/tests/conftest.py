"""
phase1/tests/conftest.py
Shared fixtures for Phase 1 test suite.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Return a temporary directory acting as project root."""
    return tmp_path


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a 260x260 RGB image on disk and return its path."""
    p = tmp_path / "test_image.jpg"
    arr = np.random.randint(0, 255, (260, 260, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    """
    Create a tiny split CSV (10 rows, 3 classes) with real images on disk.
    Returns path to the CSV.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rows = []
    for i in range(10):
        label = i % 3
        img_path = img_dir / f"img_{i:03d}.jpg"
        arr = np.random.randint(0, 255, (260, 260, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)
        rows.append({
            "image_id": f"test__{i:03d}",
            "image_path": str(img_path),
            "severity_label": label,
            "severity_name": ["mild", "moderate", "severe"][label],
            "cluster_id": i,
            "split": "train",
        })

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def minimal_config() -> dict:
    return {
        "project": {"seed": 42, "name": "test"},
        "model": {
            "backbone": "efficientnet_b2",
            "pretrained": False,
            "num_classes": 3,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "num_workers": 0,
            "pin_memory": False,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 0,
            "patience": 5,
            "mixed_precision": False,
            "label_smoothing": 0.0,
            "seed": 42,
            "checkpoint_dir": "phase1/checkpoints_test",
            "log_dir": "phase1/logs_test",
        },
        "data": {
            "splits_dir": "data/phase0_outputs/splits",
            "image_size": 64,
            "severity_map": {0: "mild", 1: "moderate", 2: "severe"},
        },
    }

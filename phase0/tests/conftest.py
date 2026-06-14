"""
phase0/tests/conftest.py
Shared pytest fixtures for Phase 0 test suite.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_jpeg(path: Path, color: tuple[int, int, int] = (128, 100, 80)) -> None:
    """Create a small solid-color 100×100 JPEG image."""
    arr = np.full((100, 100, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path), "JPEG")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal fake project structure for testing."""
    # Class folders under data/raw/ACNE04/
    class_folders = [
        "acne0_1024",
        "acne1_1024",
        "acne2_1024",
        "acne3_1024",
    ]
    colors = [
        (50, 50, 50),
        (100, 80, 60),
        (150, 120, 100),
        (200, 180, 160),
    ]

    raw_root = tmp_path / "data" / "raw" / "ACNE04"
    for folder, color in zip(class_folders, colors):
        class_dir = raw_root / folder
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img_path = class_dir / f"img_{folder}_{i:03d}.jpg"
            _make_synthetic_jpeg(img_path, color=color)

    # Output and reports directories
    (tmp_path / "data" / "phase0_outputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "phase0_outputs" / "splits").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "phase0" / "figures").mkdir(parents=True, exist_ok=True)

    # Write a minimal phase0.yaml
    config_dir = tmp_path / "phase0" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "project": {"seed": 42, "name": "acne04-phase0-test"},
        "paths": {
            "data_root": "data/raw/ACNE04",
            "outputs_root": "data/phase0_outputs",
            "reports_root": "reports/phase0",
            "figures_dir": "reports/phase0/figures",
        },
        "ingestion": {
            "format": "auto",
            "class_folders": class_folders,
            "label_mapping": {0: 0, 1: 1, 2: 2, 3: 2},
            "severity_map": {0: "mild", 1: "moderate", 2: "severe"},
            "all_folder": "all_1024",
            "sim_csv": "sim_acne.csv",
            "coco_json_filename": "Acne04-v2_annotations.json",
            "supported_extensions": [".jpg", ".jpeg", ".png", ".webp"],
        },
        "eda": {
            "sample_grid_rows": 2,
            "sample_grid_cols": 2,
            "sample_grid_seed": 42,
        },
        "quality": {
            "blur_threshold": 100.0,
            "underexposed_threshold": 40.0,
            "overexposed_threshold": 215.0,
            "low_contrast_threshold": 20.0,
            "face_confidence_threshold": 0.7,
            "face_model_selection": 1,
            "min_image_dimension": 50,  # small so 100x100 test images pass
        },
        "clustering": {
            "embedding_model": "buffalo_sc",
            "embedding_cache": True,
            "face_image_size": 112,
            "dbscan_eps_sweep": [0.30, 0.40, 0.50],
            "dbscan_min_samples": 2,
            "tsne_perplexity": 5,
            "tsne_seed": 42,
            "dendrogram_max_samples": 50,
        },
        "splits": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "distribution_tolerance_pct": 5.0,
            "seed": 42,
        },
        "report": {
            "thresholds": {
                "min_images_per_class": 1,
                "max_imbalance_ratio": 10.0,
                "max_corrupted_pct": 10.0,
                "max_leakage_risk_index": 5.0,
            }
        },
    }

    config_path = config_dir / "phase0.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    return tmp_path


@pytest.fixture
def sample_config(tmp_project: Path) -> dict:
    """Load and return the config dict for tmp_project."""
    config_path = tmp_project / "phase0" / "config" / "phase0.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_manifest_df() -> pd.DataFrame:
    """Return a small DataFrame matching the manifest.csv schema (5 rows)."""
    return pd.DataFrame(
        {
            "image_id": [f"img_{i:04d}" for i in range(5)],
            "image_path": [
                f"/fake/data/raw/ACNE04/acne{i % 4}_1024/img_{i:04d}.jpg"
                for i in range(5)
            ],
            "severity": [0, 1, 2, 3, 0],
            "severity_name": ["mild", "moderate", "severe", "very_severe", "mild"],
            "width": [1024, 1024, 512, 1024, 768],
            "height": [1024, 1024, 512, 1024, 768],
            "format": ["JPEG", "JPEG", "JPEG", "JPEG", "JPEG"],
            "file_size_bytes": [102400, 98304, 51200, 110592, 76800],
        }
    )


@pytest.fixture
def sample_quality_audit_df() -> pd.DataFrame:
    """Return a DataFrame matching quality_audit.csv schema."""
    n = 10
    return pd.DataFrame(
        {
            "image_id": [f"img_{i:04d}" for i in range(n)],
            "image_path": [
                f"/fake/data/raw/ACNE04/acne{i % 4}_1024/img_{i:04d}.jpg"
                for i in range(n)
            ],
            "severity": [i % 4 for i in range(n)],
            "is_corrupted": [False] * n,
            "blur_score": [150.0 + i * 10 for i in range(n)],
            "is_blurry": [False] * n,
            "mean_intensity": [100.0 + i * 5 for i in range(n)],
            "is_underexposed": [False] * n,
            "is_overexposed": [False] * n,
            "std_intensity": [45.0 + i * 2 for i in range(n)],
            "is_low_contrast": [False] * n,
            "face_detected": [True] * (n - 1) + [False],
            "face_confidence": [0.95] * (n - 1) + [0.0],
            "quality_pass": [True] * (n - 1) + [False],
        }
    )

"""
phase1/tests/test_dataset.py
Tests for AcneDataset — does not require torch or GPU.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from phase1.src.dataset import AcneDataset


class TestAcneDataset:
    """Unit tests for AcneDataset."""

    def test_len(self, sample_csv: Path) -> None:
        ds = AcneDataset(sample_csv)
        assert len(ds) == 10

    def test_getitem_shape_no_transform(self, sample_csv: Path) -> None:
        """Without transforms, __getitem__ returns (CHW tensor, label tensor)."""
        import torch
        ds = AcneDataset(sample_csv)
        img, label = ds[0]
        assert img.shape[0] == 3             # C
        assert label.dtype == torch.long

    def test_class_weights_sum(self, sample_csv: Path) -> None:
        ds = AcneDataset(sample_csv)
        weights = ds.class_weights()
        # Weights should be finite positive
        assert (weights > 0).all()
        assert weights.isfinite().all()

    def test_label_array(self, sample_csv: Path) -> None:
        ds = AcneDataset(sample_csv)
        labels = ds.label_array()
        assert labels.dtype == int
        assert len(labels) == len(ds)

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AcneDataset(tmp_path / "nonexistent.csv")

    def test_missing_image_skipped(self, tmp_path: Path) -> None:
        """Rows with missing images should be dropped when skip_missing=True."""
        csv_path = tmp_path / "train.csv"
        pd.DataFrame([
            {
                "image_id": "a",
                "image_path": str(tmp_path / "nonexistent.jpg"),
                "severity_label": 0,
                "severity_name": "mild",
                "cluster_id": 0,
                "split": "train",
            }
        ]).to_csv(csv_path, index=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = AcneDataset(csv_path, skip_missing=True)
            assert len(ds) == 0
            assert any("skipped" in str(w.message).lower() for w in caught)

    def test_missing_image_raises_when_not_skipping(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "train.csv"
        pd.DataFrame([
            {
                "image_id": "a",
                "image_path": str(tmp_path / "nonexistent.jpg"),
                "severity_label": 0,
                "severity_name": "mild",
                "cluster_id": 0,
                "split": "train",
            }
        ]).to_csv(csv_path, index=False)
        with pytest.raises(FileNotFoundError):
            AcneDataset(csv_path, skip_missing=False)

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame([{"image_id": "a"}]).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            AcneDataset(csv_path)

    def test_with_val_transform(self, sample_csv: Path) -> None:
        """With Albumentations transform, image should be a float tensor."""
        import torch
        from phase1.src.transforms import get_val_transforms
        ds = AcneDataset(sample_csv, transform=get_val_transforms(image_size=64))
        img, label = ds[0]
        assert img.dtype == torch.float32
        assert img.shape == (3, 64, 64)

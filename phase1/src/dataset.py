"""
phase1/src/dataset.py
AcneDataset — PyTorch Dataset for the ACNE04 3-class split CSVs produced by Phase 0.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AcneDataset(Dataset):
    """
    Loads images from a Phase-0 split CSV (train / val / test).

    Each row in the CSV must have at minimum:
        image_path    — absolute or relative path to the image file
        severity_label — integer class label (0, 1, 2)

    Parameters
    ----------
    csv_path : Path | str
        Path to one of train.csv / val.csv / test.csv.
    transform : callable | None
        Albumentations Compose pipeline.  If None, images are returned as
        numpy uint8 arrays (useful for testing).
    image_root : Path | None
        If given, image_path values in the CSV are treated as relative to
        this root.  Otherwise they are used as-is.
    skip_missing : bool
        If True, rows whose image file does not exist are silently dropped
        and a warning is logged.  If False, a FileNotFoundError is raised.
    """

    def __init__(
        self,
        csv_path: Path | str,
        transform: Optional[Callable] = None,
        image_root: Optional[Path | str] = None,
        skip_missing: bool = True,
    ) -> None:
        self.transform = transform
        self.image_root = Path(image_root) if image_root else None
        self.skip_missing = skip_missing

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"image_path", "severity_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # Resolve paths and optionally drop missing rows
        valid_rows = []
        skipped = 0
        for _, row in df.iterrows():
            p = self._resolve_path(row["image_path"])
            if not p.exists():
                if skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Image not found: {p}")
            row = row.copy()
            row["_resolved_path"] = str(p)
            valid_rows.append(row)

        if skipped:
            warnings.warn(
                f"{skipped} image(s) from {csv_path.name} were not found on disk "
                "and have been skipped.",
                UserWarning,
                stacklevel=2,
            )

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        logger.info(
            "AcneDataset loaded %d images from %s (skipped=%d)",
            len(self.df),
            csv_path.name,
            skipped,
        )

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["_resolved_path"]
        label = int(row["severity_label"])

        # Load image as RGB numpy array
        image = self._load_image(img_path)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]          # float32 tensor from ToTensorV2
        else:
            # Bare mode (testing): return as uint8 tensor
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Class-weight helpers
    # ------------------------------------------------------------------

    def class_weights(self) -> torch.FloatTensor:
        """
        Returns per-class inverse-frequency weights as a 1-D FloatTensor.

        weight[c] = N / (num_classes * count[c])

        Normalised so the mean weight is 1.0 — safe to pass directly
        to ``torch.nn.CrossEntropyLoss(weight=...)``.
        """
        labels = self.df["severity_label"].values
        num_classes = len(np.unique(labels))
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        n = len(labels)
        weights = n / (num_classes * counts)
        return torch.FloatTensor(weights)

    def label_array(self) -> np.ndarray:
        """Return all integer labels as a 1-D numpy array (for samplers)."""
        return self.df["severity_label"].values.astype(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw)
        if self.image_root and not p.is_absolute():
            return self.image_root / p
        return p

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load image as uint8 HWC RGB numpy array.  Returns grey placeholder on failure."""
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return np.array(img, dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s (%s) — using grey placeholder.", path, exc)
            return np.full((224, 224, 3), 128, dtype=np.uint8)

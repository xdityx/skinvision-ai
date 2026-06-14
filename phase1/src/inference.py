"""
phase1/src/inference.py
AcnePredictor — core inference engine for the trained EfficientNet-B2 + CORN model.

Supports:
  - Single image prediction
  - 5-view deterministic Test-Time Augmentation (TTA)
  - JSON-serialisable output dict
  - Clear validation errors on bad inputs

Output schema
-------------
{
    "image_path":          str,
    "predicted_class":     int,          # 0 | 1 | 2
    "predicted_severity":  str,          # "mild" | "moderate" | "severe"
    "confidence":          float,        # max class probability
    "class_probabilities": {             # all three class probs
        "mild": float,
        "moderate": float,
        "severe": float
    },
    "tta_enabled":         bool,
    "tta_views":           int | None,
    "model_checkpoint":    str,
    "inference_time_ms":   float
}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ─── Validation exceptions ────────────────────────────────────────────────────

class ImageNotFoundError(FileNotFoundError):
    """Raised when the image file does not exist."""

class ImageLoadError(ValueError):
    """Raised when the file exists but cannot be opened as an image."""

class ImageTooSmallError(ValueError):
    """Raised when the image is smaller than the minimum required size."""

class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the model checkpoint file does not exist."""


# ─── TTA views ───────────────────────────────────────────────────────────────

def _build_tta_transforms(image_size: int):
    """
    Return a list of 5 deterministic Albumentations pipelines for TTA.

    Views:
      0 – Standard val (resize → centre crop)
      1 – Horizontal flip + standard val
      2 – Upscale → top-left crop
      3 – Upscale → top-right crop
      4 – Upscale → bottom-centre crop
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)
    over  = int(image_size * 1.143)   # e.g. 297 for size=260
    large = int(image_size * 1.231)   # e.g. 320 for size=260
    s     = image_size
    norm  = [A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()]

    return [
        # View 0: standard val
        A.Compose([A.Resize(over, over), A.CenterCrop(s, s), *norm]),
        # View 1: horizontal flip + standard val
        A.Compose([A.Resize(over, over), A.HorizontalFlip(p=1.0), A.CenterCrop(s, s), *norm]),
        # View 2: upscale → top-left
        A.Compose([A.Resize(large, large), A.Crop(0, 0, s, s), *norm]),
        # View 3: upscale → top-right
        A.Compose([A.Resize(large, large), A.Crop(large - s, 0, large, s), *norm]),
        # View 4: upscale → bottom-centre
        A.Compose([
            A.Resize(large, large),
            A.Crop((large - s) // 2, large - s, (large - s) // 2 + s, large),
            *norm,
        ]),
    ]


# ─── Predictor ───────────────────────────────────────────────────────────────

class AcnePredictor:
    """
    Inference engine that wraps the trained EfficientNet-B2 + CORN model.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to ``best_model.pt`` saved by the trainer.
    config_path : Path | str | None
        Path to ``phase1.yaml``.  If None, the config embedded in the
        checkpoint is used (the trainer saves it there automatically).
    device : str | None
        ``"cuda"`` | ``"cpu"`` | ``"auto"`` (default).  ``"auto"``
        picks CUDA when available.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        config_path: Optional[Path | str] = None,
        device: Optional[str] = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Run training first:  python -m phase1.scripts.train"
            )

        # ── Device ──────────────────────────────────────────────────────────
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("AcnePredictor using device: %s", self.device)

        # ── Load checkpoint ──────────────────────────────────────────────────
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # ── Config: explicit file > embedded in checkpoint ───────────────────
        if config_path is not None:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif "config" in ckpt:
            self.config = ckpt["config"]
        else:
            raise ValueError(
                "No config found.  Pass --config phase1/config/phase1.yaml explicitly."
            )

        mcfg = self.config["model"]
        self.num_classes: int = mcfg["num_classes"]
        severity_map_raw: dict = self.config["data"]["severity_map"]
        self.severity_map: dict[int, str] = {
            int(k): str(v) for k, v in severity_map_raw.items()
        }
        self.class_names: list[str] = [
            self.severity_map[i] for i in sorted(self.severity_map)
        ]
        self.image_size: int = self.config["data"]["image_size"]

        # ── Model ────────────────────────────────────────────────────────────
        from phase1.src.model import EfficientNetB2Ordinal
        self.model = EfficientNetB2Ordinal(
            num_classes=self.num_classes,
            dropout=mcfg.get("dropout", 0.3),
            pretrained=False,          # weights come from checkpoint
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._checkpoint_epoch: int = ckpt.get("epoch", -1)
        self._checkpoint_val_f1: float = ckpt.get("val_f1", float("nan"))
        logger.info(
            "Loaded checkpoint  epoch=%d  val_f1=%.4f",
            self._checkpoint_epoch,
            self._checkpoint_val_f1,
        )

        # ── Transforms (lazy-built on first TTA call) ────────────────────────
        from phase1.src.transforms import get_val_transforms
        self._val_transform = get_val_transforms(self.image_size)
        self._tta_transforms: list | None = None  # built lazily

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        image_path: Path | str,
        tta: bool = False,
        save_to: Optional[Path | str] = None,
    ) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image_path : Path | str
            Path to the input image.
        tta : bool
            Enable 5-view test-time augmentation.
        save_to : Path | str | None
            If given, write the output dict as JSON to this path.

        Returns
        -------
        dict — see module docstring for schema.

        Raises
        ------
        ImageNotFoundError   if the file does not exist.
        ImageLoadError       if the file is not a valid image.
        ImageTooSmallError   if the image is below minimum dimensions.
        """
        image_path = Path(image_path).resolve()
        t0 = time.perf_counter()

        # Validate + load
        pil_image = self._validate_and_load(image_path)

        # Inference
        with torch.no_grad():
            if tta:
                probs_np = self._predict_tta(pil_image)
                n_views = len(self._get_tta_transforms())
            else:
                tensor = self._apply_val_transform(pil_image)
                probs_np = self._predict_tensor(tensor)
                n_views = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build output
        pred_class = int(np.argmax(probs_np))
        result = {
            "image_path": str(image_path),
            "predicted_class": pred_class,
            "predicted_severity": self.severity_map[pred_class],
            "confidence": round(float(probs_np[pred_class]), 6),
            "class_probabilities": {
                name: round(float(probs_np[i]), 6)
                for i, name in enumerate(self.class_names)
            },
            "tta_enabled": tta,
            "tta_views": n_views,
            "model_checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": self._checkpoint_epoch,
            "checkpoint_val_f1": round(self._checkpoint_val_f1, 6),
            "inference_time_ms": round(elapsed_ms, 2),
        }

        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            save_to.write_text(json.dumps(result, indent=2))
            logger.info("Prediction saved to %s", save_to)

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_and_load(self, image_path: Path) -> Image.Image:
        """Strict validation pipeline — raises typed exceptions on failure."""
        # 1. Existence
        if not image_path.exists():
            raise ImageNotFoundError(
                f"Image not found: {image_path}"
            )
        if not image_path.is_file():
            raise ImageLoadError(
                f"Path exists but is not a file: {image_path}"
            )

        # 2. Extension sanity check (warn only — PIL is the authority)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if image_path.suffix.lower() not in valid_exts:
            logger.warning(
                "Unexpected file extension '%s' — attempting to open anyway.",
                image_path.suffix,
            )

        # 3. PIL load + verify
        try:
            img = Image.open(image_path)
            img.verify()               # checks file integrity
        except UnidentifiedImageError:
            raise ImageLoadError(
                f"File is not a recognised image format: {image_path}"
            )
        except Exception as exc:
            raise ImageLoadError(
                f"Failed to open image {image_path}: {exc}"
            ) from exc

        # Must reopen after verify() (PIL closes after verify)
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ImageLoadError(
                f"Failed to convert image to RGB: {image_path}: {exc}"
            ) from exc

        # 4. Minimum size check
        min_dim = 64  # absolute floor; model needs at least image_size px
        w, h = img.size
        if w < min_dim or h < min_dim:
            raise ImageTooSmallError(
                f"Image is too small ({w}×{h} px). "
                f"Minimum required: {min_dim}×{min_dim} px."
            )

        return img

    def _apply_val_transform(self, pil_image: Image.Image) -> torch.Tensor:
        """Apply standard val preprocessing and return a [1, C, H, W] tensor."""
        arr = np.array(pil_image, dtype=np.uint8)
        out = self._val_transform(image=arr)
        return out["image"].unsqueeze(0).to(self.device)     # [1, C, H, W]

    def _predict_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Run forward pass, return class probabilities as float32 numpy array."""
        from phase1.src.model import corn_label_to_probs
        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)                      # [1, num_classes]
        probs = corn_label_to_probs(logits, self.num_classes)  # [1, num_classes]
        return probs.squeeze(0).cpu().float().numpy()        # [num_classes]

    def _predict_tta(self, pil_image: Image.Image) -> np.ndarray:
        """
        5-view deterministic TTA.
        Each view is preprocessed independently; probabilities are averaged.
        """
        transforms = self._get_tta_transforms()
        arr = np.array(pil_image, dtype=np.uint8)
        accumulated = np.zeros(self.num_classes, dtype=np.float32)

        for tfm in transforms:
            augmented = tfm(image=arr)
            tensor = augmented["image"].unsqueeze(0).to(self.device)
            probs = self._predict_tensor(tensor)
            accumulated += probs

        return accumulated / len(transforms)

    def _get_tta_transforms(self) -> list:
        """Lazily build and cache TTA transforms."""
        if self._tta_transforms is None:
            self._tta_transforms = _build_tta_transforms(self.image_size)
        return self._tta_transforms

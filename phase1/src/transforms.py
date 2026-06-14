"""
phase1/src/transforms.py
Albumentations augmentation pipelines for train / val / test.

ImageNet mean/std are used because EfficientNet-B2 is pretrained on ImageNet.
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 260) -> A.Compose:
    """
    Augmentation pipeline for training.

    Designed for close-up face / skin images of acne:
    - Random crops preserve lesion texture without stretching.
    - Colour jitter accounts for camera/lighting variation.
    - CoarseDropout simulates occlusion and improves robustness.
    - NO vertical flip / heavy geometric distortion — would destroy
      severity cues that depend on lesion density.
    """
    return A.Compose(
        [
            # Spatial
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=1,        # INTER_LINEAR
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                cval=0,                 # constant border fill value
                p=0.4,
            ),

            # Colour
            A.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.05,
                p=0.6,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.ToGray(p=0.05),           # rare grayscale — prevent colour bias

            # Noise / blur (light — don't destroy lesion texture)
            A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),

            # Regularisation via occlusion
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0,
                p=0.25,
            ),

            # Normalisation + tensor conversion
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),               # HWC uint8 → CHW float32
        ]
    )


def get_val_transforms(image_size: int = 260) -> A.Compose:
    """
    Deterministic pipeline for validation and test sets.

    Uses centre-crop after a slight oversize resize so we always
    see the same 260×260 region of each image.
    """
    resize_to = int(image_size * 1.143)  # ≈ 297 for image_size=260; standard practice
    return A.Compose(
        [
            A.Resize(height=resize_to, width=resize_to),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )

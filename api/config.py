"""
api/config.py
Application settings — all values can be overridden with environment variables
prefixed ACNE_ (e.g. ACNE_DEVICE=cpu).
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACNE_", case_sensitive=False)

    # Model
    checkpoint_path: str = "phase1/checkpoints/best_model.pt"
    config_path: str = "phase1/config/phase1.yaml"
    device: str = "auto"
    model_version: str = "acne-classifier-v1.0"

    # Validation
    max_file_size_mb: int = 10
    allowed_content_types: list[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

    # Face detection
    face_detection_confidence: float = 0.5
    face_model_selection: int = 1    # 1 = full-range (better for clinical photos)

    # API
    api_title: str = "Acne Severity API"
    api_version: str = "1.0.0"
    api_description: str = (
        "Ordinal acne severity classification using EfficientNet-B2 + CORN loss. "
        "Classifies face images into Mild / Moderate / Severe."
    )


# Singleton — imported by all modules
settings = Settings()

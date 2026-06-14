"""
api/models.py
Pydantic v2 request/response schemas.

These are the ONLY types the API surface exposes.
Internal paths (image_path, model_checkpoint) are never included.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ─── Responses ────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])


class ReadyResponse(BaseModel):
    ready: bool = Field(
        ...,
        description="True when the model is loaded and the server can serve predictions.",
    )
    model_loaded: bool
    checkpoint_epoch: Optional[int] = Field(
        None, description="Training epoch of the loaded checkpoint."
    )
    checkpoint_val_f1: Optional[float] = Field(
        None, description="Validation macro-F1 of the loaded checkpoint."
    )
    model_version: str


class ClassProbabilities(BaseModel):
    mild: float = Field(..., ge=0.0, le=1.0)
    moderate: float = Field(..., ge=0.0, le=1.0)
    severe: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: int = Field(
        ..., ge=0, le=2, description="0=mild, 1=moderate, 2=severe"
    )
    predicted_severity: str = Field(
        ..., examples=["mild", "moderate", "severe"]
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of the predicted class.",
    )
    class_probabilities: ClassProbabilities
    tta_enabled: bool
    tta_views: Optional[int] = Field(
        None, description="Number of TTA views used (null if TTA disabled)."
    )
    face_detected: bool = Field(
        ..., description="True when MediaPipe detected at least one full face."
    )
    face_warning: Optional[str] = Field(
        None, description="Reliability warning from relaxed face validation."
    )
    inference_time_ms: float = Field(
        ..., ge=0.0, description="End-to-end prediction time in milliseconds."
    )
    model_version: str


# ─── Error responses ──────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")


class ErrorResponse(BaseModel):
    detail: ErrorDetail

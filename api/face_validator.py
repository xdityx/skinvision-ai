"""
api/face_validator.py
MediaPipe-based face detection guard for the /predict endpoint.

Rules enforced:
  - No face detected    → FaceNotDetectedError
  - Multiple faces      → MultipleFacesError

Initialise once at app startup; reuse across requests (thread-safe read-only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ─── Typed errors (caught by the router and converted to HTTP 400) ─────────────

class FaceNotDetectedError(ValueError):
    """Raised when no face is found in the uploaded image."""

class MultipleFacesError(ValueError):
    """Raised when more than one face is found in the uploaded image."""


# ─── Detection result ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FaceDetectionResult:
    face_count: int
    max_confidence: float


# ─── Validator ────────────────────────────────────────────────────────────────

class FaceValidator:
    """
    Wraps MediaPipe FaceDetection for single-face enforcement.

    Parameters
    ----------
    min_detection_confidence : float
        Detections below this threshold are ignored.
    model_selection : int
        0 = short-range (selfies, <2 m).
        1 = full-range (clinical photos, up to 5 m).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 1,
    ) -> None:
        import mediapipe as mp

        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection,
        )
        self._min_confidence = min_detection_confidence
        logger.info(
            "FaceValidator initialised (model_selection=%d, min_conf=%.2f)",
            model_selection,
            min_detection_confidence,
        )

    def validate(self, rgb_array: np.ndarray) -> FaceDetectionResult:
        """
        Run face detection on an RGB uint8 numpy array.

        Parameters
        ----------
        rgb_array : np.ndarray shape [H, W, 3] uint8

        Returns
        -------
        FaceDetectionResult

        Raises
        ------
        FaceNotDetectedError   if face_count == 0
        MultipleFacesError     if face_count > 1
        """
        results = self._detector.process(rgb_array)

        detections = results.detections or []
        # Filter by confidence (MediaPipe already applies threshold,
        # but re-check to be explicit)
        confident = [
            d for d in detections
            if d.score and d.score[0] >= self._min_confidence
        ]
        count = len(confident)
        max_conf = max((d.score[0] for d in confident), default=0.0)

        if count == 0:
            raise FaceNotDetectedError(
                "No face detected in the uploaded image. "
                "Please upload a clear, front-facing photo."
            )
        if count > 1:
            raise MultipleFacesError(
                f"{count} faces detected. "
                "Please upload a photo containing exactly one face."
            )

        return FaceDetectionResult(face_count=count, max_confidence=max_conf)

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._detector.close()
        except Exception:  # noqa: BLE001
            pass

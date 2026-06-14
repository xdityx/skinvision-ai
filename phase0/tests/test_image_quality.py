"""
phase0/tests/test_image_quality.py
Tests for the image_quality module.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_synthetic_image(
    path: Path,
    pixel_value: int = 128,
    size: tuple[int, int] = (100, 100),
) -> Path:
    """Save a solid-color JPEG image for testing."""
    arr = np.full((*size, 3), pixel_value, dtype=np.uint8)
    img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "JPEG")
    return path


def _save_random_image(path: Path, size: tuple[int, int] = (100, 100)) -> Path:
    """Save a random-pixel JPEG image (non-blurry)."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "JPEG")
    return path


def _make_mock_mp_context():
    """Return a mock mediapipe module that reports no face detections."""
    mock_mp = MagicMock()
    mock_detection_ctx = MagicMock()
    mock_detection_ctx.__enter__ = MagicMock(
        return_value=MagicMock(
            **{"process.return_value": MagicMock(detections=None)}
        )
    )
    mock_detection_ctx.__exit__ = MagicMock(return_value=False)
    mock_mp.solutions.face_detection.FaceDetection.return_value = mock_detection_ctx
    return mock_mp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAuditNonCorruptedImage:
    def test_audit_non_corrupted_image_passes(self, tmp_path: Path) -> None:
        """A valid synthetic JPEG should have is_corrupted=False."""
        from phase0.src.image_quality import audit_single_image

        img_path = _save_random_image(tmp_path / "test_valid.jpg")

        with patch.dict("sys.modules", {"mediapipe": _make_mock_mp_context()}):
            result = audit_single_image(img_path, config={
                "blur_threshold": 100.0,
                "underexposed_threshold": 40.0,
                "overexposed_threshold": 215.0,
                "low_contrast_threshold": 20.0,
                "face_confidence_threshold": 0.7,
                "face_model_selection": 1,
                "min_image_dimension": 50,
            })

        assert result["is_corrupted"] is False

    def test_audit_reports_blur_score(self, tmp_path: Path) -> None:
        """blur_score should be a non-negative float for a valid image."""
        from phase0.src.image_quality import audit_single_image

        img_path = _save_random_image(tmp_path / "test_blur.jpg")

        with patch.dict("sys.modules", {"mediapipe": _make_mock_mp_context()}):
            result = audit_single_image(img_path, config={
                "blur_threshold": 100.0,
                "underexposed_threshold": 40.0,
                "overexposed_threshold": 215.0,
                "low_contrast_threshold": 20.0,
                "face_confidence_threshold": 0.7,
                "face_model_selection": 1,
                "min_image_dimension": 50,
            })

        assert "blur_score" in result
        assert isinstance(result["blur_score"], (int, float))
        assert result["blur_score"] >= 0.0


class TestExposureFlags:
    def test_exposure_flags_underexposed(self, tmp_path: Path) -> None:
        """Very dark image (pixel value < 40) should be flagged as underexposed."""
        from phase0.src.image_quality import audit_single_image

        # pixel_value=10 → mean intensity ~10, well below 40 threshold
        img_path = _save_synthetic_image(tmp_path / "dark.jpg", pixel_value=10)

        with patch.dict("sys.modules", {"mediapipe": _make_mock_mp_context()}):
            result = audit_single_image(img_path, config={
                "blur_threshold": 100.0,
                "underexposed_threshold": 40.0,
                "overexposed_threshold": 215.0,
                "low_contrast_threshold": 20.0,
                "face_confidence_threshold": 0.7,
                "face_model_selection": 1,
                "min_image_dimension": 50,
            })

        assert result.get("exposure_flag") == "underexposed", (
            f"Expected exposure_flag='underexposed' for dark image, got: {result}"
        )

    def test_exposure_flags_overexposed(self, tmp_path: Path) -> None:
        """Very bright image (pixel value > 215) should be flagged as overexposed."""
        from phase0.src.image_quality import audit_single_image

        # pixel_value=240 → mean intensity ~240, above 215 threshold
        img_path = _save_synthetic_image(tmp_path / "bright.jpg", pixel_value=240)

        with patch.dict("sys.modules", {"mediapipe": _make_mock_mp_context()}):
            result = audit_single_image(img_path, config={
                "blur_threshold": 100.0,
                "underexposed_threshold": 40.0,
                "overexposed_threshold": 215.0,
                "low_contrast_threshold": 20.0,
                "face_confidence_threshold": 0.7,
                "face_model_selection": 1,
                "min_image_dimension": 50,
            })

        assert result.get("exposure_flag") == "overexposed", (
            f"Expected exposure_flag='overexposed' for bright image, got: {result}"
        )


class TestQualityPassCombinesFlags:
    def test_quality_pass_false_for_corrupted(self, tmp_path: Path) -> None:
        """A corrupted (zero-byte) file should yield quality_pass=False."""
        from phase0.src.image_quality import audit_single_image

        # Create a zero-byte (corrupted) file
        img_path = tmp_path / "corrupted.jpg"
        img_path.write_bytes(b"not-a-valid-jpeg")

        with patch.dict("sys.modules", {"mediapipe": _make_mock_mp_context()}):
            result = audit_single_image(img_path, config={
                "blur_threshold": 100.0,
                "underexposed_threshold": 40.0,
                "overexposed_threshold": 215.0,
                "low_contrast_threshold": 20.0,
                "face_confidence_threshold": 0.7,
                "face_model_selection": 1,
                "min_image_dimension": 50,
            })

        assert result.get("quality_pass") is False, (
            f"Expected quality_pass=False for corrupted image, got: {result}"
        )
        assert result.get("is_corrupted") is True

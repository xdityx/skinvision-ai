"""
api/tests/test_predict.py
Tests for POST /api/v1/predict.

Covers
------
- Valid image upload → 200 + correct schema
- Invalid file type  → 400 INVALID_FILE_TYPE
- Oversized file     → 413 FILE_TOO_LARGE
- No face in strict mode → 400 NO_FACE_DETECTED
- Response schema never exposes internal paths
- TTA flag propagates correctly
- Probabilities sum to 1.0
- predicted_class ∈ {0, 1, 2}
- predicted_severity ∈ {"mild", "moderate", "severe"}
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from api.tests.conftest import _need_checkpoint

_PREDICT_URL = "/api/v1/predict"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _upload(client, data: bytes, filename: str = "face.jpg",
            content_type: str = "image/jpeg", tta: bool = False,
            strict_face: bool = False):
    params = {"tta": "true"} if tta else {}
    if strict_face:
        params["strict_face"] = "true"
    return client.post(
        _PREDICT_URL,
        files={"file": (filename, io.BytesIO(data), content_type)},
        params=params,
    )


# ─── Valid image tests ─────────────────────────────────────────────────────────

@_need_checkpoint
class TestValidPrediction:

    def test_real_face_returns_200(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available (dataset not found)")
        r = _upload(client, real_face_jpg)
        assert r.status_code == 200, r.text

    def test_response_schema_keys(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        required = {
            "predicted_class", "predicted_severity", "confidence",
            "class_probabilities", "tta_enabled", "tta_views",
            "face_detected", "face_warning", "inference_time_ms",
            "model_version",
        }
        assert required.issubset(body.keys())

    def test_old_response_fields_still_work(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        assert isinstance(body["predicted_class"], int)
        assert isinstance(body["predicted_severity"], str)
        assert isinstance(body["confidence"], float)
        assert isinstance(body["class_probabilities"], dict)
        assert isinstance(body["tta_enabled"], bool)
        assert "tta_views" in body
        assert isinstance(body["inference_time_ms"], float)
        assert isinstance(body["model_version"], str)

    def test_no_internal_paths_in_response(self, client, real_face_jpg) -> None:
        """image_path and model_checkpoint must never appear in the response."""
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        body_str = str(body)
        assert "image_path" not in body_str
        assert "model_checkpoint" not in body_str
        assert "checkpoints" not in body_str

    def test_predicted_class_valid_range(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        assert body["predicted_class"] in {0, 1, 2}

    def test_predicted_severity_valid(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        assert body["predicted_severity"] in {"mild", "moderate", "severe"}

    def test_probabilities_sum_to_one(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body  = _upload(client, real_face_jpg).json()
        probs = body["class_probabilities"]
        total = probs["mild"] + probs["moderate"] + probs["severe"]
        # Tolerance of 5e-4: the API rounds each probability to 6dp individually,
        # so 3 values can accumulate up to ~3 × 5e-7 rounding error per value;
        # the un-rounded CORN probabilities provably sum to exactly 1.0.
        assert abs(total - 1.0) < 5e-4, f"Probabilities sum to {total} (delta={abs(total-1.0):.2e})"

    def test_probabilities_non_negative(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        probs = _upload(client, real_face_jpg).json()["class_probabilities"]
        for name, p in probs.items():
            assert p >= 0.0, f"Negative probability for {name}: {p}"

    def test_confidence_equals_max_prob(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body   = _upload(client, real_face_jpg).json()
        probs  = body["class_probabilities"]
        max_p  = max(probs.values())
        assert abs(body["confidence"] - max_p) < 1e-4

    def test_inference_time_positive(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        assert body["inference_time_ms"] > 0

    def test_tta_disabled_by_default(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg, tta=False).json()
        assert body["tta_enabled"] is False
        assert body["tta_views"] is None

    def test_tta_enabled_flag(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg, tta=True).json()
        assert body["tta_enabled"] is True
        assert body["tta_views"] == 5

    def test_model_version_present(self, client, real_face_jpg) -> None:
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        body = _upload(client, real_face_jpg).json()
        assert isinstance(body["model_version"], str)
        assert len(body["model_version"]) > 0

    def test_png_upload_works(self, client, real_face_jpg) -> None:
        """Re-encode the face JPEG as PNG; should still return 200."""
        if real_face_jpg is None:
            pytest.skip("Real face fixture not available")
        from PIL import Image
        img = Image.open(io.BytesIO(real_face_jpg)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        r = _upload(client, png_bytes, filename="face.png", content_type="image/png")
        assert r.status_code == 200, r.text


# ─── File type validation ──────────────────────────────────────────────────────

@_need_checkpoint
class TestFileTypeValidation:

    def test_pdf_rejected_400(self, client, solid_color_jpg) -> None:
        r = _upload(client, solid_color_jpg,
                    filename="doc.pdf", content_type="application/pdf")
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "INVALID_FILE_TYPE"

    def test_gif_extension_rejected(self, client, solid_color_jpg) -> None:
        r = _upload(client, solid_color_jpg,
                    filename="anim.gif", content_type="image/gif")
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "INVALID_FILE_TYPE"

    def test_txt_file_rejected(self, client) -> None:
        r = _upload(client, b"hello world",
                    filename="notes.txt", content_type="text/plain")
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "INVALID_FILE_TYPE"

    def test_no_extension_rejected(self, client, solid_color_jpg) -> None:
        r = _upload(client, solid_color_jpg,
                    filename="imagefile", content_type="application/octet-stream")
        assert r.status_code == 400

    def test_webp_accepted_type_check(self, client, solid_color_jpg) -> None:
        """webp is allowed; will pass type check but may fail face detection."""
        img = Image.open(io.BytesIO(solid_color_jpg)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="WEBP")
        r = _upload(client, buf.getvalue(),
                    filename="face.webp", content_type="image/webp")
        # Should pass file-type check (may fail face detection → 400)
        assert r.status_code in {200, 400}
        if r.status_code == 400:
            # Must be a face error, not a file-type error
            assert r.json()["detail"]["code"] != "INVALID_FILE_TYPE"


# ─── File size validation ──────────────────────────────────────────────────────

@_need_checkpoint
class TestFileSizeValidation:

    def test_oversized_file_rejected_413(self, client, oversized_jpg) -> None:
        r = _upload(client, oversized_jpg)
        assert r.status_code == 413
        assert r.json()["detail"]["code"] == "FILE_TOO_LARGE"

    def test_error_message_mentions_limit(self, client, oversized_jpg) -> None:
        body = _upload(client, oversized_jpg).json()
        assert "10" in body["detail"]["message"]   # mentions 10 MB limit


# ─── Face detection validation ────────────────────────────────────────────────

@_need_checkpoint
class TestFaceDetection:

    def test_no_face_solid_color_rejected(self, client, solid_color_jpg) -> None:
        """Solid green image has no face — should return 400 NO_FACE_DETECTED."""
        r = _upload(client, solid_color_jpg, strict_face=True)
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "NO_FACE_DETECTED"

    def test_no_face_random_noise_rejected(self, client, random_noise_jpg) -> None:
        """Random noise image has no face."""
        r = _upload(client, random_noise_jpg, strict_face=True)
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "NO_FACE_DETECTED"

    def test_no_face_allowed_with_warning_when_not_strict(self, client, solid_color_jpg) -> None:
        r = _upload(client, solid_color_jpg, strict_face=False)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["face_detected"] is False
        assert body["face_warning"] == "No full face detected. Prediction may be less reliable."
        assert body["predicted_class"] in {0, 1, 2}

    def test_no_face_error_message_helpful(self, client, solid_color_jpg) -> None:
        """Error message should guide the user."""
        body = _upload(client, solid_color_jpg, strict_face=True).json()
        msg  = body["detail"]["message"].lower()
        assert any(word in msg for word in ["face", "photo", "upload"])

    def test_response_includes_face_fields(self, client, solid_color_jpg) -> None:
        body = _upload(client, solid_color_jpg).json()
        assert isinstance(body["face_detected"], bool)
        assert body["face_warning"] is None or isinstance(body["face_warning"], str)

    def test_multiple_faces_rejected_when_strict(self, app, client, solid_color_jpg) -> None:
        from api.face_validator import MultipleFacesError

        class MultipleFaceValidator:
            def validate(self, rgb_array):
                raise MultipleFacesError(
                    "2 faces detected. Please upload a photo containing exactly one face."
                )

        original = app.state.face_validator
        app.state.face_validator = MultipleFaceValidator()
        try:
            r = _upload(client, solid_color_jpg, strict_face=True)
        finally:
            app.state.face_validator = original

        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "MULTIPLE_FACES_DETECTED"

    def test_multiple_faces_allowed_with_warning_when_not_strict(
        self, app, client, solid_color_jpg
    ) -> None:
        from api.face_validator import MultipleFacesError

        class MultipleFaceValidator:
            def validate(self, rgb_array):
                raise MultipleFacesError(
                    "2 faces detected. Please upload a photo containing exactly one face."
                )

        original = app.state.face_validator
        app.state.face_validator = MultipleFaceValidator()
        try:
            r = _upload(client, solid_color_jpg, strict_face=False)
        finally:
            app.state.face_validator = original

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["face_detected"] is True
        assert body["face_warning"] == "Multiple faces detected. Prediction may be unreliable."


# ─── Error schema validation ───────────────────────────────────────────────────

@_need_checkpoint
class TestErrorSchema:

    def test_error_has_code_and_message(self, client, solid_color_jpg) -> None:
        body = _upload(client, solid_color_jpg, strict_face=True).json()
        assert "code" in body["detail"]
        assert "message" in body["detail"]
        assert isinstance(body["detail"]["code"], str)
        assert isinstance(body["detail"]["message"], str)

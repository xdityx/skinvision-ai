"""
api/routers/predict.py
POST /api/v1/predict — main inference endpoint.

Request
-------
multipart/form-data:
  file  : UploadFile   — the face image
  tta   : bool = False — enable 5-view test-time augmentation (query param)

Response (200)
--------------
PredictionResponse — see api/models.py

Error responses
---------------
400 NO_FACE_DETECTED          — no face found in strict mode
400 MULTIPLE_FACES_DETECTED   — more than one face found in strict mode
400 INVALID_FILE_TYPE     — unsupported file extension / content-type
413 FILE_TOO_LARGE        — image exceeds MAX_FILE_SIZE_MB
422 UNPROCESSABLE_ENTITY  — malformed multipart request (FastAPI default)
500 INFERENCE_ERROR       — unexpected model failure
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status
from PIL import Image

from api.config import settings
from api.dependencies import get_face_validator, get_predictor
from api.face_validator import FaceNotDetectedError, FaceValidator, MultipleFacesError
from api.models import ClassProbabilities, ErrorDetail, PredictionResponse
from phase1.src.inference import AcnePredictor, ImageLoadError, ImageTooSmallError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Prediction"])

_MAX_BYTES = settings.max_file_size_mb * 1024 * 1024
_NO_FACE_WARNING = "No full face detected. Prediction may be less reliable."
_MULTIPLE_FACES_WARNING = "Multiple faces detected. Prediction may be unreliable."


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _error(code: str, message: str, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _validate_file_type(file: UploadFile) -> None:
    """Reject files whose extension or content-type is not in the allowlist."""
    ext = Path(file.filename or "").suffix.lower()
    ct  = (file.content_type or "").split(";")[0].strip().lower()

    ext_ok = ext in settings.allowed_extensions
    ct_ok  = ct  in settings.allowed_content_types

    if not ext_ok or not ct_ok:
        raise _error(
            code="INVALID_FILE_TYPE",
            message=(
                f"Unsupported file type (extension='{ext}', content_type='{ct}'). "
                f"Allowed: {sorted(settings.allowed_extensions)}"
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )


async def _read_and_check_size(file: UploadFile) -> bytes:
    """Read upload in chunks; raise 413 if it exceeds the size limit."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 64 * 1024  # 64 KB

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_BYTES:
            raise _error(
                code="FILE_TOO_LARGE",
                message=(
                    f"File exceeds the {settings.max_file_size_mb} MB limit. "
                    f"Please compress or resize the image."
                ),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _bytes_to_rgb_array(data: bytes) -> np.ndarray:
    """Decode image bytes to an RGB uint8 HWC numpy array."""
    import io
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _check_face_status(
    rgb_array: np.ndarray,
    face_validator: FaceValidator,
    strict_face: bool,
) -> tuple[bool, str | None]:
    """Return face status, or raise when strict validation is enabled."""
    try:
        face_validator.validate(rgb_array)
        return True, None
    except FaceNotDetectedError as exc:
        if strict_face:
            raise _error(
                code="NO_FACE_DETECTED",
                message=str(exc),
                status_code=status.HTTP_400_BAD_REQUEST,
            ) from exc
        return False, _NO_FACE_WARNING
    except MultipleFacesError as exc:
        if strict_face:
            raise _error(
                code="MULTIPLE_FACES_DETECTED",
                message=str(exc),
                status_code=status.HTTP_400_BAD_REQUEST,
            ) from exc
        return True, _MULTIPLE_FACES_WARNING


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": None, "description": "Validation error (face / file type)"},
        413: {"model": None, "description": "File too large"},
        500: {"model": None, "description": "Inference failure"},
    },
    summary="Predict acne severity",
    description=(
        "Upload a face image and receive an ordinal severity classification: "
        "**mild**, **moderate**, or **severe**.\n\n"
        "By default, partial acne-region crops are allowed. Set "
        "`strict_face=true` to require exactly one full face. "
        "Enable `tta` for more robust predictions (5-view averaging, ~5× slower)."
    ),
)
async def predict(
    file: UploadFile,
    tta: bool = Query(default=False, description="Enable 5-view test-time augmentation."),
    strict_face: bool = Query(
        default=False,
        description="Reject uploads unless exactly one full face is detected.",
    ),
    predictor: AcnePredictor = Depends(get_predictor),
    face_validator: FaceValidator = Depends(get_face_validator),
) -> PredictionResponse:

    # ── 1. Validate file type ────────────────────────────────────────────────
    _validate_file_type(file)

    # ── 2. Read + enforce size limit ─────────────────────────────────────────
    image_bytes = await _read_and_check_size(file)

    # ── 3. Decode to RGB array (needed for face detection) ───────────────────
    try:
        rgb_array = _bytes_to_rgb_array(image_bytes)
    except Exception as exc:
        raise _error(
            code="IMAGE_DECODE_ERROR",
            message=f"Could not decode the uploaded file as an image: {exc}",
            status_code=status.HTTP_400_BAD_REQUEST,
        ) from exc

    # ── 4. Face detection guard ───────────────────────────────────────────────
    face_detected, face_warning = _check_face_status(
        rgb_array=rgb_array,
        face_validator=face_validator,
        strict_face=strict_face,
    )

    # ── 5. Write to a temporary file for AcnePredictor ───────────────────────
    # On Windows, NamedTemporaryFile must be closed before another process
    # opens it, so we use delete=False and clean up manually.
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False
        ) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        # ── 6. Run inference ──────────────────────────────────────────────────
        try:
            raw = predictor.predict(image_path=tmp_path, tta=tta)
        except (ImageLoadError, ImageTooSmallError) as exc:
            raise _error(
                code="IMAGE_INVALID",
                message=str(exc),
                status_code=status.HTTP_400_BAD_REQUEST,
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected inference error for file %s", file.filename)
            raise _error(
                code="INFERENCE_ERROR",
                message="An unexpected error occurred during inference. Please try again.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    finally:
        # Always clean up the temp file
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── 7. Build response (no internal paths) ────────────────────────────────
    probs = raw["class_probabilities"]
    return PredictionResponse(
        predicted_class=raw["predicted_class"],
        predicted_severity=raw["predicted_severity"],
        confidence=raw["confidence"],
        class_probabilities=ClassProbabilities(
            mild=probs["mild"],
            moderate=probs["moderate"],
            severe=probs["severe"],
        ),
        tta_enabled=raw["tta_enabled"],
        tta_views=raw["tta_views"],
        face_detected=face_detected,
        face_warning=face_warning,
        inference_time_ms=raw["inference_time_ms"],
        model_version=settings.model_version,
    )

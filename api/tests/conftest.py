"""
api/tests/conftest.py
Shared pytest fixtures for API tests.

Strategy
--------
The TestClient uses the real FastAPI app (including lifespan), so the actual
model checkpoint and MediaPipe are loaded once per test session via the
session-scoped `client` fixture.

This means:
  - Tests run against the real model (no mocking needed for the model itself).
  - FaceValidator is replaced per-test where needed via app.state override.
  - Image fixtures are generated in-memory with PIL (no disk I/O).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT   = PROJECT_ROOT / "phase1" / "checkpoints" / "best_model.pt"

_need_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="best_model.pt not found — run training first",
)


# ── App / client ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Create the FastAPI application (loads model once for the session)."""
    from api.main import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """TestClient that runs the full lifespan (model load + warm-up)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Image factories ───────────────────────────────────────────────────────────

def _encode_image(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def solid_color_jpg() -> bytes:
    """
    Plain 300×300 green JPEG — no face.
    Used for no-face rejection tests.
    """
    img = Image.fromarray(
        np.full((300, 300, 3), [60, 180, 60], dtype=np.uint8)
    )
    return _encode_image(img, "JPEG")


@pytest.fixture(scope="session")
def random_noise_jpg() -> bytes:
    """Random noise JPEG — definitely no face."""
    arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    return _encode_image(Image.fromarray(arr), "JPEG")


@pytest.fixture(scope="session")
def oversized_jpg() -> bytes:
    """
    A 11 MB payload constructed by repeating valid JPEG bytes.
    This exceeds the 10 MB limit and should be rejected with 413.
    """
    base = _encode_image(
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)), "JPEG"
    )
    # Pad to 11 MB with JPEG comment bytes (safe, ignored by decoders)
    padding = b"\xff\xfe" + b"X" * (11 * 1024 * 1024 - len(base))
    return base + padding


@pytest.fixture(scope="session")
def real_face_jpg() -> bytes | None:
    """
    A real ACNE04 image that contains exactly one face.
    Returns None if the dataset is not available (test will be skipped).
    """
    candidate = PROJECT_ROOT / "data" / "raw" / "ACNE04" / "acne0_1024" / "levle0_1.jpg"
    if candidate.exists():
        return candidate.read_bytes()
    return None

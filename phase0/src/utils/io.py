"""
phase0/src/utils/io.py
Safe I/O helpers for image loading, hashing, and metadata extraction.

All functions return sentinel values (None / "unknown") on failure rather
than raising, so callers can handle bad files gracefully.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------

def safe_load_pil(path: Path) -> Optional[Image.Image]:
    """Open an image with PIL, verify it is not truncated, then reopen.

    PIL's ``verify()`` call consumes the file object, so the image must be
    reopened after verification.  This two-pass approach catches most forms
    of corruption without fully decoding the pixel data on the first pass.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        A valid :class:`PIL.Image.Image` object, or ``None`` on any failure.
    """
    try:
        # First pass: verify file integrity
        with Image.open(path) as img:
            img.verify()
        # Second pass: actually load the image (verify() closes the file)
        return Image.open(path)
    except (
        FileNotFoundError,
        OSError,
        UnidentifiedImageError,
        Exception,
    ):
        return None


def safe_load_cv2(path: Path) -> Optional[np.ndarray]:
    """Read an image file with OpenCV (BGR channel order), supporting Unicode paths.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        A NumPy ``ndarray`` (H × W × C, dtype=uint8), or ``None`` on any
        failure (file not found, unsupported format, read error, etc.).
    """
    try:
        # Read raw bytes first to bypass Windows Unicode path limitation in cv2.imread
        with open(path, "rb") as fh:
            file_bytes = np.frombuffer(fh.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads the file in *chunk_size*-byte chunks to avoid loading large images
    into memory all at once.

    Args:
        path:       Path to the file.
        chunk_size: Read buffer size in bytes (default 8192).

    Returns:
        Lowercase hex string of the SHA-256 digest.

    Raises:
        OSError: If the file cannot be opened or read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

# Mapping from lower-case suffix to canonical format name
_FORMAT_MAP: dict[str, str] = {
    ".jpg":  "jpeg",
    ".jpeg": "jpeg",
    ".png":  "png",
    ".webp": "webp",
}


def get_image_format(path: Path) -> str:
    """Map a file's extension to a canonical image format string.

    Args:
        path: Path to the image file (only the suffix is inspected).

    Returns:
        One of ``"jpeg"``, ``"png"``, ``"webp"``, or ``"unknown"``.
    """
    return _FORMAT_MAP.get(path.suffix.lower(), "unknown")


def get_image_dimensions(path: Path) -> Optional[tuple[int, int]]:
    """Return ``(width, height)`` for an image without fully decoding it.

    Uses :func:`PIL.Image.open` which reads only the header for most formats,
    making this significantly faster than loading pixel data.

    Args:
        path: Path to the image file.

    Returns:
        ``(width, height)`` as integers, or ``None`` on any failure.
    """
    try:
        with Image.open(path) as img:
            return img.size   # PIL returns (width, height)
    except Exception:  # noqa: BLE001
        return None

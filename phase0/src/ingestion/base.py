"""
phase0/src/ingestion/base.py
Abstract base classes and shared data structures for dataset ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates.

    Attributes:
        x: Left edge (x-coordinate of the top-left corner).
        y: Top edge (y-coordinate of the top-left corner).
        w: Width in pixels.
        h: Height in pixels.
    """

    x: float
    y: float
    w: float
    h: float


@dataclass
class ImageRecord:
    """Represents a single labelled image entry in the dataset.

    Attributes:
        image_id:       Unique string identifier (e.g. ``"acne0_1024__img001"``).
        image_path:     Absolute :class:`~pathlib.Path` to the image file.
        severity_label: Integer grade — 0 (mild), 1 (moderate), 2 (severe),
                        3 (very severe).
        bboxes:         List of :class:`BoundingBox` annotations (may be empty
                        for image-level-only datasets).
    """

    image_id: str
    image_path: Path
    severity_label: int          # 0 = mild, 1 = moderate, 2 = severe, 3 = very_severe
    bboxes: List[BoundingBox] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatasetFormatError(Exception):
    """Raised when the dataset directory layout or file format is unrecognised."""


class DatasetNotFoundError(Exception):
    """Raised when the dataset root directory or a required sub-path is absent."""


# ---------------------------------------------------------------------------
# Abstract adapter interface
# ---------------------------------------------------------------------------

class AnnotationAdapter(ABC):
    """Strategy interface for reading different dataset annotation formats.

    Concrete subclasses translate a specific on-disk layout (folder structure,
    COCO JSON, Pascal VOC XML, …) into a uniform stream of
    :class:`ImageRecord` objects.
    """

    @abstractmethod
    def parse(self) -> Iterator[ImageRecord]:
        """Yield :class:`ImageRecord` objects for every image in the dataset.

        Yields:
            :class:`ImageRecord` instances, one per image.
        """
        ...

    @abstractmethod
    def validate_source(self) -> bool:
        """Validate that the data source is present and structurally correct.

        Returns:
            ``True`` if the source is valid.

        Raises:
            :class:`DatasetNotFoundError`: If required files or folders are absent.
            :class:`DatasetFormatError`:   If the source exists but is malformed.
        """
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """A stable, human-readable identifier for this annotation format.

        Used in manifests and ingestion logs to record which adapter produced
        the data (e.g. ``"folder_structure_acne04_kaggle"``).
        """
        ...

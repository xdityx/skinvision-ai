"""
phase0/src/ingestion/folder_adapter.py
Adapter for the ACNE04 Kaggle dataset stored as four class folders.

Expected layout:
    <data_root>/
        acne0_1024/   <- Mild        (severity 0)
        acne1_1024/   <- Moderate    (severity 1)
        acne2_1024/   <- Severe      (severity 2)
        acne3_1024/   <- Very Severe (severity 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from phase0.src.ingestion.base import (
    AnnotationAdapter,
    BoundingBox,
    DatasetNotFoundError,
    ImageRecord,
)
from phase0.src.utils.logging import get_logger


class FolderStructureAdapter(AnnotationAdapter):
    """Read ACNE04 images organised into four class-labelled directories.

    The severity grade is encoded in the folder name: ``acneN_1024`` → grade N.

    Args:
        data_root: Path to the directory that contains the four class folders.
        config:    Full project config dictionary (not used directly by this
                   adapter but stored for consistency with the factory API).
    """

    CLASS_FOLDERS: list[str] = [
        "acne0_1024",   # grade 0 — Mild
        "acne1_1024",   # grade 1 — Moderate
        "acne2_1024",   # grade 2 — Severe
        "acne3_1024",   # grade 3 — Very Severe
    ]

    IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp"}
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data_root: Path, config: dict) -> None:
        self.data_root: Path = Path(data_root)
        self.config: dict = config
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # AnnotationAdapter interface
    # ------------------------------------------------------------------

    @property
    def format_name(self) -> str:
        """Stable identifier for this adapter."""
        return "folder_structure_acne04_kaggle"

    def validate_source(self) -> bool:
        """Verify that all four class directories exist under *data_root*.

        Returns:
            ``True`` if all folders are present.

        Raises:
            :class:`~phase0.src.ingestion.base.DatasetNotFoundError`: If one or
                more class folders are missing, with a message listing them.
        """
        missing = [
            folder
            for folder in self.CLASS_FOLDERS
            if not (self.data_root / folder).is_dir()
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise DatasetNotFoundError(
                f"FolderStructureAdapter: the following required class folders "
                f"are missing under '{self.data_root}':\n"
                f"  {missing_str}\n"
                f"Expected all of: {self.CLASS_FOLDERS}"
            )
        return True

    def parse(self) -> Iterator[ImageRecord]:
        """Yield one :class:`~phase0.src.ingestion.base.ImageRecord` per image.

        Iterates the class folders **in order** (grade 0 → 3).  Within each
        folder, files are yielded in **sorted** (lexicographic) order for
        reproducibility.

        Yields:
            :class:`~phase0.src.ingestion.base.ImageRecord` instances.
        """
        ingestion_cfg = self.config.get("ingestion", {})
        # YAML keys could be integers, normalise to int just in case
        label_map = {int(k): int(v) for k, v in ingestion_cfg.get("label_mapping", {}).items()}

        for folder_name in self.CLASS_FOLDERS:
            folder_path = self.data_root / folder_name

            # Extract the severity grade from the 5th character: "acneN_1024"[4]
            raw_severity = int(folder_name[4])
            severity = label_map.get(raw_severity, raw_severity)

            image_files = sorted(
                p
                for p in folder_path.iterdir()
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
            )

            self._logger.info(
                "Parsed folder '%s' (raw_severity=%d, mapped_severity=%d): %d image(s) found.",
                folder_name,
                raw_severity,
                severity,
                len(image_files),
            )

            for image_path in image_files:
                yield ImageRecord(
                    image_id=f"{folder_name}__{image_path.stem}",
                    image_path=image_path.resolve(),
                    severity_label=severity,
                    bboxes=[],
                )

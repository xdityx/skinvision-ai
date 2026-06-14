"""
phase0/src/ingestion/metadata_jsonl_adapter.py
=============================================
Adapter for the ACNE04 Kaggle dataset stored in a single merged folder
with a `metadata.jsonl` metadata file (e.g. `all_1024/metadata.jsonl`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from phase0.src.ingestion.base import (
    AnnotationAdapter,
    DatasetNotFoundError,
    ImageRecord,
)
from phase0.src.utils.logging import get_logger


class MetadataJsonlAdapter(AnnotationAdapter):
    """Read ACNE04 images from a merged directory using `metadata.jsonl` labels.

    The severity grade is parsed from the file_name field (e.g. `levle3_87.jpg` -> grade 3).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data_root: Path, config: dict) -> None:
        self.data_root: Path = Path(data_root)
        self.config: dict = config
        self._logger = get_logger(__name__)

        ingestion_cfg = config.get("ingestion", {})
        self.all_folder_name: str = ingestion_cfg.get("all_folder", "all_1024")
        self.metadata_filename: str = "metadata.jsonl"
        self.metadata_path: Path = self.data_root / self.all_folder_name / self.metadata_filename

    # ------------------------------------------------------------------
    # AnnotationAdapter interface
    # ------------------------------------------------------------------

    @property
    def format_name(self) -> str:
        """Stable identifier for this adapter."""
        return "metadata_jsonl_acne04"

    def validate_source(self) -> bool:
        """Verify that the all_folder and metadata.jsonl exist under *data_root*.

        Returns:
            ``True`` if files are present.

        Raises:
            :class:`~phase0.src.ingestion.base.DatasetNotFoundError`: If metadata.jsonl is missing.
        """
        all_dir = self.data_root / self.all_folder_name
        if not all_dir.is_dir():
            raise DatasetNotFoundError(
                f"MetadataJsonlAdapter: the folder '{all_dir}' is missing under '{self.data_root}'."
            )
        if not self.metadata_path.is_file():
            raise DatasetNotFoundError(
                f"MetadataJsonlAdapter: metadata file '{self.metadata_path}' is missing."
            )
        return True

    def parse(self) -> Iterator[ImageRecord]:
        """Yield one :class:`~phase0.src.ingestion.base.ImageRecord` per line in metadata.jsonl.

        Yields:
            :class:`~phase0.src.ingestion.base.ImageRecord` instances.
        """
        self._logger.info("Opening metadata file: '%s'", self.metadata_path)
        records_count = 0

        with open(self.metadata_path, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    file_name = data["file_name"]
                    
                    # Parse severity from filename prefix e.g., "levle3_87.jpg" -> severity 3
                    stem = Path(file_name).stem
                    if not stem.startswith("levle"):
                        self._logger.warning("Line %d: Filename '%s' does not start with 'levle'. Skipping.", line_idx, file_name)
                        continue
                    
                    parts = stem.split("_")
                    raw_severity = int(parts[0][5:])
                    
                    ingestion_cfg = self.config.get("ingestion", {})
                    label_map = {int(k): int(v) for k, v in ingestion_cfg.get("label_mapping", {}).items()}
                    severity = label_map.get(raw_severity, raw_severity)
                    
                    image_path = self.data_root / self.all_folder_name / file_name
                    
                    records_count += 1
                    yield ImageRecord(
                        image_id=f"{self.all_folder_name}__{stem}",
                        image_path=image_path.resolve(),
                        severity_label=severity,
                        bboxes=[],
                    )
                except Exception as exc:
                    self._logger.warning("Line %d: failed to parse metadata record: %s. Skipping.", line_idx, exc)

        self._logger.info(
            "Parsed metadata file '%s': %d image records generated.",
            self.metadata_filename,
            records_count,
        )

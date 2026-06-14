"""
phase0/src/ingestion/coco_json_adapter.py
Stub adapter for a future COCO-format annotation JSON.

This class is intentionally incomplete.  It will be fully implemented once
bounding-box annotations become available for the ACNE04 dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from phase0.src.ingestion.base import (
    AnnotationAdapter,
    DatasetNotFoundError,
    ImageRecord,
)
from phase0.src.utils.logging import get_logger


class CocoJsonAdapter(AnnotationAdapter):
    """Reserved adapter for COCO-format bounding-box annotations.

    .. warning::
        This is a stub.  Calling :meth:`parse` will always raise
        :class:`NotImplementedError`.  Use
        :class:`~phase0.src.ingestion.folder_adapter.FolderStructureAdapter`
        for the current dataset.

    Args:
        data_root: Path to the dataset root directory.
        config:    Full project config dictionary.  The JSON annotation file
                   path is read from ``config["ingestion"]["coco_json_path"]``.
    """

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
        return "coco_json_acne04"

    def validate_source(self) -> bool:
        """Check that the configured COCO JSON file exists.

        The path is read from ``config["ingestion"]["coco_json_path"]``.

        Returns:
            ``True`` if the file exists.

        Raises:
            :class:`~phase0.src.ingestion.base.DatasetNotFoundError`: If the
                JSON file is absent.
            :class:`KeyError`: If ``coco_json_path`` is not present in config.
        """
        json_path = Path(self.config["ingestion"]["coco_json_path"])
        if not json_path.exists():
            raise DatasetNotFoundError(
                f"CocoJsonAdapter: annotation file not found: '{json_path}'\n"
                f"Ensure 'ingestion.coco_json_path' in phase0.yaml points to "
                f"a valid COCO JSON file."
            )
        self._logger.info("CocoJsonAdapter: annotation file found at '%s'.", json_path)
        return True

    def parse(self) -> Iterator[ImageRecord]:
        """Not implemented — reserved for future use.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "CocoJsonAdapter is reserved for future use when bounding box "
            "annotations are available."
        )

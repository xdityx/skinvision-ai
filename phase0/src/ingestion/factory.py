"""
phase0/src/ingestion/factory.py
Factory that selects and instantiates the correct AnnotationAdapter.

Auto-detection priority:
  1. acne0_1024/ directory present  → FolderStructureAdapter
  2. COCO JSON filename present     → CocoJsonAdapter
  3. Neither found                  → DatasetFormatError

If ``config["ingestion"]["format"]`` is not ``"auto"``, the named adapter
is instantiated directly without auto-detection.
"""

from __future__ import annotations

from pathlib import Path

from phase0.src.ingestion.base import (
    AnnotationAdapter,
    DatasetFormatError,
)
from phase0.src.ingestion.coco_json_adapter import CocoJsonAdapter
from phase0.src.ingestion.folder_adapter import FolderStructureAdapter
from phase0.src.ingestion.metadata_jsonl_adapter import MetadataJsonlAdapter
from phase0.src.utils.logging import get_logger

_logger = get_logger(__name__)

# Map of format name string → adapter class
_ADAPTER_REGISTRY: dict[str, type[AnnotationAdapter]] = {
    "folder": FolderStructureAdapter,
    "folder_structure": FolderStructureAdapter,
    "folder_structure_acne04_kaggle": FolderStructureAdapter,
    "coco": CocoJsonAdapter,
    "coco_json": CocoJsonAdapter,
    "coco_json_acne04": CocoJsonAdapter,
    "metadata_jsonl": MetadataJsonlAdapter,
    "metadata_jsonl_acne04": MetadataJsonlAdapter,
}


class AnnotationAdapterFactory:
    """Factory for creating :class:`~phase0.src.ingestion.base.AnnotationAdapter` instances."""

    def __init__(self, config: dict, project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)

        # Resolve data_root from config paths
        paths_cfg = config.get("paths", {})
        data_root_str = paths_cfg.get("data_root", "data/raw/ACNE04")
        self.data_root = Path(data_root_str)
        if not self.data_root.is_absolute():
            self.data_root = self.project_root / self.data_root

    def create(self) -> AnnotationAdapter:
        """Create the adapter instance using resolved paths and config."""
        return self._create(self.data_root, self.config)

    @staticmethod
    def _create(data_root: Path, config: dict) -> AnnotationAdapter:
        """Return the appropriate :class:`AnnotationAdapter` for *data_root*."""
        data_root = Path(data_root)
        fmt: str = config.get("ingestion", {}).get("format", "auto").strip().lower()

        # ------------------------------------------------------------------
        # Explicit format requested — look up in registry
        # ------------------------------------------------------------------
        if fmt != "auto":
            adapter_cls = _ADAPTER_REGISTRY.get(fmt)
            if adapter_cls is None:
                raise DatasetFormatError(
                    f"AnnotationAdapterFactory: unknown format '{fmt}'. "
                    f"Known formats: {sorted(_ADAPTER_REGISTRY.keys())}"
                )
            adapter = adapter_cls(data_root, config)
            _logger.info(
                "AnnotationAdapterFactory: using explicitly configured format '%s' → %s.",
                fmt,
                type(adapter).__name__,
            )
            return adapter

        # ------------------------------------------------------------------
        # Auto-detection
        # ------------------------------------------------------------------
        _logger.info(
            "AnnotationAdapterFactory: auto-detecting format under '%s' …", data_root
        )

        # Priority 1: metadata.jsonl layout (complete ACNE04 in all_1024)
        all_folder_name = config.get("ingestion", {}).get("all_folder", "all_1024")
        if (data_root / all_folder_name / "metadata.jsonl").is_file():
            adapter = MetadataJsonlAdapter(data_root, config)
            _logger.info(
                "AnnotationAdapterFactory: detected metadata.jsonl layout under '%s' → %s.",
                all_folder_name,
                type(adapter).__name__,
            )
            return adapter

        # Priority 2: folder structure (ACNE04 Kaggle)
        if (data_root / "acne0_1024").is_dir():
            adapter = FolderStructureAdapter(data_root, config)
            _logger.info(
                "AnnotationAdapterFactory: detected folder-structure layout → %s.",
                type(adapter).__name__,
            )
            return adapter

        # Priority 3: COCO JSON file
        coco_json_path_raw: str = (
            config.get("ingestion", {}).get("coco_json_path", "")
        )
        if coco_json_path_raw:
            candidate = Path(coco_json_path_raw)
            if candidate.exists():
                adapter = CocoJsonAdapter(data_root, config)
                _logger.info(
                    "AnnotationAdapterFactory: detected COCO JSON at '%s' → %s.",
                    candidate,
                    type(adapter).__name__,
                )
                return adapter

        # Nothing matched — list what is actually present for diagnostics
        try:
            found_items = sorted(
                f"{'[DIR] ' if p.is_dir() else '[FILE]'} {p.name}"
                for p in data_root.iterdir()
            )
            found_str = "\n  ".join(found_items) if found_items else "(directory is empty)"
        except OSError:
            found_str = "(unable to list directory contents)"

        raise DatasetFormatError(
            f"AnnotationAdapterFactory: could not auto-detect a supported "
            f"dataset format under '{data_root}'.\n"
            f"Files/folders found:\n  {found_str}\n"
            f"Expected either:\n"
            f"  • An 'acne0_1024/' sub-directory (folder-structure layout), or\n"
            f"  • A COCO JSON file path configured at "
            f"config['ingestion']['coco_json_path']."
        )

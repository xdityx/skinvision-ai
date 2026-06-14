"""
phase0/tests/ingestion/test_factory.py
Tests for AnnotationAdapterFactory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from phase0.src.ingestion.factory import AnnotationAdapterFactory
from phase0.src.ingestion.folder_adapter import FolderStructureAdapter
from phase0.src.ingestion.coco_json_adapter import CocoJsonAdapter
from phase0.src.ingestion.base import DatasetFormatError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    tmp_project: Path,
    format_: str = "auto",
    class_folders: list[str] | None = None,
    json_filename: str = "Acne04-v2_annotations.json",
) -> dict:
    data_root = tmp_project / "data" / "raw" / "ACNE04"
    return {
        "paths": {"data_root": "data/raw/ACNE04"},
        "ingestion": {
            "format": format_,
            "class_folders": class_folders
            or ["acne0_1024", "acne1_1024", "acne2_1024", "acne3_1024"],
            "severity_map": {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"},
            "coco_json_filename": json_filename,
            "coco_json_path": str(data_root / json_filename),
            "supported_extensions": [".jpg", ".jpeg", ".png", ".webp"],
        },
    }


def _write_dummy_coco_json(project_root: Path, filename: str = "Acne04-v2_annotations.json") -> None:
    data_root = project_root / "data" / "raw" / "ACNE04"
    data_root.mkdir(parents=True, exist_ok=True)
    with open(data_root / filename, "w", encoding="utf-8") as f:
        json.dump({"images": [], "annotations": [], "categories": []}, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnnotationAdapterFactory:
    def test_creates_folder_adapter_when_acne_dirs_exist(self, tmp_project: Path) -> None:
        """Auto-detection should select FolderStructureAdapter when all class dirs exist."""
        cfg = _make_config(tmp_project)
        factory = AnnotationAdapterFactory(cfg, tmp_project)
        adapter = factory.create()
        assert isinstance(adapter, FolderStructureAdapter)

    def test_creates_coco_adapter_when_json_exists_and_no_dirs(self, tmp_project: Path) -> None:
        """Auto-detection should select CocoJsonAdapter when only a JSON file is present."""
        # Remove acne class directories
        import shutil
        raw_root = tmp_project / "data" / "raw" / "ACNE04"
        for folder in ["acne0_1024", "acne1_1024", "acne2_1024", "acne3_1024"]:
            folder_path = raw_root / folder
            if folder_path.exists():
                shutil.rmtree(folder_path)

        # Place only the COCO JSON
        _write_dummy_coco_json(tmp_project)

        cfg = _make_config(tmp_project)
        factory = AnnotationAdapterFactory(cfg, tmp_project)
        adapter = factory.create()
        assert isinstance(adapter, CocoJsonAdapter)

    def test_raises_error_when_format_unknown(self, tmp_project: Path) -> None:
        """When no known files are found, factory should raise DatasetFormatError."""
        import shutil

        # Remove all known data
        raw_root = tmp_project / "data" / "raw" / "ACNE04"
        if raw_root.exists():
            shutil.rmtree(raw_root)
        raw_root.mkdir(parents=True, exist_ok=True)

        cfg = _make_config(tmp_project)
        factory = AnnotationAdapterFactory(cfg, tmp_project)
        with pytest.raises(DatasetFormatError):
            factory.create()

    def test_explicit_format_override(self, tmp_project: Path) -> None:
        """Explicit format='folder' in config forces FolderStructureAdapter."""
        # Write JSON file too — explicit override should ignore it
        _write_dummy_coco_json(tmp_project)

        cfg = _make_config(tmp_project, format_="folder")
        factory = AnnotationAdapterFactory(cfg, tmp_project)
        adapter = factory.create()
        assert isinstance(adapter, FolderStructureAdapter)

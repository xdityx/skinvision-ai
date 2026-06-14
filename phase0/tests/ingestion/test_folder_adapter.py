"""
phase0/tests/ingestion/test_folder_adapter.py
Tests for FolderStructureAdapter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from phase0.src.ingestion.folder_adapter import FolderStructureAdapter
from phase0.src.ingestion.base import ImageRecord, DatasetNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(class_folders: list[str] | None = None) -> dict:
    return {
        "paths": {"data_root": "data/raw/ACNE04"},
        "ingestion": {
            "class_folders": class_folders
            or ["acne0_1024", "acne1_1024", "acne2_1024", "acne3_1024"],
            "label_mapping": {0: 0, 1: 1, 2: 2, 3: 2},
            "severity_map": {0: "mild", 1: "moderate", 2: "severe"},
            "supported_extensions": [".jpg", ".jpeg", ".png", ".webp"],
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFolderAdapterValidate:
    def test_validate_source_passes_with_all_folders(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = FolderStructureAdapter(data_root, cfg)
        # Should not raise
        result = adapter.validate_source()
        assert result is True

    def test_validate_source_raises_when_folder_missing(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        import shutil
        shutil.rmtree(data_root / "acne3_1024")
        adapter = FolderStructureAdapter(data_root, cfg)
        with pytest.raises(DatasetNotFoundError):
            adapter.validate_source()


class TestFolderAdapterParse:
    def _get_records(self, tmp_project: Path) -> list[ImageRecord]:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = FolderStructureAdapter(data_root, cfg)
        return list(adapter.parse())

    def test_parse_yields_correct_record_count(self, tmp_project: Path) -> None:
        # 4 class folders × 3 images each = 12
        records = self._get_records(tmp_project)
        assert len(records) == 12

    def test_parse_image_ids_are_unique(self, tmp_project: Path) -> None:
        records = self._get_records(tmp_project)
        ids = [r.image_id for r in records]
        assert len(ids) == len(set(ids)), "Duplicate image_ids found"

    def test_parse_severity_labels_correct(self, tmp_project: Path) -> None:
        records = self._get_records(tmp_project)
        by_severity: dict[int, list[ImageRecord]] = {}
        for r in records:
            by_severity.setdefault(r.severity_label, []).append(r)

        assert 0 in by_severity, "No records with severity 0"
        assert 3 not in by_severity, "Severity 3 was not mapped to 2"
        assert 2 in by_severity, "No records with severity 2"

        # acne0_1024 → severity 0
        sev0_records = by_severity[0]
        for r in sev0_records:
            assert "acne0_1024" in str(r.image_path), (
                f"Severity 0 record has wrong path: {r.image_path}"
            )

        # acne3_1024 → mapped to severity 2
        sev2_records = by_severity[2]
        acne3_paths = [str(r.image_path) for r in sev2_records]
        assert any("acne3_1024" in p for p in acne3_paths), "No acne3 records mapped to severity 2"

    def test_parse_image_paths_are_absolute(self, tmp_project: Path) -> None:
        records = self._get_records(tmp_project)
        for r in records:
            assert Path(r.image_path).is_absolute(), (
                f"image_path is not absolute: {r.image_path}"
            )


class TestFolderAdapterFormatName:
    def test_format_name(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = FolderStructureAdapter(data_root, cfg)
        name = adapter.format_name
        assert isinstance(name, str)
        assert len(name) > 0
        assert "folder" in name.lower()

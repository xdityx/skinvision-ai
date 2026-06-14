"""
phase0/tests/ingestion/test_coco_json_adapter.py
Tests for CocoJsonAdapter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from phase0.src.ingestion.coco_json_adapter import CocoJsonAdapter
from phase0.src.ingestion.base import DatasetNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_project: Path, json_filename: str = "Acne04-v2_annotations.json") -> dict:
    data_root = tmp_project / "data" / "raw" / "ACNE04"
    return {
        "paths": {"data_root": "data/raw/ACNE04"},
        "ingestion": {
            "coco_json_filename": json_filename,
            "coco_json_path": str(data_root / json_filename),
            "supported_extensions": [".jpg", ".jpeg", ".png"],
        },
    }


def _write_dummy_coco_json(project_root: Path, json_filename: str) -> Path:
    """Write a minimal COCO JSON file into the data_root directory."""
    data_root = project_root / "data" / "raw" / "ACNE04"
    data_root.mkdir(parents=True, exist_ok=True)
    json_path = data_root / json_filename
    dummy = {
        "images": [{"id": 1, "file_name": "img_0001.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0}],
        "categories": [{"id": 0, "name": "mild"}],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dummy, f)
    return json_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCocoJsonAdapterParse:
    def test_parse_raises_not_implemented(self, tmp_project: Path) -> None:
        """CocoJsonAdapter.parse() is not yet implemented."""
        _write_dummy_coco_json(tmp_project, "Acne04-v2_annotations.json")
        cfg = _make_config(tmp_project)
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = CocoJsonAdapter(data_root, cfg)
        with pytest.raises(NotImplementedError):
            list(adapter.parse())


class TestCocoJsonAdapterValidate:
    def test_validate_source_raises_when_json_missing(self, tmp_project: Path) -> None:
        """Validation should raise DatasetNotFoundError if the JSON file is absent."""
        cfg = _make_config(tmp_project, "nonexistent_annotations.json")
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = CocoJsonAdapter(data_root, cfg)
        with pytest.raises(DatasetNotFoundError):
            adapter.validate_source()

    def test_validate_source_passes_when_json_present(self, tmp_project: Path) -> None:
        """Validation should succeed when the JSON file exists."""
        _write_dummy_coco_json(tmp_project, "Acne04-v2_annotations.json")
        cfg = _make_config(tmp_project)
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = CocoJsonAdapter(data_root, cfg)
        result = adapter.validate_source()
        assert result is True


class TestCocoJsonAdapterFormatName:
    def test_format_name(self, tmp_project: Path) -> None:
        cfg = _make_config(tmp_project)
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = CocoJsonAdapter(data_root, cfg)
        name = adapter.format_name
        assert isinstance(name, str)
        assert len(name) > 0
        assert "coco" in name.lower()

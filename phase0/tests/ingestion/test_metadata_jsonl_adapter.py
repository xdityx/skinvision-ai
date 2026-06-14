"""
phase0/tests/ingestion/test_metadata_jsonl_adapter.py
Tests for MetadataJsonlAdapter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from phase0.src.ingestion.metadata_jsonl_adapter import MetadataJsonlAdapter
from phase0.src.ingestion.base import ImageRecord, DatasetNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(all_folder: str = "all_1024") -> dict:
    return {
        "paths": {
            "data_root": "data/raw/ACNE04",
            "outputs_root": "data/phase0_outputs",
        },
        "ingestion": {
            "all_folder": all_folder,
            "label_mapping": {0: 0, 1: 1, 2: 2, 3: 2},
            "severity_map": {0: "mild", 1: "moderate", 2: "severe"},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMetadataJsonlAdapterValidate:
    def test_validate_source_passes_when_files_present(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        
        # Create a mock all_1024 folder and metadata.jsonl file
        all_dir = data_root / "all_1024"
        all_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = all_dir / "metadata.jsonl"
        metadata_file.write_text('{"file_name": "levle0_1.jpg", "prompt": "photo of acne0"}', encoding="utf-8")

        adapter = MetadataJsonlAdapter(data_root, cfg)
        result = adapter.validate_source()
        assert result is True

    def test_validate_source_raises_when_all_folder_missing(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        adapter = MetadataJsonlAdapter(data_root, cfg)
        # all_1024 directory is not created, should raise DatasetNotFoundError
        with pytest.raises(DatasetNotFoundError):
            adapter.validate_source()

    def test_validate_source_raises_when_jsonl_missing(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        
        # Create all_1024 but no metadata.jsonl
        all_dir = data_root / "all_1024"
        all_dir.mkdir(parents=True, exist_ok=True)

        adapter = MetadataJsonlAdapter(data_root, cfg)
        with pytest.raises(DatasetNotFoundError):
            adapter.validate_source()


class TestMetadataJsonlAdapterParse:
    def test_parse_yields_correct_records(self, tmp_project: Path) -> None:
        cfg = _make_config()
        data_root = tmp_project / "data" / "raw" / "ACNE04"
        
        # Create all_1024 and metadata.jsonl
        all_dir = data_root / "all_1024"
        all_dir.mkdir(parents=True, exist_ok=True)
        
        lines = [
            '{"file_name": "levle0_1.jpg", "prompt": "photo of acne0"}',
            '{"file_name": "levle1_10.jpg", "prompt": "photo of acne1"}',
            '{"file_name": "levle3_99.jpg", "prompt": "photo of acne3"}',
            '# this is a comment or empty line',
            '{"file_name": "invalid_name.jpg", "prompt": "photo of acne1"}', # starts without levle -> skipped
        ]
        metadata_file = all_dir / "metadata.jsonl"
        metadata_file.write_text("\n".join(lines), encoding="utf-8")

        adapter = MetadataJsonlAdapter(data_root, cfg)
        records = list(adapter.parse())
        
        assert len(records) == 3
        
        assert records[0].image_id == "all_1024__levle0_1"
        assert records[0].severity_label == 0
        assert records[0].image_path.name == "levle0_1.jpg"
        
        assert records[1].image_id == "all_1024__levle1_10"
        assert records[1].severity_label == 1
        
        assert records[2].image_id == "all_1024__levle3_99"
        assert records[2].severity_label == 2

        # Path must be absolute
        assert records[0].image_path.is_absolute()


class TestMetadataJsonlAdapterFormatName:
    def test_format_name(self) -> None:
        cfg = _make_config()
        adapter = MetadataJsonlAdapter(Path("."), cfg)
        name = adapter.format_name
        assert isinstance(name, str)
        assert len(name) > 0
        assert "metadata_jsonl" in name.lower()

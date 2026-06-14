"""
phase0/tests/test_eda.py
Tests for the EDA module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest_df(n_per_class: int = 5) -> pd.DataFrame:
    """Build a synthetic manifest DataFrame for testing."""
    rows = []
    for sev in range(4):
        for i in range(n_per_class):
            rows.append(
                {
                    "image_id": f"img_{sev}_{i:03d}",
                    "image_path": f"/fake/acne{sev}_1024/img_{sev}_{i:03d}.jpg",
                    "severity_label": sev,
                    "severity_name": ["mild", "moderate", "severe", "very_severe"][sev],
                    "width": 1024 if sev < 3 else 512,
                    "height": 1024 if sev < 3 else 512,
                    "format": "JPEG",
                    "file_size_bytes": 100_000,
                }
            )
    # Make one class much larger for IR test
    for i in range(15):
        rows.append(
            {
                "image_id": f"img_0_extra_{i:03d}",
                "image_path": f"/fake/acne0_1024/img_0_extra_{i:03d}.jpg",
                "severity_label": 0,
                "severity_name": "mild",
                "width": 1024,
                "height": 1024,
                "format": "JPEG",
                "file_size_bytes": 100_000,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassDistribution:
    def test_class_distribution_computes_ir(self, tmp_project: Path) -> None:
        """IR should equal max_count / min_count across severity classes."""
        from phase0.src.eda import DatasetAnalyser

        df = _make_manifest_df(n_per_class=5)
        outputs_root = tmp_project / "data" / "phase0_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        manifest_path = outputs_root / "manifest.csv"
        df.to_csv(manifest_path, index=False)

        config = {
            "paths": {
                "outputs_root": "data/phase0_outputs",
                "figures_dir": "reports/phase0/figures",
            },
            "ingestion": {
                "severity_map": {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"},
            },
            "quality": {
                "min_image_dimension": 224,
            }
        }

        # Patch matplotlib to avoid GUI window/saving issues
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            analyser = DatasetAnalyser(manifest_path, config, tmp_project)
            result = analyser.analyse_class_distribution()

        counts = df.groupby("severity_label").size()
        expected_ir = counts.max() / counts.min()

        assert "imbalance_ratio" in result
        assert abs(result["imbalance_ratio"] - expected_ir) < 1e-6

    def test_class_distribution_saves_plot(self, tmp_project: Path) -> None:
        """After computing distribution, a PNG plot should be saved."""
        from phase0.src.eda import DatasetAnalyser

        df = _make_manifest_df()
        outputs_root = tmp_project / "data" / "phase0_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        manifest_path = outputs_root / "manifest.csv"
        df.to_csv(manifest_path, index=False)

        config = {
            "paths": {
                "outputs_root": "data/phase0_outputs",
                "figures_dir": "reports/phase0/figures",
            },
            "ingestion": {
                "severity_map": {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"},
            },
            "quality": {
                "min_image_dimension": 224,
            }
        }

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            analyser = DatasetAnalyser(manifest_path, config, tmp_project)
            analyser.analyse_class_distribution()

        png_files = list(analyser.figures_dir.glob("*.png"))
        assert len(png_files) >= 1, "No PNG plot was saved"


class TestResolutionAnalysis:
    def test_resolution_analysis_returns_stats(self, tmp_project: Path) -> None:
        """Resolution analysis result must contain width_stats keys."""
        from phase0.src.eda import DatasetAnalyser

        outputs_root = tmp_project / "data" / "phase0_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)

        rows = []
        for sev in range(4):
            for i in range(3):
                img_path = tmp_project / f"img_{sev}_{i}.jpg"
                from PIL import Image
                import numpy as np
                Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)

                rows.append({
                    "image_id": f"img_{sev}_{i}",
                    "image_path": str(img_path),
                    "severity_label": sev,
                    "severity_name": ["mild", "moderate", "severe", "very_severe"][sev],
                })
        df = pd.DataFrame(rows)
        manifest_path = outputs_root / "manifest.csv"
        df.to_csv(manifest_path, index=False)

        config = {
            "paths": {
                "outputs_root": "data/phase0_outputs",
                "figures_dir": "reports/phase0/figures",
            },
            "ingestion": {
                "severity_map": {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"},
            },
            "quality": {
                "min_image_dimension": 50,
            }
        }

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            analyser = DatasetAnalyser(manifest_path, config, tmp_project)
            result = analyser.analyse_resolution()

        assert "width_stats" in result
        width_stats = result["width_stats"]
        for key in ("mean", "min", "max", "median"):
            assert key in width_stats, f"Missing key '{key}' in width_stats"


class TestRunEda:
    def test_run_saves_eda_stats_json(self, tmp_project: Path) -> None:
        """run_eda() should produce eda_stats.json in the outputs directory."""
        import yaml
        from phase0.src.eda import run_eda

        outputs_root = tmp_project / "data" / "phase0_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)

        rows = []
        for sev in range(4):
            for i in range(3):
                img_path = tmp_project / f"img_{sev}_{i}.jpg"
                from PIL import Image
                import numpy as np
                Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)

                rows.append({
                    "image_id": f"img_{sev}_{i}",
                    "image_path": str(img_path),
                    "severity_label": sev,
                    "severity_name": ["mild", "moderate", "severe", "very_severe"][sev],
                })
        df = pd.DataFrame(rows)
        df.to_csv(outputs_root / "manifest.csv", index=False)

        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
            run_eda(config_path, tmp_project)

        eda_stats_path = outputs_root / "eda_stats.json"
        assert eda_stats_path.exists(), "eda_stats.json was not created"

        with open(eda_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        assert "class_distribution" in stats or len(stats) > 0

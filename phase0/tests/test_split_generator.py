"""
phase0/tests/test_split_generator.py
Tests for the split_generator module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest_and_clusters(
    n_per_class: int = 20,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic manifest and cluster_assignments DataFrames."""
    rng = np.random.default_rng(seed)

    manifest_rows = []
    cluster_rows = []
    cluster_id = 0
    img_counter = 0

    for sev in range(n_classes):
        # Group images into clusters of ~3
        for group in range(n_per_class // 3):
            cluster_id += 1
            for k in range(3):
                iid = f"img_{img_counter:05d}"
                img_counter += 1
                manifest_rows.append(
                    {
                        "image_id": iid,
                        "image_path": f"/fake/acne{sev}_1024/{iid}.jpg",
                        "severity_label": sev,
                        "severity_name": ["mild", "moderate", "severe"][sev],
                        "width": 1024,
                        "height": 1024,
                        "format": "JPEG",
                        "file_size_bytes": 100_000,
                        "quality_pass": True,
                    }
                )
                cluster_rows.append({
                    "image_id": iid,
                    "cluster_id": cluster_id,
                    "cluster_size": 3,
                    "embedding_extracted": True,
                })

        # Add some singletons (cluster_id = -1)
        for _ in range(2):
            iid = f"img_{img_counter:05d}"
            img_counter += 1
            manifest_rows.append(
                {
                    "image_id": iid,
                    "image_path": f"/fake/acne{sev}_1024/{iid}.jpg",
                    "severity_label": sev,
                    "severity_name": ["mild", "moderate", "severe"][sev],
                    "width": 1024,
                    "height": 1024,
                    "format": "JPEG",
                    "file_size_bytes": 100_000,
                    "quality_pass": True,
                }
            )
            cluster_rows.append({
                "image_id": iid,
                "cluster_id": -1,
                "cluster_size": 1,
                "embedding_extracted": False,
            })

    return pd.DataFrame(manifest_rows), pd.DataFrame(cluster_rows)


def _write_inputs(
    tmp_project: Path,
    manifest_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> None:
    outputs = tmp_project / "data" / "phase0_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(outputs / "quality_audit.csv", index=False)
    cluster_df.to_csv(outputs / "cluster_assignments.csv", index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitCoverage:
    def test_splits_cover_all_images(self, tmp_project: Path) -> None:
        """train + val + test should contain every image_id exactly once."""
        from phase0.src.split_generator import run_split_generation

        manifest_df, cluster_df = _make_manifest_and_clusters()
        _write_inputs(tmp_project, manifest_df, cluster_df)
        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        run_split_generation(config_path, tmp_project)

        splits_dir = tmp_project / "data" / "phase0_outputs" / "splits"
        all_ids: list[str] = []
        for split_name in ("train", "val", "test"):
            split_file = splits_dir / f"{split_name}.csv"
            assert split_file.exists(), f"{split_name}.csv missing"
            df = pd.read_csv(split_file)
            all_ids.extend(df["image_id"].tolist())

        assert sorted(all_ids) == sorted(manifest_df["image_id"].tolist()), (
            "Not all images appear exactly once across splits"
        )

    def test_no_duplicate_images_across_splits(self, tmp_project: Path) -> None:
        """No image_id should appear in more than one split."""
        from phase0.src.split_generator import run_split_generation

        manifest_df, cluster_df = _make_manifest_and_clusters()
        _write_inputs(tmp_project, manifest_df, cluster_df)
        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        run_split_generation(config_path, tmp_project)

        splits_dir = tmp_project / "data" / "phase0_outputs" / "splits"
        split_id_sets: dict[str, set] = {}
        for split_name in ("train", "val", "test"):
            df = pd.read_csv(splits_dir / f"{split_name}.csv")
            split_id_sets[split_name] = set(df["image_id"].tolist())

        train_val_overlap = split_id_sets["train"] & split_id_sets["val"]
        train_test_overlap = split_id_sets["train"] & split_id_sets["test"]
        val_test_overlap = split_id_sets["val"] & split_id_sets["test"]

        assert not train_val_overlap, f"train/val overlap: {train_val_overlap}"
        assert not train_test_overlap, f"train/test overlap: {train_test_overlap}"
        assert not val_test_overlap, f"val/test overlap: {val_test_overlap}"


class TestClusterLeakage:
    def test_no_cluster_leakage(self, tmp_project: Path) -> None:
        """No cluster_id (excluding -1) should appear in two different splits."""
        from phase0.src.split_generator import run_split_generation

        manifest_df, cluster_df = _make_manifest_and_clusters()
        _write_inputs(tmp_project, manifest_df, cluster_df)
        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        run_split_generation(config_path, tmp_project)

        splits_dir = tmp_project / "data" / "phase0_outputs" / "splits"
        # Build a merged table: image_id → split
        all_rows = []
        for split_name in ("train", "val", "test"):
            df = pd.read_csv(splits_dir / f"{split_name}.csv")
            df["split"] = split_name
            all_rows.append(df)

        merged = pd.concat(all_rows, ignore_index=True)

        # Check: for each cluster_id != -1, all images are in the same split
        for cid, group in merged[merged["cluster_id"] != -1].groupby("cluster_id"):
            splits_for_cluster = group["split"].unique()
            assert len(splits_for_cluster) == 1, (
                f"Cluster {cid} appears in multiple splits: {splits_for_cluster}"
            )


class TestSplitRatios:
    def test_split_ratios_approximate_targets(self, tmp_project: Path) -> None:
        """Each split's fraction should be within 5% of the configured target."""
        from phase0.src.split_generator import run_split_generation

        manifest_df, cluster_df = _make_manifest_and_clusters(n_per_class=30)
        _write_inputs(tmp_project, manifest_df, cluster_df)
        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        run_split_generation(config_path, tmp_project)

        splits_dir = tmp_project / "data" / "phase0_outputs" / "splits"
        counts = {}
        for split_name in ("train", "val", "test"):
            df = pd.read_csv(splits_dir / f"{split_name}.csv")
            counts[split_name] = len(df)

        total = sum(counts.values())
        targets = {"train": 0.70, "val": 0.15, "test": 0.15}
        tolerance = 0.05  # 5%

        for split_name, target in targets.items():
            actual = counts[split_name] / total
            assert abs(actual - target) <= tolerance, (
                f"{split_name} ratio {actual:.3f} deviates more than {tolerance} from {target}"
            )


class TestSplitSummaryJson:
    def test_split_summary_json_created(self, tmp_project: Path) -> None:
        """run_split_generation() must produce split_summary.json."""
        from phase0.src.split_generator import run_split_generation

        manifest_df, cluster_df = _make_manifest_and_clusters()
        _write_inputs(tmp_project, manifest_df, cluster_df)
        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        run_split_generation(config_path, tmp_project)

        summary_path = tmp_project / "data" / "phase0_outputs" / "splits" / "split_summary.json"
        assert summary_path.exists(), "split_summary.json was not created"

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert isinstance(summary, dict)


class TestSingletonClusters:
    def test_singleton_clusters_distributed(self, tmp_project: Path) -> None:
        """Images with cluster_id=-1 (singletons) should appear in splits."""
        from phase0.src.split_generator import run_split_generation

        manifest_df, cluster_df = _make_manifest_and_clusters()
        _write_inputs(tmp_project, manifest_df, cluster_df)
        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        run_split_generation(config_path, tmp_project)

        splits_dir = tmp_project / "data" / "phase0_outputs" / "splits"
        all_ids_in_splits: set[str] = set()
        for split_name in ("train", "val", "test"):
            df = pd.read_csv(splits_dir / f"{split_name}.csv")
            all_ids_in_splits.update(df["image_id"].tolist())

        # All singleton images should be in some split
        singleton_ids = set(cluster_df[cluster_df["cluster_id"] == -1]["image_id"])
        missing = singleton_ids - all_ids_in_splits
        assert not missing, f"Singleton images not in any split: {missing}"

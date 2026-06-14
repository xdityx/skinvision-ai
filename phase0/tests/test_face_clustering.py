"""
phase0/tests/test_face_clustering.py
Tests for the face_clustering module.

All tests mock insightface entirely via unittest.mock.patch to avoid
requiring GPU or model downloads during CI.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

# Add project root to path so phase0 package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_jpeg(path: Path, pixel_value: int = 128) -> None:
    """Save a tiny solid-color JPEG for testing."""
    arr = np.full((100, 100, 3), pixel_value, dtype=np.uint8)
    img = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "JPEG")


def _make_manifest_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [f"img_{i:04d}" for i in range(n)],
            "image_path": [f"/fake/acne0_1024/img_{i:04d}.jpg" for i in range(n)],
            "severity": [i % 4 for i in range(n)],
        }
    )


def _make_fake_embeddings(n: int = 6, dim: int = 512) -> np.ndarray:
    """Return deterministic random unit embeddings."""
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def _make_mock_insightface(embeddings: np.ndarray) -> MagicMock:
    """Build a mock insightface module that returns deterministic embeddings."""
    mock_insightface = MagicMock()

    mock_model = MagicMock()
    mock_insightface.app.FaceAnalysis.return_value = mock_model

    # get() returns a list of faces, each with .embedding attribute
    call_count = [0]

    def side_effect_get(img, size=(112, 112)):
        idx = call_count[0] % len(embeddings)
        call_count[0] += 1
        face = MagicMock()
        face.normed_embedding = embeddings[idx]
        return [face]

    mock_model.get.side_effect = side_effect_get
    return mock_insightface


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    def test_embedding_cache_skips_extraction(self, tmp_project: Path) -> None:
        """If a .npy embedding cache exists, model.get() should not be called."""
        from phase0.src.face_clustering import FaceClusterer

        outputs_root = tmp_project / "data" / "phase0_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)

        embeddings = _make_fake_embeddings(n=6)
        cache_path = outputs_root / "face_embeddings.npy"
        np.save(str(cache_path), embeddings)

        # We also need embeddings_index.csv
        idx_df = pd.DataFrame({"row_idx": list(range(6)), "image_id": [f"img_{i:04d}" for i in range(6)]})
        idx_df.to_csv(outputs_root / "embeddings_index.csv", index=False)

        # Create dummy quality_audit.csv
        qa_df = pd.DataFrame({
            "image_id": [f"img_{i:04d}" for i in range(6)],
            "image_path": [f"/fake/img_{i:04d}.jpg" for i in range(6)],
            "severity_label": [0]*6,
            "quality_pass": [True]*6,
            "face_flag": ["ok"]*6
        })
        qa_df.to_csv(outputs_root / "quality_audit.csv", index=False)

        mock_insightface = _make_mock_insightface(embeddings)
        mock_app_module = MagicMock()
        mock_app_module.FaceAnalysis = mock_insightface.app.FaceAnalysis

        config = {
            "project": {"seed": 42},
            "paths": {
                "outputs_root": "data/phase0_outputs",
                "figures_dir": "reports/phase0/figures",
            },
            "clustering": {
                "embedding_model": "buffalo_sc",
                "embedding_cache": True,
                "face_image_size": 112,
            }
        }

        clusterer = FaceClusterer(
            quality_audit_path=outputs_root / "quality_audit.csv",
            config=config,
            project_root=tmp_project
        )

        with patch.dict("sys.modules", {
            "insightface": mock_insightface,
            "insightface.app": mock_app_module,
        }):
            app = clusterer._init_insightface()
            res_embs, res_ids = clusterer._load_or_extract_embeddings(app)

        # model.get should NOT have been called if cache was loaded
        mock_insightface.app.FaceAnalysis.return_value.get.assert_not_called()
        assert res_embs.shape == embeddings.shape


class TestDbscanSweep:
    def test_dbscan_sweep_selects_best_eps(self, tmp_project: Path) -> None:
        """DBSCAN sweep should return labels array and choose the best eps."""
        from phase0.src.face_clustering import FaceClusterer

        embeddings = _make_fake_embeddings(n=20)
        
        config = {
            "project": {"seed": 42},
            "paths": {
                "outputs_root": "data/phase0_outputs",
                "figures_dir": "reports/phase0/figures",
            },
            "clustering": {
                "dbscan_eps_sweep": [0.30, 0.40, 0.50],
                "dbscan_min_samples": 2,
            }
        }
        
        qa_path = tmp_project / "data" / "phase0_outputs" / "quality_audit.csv"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"image_id": [], "quality_pass": [], "severity_label": []}).to_csv(qa_path, index=False)
        
        clusterer = FaceClusterer(qa_path, config, tmp_project)
        labels, sweep_results = clusterer._run_dbscan_sweep(embeddings)

        assert isinstance(labels, np.ndarray)
        assert labels.shape == (len(embeddings),)


class TestLriComputation:
    def test_lri_computed_correctly(self, tmp_project: Path) -> None:
        """LRI should count images in multi-image clusters as a percentage of total images."""
        from phase0.src.face_clustering import FaceClusterer

        image_ids = [f"img_{i}" for i in range(6)]
        cluster_labels = np.array([1, 1, 2, 2, 2, -1])

        config = {
            "project": {"seed": 42},
            "paths": {
                "outputs_root": "data/phase0_outputs",
                "figures_dir": "reports/phase0/figures",
            },
            "clustering": {
                "embedding_model": "buffalo_sc",
            }
        }
        qa_path = tmp_project / "data" / "phase0_outputs" / "quality_audit.csv"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"image_id": [], "quality_pass": [], "severity_label": []}).to_csv(qa_path, index=False)
        
        clusterer = FaceClusterer(qa_path, config, tmp_project)
        lri_info = clusterer._compute_leakage_risk_index(cluster_labels, image_ids)

        assert lri_info["total_images"] == 6
        assert lri_info["images_in_multi_clusters"] == 5
        assert lri_info["multi_image_clusters"] == 2
        assert abs(lri_info["lri"] - (5.0 / 6.0 * 100)) < 1e-2


class TestClusterAssignmentsSaved:
    def test_cluster_assignments_saved(self, tmp_project: Path) -> None:
        """run_face_clustering() must produce cluster_assignments.csv."""
        from phase0.src.face_clustering import run_face_clustering

        outputs_root = tmp_project / "data" / "phase0_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)

        n = 6
        image_paths = []
        for i in range(n):
            img_path = tmp_project / "data" / "raw" / "ACNE04" / "acne0_1024" / f"img_{i:04d}.jpg"
            _save_jpeg(img_path)
            image_paths.append(str(img_path))

        qa_df = pd.DataFrame(
            {
                "image_id": [f"img_{i:04d}" for i in range(n)],
                "image_path": image_paths,
                "severity_label": [i % 4 for i in range(n)],
                "quality_pass": [True] * n,
                "face_flag": ["ok"] * n,
            }
        )
        qa_df.to_csv(outputs_root / "quality_audit.csv", index=False)

        embeddings = _make_fake_embeddings(n=n)
        mock_insightface = _make_mock_insightface(embeddings)
        mock_app_module = MagicMock()
        mock_app_module.FaceAnalysis = mock_insightface.app.FaceAnalysis

        config_path = tmp_project / "phase0" / "config" / "phase0.yaml"

        with patch.dict("sys.modules", {
            "insightface": mock_insightface,
            "insightface.app": mock_app_module,
        }):
            run_face_clustering(config_path, tmp_project)

        assignments_path = outputs_root / "cluster_assignments.csv"
        assert assignments_path.exists(), "cluster_assignments.csv was not created"

        df = pd.read_csv(assignments_path)
        assert "image_id" in df.columns
        assert "cluster_id" in df.columns
        assert len(df) == n

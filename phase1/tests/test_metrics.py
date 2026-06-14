"""
phase1/tests/test_metrics.py
Tests for the metrics module.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_perfect_predictions(self) -> None:
        from phase1.src.metrics import compute_metrics
        y = [0, 0, 1, 1, 2, 2]
        m = compute_metrics(y, y, num_classes=3)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["qwk"] == pytest.approx(1.0)

    def test_random_predictions_qwk_near_zero(self) -> None:
        """QWK for random predictions should be near 0 (not necessarily exact 0)."""
        from phase1.src.metrics import compute_metrics
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 3, size=200).tolist()
        y_pred = rng.integers(0, 3, size=200).tolist()
        m = compute_metrics(y_true, y_pred, num_classes=3)
        assert -0.3 < m["qwk"] < 0.3

    def test_confusion_matrix_shape(self) -> None:
        from phase1.src.metrics import compute_metrics
        y = [0, 1, 2, 0, 1, 2]
        m = compute_metrics(y, y, num_classes=3)
        cm = np.array(m["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_per_class_accuracy_perfect(self) -> None:
        from phase1.src.metrics import compute_metrics
        y = [0, 0, 1, 1, 2, 2]
        m = compute_metrics(y, y, num_classes=3, class_names=["a", "b", "c"])
        assert m["per_class_accuracy"]["a"] == pytest.approx(1.0)
        assert m["per_class_accuracy"]["b"] == pytest.approx(1.0)
        assert m["per_class_accuracy"]["c"] == pytest.approx(1.0)

    def test_macro_f1_imbalanced(self) -> None:
        """macro-F1 should penalise poor minority class performance."""
        from phase1.src.metrics import compute_metrics
        # Always predict class 1 (the majority)
        y_true = [0] * 5 + [1] * 50 + [2] * 5
        y_pred = [1] * 60
        m = compute_metrics(y_true, y_pred, num_classes=3)
        # macro_f1 should be much lower than accuracy
        assert m["macro_f1"] < m["accuracy"]

    def test_qwk_adjacent_errors_better_than_distant(self) -> None:
        """QWK penalises distant ordinal mistakes more than adjacent ones."""
        from phase1.src.metrics import compute_metrics
        # Off-by-one errors
        y_true = [0, 1, 2]
        y_adjacent = [1, 2, 1]     # all off by 1
        y_distant = [2, 0, 0]      # max distance errors
        m_adj = compute_metrics(y_true, y_adjacent, num_classes=3)
        m_dist = compute_metrics(y_true, y_distant, num_classes=3)
        assert m_adj["qwk"] > m_dist["qwk"]


class TestPlotConfusionMatrix:
    """Test that confusion matrix PNG is saved without errors."""

    def test_saves_file(self, tmp_path) -> None:
        from phase1.src.metrics import plot_confusion_matrix
        cm = [[10, 2, 0], [1, 15, 3], [0, 1, 8]]
        out = plot_confusion_matrix(
            cm, ["mild", "moderate", "severe"], tmp_path / "cm.png"
        )
        assert out.exists()
        assert out.stat().st_size > 0


class TestSaveMetricsJson:
    """Test JSON serialisation of metrics dict."""

    def test_saves_json(self, tmp_path) -> None:
        import json
        from phase1.src.metrics import save_metrics_json
        metrics = {"accuracy": 0.8, "qwk": 0.65, "confusion_matrix": [[1, 2], [3, 4]]}
        out = save_metrics_json(metrics, tmp_path / "metrics.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["accuracy"] == pytest.approx(0.8)

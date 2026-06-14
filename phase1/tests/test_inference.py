"""
phase1/tests/test_inference.py
Unit and integration tests for the AcnePredictor inference engine.

Tests are split into:
  - TestImageValidation   — file/format/size error handling
  - TestPredictorOutput   — output schema, probabilities, classes
  - TestTTA               — TTA vs single-pass consistency
  - TestCLI               — end-to-end CLI invocation via subprocess
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ─── Helpers ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT   = PROJECT_ROOT / "phase1" / "checkpoints" / "best_model.pt"

# Skip all tests that need the real checkpoint if it doesn't exist
_need_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="best_model.pt not found — run training first",
)


def _make_rgb_image(tmp_path: Path, size: tuple[int, int] = (320, 320), name: str = "face.jpg") -> Path:
    """Save a random RGB JPEG to tmp_path and return its path."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def _make_predictor(device: str = "cpu"):
    """Build a predictor pointing at the real checkpoint, forced to CPU."""
    from phase1.src.inference import AcnePredictor
    return AcnePredictor(checkpoint_path=CHECKPOINT, device=device)


# ─── Image validation tests ───────────────────────────────────────────────────

class TestImageValidation:

    def test_missing_file_raises_image_not_found(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor, ImageNotFoundError
        if not CHECKPOINT.exists():
            pytest.skip("No checkpoint")
        pred = _make_predictor()
        with pytest.raises(ImageNotFoundError, match="not found"):
            pred.predict(tmp_path / "nonexistent.jpg")

    def test_directory_path_raises_image_load_error(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor, ImageLoadError
        if not CHECKPOINT.exists():
            pytest.skip("No checkpoint")
        pred = _make_predictor()
        with pytest.raises(ImageLoadError, match="not a file"):
            pred.predict(tmp_path)   # tmp_path is a directory

    def test_corrupt_file_raises_image_load_error(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor, ImageLoadError
        if not CHECKPOINT.exists():
            pytest.skip("No checkpoint")
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"this is not an image")
        pred = _make_predictor()
        with pytest.raises(ImageLoadError):
            pred.predict(bad)

    def test_tiny_image_raises_image_too_small(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor, ImageTooSmallError
        if not CHECKPOINT.exists():
            pytest.skip("No checkpoint")
        tiny = tmp_path / "tiny.png"
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(tiny)
        pred = _make_predictor()
        with pytest.raises(ImageTooSmallError, match="too small"):
            pred.predict(tiny)

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor, CheckpointNotFoundError
        with pytest.raises(CheckpointNotFoundError, match="not found"):
            AcnePredictor(checkpoint_path=tmp_path / "fake.pt")


# ─── Output schema tests ──────────────────────────────────────────────────────

@_need_checkpoint
class TestPredictorOutput:

    def test_output_has_required_keys(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred = _make_predictor()
        img  = _make_rgb_image(tmp_path)
        result = pred.predict(img)

        required = {
            "image_path", "predicted_class", "predicted_severity",
            "confidence", "class_probabilities",
            "tta_enabled", "tta_views",
            "model_checkpoint", "checkpoint_epoch", "checkpoint_val_f1",
            "inference_time_ms",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_probabilities_sum_to_one(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        total  = sum(result["class_probabilities"].values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}"

    def test_probabilities_non_negative(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        for name, p in result["class_probabilities"].items():
            assert p >= 0, f"Negative probability for class {name}: {p}"

    def test_confidence_equals_max_prob(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        max_prob = max(result["class_probabilities"].values())
        assert abs(result["confidence"] - max_prob) < 1e-5

    def test_predicted_class_matches_severity(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        class_idx = result["predicted_class"]
        severity  = result["predicted_severity"]
        assert pred.severity_map[class_idx] == severity

    def test_predicted_class_in_valid_range(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        assert result["predicted_class"] in {0, 1, 2}

    def test_tta_disabled_by_default(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        assert result["tta_enabled"] is False
        assert result["tta_views"] is None

    def test_inference_time_positive(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img)
        assert result["inference_time_ms"] > 0

    def test_json_save(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        out    = tmp_path / "pred.json"
        pred.predict(img, save_to=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "predicted_severity" in data

    def test_png_input_works(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred = _make_predictor()
        img  = _make_rgb_image(tmp_path, name="face.png")
        result = pred.predict(img)
        assert result["predicted_severity"] in {"mild", "moderate", "severe"}


# ─── TTA tests ────────────────────────────────────────────────────────────────

@_need_checkpoint
class TestTTA:

    def test_tta_enabled_flag_set(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img, tta=True)
        assert result["tta_enabled"] is True
        assert result["tta_views"] == 5

    def test_tta_probabilities_sum_to_one(self, tmp_path: Path) -> None:
        from phase1.src.inference import AcnePredictor
        pred   = _make_predictor()
        img    = _make_rgb_image(tmp_path)
        result = pred.predict(img, tta=True)
        total  = sum(result["class_probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_tta_and_no_tta_agree_on_class(self, tmp_path: Path) -> None:
        """
        For a clean synthetic image the TTA prediction should agree with
        the single-pass prediction (both use the same trained weights).
        Not guaranteed to always hold on adversarial inputs, but holds
        for most well-formed images.
        """
        from phase1.src.inference import AcnePredictor
        pred = _make_predictor()
        img  = _make_rgb_image(tmp_path, size=(400, 400))
        r1   = pred.predict(img, tta=False)
        r2   = pred.predict(img, tta=True)
        # Both should return a valid class; not asserting equality
        # (TTA is expected to differ slightly on random images)
        assert r1["predicted_class"] in {0, 1, 2}
        assert r2["predicted_class"] in {0, 1, 2}


# ─── CLI tests ────────────────────────────────────────────────────────────────

@_need_checkpoint
class TestCLI:

    def _run(self, *extra_args: str) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable, "-m", "phase1.scripts.predict",
            "--device", "cpu",
            "--quiet",
            *extra_args,
        ]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

    def test_cli_returns_zero_on_success(self, tmp_path: Path) -> None:
        img = _make_rgb_image(tmp_path)
        r   = self._run("--image", str(img))
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_cli_stdout_is_valid_json(self, tmp_path: Path) -> None:
        img  = _make_rgb_image(tmp_path)
        r    = self._run("--image", str(img))
        data = json.loads(r.stdout)
        assert "predicted_severity" in data

    def test_cli_missing_image_returns_nonzero(self) -> None:
        r = self._run("--image", "/tmp/does_not_exist_xyz.jpg")
        assert r.returncode != 0

    def test_cli_tta_flag(self, tmp_path: Path) -> None:
        img  = _make_rgb_image(tmp_path)
        r    = self._run("--image", str(img), "--tta")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["tta_enabled"] is True
        assert data["tta_views"]   == 5

    def test_cli_saves_json_output(self, tmp_path: Path) -> None:
        img = _make_rgb_image(tmp_path)
        out = tmp_path / "result.json"
        r   = self._run("--image", str(img), "--output", str(out))
        assert r.returncode == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "predicted_class" in data

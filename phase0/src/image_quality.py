"""
phase0/src/image_quality.py
============================
Image quality audit for the ACNE04 dataset.

Provides ImageQualityAuditor, which examines every image in the manifest
for corruption, blur, exposure problems, and face-detection quality,
then writes a quality_audit.csv and summary visualisations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from phase0.src.utils.logging import get_logger
from phase0.src.utils.io import safe_load_pil, safe_load_cv2
from phase0.src.utils.visualisation import save_bar_chart, save_kde_plot


class ImageQualityAuditor:
    """Audit image quality across the full ACNE04 manifest."""

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        manifest_path: Path,
        config: dict,
        project_root: Path,
    ) -> None:
        """
        Parameters
        ----------
        manifest_path : Path
            Absolute path to ``manifest.csv``.
        config : dict
            Full Phase-0 config dict (parsed from ``phase0.yaml``).
        project_root : Path
            Absolute path to the project root directory.
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.project_root = project_root

        # ── Manifest ──────────────────────────────────────────────────── #
        self.manifest_path = manifest_path
        self.df = pd.read_csv(manifest_path)
        self.logger.info(
            "Loaded manifest with %d rows from %s", len(self.df), manifest_path
        )

        # ── Quality thresholds ─────────────────────────────────────────── #
        q = config["quality"]
        self._blur_threshold: float = float(q["blur_threshold"])
        self._underexposed_threshold: float = float(q["underexposed_threshold"])
        self._overexposed_threshold: float = float(q["overexposed_threshold"])
        self._low_contrast_threshold: float = float(q["low_contrast_threshold"])
        self._face_confidence_threshold: float = float(q["face_confidence_threshold"])
        self._face_model_selection: int = int(q["face_model_selection"])

        # ── Output directories ─────────────────────────────────────────── #
        self.figures_dir: Path = project_root / config["paths"]["figures_dir"]
        self.outputs_root: Path = project_root / config["paths"]["outputs_root"]
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

        # ── Severity metadata ──────────────────────────────────────────── #
        raw_map: dict = config["ingestion"]["severity_map"]
        self._severity_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}
        self._severity_names: list[str] = [
            self._severity_map[i] for i in sorted(self._severity_map)
        ]

        # ── MediaPipe face detector (lazy import, single instance) ─────── #
        import mediapipe as mp  # noqa: PLC0415  — intentional lazy import

        self._mp_face_detection = mp.solutions.face_detection
        self._face_detector = self._mp_face_detection.FaceDetection(
            model_selection=self._face_model_selection,
            min_detection_confidence=0.3,  # low threshold; we filter ourselves
        ).__enter__()

        self.logger.info(
            "MediaPipe FaceDetection initialised  "
            "(model_selection=%d, quality threshold=%.2f)",
            self._face_model_selection,
            self._face_confidence_threshold,
        )

    # ------------------------------------------------------------------ #
    # Core audit logic                                                     #
    # ------------------------------------------------------------------ #

    def audit_single_image(self, row: pd.Series) -> dict:
        """Compute quality metrics for a single image row from the manifest.

        Parameters
        ----------
        row : pd.Series
            One row of the manifest DataFrame.

        Returns
        -------
        dict
            Quality metric dictionary (see module docstring for full schema).
        """
        import cv2  # noqa: PLC0415

        filepath_raw = str(row["image_path"])
        img_path = Path(filepath_raw)
        if not img_path.is_absolute():
            img_path = self.project_root / img_path

        # ── Defaults (used on error) ────────────────────────────────────── #
        result: dict[str, Any] = {
            "is_corrupted": False,
            "blur_score": -1.0,
            "is_blurry": False,
            "mean_intensity": -1.0,
            "std_intensity": -1.0,
            "exposure_flag": "ok",
            "face_count": -1,
            "face_bbox": "null",
            "face_confidence": -1.0,
            "face_flag": "detection_failed",
            "quality_pass": False,
        }

        # ── Step 1: Corruption check ───────────────────────────────────── #
        pil_img = safe_load_pil(img_path)  # returns None if unreadable/corrupt
        cv_img = safe_load_cv2(img_path)   # returns None if unreadable/corrupt

        if pil_img is None or cv_img is None:
            result["is_corrupted"] = True
            result["face_flag"] = "detection_failed"
            result["quality_pass"] = False
            return result

        # ── Step 2: Blur (Laplacian variance) ─────────────────────────── #
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result["blur_score"] = round(blur_score, 4)
        result["is_blurry"] = blur_score < self._blur_threshold

        # ── Step 3: Exposure / contrast ───────────────────────────────── #
        mean_intensity = float(gray.mean())
        std_intensity = float(gray.std())
        result["mean_intensity"] = round(mean_intensity, 4)
        result["std_intensity"] = round(std_intensity, 4)

        if mean_intensity < self._underexposed_threshold:
            exposure_flag = "underexposed"
        elif mean_intensity > self._overexposed_threshold:
            exposure_flag = "overexposed"
        elif std_intensity < self._low_contrast_threshold:
            exposure_flag = "low_contrast"
        else:
            exposure_flag = "ok"
        result["exposure_flag"] = exposure_flag

        # ── Step 4: Face detection via MediaPipe ──────────────────────── #
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        try:
            detection_result = self._face_detector.process(rgb_img)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("MediaPipe process() raised %s for %s", exc, img_path)
            result["face_flag"] = "detection_failed"
            result["quality_pass"] = False
            return result

        detections = (
            detection_result.detections
            if detection_result and detection_result.detections
            else []
        )

        if not detections:
            result["face_count"] = 0
            result["face_flag"] = "no_face"
            result["face_confidence"] = -1.0
            result["face_bbox"] = "null"
        else:
            # Filter by our threshold and get confidences
            confident = [
                d
                for d in detections
                if _detection_score(d) >= self._face_confidence_threshold
            ]
            result["face_count"] = len(detections)

            # Primary detection = highest confidence
            primary = max(detections, key=lambda d: _detection_score(d))
            max_conf = _detection_score(primary)
            result["face_confidence"] = round(max_conf, 4)

            # Bounding box (relative coords, already 0-1)
            bbox = primary.location_data.relative_bounding_box
            result["face_bbox"] = json.dumps(
                [
                    round(bbox.xmin, 6),
                    round(bbox.ymin, 6),
                    round(bbox.width, 6),
                    round(bbox.height, 6),
                ]
            )

            if len(detections) > 1:
                result["face_flag"] = "multi_face"
            elif max_conf < self._face_confidence_threshold:
                result["face_flag"] = "low_confidence"
            else:
                result["face_flag"] = "ok"

        # ── Step 5: Overall pass/fail ──────────────────────────────────── #
        result["quality_pass"] = (
            not result["is_corrupted"]
            and not result["is_blurry"]
            and result["exposure_flag"] == "ok"
            and result["face_flag"] in {"ok", "low_confidence"}
        )

        return result

    # ------------------------------------------------------------------ #
    # Plotting helpers                                                     #
    # ------------------------------------------------------------------ #

    def _save_summary_plots(self, df: pd.DataFrame) -> None:
        """Persist quality flag summary bar chart and blur KDE plot.

        Parameters
        ----------
        df : pd.DataFrame
            Merged quality-audit DataFrame (manifest + quality columns).
        """
        # ── Flag counts ────────────────────────────────────────────────── #
        flag_columns = {
            "is_corrupted": "Corrupted",
            "is_blurry": "Blurry",
            "underexposed": "Underexposed",
            "overexposed": "Overexposed",
            "low_contrast": "Low Contrast",
            "no_face": "No Face",
            "multi_face": "Multi-face",
        }

        flag_counts: list[int] = []
        flag_labels: list[str] = []
        flag_colors: list[str] = []

        for col_or_flag, label in flag_columns.items():
            if col_or_flag == "is_corrupted":
                count = int(df["is_corrupted"].sum())
            elif col_or_flag == "is_blurry":
                count = int(df["is_blurry"].sum())
            else:
                count = int((df["exposure_flag"] == col_or_flag).sum() if col_or_flag in {"underexposed", "overexposed", "low_contrast"} else (df["face_flag"] == col_or_flag).sum())

            flag_counts.append(count)
            flag_labels.append(label)
            flag_colors.append("#e05c5c" if count > 0 else "#4caf50")

        flags_figure_path = self.figures_dir / "03_quality_flags_summary.png"
        save_bar_chart(
            values=flag_counts,
            labels=flag_labels,
            title="Quality Flag Summary",
            output_path=flags_figure_path,
            horizontal=True,
            bar_colors=flag_colors,
            annotations=[str(c) for c in flag_counts],
        )
        self.logger.info("Quality flags chart saved → %s", flags_figure_path)

        # ── Blur KDE ───────────────────────────────────────────────────── #
        blur_figure_path = self.figures_dir / "03_blur_score_distribution.png"
        valid_df = df[df["blur_score"] >= 0].copy()

        series_data: dict[str, list[float]] = {}
        for idx, name in self._severity_map.items():
            subset = valid_df[valid_df["severity_label"] == idx]["blur_score"].tolist()
            if subset:
                series_data[name] = subset

        save_kde_plot(
            series=series_data,
            title="Blur Score Distribution by Severity",
            xlabel="Laplacian Variance (blur score)",
            output_path=blur_figure_path,
            vlines=[self._blur_threshold],
            vline_labels=["blur_threshold"],
            vline_style="--",
        )
        self.logger.info("Blur KDE plot saved → %s", blur_figure_path)

    # ------------------------------------------------------------------ #
    # Public run method                                                    #
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """Audit all images and persist results.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame containing manifest columns + quality columns.
        """
        from tqdm import tqdm  # noqa: PLC0415

        records: list[dict] = []

        try:
            for _, row in tqdm(
                self.df.iterrows(),
                total=len(self.df),
                desc="Auditing images",
                unit="img",
            ):
                quality_metrics = self.audit_single_image(row)
                records.append(quality_metrics)
        finally:
            # Ensure MediaPipe detector is properly closed
            try:
                self._face_detector.__exit__(None, None, None)
                self.logger.debug("MediaPipe FaceDetection context closed.")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Error closing MediaPipe detector: %s", exc)

        quality_df = pd.DataFrame(records)
        merged_df = pd.concat(
            [self.df.reset_index(drop=True), quality_df.reset_index(drop=True)],
            axis=1,
        )

        # ── Persist CSV ────────────────────────────────────────────────── #
        output_csv = self.outputs_root / "quality_audit.csv"
        merged_df.to_csv(output_csv, index=False)
        self.logger.info("Quality audit CSV saved → %s", output_csv)

        # ── Summary plots ──────────────────────────────────────────────── #
        self._save_summary_plots(merged_df)

        # ── Log summary ────────────────────────────────────────────────── #
        total = len(merged_df)
        n_pass = int(merged_df["quality_pass"].sum())
        n_corrupted = int(merged_df["is_corrupted"].sum())
        n_blurry = int(merged_df["is_blurry"].sum())
        face_flag_counts = merged_df["face_flag"].value_counts().to_dict()
        exposure_flag_counts = merged_df["exposure_flag"].value_counts().to_dict()

        self.logger.info(
            "Quality audit complete | total=%d | pass=%d (%.1f%%) | "
            "corrupted=%d | blurry=%d",
            total, n_pass, n_pass / total * 100 if total else 0,
            n_corrupted, n_blurry,
        )
        self.logger.info("Face flag breakdown: %s", face_flag_counts)
        self.logger.info("Exposure flag breakdown: %s", exposure_flag_counts)

        return merged_df


# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #

def _detection_score(detection: Any) -> float:
    """Extract the confidence score from a MediaPipe detection object."""
    try:
        return float(detection.score[0])
    except (AttributeError, IndexError, TypeError):
        return 0.0


# --------------------------------------------------------------------------- #
# Module-level convenience function                                            #
# --------------------------------------------------------------------------- #

def run_quality_audit(config_path: Path, project_root: Path) -> pd.DataFrame:
    """Load config and run the full quality audit pipeline.

    Parameters
    ----------
    config_path : Path
        Path to ``phase0.yaml``.
    project_root : Path
        Absolute project root.

    Returns
    -------
    pd.DataFrame
        Merged quality-audit DataFrame.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    outputs_root = project_root / config["paths"]["outputs_root"]
    manifest_path = outputs_root / "manifest.csv"

    auditor = ImageQualityAuditor(
        manifest_path=manifest_path,
        config=config,
        project_root=project_root,
    )
    return auditor.run()


def audit_single_image(img_path: Path, config: dict) -> dict:
    """Convenience helper to audit a single image file without creating a manifest.

    Mainly used for testing.
    """
    import pandas as pd
    img_path = Path(img_path)
    
    auditor_config = {
        "paths": {
            "outputs_root": "data/phase0_outputs",
            "figures_dir": "reports/phase0/figures",
        },
        "ingestion": {
            "severity_map": {0: "mild", 1: "moderate", 2: "severe", 3: "very_severe"},
        },
        "quality": config
    }
    
    temp_dir = img_path.parent
    temp_manifest = temp_dir / f"temp_manifest_{img_path.stem}.csv"
    pd.DataFrame([{"image_id": img_path.stem, "image_path": str(img_path)}]).to_csv(temp_manifest, index=False)
    
    try:
        auditor = ImageQualityAuditor(
            manifest_path=temp_manifest,
            config=auditor_config,
            project_root=temp_dir,
        )
        row = auditor.df.iloc[0]
        res = auditor.audit_single_image(row)
        auditor._face_detector.__exit__(None, None, None)
    finally:
        if temp_manifest.exists():
            try:
                temp_manifest.unlink()
            except OSError:
                pass
                
    return res


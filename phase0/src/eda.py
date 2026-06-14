"""
phase0/src/eda.py
=================
Exploratory Data Analysis for the ACNE04 dataset.

Provides DatasetAnalyser, which computes class distribution statistics,
resolution statistics, sample grids, and all-folder discrepancy analysis.
All results are persisted to disk and returned as plain dicts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from phase0.src.utils.logging import get_logger
from phase0.src.utils.io import get_image_dimensions
from phase0.src.utils.visualisation import (
    save_bar_chart,
    save_scatter,
    save_image_grid,
    save_stacked_bar,
)


class DatasetAnalyser:
    """Analyse the ACNE04 dataset and persist EDA artefacts."""

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

        # ── Output directories ─────────────────────────────────────────── #
        self.figures_dir: Path = project_root / config["paths"]["figures_dir"]
        self.outputs_root: Path = project_root / config["paths"]["outputs_root"]
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

        # ── Severity metadata ──────────────────────────────────────────── #
        raw_map: dict = config["ingestion"]["severity_map"]
        # YAML may have loaded integer keys; normalise to int
        self._severity_map: dict[int, str] = {int(k): v for k, v in raw_map.items()}
        self.severity_names: list[str] = [
            self._severity_map[i] for i in sorted(self._severity_map)
        ]  # ['mild', 'moderate', 'severe', 'very_severe']

        # ── EDA sub-config ─────────────────────────────────────────────── #
        eda_cfg = config.get("eda", {})
        self._grid_rows: int = eda_cfg.get("sample_grid_rows", 4)
        self._grid_cols: int = eda_cfg.get("sample_grid_cols", 4)
        self._grid_seed: int = eda_cfg.get("sample_grid_seed", 42)

        self._min_dim: int = config["quality"]["min_image_dimension"]

        self.logger.debug(
            "figures_dir=%s  outputs_root=%s", self.figures_dir, self.outputs_root
        )

    # ------------------------------------------------------------------ #
    # Public analysis methods                                             #
    # ------------------------------------------------------------------ #

    def analyse_class_distribution(self) -> dict:
        """Count images per severity class and save a bar chart.

        Returns
        -------
        dict
            ``class_counts``      – {severity_name: count}
            ``imbalance_ratio``   – max_count / min_count
            ``recommended_weights`` – inverse-frequency weights normalised
                                      so they sum to 4.0.
            ``figure_path``       – str path to the saved figure.
        """
        # ── Counts ────────────────────────────────────────────────────── #
        # The manifest must contain a column 'severity_label' (int 0-3)
        counts_series = (
            self.df["severity_label"]
            .value_counts()
            .sort_index()
        )

        class_counts: dict[str, int] = {
            self._severity_map[int(idx)]: int(cnt)
            for idx, cnt in counts_series.items()
            if int(idx) in self._severity_map
        }

        # Fill missing classes with 0
        for name in self.severity_names:
            class_counts.setdefault(name, 0)

        counts_ordered = [class_counts[n] for n in self.severity_names]
        total = sum(counts_ordered)
        max_count = max(counts_ordered)
        min_count = min(c for c in counts_ordered if c > 0)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        # ── Inverse-frequency weights ─────────────────────────────────── #
        raw_weights = [
            1.0 / c if c > 0 else 0.0 for c in counts_ordered
        ]
        weight_sum = sum(raw_weights)
        n_classes = len(self.severity_names)
        normalised_weights = (
            [w / weight_sum * n_classes for w in raw_weights]
            if weight_sum > 0
            else [1.0] * n_classes
        )
        recommended_weights: dict[str, float] = {
            name: round(w, 6)
            for name, w in zip(self.severity_names, normalised_weights)
        }

        # ── Figure ────────────────────────────────────────────────────── #
        labels = [
            f"{name}  (n={cnt})"
            for name, cnt in zip(self.severity_names, counts_ordered)
        ]
        percentages = [
            cnt / total * 100 if total > 0 else 0.0 for cnt in counts_ordered
        ]
        annotations = [
            f"{cnt} ({pct:.1f}%)"
            for cnt, pct in zip(counts_ordered, percentages)
        ]
        figure_path = self.figures_dir / "01_class_distribution.png"
        save_bar_chart(
            values=counts_ordered,
            labels=labels,
            title=f"Class Distribution (IR = {imbalance_ratio:.2f})",
            output_path=figure_path,
            horizontal=True,
            annotations=annotations,
        )
        self.logger.info(
            "Class distribution saved → %s  |  IR=%.2f", figure_path, imbalance_ratio
        )

        return {
            "class_counts": class_counts,
            "imbalance_ratio": round(imbalance_ratio, 4),
            "recommended_weights": recommended_weights,
            "figure_path": str(figure_path),
        }

    # ------------------------------------------------------------------ #

    def analyse_resolution(self) -> dict:
        """Measure image resolution for every existing file in the manifest.

        Returns
        -------
        dict
            ``width_stats``, ``height_stats`` – descriptive stats dicts.
            ``n_below_min_dimension``          – count of images below threshold.
            ``total_measured``                 – number of images successfully measured.
            ``figure_path``                    – str path to the saved figure.
        """
        widths: list[float] = []
        heights: list[float] = []
        colors: list[int] = []   # severity_label as int
        missing = 0

        for _, row in self.df.iterrows():
            img_path = Path(str(row["image_path"]))
            if not img_path.is_absolute():
                img_path = self.project_root / img_path
            if not img_path.exists():
                missing += 1
                continue
            dims = get_image_dimensions(img_path)
            if dims is None:
                missing += 1
                continue
            w, h = dims
            widths.append(float(w))
            heights.append(float(h))
            colors.append(int(row["severity_label"]))

        if missing:
            self.logger.warning(
                "%d image(s) could not be measured (missing or unreadable).", missing
            )

        total_measured = len(widths)

        def _stats(values: list[float]) -> dict:
            if not values:
                return {
                    "min": None, "max": None, "mean": None,
                    "median": None, "p5": None, "p95": None,
                }
            arr = np.array(values)
            return {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
            }

        width_stats = _stats(widths)
        height_stats = _stats(heights)

        n_below = sum(
            1
            for w, h in zip(widths, heights)
            if w < self._min_dim or h < self._min_dim
        )

        # ── Scatter figure ────────────────────────────────────────────── #
        # Map severity int → colour string for scatter
        colour_palette = ["#4e9af1", "#f1a94e", "#e05c5c", "#8e44ad"]
        point_colors = [colour_palette[c % 4] for c in colors]
        figure_path = self.figures_dir / "02_resolution_scatter.png"
        save_scatter(
            x=widths,
            y=heights,
            colors=point_colors,
            title="Image Resolution Distribution",
            xlabel="Width (px)",
            ylabel="Height (px)",
            output_path=figure_path,
            alpha=0.5,
            vlines=[self._min_dim],
            hlines=[self._min_dim],
            vline_style="--",
            hline_style="--",
            legend_labels={
                colour_palette[i]: self.severity_names[i]
                for i in range(len(self.severity_names))
            },
        )
        self.logger.info(
            "Resolution scatter saved → %s  |  measured=%d  below_min=%d",
            figure_path, total_measured, n_below,
        )

        return {
            "width_stats": width_stats,
            "height_stats": height_stats,
            "n_below_min_dimension": n_below,
            "total_measured": total_measured,
            "figure_path": str(figure_path),
        }

    # ------------------------------------------------------------------ #

    def generate_sample_grids(self) -> list[Path]:
        """Save one sample image grid per severity class.

        Returns
        -------
        list[Path]
            Four output paths (one per severity class, in order 0-3).
        """
        rng = random.Random(self._grid_seed)
        max_samples = self._grid_rows * self._grid_cols
        saved_paths: list[Path] = []

        for class_idx, severity_name in enumerate(self.severity_names):
            subset = self.df[self.df["severity_label"] == class_idx]
            filepaths: list[Path] = []
            for fp in subset["image_path"].tolist():
                p = Path(str(fp))
                if not p.is_absolute():
                    p = self.project_root / p
                if p.exists():
                    filepaths.append(p)

            n_sample = min(max_samples, len(filepaths))
            sampled = rng.sample(filepaths, n_sample) if n_sample > 0 else []

            output_path = self.figures_dir / f"02_sample_grid_{severity_name}.png"
            if sampled:
                save_image_grid(
                    image_paths=sampled,
                    title=f"Samples: {severity_name}",
                    output_path=output_path,
                    rows=self._grid_rows,
                    cols=self._grid_cols,
                )
                self.logger.info(
                    "Sample grid (%s) saved → %s  |  %d images",
                    severity_name, output_path, len(sampled),
                )
            else:
                self.logger.warning(
                    "No images found for class '%s'; skipping grid.", severity_name
                )

            saved_paths.append(output_path)

        return saved_paths

    # ------------------------------------------------------------------ #

    def analyse_all_folder_discrepancy(
        self, all_folder_index_path: Path
    ) -> dict | None:
        """Compare class-folder counts vs the all_1024 folder index.

        Parameters
        ----------
        all_folder_index_path : Path
            Path to the ``all_folder_index.csv`` produced during ingestion.
            If the file does not exist, returns ``None``.

        Returns
        -------
        dict | None
        """
        if not all_folder_index_path.exists():
            self.logger.info(
                "all_folder_index not found at %s; skipping discrepancy analysis.",
                all_folder_index_path,
            )
            return None

        all_df = pd.read_csv(all_folder_index_path)
        self.logger.info(
            "Loaded all_folder_index with %d rows from %s",
            len(all_df), all_folder_index_path,
        )

        # ── Counts from class folders (manifest) ─────────────────────── #
        class_folder_counts: dict[str, int] = {}
        for idx, name in self._severity_map.items():
            folder_name = f"acne{idx}_1024"
            cnt = int((self.df["severity_label"] == idx).sum())
            class_folder_counts[folder_name] = cnt

        # ── Count from all_1024 ───────────────────────────────────────── #
        all_filenames = set(all_df["filename"].tolist()) if "filename" in all_df.columns else set()

        # Images in all_1024 that are also in the manifest (labelled)
        manifest_filenames = set()
        if "filename" in self.df.columns:
            manifest_filenames = set(self.df["filename"].tolist())

        in_class_folder = len(manifest_filenames & all_filenames)
        unlabelled = len(all_filenames - manifest_filenames)

        # Duplicates within the all_1024 index itself
        duplicate_in_all = int(all_df.duplicated(subset=["filename"]).sum()) if "filename" in all_df.columns else 0

        # ── Stacked bar ───────────────────────────────────────────────── #
        figure_path = self.figures_dir / "01_all_folder_discrepancy.png"
        categories = list(class_folder_counts.keys()) + ["unlabelled"]
        class_vals = list(class_folder_counts.values()) + [0]  # unlabelled has no class split
        all_vals = [0] * len(class_folder_counts) + [unlabelled]

        # Reframe: for each category show "in_class_folder" vs "only_in_all"
        # Stack layer 1: present in class folder; layer 2: only in all_1024
        class_folder_in_all = []
        class_folder_not_in_all = []
        for folder_name, cnt in class_folder_counts.items():
            # Can't distinguish per-folder without more info; use totals
            class_folder_in_all.append(cnt)
            class_folder_not_in_all.append(0)
        class_folder_in_all.append(0)
        class_folder_not_in_all.append(unlabelled)

        save_stacked_bar(
            categories=categories,
            series={
                "In class folders": class_folder_in_all,
                "Unlabelled (all_1024 only)": class_folder_not_in_all,
            },
            title="Class Folder vs all_1024 Discrepancy",
            output_path=figure_path,
            xlabel="Folder",
            ylabel="Image Count",
        )
        self.logger.info("Discrepancy chart saved → %s", figure_path)

        return {
            "in_class_folder": in_class_folder,
            "unlabelled": unlabelled,
            "duplicate_in_all": duplicate_in_all,
            "class_folder_counts": class_folder_counts,
            "figure_path": str(figure_path),
        }

    # ------------------------------------------------------------------ #

    def run(self, all_folder_index_path: Path | None = None) -> dict:
        """Execute all analysis steps and persist combined results.

        Parameters
        ----------
        all_folder_index_path : Path | None
            Optional path to ``all_folder_index.csv``.

        Returns
        -------
        dict
            Aggregated EDA statistics dictionary.
        """
        self.logger.info("=== EDA run started ===")

        stats: dict = {}

        # 1. Class distribution
        dist_result = self.analyse_class_distribution()
        stats["class_distribution"] = dist_result

        # 2. Resolution analysis
        res_result = self.analyse_resolution()
        stats["resolution"] = res_result

        # 3. Sample grids (paths converted to str for JSON serialisability)
        grid_paths = self.generate_sample_grids()
        stats["sample_grid_paths"] = [str(p) for p in grid_paths]

        # 4. All-folder discrepancy (optional)
        if all_folder_index_path is not None:
            disc_result = self.analyse_all_folder_discrepancy(all_folder_index_path)
        else:
            disc_result = None
        stats["all_folder_discrepancy"] = disc_result

        # ── Persist ───────────────────────────────────────────────────── #
        eda_stats_path = self.outputs_root / "eda_stats.json"
        with open(eda_stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, default=str)
        self.logger.info("EDA stats written → %s", eda_stats_path)

        # ── Summary log ───────────────────────────────────────────────── #
        cc = dist_result["class_counts"]
        ir = dist_result["imbalance_ratio"]
        tm = res_result["total_measured"]
        nb = res_result["n_below_min_dimension"]
        w_mean = res_result["width_stats"].get("mean")
        h_mean = res_result["height_stats"].get("mean")

        self.logger.info(
            "EDA summary | classes=%s | IR=%.2f | measured=%d | "
            "below_min_dim=%d | mean_resolution=%.0fx%.0f",
            cc, ir, tm, nb,
            w_mean if w_mean else 0,
            h_mean if h_mean else 0,
        )
        self.logger.info("=== EDA run complete ===")

        return stats


# --------------------------------------------------------------------------- #
# Module-level convenience function                                            #
# --------------------------------------------------------------------------- #

def run_eda(config_path: Path, project_root: Path) -> dict:
    """Load config and run the full EDA pipeline.

    Parameters
    ----------
    config_path : Path
        Path to ``phase0.yaml``.
    project_root : Path
        Absolute project root.

    Returns
    -------
    dict
        Aggregated EDA statistics.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    outputs_root = project_root / config["paths"]["outputs_root"]
    manifest_path = outputs_root / "manifest.csv"

    all_folder_index_path = outputs_root / "all_folder_index.csv"

    analyser = DatasetAnalyser(
        manifest_path=manifest_path,
        config=config,
        project_root=project_root,
    )
    return analyser.run(all_folder_index_path=all_folder_index_path)

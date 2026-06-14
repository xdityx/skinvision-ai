"""
phase0/src/split_generator.py
──────────────────────────────
Cluster-aware stratified split generation that prevents identity leakage
across train / val / test splits.

Algorithm
---------
For each acne-severity bucket (mild, moderate, severe, very_severe):
  1. Gather all multi-image clusters assigned to that bucket.
  2. Shuffle with a fixed seed for reproducibility.
  3. Allocate 70 % → train, 15 % → val, 15 % → test.
  4. Singleton images (cluster_id == -1) are treated as individual groups
     and allocated with the same proportions per severity bucket.
  5. Expand cluster → image assignments so every image gets a split label.

Usage
-----
    from phase0.src.split_generator import run_split_generation
    summary = run_split_generation(
        config_path=Path("phase0/config/phase0.yaml"),
        project_root=Path("."),
    )
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from phase0.src.utils.logging import get_logger
from phase0.src.utils.visualisation import save_stacked_bar, save_bar_chart

# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_MAP: dict[int, str] = {0: "mild", 1: "moderate", 2: "severe"}
SPLIT_NAMES = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
class SplitGenerator:
    """
    Cluster-aware, stratified train / val / test split generator.

    Parameters
    ----------
    quality_audit_path : path to quality_audit.csv
    cluster_path       : path to cluster_assignments.csv
    config             : parsed phase0.yaml as a dict
    project_root       : absolute path to the project root
    """

    def __init__(
        self,
        quality_audit_path: Path,
        cluster_path: Path,
        config: dict,
        project_root: Path,
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.project_root = project_root

        # ── paths ─────────────────────────────────────────────────────────────
        path_cfg = config["paths"]
        self.outputs_root: Path = project_root / path_cfg["outputs_root"]
        self.splits_dir: Path = self.outputs_root / "splits"
        self.figures_dir: Path = project_root / path_cfg["figures_dir"]
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # ── split ratios ──────────────────────────────────────────────────────
        split_cfg = config["splits"]
        self.train_ratio: float = float(split_cfg["train_ratio"])
        self.val_ratio: float = float(split_cfg["val_ratio"])
        self.test_ratio: float = float(split_cfg["test_ratio"])
        self.tolerance_pct: float = float(split_cfg["distribution_tolerance_pct"])
        self.seed: int = int(split_cfg["seed"])

        # ── data ──────────────────────────────────────────────────────────────
        self.logger.info("Loading quality audit: %s", quality_audit_path)
        qa_df = pd.read_csv(quality_audit_path)
        qa_df["image_id"] = qa_df["image_id"].astype(str)
        qa_df["quality_pass"] = qa_df["quality_pass"].astype(bool)
        qa_df["severity_label"] = qa_df["severity_label"].astype(int)

        self.logger.info("Loading cluster assignments: %s", cluster_path)
        cl_df = pd.read_csv(cluster_path)
        cl_df["image_id"] = cl_df["image_id"].astype(str)
        cl_df["cluster_id"] = cl_df["cluster_id"].astype(int)

        # Merge on image_id — keep all quality-audit rows
        self.merged_df: pd.DataFrame = qa_df.merge(
            cl_df[["image_id", "cluster_id", "cluster_size", "embedding_extracted"]],
            on="image_id",
            how="left",
        )
        # Fill missing cluster info for any images not in cluster_assignments
        self.merged_df["cluster_id"] = (
            self.merged_df["cluster_id"].fillna(-1).astype(int)
        )
        self.merged_df["cluster_size"] = (
            self.merged_df["cluster_size"].fillna(1).astype(int)
        )
        self.merged_df["embedding_extracted"] = (
            self.merged_df["embedding_extracted"].fillna(False).astype(bool)
        )
        self.logger.info(
            "Merged dataset: %d images, %d unique clusters",
            len(self.merged_df),
            self.merged_df["cluster_id"].nunique(),
        )

        raw_map = config.get("ingestion", {}).get("severity_map", {0: "mild", 1: "moderate", 2: "severe"})
        self.severity_map = {int(k): v for k, v in raw_map.items()}

    # ─────────────────────────────────────────────────────────────────────
    # Cluster → severity assignment
    # ─────────────────────────────────────────────────────────────────────

    def _assign_cluster_severity(self, cluster_df: pd.DataFrame) -> pd.Series:
        """
        For every cluster_id >= 0, compute the mode severity_label across
        its member images.  Ties are broken by choosing the lower severity
        index for reproducibility.

        Returns
        -------
        pd.Series : index = cluster_id (int), values = severity_label (int)
        """
        multi = cluster_df[cluster_df["cluster_id"] >= 0].copy()
        if multi.empty:
            return pd.Series(dtype=int)

        def _mode_sev(group: pd.DataFrame) -> int:
            counts = group["severity_label"].value_counts()
            return int(counts.idxmax())

        cluster_severity = (
            multi.groupby("cluster_id")
            .apply(_mode_sev)
            .rename("representative_severity")
        )
        return cluster_severity  # type: ignore[return-value]

    # ─────────────────────────────────────────────────────────────────────
    # Greedy stratified split
    # ─────────────────────────────────────────────────────────────────────

    def _greedy_stratified_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign each image a split label while keeping all images from the
        same cluster in the same split.

        Returns df with an added 'split' column ('train'/'val'/'test').
        """
        rng = random.Random(self.seed)
        df = df.copy()
        df["split"] = ""

        # ── Assign multi-image clusters (cluster_id >= 0) ──────────────────
        cluster_sev = self._assign_cluster_severity(df)

        for sev_lbl in sorted(self.severity_map.keys()):
            cluster_ids = sorted(cluster_sev[cluster_sev == sev_lbl].index.tolist())
            rng.shuffle(cluster_ids)
            n = len(cluster_ids)
            n_train = max(1, round(n * self.train_ratio)) if n >= 1 else 0
            n_val = max(0, round(n * self.val_ratio)) if n >= 2 else 0
            # test gets the remainder
            n_test = n - n_train - n_val

            if n_test < 0:
                # Edge case: very few clusters — give all to train
                n_train = n
                n_val = 0
                n_test = 0

            train_clusters = set(cluster_ids[:n_train])
            val_clusters = set(cluster_ids[n_train : n_train + n_val])
            test_clusters = set(cluster_ids[n_train + n_val :])

            mask = df["cluster_id"].isin(train_clusters)
            df.loc[mask, "split"] = "train"
            mask = df["cluster_id"].isin(val_clusters)
            df.loc[mask, "split"] = "val"
            mask = df["cluster_id"].isin(test_clusters)
            df.loc[mask, "split"] = "test"

        # ── Assign singletons (cluster_id == -1) per severity bucket ──────
        for sev_lbl in sorted(self.severity_map.keys()):
            singleton_mask = (df["cluster_id"] == -1) & (df["severity_label"] == sev_lbl)
            singleton_idx = df.index[singleton_mask].tolist()
            rng.shuffle(singleton_idx)
            n = len(singleton_idx)
            n_train = max(1, round(n * self.train_ratio)) if n >= 1 else 0
            n_val = max(0, round(n * self.val_ratio)) if n >= 2 else 0
            n_test = n - n_train - n_val
            if n_test < 0:
                n_train = n
                n_val = 0
                n_test = 0

            df.loc[singleton_idx[:n_train], "split"] = "train"
            df.loc[singleton_idx[n_train : n_train + n_val], "split"] = "val"
            df.loc[singleton_idx[n_train + n_val :], "split"] = "test"

        # ── Handle any stragglers still unlabelled ─────────────────────────
        unlabelled = df["split"] == ""
        if unlabelled.sum():
            self.logger.warning(
                "%d images could not be split — assigning to 'train'.", unlabelled.sum()
            )
            df.loc[unlabelled, "split"] = "train"

        return df

    # ─────────────────────────────────────────────────────────────────────
    # Verification
    # ─────────────────────────────────────────────────────────────────────

    def _verify_splits(self, df: pd.DataFrame) -> dict:
        """
        Run a battery of post-hoc checks to guarantee split integrity.

        Returns
        -------
        dict : {check_name: bool, ..., 'overall_pass': bool}

        Raises
        ------
        ValueError if any check fails.
        """
        checks: dict[str, bool] = {}
        errors: list[str] = []

        # 1. No cluster appears in two splits
        multi_clusters = df[df["cluster_id"] >= 0]
        cluster_splits = multi_clusters.groupby("cluster_id")["split"].nunique()
        check1 = bool((cluster_splits <= 1).all())
        checks["no_cluster_in_multiple_splits"] = check1
        if not check1:
            bad = cluster_splits[cluster_splits > 1].index.tolist()
            errors.append(
                f"Clusters appear in >1 split: {bad[:10]}{'…' if len(bad) > 10 else ''}"
            )

        # 2. All input images appear in exactly one split
        check2 = len(df) == df["image_id"].nunique()
        checks["all_images_assigned"] = check2
        if not check2:
            errors.append(
                f"Duplicate image_ids found in merged df ({len(df) - df['image_id'].nunique()} dups)."
            )

        # 3. No duplicate image_ids across splits
        split_counts = df.groupby("image_id")["split"].nunique()
        check3 = bool((split_counts == 1).all())
        checks["no_duplicate_across_splits"] = check3
        if not check3:
            bad_ids = split_counts[split_counts > 1].index.tolist()
            errors.append(
                f"image_ids appear in >1 split: {bad_ids[:5]}{'…' if len(bad_ids) > 5 else ''}"
            )

        # 4. Per-class distribution within tolerance_pct of target
        tol = self.tolerance_pct
        target = {"train": self.train_ratio, "val": self.val_ratio, "test": self.test_ratio}
        check4 = True
        for sev in sorted(self.severity_map.keys()):
            sev_total = (df["severity_label"] == sev).sum()
            if sev_total == 0:
                continue
            for split_name, t_ratio in target.items():
                actual_n = ((df["severity_label"] == sev) & (df["split"] == split_name)).sum()
                actual_pct = actual_n / sev_total * 100
                expected_pct = t_ratio * 100
                deviation = abs(actual_pct - expected_pct)
                if deviation > tol:
                    check4 = False
                    errors.append(
                        f"Class {sev} ({self.severity_map[sev]}) in '{split_name}': "
                        f"actual={actual_pct:.1f}% expected={expected_pct:.1f}% "
                        f"deviation={deviation:.1f}% > tol={tol}%"
                    )
        checks["class_distribution_within_tolerance"] = check4

        overall = all(checks.values())
        checks["overall_pass"] = overall

        if errors:
            msg = "Split verification FAILED:\n" + "\n".join(f"  • {e}" for e in errors)
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("All split verification checks passed ✓")
        return checks

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full split-generation pipeline.

        Returns
        -------
        split_summary : dict  (also written to split_summary.json)
        """
        # 1. Generate splits
        split_df = self._greedy_stratified_split(self.merged_df)

        # 2. Verify
        verification = self._verify_splits(split_df)

        # 3. Save per-split CSVs
        out_cols = [
            "image_id", "image_path", "severity_label", "severity_name",
            "cluster_id", "split",
        ]
        # Ensure severity_name is present
        if "severity_name" not in split_df.columns:
            split_df["severity_name"] = split_df["severity_label"].map(self.severity_map)

        for split_name in SPLIT_NAMES:
            subset = split_df[split_df["split"] == split_name][out_cols].reset_index(drop=True)
            out_path = self.splits_dir / f"{split_name}.csv"
            subset.to_csv(out_path, index=False)
            self.logger.info("Saved %s split → %s (%d rows)", split_name, out_path, len(subset))

        # 4. Build summary
        split_summary_splits: dict[str, Any] = {}
        for split_name in SPLIT_NAMES:
            sub = split_df[split_df["split"] == split_name]
            n_images = len(sub)
            n_clusters = sub[sub["cluster_id"] >= 0]["cluster_id"].nunique()
            class_dist: dict[str, int] = {}
            for sev_lbl, sev_name in self.severity_map.items():
                class_dist[sev_name] = int((sub["severity_label"] == sev_lbl).sum())
            split_summary_splits[split_name] = {
                "n_images": n_images,
                "n_clusters": n_clusters,
                "class_distribution": class_dist,
            }

        split_summary: dict = {
            "seed": self.seed,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "total_input_images": len(split_df),
            "splits": split_summary_splits,
            "verification": verification,
        }

        summary_path = self.splits_dir / "split_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(split_summary, f, indent=2)
        self.logger.info("Split summary saved → %s", summary_path)

        # 5. Log table
        sorted_labels = sorted(self.severity_map.keys())
        header = f"{'Split':<8} {'Total':>7} {'Clusters':>10} " + "  ".join(
            f"{self.severity_map[s]:>10}" for s in sorted_labels
        )
        self.logger.info("\n" + "═" * len(header))
        self.logger.info(header)
        self.logger.info("─" * len(header))
        for split_name in SPLIT_NAMES:
            info = split_summary_splits[split_name]
            cd = info["class_distribution"]
            row = (
                f"{split_name:<8} {info['n_images']:>7} {info['n_clusters']:>10}  "
                + "  ".join(
                    f"{cd.get(self.severity_map[s], 0):>10}" for s in sorted_labels
                )
            )
            self.logger.info(row)
        self.logger.info("═" * len(header))

        # 6. Visualisation — grouped bar chart
        self._save_split_distribution_chart(split_df)

        return split_summary

    # ─────────────────────────────────────────────────────────────────────
    # Visualisation helper
    # ─────────────────────────────────────────────────────────────────────

    def _save_split_distribution_chart(self, split_df: pd.DataFrame) -> None:
        """Grouped bar chart: train / val / test side-by-side per severity class."""
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        sorted_labels = sorted(self.severity_map.keys())
        severity_names = [self.severity_map[i] for i in sorted_labels]
        split_colors = {"train": "#5C6BC0", "val": "#26A69A", "test": "#EF5350"}

        x = np.arange(len(severity_names))
        width = 0.25
        offsets = {"train": -width, "val": 0.0, "test": width}

        fig, ax = plt.subplots(figsize=(10, 6))
        for split_name in SPLIT_NAMES:
            sub = split_df[split_df["split"] == split_name]
            counts = [int((sub["severity_label"] == sev).sum()) for sev in sorted_labels]
            bars = ax.bar(
                x + offsets[split_name],
                counts,
                width=width * 0.95,
                label=split_name.capitalize(),
                color=split_colors[split_name],
                alpha=0.88,
                edgecolor="white",
            )
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01 if max(counts) > 0 else 1.0,
                    str(count),
                    ha="center", va="bottom", fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([sn.replace("_", " ").title() for sn in severity_names])
        ax.set_title("Class Distribution per Split", fontsize=14)
        ax.set_xlabel("Acne Severity")
        ax.set_ylabel("Number of Images")
        ax.legend()
        fig.tight_layout()

        out_path = self.figures_dir / "05_split_class_distribution.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_split_generation(
    config_path: Path,
    project_root: Path,
) -> dict:
    """
    Load config and run the full split-generation pipeline.

    Parameters
    ----------
    config_path  : Path to phase0.yaml
    project_root : Absolute path to the project root

    Returns
    -------
    split_summary : dict
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outputs_root = project_root / config["paths"]["outputs_root"]
    quality_audit_path = outputs_root / "quality_audit.csv"
    cluster_path = outputs_root / "cluster_assignments.csv"

    generator = SplitGenerator(
        quality_audit_path=quality_audit_path,
        cluster_path=cluster_path,
        config=config,
        project_root=project_root,
    )
    return generator.run()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0 — Split Generation")
    parser.add_argument("--config", default="phase0/config/phase0.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    run_split_generation(
        config_path=Path(args.config),
        project_root=Path(args.project_root).resolve(),
    )

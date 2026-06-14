"""
phase0/src/report_generator.py
────────────────────────────────
Aggregates all Phase 0 artefacts (EDA stats, quality audit, clustering
results, split summary) into a structured feasibility report in both
Markdown and JSON formats.

The report drives the go / no-go decision before investing in Phase 1
model training.

Usage
-----
    from phase0.src.report_generator import run_report_generation
    report = run_report_generation(
        config_path=Path("phase0/config/phase0.yaml"),
        project_root=Path("."),
    )
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from phase0.src.utils.logging import get_logger

# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_MAP: dict[int, str] = {0: "mild", 1: "moderate", 2: "severe"}
SPLIT_NAMES = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────────────────
class FeasibilityReportGenerator:
    """
    Reads all Phase 0 output artefacts and produces a comprehensive
    feasibility report for the ACNE04 dataset.
    """

    def __init__(self, config: dict, project_root: Path) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.project_root = project_root

        # ── paths ─────────────────────────────────────────────────────────────
        path_cfg = config["paths"]
        self.outputs_root: Path = project_root / path_cfg["outputs_root"]
        self.figures_dir: Path = project_root / path_cfg["figures_dir"]
        self.splits_dir: Path = self.outputs_root / "splits"
        self.reports_dir: Path = project_root / path_cfg["reports_root"]
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # ── report config ─────────────────────────────────────────────────────
        self.thresholds: dict = config["report"]["thresholds"]
        self.seed: int = config["project"]["seed"]

        raw_map = config.get("ingestion", {}).get("severity_map", {0: "mild", 1: "moderate", 2: "severe"})
        self.severity_map = {int(k): v for k, v in raw_map.items()}

    # ─────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────

    def _safe_load_json(self, path: Path) -> dict | None:
        """Load a JSON file, returning None and logging a warning on failure."""
        if not path.exists():
            self.logger.warning("Missing file (expected): %s", path)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning("Could not parse %s: %s", path, exc)
            return None

    def _safe_load_csv(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            self.logger.warning("Missing CSV (expected): %s", path)
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            self.logger.warning("Could not read CSV %s: %s", path, exc)
            return None

    def _load_all_stats(self) -> dict:
        """
        Aggregate statistics from every Phase 0 output artefact.

        Returns a unified dict consumed by the downstream gate / render
        methods.
        """
        stats: dict[str, Any] = {}

        # ── JSON artefacts ────────────────────────────────────────────────────
        stats["eda_stats"] = self._safe_load_json(
            self.outputs_root / "eda_stats.json"
        )
        stats["ingestion_log"] = self._safe_load_json(
            self.outputs_root / "ingestion_log.json"
        )
        stats["clustering_sweep"] = self._safe_load_json(
            self.outputs_root / "clustering_sweep.json"
        )
        stats["split_summary"] = self._safe_load_json(
            self.splits_dir / "split_summary.json"
        )

        # ── Quality audit CSV ─────────────────────────────────────────────────
        qa_df = self._safe_load_csv(self.outputs_root / "quality_audit.csv")
        if qa_df is not None:
            qa_df["quality_pass"] = qa_df["quality_pass"].astype(bool)
            qa_df["severity_label"] = qa_df["severity_label"].astype(int)

            total = len(qa_df)
            quality_pass_n = int(qa_df["quality_pass"].sum())
            pct_corrupted = round((1 - quality_pass_n / total) * 100, 2) if total else 0.0

            # Per-flag counts (face_flag column)
            flag_counts: dict[str, int] = {}
            if "face_flag" in qa_df.columns:
                flag_counts = qa_df["face_flag"].value_counts().to_dict()
                flag_counts = {str(k): int(v) for k, v in flag_counts.items()}

            # Blur / exposure stats from quality_audit columns if present
            blur_stats: dict = {}
            if "blur_score" in qa_df.columns:
                blur_stats = {
                    "mean": round(float(qa_df["blur_score"].mean()), 2),
                    "std": round(float(qa_df["blur_score"].std()), 2),
                    "min": round(float(qa_df["blur_score"].min()), 2),
                    "max": round(float(qa_df["blur_score"].max()), 2),
                }
            exposure_stats: dict = {}
            if "mean_intensity" in qa_df.columns:
                exposure_stats = {
                    "mean": round(float(qa_df["mean_intensity"].mean()), 2),
                    "std": round(float(qa_df["mean_intensity"].std()), 2),
                }

            # Per-severity class counts in quality_pass images
            class_counts: dict[str, int] = {}
            for sev_lbl, sev_name in self.severity_map.items():
                class_counts[sev_name] = int(
                    (
                        (qa_df["severity_label"] == sev_lbl) & qa_df["quality_pass"]
                    ).sum()
                )

            stats["quality_audit"] = {
                "total": total,
                "quality_pass": quality_pass_n,
                "pct_corrupted": pct_corrupted,
                "flag_counts": flag_counts,
                "blur_stats": blur_stats,
                "exposure_stats": exposure_stats,
                "class_counts_quality_pass": class_counts,
            }
        else:
            stats["quality_audit"] = None

        # ── Cluster assignments ────────────────────────────────────────────────
        cl_df = self._safe_load_csv(self.outputs_root / "cluster_assignments.csv")
        if cl_df is not None:
            cl_df["cluster_id"] = cl_df["cluster_id"].astype(int)
            cl_df["embedding_extracted"] = cl_df["embedding_extracted"].astype(bool)

            total_emb = int(cl_df["embedding_extracted"].sum())
            images_in_multi = int(
                cl_df[(cl_df["cluster_id"] >= 0) & (cl_df["cluster_size"] > 1)].shape[0]
            )
            lri = round(images_in_multi / max(total_emb, 1) * 100, 4)
            n_clusters = int(cl_df[cl_df["cluster_id"] >= 0]["cluster_id"].nunique())

            stats["clustering"] = {
                "total_embeddings": total_emb,
                "lri": lri,
                "images_in_multi_clusters": images_in_multi,
                "n_clusters": n_clusters,
            }
        else:
            stats["clustering"] = None

        # Merge best_eps from sweep
        if stats["clustering_sweep"]:
            stats["clustering"] = stats.get("clustering") or {}
            stats["clustering"]["best_eps"] = stats["clustering_sweep"].get("best_eps")
            stats["clustering"]["best_silhouette"] = stats["clustering_sweep"].get(
                "best_silhouette"
            )

        self.logger.info("All stats loaded successfully.")
        return stats

    # ─────────────────────────────────────────────────────────────────────
    # Decision gates
    # ─────────────────────────────────────────────────────────────────────

    def _compute_decision_gates(self, stats: dict) -> dict:
        """
        Evaluate the five feasibility decision gates.

        Returns
        -------
        dict : {gate_name: bool, ...}
        """
        gates: dict[str, bool] = {}
        thr = self.thresholds

        # Gate 1: Sufficient images per class in train split
        min_per_class: int = int(thr["min_images_per_class"])
        gate1 = False
        if stats.get("split_summary"):
            train_dist = (
                stats["split_summary"]
                .get("splits", {})
                .get("train", {})
                .get("class_distribution", {})
            )
            if train_dist:
                gate1 = min(train_dist.values()) >= min_per_class
        gates["sufficient_images_all_classes"] = gate1

        # Gate 2: Acceptable imbalance ratio (max_class / min_class in training set)
        max_ir: float = float(thr["max_imbalance_ratio"])
        gate2 = False
        if stats.get("split_summary"):
            train_dist = (
                stats["split_summary"]
                .get("splits", {})
                .get("train", {})
                .get("class_distribution", {})
            )
            if train_dist and min(train_dist.values()) > 0:
                ir = max(train_dist.values()) / min(train_dist.values())
                gate2 = ir <= max_ir
        gates["acceptable_imbalance_ratio"] = gate2

        # Gate 3: Low corrupted image percentage
        max_corrupt: float = float(thr["max_corrupted_pct"])
        gate3 = False
        if stats.get("quality_audit") and stats["quality_audit"] is not None:
            pct = float(stats["quality_audit"].get("pct_corrupted", 100.0))
            gate3 = pct <= max_corrupt
        gates["low_corrupted_pct"] = gate3

        # Gate 4: Low leakage risk index
        max_lri: float = float(thr["max_leakage_risk_index"])
        gate4 = False
        if stats.get("clustering") and stats["clustering"] is not None:
            lri = float(stats["clustering"].get("lri", 100.0))
            gate4 = lri <= max_lri
        gates["low_leakage_risk"] = gate4

        # Gate 5: Split verification passed
        gate5 = False
        if stats.get("split_summary"):
            gate5 = bool(
                stats["split_summary"]
                .get("verification", {})
                .get("overall_pass", False)
            )
        gates["split_verification_passed"] = gate5

        self.logger.info("Decision gates: %s", gates)
        return gates

    # ─────────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────────

    def _determine_verdict(self, gates: dict) -> str:
        """
        FEASIBLE        → all gates pass
        NOT_FEASIBLE    → split_verification_passed=False OR
                          sufficient_images_all_classes=False
        MARGINAL        → any other gate fails
        """
        if all(gates.values()):
            return "FEASIBLE"
        if (
            not gates.get("split_verification_passed", True)
            or not gates.get("sufficient_images_all_classes", True)
        ):
            return "NOT_FEASIBLE"
        return "MARGINAL"

    # ─────────────────────────────────────────────────────────────────────
    # Recommendations
    # ─────────────────────────────────────────────────────────────────────

    def _build_recommendations(self, stats: dict, gates: dict) -> list[str]:
        """Dynamically build a prioritised list of Phase 1 recommendations."""
        recs: list[str] = []

        train_dist = (
            stats.get("split_summary", {})
            .get("splits", {})
            .get("train", {})
            .get("class_distribution", {})
        )

        # ── Imbalance ─────────────────────────────────────────────────────────
        if not gates.get("acceptable_imbalance_ratio", True):
            if train_dist and min(train_dist.values()) > 0:
                max_count = max(train_dist.values())
                weights = {
                    k: round(max_count / max(v, 1), 3) for k, v in train_dist.items()
                }
                recs.append(
                    f"High class imbalance detected (IR > {self.thresholds['max_imbalance_ratio']:.0f}). "
                    f"Use `WeightedRandomSampler` in Phase 1 DataLoader with weights: {weights}. "
                    "Also consider `torch.nn.CrossEntropyLoss(weight=class_weights)` to penalise "
                    "majority class over-confidence."
                )
            else:
                recs.append(
                    "High class imbalance detected. Use `WeightedRandomSampler` and a "
                    "class-weighted loss in Phase 1."
                )

        if train_dist:
            for c_name, c_cnt in train_dist.items():
                if c_cnt < 80:
                    recs.append(
                        f"Class '{c_name.replace('_', ' ').title()}' training images are critically low "
                        f"(n={c_cnt} < 80). Supplement with DermNet NZ acne images "
                        "or apply over-sampling with strong augmentation "
                        "(RandomHorizontalFlip, ColorJitter, RandomAffine)."
                    )

        # ── Leakage warnings ──────────────────────────────────────────────────
        lri = 0.0
        if stats.get("clustering") and stats["clustering"]:
            lri = float(stats["clustering"].get("lri", 0.0))

        if lri > 15.0:
            recs.append(
                f"CRITICAL — Leakage Risk Index is {lri:.1f}% (> 15%). "
                "A large fraction of images may belong to the same individual. "
                "Consider curating a dataset with verified unique subjects before "
                "proceeding to Phase 1."
            )
        elif lri > 5.0:
            recs.append(
                f"Moderate leakage risk (LRI={lri:.1f}%). "
                "Cluster-aware splitting has been applied to mitigate leakage, "
                "but treat reported metrics with caution — inter-subject variation "
                "may be underestimated."
            )

        # ── Frozen test set ───────────────────────────────────────────────────
        recs.append(
            "Test split is frozen — do NOT use for hyperparameter tuning, "
            "architecture search, or threshold calibration. "
            "Reserve exclusively for final model evaluation."
        )

        # ── Hardware-aware model recommendation ───────────────────────────────
        recs.append(
            "For RTX 3050 4 GB VRAM: Use EfficientNet-B2 (imagenet pretrained) "
            "with mixed precision (torch.cuda.amp). Batch size 32, gradient accumulation 2. "
            "Expected VRAM peak ≈ 3.6 GB — leaves headroom for augmentation on-GPU. "
            "If OOM: switch to EfficientNet-B1 or reduce crop to 224×224."
        )

        return recs

    # ─────────────────────────────────────────────────────────────────────
    # Markdown rendering
    # ─────────────────────────────────────────────────────────────────────

    def _gate_icon(self, passed: bool) -> str:
        return "✅" if passed else "❌"

    def _render_markdown(
        self,
        stats: dict,
        gates: dict,
        verdict: str,
        recommendations: list[str],
    ) -> str:
        lines: list[str] = []

        # ── helpers ───────────────────────────────────────────────────────────
        def h1(t: str) -> None:
            lines.append(f"# {t}\n")

        def h2(t: str) -> None:
            lines.append(f"\n## {t}\n")

        def h3(t: str) -> None:
            lines.append(f"\n### {t}\n")

        def para(*args: str) -> None:
            lines.append(" ".join(args) + "\n")

        def blank() -> None:
            lines.append("")

        # ── Title ─────────────────────────────────────────────────────────────
        h1("Phase 0 Feasibility Report — ACNE04 Dataset")
        para(
            f"**Generated:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
            f"**Seed:** {self.seed}",
        )

        # ─────────────────────────────────────────────────────────────────
        # 1. Executive Summary
        # ─────────────────────────────────────────────────────────────────
        h2("1. Executive Summary")

        verdict_badge = {
            "FEASIBLE": "🟢 **FEASIBLE**",
            "MARGINAL": "🟡 **MARGINAL**",
            "NOT_FEASIBLE": "🔴 **NOT FEASIBLE**",
        }.get(verdict, verdict)
        para(f"**Overall verdict:** {verdict_badge}")
        blank()

        para("**Decision Gates:**")
        blank()
        gate_labels = {
            "sufficient_images_all_classes": "Sufficient images in all training classes",
            "acceptable_imbalance_ratio": "Acceptable class imbalance ratio",
            "low_corrupted_pct": "Low corrupted image percentage",
            "low_leakage_risk": "Low face identity leakage risk",
            "split_verification_passed": "Split verification passed",
        }
        lines.append("| Gate | Status |")
        lines.append("|------|--------|")
        for gate_key, gate_label in gate_labels.items():
            icon = self._gate_icon(gates.get(gate_key, False))
            lines.append(f"| {gate_label} | {icon} |")
        blank()

        if stats.get("quality_audit"):
            qa = stats["quality_audit"]
            para(
                f"**Total images:** {qa['total']:,} — "
                f"**Quality pass:** {qa['quality_pass']:,} "
                f"({100 - qa['pct_corrupted']:.1f}%)"
            )

        para("**Top recommendations:**")
        for i, rec in enumerate(recommendations[:3], 1):
            lines.append(f"{i}. {rec}")
        blank()

        # ─────────────────────────────────────────────────────────────────
        # 2. Dataset Inventory
        # ─────────────────────────────────────────────────────────────────
        h2("2. Dataset Inventory")

        ing = stats.get("ingestion_log")
        if ing:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in ing.items():
                if not isinstance(v, dict):
                    lines.append(f"| {k} | {v} |")
            blank()
        else:
            para("_ingestion_log.json not available._")

        # ─────────────────────────────────────────────────────────────────
        # 3. Note on sim_acne.csv
        # ─────────────────────────────────────────────────────────────────
        h2("3. Note on `sim_acne.csv`")
        para(
            "`sim_acne.csv` has been **excluded** from all training artefacts. "
            "Investigation confirmed it is a single-subject (N-of-1) longitudinal "
            "trial artefact containing repeated photos of one individual across multiple "
            "severity grades. Including it would introduce severe within-subject leakage "
            "and artificially inflate cross-class identity overlap."
        )

        # ─────────────────────────────────────────────────────────────────
        # 4. Class Distribution
        # ─────────────────────────────────────────────────────────────────
        h2("4. Class Distribution (Quality-Pass Images)")

        qa = stats.get("quality_audit")
        if qa and qa.get("class_counts_quality_pass"):
            cc = qa["class_counts_quality_pass"]
            total_qp = sum(cc.values())
            max_count = max(cc.values()) if cc else 1
            min_count = max(min(cc.values()), 1) if cc else 1
            ir = round(max_count / min_count, 2)

            lines.append("| Class | Severity | Count | % of QP Total |")
            lines.append("|-------|----------|-------|---------------|")
            for sev_lbl, sev_name in self.severity_map.items():
                count = cc.get(sev_name, 0)
                pct = round(count / total_qp * 100, 1) if total_qp else 0.0
                lines.append(f"| {sev_lbl} | {sev_name.replace('_', ' ').title()} | {count:,} | {pct}% |")
            blank()

            para(f"**Imbalance Ratio (max/min):** {ir:.2f}")

            # Class weights
            if min_count > 0:
                weights = {
                    sev_name.replace("_", " ").title(): round(max_count / max(cc.get(sev_name, 1), 1), 3)
                    for sev_name in self.severity_map.values()
                }
                para(f"**Suggested class weights:** `{weights}`")
        else:
            para("_Quality audit data not available._")

        # ─────────────────────────────────────────────────────────────────
        # 5. Image Quality
        # ─────────────────────────────────────────────────────────────────
        h2("5. Image Quality Analysis")

        if qa:
            flag_counts = qa.get("flag_counts", {})
            if flag_counts:
                lines.append("| Face Flag | Count |")
                lines.append("|-----------|-------|")
                for flag, cnt in sorted(flag_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"| `{flag}` | {cnt:,} |")
                blank()

            blur = qa.get("blur_stats", {})
            if blur:
                para(
                    f"**Blur score (Laplacian variance):** "
                    f"mean={blur['mean']}, std={blur['std']}, "
                    f"min={blur['min']}, max={blur['max']}"
                )

            exp = qa.get("exposure_stats", {})
            if exp:
                para(
                    f"**Mean pixel intensity:** "
                    f"mean={exp['mean']}, std={exp['std']}"
                )

            para(
                f"**Corrupted / filtered images:** "
                f"{qa['total'] - qa['quality_pass']:,} "
                f"({qa['pct_corrupted']}% of total) "
                f"{self._gate_icon(gates.get('low_corrupted_pct', False))}"
            )
        else:
            para("_Quality audit data not available._")

        # ─────────────────────────────────────────────────────────────────
        # 6. Face Clustering & Leakage Analysis
        # ─────────────────────────────────────────────────────────────────
        h2("6. Face Clustering & Leakage Analysis")

        cl = stats.get("clustering")
        if cl:
            best_eps = cl.get("best_eps", "N/A")
            best_sil = cl.get("best_silhouette")
            lri_val = cl.get("lri", 0.0)
            n_clusters = cl.get("n_clusters", 0)

            if lri_val <= 5.0:
                leakage_verdict = "✅ SAFE"
            elif lri_val <= 15.0:
                leakage_verdict = "⚠️ AT_RISK"
            else:
                leakage_verdict = "❌ HIGH_RISK"

            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| InsightFace model | `{self.config['clustering']['embedding_model']}` |")
            lines.append(f"| DBSCAN best ε | {best_eps} |")
            lines.append(f"| Best silhouette score | {f'{best_sil:.4f}' if best_sil is not None else 'N/A'} |")
            lines.append(f"| Non-noise clusters | {n_clusters:,} |")
            lines.append(f"| Images in multi-image clusters | {cl.get('images_in_multi_clusters', 'N/A'):,} |")
            lines.append(f"| Leakage Risk Index (LRI) | {lri_val:.2f}% |")
            lines.append(f"| Leakage verdict | {leakage_verdict} |")
            blank()

            # DBSCAN sweep table
            sweep = stats.get("clustering_sweep")
            if sweep and sweep.get("params"):
                h3("DBSCAN Hyperparameter Sweep")
                lines.append("| ε | Clusters | Noise | Silhouette |")
                lines.append("|---|----------|-------|------------|")
                for p in sweep["params"]:
                    sil_str = f"{p['silhouette_score']:.4f}" if p["silhouette_score"] is not None else "N/A"
                    star = " ★" if p["eps"] == best_eps else ""
                    lines.append(
                        f"| {p['eps']}{star} | {p['n_clusters']} | {p['n_noise']} | {sil_str} |"
                    )
                blank()
        else:
            para("_Clustering data not available._")

        # ─────────────────────────────────────────────────────────────────
        # 7. Split Summary
        # ─────────────────────────────────────────────────────────────────
        h2("7. Train / Val / Test Split Summary")

        ss = stats.get("split_summary")
        if ss:
            splits_data = ss.get("splits", {})
            sev_names_ordered = [self.severity_map[i] for i in sorted(self.severity_map.keys())]
            header_cols = ["Split", "Total", "Clusters"] + [
                sn.replace("_", " ").title() for sn in sev_names_ordered
            ]
            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
            for split_name in SPLIT_NAMES:
                info = splits_data.get(split_name, {})
                cd = info.get("class_distribution", {})
                row_vals = [
                    split_name.capitalize(),
                    str(info.get("n_images", 0)),
                    str(info.get("n_clusters", 0)),
                ] + [str(cd.get(sn, 0)) for sn in sev_names_ordered]
                lines.append("| " + " | ".join(row_vals) + " |")
            blank()

            verification = ss.get("verification", {})
            overall = verification.get("overall_pass", False)
            para(f"**Split verification:** {self._gate_icon(overall)}")
            for check_name, result in verification.items():
                if check_name == "overall_pass":
                    continue
                icon = self._gate_icon(bool(result))
                para(f"  - {check_name.replace('_', ' ').title()}: {icon}")
        else:
            para("_Split summary not available._")

        # ─────────────────────────────────────────────────────────────────
        # 8. Phase 1 Recommendations
        # ─────────────────────────────────────────────────────────────────
        h2("8. Phase 1 Recommendations")

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
            blank()

        blank()
        lines.append("---")
        para("*End of Phase 0 Feasibility Report*")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Load all stats, evaluate gates, render and save the feasibility
        report in both Markdown and JSON formats.

        Returns
        -------
        report_json : dict
        """
        # 1. Load
        stats = self._load_all_stats()

        # 2. Gates
        gates = self._compute_decision_gates(stats)

        # 3. Verdict
        verdict = self._determine_verdict(gates)

        # 4. Recommendations
        recommendations = self._build_recommendations(stats, gates)

        # 5. Render Markdown
        markdown_str = self._render_markdown(stats, gates, verdict, recommendations)

        # 6. Save Markdown
        md_path = self.reports_dir / "feasibility_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_str)
        self.logger.info("Markdown report saved → %s", md_path)

        # 7. Build and save JSON
        report_json: dict = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "verdict": verdict,
            "gates": gates,
            "recommendations": recommendations,
            "stats": {
                "quality_audit": stats.get("quality_audit"),
                "clustering": stats.get("clustering"),
                "split_summary": stats.get("split_summary"),
                "ingestion_log": stats.get("ingestion_log"),
            },
            "thresholds": self.thresholds,
        }

        json_dir = self.outputs_root / "reports"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / "feasibility_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, default=str)
        self.logger.info("JSON report saved → %s", json_path)

        # 8. Prominent verdict log
        verdict_line = {
            "FEASIBLE": "🟢  VERDICT: FEASIBLE — proceed to Phase 1 training.",
            "MARGINAL": "🟡  VERDICT: MARGINAL — review recommendations before Phase 1.",
            "NOT_FEASIBLE": "🔴  VERDICT: NOT FEASIBLE — address critical issues first.",
        }.get(verdict, verdict)

        self.logger.info(
            "\n"
            "═══════════════════════════════════════════════════════════\n"
            "  %s\n"
            "═══════════════════════════════════════════════════════════",
            verdict_line,
        )

        return report_json


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_report_generation(
    config_path: Path,
    project_root: Path,
) -> dict:
    """
    Load config and execute the full report-generation pipeline.

    Parameters
    ----------
    config_path  : Path to phase0.yaml
    project_root : Absolute path to the project root

    Returns
    -------
    report_json : dict
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generator = FeasibilityReportGenerator(config=config, project_root=project_root)
    return generator.run()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0 — Feasibility Report")
    parser.add_argument("--config", default="phase0/config/phase0.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    run_report_generation(
        config_path=Path(args.config),
        project_root=Path(args.project_root).resolve(),
    )

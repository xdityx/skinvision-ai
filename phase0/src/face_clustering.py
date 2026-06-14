"""
phase0/src/face_clustering.py
─────────────────────────────
Face embedding extraction via InsightFace (buffalo_sc) and DBSCAN-based
identity clustering to quantify cross-split leakage risk.

Usage
-----
    from phase0.src.face_clustering import run_face_clustering
    df = run_face_clustering(config_path=Path("phase0/config/phase0.yaml"),
                             project_root=Path("."))
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ── internal utilities ────────────────────────────────────────────────────────
from phase0.src.utils.logging import get_logger
from phase0.src.utils.io import safe_load_cv2
from phase0.src.utils.visualisation import save_tsne_plot, save_bar_chart


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_MAP: dict[int, str] = {0: "mild", 1: "moderate", 2: "severe"}


# ─────────────────────────────────────────────────────────────────────────────
class FaceClusterer:
    """
    Orchestrates InsightFace embedding extraction, DBSCAN hyperparameter
    sweep, leakage-risk analysis, and diagnostic visualisations for the
    ACNE04 dataset.
    """

    def __init__(
        self,
        quality_audit_path: Path,
        config: dict,
        project_root: Path,
    ) -> None:
        self.logger = get_logger(__name__)
        self.config = config
        self.project_root = project_root

        # ── paths ─────────────────────────────────────────────────────────────
        path_cfg = config["paths"]
        self.outputs_root: Path = project_root / path_cfg["outputs_root"]
        self.figures_dir: Path = project_root / path_cfg["figures_dir"]
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # ── clustering config ─────────────────────────────────────────────────
        self.clustering_cfg = config["clustering"]
        self.seed: int = config["project"]["seed"]

        # ── quality audit ─────────────────────────────────────────────────────
        self.logger.info("Loading quality audit: %s", quality_audit_path)
        self.quality_df: pd.DataFrame = pd.read_csv(quality_audit_path)
        # Normalise column types
        self.quality_df["quality_pass"] = self.quality_df["quality_pass"].astype(bool)
        self.quality_df["severity_label"] = self.quality_df["severity_label"].astype(int)
        self.logger.info(
            "Quality audit loaded: %d total images, %d quality_pass=True",
            len(self.quality_df),
            self.quality_df["quality_pass"].sum(),
        )

        raw_map = config.get("ingestion", {}).get("severity_map", {0: "mild", 1: "moderate", 2: "severe"})
        self.severity_map = {int(k): v for k, v in raw_map.items()}

    # ─────────────────────────────────────────────────────────────────────
    # InsightFace initialisation
    # ─────────────────────────────────────────────────────────────────────

    def _init_insightface(self) -> Any:
        """
        Initialise InsightFace FaceAnalysis with CUDA → CPU fallback.

        Returns
        -------
        app : insightface.app.FaceAnalysis
        """
        try:
            import insightface  # noqa: PLC0415
            from insightface.app import FaceAnalysis  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. "
                "Install it with:  pip install insightface onnxruntime"
            ) from exc

        model_name: str = self.clustering_cfg["embedding_model"]
        self.logger.info("Initialising InsightFace model: %s", model_name)

        det_size_raw = self.clustering_cfg.get("det_size", [640, 640])
        det_size = (int(det_size_raw[0]), int(det_size_raw[1]))
        self.logger.info("Setting InsightFace detection size to %s", det_size)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name=model_name, providers=providers)
        app.prepare(ctx_id=0, det_size=det_size)

        self.logger.info("InsightFace model ready.")
        return app

    # ─────────────────────────────────────────────────────────────────────
    # Embedding extraction
    # ─────────────────────────────────────────────────────────────────────

    def _extract_embeddings(
        self, app: Any
    ) -> tuple[np.ndarray, list[str]]:
        """
        Iterate over quality-pass images, extract the first-face embedding.

        Returns
        -------
        embeddings : np.ndarray, shape (N, 512)
        image_ids  : list[str], length N
        """
        eligible = self.quality_df[
            (self.quality_df["quality_pass"]) & (self.quality_df["face_flag"] == "ok")
        ].copy()

        self.logger.info(
            "Extracting embeddings from %d eligible images …", len(eligible)
        )

        embeddings: list[np.ndarray] = []
        image_ids: list[str] = []

        for _, row in tqdm(
            eligible.iterrows(), total=len(eligible), desc="Extracting embeddings"
        ):
            img_id: str = str(row["image_id"])
            img_path: Path = Path(row["image_path"])

            img = safe_load_cv2(img_path)
            if img is None:
                self.logger.warning("Could not load image %s — skipping.", img_id)
                continue

            faces = app.get(img)
            if not faces:
                self.logger.warning(
                    "No face detected in image %s — skipping.", img_id
                )
                continue

            emb: np.ndarray = faces[0].normed_embedding  # already L2-normalised, 512-d
            embeddings.append(emb)
            image_ids.append(img_id)

        if not embeddings:
            raise RuntimeError(
                "No embeddings were extracted. "
                "Check image paths and face detection configuration."
            )

        embeddings_array = np.stack(embeddings, axis=0)
        self.logger.info(
            "Extracted %d embeddings (shape %s).", len(image_ids), embeddings_array.shape
        )
        return embeddings_array, image_ids

    # ─────────────────────────────────────────────────────────────────────
    # Cache logic
    # ─────────────────────────────────────────────────────────────────────

    def _load_or_extract_embeddings(
        self, app: Any
    ) -> tuple[np.ndarray, list[str]]:
        """
        Return cached embeddings if available and caching is enabled;
        otherwise run extraction and persist to disk.
        """
        npy_path = self.outputs_root / "face_embeddings.npy"
        idx_path = self.outputs_root / "embeddings_index.csv"
        use_cache: bool = bool(self.clustering_cfg.get("embedding_cache", True))

        if use_cache and npy_path.exists() and idx_path.exists():
            self.logger.info("Loading cached embeddings from %s", npy_path)
            embeddings = np.load(npy_path)
            idx_df = pd.read_csv(idx_path)
            image_ids: list[str] = idx_df["image_id"].tolist()
            self.logger.info(
                "Cache hit: %d embeddings loaded.", len(image_ids)
            )
            return embeddings, image_ids

        self.logger.info("Cache miss — running full embedding extraction.")
        embeddings, image_ids = self._extract_embeddings(app)

        # Persist
        np.save(npy_path, embeddings)
        idx_df = pd.DataFrame(
            {"row_idx": list(range(len(image_ids))), "image_id": image_ids}
        )
        idx_df.to_csv(idx_path, index=False)
        self.logger.info("Embeddings saved → %s | index → %s", npy_path, idx_path)
        return embeddings, image_ids

    # ─────────────────────────────────────────────────────────────────────
    # DBSCAN sweep
    # ─────────────────────────────────────────────────────────────────────

    def _run_dbscan_sweep(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Sweep eps values via DBSCAN (cosine metric) and choose the value that
        maximises silhouette score.

        Returns
        -------
        best_labels   : np.ndarray, shape (N,)
        sweep_results : dict  (serialisable)
        """
        from sklearn.cluster import DBSCAN  # noqa: PLC0415
        from sklearn.metrics import silhouette_score  # noqa: PLC0415

        eps_values: list[float] = self.clustering_cfg["dbscan_eps_sweep"]
        min_samples: int = int(self.clustering_cfg["dbscan_min_samples"])

        sweep_results: dict = {"params": [], "best_eps": None}
        best_silhouette = -1.0
        best_eps = eps_values[0]
        best_labels: np.ndarray | None = None

        self.logger.info(
            "Starting DBSCAN sweep over eps=%s with min_samples=%d",
            eps_values, min_samples,
        )

        for eps in eps_values:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
            labels = db.fit_predict(embeddings)

            unique_labels = set(labels)
            n_clusters = len(unique_labels - {-1})
            n_noise = int((labels == -1).sum())

            sil: float | None = None
            if n_clusters >= 2:
                # Silhouette needs at least 2 non-noise samples per cluster; guard:
                non_noise_mask = labels != -1
                if non_noise_mask.sum() >= 2 and n_clusters >= 2:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            sil = float(
                                silhouette_score(
                                    embeddings[non_noise_mask],
                                    labels[non_noise_mask],
                                    metric="cosine",
                                    sample_size=min(2000, non_noise_mask.sum()),
                                    random_state=self.seed,
                                )
                            )
                        except Exception:
                            sil = None

            record = {
                "eps": eps,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette_score": sil,
            }
            sweep_results["params"].append(record)

            self.logger.info(
                "  eps=%.2f → clusters=%d  noise=%d  silhouette=%s",
                eps, n_clusters, n_noise,
                f"{sil:.4f}" if sil is not None else "N/A",
            )

            if sil is not None and sil > best_silhouette:
                best_silhouette = sil
                best_eps = eps

        self.logger.info("Best eps=%.2f (silhouette=%.4f)", best_eps, best_silhouette)
        sweep_results["best_eps"] = best_eps
        sweep_results["best_silhouette"] = best_silhouette

        # Refit with best eps
        db_best = DBSCAN(
            eps=best_eps, min_samples=min_samples, metric="cosine", n_jobs=-1
        )
        best_labels = db_best.fit_predict(embeddings)

        # Persist sweep results
        sweep_path = self.outputs_root / "clustering_sweep.json"
        with open(sweep_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2)
        self.logger.info("Sweep results saved → %s", sweep_path)

        return best_labels, sweep_results

    # ─────────────────────────────────────────────────────────────────────
    # Leakage Risk Index
    # ─────────────────────────────────────────────────────────────────────

    def _compute_leakage_risk_index(
        self, labels: np.ndarray, image_ids: list[str]
    ) -> dict:
        """
        LRI = (images_in_multi_image_clusters / total_images) × 100

        Returns dict with lri, multi_image_clusters, images_in_multi_clusters,
        total_images.
        """
        total_images = len(image_ids)
        cluster_sizes: dict[int, int] = {}
        for lbl in labels:
            if lbl < 0:
                continue
            cluster_sizes[lbl] = cluster_sizes.get(lbl, 0) + 1

        multi_image_clusters = sum(1 for s in cluster_sizes.values() if s > 1)
        images_in_multi = sum(s for s in cluster_sizes.values() if s > 1)
        lri = (images_in_multi / total_images * 100) if total_images else 0.0

        result = {
            "lri": round(lri, 4),
            "multi_image_clusters": multi_image_clusters,
            "images_in_multi_clusters": images_in_multi,
            "total_images": total_images,
        }
        self.logger.info(
            "Leakage Risk Index: %.2f%% (%d images in %d multi-image clusters)",
            lri, images_in_multi, multi_image_clusters,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Cluster assignments CSV
    # ─────────────────────────────────────────────────────────────────────

    def _save_cluster_assignments(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        image_ids: list[str],
    ) -> pd.DataFrame:
        """
        Build and persist cluster_assignments.csv containing every image in
        quality_audit.csv (not only those with embeddings).

        Returns the full cluster assignments DataFrame.
        """
        # ── Build per-embedding rows ──────────────────────────────────────────
        emb_df = pd.DataFrame({"image_id": image_ids, "cluster_id": labels.tolist()})
        emb_df["embedding_extracted"] = True

        # ── Cluster sizes (label -1 = noise → singleton) ──────────────────────
        size_map: dict[int, int] = (
            emb_df[emb_df["cluster_id"] >= 0]
            .groupby("cluster_id")["image_id"]
            .count()
            .to_dict()
        )
        # Noise / unclustered images: size = 1
        emb_df["cluster_size"] = emb_df["cluster_id"].apply(
            lambda c: size_map.get(c, 1) if c >= 0 else 1
        )

        # ── Intra-cluster severity range ──────────────────────────────────────
        # Join severity from quality_df
        sev_map: dict[str, int] = (
            self.quality_df.set_index("image_id")["severity_label"].to_dict()
        )
        emb_df["severity_label"] = emb_df["image_id"].map(sev_map)

        # Compute per-cluster severity range
        cluster_sev_range: dict[int, int] = {}
        for cid, grp in emb_df[emb_df["cluster_id"] >= 0].groupby("cluster_id"):
            mn = grp["severity_label"].min()
            mx = grp["severity_label"].max()
            cluster_sev_range[int(cid)] = int(mx - mn)

        emb_df["intra_cluster_severity_range"] = emb_df["cluster_id"].apply(
            lambda c: cluster_sev_range.get(int(c), 0) if c >= 0 else 0
        )
        emb_df["leakage_risk_flag"] = emb_df["cluster_size"] > 1

        # ── Add images with no embedding ──────────────────────────────────────
        all_ids = set(self.quality_df["image_id"].astype(str))
        extracted_ids = set(image_ids)
        missing_ids = all_ids - extracted_ids

        if missing_ids:
            missing_rows = self.quality_df[
                self.quality_df["image_id"].astype(str).isin(missing_ids)
            ][["image_id", "severity_label"]].copy()
            missing_rows["image_id"] = missing_rows["image_id"].astype(str)
            missing_rows["cluster_id"] = -1
            missing_rows["cluster_size"] = 1
            missing_rows["embedding_extracted"] = False
            missing_rows["intra_cluster_severity_range"] = 0
            missing_rows["leakage_risk_flag"] = False
            emb_df = pd.concat([emb_df, missing_rows], ignore_index=True)

        # ── severity_name ─────────────────────────────────────────────────────
        emb_df["severity_name"] = emb_df["severity_label"].map(self.severity_map)

        # ── Final column order ────────────────────────────────────────────────
        out_cols = [
            "image_id",
            "cluster_id",
            "cluster_size",
            "embedding_extracted",
            "severity_label",
            "severity_name",
            "intra_cluster_severity_range",
            "leakage_risk_flag",
        ]
        out_df = emb_df[out_cols].reset_index(drop=True)

        out_path = self.outputs_root / "cluster_assignments.csv"
        out_df.to_csv(out_path, index=False)
        self.logger.info("Cluster assignments saved → %s  (%d rows)", out_path, len(out_df))
        return out_df

    # ─────────────────────────────────────────────────────────────────────
    # Visualisations
    # ─────────────────────────────────────────────────────────────────────

    def _save_visualisations(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        image_ids: list[str],
        cluster_df: pd.DataFrame,
    ) -> None:
        """Produce t-SNE plots, cluster-size histogram, and dendrogram."""
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib.cm as cm  # noqa: PLC0415
        from sklearn.manifold import TSNE  # noqa: PLC0415
        import scipy.cluster.hierarchy as sch  # noqa: PLC0415

        rng = np.random.default_rng(self.clustering_cfg["tsne_seed"])
        n_samples = len(embeddings)
        max_tsne = 1000

        # ── Sub-sample for t-SNE if needed ───────────────────────────────────
        if n_samples > max_tsne:
            idx = rng.choice(n_samples, size=max_tsne, replace=False)
            emb_sub = embeddings[idx]
            labels_sub = labels[idx]
            ids_sub = [image_ids[i] for i in idx]
        else:
            emb_sub = embeddings
            labels_sub = labels
            ids_sub = image_ids

        self.logger.info(
            "Running t-SNE on %d samples (perplexity=%d) …",
            len(emb_sub), self.clustering_cfg["tsne_perplexity"],
        )
        tsne = TSNE(
            n_components=2,
            perplexity=min(self.clustering_cfg["tsne_perplexity"], len(emb_sub) - 1),
            random_state=self.clustering_cfg["tsne_seed"],
            n_jobs=-1,
        )
        coords_2d = tsne.fit_transform(emb_sub)

        # ── t-SNE coloured by cluster ─────────────────────────────────────────
        unique_clusters = sorted(set(labels_sub))
        n_colors = len(unique_clusters)
        cmap = cm.get_cmap("tab20", max(n_colors, 1))
        cluster_color_map = {c: cmap(i) for i, c in enumerate(unique_clusters)}

        fig, ax = plt.subplots(figsize=(10, 8))
        for cid in unique_clusters:
            mask = labels_sub == cid
            label = f"Noise" if cid == -1 else f"Cluster {cid}"
            color = "lightgray" if cid == -1 else cluster_color_map[cid]
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=[color], label=label, s=18, alpha=0.7, linewidths=0,
            )
        ax.set_title("t-SNE — coloured by cluster identity", fontsize=14)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        if n_colors <= 20:
            ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        out_path = self.figures_dir / "04_tsne_by_cluster.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved → %s", out_path)

        # ── t-SNE coloured by severity ────────────────────────────────────────
        sev_map_local: dict[str, int] = (
            self.quality_df.set_index("image_id")["severity_label"].to_dict()
        )
        sevs_sub = np.array(
            [sev_map_local.get(iid, -1) for iid in ids_sub], dtype=float
        )
        color_list = ["#4CAF50", "#FFC107", "#FF5722", "#B71C1C"]
        sev_cmap = {lbl: color_list[idx % len(color_list)] for idx, lbl in enumerate(sorted(self.severity_map.keys()))}
        sev_labels_text = {lbl: name.replace("_", " ").title() for lbl, name in self.severity_map.items()}

        fig, ax = plt.subplots(figsize=(10, 8))
        for sev_lbl in sorted(self.severity_map.keys()):
            mask = sevs_sub == sev_lbl
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=sev_cmap[sev_lbl],
                label=sev_labels_text[sev_lbl],
                s=18, alpha=0.7, linewidths=0,
            )
        unknown_mask = sevs_sub == -1
        if unknown_mask.sum():
            ax.scatter(
                coords_2d[unknown_mask, 0], coords_2d[unknown_mask, 1],
                c="lightgray", label="Unknown", s=18, alpha=0.5, linewidths=0,
            )
        ax.set_title("t-SNE — coloured by severity label", fontsize=14)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_path = self.figures_dir / "04_tsne_by_severity.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved → %s", out_path)

        # ── Cluster size histogram ────────────────────────────────────────────
        non_noise_df = cluster_df[cluster_df["cluster_id"] >= 0]
        if not non_noise_df.empty:
            cluster_sizes = (
                non_noise_df.groupby("cluster_id")["image_id"].count().values
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(cluster_sizes, bins=range(1, cluster_sizes.max() + 2),
                    color="#5C6BC0", edgecolor="white", rwidth=0.85)
            ax.set_title("Cluster Size Distribution (non-noise clusters)", fontsize=13)
            ax.set_xlabel("Cluster size (# images)")
            ax.set_ylabel("Count")
            ax.set_xticks(range(1, cluster_sizes.max() + 1))
            fig.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No non-noise clusters found.",
                    ha="center", va="center", fontsize=12)
        out_path = self.figures_dir / "04_cluster_size_histogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved → %s", out_path)

        # ── Dendrogram ────────────────────────────────────────────────────────
        max_dend = int(self.clustering_cfg.get("dendrogram_max_samples", 200))
        n_dend = min(n_samples, max_dend)
        dend_idx = rng.choice(n_samples, size=n_dend, replace=False)
        dend_sub = embeddings[dend_idx]

        fig, ax = plt.subplots(figsize=(14, 6))
        linkage_matrix = sch.linkage(dend_sub, method="ward")
        sch.dendrogram(
            linkage_matrix,
            ax=ax,
            truncate_mode="lastp",
            p=30,
            leaf_rotation=90.0,
            leaf_font_size=8,
            show_contracted=True,
        )
        ax.set_title(
            f"Hierarchical Clustering Dendrogram (subsample n={n_dend})", fontsize=13
        )
        ax.set_xlabel("Image index")
        ax.set_ylabel("Ward distance")
        fig.tight_layout()
        out_path = self.figures_dir / "04_cluster_dendrogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved → %s", out_path)

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Full clustering pipeline.

        Returns
        -------
        cluster_df : pd.DataFrame  (cluster_assignments.csv contents)
        """
        # 1. Initialise model
        app = self._init_insightface()

        # 2. Load / extract embeddings
        embeddings, image_ids = self._load_or_extract_embeddings(app)

        # 3. DBSCAN sweep
        labels, sweep_results = self._run_dbscan_sweep(embeddings)

        # 4. Leakage Risk Index
        lri_info = self._compute_leakage_risk_index(labels, image_ids)

        # 5. Cluster assignments CSV
        cluster_df = self._save_cluster_assignments(embeddings, labels, image_ids)

        # 6. Visualisations
        self._save_visualisations(embeddings, labels, image_ids, cluster_df)

        # 7. Final summary log
        best_eps: float = sweep_results["best_eps"]
        n_clusters: int = sweep_results["params"][
            next(
                i for i, p in enumerate(sweep_results["params"])
                if p["eps"] == best_eps
            )
        ]["n_clusters"]
        lri = lri_info["lri"]

        if lri <= 5.0:
            leakage_verdict = "SAFE"
        elif lri <= 15.0:
            leakage_verdict = "AT_RISK"
        else:
            leakage_verdict = "HIGH_RISK"

        self.logger.info(
            "\n"
            "══════════════════ Face Clustering Summary ══════════════════\n"
            "  Best eps          : %.2f\n"
            "  N clusters        : %d\n"
            "  Leakage Risk Index: %.2f%%\n"
            "  Leakage verdict   : %s\n"
            "═════════════════════════════════════════════════════════════",
            best_eps, n_clusters, lri, leakage_verdict,
        )

        return cluster_df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_face_clustering(
    config_path: Path,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load config from *config_path* and execute the full face-clustering
    pipeline relative to *project_root*.

    Parameters
    ----------
    config_path  : Path to phase0.yaml (absolute or relative to cwd)
    project_root : Absolute path to the project root directory

    Returns
    -------
    cluster_df : pd.DataFrame
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    quality_audit_path = (
        project_root / config["paths"]["outputs_root"] / "quality_audit.csv"
    )

    clusterer = FaceClusterer(
        quality_audit_path=quality_audit_path,
        config=config,
        project_root=project_root,
    )
    return clusterer.run()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 0 — Face Clustering")
    parser.add_argument(
        "--config",
        default="phase0/config/phase0.yaml",
        help="Path to phase0.yaml",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory",
    )
    args = parser.parse_args()

    run_face_clustering(
        config_path=Path(args.config),
        project_root=Path(args.project_root).resolve(),
    )

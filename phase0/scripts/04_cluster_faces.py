#!/usr/bin/env python3
"""Run InsightFace-based face clustering on the ACNE04 dataset.

Extracts face embeddings (or loads from cache), runs DBSCAN with an
eps sweep, and writes cluster_assignments.csv to data/phase0_outputs/.
"""
import argparse
import sys
import time
from pathlib import Path

# Add phase0/src to Python path
PHASE0_ROOT = Path(__file__).parent.parent
SRC_ROOT = PHASE0_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Also add project root for consistent data/ paths
PROJECT_ROOT = PHASE0_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import get_logger
from ingestion.manifest_builder import load_config, get_project_root
from face_clustering import run_face_clustering

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 face clustering."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PHASE0_ROOT / "config" / "phase0.yaml",
        help="Path to phase0.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cluster_assignments.csv and delete cached embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    logger.info(f"Starting face clustering  config={args.config}")

    cfg = load_config(args.config)
    project_root = get_project_root()
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    assignments_path = outputs_root / "cluster_assignments.csv"

    if assignments_path.exists() and not args.force:
        logger.info(
            f"cluster_assignments.csv already exists at {assignments_path}. "
            "Use --force to overwrite. Skipping."
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Done in {elapsed:.1f}s (skipped)")
        return

    # If --force: also delete cached embeddings so they are re-extracted
    if args.force:
        embeddings_cache = outputs_root / "embeddings.npy"
        if embeddings_cache.exists():
            embeddings_cache.unlink()
            logger.info(f"Deleted embedding cache at {embeddings_cache}")

        # Also delete any per-image npy caches in outputs dir
        for npy_file in outputs_root.glob("*.npy"):
            npy_file.unlink()
            logger.info(f"Deleted cached embeddings: {npy_file}")

    run_face_clustering(args.config, project_root)

    # Print LRI and cluster verdict
    if assignments_path.exists():
        import pandas as pd
        df = pd.read_csv(assignments_path)

        n_clusters = df["cluster_id"].nunique() if "cluster_id" in df.columns else "N/A"
        lri = "N/A"
        verdict = "N/A"

        # Try reading from clustering_summary.json if present
        summary_path = outputs_root / "clustering_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            lri = summary.get("leakage_risk_index", lri)
            verdict = summary.get("verdict", verdict)

        print(f"\nFace clustering results:")
        print(f"  Clusters found        : {n_clusters}")
        print(f"  Leakage Risk Index    : {lri}")
        print(f"  Verdict               : {verdict}")

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

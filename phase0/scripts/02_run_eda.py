#!/usr/bin/env python3
"""Run exploratory data analysis on the ACNE04 dataset.

Reads manifest.csv, computes class distribution, resolution statistics,
and writes plots + eda_stats.json to data/phase0_outputs/.
"""
import argparse
import json
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
from eda import run_eda

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 exploratory data analysis."
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
        help="Overwrite existing eda_stats.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    logger.info(f"Starting EDA  config={args.config}")

    cfg = load_config(args.config)
    project_root = get_project_root()
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    eda_stats_path = outputs_root / "eda_stats.json"

    if eda_stats_path.exists() and not args.force:
        logger.info(
            f"eda_stats.json already exists at {eda_stats_path}. "
            "Use --force to overwrite. Skipping."
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Done in {elapsed:.1f}s (skipped)")
        return

    run_eda(args.config, project_root)

    # Print summary from the generated JSON
    if eda_stats_path.exists():
        with open(eda_stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        ir = stats.get("class_distribution", {}).get("imbalance_ratio", "N/A")
        print(f"\nImbalance Ratio (IR) : {ir}")

        res_stats = stats.get("resolution", {})
        width_stats = res_stats.get("width_stats", {})
        height_stats = res_stats.get("height_stats", {})
        if width_stats:
            print("Resolution summary (width):")
            for k, v in width_stats.items():
                print(f"  {k}: {v}")
        if height_stats:
            print("Resolution summary (height):")
            for k, v in height_stats.items():
                print(f"  {k}: {v}")

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

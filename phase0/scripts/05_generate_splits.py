#!/usr/bin/env python3
"""Generate train/val/test splits respecting cluster boundaries.

Reads manifest.csv and cluster_assignments.csv, performs cluster-aware
stratified splitting, and writes splits/ CSVs to data/phase0_outputs/.
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
from split_generator import run_split_generation

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 train/val/test split generation."
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
        help="Overwrite existing splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    logger.info(f"Starting split generation  config={args.config}")

    cfg = load_config(args.config)
    project_root = get_project_root()
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    train_path = outputs_root / "splits" / "train.csv"

    if train_path.exists() and not args.force:
        logger.info(
            f"splits/train.csv already exists at {train_path}. "
            "Use --force to overwrite. Skipping."
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Done in {elapsed:.1f}s (skipped)")
        return

    run_split_generation(args.config, project_root)

    # Print split counts per class
    splits_dir = outputs_root / "splits"
    split_names = ["train", "val", "test"]
    print("\nSplit counts:")

    for split_name in split_names:
        split_file = splits_dir / f"{split_name}.csv"
        if split_file.exists():
            import pandas as pd
            df = pd.read_csv(split_file)
            total = len(df)
            print(f"\n  {split_name.upper()} ({total} images):")
            if "severity" in df.columns:
                per_class = df.groupby("severity").size()
                for sev, cnt in per_class.items():
                    print(f"    severity={sev}: {cnt}")
            elif "label" in df.columns:
                per_class = df.groupby("label").size()
                for label, cnt in per_class.items():
                    print(f"    {label}: {cnt}")

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

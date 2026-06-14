#!/usr/bin/env python3
"""Audit image quality across the ACNE04 dataset.

Checks for blur, exposure issues, corrupted files, and face detection
confidence. Writes quality_audit.csv to data/phase0_outputs/.
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
from image_quality import run_quality_audit

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 image quality audit."
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
        help="Overwrite existing quality_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    logger.info(f"Starting quality audit  config={args.config}")

    cfg = load_config(args.config)
    project_root = get_project_root()
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    audit_path = outputs_root / "quality_audit.csv"

    if audit_path.exists() and not args.force:
        logger.info(
            f"quality_audit.csv already exists at {audit_path}. "
            "Use --force to overwrite. Skipping."
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Done in {elapsed:.1f}s (skipped)")
        return

    run_quality_audit(args.config, project_root)

    # Print pass rate summary
    if audit_path.exists():
        import pandas as pd
        df = pd.read_csv(audit_path)
        if "quality_pass" in df.columns:
            total = len(df)
            passed = df["quality_pass"].sum()
            rate = 100.0 * passed / total if total > 0 else 0.0
            print(f"\nQuality audit results:")
            print(f"  Total images : {total}")
            print(f"  Passed       : {passed}")
            print(f"  Pass rate    : {rate:.1f}%")

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

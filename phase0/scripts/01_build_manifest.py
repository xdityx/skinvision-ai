#!/usr/bin/env python3
"""Build the dataset manifest CSV from raw ACNE04 data.

Reads the project config, auto-detects (or reads) the dataset format,
and writes data/phase0_outputs/manifest.csv with one row per image.
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
from ingestion.factory import AnnotationAdapterFactory
from ingestion.manifest_builder import ManifestBuilder, load_config, get_project_root

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 0 dataset manifest CSV."
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
        help="Overwrite existing manifest.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    logger.info(f"Starting manifest build  config={args.config}")

    cfg = load_config(args.config)
    project_root = get_project_root()
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    manifest_path = outputs_root / "manifest.csv"

    if manifest_path.exists() and not args.force:
        logger.info(
            f"manifest.csv already exists at {manifest_path}. "
            "Use --force to overwrite. Skipping."
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Done in {elapsed:.1f}s (skipped)")
        return

    outputs_root.mkdir(parents=True, exist_ok=True)

    factory = AnnotationAdapterFactory(cfg, project_root)
    adapter = factory.create()
    builder = ManifestBuilder(cfg, project_root)
    df = builder.build(adapter)

    df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved to {manifest_path}  rows={len(df)}")

    # Per-class breakdown
    if "severity" in df.columns:
        breakdown = df.groupby("severity").size()
        logger.info("Per-class image counts:")
        for sev, count in breakdown.items():
            logger.info(f"  severity={sev}  count={count}")
    elif "label" in df.columns:
        breakdown = df.groupby("label").size()
        logger.info("Per-class image counts:")
        for label, count in breakdown.items():
            logger.info(f"  label={label}  count={count}")

    print(f"\nManifest rows : {len(df)}")
    if "severity" in df.columns:
        print("Per-class breakdown:")
        print(df.groupby("severity").size().to_string())

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

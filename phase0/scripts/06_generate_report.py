#!/usr/bin/env python3
"""Generate the final Phase 0 HTML/Markdown dataset readiness report.

Always regenerates — the report is fast and reads from existing output files.
Writes the report to reports/phase0/.
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
from report_generator import run_report_generation

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Phase 0 dataset readiness report."
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
        help="(No effect — report is always regenerated.)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    logger.info(f"Starting report generation  config={args.config}")

    cfg = load_config(args.config)
    project_root = get_project_root()
    reports_root = project_root / cfg["paths"]["reports_root"]

    result = run_report_generation(args.config, project_root)

    # Determine report path from result or scan reports directory
    report_path = None
    if isinstance(result, dict):
        report_path = result.get("report_path")
        verdict = result.get("verdict", "N/A")
    else:
        verdict = "N/A"
        # Try to find any report file written
        for pattern in ["*.html", "*.md", "report_*"]:
            matches = list(reports_root.glob(pattern))
            if matches:
                report_path = matches[-1]
                break

    print(f"\nReport generation results:")
    print(f"  Verdict     : {verdict}")
    if report_path:
        print(f"  Report path : {report_path}")
    else:
        print(f"  Reports dir : {reports_root}")

    elapsed = time.perf_counter() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

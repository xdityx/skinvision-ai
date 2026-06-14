#!/usr/bin/env python3
"""Run all Phase 0 pipeline steps in sequence.

Executes steps 1-6 in order, checking prerequisites before each step.
Prints a summary table at the end with status and elapsed time per step.

Usage:
    python run_all.py
    python run_all.py --force
    python run_all.py --steps 1,2,3
    python run_all.py --steps 4,5,6 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

# Ensure standard output and error streams handle UTF-8 and Unicode characters correctly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


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

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STATUS_DONE = "DONE"
STATUS_SKIPPED = "SKIPPED"
STATUS_FAILED = "FAILED"
STATUS_NOT_RUN = "NOT_RUN"


@dataclass
class StepResult:
    number: int
    name: str
    status: str = STATUS_NOT_RUN
    elapsed: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Phase 0 pipeline steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Pass --force to all steps (overwrite existing outputs).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated list of step numbers to run (default: 1,2,3,4,5,6).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prerequisite checkers
# ---------------------------------------------------------------------------

def _check_manifest(outputs_root: Path) -> Optional[str]:
    p = outputs_root / "manifest.csv"
    if not p.exists():
        return f"manifest.csv not found at {p}. Run Step 1 first."
    return None


def _check_eda(outputs_root: Path) -> Optional[str]:
    p = outputs_root / "eda_stats.json"
    if not p.exists():
        return f"eda_stats.json not found at {p}. Run Step 2 first."
    return None


def _check_quality(outputs_root: Path) -> Optional[str]:
    p = outputs_root / "quality_audit.csv"
    if not p.exists():
        return f"quality_audit.csv not found at {p}. Run Step 3 first."
    return None


def _check_clusters(outputs_root: Path) -> Optional[str]:
    p = outputs_root / "cluster_assignments.csv"
    if not p.exists():
        return f"cluster_assignments.csv not found at {p}. Run Step 4 first."
    return None


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def _run_step1(config: Path, project_root: Path, force: bool) -> None:
    """Build manifest."""
    from ingestion.factory import AnnotationAdapterFactory
    from ingestion.manifest_builder import ManifestBuilder, load_config, get_project_root

    cfg = load_config(config)
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    manifest_path = outputs_root / "manifest.csv"

    if manifest_path.exists() and not force:
        logger.info("manifest.csv exists — skipping (use --force to overwrite).")
        return

    outputs_root.mkdir(parents=True, exist_ok=True)
    factory = AnnotationAdapterFactory(cfg, project_root)
    adapter = factory.create()
    builder = ManifestBuilder(cfg, project_root)
    df = builder.build(adapter)
    df.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}  rows={len(df)}")

    if "severity" in df.columns:
        for sev, cnt in df.groupby("severity").size().items():
            logger.info(f"  severity={sev}: {cnt}")


def _run_step2(config: Path, project_root: Path, force: bool) -> None:
    """Run EDA."""
    from ingestion.manifest_builder import load_config
    from eda import run_eda

    cfg = load_config(config)
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    eda_path = outputs_root / "eda_stats.json"

    if eda_path.exists() and not force:
        logger.info("eda_stats.json exists — skipping.")
        return

    run_eda(config, project_root)
    logger.info(f"EDA stats saved: {eda_path}")


def _run_step3(config: Path, project_root: Path, force: bool) -> None:
    """Audit quality."""
    from ingestion.manifest_builder import load_config
    from image_quality import run_quality_audit

    cfg = load_config(config)
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    audit_path = outputs_root / "quality_audit.csv"

    if audit_path.exists() and not force:
        logger.info("quality_audit.csv exists — skipping.")
        return

    run_quality_audit(config, project_root)
    logger.info(f"Quality audit saved: {audit_path}")


def _run_step4(config: Path, project_root: Path, force: bool) -> None:
    """Cluster faces."""
    from ingestion.manifest_builder import load_config
    from face_clustering import run_face_clustering

    cfg = load_config(config)
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    assignments_path = outputs_root / "cluster_assignments.csv"

    if assignments_path.exists() and not force:
        logger.info("cluster_assignments.csv exists — skipping.")
        return

    if force:
        for npy_file in outputs_root.glob("*.npy"):
            npy_file.unlink()
            logger.info(f"Deleted embedding cache: {npy_file}")

    run_face_clustering(config, project_root)
    logger.info(f"Cluster assignments saved: {assignments_path}")


def _run_step5(config: Path, project_root: Path, force: bool) -> None:
    """Generate splits."""
    from ingestion.manifest_builder import load_config
    from split_generator import run_split_generation

    cfg = load_config(config)
    outputs_root = project_root / cfg["paths"]["outputs_root"]
    train_path = outputs_root / "splits" / "train.csv"

    if train_path.exists() and not force:
        logger.info("splits/train.csv exists — skipping.")
        return

    run_split_generation(config, project_root)
    logger.info(f"Splits saved under: {outputs_root / 'splits'}")


def _run_step6(config: Path, project_root: Path, force: bool) -> None:  # noqa: ARG001
    """Generate report (always runs)."""
    from report_generator import run_report_generation

    result = run_report_generation(config, project_root)
    if isinstance(result, dict):
        verdict = result.get("verdict", "N/A")
        report_path = result.get("report_path", "N/A")
        logger.info(f"Report generated  verdict={verdict}  path={report_path}")
    else:
        logger.info("Report generation complete.")


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

@dataclass
class StepDef:
    number: int
    description: str
    runner: Callable
    prerequisites: List[Callable] = field(default_factory=list)


def _build_step_registry(outputs_root: Path) -> List[StepDef]:
    return [
        StepDef(
            number=1,
            description="Build Manifest",
            runner=_run_step1,
            prerequisites=[],
        ),
        StepDef(
            number=2,
            description="Run EDA",
            runner=_run_step2,
            prerequisites=[lambda: _check_manifest(outputs_root)],
        ),
        StepDef(
            number=3,
            description="Audit Image Quality",
            runner=_run_step3,
            prerequisites=[lambda: _check_manifest(outputs_root)],
        ),
        StepDef(
            number=4,
            description="Cluster Faces",
            runner=_run_step4,
            prerequisites=[lambda: _check_manifest(outputs_root)],
        ),
        StepDef(
            number=5,
            description="Generate Train/Val/Test Splits",
            runner=_run_step5,
            prerequisites=[
                lambda: _check_manifest(outputs_root),
                lambda: _check_clusters(outputs_root),
            ],
        ),
        StepDef(
            number=6,
            description="Generate Readiness Report",
            runner=_run_step6,
            prerequisites=[
                lambda: _check_manifest(outputs_root),
                lambda: _check_eda(outputs_root),
                lambda: _check_quality(outputs_root),
                lambda: _check_clusters(outputs_root),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: List[StepResult]) -> None:
    sep = "-" * 62
    print(f"\n{'='*62}")
    print("PHASE 0 PIPELINE SUMMARY")
    print(f"{'='*62}")
    print(f"{'Step':<6} {'Name':<32} {'Status':<10} {'Elapsed':>8}")
    print(sep)
    for r in results:
        elapsed_str = f"{r.elapsed:.1f}s" if r.elapsed > 0 else "—"
        print(f"{r.number:<6} {r.name:<32} {r.status:<10} {elapsed_str:>8}")
        if r.error:
            print(f"       ERROR: {r.error}")
    print(sep)
    any_failed = any(r.status == STATUS_FAILED for r in results)
    overall = "FAILED" if any_failed else "SUCCESS"
    total_elapsed = sum(r.elapsed for r in results)
    print(f"Overall: {overall}  |  Total elapsed: {total_elapsed:.1f}s")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse requested step numbers
    try:
        requested_steps = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
    except ValueError:
        logger.error(f"Invalid --steps value: {args.steps!r}. Must be comma-separated ints.")
        sys.exit(1)

    cfg = load_config(args.config)
    project_root = get_project_root()
    outputs_root = project_root / cfg["paths"]["outputs_root"]

    step_defs = _build_step_registry(outputs_root)

    # Filter to requested steps
    steps_to_run = [s for s in step_defs if s.number in requested_steps]
    if not steps_to_run:
        logger.error(f"No valid steps found for --steps={args.steps!r}.")
        sys.exit(1)

    results: List[StepResult] = []
    any_failed = False

    for step in steps_to_run:
        banner = f"\n{'='*60}\nStep {step.number}: {step.description}\n{'='*60}"
        print(banner)
        logger.info(f"Starting Step {step.number}: {step.description}")

        result = StepResult(number=step.number, name=step.description)
        step_start = time.perf_counter()

        # Check prerequisites
        prereq_error: Optional[str] = None
        for check_fn in step.prerequisites:
            prereq_error = check_fn()
            if prereq_error:
                break

        if prereq_error:
            logger.error(f"Prerequisite check failed: {prereq_error}")
            result.status = STATUS_FAILED
            result.error = prereq_error
            result.elapsed = time.perf_counter() - step_start
            results.append(result)
            any_failed = True
            continue

        # Run the step
        try:
            step.runner(args.config, project_root, args.force)
            result.status = STATUS_DONE
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Step {step.number} failed with exception: {exc}")
            result.status = STATUS_FAILED
            result.error = str(exc)
            any_failed = True

        result.elapsed = time.perf_counter() - step_start
        results.append(result)
        logger.info(
            f"Step {step.number} {result.status} in {result.elapsed:.1f}s"
        )

    _print_summary(results)

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

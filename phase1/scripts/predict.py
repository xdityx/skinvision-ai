"""
phase1/scripts/predict.py
CLI entry point for single-image acne severity prediction.

Usage
-----
# Basic prediction:
    python -m phase1.scripts.predict --image path/to/face.jpg

# With TTA (5-view averaging):
    python -m phase1.scripts.predict --image path/to/face.jpg --tta

# Save output JSON:
    python -m phase1.scripts.predict --image path/to/face.jpg --output result.json

# Specify checkpoint and config explicitly:
    python -m phase1.scripts.predict \\
        --image path/to/face.jpg \\
        --checkpoint phase1/checkpoints/best_model.pt \\
        --config phase1/config/phase1.yaml

# Force CPU (useful for testing without GPU):
    python -m phase1.scripts.predict --image path/to/face.jpg --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase1.src.inference import (
    AcnePredictor,
    CheckpointNotFoundError,
    ImageLoadError,
    ImageNotFoundError,
    ImageTooSmallError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase1.predict")

# ── Default paths ─────────────────────────────────────────────────────────────
_DEFAULT_CHECKPOINT = PROJECT_ROOT / "phase1" / "checkpoints" / "best_model.pt"
_DEFAULT_CONFIG     = PROJECT_ROOT / "phase1" / "config"      / "phase1.yaml"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Acne Severity Predictor — EfficientNet-B2 + CORN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--image", "-i",
        type=Path,
        required=True,
        metavar="IMAGE",
        help="Path to the input face image (.jpg / .png / .webp).",
    )
    p.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        metavar="PT",
        help=f"Path to model checkpoint (default: {_DEFAULT_CHECKPOINT}).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="YAML",
        help=(
            "Path to phase1.yaml.  If omitted, config embedded in the "
            "checkpoint is used."
        ),
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="Enable 5-view Test-Time Augmentation (slightly slower, more robust).",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        metavar="JSON",
        help="If provided, save the prediction result to this JSON file.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device (default: auto — uses CUDA if available).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress INFO logs; only print the JSON result.",
    )
    return p.parse_args()


# ── Pretty printer ────────────────────────────────────────────────────────────

_SEVERITY_TAG = {
    "mild":     "[MILD]    ",
    "moderate": "[MODERATE]",
    "severe":   "[SEVERE]  ",
}

def _print_result(result: dict) -> None:
    """Print a human-readable summary to stdout (ASCII-safe for Windows)."""
    sev   = result["predicted_severity"]
    tag   = _SEVERITY_TAG.get(sev, f"[{sev.upper()}]")
    conf  = result["confidence"] * 100
    probs = result["class_probabilities"]
    tta   = "enabled (5 views)" if result["tta_enabled"] else "disabled"
    ms    = result["inference_time_ms"]

    sep = "-" * 52
    print()
    print(sep)
    print("  Acne Severity Prediction")
    print(sep)
    print(f"  Image       : {Path(result['image_path']).name}")
    print(f"  Prediction  : {tag}  ({conf:.1f}% confidence)")
    print()
    print("  Class probabilities:")
    for name, prob in probs.items():
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        marker = " <--" if name == sev else ""
        print(f"    {name:>10}  {bar}  {prob * 100:5.1f}%{marker}")
    print()
    print(f"  TTA         : {tta}")
    print(f"  Inference   : {ms:.1f} ms")
    print(f"  Checkpoint  : epoch {result['checkpoint_epoch']} "
          f"(val F1 = {result['checkpoint_val_f1']:.4f})")
    if result.get("output_path"):
        print(f"  Output JSON : {result['output_path']}")
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    """Entry point. Returns exit code (0=success, 1=error)."""
    # Reconfigure stdout to UTF-8 so non-ASCII paths (e.g. Chinese folder names)
    # can be printed on Windows without cp1252 UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # ── Load predictor ────────────────────────────────────────────────────────
    try:
        predictor = AcnePredictor(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
        )
    except CheckpointNotFoundError as exc:
        logger.error("❌  Checkpoint error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("❌  Failed to initialise predictor: %s", exc)
        return 1

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = predictor.predict(
            image_path=args.image,
            tta=args.tta,
            save_to=args.output,
        )
    except ImageNotFoundError as exc:
        logger.error("❌  Image not found: %s", exc)
        return 1
    except ImageLoadError as exc:
        logger.error("❌  Cannot load image: %s", exc)
        return 1
    except ImageTooSmallError as exc:
        logger.error("❌  Image too small: %s", exc)
        return 1
    except Exception as exc:
        logger.error("❌  Inference failed: %s", exc, exc_info=True)
        return 1

    # ── Output ────────────────────────────────────────────────────────────────
    if args.output:
        result["output_path"] = str(args.output.resolve())

    if not args.quiet:
        _print_result(result)

    # Always print the raw JSON to stdout (for piping / programmatic use)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

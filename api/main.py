"""
api/main.py
FastAPI application factory with lifespan-based startup.

Startup sequence
----------------
1. Load AcnePredictor from checkpoint (GPU if available)
2. Warm up with a dummy 260×260 tensor (eliminates first-request CUDA spike)
3. Initialise FaceValidator (MediaPipe)
4. Store both in app.state for dependency injection

Run locally
-----------
    uvicorn api.main:app --reload --port 8000

API docs
--------
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import health, predict

# Ensure project root is importable when run from any working directory
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api.main")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and validators at startup; release at shutdown."""

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Loading AcnePredictor from %s ...", settings.checkpoint_path)
    from phase1.src.inference import AcnePredictor, CheckpointNotFoundError

    try:
        predictor = AcnePredictor(
            checkpoint_path=_PROJECT_ROOT / settings.checkpoint_path,
            config_path=_PROJECT_ROOT / settings.config_path,
            device=settings.device,
        )
    except CheckpointNotFoundError as exc:
        logger.critical("Checkpoint not found: %s", exc)
        raise RuntimeError(
            f"Cannot start API — checkpoint not found: {settings.checkpoint_path}"
        ) from exc

    # ── GPU warm-up ──────────────────────────────────────────────────────────
    logger.info("Warming up model with dummy input ...")
    try:
        dummy = torch.zeros(
            1, 3, predictor.image_size, predictor.image_size,
            device=predictor.device,
        )
        with torch.no_grad():
            _ = predictor.model(dummy)
        logger.info("GPU warm-up complete.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up failed (non-fatal): %s", exc)

    app.state.predictor = predictor

    # ── Face validator ────────────────────────────────────────────────────────
    logger.info("Initialising FaceValidator ...")
    from api.face_validator import FaceValidator

    face_validator = FaceValidator(
        min_detection_confidence=settings.face_detection_confidence,
        model_selection=settings.face_model_selection,
    )
    app.state.face_validator = face_validator
    logger.info("API ready.")

    yield  # ── server is running ──────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing resources ...")
    face_validator.close()
    logger.info("Shutdown complete.")


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — permissive for local development; tighten for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)

    return app


app = create_app()

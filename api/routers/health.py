"""
api/routers/health.py
GET /health  — liveness probe (always 200 if process is running)
GET /ready   — readiness probe (200 only when model is loaded)
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.config import settings
from api.models import HealthResponse, ReadyResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 OK as long as the server process is running.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Returns ready=true when the model is loaded and the server can "
        "serve predictions.  Use this as a Kubernetes readiness probe."
    ),
)
async def ready(request: Request) -> ReadyResponse:
    predictor = getattr(request.app.state, "predictor", None)
    loaded = predictor is not None

    return ReadyResponse(
        ready=loaded,
        model_loaded=loaded,
        checkpoint_epoch=predictor._checkpoint_epoch if loaded else None,
        checkpoint_val_f1=predictor._checkpoint_val_f1 if loaded else None,
        model_version=settings.model_version,
    )

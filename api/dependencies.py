"""
api/dependencies.py
FastAPI dependency providers.

Both AcnePredictor and FaceValidator are singletons stored in app.state
and accessed via these dependency functions.
"""
from __future__ import annotations

from fastapi import Request

from phase1.src.inference import AcnePredictor
from api.face_validator import FaceValidator


def get_predictor(request: Request) -> AcnePredictor:
    """Return the application-level AcnePredictor instance."""
    return request.app.state.predictor


def get_face_validator(request: Request) -> FaceValidator:
    """Return the application-level FaceValidator instance."""
    return request.app.state.face_validator

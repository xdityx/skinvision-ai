"""Tests for Phase 3 ingredient guidance."""
from __future__ import annotations

import io

import pytest

from api.recommendations.engine import build_ingredient_guidance
from api.recommendations.loader import REQUIRED_SEVERITIES, load_knowledge_base
from api.tests.conftest import _need_checkpoint

_PREDICT_URL = "/api/v1/predict"


class FakePredictor:
    def __init__(self, severity: str = "mild", confidence: float = 0.82) -> None:
        self.severity = severity
        self.confidence = confidence

    def predict(self, image_path, tta: bool = False):
        probabilities = {
            "mild": 0.82,
            "moderate": 0.12,
            "severe": 0.06,
        }
        if self.severity == "moderate":
            probabilities = {
                "mild": 0.21,
                "moderate": self.confidence,
                "severe": 1.0 - self.confidence - 0.21,
            }
        elif self.severity == "severe":
            probabilities = {
                "mild": 0.08,
                "moderate": 1.0 - self.confidence - 0.08,
                "severe": self.confidence,
            }
        elif self.confidence != 0.82:
            probabilities = {
                "mild": self.confidence,
                "moderate": 0.30,
                "severe": 1.0 - self.confidence - 0.30,
            }

        class_index = {"mild": 0, "moderate": 1, "severe": 2}[self.severity]
        return {
            "predicted_class": class_index,
            "predicted_severity": self.severity,
            "confidence": self.confidence,
            "class_probabilities": probabilities,
            "tta_enabled": tta,
            "tta_views": 5 if tta else None,
            "inference_time_ms": 12.5,
        }


def _upload(client, data: bytes):
    return client.post(
        _PREDICT_URL,
        files={"file": ("face.jpg", io.BytesIO(data), "image/jpeg")},
        params={"strict_face": "false"},
    )


def test_kb_loads() -> None:
    kb = load_knowledge_base()
    assert kb["schema_version"] == "acne_ingredients.v1"
    assert kb["ingredients"]


def test_all_severity_profiles_exist() -> None:
    kb = load_knowledge_base()
    assert REQUIRED_SEVERITIES.issubset(kb["severity_profiles"])


def test_every_ingredient_reference_exists() -> None:
    kb = load_knowledge_base()
    ingredient_ids = set(kb["ingredients"])
    for severity, profile in kb["severity_profiles"].items():
        missing = set(profile["ingredient_ids"]) - ingredient_ids
        assert not missing, f"{severity} references missing ingredients: {missing}"


def test_low_confidence_creates_confidence_warning() -> None:
    guidance = build_ingredient_guidance("mild", confidence=0.59)
    assert guidance.confidence_warning is not None
    assert "below 60%" in guidance.confidence_warning


def test_retinoid_guidance_has_required_cautions() -> None:
    guidance = build_ingredient_guidance("moderate", confidence=0.90)
    retinoid = next(
        item for item in guidance.recommended_ingredients
        if item.id == "adapalene_or_retinoid"
    )
    cautions = " ".join(retinoid.cautions).lower()
    assert "irritation" in cautions
    assert "sun sensitivity" in cautions
    assert "pregnancy" in cautions


def test_severe_guidance_emphasizes_dermatologist_evaluation() -> None:
    guidance = build_ingredient_guidance("severe", confidence=0.90)
    message = guidance.dermatologist_escalation.message.lower()
    assert guidance.dermatologist_escalation.level == "strongly_recommended"
    assert "dermatologist" in message
    assert "prescription" in message


def test_no_product_names_or_urls_in_ingredient_recommendations() -> None:
    banned_terms = [
        "http://",
        "https://",
        "www.",
        "differin",
        "retin-a",
        "accutane",
        "proactiv",
        "benzaclin",
        "duac",
        "epiduo",
    ]

    for severity in REQUIRED_SEVERITIES:
        guidance = build_ingredient_guidance(severity, confidence=0.90)
        parts = []
        for ingredient in guidance.recommended_ingredients:
            parts.extend([ingredient.name, ingredient.why, " ".join(ingredient.cautions)])
        text = " ".join(parts).lower()
        assert not any(term in text for term in banned_terms)


@_need_checkpoint
def test_predict_api_includes_ingredient_guidance(app, client, solid_color_jpg) -> None:
    original = app.state.predictor
    app.state.predictor = FakePredictor(severity="mild", confidence=0.82)
    try:
        response = _upload(client, solid_color_jpg)
    finally:
        app.state.predictor = original

    assert response.status_code == 200, response.text
    body = response.json()
    assert "ingredient_guidance" in body
    guidance = body["ingredient_guidance"]
    assert guidance["severity"] == "mild"
    assert guidance["schema_version"] == "acne_ingredients.v1"
    assert guidance["recommended_ingredients"]
    assert guidance["routine_guidance"]
    assert guidance["cautions"]
    assert guidance["dermatologist_escalation"]


@_need_checkpoint
def test_predict_api_low_confidence_includes_warning(app, client, solid_color_jpg) -> None:
    original = app.state.predictor
    app.state.predictor = FakePredictor(severity="mild", confidence=0.58)
    try:
        response = _upload(client, solid_color_jpg)
    finally:
        app.state.predictor = original

    assert response.status_code == 200, response.text
    warning = response.json()["ingredient_guidance"]["confidence_warning"]
    assert warning is not None
    assert "below 60%" in warning


@_need_checkpoint
def test_old_prediction_fields_still_exist(app, client, solid_color_jpg) -> None:
    original = app.state.predictor
    app.state.predictor = FakePredictor(severity="mild", confidence=0.82)
    try:
        response = _upload(client, solid_color_jpg)
    finally:
        app.state.predictor = original

    assert response.status_code == 200, response.text
    body = response.json()
    old_fields = {
        "predicted_class",
        "predicted_severity",
        "confidence",
        "class_probabilities",
        "tta_enabled",
        "tta_views",
        "face_detected",
        "face_warning",
        "inference_time_ms",
        "model_version",
    }
    assert old_fields.issubset(body)

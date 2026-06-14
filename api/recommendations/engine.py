"""Rule-based ingredient guidance builder."""
from __future__ import annotations

from typing import Any

from api.recommendations.loader import REQUIRED_SEVERITIES, load_knowledge_base
from api.recommendations.schemas import (
    DermatologistEscalation,
    IngredientGuidance,
    IngredientRecommendation,
    RoutineGuidance,
)

LOW_CONFIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_WARNING = (
    "Prediction confidence is below 60%, so use this ingredient guidance cautiously "
    "and consider retaking the photo in clear, even lighting."
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def build_ingredient_guidance(
    severity: str,
    confidence: float,
    kb: dict[str, Any] | None = None,
) -> IngredientGuidance:
    """Return ingredient guidance for the predicted acne severity."""
    normalized = severity.strip().lower()
    if normalized not in REQUIRED_SEVERITIES:
        raise ValueError(f"Unsupported acne severity: {severity}")

    knowledge_base = kb or load_knowledge_base()
    profile = knowledge_base["severity_profiles"][normalized]
    ingredients = knowledge_base["ingredients"]

    recommended = []
    for ingredient_id in profile["ingredient_ids"]:
        item = ingredients[ingredient_id]
        recommended.append(
            IngredientRecommendation(
                id=ingredient_id,
                name=item["name"],
                category=item["category"],
                priority=item.get("priority", "supportive"),
                why=item["why"],
                cautions=_as_list(item.get("cautions")),
            )
        )

    routine = knowledge_base["routine_guidance"][profile["routine_guidance_id"]]
    escalation = knowledge_base["dermatologist_escalation"][
        profile["dermatologist_escalation_id"]
    ]

    confidence_warning = (
        LOW_CONFIDENCE_WARNING if confidence < LOW_CONFIDENCE_THRESHOLD else None
    )

    return IngredientGuidance(
        schema_version=knowledge_base["schema_version"],
        severity=normalized,
        disclaimer=knowledge_base["disclaimer"],
        recommended_ingredients=recommended,
        routine_guidance=RoutineGuidance(
            morning=_as_list(routine.get("morning")),
            evening=_as_list(routine.get("evening")),
            general=_as_list(routine.get("general")),
        ),
        cautions=_as_list(profile.get("cautions")),
        dermatologist_escalation=DermatologistEscalation(
            level=escalation["level"],
            message=escalation["message"],
        ),
        confidence_warning=confidence_warning,
    )

"""Load and validate the versioned ingredient knowledge base."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

KB_PATH = Path(__file__).resolve().parent / "kb" / "acne_ingredients.v1.yaml"
REQUIRED_SEVERITIES = {"mild", "moderate", "severe"}


class KnowledgeBaseError(ValueError):
    """Raised when the ingredient knowledge base is missing or invalid."""


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise KnowledgeBaseError(f"{name} must be a mapping")
    return data


def validate_knowledge_base(kb: dict[str, Any]) -> None:
    """Validate the fields needed by the recommendation engine."""
    for key in [
        "schema_version",
        "disclaimer",
        "ingredients",
        "severity_profiles",
        "routine_guidance",
        "dermatologist_escalation",
    ]:
        if key not in kb:
            raise KnowledgeBaseError(f"Missing required KB key: {key}")

    ingredients = _require_mapping(kb["ingredients"], "ingredients")
    profiles = _require_mapping(kb["severity_profiles"], "severity_profiles")
    routines = _require_mapping(kb["routine_guidance"], "routine_guidance")
    escalation = _require_mapping(
        kb["dermatologist_escalation"], "dermatologist_escalation"
    )

    missing_profiles = REQUIRED_SEVERITIES - set(profiles)
    if missing_profiles:
        raise KnowledgeBaseError(
            f"Missing severity profiles: {sorted(missing_profiles)}"
        )

    for severity, profile in profiles.items():
        profile = _require_mapping(profile, f"severity_profiles.{severity}")
        ingredient_ids = profile.get("ingredient_ids")
        if not isinstance(ingredient_ids, list) or not ingredient_ids:
            raise KnowledgeBaseError(
                f"severity_profiles.{severity}.ingredient_ids must be a non-empty list"
            )

        unknown_ids = [item for item in ingredient_ids if item not in ingredients]
        if unknown_ids:
            raise KnowledgeBaseError(
                f"severity_profiles.{severity} references unknown ingredients: {unknown_ids}"
            )

        routine_id = profile.get("routine_guidance_id")
        escalation_id = profile.get("dermatologist_escalation_id")
        if routine_id not in routines:
            raise KnowledgeBaseError(
                f"severity_profiles.{severity} references unknown routine: {routine_id}"
            )
        if escalation_id not in escalation:
            raise KnowledgeBaseError(
                f"severity_profiles.{severity} references unknown escalation: {escalation_id}"
            )


@lru_cache(maxsize=1)
def load_knowledge_base(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML ingredient knowledge base and validate references."""
    kb_path = path or KB_PATH
    if not kb_path.exists():
        raise KnowledgeBaseError(f"Ingredient knowledge base not found: {kb_path}")

    with kb_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    kb = _require_mapping(data, "knowledge base")
    validate_knowledge_base(kb)
    return kb

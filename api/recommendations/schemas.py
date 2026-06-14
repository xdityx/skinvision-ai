"""Pydantic schemas for ingredient guidance responses."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngredientRecommendation(BaseModel):
    id: str
    name: str
    category: str
    priority: str
    why: str
    cautions: list[str] = Field(default_factory=list)


class RoutineGuidance(BaseModel):
    morning: list[str] = Field(default_factory=list)
    evening: list[str] = Field(default_factory=list)
    general: list[str] = Field(default_factory=list)


class DermatologistEscalation(BaseModel):
    level: str
    message: str


class IngredientGuidance(BaseModel):
    schema_version: str
    severity: str
    disclaimer: str
    recommended_ingredients: list[IngredientRecommendation]
    routine_guidance: RoutineGuidance
    cautions: list[str] = Field(default_factory=list)
    dermatologist_escalation: DermatologistEscalation
    confidence_warning: str | None = None

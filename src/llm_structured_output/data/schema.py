"""Pydantic schemas for model predictions and evaluation records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class Prediction(BaseModel):
    """The strict structured-output schema every model response must satisfy."""

    model_config = ConfigDict(extra="forbid")
    intent: str


def validate_prediction(payload: Any) -> Prediction:
    """Validate a parsed payload against the prediction schema.

    Raises :class:`pydantic.ValidationError` when the payload is valid JSON but
    does not match the schema (missing/wrong-typed ``intent`` or extra keys).
    """
    return Prediction.model_validate(payload)


def is_valid_prediction(payload: Any) -> bool:
    """Return True when ``payload`` conforms to :class:`Prediction`."""
    try:
        validate_prediction(payload)
        return True
    except ValidationError:
        return False

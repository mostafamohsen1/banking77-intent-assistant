"""Strict JSON output validation: extract, parse and schema-check model output."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_structured_output.data.schema import Prediction, validate_prediction

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_candidate(raw: str) -> str | None:
    """Extract the first JSON object from a raw model response.

    Handles fenced code blocks and leading/trailing prose. Returns ``None`` when
    no JSON object can be located.
    """
    if not raw:
        return None
    fence = _FENCE_RE.search(raw)
    if fence:
        candidate = fence.group(1).strip()
        if candidate.startswith("{"):
            return candidate
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        return raw[start : end + 1]
    return None


def parse_model_output(raw: str) -> tuple[Prediction | None, str | None]:
    """Parse a raw response into a validated :class:`Prediction`.

    Returns ``(prediction, error)`` where exactly one of the two is not None.
    Errors are stable machine-readable codes used by the error analysis:
    ``invalid_json``, ``schema_violation``, ``empty_output``.
    """
    if raw is None or not raw.strip():
        return None, "empty_output"
    candidate = extract_json_candidate(raw)
    if candidate is None:
        return None, "invalid_json"
    try:
        payload: Any = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "invalid_json"
    try:
        return validate_prediction(payload), None
    except Exception:
        return None, "schema_violation"

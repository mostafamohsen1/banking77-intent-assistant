"""Tests for the strict JSON output validator and the prediction schema."""

from __future__ import annotations

import json

import pytest

from llm_structured_output.data.schema import is_valid_prediction, validate_prediction
from llm_structured_output.evaluation.json_validator import (
    extract_json_candidate,
    parse_model_output,
)

VALID = json.dumps({"intent": "card_arrival"})


def test_valid_prediction_passes():
    p = validate_prediction(json.loads(VALID))
    assert p.intent == "card_arrival"
    assert is_valid_prediction({"intent": "x"})


def test_schema_rejects_missing_wrong_type_and_extra():
    assert not is_valid_prediction({})
    assert not is_valid_prediction({"intent": 42})
    assert not is_valid_prediction({"intent": "x", "confidence": 0.9})
    with pytest.raises(Exception):
        validate_prediction({"nope": True})


def test_parse_plain_json():
    pred, err = parse_model_output(VALID)
    assert err is None
    assert pred.intent == "card_arrival"


def test_parse_json_in_markdown_fence():
    raw = f"Here you go:\n```json\n{VALID}\n```\nHope that helps!"
    pred, err = parse_model_output(raw)
    assert err is None and pred.intent == "card_arrival"


def test_parse_json_with_leading_prose():
    pred, err = parse_model_output(f"The intent is {VALID} — done.")
    assert err is None and pred.intent == "card_arrival"


@pytest.mark.parametrize(
    "raw,expected_error",
    [
        ("", "empty_output"),
        (None, "empty_output"),
        ("I am sorry, I cannot classify that.", "invalid_json"),
        ("{intent: card_arrival}", "invalid_json"),  # not valid JSON (unquoted key)
        ('{"label": "card_arrival"}', "schema_violation"),
        ('{"intent": "card_arrival", "extra": 1}', "schema_violation"),
        ('{"intent": 123}', "schema_violation"),
    ],
)
def test_parse_failures_return_error_codes(raw, expected_error):
    pred, err = parse_model_output(raw)
    assert pred is None
    assert err == expected_error


def test_extract_json_candidate_prefers_fence():
    assert extract_json_candidate("```json\n{}\n```") == "{}"
    assert extract_json_candidate("no braces here") is None

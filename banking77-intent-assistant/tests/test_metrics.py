"""Tests for metrics computation and error analysis artifacts."""

from __future__ import annotations

import json

import pytest

from llm_structured_output.evaluation.error_analysis import (
    build_confusion_matrix,
    confusion_pairs,
    write_error_records,
)
from llm_structured_output.evaluation.metrics import compute_metrics, save_metrics


def test_compute_metrics_counts(sample_records, labels):
    m = compute_metrics(sample_records)
    assert m["num_examples"] == 4
    assert m["valid_json_rate"] == 0.5  # 2 of 4 parse to the schema
    assert m["invalid_outputs"] == 2
    assert m["exact_match"] == 0.25  # only the first record matches
    assert m["error_breakdown"] == {"invalid_json": 1, "unknown_label": 1}
    # Over parsed outputs only: 1 correct of 2.
    assert m["accuracy"] == 0.5


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([])


def test_write_error_records(tmp_path, sample_records):
    path = tmp_path / "errors.jsonl"
    n = write_error_records(sample_records, path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert n == 3  # everything except the exact match
    keys = {"input", "raw_output", "true_label", "pred_label", "error_type"}
    assert all(set(line) == keys for line in lines)
    types = {line["error_type"] for line in lines}
    assert types == {"invalid_json", "unknown_label", "wrong_class"}


def test_confusion_matrix_and_pairs(sample_records, labels):
    matrix = build_confusion_matrix(sample_records, labels)
    assert matrix["labels"] == labels
    assert matrix["off_vocabulary_outputs"] == 2
    total = sum(sum(row) for row in matrix["matrix"])
    assert total == 2  # only predictions with in-vocabulary labels land in the matrix
    pairs = confusion_pairs(sample_records)
    assert pairs[0] == {"true_label": labels[2], "pred_label": labels[1], "count": 1}


def test_save_metrics_roundtrip(tmp_path):
    m = {"accuracy": 0.5}
    path = save_metrics(m, tmp_path / "m.json")
    assert json.loads(path.read_text()) == m

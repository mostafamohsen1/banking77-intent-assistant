"""Tests for the TF-IDF baseline on synthetic data."""

from __future__ import annotations

import base64

from conftest import FakeDataset

from llm_structured_output.baseline.tfidf import evaluate_baseline, train_baseline
from llm_structured_output.baseline.web_export import evaluate_web_model, export_web_model


def rows(n, offset=0):
    phrases = [
        "my card has not arrived yet please check delivery",
        "you charged me the wrong exchange rate on my payment",
        "what is the current exchange rate for dollars",
        "my top up failed and money did not arrive",
        "the recipient has not received my bank transfer",
        "i need to verify my identity with a document",
    ]
    return FakeDataset(
        [
            {
                "text": phrases[(offset + i) % len(phrases)] + f" #{offset + i}",
                "label": (offset + i) % len(phrases),
            }
            for i in range(n)
        ]
    )


def test_web_export_matches_float_model():
    label_names = [f"intent_{i}" for i in range(6)]
    pipeline = train_baseline(rows(120), rows(30, offset=120), seed=42)
    test = rows(30, offset=150)

    model = export_web_model(pipeline, label_names, min_confidence=0.15)
    n_features = len(model["idf"])
    assert len(model["vocabulary"]) == n_features
    assert len(base64.b64decode(model["coef_int8_b64"])) == len(label_names) * n_features

    metrics = evaluate_web_model(model, pipeline, test)
    assert metrics["num_test_examples"] == len(test)
    assert metrics["agreement_with_float_model"] == 1.0
    assert metrics["accuracy"] == metrics["float_model_accuracy"]


def test_baseline_learns_separable_classes(labels):
    train = rows(120)
    validation = rows(30, offset=120)
    test = rows(30, offset=150)
    label_names = [f"intent_{i}" for i in range(6)]

    pipeline = train_baseline(train, validation, seed=42)
    metrics = evaluate_baseline(pipeline, test, label_names)
    assert metrics["accuracy"] > 0.9
    assert metrics["num_labels"] == 6
    assert len(metrics["classification_report"]) > 0

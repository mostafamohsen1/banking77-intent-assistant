"""Export the TF-IDF baseline as a compact JSON model a browser can run."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def export_web_model(
    pipeline: Pipeline, labels: list[str], min_confidence: float
) -> dict[str, Any]:
    """Serialize vocabulary, idf and int8-quantized coefficients (one scale per class)."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if list(clf.classes_) != list(range(len(labels))):
        raise ValueError("Classifier classes do not line up with the label list.")

    scale = np.abs(clf.coef_).max(axis=1) / 127.0
    scale[scale == 0.0] = 1.0
    quantized = np.round(clf.coef_ / scale[:, None]).astype(np.int8)
    return {
        "labels": list(labels),
        "vocabulary": {term: int(idx) for term, idx in tfidf.vocabulary_.items()},
        "idf": [round(float(v), 5) for v in tfidf.idf_],
        "intercept": [round(float(v), 5) for v in clf.intercept_],
        "coef_scale": [float(v) for v in scale],
        "coef_int8_b64": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "min_confidence": float(min_confidence),
    }


def web_model_probabilities(
    model: dict[str, Any], pipeline: Pipeline, texts: list[str]
) -> np.ndarray:
    """Class probabilities computed from the exported (quantized) weights."""
    n_labels = len(model["labels"])
    coef = np.frombuffer(base64.b64decode(model["coef_int8_b64"]), dtype=np.int8)
    coef = coef.reshape(n_labels, -1) * np.asarray(model["coef_scale"])[:, None]
    features = pipeline.named_steps["tfidf"].transform(texts)
    logits = np.asarray(features @ coef.T) + np.asarray(model["intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate_web_model(model: dict[str, Any], pipeline: Pipeline, test: Any) -> dict[str, Any]:
    """Score the exported model on the test split, next to the float model it came from."""
    texts = list(test["text"])
    gold = np.asarray(test["label"])
    probabilities = web_model_probabilities(model, pipeline, texts)
    preds = probabilities.argmax(axis=1)
    float_preds = np.asarray(pipeline.predict(texts))
    return {
        "accuracy": float(accuracy_score(gold, preds)),
        "macro_f1": float(f1_score(gold, preds, average="macro")),
        "float_model_accuracy": float(accuracy_score(gold, float_preds)),
        "float_model_macro_f1": float(f1_score(gold, float_preds, average="macro")),
        "agreement_with_float_model": float((preds == float_preds).mean()),
        "below_min_confidence_rate": float(
            (probabilities.max(axis=1) < model["min_confidence"]).mean()
        ),
        "min_confidence": model["min_confidence"],
        "num_test_examples": int(len(gold)),
        "num_labels": len(model["labels"]),
        "num_features": len(model["idf"]),
    }

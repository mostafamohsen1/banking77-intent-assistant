"""TF-IDF + Logistic Regression baseline over the Banking77 classes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_baseline(train: Any, validation: Any, seed: int) -> Pipeline:
    """Train a TF-IDF (word + char n-grams) logistic-regression pipeline."""
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=50_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=10.0,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train["text"], train["label"])
    val_acc = accuracy_score(validation["label"], pipeline.predict(validation["text"]))
    logger.info("Baseline validation accuracy: %.4f", val_acc)
    return pipeline


def evaluate_baseline(pipeline: Pipeline, test: Any, labels: list[str]) -> dict[str, Any]:
    """Compute the baseline metrics on the official test split."""
    preds = pipeline.predict(test["text"])
    return {
        "accuracy": float(accuracy_score(test["label"], preds)),
        "macro_f1": float(f1_score(test["label"], preds, average="macro")),
        "weighted_f1": float(f1_score(test["label"], preds, average="weighted")),
        "num_test_examples": len(test),
        "num_labels": len(labels),
        "classification_report": classification_report(
            test["label"], preds, target_names=labels, output_dict=True, zero_division=0
        ),
    }


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars to built-in types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def save_baseline_outputs(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    test: Any,
    metrics_file: str | Path,
    predictions_file: str | Path,
) -> None:
    """Persist baseline metrics (JSON) and per-example predictions (JSONL)."""
    metrics_file = Path(metrics_file)
    predictions_file = Path(predictions_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_file.write_text(json.dumps(_to_builtin(metrics), indent=2), encoding="utf-8")
    logger.info("Saved baseline metrics to %s", metrics_file)

    preds = pipeline.predict(test["text"])
    with predictions_file.open("w", encoding="utf-8") as fh:
        for text, true_id, pred_id in zip(test["text"], test["label"], preds, strict=True):
            fh.write(
                json.dumps(
                    {
                        "input": text,
                        "true_label": true_id,
                        "pred_label": str(pred_id),
                        "valid_json": True,  # the baseline is deterministic by construction
                        "exact_match": bool(true_id == pred_id),
                    }
                )
                + "\n"
            )
    logger.info("Saved baseline predictions to %s", predictions_file)

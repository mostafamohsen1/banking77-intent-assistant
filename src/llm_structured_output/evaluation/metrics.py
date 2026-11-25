"""Evaluation metrics from structured prediction records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute headline metrics from per-example prediction records.

    Each record must contain ``true_label`` (str) and ``pred_label`` (str | None,
    None when the output was not valid JSON). Additionally reports:

    - ``valid_json_rate``: fraction of responses that parsed to the schema
    - ``invalid_outputs``: count of unparseable / schema-violating responses
    - ``error_breakdown``: counts by machine-readable error type
    - ``exact_match``: accuracy over all examples (invalid output = wrong)
    """
    records = list(records)
    n = len(records)
    if n == 0:
        raise ValueError("No prediction records to evaluate.")

    y_true: list[str] = []
    y_pred: list[str | None] = []
    error_breakdown: dict[str, int] = {}
    valid = 0
    for rec in records:
        y_true.append(rec["true_label"])
        pred = rec.get("pred_label")
        y_pred.append(pred)
        if pred is not None:
            valid += 1
        else:
            err = rec.get("error", "unknown_error")
            error_breakdown[err] = error_breakdown.get(err, 0) + 1

    accuracy_overall = accuracy_score(y_true, ["__INVALID__" if p is None else p for p in y_pred])

    parsed = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if p is not None]
    if parsed:
        pt = [t for t, _ in parsed]
        pp = [p for _, p in parsed]
        metrics = {
            "accuracy": float(accuracy_score(pt, pp)),
            "macro_f1": float(f1_score(pt, pp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(pt, pp, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(pt, pp, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(pt, pp, average="macro", zero_division=0)),
        }
    else:
        metrics = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }

    return {
        **metrics,
        "exact_match": float(accuracy_overall),
        "valid_json_rate": valid / n,
        "invalid_outputs": n - valid,
        "num_examples": n,
        "error_breakdown": error_breakdown,
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Persist metrics as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path

"""Error analysis: error file, confusion matrix and confusion pairs."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_error_records(
    records: list[dict[str, Any]],
    errors_file: str | Path,
) -> int:
    """Write every failed example to JSONL with a machine-readable error type.

    Failure = invalid JSON, schema violation, unknown label or wrong class.
    Each record includes the raw output, true label, predicted label (if any)
    and the error category.
    """
    errors_file = Path(errors_file)
    errors_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with errors_file.open("w", encoding="utf-8") as fh:
        for rec in records:
            if rec.get("exact_match"):
                continue
            error_type = rec.get("error")
            if error_type is None:
                error_type = "wrong_class"
            payload = {
                "input": rec.get("input"),
                "raw_output": rec.get("raw_output"),
                "true_label": rec.get("true_label"),
                "pred_label": rec.get("pred_label"),
                "error_type": error_type,
            }
            fh.write(json.dumps(payload) + "\n")
            n += 1
    logger.info("Wrote %d error records to %s", n, errors_file)
    return n


def confusion_pairs(records: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    """Most frequent (true_label -> pred_label) confusions, ranked by count."""
    counter = Counter(
        (rec["true_label"], rec["pred_label"])
        for rec in records
        if not rec.get("exact_match") and rec.get("pred_label") is not None
    )
    return [
        {"true_label": true_label, "pred_label": pred_label, "count": count}
        for (true_label, pred_label), count in counter.most_common(top_k)
    ]


def build_confusion_matrix(
    records: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Build a machine-readable confusion matrix over the Banking77 classes."""
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    off_vocab = 0
    for rec in records:
        pred = rec.get("pred_label")
        if pred is None:
            off_vocab += 1
            continue
        matrix[index[rec["true_label"]]][index[pred]] += 1
    return {
        "labels": labels,
        "matrix": matrix,
        "off_vocabulary_outputs": off_vocab,
        "confusion_pairs": confusion_pairs(records),
    }


def save_confusion_matrix(matrix: dict[str, Any], path: str | Path) -> Path:
    """Persist the confusion matrix JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    logger.info("Saved confusion matrix to %s", path)
    return path


def plot_confusion_matrix(matrix: dict[str, Any], path: str | Path) -> Path | None:
    """Optionally render the confusion matrix heatmap as PNG (needs matplotlib).

    With 77 classes the raw matrix is unreadable, so only the rows (true labels)
    involved in the top confusions are shown.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed; skipping confusion matrix PNG")
        return None

    labels: list[str] = matrix["labels"]
    mat = np.asarray(matrix["matrix"], dtype=float)
    pairs = matrix["confusion_pairs"][:10]
    focus = sorted({p["true_label"] for p in pairs} | {p["pred_label"] for p in pairs})
    idx = [labels.index(label) for label in focus]
    sub = mat[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(1.1 * len(focus) + 2, 1.1 * len(focus) + 2))
    im = ax.imshow(sub, cmap="Blues")
    ax.set_xticks(range(len(focus)), labels=focus, rotation=90)
    ax.set_yticks(range(len(focus)), labels=focus)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (labels involved in top confusions)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", path)
    return path

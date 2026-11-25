#!/usr/bin/env python3
"""Train the TF-IDF + Logistic Regression baseline and save its metrics."""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

from llm_structured_output.baseline.tfidf import (
    evaluate_baseline,
    save_baseline_outputs,
    train_baseline,
)
from llm_structured_output.config import Config, set_seed
from llm_structured_output.data.load_dataset import load_banking77

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/banking77.yaml", help="YAML config path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    train, validation, test, labels = load_banking77(config)

    pipeline = train_baseline(train, validation, config.seed)
    metrics = evaluate_baseline(pipeline, test, labels)
    print(
        f"Baseline test accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    save_baseline_outputs(
        pipeline,
        metrics,
        test,
        metrics_file=config.output.get("tfidf_metrics_file", "outputs/tfidf_metrics.json"),
        predictions_file=config.output.get(
            "tfidf_predictions_file", "outputs/tfidf_predictions.jsonl"
        ),
    )


if __name__ == "__main__":
    main()

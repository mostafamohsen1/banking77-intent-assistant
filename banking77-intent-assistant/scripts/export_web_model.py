#!/usr/bin/env python3
"""Export the TF-IDF classifier so index.html can classify in the browser.

A static host (such as the preview) has no /api/ask endpoint, so the page falls
back to this file and runs the same weights scripts/webapp.py serves. The
coefficients are quantized to int8 per class; the script reports how the
quantized model scores on the official Banking77 test split.

Usage: .venv/bin/python scripts/export_web_model.py [--output outputs/web_model.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from datasets import load_dataset

    from llm_structured_output.baseline.tfidf import train_baseline
    from llm_structured_output.baseline.web_export import evaluate_web_model, export_web_model
    from llm_structured_output.config import Config, set_seed
    from llm_structured_output.data.load_dataset import extract_labels

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/banking77.yaml", help="YAML config path")
    parser.add_argument("--output", default=None, help="overrides web_model.model_file")
    parser.add_argument("--metrics-output", default=None, help="overrides web_model.metrics_file")
    parser.add_argument("--seed", type=int, default=None, help="overrides project.seed")
    parser.add_argument(
        "--min-confidence", type=float, default=None, help="overrides web_model.min_confidence"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    settings = config.raw.get("web_model", {})
    output = Path(args.output or settings.get("model_file", "outputs/web_model.json"))
    metrics_output = Path(
        args.metrics_output or settings.get("metrics_file", "outputs/web_model_metrics.json")
    )
    seed = config.seed if args.seed is None else args.seed
    min_confidence = float(
        settings.get("min_confidence", 0.15) if args.min_confidence is None else args.min_confidence
    )

    set_seed(seed)
    raw = load_dataset(config.hf_dataset_id)
    train = raw[config.train_split]
    test = raw[config.test_split]
    labels = extract_labels(train, test)

    pipeline = train_baseline(train, train, seed=seed)
    model = export_web_model(pipeline, labels, min_confidence)
    metrics = evaluate_web_model(model, pipeline, test)
    model["test_accuracy"] = metrics["accuracy"]
    model["num_test_examples"] = metrics["num_test_examples"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    metrics.update(
        config=args.config,
        dataset=config.hf_dataset_id,
        train_split=config.train_split,
        num_train_examples=len(train),
        test_split=config.test_split,
        seed=seed,
        model_file=str(output),
        model_file_bytes=output.stat().st_size,
    )
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"Wrote {output} ({metrics['model_file_bytes'] / 1e6:.2f} MB) and {metrics_output}: "
        f"{metrics['num_labels']} labels, {metrics['num_features']} features, "
        f"test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"(float model {metrics['float_model_accuracy']:.4f}, "
        f"agreement {metrics['agreement_with_float_model']:.4f})"
    )


if __name__ == "__main__":
    main()

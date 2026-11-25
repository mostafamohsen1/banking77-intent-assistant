#!/usr/bin/env python3
"""Evaluate a fine-tuned LoRA checkpoint on the Banking77 test split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")

from llm_structured_output.config import Config, set_seed
from llm_structured_output.data.load_dataset import load_banking77
from llm_structured_output.evaluation.error_analysis import (
    build_confusion_matrix,
    plot_confusion_matrix,
    save_confusion_matrix,
    write_error_records,
)
from llm_structured_output.evaluation.evaluate import evaluate_lora_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/banking77.yaml", help="YAML config path")
    parser.add_argument(
        "--adapter",
        required=False,
        help="Path to the LoRA adapter (default: latest under output.checkpoint_dir)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)

    adapter = args.adapter or _latest_checkpoint(config.output.get("checkpoint_dir", "checkpoints"))
    if adapter is None:
        raise SystemExit("No adapter path given and no checkpoint found under the checkpoint dir.")
    print(f"Evaluating adapter: {adapter}")

    _train, _validation, test, labels = load_banking77(config)
    metrics, records = evaluate_lora_checkpoint(adapter, test, labels, config)
    print(
        f"Test accuracy={metrics['accuracy']:.4f} exact_match={metrics['exact_match']:.4f} "
        f"valid_json_rate={metrics['valid_json_rate']:.4f}"
    )

    write_error_records(records, config.output.get("errors_file", "outputs/errors.jsonl"))
    matrix = build_confusion_matrix(records, labels)
    save_confusion_matrix(
        matrix, config.output.get("confusion_matrix_file", "outputs/confusion_matrix.json")
    )
    try:
        plot_confusion_matrix(matrix, "outputs/confusion_matrix.png")
    except Exception:  # plotting is best-effort on headless servers
        pass


def _latest_checkpoint(checkpoint_dir: str) -> str | None:
    base = Path(checkpoint_dir)
    if not base.exists():
        return None
    candidates = sorted(
        (p for p in base.rglob("*") if (p / "adapter_config.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


if __name__ == "__main__":
    main()

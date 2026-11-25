"""Batch evaluation of a fine-tuned (or base) model on the Banking77 test split."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from llm_structured_output.data.prepare_dataset import build_prompt, target_output
from llm_structured_output.data.schema import Prediction
from llm_structured_output.evaluation.json_validator import parse_model_output
from llm_structured_output.evaluation.metrics import compute_metrics, save_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_predictions(
    model: Any,
    tokenizer: Any,
    test: Any,
    labels: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate and validate structured predictions for every test example."""
    label_set = set(labels)
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    prompts = [build_prompt(tokenizer, row["text"], labels) for row in test]
    records: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="evaluating", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        batch_rows = test.select(range(start, min(start + batch_size, len(test))))
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        do_sample = temperature > 0.0
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for row, raw in zip(batch_rows, responses, strict=True):
            true_id = row["label"]
            true_label = labels[true_id] if isinstance(true_id, int) else true_id
            prediction, error = parse_model_output(raw)
            pred_label: str | None = None
            if prediction is not None:
                # Off-vocabulary labels count as a schema violation, not a class.
                if prediction.intent in label_set:
                    pred_label = prediction.intent
                else:
                    pred_label = None
                    error = "unknown_label"
            records.append(
                {
                    "input": row["text"],
                    "raw_output": raw,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "error": error,
                    "valid_json": prediction is not None and pred_label is not None,
                    "exact_match": pred_label == true_label,
                }
            )
    return records


def evaluate_model(
    model: Any,
    tokenizer: Any,
    test: Any,
    labels: list[str],
    config: Any,
    predictions_file: str | Path | None = None,
    metrics_file: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run generation on the test split, compute metrics and persist artifacts."""
    ev = config.evaluation
    records = generate_predictions(
        model,
        tokenizer,
        test,
        labels,
        batch_size=int(ev.get("batch_size", 8)),
        max_new_tokens=int(ev.get("max_new_tokens", 64)),
        temperature=float(ev.get("temperature", 0.0)),
    )
    metrics = compute_metrics(records)
    logger.info(
        "Test results: accuracy=%.4f exact_match=%.4f valid_json=%.4f",
        metrics["accuracy"],
        metrics["exact_match"],
        metrics["valid_json_rate"],
    )
    if predictions_file:
        predictions_file = Path(predictions_file)
        predictions_file.parent.mkdir(parents=True, exist_ok=True)
        with predictions_file.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        logger.info("Saved predictions to %s", predictions_file)
    if metrics_file:
        save_metrics(metrics, metrics_file)
        logger.info("Saved metrics to %s", metrics_file)
    return metrics, records


def evaluate_lora_checkpoint(
    adapter_path: str | Path,
    test: Any,
    labels: list[str],
    config: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a saved LoRA adapter and evaluate it end to end."""
    from llm_structured_output.training.train import load_trained_model

    model, tokenizer = load_trained_model(adapter_path, config)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_model(
        model,
        tokenizer,
        test,
        labels,
        config,
        predictions_file=config.output.get("predictions_file"),
        metrics_file=config.output.get("metrics_file"),
    )


def example_target(intent: str) -> str:
    """Expose the canonical target string (used by docs and tests)."""
    return target_output(intent)


__all__ = [
    "Prediction",
    "evaluate_lora_checkpoint",
    "evaluate_model",
    "example_target",
    "generate_predictions",
]

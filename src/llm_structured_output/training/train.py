"""SFT fine-tuning of a chat model with LoRA adapters."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from llm_structured_output.config import Config, resolve_torch_dtype, save_run_config
from llm_structured_output.data.prepare_dataset import build_prompt, target_output
from llm_structured_output.training.callbacks import MetricLoggerCallback
from llm_structured_output.training.lora import apply_lora

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: Config):
    """Load the base causal LM and tokenizer according to the config."""
    dtype = resolve_torch_dtype(config.torch_dtype)
    kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = dtype
    else:
        logger.info("CUDA not available; loading model in float32 on CPU")
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Loaded model %s (dtype=%s)", config.model_name, kwargs["torch_dtype"])
    return model, tokenizer


def run_training(
    config: Config, train: Any, validation: Any, labels: list[str] | None = None
) -> Any:
    """Run LoRA SFT and return the path of the final adapter checkpoint.

    The dataset is passed as ``prompt``/``completion`` pairs so TRL computes
    the loss on the assistant JSON only (completion-only loss) instead of
    also teaching the model to parrot the prompt.
    """
    model, tokenizer = load_model_and_tokenizer(config)

    def _to_prompt_completion(row: dict[str, Any]) -> dict[str, str]:
        label = row["label"]
        intent = labels[label] if labels and isinstance(label, int) else label
        return {
            "prompt": build_prompt(tokenizer, row["text"], labels),
            "completion": target_output(intent),
        }

    keep = {"prompt", "completion"}
    train_pc = train.map(
        _to_prompt_completion,
        remove_columns=[c for c in train.column_names if c not in keep],
        desc="formatting train",
    )
    val_pc = validation.map(
        _to_prompt_completion,
        remove_columns=[c for c in validation.column_names if c not in keep],
        desc="formatting validation",
    )

    model = apply_lora(model, config.lora)

    t = config.training
    output_dir = config.output.get("checkpoint_dir", "checkpoints")
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(t.get("epochs", 2)),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 8)),
        warmup_steps=max(1, int(t.get("warmup_ratio", 0.05) * 100)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 250)),
        eval_strategy="steps",
        eval_steps=int(t.get("eval_steps", 250)),
        save_total_limit=int(t.get("save_total_limit", 3)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        report_to=[],
        seed=config.seed,
        packing=False,
        completion_only_loss=True,
        max_length=config.max_length,
        use_cpu=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )

    metric_logger = MetricLoggerCallback(output_dir)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_pc,
        eval_dataset=val_pc,
        processing_class=tokenizer,
        callbacks=[metric_logger],
    )

    start = time.time()
    resume = config.training.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume)
    duration = time.time() - start
    logger.info("Training finished in %.1f s", duration)

    final_dir = trainer.save_model()
    metric_logger.save()
    save_run_config(config, training_duration_seconds=duration)
    logger.info("Final adapter saved to %s", final_dir)
    return final_dir


def load_trained_model(adapter_path: str | Any, config: Config | None = None):
    """Load a base model merged with a saved LoRA adapter for evaluation."""
    path = str(adapter_path)
    kwargs: dict[str, Any] = {}
    if config is not None and torch.cuda.is_available():
        kwargs["torch_dtype"] = resolve_torch_dtype(config.torch_dtype)
    model = AutoPeftModelForCausalLM.from_pretrained(path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

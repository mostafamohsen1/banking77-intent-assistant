"""LoRA (PEFT) adapter construction."""

from __future__ import annotations

import logging
from typing import Any

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def build_lora_config(settings: dict[str, Any]) -> LoraConfig:
    """Build a ``LoraConfig`` from the ``lora`` section of the experiment YAML."""
    lora = LoraConfig(
        r=int(settings.get("r", 16)),
        lora_alpha=int(settings.get("alpha", 32)),
        lora_dropout=float(settings.get("dropout", 0.05)),
        bias=settings.get("bias", "none"),
        task_type=settings.get("task_type", "CAUSAL_LM"),
        target_modules=list(settings.get("target_modules", ["q_proj", "v_proj"])),
    )
    logger.info(
        "LoRA config: r=%d alpha=%d dropout=%.2f targets=%s",
        lora.r,
        lora.lora_alpha,
        lora.lora_dropout,
        lora.target_modules,
    )
    return lora


def apply_lora(model: Any, settings: dict[str, Any]) -> Any:
    """Wrap a causal LM with LoRA adapters and report trainable parameters."""
    model = get_peft_model(model, build_lora_config(settings))
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)", trainable, total, pct
    )
    model.print_trainable_parameters()
    return model

"""Configuration loading and experiment reproducibility helpers."""

from __future__ import annotations

import json
import logging
import platform
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/banking77.yaml")


@dataclass
class Config:
    """Typed view over the experiment YAML configuration.

    The YAML remains the source of truth; this dataclass validates and
    centralizes access so no section of the codebase reads raw dicts.
    """

    project_name: str = "banking77-structured-llm"
    seed: int = 42

    dataset: str = "banking77"
    hf_dataset_id: str = "PolyAI/banking77"
    train_split: str = "train"
    test_split: str = "test"
    validation_ratio: float = 0.1
    max_length: int = 512
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False

    training: dict[str, Any] = field(default_factory=dict)
    lora: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a YAML configuration file into a validated :class:`Config`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cls(
            project_name=data.get("project", {}).get("name", cls.project_name),
            seed=int(data.get("project", {}).get("seed", 42)),
            dataset=data.get("data", {}).get("dataset", "banking77"),
            hf_dataset_id=data.get("data", {}).get("hf_dataset_id", "PolyAI/banking77"),
            train_split=data.get("data", {}).get("train_split", "train"),
            test_split=data.get("data", {}).get("test_split", "test"),
            validation_ratio=float(data.get("data", {}).get("validation_ratio", 0.1)),
            max_length=int(data.get("data", {}).get("max_length", 512)),
            max_train_samples=data.get("data", {}).get("max_train_samples"),
            max_validation_samples=data.get("data", {}).get("max_validation_samples"),
            max_test_samples=data.get("data", {}).get("max_test_samples"),
            model_name=data.get("model", {}).get("name", cls.model_name),
            torch_dtype=data.get("model", {}).get("torch_dtype", "bfloat16"),
            trust_remote_code=bool(data.get("model", {}).get("trust_remote_code", False)),
            training=data.get("training", {}),
            lora=data.get("lora", {}),
            evaluation=data.get("evaluation", {}),
            output=data.get("output", {}),
            raw=data,
        )
        if not 0.0 < cfg.validation_ratio < 1.0:
            raise ValueError(f"validation_ratio must be in (0, 1), got {cfg.validation_ratio}")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the configuration."""
        return asdict(self)

    def ensure_output_dirs(self) -> None:
        """Create the output and checkpoint directories if missing."""
        Path(self.output.get("checkpoint_dir", "checkpoints")).mkdir(parents=True, exist_ok=True)
        for key in ("metrics_file", "predictions_file", "errors_file", "run_config_file"):
            value = self.output.get(key)
            if value:
                Path(value).parent.mkdir(parents=True, exist_ok=True)


def resolve_torch_dtype(name: str) -> torch.dtype:
    """Map a YAML dtype string to a ``torch.dtype``."""
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = (name or "float32").lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported torch_dtype '{name}'. Supported values: {sorted(set(mapping))}"
        )
    return mapping[key]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu(expected_gb: float = 20.0) -> dict[str, Any]:
    """Report GPU availability with a helpful message when unsuitable."""
    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        info["message"] = (
            "CUDA is not available. Full fine-tuning requires a GPU; use the smoke test "
            "config (configs/smoke_test.yaml) for a CPU pipeline check."
        )
        return info
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    info.update(
        {
            "device": props.name,
            "total_memory_gb": round(total_gb, 1),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    if total_gb < expected_gb:
        info["message"] = (
            f"GPU has {total_gb:.1f} GB memory; the full experiment assumes ~{expected_gb:.0f} GB. "
            "Reduce batch size / sequence length or use a smaller model."
        )
    logger.info("GPU detected: %s (%.1f GB)", info.get("device"), total_gb)
    return info


def save_run_config(config: Config, training_duration_seconds: float | None = None) -> Path:
    """Persist model/data/training metadata and environment versions for reproducibility."""
    import datasets
    import peft
    import sklearn
    import transformers
    import trl

    payload = {
        "config": config.to_dict(),
        "environment": {
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "datasets": datasets.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "training_duration_seconds": training_duration_seconds,
    }
    out_path = Path(config.output.get("run_config_file", "outputs/run_config.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Saved run configuration to %s", out_path)
    return out_path

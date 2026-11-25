"""Tests for config loading and reproducibility helpers."""

from __future__ import annotations

import pytest

from llm_structured_output.config import Config, resolve_torch_dtype


def test_load_banking77_config():
    cfg = Config.from_yaml("configs/banking77.yaml")
    assert cfg.model_name.startswith("Qwen/")
    assert cfg.hf_dataset_id == "PolyAI/banking77"
    assert cfg.seed == 42
    assert cfg.validation_ratio == 0.1


def test_load_smoke_config():
    cfg = Config.from_yaml("configs/smoke_test.yaml")
    assert cfg.max_train_samples == 100
    assert cfg.torch_dtype == "float32"
    assert cfg.output["metrics_file"].startswith("outputs/")


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "nope.yaml")


def test_invalid_validation_ratio_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  validation_ratio: 1.5\n")
    with pytest.raises(ValueError):
        Config.from_yaml(p)


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("bf16") is not None
    assert resolve_torch_dtype("float32") is not None
    with pytest.raises(ValueError):
        resolve_torch_dtype("float64")

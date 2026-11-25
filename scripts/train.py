#!/usr/bin/env python3
"""Run LoRA SFT fine-tuning on Banking77 (requires a GPU)."""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train")

sys.path.insert(0, "src")

from llm_structured_output.config import Config, check_gpu, set_seed  # noqa: E402
from llm_structured_output.data.load_dataset import load_banking77  # noqa: E402
from llm_structured_output.training.train import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/banking77.yaml", help="YAML config path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    gpu = check_gpu()
    if not gpu["cuda_available"] and config.torch_dtype.lower() not in {"float32", "fp32"}:
        raise SystemExit(
            "CUDA GPU required for fine-tuning. Run scripts/train.py on a GPU machine, "
            "or use configs/smoke_test.yaml (float32) for a tiny CPU pipeline check."
        )
    if not gpu["cuda_available"]:
        logger.warning("Running fine-tuning on CPU (float32 smoke config) — slow by design.")

    train, validation, test, labels = load_banking77(config)
    run_training(config, train, validation, labels)


if __name__ == "__main__":
    main()

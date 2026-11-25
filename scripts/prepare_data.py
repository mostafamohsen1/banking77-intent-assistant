#!/usr/bin/env python3
"""Download Banking77, split it and export the processed JSONL splits."""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

from llm_structured_output.config import Config, set_seed
from llm_structured_output.data.load_dataset import load_banking77
from llm_structured_output.data.prepare_dataset import export_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/banking77.yaml", help="YAML config path")
    parser.add_argument(
        "--out-dir", default="data/processed", help="Output directory for processed JSONL"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.seed)
    train, validation, test, labels = load_banking77(config)
    paths = export_processed(train, validation, test, args.out_dir)
    print(f"Labels ({len(labels)}): {labels[:5]} ...")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save a standalone checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model id or path (e.g. Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument("--adapter", required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--out", required=True, help="Output directory for the merged model")
    args = parser.parse_args()

    print(f"Loading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Attaching adapter {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging and unloading adapters...")
    model = model.merge_and_unload()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")
    print("Serve it with: ./scripts/serve.sh <model-path>")


if __name__ == "__main__":
    main()

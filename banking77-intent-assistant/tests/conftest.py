"""Shared fixtures: synthetic Banking77-like data and a fake tokenizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LABELS = [
    "card_arrival",
    "card_payment_wrong_exchange_rate",
    "exchange_rate",
    "top_up_failed",
    "transfer_not_received_by_recipient",
    "verify_my_identity",
]


class FakeTokenizer:
    """Minimal chat-template tokenizer stub (Qwen-Chat style markers)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        longest = max(len(t.split()) for t in texts)
        return {
            "input_ids": [[0] * longest for _ in texts],
            "attention_mask": [[1] * longest for _ in texts],
        }


class FakeDataset(list):
    """List-of-dicts that quacks like a HF Dataset for the pieces we use."""

    features = {"label": None}

    def __init__(self, rows):
        super().__init__(rows)

    def select(self, indices):
        return FakeDataset([self[i] for i in indices])

    def map(self, fn):
        return FakeDataset([fn(dict(row)) for row in self])

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        if isinstance(key, slice):
            return FakeDataset(list.__getitem__(self, key))
        return [row[key] for row in self]


@pytest.fixture
def labels():
    return list(LABELS)


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def mini_splits():
    def rows(n, offset=0):
        return FakeDataset(
            [
                {
                    "text": f"message {offset + i} about card and money",
                    "label": LABELS[(offset + i) % len(LABELS)],
                }
                for i in range(n)
            ]
        )

    return rows(40), rows(10, offset=40), rows(12, offset=50)


@pytest.fixture
def sample_records(labels):
    """One clean hit, one wrong-class, one invalid-JSON and one off-vocab record."""
    return [
        {"input": "m1", "raw_output": json.dumps({"intent": labels[0]}), "true_label": labels[0],
         "pred_label": labels[0], "error": None, "valid_json": True, "exact_match": True},
        {"input": "m2", "raw_output": json.dumps({"intent": labels[1]}), "true_label": labels[2],
         "pred_label": labels[1], "error": None, "valid_json": True, "exact_match": False},
        {"input": "m3", "raw_output": "sorry, I cannot help with that", "true_label": labels[3],
         "pred_label": None, "error": "invalid_json", "valid_json": False, "exact_match": False},
        {"input": "m4", "raw_output": json.dumps({"intent": "made_up_intent"}),
         "true_label": labels[4], "pred_label": None,
         "error": "unknown_label", "valid_json": False, "exact_match": False},
    ]

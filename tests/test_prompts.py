"""Tests for prompt formatting and chat-template handling."""

from __future__ import annotations

import json

from llm_structured_output.data.prepare_dataset import (
    SYSTEM_PROMPT,
    build_prompt,
    format_example,
    target_output,
)


def test_target_output_is_canonical_json(labels):
    assert target_output(labels[0]) == json.dumps({"intent": labels[0]})


def test_training_example_includes_assistant_target(fake_tokenizer, labels):
    text = "my card never arrived"
    rendered = format_example(fake_tokenizer, text, labels[0])
    assert SYSTEM_PROMPT in rendered
    assert text in rendered
    assert f'{{"intent": "{labels[0]}"}}' in rendered
    assert "<|im_start|>assistant" in rendered


def test_inference_prompt_stops_at_assistant_header(fake_tokenizer):
    rendered = build_prompt(fake_tokenizer, "where is my card?")
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "assistant\n{" not in rendered
    assert "where is my card?" in rendered


def test_system_prompt_demands_json_only():
    assert "ONLY valid JSON" in SYSTEM_PROMPT
    assert '"intent"' in SYSTEM_PROMPT

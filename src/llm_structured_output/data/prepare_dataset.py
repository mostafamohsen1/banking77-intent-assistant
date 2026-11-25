"""Prompt construction, chat-template formatting and processed-dataset export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a banking customer-support intent classification assistant.\n"
    "Classify the user's message into exactly one Banking77 intent.\n"
    "The intent must be exactly one of the allowed intents listed in the message.\n"
    "Return ONLY valid JSON.\n"
    'The JSON schema is: {"intent": "string"}\n'
    "Do not include explanations, markdown, code fences, or additional text."
)


def target_output(intent: str) -> str:
    """The canonical assistant target string for a given intent label."""
    return json.dumps({"intent": intent})


def build_user_message(text: str, labels: list[str] | None = None) -> str:
    """The user-turn content for a support message.

    When ``labels`` is given the allowed intent names are appended so the
    model can only pick from the real Banking77 label vocabulary.
    """
    message = text.strip()
    if labels:
        message += (
            "\n\nAllowed intents (reply with exactly one of these):\n"
            + ", ".join(labels)
        )
    return message


def format_example(
    tokenizer: Any, text: str, intent: str | None, labels: list[str] | None = None
) -> str:
    """Render one example with the tokenizer's native chat template.

    When ``intent`` is given the assistant turn contains the JSON target
    (training); otherwise the prompt ends at the assistant header (inference).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(text, labels)},
    ]
    if intent is not None:
        messages.append({"role": "assistant", "content": target_output(intent)})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=(intent is None)
    )


def build_prompt(tokenizer: Any, text: str, labels: list[str] | None = None) -> str:
    """Render an inference prompt (no assistant content) with the chat template."""
    return format_example(tokenizer, text, intent=None, labels=labels)


def prepare_splits(
    train: Any,
    validation: Any,
    test: Any,
    tokenizer: Any,
    labels: list[str] | None = None,
) -> tuple[Any, Any, Any]:
    """Attach ``target_text`` (chat-formatted) and ``target_json`` to each split."""
    def _map(split: Any, with_target: bool) -> Any:
        def _add(row: dict[str, Any]) -> dict[str, Any]:
            intent = row["label"] if with_target else None
            return {
                "target_json": target_output(row["label"]) if with_target else None,
                "text": format_example(tokenizer, row["text"], intent, labels),
            }

        return split.map(_add)

    train = _map(train, True)
    validation = _map(validation, True)
    test = _map(test, False)
    logger.info("Prepared chat-formatted splits: train=%d validation=%d test=%d",
                len(train), len(validation), len(test))
    return train, validation, test


def export_processed(
    train: Any,
    validation: Any,
    test: Any,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write the prepared splits to JSONL under ``data/processed``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        path = out_dir / f"{name}.jsonl"
        split.to_json(str(path))
        paths[name] = path
        logger.info("Wrote %d rows to %s", len(split), path)
    return paths

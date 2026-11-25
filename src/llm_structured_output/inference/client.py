"""OpenAI-compatible client with strict output validation and bounded retries."""

from __future__ import annotations

import logging
import time

import requests

from llm_structured_output.data.prepare_dataset import SYSTEM_PROMPT, build_user_message
from llm_structured_output.data.schema import Prediction
from llm_structured_output.evaluation.json_validator import parse_model_output

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "banking77-structured"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_INPUT_CHARS = 2048
DEFAULT_MAX_RETRIES = 2


class IntentClient:
    """Client for the vLLM OpenAI-compatible endpoint.

    Validates every response against the structured-output schema and retries
    (bounded, with exponential backoff) on invalid JSON or schema violations.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        temperature: float = 0.0,
        max_tokens: int = 64,
        labels: list[str] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_input_chars = max_input_chars
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.labels = labels
        self.session = session or requests.Session()

    def _post_chat(self, user_message: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(user_message, self.labels)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self.session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def predict(self, message: str) -> Prediction:
        """Classify one message; raise :class:`ValueError` after the retries fail."""
        message = message[: self.max_input_chars]
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            raw = self._post_chat(message)
            prediction, error = parse_model_output(raw)
            if prediction is not None:
                return prediction
            last_error = error
            logger.warning(
                "Attempt %d produced invalid output (%s); retrying", attempt + 1, error
            )
            time.sleep(min(2**attempt, 5))
        raise ValueError(
            f"Model failed to return a valid prediction after {self.max_retries + 1} "
            f"attempts (last error: {last_error})"
        )

    def health(self) -> bool:
        """Return True when the endpoint answers the /models list."""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=self.timeout)
            return response.ok
        except requests.RequestException:
            return False

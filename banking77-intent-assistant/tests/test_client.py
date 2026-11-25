"""Tests for the inference client against a stubbed HTTP endpoint."""

from __future__ import annotations

import json as jsonlib

import pytest
import requests

from llm_structured_output.inference.client import IntentClient


class StubSession:
    """Replaces requests.Session; returns queued responses per call."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        response = requests.Response()
        response.status_code = 200
        response._content = jsonlib.dumps(
            {"choices": [{"message": {"content": payload}}]}
        ).encode()
        return response

    def get(self, url, timeout=None):
        response = requests.Response()
        response.status_code = 200
        return response


def test_predict_validates_and_returns_prediction(labels):
    client = IntentClient(session=StubSession([jsonlib.dumps({"intent": labels[0]})]))
    assert client.predict("card lost").intent == labels[0]


def test_predict_retries_on_invalid_json_then_succeeds(labels):
    session = StubSession(["garbage answer", jsonlib.dumps({"intent": labels[1]})])
    client = IntentClient(session=session)
    assert client.predict("card lost").intent == labels[1]
    assert session.calls == 2


def test_predict_raises_after_bounded_retries():
    client = IntentClient(session=StubSession(["still not json"]), max_retries=2)
    with pytest.raises(ValueError, match="valid prediction"):
        client.predict("card lost")


def test_input_truncated_to_max_length():
    seen = {}

    class Recorder(StubSession):
        def post(self, url, json=None, timeout=None):
            seen["content"] = json["messages"][1]["content"]
            return super().post(url, json=json, timeout=timeout)

    client = IntentClient(session=Recorder([jsonlib.dumps({"intent": "x"})]), max_input_chars=10)
    client.predict("a" * 500)
    assert len(seen["content"]) == 10


def test_health_ok():
    client = IntentClient(session=StubSession([]))
    assert client.health() is True

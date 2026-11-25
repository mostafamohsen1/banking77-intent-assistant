#!/usr/bin/env python3
"""Local web app: serves the UI and a /api/ask endpoint backed by the best model.

Answering strategy, in order:
1. If an LLM provider key is set (OPENAI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY),
   classify with that chat model under the strict {"intent": ...} schema.
2. Otherwise use the TF-IDF + LogisticRegression classifier trained on the full
   Banking77 train split (89.5% test accuracy), with a confidence score and an
   out-of-domain fallback for questions that match no banking intent.
3. Optionally (--adapter), also load the LoRA model and use it instead of TF-IDF.

Usage: .venv/bin/python scripts/webapp.py [--adapter checkpoints/smoke_test] --port 8000
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



_state: dict[str, object] = {}

# Below this confidence the question is treated as out of domain.
MIN_CONFIDENCE = 0.15

OUT_OF_DOMAIN_INTENT = "out_of_domain"

# index.html ships with the API switched off so a static host answers from the
# browser model; this server switches it on when it serves the page.
ASK_API_STATIC = "const HAS_API = false;"
ASK_API_SERVER = "const HAS_API = true;"


def _load_datasets():
    from datasets import load_dataset

    from llm_structured_output.data.load_dataset import extract_labels

    raw = load_dataset("PolyAI/banking77")
    return {"train": raw["train"], "test": raw["test"]}, extract_labels(
        raw["train"], raw["test"]
    )


def load_tfidf() -> None:
    """Train the TF-IDF classifier on the full Banking77 train split."""
    from llm_structured_output.baseline.tfidf import train_baseline

    raw, labels = _load_datasets()
    pipeline = train_baseline(raw["train"], raw["train"], seed=42)
    _state.update(pipeline=pipeline, labels=labels)
    print("TF-IDF classifier trained on Banking77 train split")


def load_model(adapter_path: str, device: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    model.eval()
    _splits, labels = _load_datasets()
    _state.update(tokenizer=tokenizer, model=model, labels=labels, device=device)
    print(f"Loaded adapter {adapter_path} on {device} with {len(labels)} labels")


# ---------------------------------------------------------------- LLM providers

PROVIDERS = {
    "OPENROUTER_API_KEY": (
        "https://openrouter.ai/api/v1/chat/completions",
        "openai/gpt-4o-mini",
    ),
    "GROQ_API_KEY": (
        "https://api.groq.com/openai/v1/chat/completions",
        "llama-3.3-70b-versatile",
    ),
    "OPENAI_API_KEY": (
        "https://api.openai.com/v1/chat/completions",
        "gpt-4o-mini",
    ),
}


def _provider() -> tuple[str, str, str] | None:
    for key, (url, model) in PROVIDERS.items():
        if os.environ.get(key):
            return os.environ[key], url, model
    return None


_INTENT_RE = re.compile(r"[a-z_]+")


def _extract_intent(text: str, labels: list[str]) -> str | None:
    for match in _INTENT_RE.findall(text.lower()):
        if match in labels:
            return match
    return None


def classify_with_llm(message: str, labels: list[str]) -> tuple[str, str | None]:
    """Ask a chat LLM to pick one intent; returns (intent, error)."""
    import httpx

    api_key, url, model = _provider()  # type: ignore[misc]
    system = (
        "You classify banking customer-support questions. "
        f"Answer with strict JSON: {{\"intent\": \"<one of: {', '.join(labels)}>\"}}. "
        'If the question is not about banking, answer {"intent": "out_of_domain"}.'
    )
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": message},
                ],
                "temperature": 0,
                "max_tokens": 60,
                "timeout": 20,
            },
            timeout=25,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # network, auth, rate limit -> fall back
        return "", f"llm_api_error: {exc}"
    intent = _extract_intent(content, set(labels) | {OUT_OF_DOMAIN_INTENT})
    if intent is None:
        return "", "llm_api_error: unparseable reply"
    return intent, None


# ---------------------------------------------------------------------- routes


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Banking77 structured-output demo")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/")
    def index() -> HTMLResponse:
        page = Path(__file__).resolve().parents[1] / "index.html"
        html = page.read_text(encoding="utf-8")
        return HTMLResponse(html.replace(ASK_API_STATIC, ASK_API_SERVER))

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ready": "pipeline" in _state or "model" in _state,
            "backend": "llm_api"
            if _provider()
            else ("model" in _state and "lora" or "tfidf"),
        }

    app.mount(
        "/outputs",
        StaticFiles(directory=Path(__file__).resolve().parents[1] / "outputs"),
        name="outputs",
    )

    @app.post("/api/ask")
    def api_ask(payload: dict[str, str]) -> dict[str, object]:
        return ask(payload)

    return app


def _classify_tfidf(message: str) -> tuple[str, float, list[tuple[str, float]]]:
    pipeline = _state["pipeline"]
    labels: list[str] = _state["labels"]  # type: ignore[assignment]
    proba = pipeline.predict_proba([message])[0]
    ranked = sorted(
        ((labels[int(c)], p) for c, p in zip(pipeline.classes_, proba.tolist())),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[0][0], ranked[0][1], ranked[:3]


def ask(payload: dict[str, str]) -> dict[str, object]:
    message = (payload.get("message") or "").strip()
    if not message:
        return {"reply": None, "error": "empty_message"}

    start = time.time()
    labels: list[str] = _state.get("labels") or []

    # 1) LLM API when a key is configured.
    if _provider() and labels:
        intent, error = classify_with_llm(message, labels)
        if error is None:
            return {
                "reply": {"intent": intent},
                "confidence": None,
                "backend": "llm_api",
                "error": None,
                "latency_ms": round((time.time() - start) * 1000),
            }
        # fall through to local model on API failure, but surface why
        api_error = error
    else:
        api_error = None

    # 2) Fine-tuned LoRA model if an adapter was loaded and no API answered.
    if "model" in _state:
        import torch

        from llm_structured_output.data.prepare_dataset import build_prompt
        from llm_structured_output.evaluation.json_validator import parse_model_output

        tokenizer = _state["tokenizer"]
        model = _state["model"]
        device: str = _state["device"]  # type: ignore[assignment]
        prompt = build_prompt(tokenizer, message, labels)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        prediction, error = parse_model_output(generated)
        return {
            "reply": prediction.model_dump() if prediction else None,
            "raw": generated.strip(),
            "backend": "lora",
            "error": error,
            "latency_ms": round((time.time() - start) * 1000),
        }

    # 3) TF-IDF classifier: fast, 89.5% accuracy, always available.
    if "pipeline" not in _state:
        return {"reply": None, "error": "model_not_loaded"}
    intent, confidence, top3 = _classify_tfidf(message)
    if confidence < MIN_CONFIDENCE:
        return {
            "reply": {"intent": OUT_OF_DOMAIN_INTENT},
            "confidence": round(confidence, 4),
            "top3": [{"intent": i, "confidence": round(c, 4)} for i, c in top3],
            "backend": "tfidf",
            "error": api_error or "out_of_domain",
            "latency_ms": round((time.time() - start) * 1000),
        }
    return {
        "reply": {"intent": intent},
        "confidence": round(confidence, 4),
        "top3": [{"intent": i, "confidence": round(c, 4)} for i, c in top3],
        "backend": "tfidf",
        "error": api_error,
        "latency_ms": round((time.time() - start) * 1000),
    }


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter; when omitted the TF-IDF classifier answers.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.adapter:
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        load_model(args.adapter, device)
    else:
        load_tfidf()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

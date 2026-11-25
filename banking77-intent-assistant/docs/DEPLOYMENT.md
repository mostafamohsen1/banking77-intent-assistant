# Deployment guide

## 1. Merge the adapter

```bash
python scripts/merge_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter checkpoints/<checkpoint-with-adapter_config.json> \
  --out checkpoints/merged
```

The merged directory is a standalone causal LM — no PEFT required at inference time.

## 2. Serve with vLLM

```bash
./scripts/serve.sh checkpoints/merged 8000
```

- Endpoint: OpenAI-compatible `POST /v1/chat/completions` on port 8000.
- Served model name: `banking77-structured`.
- `--max-model-len 1024` — prompts are short; this keeps KV cache small.
- Install with `pip install 'llm-structured-output[vllm]'` (vLLM is not in the base
  requirements because it pins its own torch version).

## 3. Client contract

`llm_structured_output.inference.client.IntentClient`:

- enforces the system prompt (schema instruction) server-side as well as in training;
- truncates input to `max_input_chars` (default 2048) — a hard input-length bound;
- validates every completion against the `{"intent": str}` schema;
- retries invalid output up to `max_retries` (default 2) with exponential backoff (2s, 4s);
- applies a per-request `timeout` (default 30 s) and `max_tokens` (default 64) cap;
- raises `ValueError` when no valid prediction is produced, so callers can fall back.

```python
from llm_structured_output.inference.client import IntentClient

client = IntentClient()  # http://localhost:8000/v1
print(client.predict("ATM swallowed my card").intent)
```

## Health and monitoring

- `IntentClient.health()` checks `GET /v1/models` — use it for readiness probes.
- Every response that fails validation is already logged client-side; ship those logs with
  the retry outcome to track production `valid_json_rate`.

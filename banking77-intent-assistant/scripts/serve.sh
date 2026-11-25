#!/usr/bin/env bash
# Serve the fine-tuned model with vLLM (OpenAI-compatible API on :8000).
# Usage: ./scripts/serve.sh <merged-model-path> [port]
set -euo pipefail

MODEL_PATH="${1:?Usage: serve.sh <merged-model-path> [port]}"
PORT="${2:-8000}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM is not installed. Install it with: pip install 'llm-structured-output[vllm]'" >&2
  exit 1
fi

echo "Serving ${MODEL_PATH} on port ${PORT}..."
exec vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name banking77-structured \
  --max-model-len 1024

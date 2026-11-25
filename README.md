# Structured-Output LLM Fine-Tuning — Banking77 Intent Classifier

Fine-tune a small open-source chat LLM (default: **Qwen2.5-1.5B-Instruct**) with **LoRA + SFT**
to map customer-support messages to a strict JSON schema:

```json
{"intent": "card_arrival"}
```

on the **Banking77** benchmark (77 fine-grained banking intents). Includes a classical
TF-IDF baseline, strict output validation, evaluation with per-error breakdown and a
confusion matrix over the 77 classes, and a served inference client over an
OpenAI-compatible API.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 1. Download and prepare the data (CPU)
python scripts/prepare_data.py --config configs/banking77.yaml

# 2. Classical baseline (CPU, ~2 minutes)
python scripts/run_baseline.py --config configs/banking77.yaml

# 3. Fine-tune (GPU required — see Hardware notes)
python scripts/train.py --config configs/banking77.yaml

# 4. Evaluate the checkpoint (GPU recommended)
python scripts/evaluate.py --config configs/banking77.yaml

# 5. Merge the adapter, serve it, and query it
python scripts/merge_adapter.py --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter checkpoints/<run> --out checkpoints/merged
./scripts/serve.sh checkpoints/merged 8000
python scripts/client.py "I have been charged twice this month"
```

### CPU smoke test (no GPU)

```bash
python scripts/prepare_data.py --config configs/smoke_test.yaml
python scripts/train.py     --config configs/smoke_test.yaml   # tiny LoRA run
python scripts/evaluate.py  --config configs/smoke_test.yaml
```

The smoke config caps train/val/test at 100/25/50 examples and uses the 0.5B model;
it verifies the full pipeline end to end in minutes on CPU — it does **not** produce
meaningful quality.

## Repository layout

```
├── configs/
│   ├── banking77.yaml          # full experiment config (GPU)
│   └── smoke_test.yaml         # CPU end-to-end smoke test
├── src/llm_structured_output/
│   ├── config.py               # YAML config, seeds, run metadata
│   ├── data/
│   │   ├── load_dataset.py     # Banking77 loading, leak-free split, labels
│   │   ├── prepare_dataset.py  # prompts, chat template, processed export
│   │   └── schema.py           # Pydantic schema: {"intent": str}
│   ├── baseline/tfidf.py       # TF-IDF + Logistic Regression baseline
│   ├── training/
│   │   ├── train.py            # SFTTrainer + LoRA fine-tuning
│   │   ├── lora.py             # LoRA config construction
│   │   └── callbacks.py        # eval-metric history, best checkpoint
│   ├── evaluation/
│   │   ├── json_validator.py   # strict JSON extraction + schema validation
│   │   ├── metrics.py          # accuracy / F1 / valid_json_rate / exact_match
│   │   ├── evaluate.py         # generation loop over the test split
│   │   └── error_analysis.py   # errors.jsonl, confusion matrix (json + png)
│   └── inference/client.py     # OpenAI-compatible client with bounded retries
├── scripts/                    # CLI entry points (prepare/baseline/train/evaluate/merge/serve/client)
├── tests/                      # CPU-only pytest suite
├── data/                       # raw HF cache + processed JSONL (git-ignored)
├── checkpoints/                # LoRA adapters and merged models (git-ignored)
└── outputs/                    # metrics, predictions, errors, configs (git-ignored)
```

## The task

Given a raw customer-support message, the model must return **only** valid JSON matching
`{"intent": "string"}` where the string is one of the 77 Banking77 intents. Everything
else — markdown, explanations, off-vocabulary labels, malformed JSON — counts as a
failure and is recorded in `outputs/errors.jsonl`.

## Data and split policy

- Source: Hugging Face `PolyAI/banking77` (~10k train / 3k test).
- The **official test split is never modified, resampled or leaked**; it is used only at
  evaluation time.
- The validation set is split out of the training data only (10%, seed 42).
- All processed artifacts are reproducible via `scripts/prepare_data.py` and git-ignored.

## Baseline

TF-IDF (word 1–2 grams + sublinear TF) feeding a balanced multinomial logistic regression.
Run it before touching a GPU — it calibrates expectations (typical test accuracy is in the
low 90s on Banking77). Results land in `outputs/tfidf_metrics.json`.

## Fine-tuning

- **Method:** LoRA (r=16, α=32, dropout 0.05) on all attention projections of Qwen2.5-1.5B-Instruct,
  trained with TRL's `SFTTrainer` on chat-template-formatted examples.
- **Targets:** `q_proj, k_proj, v_proj, o_proj`; `task_type=CAUSAL_LM`.
- **Hyperparameters** (see `configs/banking77.yaml`): 2 epochs, lr 2e-4 cosine, effective
  batch 32 (4 × 8 grad-accum), bf16, gradient checkpointing.
- Eval metrics and the best eval loss are tracked in `checkpoints/**/training_metrics.json`.

## Evaluation

For every test example the harness records raw output, parsed prediction, error type and
exact match, then reports:

- **accuracy / macro-F1 / weighted-F1** over valid predictions
- **exact_match** — invalid or off-vocabulary output counts as wrong (end-to-end quality)
- **valid_json_rate** — fraction of responses that satisfy the schema
- **error breakdown** — `invalid_json` / `schema_violation` / `empty_output` / `unknown_label`
- **confusion matrix** over the 77 classes → `outputs/confusion_matrix.json` (+ optional PNG)

Artifacts: `outputs/metrics.json`, `outputs/predictions.jsonl`, `outputs/errors.jsonl`,
`outputs/run_config.json` (data + training + environment metadata).

## Deployment

1. Merge the adapter: `python scripts/merge_adapter.py ...`
2. Serve with vLLM (OpenAI-compatible): `./scripts/serve.sh <merged-model-path> [port]`
3. Query with validated retries: `python scripts/client.py "message"` or
   `IntentClient(base_url=...).predict("message")` from Python.

The client enforces the JSON schema on every response, truncates input to a configured
bound, applies a per-request timeout and bounded exponential-backoff retries, and raises
`ValueError` when the model never returns a valid prediction. Details in `docs/DEPLOYMENT.md`.

## Hardware notes

- Full fine-tuning of the 1.5B model assumes a ~24 GB GPU (bf16 + gradient checkpointing).
- CPU-only machines: run `configs/smoke_test.yaml` and the TF-IDF baseline; training steps
  still execute, just slowly.
- No data, checkpoints or outputs are committed; everything is reproducible from the configs.

## Documentation

- `docs/TRAINING.md` — fine-tuning in depth (hyperparameters, LoRA choices, resume, expected metrics)
- `docs/EVALUATION.md` — metrics, error taxonomy, confusion matrix
- `docs/DEPLOYMENT.md` — vLLM serving and the inference client contract

## Development

```bash
pytest                # CPU-only test suite
ruff check src tests  # lint
```

## License

MIT — see [LICENSE](LICENSE).

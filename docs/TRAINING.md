# Training guide

## Pipeline

```
scripts/prepare_data.py → scripts/train.py → scripts/evaluate.py
```

1. **Data** — `PolyAI/banking77` is loaded via `datasets`, the official train split is
   shuffled (seed 42) and split 90/10 into train/validation; the official test split is
   passed through untouched. Every example is rendered with the tokenizer's **native chat
   template**: system prompt (schema instruction), user message, and — for training only —
   an assistant turn containing exactly `{"intent": "<label>"}`.

2. **Model** — `Qwen/Qwen2.5-1.5B-Instruct` loaded in bf16 on GPU (float32 on CPU).
   The tokenizer gets `pad_token = eos_token` when missing.

3. **LoRA** — PEFT adapters (r=16, α=32, dropout 0.05, `CAUSAL_LM`) on `q_proj/k_proj/v_proj/o_proj`.
   Roughly 0.5–1% of parameters are trainable; `train.py` logs the exact count at startup.

4. **SFT** — TRL `SFTTrainer`, `packing=False`, `completion_only_loss=False` (full-sequence
   loss; the schema tokens are few, and full-sequence loss teaches the format), cosine
   schedule with 5% warmup, gradient checkpointing on.

## Hyperparameters (configs/banking77.yaml)

| Parameter | Value |
|---|---|
| Epochs | 2 |
| Learning rate | 2e-4 (cosine, 5% warmup) |
| Effective batch | 4 × 8 grad-accum = 32 |
| Max sequence length | 512 |
| Precision | bf16 |
| Eval / save every | 250 steps |

## Resume and checkpoints

Set `training.resume_from_checkpoint` in the YAML (or edit `configs/banking77.yaml`
temporarily) to a `checkpoints/checkpoint-<step>` path; `trainer.train(resume_from_checkpoint=...)`
continues from the saved optimizer state. `save_total_limit: 3` keeps disk use bounded.

## What is tracked

- `checkpoints/**/training_metrics.json` — full eval-loss history and the best eval loss
  (via `MetricLoggerCallback`).
- `outputs/run_config.json` — the complete config plus library versions, GPU name and
  training duration, written at the end of every run.

## Expected results

On full Banking77 with the 1.5B model you can expect test **accuracy in the 0.85–0.92**
range and **valid_json_rate > 0.97** after epoch 1 (the JSON format is learned within a few
hundred steps; class boundaries take longer). The TF-IDF baseline typically lands around
0.91–0.93 accuracy — the fine-tuned model's value is the *interface* (natural-language
instruction following, robustness to phrasing, served behind a strict validator), not raw
accuracy alone. Record your actual numbers in `docs/RESULTS.md`.

## CPU smoke test

`configs/smoke_test.yaml` uses Qwen2.5-0.5B-Instruct, 100 train / 25 val / 50 test examples,
1 epoch, LoRA r=8 on `q_proj/v_proj`. It runs end to end (data → training → evaluation →
error artifacts) on a CPU laptop in minutes and exists to verify plumbing, not quality.

# Results

> Fill this table in after running `scripts/run_baseline.py` and `scripts/evaluate.py`
> on a GPU machine. The values below are placeholders — replace them with your own.

| Model | Test accuracy | exact_match | valid_json_rate | macro F1 |
|---|---|---|---|---|
| TF-IDF + LogReg (baseline) | _fill in_ | — | 1.0 (deterministic) | _fill in_ |
| Qwen2.5-1.5B-Instruct + LoRA | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

## Error summary

- Top confusion pairs (from `outputs/confusion_matrix.json`): _fill in_
- Most common error type (from `outputs/errors.jsonl`): _fill in_

## Run provenance

`outputs/run_config.json` records the config, library versions, GPU and training duration
for each run. Keep one file per meaningful run.

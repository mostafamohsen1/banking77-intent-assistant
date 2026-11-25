# Evaluation guide

## Protocol

`scripts/evaluate.py` loads the official Banking77 **test split** (never seen during
training or model selection) and, for every example:

1. builds the inference prompt with the chat template (`add_generation_prompt=True`),
2. generates greedily (`temperature=0`, `max_new_tokens=64`),
3. extracts the first JSON object (`evaluation/json_validator.py`) and validates it against
   the Pydantic schema `{"intent": str}` with `extra="forbid"`,
4. rejects intents outside the 77-label vocabulary as `unknown_label`.

## Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | Classification accuracy over examples whose output parsed to a valid, in-vocabulary intent |
| `macro_f1` | Unweighted mean of per-class F1 over the 77 intents |
| `weighted_f1` | Support-weighted per-class F1 |
| `exact_match` | Strict end-to-end score: any invalid, off-vocabulary or wrong output counts as failure |
| `valid_json_rate` | Fraction of responses satisfying the schema |
| `error_breakdown` | Counts per machine-readable failure type |

`exact_match` is the headline number for a deployment decision; `accuracy` isolates the
classification ability from format failures.

## Error taxonomy

`outputs/errors.jsonl` contains every failed example with `error_type`:

- `invalid_json` — output contained no parseable JSON object
- `schema_violation` — valid JSON that does not match `{"intent": str}` (missing key, wrong
  type, extra keys)
- `empty_output` — the model returned nothing
- `unknown_label` — valid JSON with an intent outside the 77-label vocabulary
- `wrong_class` — valid, in-vocabulary, but incorrect prediction

## Confusion matrix

`outputs/confusion_matrix.json` holds the full 77×77 matrix, the per-pair confusion counts
(`confusion_pairs`, ranked) and the number of off-vocabulary outputs. The optional PNG
(`outputs/confusion_matrix.png`) plots only the labels involved in the top confusions — a
77-class full heatmap is unreadable.

## Baseline comparison

`outputs/tfidf_metrics.json` (from `scripts/run_baseline.py`) uses the same test split, so
the two metric files are directly comparable. Report both in `docs/RESULTS.md`.

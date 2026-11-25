# Data

- `raw/` — the Hugging Face `datasets` library downloads and caches Banking77 here
  (or in the shared HF cache) when `scripts/prepare_data.py` runs. Nothing here is committed.
- `processed/` — normalized train/validation/test JSONL produced by `scripts/prepare_data.py`.
  The official Banking77 **test split is never modified, resampled or leaked into training**;
  the validation set is split out of the training data only (default 10%, seed 42).

Downloaded datasets and processed artifacts are reproducible from the config and are
therefore git-ignored.

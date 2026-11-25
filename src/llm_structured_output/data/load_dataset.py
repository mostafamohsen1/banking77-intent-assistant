"""Banking77 dataset loading with a strict, leak-free split policy."""

from __future__ import annotations

import logging
from typing import Any

from datasets import ClassLabel, load_dataset

logger = logging.getLogger(__name__)


def load_banking77(config: Any) -> tuple[Any, Any, Any, list[str]]:
    """Load Banking77 and produce ``(train, validation, test, labels)``.

    The official test split is returned untouched. The validation set is carved
    out of the *training* data only, using the configured seed and ratio.
    Optional ``max_*_samples`` caps are applied after splitting (for smoke runs).
    """
    ds = load_dataset(config.hf_dataset_id)
    train_raw = ds[config.train_split]
    test_raw = ds[config.test_split]
    logger.info(
        "Loaded %s: %d train / %d test rows from official splits",
        config.hf_dataset_id,
        len(train_raw),
        len(test_raw),
    )

    split = train_raw.train_test_split(
        test_size=config.validation_ratio,
        seed=config.seed,
        shuffle=True,
    )
    train = split["train"]
    validation = split["test"]

    if config.max_train_samples:
        train = train.select(range(min(config.max_train_samples, len(train))))
    if config.max_validation_samples:
        validation = validation.select(
            range(min(config.max_validation_samples, len(validation)))
        )
    if config.max_test_samples:
        test = test_raw.select(range(min(config.max_test_samples, len(test_raw))))
    else:
        test = test_raw

    labels = extract_labels(train_raw, test_raw)
    logger.info(
        "Split sizes after preparation: train=%d validation=%d test=%d (%d labels)",
        len(train),
        len(validation),
        len(test),
        len(labels),
    )
    return train, validation, test, labels


def extract_labels(*splits: Any) -> list[str]:
    """Extract the ordered intent label list from a ClassLabel feature.

    Falls back to sorted unique text labels when the feature is not ClassLabel.
    """
    for split in splits:
        feature = split.features.get("label")
        if isinstance(feature, ClassLabel):
            return list(feature.names)
    unique = {row["label"] for split in splits for row in split}
    return sorted(unique)  # type: ignore[return-value]


def label_to_id_map(labels: list[str]) -> dict[str, int]:
    """Map intent name -> integer id, consistent with the ClassLabel ordering."""
    return {name: idx for idx, name in enumerate(labels)}

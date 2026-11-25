"""Tests for dataset splitting and label handling (no network, no GPU)."""

from __future__ import annotations

from llm_structured_output.data.load_dataset import extract_labels, label_to_id_map


def test_label_to_id_map_is_bijective(labels):
    mapping = label_to_id_map(labels)
    assert len(mapping) == len(labels)
    assert sorted(mapping.values()) == list(range(len(labels)))


def test_extract_labels_falls_back_to_sorted_unique(labels):
    dataset = [{"label": lbl} for lbl in ["b_intent", "a_intent", "b_intent"]]
    ds = _FakeDS(dataset)
    assert extract_labels(ds, ds) == ["a_intent", "b_intent"]


class _FakeDS(list):
    features = {"label": object()}

    def __init__(self, rows):
        super().__init__(rows)


def test_extract_labels_from_class_feature(labels):
    from datasets import ClassLabel

    ds = _FakeDS([{"label": 0}])
    ds.features = {"label": ClassLabel(names=labels)}
    assert extract_labels(ds) == labels

"""The loop classifier and distillation share prompt-grouped validation."""

from collections import Counter

import numpy as np

from looped_mechanisms.inputs import SOURCES, Item
from looped_mechanisms.layer_classifier import prediction_summary
from looped_mechanisms.probe_study import VALIDATION_PER_SOURCE, validation_ids


def test_validation_holds_out_whole_prompt_ids_by_dataset() -> None:
    items = [
        Item(f"{source_index}-{rank}", source.name, rank, rank, f"prompt {rank}")
        for source_index, source in enumerate(SOURCES)
        for rank in range(64)
    ]
    identifiers = validation_ids(items)
    counts = Counter(item.source for item in items if item.identifier in identifiers)
    assert counts == {source.name: VALIDATION_PER_SOURCE for source in SOURCES}
    assert identifiers == validation_ids(list(reversed(items)))
    assert len(identifiers) == 96


def test_layer_prediction_summary_uses_physical_layer_labels() -> None:
    confusion = np.eye(12, dtype=np.int64)
    confusion[0, 2] = 1
    rows = prediction_summary(confusion)

    assert rows[0] == {
        "layer": 6,
        "counts": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "examples": 2,
        "mean": 7.0,
        "variance": 1.0,
    }
    assert rows[-1]["layer"] == rows[-1]["mean"] == 17
    assert rows[-1]["variance"] == 0

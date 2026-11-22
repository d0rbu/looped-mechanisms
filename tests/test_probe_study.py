"""The loop classifier and distillation share prompt-grouped validation."""

import json
from collections import Counter
from pathlib import Path

from looped_mechanisms.design import LOCAL, NDIF, conditions
from looped_mechanisms.inputs import SOURCES, Item, render
from looped_mechanisms.probe_study import (
    SWEEP_RUNS,
    VALIDATION_PER_SOURCE,
    publish_catalog,
    validation_ids,
)
from looped_mechanisms.stream_sft import assistant_mask


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


def test_stream_sft_masks_only_assistant_content() -> None:
    text = render([("system", "System context"), ("user", "A request"), ("assistant", "An answer")])
    offsets = [(index, index + 1) for index in range(len(text))]
    included = assistant_mask(text, offsets)
    assert "".join(char for char, take in zip(text, included, strict=True) if take) == "An answer"


def test_classifier_catalog_covers_recorded_multiloop_conditions(tmp_path: Path) -> None:
    expected = {
        (backend, run) for backend in (LOCAL, NDIF) for run in conditions(backend) if run.passes > 1
    }
    assert set(SWEEP_RUNS) == expected
    assert len(SWEEP_RUNS) == 18
    scores = tmp_path / LOCAL / "variable-02"
    scores.mkdir(parents=True)
    (scores / "scores.json").write_text(json.dumps({"layers": [{}] * 12}))
    publish_catalog(tmp_path)
    runs = json.loads((tmp_path / "index.json").read_text())["runs"]
    assert [(run["id"], run["projections"]) for run in runs] == [("variable-02", False)]

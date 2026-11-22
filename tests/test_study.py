import json
from pathlib import Path

import numpy as np
import pytest

from looped_mechanisms.design import HIDDEN, LOCAL, NDIF, POSITIONS, Condition, conditions, visits
from looped_mechanisms.inputs import Item, choice_prompt
from looped_mechanisms.probe_study import load_core
from looped_mechanisms.report import build_report
from looped_mechanisms.results import complete, positions, result_dir, save_result


@pytest.mark.parametrize("passes", range(1, 17))
def test_visit_order_covers_each_physical_layer_and_pass(passes: int) -> None:
    layout = visits(Condition("variable", passes))
    assert len(layout) == 12 + 12 * passes
    assert sorted(visit.slot for visit in layout) == list(range(len(layout)))
    assert [visit.layer for visit in layout[:6]] == list(range(6))
    assert [visit.layer for visit in layout[-6:]] == list(range(18, 24))
    for pass_number in range(1, passes + 1):
        assert [visit.layer for visit in layout if visit.pass_number == pass_number] == list(
            range(6, 18)
        )


def test_matrix_and_scoring_prompt() -> None:
    assert len(conditions(LOCAL)) == 17
    assert [condition.passes for condition in conditions(NDIF)] == [3, 3]
    with pytest.raises(ValueError, match="Pass count"):
        Condition("fixed", 16)
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        Condition("unknown", 1)
    assert choice_prompt("Pick", ["a", "b", "c", "d"]).endswith("<|message_model|><|content_text|>")
    assert positions(4) == [0, 1, 2, 3]
    assert positions(33) == [8, 16, 24, 32]
    with pytest.raises(ValueError, match="only 3 tokens"):
        positions(3)


def test_committed_artifacts_feed_report(tmp_path: Path) -> None:
    condition = Condition("variable", 16)
    item = Item("arc-test", "ARC-Challenge", 0, 9, "A question", 1)
    path = result_dir(tmp_path, LOCAL, condition)
    token_ids = list(range(17))
    vectors = np.ones((204, POSITIONS, HIDDEN), dtype=np.float32)
    vectors[18, -1] = -1
    assert not complete(path, item, "test-runtime")
    save_result(path, condition, item, "test-runtime", token_ids, vectors, [0, 7, 2, 1])
    assert complete(path, item, "test-runtime")
    with pytest.raises(ValueError, match="provenance"):
        complete(path, item, "different-runtime")
    marker = json.loads((path / "arc-test.json").read_text())
    assert marker["cosine"][18] == pytest.approx(-1)
    assert marker["cosine"][6] is None

    with np.load(path / "arc-test.npz") as arrays:
        assert arrays["active_block"].dtype == np.float16
        assert arrays["active_block"].shape == (204, 4, 2304)
        assert arrays["token_ids"].tolist() == token_ids
        assert arrays["positions"].tolist() == [4, 8, 12, 16]

    report = build_report(tmp_path)
    series = report["backends"][LOCAL][-1]["sources"]["ARC-Challenge"]
    assert series["correct"] == 1
    assert series["evaluated"] == 1
    assert len(series["rms"]) == 204
    (path / "arc-test.npz").unlink()
    with pytest.raises(ValueError, match="provenance"):
        complete(path, item, "test-runtime")
    with pytest.raises(ValueError, match="Invalid activation"):
        build_report(tmp_path)


def test_reject_incomplete_or_invalid_vectors(tmp_path: Path) -> None:
    condition = Condition("variable", 1)
    item = Item("invalid", "Dolci Tool Use", 0, 9, "A trace")
    vectors = np.ones((24, POSITIONS, HIDDEN), dtype=np.float32)
    with pytest.raises(ValueError, match="invalid activations"):
        save_result(tmp_path, condition, item, "test", [1, 2, 3, 4], vectors[1:], None)
    vectors[2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid activations"):
        save_result(tmp_path, condition, item, "test", [1, 2, 3, 4], vectors, None)
    vectors[2, 0, 0] = 1
    with pytest.raises(ValueError, match="Invalid answer"):
        save_result(tmp_path, condition, item, "test", [1, 2, 3, 4], vectors, [0, 1, 2, 3])
    assert not list(tmp_path.iterdir())


def test_compact_capture_preserves_layer_vectors_and_provenance(tmp_path: Path) -> None:
    condition = Condition("variable", 16)
    item = Item("sample", "SmolTalk train", 0, 0, "A prompt")
    layout = visits(condition)
    vectors = np.broadcast_to(
        np.arange(len(layout), dtype=np.float32)[:, None, None],
        (len(layout), POSITIONS, HIDDEN),
    )
    selected = [i for i, visit in enumerate(layout) if visit.layer == 6]
    roots = (tmp_path / "full", tmp_path / "compact")
    full, compact = (result_dir(root, LOCAL, condition) for root in roots)
    save_result(full, condition, item, "test-runtime", [1, 2, 3, 4], vectors, None)
    save_result(
        compact,
        condition,
        item,
        "test-runtime",
        [1, 2, 3, 4],
        vectors[selected],
        None,
        capture_layer=6,
    )
    expected = load_core(roots[0], LOCAL, condition, [item], layer=6)
    out = np.empty_like(expected)
    result = load_core(roots[1], LOCAL, condition, [item], layer=6, capture_layer=6, out=out)
    assert result is out
    assert np.array_equal(result, expected)
    assert complete(compact, item, "test-runtime", capture_layer=6)
    with pytest.raises(ValueError, match="provenance"):
        complete(compact, item, "test-runtime")
    with pytest.raises(ValueError, match="only its recorded layer"):
        load_core(roots[1], LOCAL, condition, [item], layer=7, capture_layer=6)

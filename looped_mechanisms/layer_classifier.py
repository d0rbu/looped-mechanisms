"""Classify physical recurrent layers on held-out Inkling activations."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from looped_mechanisms.design import HIDDEN, POSITIONS, Condition, visits
from looped_mechanisms.inputs import Item, load_items
from looped_mechanisms.probe_study import RUNS, load_core, validation_ids
from looped_mechanisms.results import atomic_json

LAYERS = np.arange(6, 18)


def prediction_summary(confusion: NDArray[np.int64]) -> list[dict[str, object]]:
    if confusion.shape != (len(LAYERS), len(LAYERS)) or (confusion < 0).any():
        raise ValueError("Expected a 12-by-12 matrix of nonnegative counts")
    totals = confusion.sum(axis=1)
    if (totals == 0).any():
        raise ValueError("Every true layer must have at least one prediction")
    means = (confusion @ LAYERS) / totals
    variance = (confusion * (LAYERS - means[:, None]) ** 2).sum(axis=1) / totals
    return [
        {
            "layer": int(layer),
            "counts": counts.tolist(),
            "examples": int(count),
            "mean": float(mean),
            "variance": float(var),
        }
        for layer, counts, count, mean, var in zip(
            LAYERS, confusion, totals, means, variance, strict=True
        )
    ]


def train_condition(
    artifacts: Path,
    output: Path,
    heads: Path,
    backend: str,
    condition: Condition,
    items: list[Item],
    validation: NDArray[np.bool_],
) -> None:
    from sklearn.linear_model import RidgeClassifier  # ty: ignore[unresolved-import]

    values = load_core(artifacts, backend, condition, items)
    layout = [visit.layer for visit in visits(condition) if visit.pass_number == 1]
    if not np.array_equal(layout, LAYERS):
        raise ValueError("Activation layer order does not match the classifier labels")
    train_data = values[~validation].reshape(-1, HIDDEN).astype(np.float64)
    train_labels = np.tile(
        np.repeat(LAYERS, POSITIONS), np.count_nonzero(~validation) * condition.passes
    )
    head = RidgeClassifier(alpha=10.0, copy_X=False).fit(train_data, train_labels)
    del train_data
    if head.coef_.shape != (len(LAYERS), HIDDEN) or not np.array_equal(head.classes_, LAYERS):
        raise ValueError("Unexpected classifier head or label order")

    val_labels = np.tile(
        np.repeat(LAYERS, POSITIONS), np.count_nonzero(validation) * condition.passes
    )
    predictions = head.predict(values[validation].reshape(-1, HIDDEN).astype(np.float64))
    confusion = np.zeros((len(LAYERS), len(LAYERS)), dtype=np.int64)
    np.add.at(confusion, (val_labels - LAYERS[0], predictions - LAYERS[0]), 1)
    heads.mkdir(parents=True, exist_ok=True)
    np.savez(
        heads / f"{backend}-{condition.name}-recurrent-layer.npz",
        weights=head.coef_,
        bias=head.intercept_,
        labels=LAYERS,
    )
    directory = output / backend / condition.name
    directory.mkdir(parents=True, exist_ok=True)
    accuracy = float(np.trace(confusion) / confusion.sum())
    atomic_json(
        directory / "layer-predictions.json",
        {
            "labels": LAYERS.tolist(),
            "accuracy": accuracy,
            "validation_prompts": int(np.count_nonzero(validation)),
            "positions": POSITIONS,
            "passes": condition.passes,
            "rows": prediction_summary(confusion),
        },
    )
    print(
        f"{backend}/{condition.name}: recurrent-layer validation accuracy {accuracy:.3%}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("site/loop-study/v1"))
    parser.add_argument("--heads", type=Path, default=Path("artifacts/loop-heads"))
    args = parser.parse_args()
    items = load_items(args.cohort)
    held_out = validation_ids(items)
    validation = np.array([item.identifier in held_out for item in items])
    for backend, condition in RUNS:
        train_condition(
            args.artifacts, args.output, args.heads, backend, condition, items, validation
        )


if __name__ == "__main__":
    main()

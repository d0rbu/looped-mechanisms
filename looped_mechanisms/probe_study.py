"""Fit held-out loop classifiers and build loop-colored projections."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from looped_mechanisms.design import HIDDEN, LOCAL, NDIF, POSITIONS, Condition, visits
from looped_mechanisms.inputs import SEED, SOURCES, Item, load_items
from looped_mechanisms.results import atomic_json, complete, result_dir

RUNS = (
    (NDIF, Condition("fixed", 3)),
    (LOCAL, Condition("variable", 16)),
    (NDIF, Condition("variable", 3)),
    (LOCAL, Condition("variable", 3)),
)
METHODS = ("random", "pca", "tsne", "umap", "lda")
VALIDATION_PER_SOURCE = 16


def validation_ids(items: list[Item]) -> set[str]:
    rng = np.random.default_rng(SEED)
    held_out: set[str] = set()
    for source in SOURCES:
        identifiers = sorted(item.identifier for item in items if item.source == source.name)
        if len(identifiers) <= VALIDATION_PER_SOURCE:
            raise ValueError(f"Incomplete cohort for {source.name}")
        held_out.update(rng.choice(identifiers, VALIDATION_PER_SOURCE, replace=False).tolist())
    return held_out


def load_core(
    root: Path, backend: str, condition: Condition, items: list[Item]
) -> NDArray[np.float16]:
    folder = result_dir(root, backend, condition)
    layout = visits(condition)
    core = [index for index, visit in enumerate(layout) if visit.pass_number is not None]
    values = np.empty((len(items), condition.passes, 12, POSITIONS, HIDDEN), dtype=np.float16)

    runtime = json.loads((folder / f"{items[0].identifier}.json").read_text())["runtime"]
    for index, item in enumerate(items):
        if not complete(folder, item, runtime):
            raise ValueError(f"Missing activation for {item.identifier} in {folder}")
        with np.load(folder / f"{item.identifier}.npz") as saved:
            values[index] = saved["active_block"][core].reshape(
                condition.passes, 12, POSITIONS, HIDDEN
            )
    return values


def run(
    root: Path,
    output: Path,
    heads: Path,
    backend: str,
    condition: Condition,
    items: list[Item],
) -> None:
    from sklearn.decomposition import PCA  # ty: ignore[unresolved-import]
    from sklearn.discriminant_analysis import (  # ty: ignore[unresolved-import]
        LinearDiscriminantAnalysis,
    )
    from sklearn.linear_model import RidgeClassifier  # ty: ignore[unresolved-import]
    from sklearn.manifold import TSNE  # ty: ignore[unresolved-import]
    from sklearn.metrics import confusion_matrix  # ty: ignore[unresolved-import]
    from umap import UMAP  # ty: ignore[unresolved-import]

    directory = output / backend / condition.name
    directory.mkdir(parents=True, exist_ok=True)
    heads.mkdir(parents=True, exist_ok=True)
    values = load_core(root, backend, condition, items)
    held_out = validation_ids(items)
    validation = np.array([item.identifier in held_out for item in items])
    val_indices = np.flatnonzero(validation)
    labels = np.broadcast_to(
        np.arange(condition.passes)[None, :, None], (len(items), condition.passes, POSITIONS)
    )
    train_y = labels[~validation].ravel()
    val_y = labels[validation].ravel()
    rng = np.random.default_rng(SEED)
    directions = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))[0].astype(np.float32)
    scores = []
    for offset in range(12):
        layer = offset + 6
        train_x = values[~validation, :, offset].reshape(-1, HIDDEN).astype(np.float32)
        val_x = values[validation, :, offset].reshape(-1, HIDDEN).astype(np.float32)
        head = RidgeClassifier(alpha=10.0).fit(train_x.astype(np.float64), train_y)
        expected = (condition.passes, HIDDEN)
        if head.coef_.shape != expected:
            raise ValueError(f"Expected a {expected} loop head, got {head.coef_.shape}")
        predicted = head.predict(val_x)
        matrix = confusion_matrix(val_y, predicted, labels=np.arange(condition.passes))
        scores.append(
            {
                "layer": layer,
                "accuracy": float(np.mean(predicted == val_y)),
                "correct": int(np.count_nonzero(predicted == val_y)),
                "examples": len(val_y),
                "confusion": matrix.tolist(),
            }
        )
        np.savez(
            heads / f"{backend}-{condition.name}-{layer:02d}.npz",
            weights=head.coef_,
            bias=head.intercept_,
        )

        pca = PCA(n_components=50, svd_solver="randomized", random_state=SEED)
        low_train = pca.fit_transform(train_x)
        plot_x = values[validation, :, offset, -1].reshape(-1, HIDDEN).astype(np.float32)
        low_plot = pca.transform(plot_x)
        lda = LinearDiscriminantAnalysis(n_components=2)
        methods = {
            "random": (plot_x - pca.mean_) @ directions,
            "pca": low_plot[:, :2],
            "tsne": TSNE(
                n_components=2,
                perplexity=30,
                max_iter=550,
                init="pca",
                random_state=SEED,
            ).fit_transform(low_plot),
            "umap": UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=SEED,
                n_jobs=1,
                compatibility_layout=True,
            ).fit_transform(low_plot),
            "lda": lda.fit(low_train, train_y).transform(low_plot),
        }
        atomic_json(
            directory / f"layer-{layer:02d}.json",
            {
                "layer": layer,
                "methods": {name: np.round(points, 5).tolist() for name, points in methods.items()},
                "items": [
                    {"id": items[index].identifier, "source": items[index].source}
                    for index in val_indices
                ],
                "loops": condition.passes,
            },
        )
        atomic_json(
            directory / "scores.json",
            {
                "head_dim": condition.passes,
                "train_prompts": int(np.count_nonzero(~validation)),
                "validation_prompts": int(np.count_nonzero(validation)),
                "positions": POSITIONS,
                "layers": scores,
            },
        )
        print(f"{backend}/{condition.name} layer {layer}: {scores[-1]['accuracy']:.3%}", flush=True)


def publish_catalog(output: Path) -> None:
    available = []
    for backend, condition in RUNS:
        directory = output / backend / condition.name
        scores_path = directory / "scores.json"
        if scores_path.is_file() and len(json.loads(scores_path.read_text())["layers"]) == 12:
            available.append({"backend": backend, "id": condition.name, **asdict(condition)})
    atomic_json(
        output / "index.json",
        {
            "runs": available,
            "methods": METHODS,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("site/loop-study/v1"))
    parser.add_argument("--heads", type=Path, default=Path("artifacts/loop-heads"))
    runs = {f"{backend}/{condition.name}": (backend, condition) for backend, condition in RUNS}
    parser.add_argument("--run", choices=runs)
    args = parser.parse_args()
    items = load_items(args.cohort)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [runs[args.run]] if args.run else RUNS
    for backend, condition in selected:
        run(args.artifacts, args.output, args.heads, backend, condition, items)
    publish_catalog(args.output)


if __name__ == "__main__":
    main()

"""Fit held-out loop classifiers and build loop-colored projections."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from looped_mechanisms.design import HIDDEN, LOCAL, NDIF, POSITIONS, Condition, visits
from looped_mechanisms.inputs import SEED, SOURCES, Item, load_items
from looped_mechanisms.loop_classifier import BASE_WIDTH, EPOCHS, fit, parameter_count
from looped_mechanisms.results import atomic_json, complete, result_dir

RUNS = (
    (NDIF, Condition("fixed", 3)),
    (LOCAL, Condition("variable", 16)),
    (NDIF, Condition("variable", 3)),
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
    root: Path,
    backend: str,
    condition: Condition,
    items: list[Item],
    *,
    layer: int | None = None,
    capture_layer: int | None = None,
    out: NDArray[np.float16] | None = None,
) -> NDArray[np.float16]:
    folder = result_dir(root, backend, condition)
    layout = visits(condition)
    if capture_layer is not None:
        if capture_layer != layer:
            raise ValueError("A compact capture can supply only its recorded layer")
        layout = tuple(
            visit
            for visit in layout
            if visit.layer == capture_layer and visit.pass_number is not None
        )
    core = [
        index
        for index, visit in enumerate(layout)
        if visit.pass_number is not None and (layer is None or visit.layer == layer)
    ]
    if not core:
        raise ValueError(f"Physical layer {layer} does not recur")
    layers = len(core) // condition.passes
    shape = (len(items), condition.passes, layers, POSITIONS, HIDDEN)
    values = np.empty(shape, dtype=np.float16) if out is None else out
    if values.shape != shape or values.dtype != np.float16:
        raise ValueError(f"Expected capture destination of shape {shape} and dtype float16")

    runtime = json.loads((folder / f"{items[0].identifier}.json").read_text())["runtime"]
    for index, item in enumerate(items):
        if not complete(folder, item, runtime, capture_layer=capture_layer):
            raise ValueError(f"Missing activation for {item.identifier} in {folder}")
        with np.load(folder / f"{item.identifier}.npz") as saved:
            values[index] = saved["active_block"][core].reshape(
                condition.passes, layers, POSITIONS, HIDDEN
            )
    return values


def run(
    root: Path,
    output: Path,
    heads: Path,
    backend: str,
    condition: Condition,
    items: list[Item],
    *,
    device_name: str = "cpu",
    classifiers_only: bool = False,
    extra_items: tuple[Item, ...] = (),
    extra_root: Path | None = None,
    extra_sha256: str | None = None,
) -> None:
    directory = output / backend / condition.name
    directory.mkdir(parents=True, exist_ok=True)
    held_out = validation_ids(items)
    if extra_items:
        if extra_root is None or extra_sha256 is None:
            raise ValueError("Extended probes require a capture root and cohort digest")
        if len({item.identifier for item in [*items, *extra_items]}) != len(items) + len(
            extra_items
        ):
            raise ValueError("Expanded sample overlaps the base prompt IDs")
        if len({item.text for item in [*items, *extra_items]}) != len(items) + len(extra_items):
            raise ValueError("Expanded sample has duplicate prompts")
        shape = (len(items) + len(extra_items), condition.passes, 12, POSITIONS, HIDDEN)
        values = np.empty(shape, dtype=np.float16)
        load_core(root, backend, condition, items, out=values[: len(items)])
        load_core(extra_root, backend, condition, list(extra_items), out=values[len(items) :])
        items = [*items, *extra_items]
    else:
        values = load_core(root, backend, condition, items)
    validation = np.array([item.identifier in held_out for item in items])
    training_count = int(np.count_nonzero(~validation))
    checkpoint_suffix = f"-prompts-{training_count}" if extra_items else ""
    scores = []
    for offset in range(12):
        layer = offset + 6
        score = fit(
            values[~validation, :, offset],
            values[validation, :, offset],
            layer=layer,
            width=BASE_WIDTH,
            batch_size=256,
            device_name=device_name,
            checkpoint=heads
            / f"{backend}-{condition.name}-{layer:02d}-width-{BASE_WIDTH}{checkpoint_suffix}.pt",
            lineage={
                "validation_ids": sorted(held_out),
                "train_prompts": training_count,
                "extra_cohort_sha256": extra_sha256,
            },
        )
        scores.append(score)
        print(f"{backend}/{condition.name} layer {layer}: {score['accuracy']:.3%}", flush=True)
    atomic_json(
        directory / "scores.json",
        {
            "head_dim": condition.passes,
            "hidden_units": BASE_WIDTH,
            "parameters_per_classifier": parameter_count(BASE_WIDTH, condition.passes),
            "epochs": EPOCHS,
            "train_prompts": training_count,
            "extra_cohort_sha256": extra_sha256,
            "validation_prompts": int(np.count_nonzero(validation)),
            "positions": POSITIONS,
            "layers": scores,
        },
    )
    if not classifiers_only:
        projections(directory, condition, items, values, validation)


def projections(
    directory: Path,
    condition: Condition,
    items: list[Item],
    values: NDArray[np.float16],
    validation: NDArray[np.bool_],
) -> None:
    from sklearn.decomposition import PCA  # ty: ignore[unresolved-import]
    from sklearn.discriminant_analysis import (  # ty: ignore[unresolved-import]
        LinearDiscriminantAnalysis,
    )
    from sklearn.manifold import TSNE  # ty: ignore[unresolved-import]
    from umap import UMAP  # ty: ignore[unresolved-import]

    val_indices = np.flatnonzero(validation)
    train_y = np.tile(
        np.repeat(np.arange(condition.passes), POSITIONS), np.count_nonzero(~validation)
    )
    rng = np.random.default_rng(SEED)
    directions = np.linalg.qr(rng.standard_normal((HIDDEN, 2)))[0].astype(np.float32)
    for offset in range(12):
        layer = offset + 6
        train_x = values[~validation, :, offset].reshape(-1, HIDDEN).astype(np.float32)
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
    parser.add_argument("--extra-cohort", type=Path)
    parser.add_argument("--extra-artifacts", type=Path)
    runs = {
        f"{backend}/{condition.name}": (backend, condition)
        for backend, condition in (*RUNS, (LOCAL, Condition("variable", 3)))
    }
    parser.add_argument("--run", choices=runs, action="append")
    parser.add_argument("--classifiers-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    items = load_items(args.cohort)
    extra_items = ()
    extra_sha256 = None
    if args.extra_cohort is not None:
        raw = args.extra_cohort.read_bytes()
        extra_items = tuple(Item(**row) for row in json.loads(raw))
        extra_sha256 = hashlib.sha256(raw).hexdigest()
        if (
            args.extra_artifacts is None
            or len(extra_items) + len(items) - len(validation_ids(items)) != 2_880
        ):
            parser.error(
                "An expanded baseline requires --extra-artifacts and exactly 2,880 training prompts"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [runs[name] for name in args.run] if args.run else RUNS
    for backend, condition in selected:
        run(
            args.artifacts,
            args.output,
            args.heads,
            backend,
            condition,
            items,
            device_name=args.device,
            classifiers_only=args.classifiers_only,
            extra_items=extra_items,
            extra_root=args.extra_artifacts,
            extra_sha256=extra_sha256,
        )
    publish_catalog(args.output)


if __name__ == "__main__":
    main()

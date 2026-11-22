"""Build Gaussianity summaries and cross-loop neighbors from saved activations."""

import argparse
import hashlib
import importlib
import json
from codecs import getincrementaldecoder
from collections.abc import Iterator
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from looped_mechanisms.design import (
    HIDDEN,
    ITEMS_PER_SOURCE,
    LOCAL,
    NDIF,
    POSITIONS,
    Condition,
    visits,
)
from looped_mechanisms.design import conditions as study_conditions
from looped_mechanisms.inputs import SOURCES, Item, load_items
from looped_mechanisms.results import atomic_json, complete, result_dir

NEIGHBORS = 3
PROJECTIONS = 32
TOKENIZER_SHA256 = "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"
RECORD = np.dtype([("index", "<u4"), ("distance", "<f4")])


def token_decoder(path: Path) -> tuple[dict[int, bytes], dict[int, str]]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Expected the pinned tokenizer shared by both Inkling checkpoints")
    tokenizer = json.loads(payload)
    # ByteLevel BPE maps printable bytes to themselves and gives the rest
    # consecutive Unicode code points starting at 256.
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    other = [byte for byte in range(256) if byte not in printable]
    alphabet = {chr(byte): byte for byte in printable}
    alphabet.update({chr(256 + i): byte for i, byte in enumerate(other)})
    vocab = {
        index: bytes(alphabet[char] for char in token)
        for token, index in tokenizer["model"]["vocab"].items()
    }
    added = {token["id"]: token["content"] for token in tokenizer["added_tokens"]}
    return vocab, added


def decode_tokens(ids: list[int], vocab: dict[int, bytes], added: dict[int, str]) -> list[str]:
    decoder = getincrementaldecoder("utf-8")(errors="replace")
    pieces = []
    for index in ids:
        if index in added:
            pieces.append(decoder.decode(b"", final=True) + added[index])
            decoder.reset()
        else:
            pieces.append(decoder.decode(vocab[index]))
    pieces[-1] += decoder.decode(b"", final=True)
    return pieces


def load_condition(
    root: Path, backend: str, condition: Condition, items: list[Item]
) -> tuple[NDArray[np.float16], list[dict[str, Any]]]:
    directory = result_dir(root, backend, condition)
    vectors = np.empty((len(visits(condition)), len(items) * POSITIONS, HIDDEN), dtype=np.float16)
    examples = []
    runtime = json.loads((directory / f"{items[0].identifier}.json").read_text())["runtime"]
    for i, item in enumerate(items):
        if not complete(directory, item, runtime):
            raise ValueError(f"Missing artifact for {condition.name}/{item.identifier}")
        with np.load(directory / f"{item.identifier}.npz") as saved:
            vectors[:, i * POSITIONS : (i + 1) * POSITIONS] = saved["active_block"]
            examples.append(
                {
                    "id": item.identifier,
                    "source": item.source,
                    "row": item.row,
                    "positions": saved["positions"].tolist(),
                    "ids": saved["token_ids"].tolist(),
                }
            )
    return vectors, examples


def normal_scores(samples: NDArray[Any], quantiles: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = samples - samples.mean(axis=0)
    scale = centered.std(axis=0)
    variable = centered[:, scale > 0] / scale[scale > 0]
    ordered = np.sort(variable, axis=0)
    return np.sqrt(np.mean((ordered - quantiles[:, None]) ** 2, axis=0))


def gaussianity(
    matrix: NDArray[np.float32], directions: NDArray[np.float32], quantiles: NDArray[np.float64]
) -> dict[str, float]:
    scores = normal_scores(matrix @ directions, quantiles)
    centered = matrix - matrix.mean(axis=0)
    squared = centered**2
    variance = squared.mean(axis=0)
    usable = variance > 0
    fourth = np.einsum("ij,ij->j", squared, squared) / len(matrix)
    excess = np.abs(fourth[usable] / variance[usable] ** 2 - 3)
    return {
        "qq_rmse": round(float(np.mean(scores)), 6),
        "median_abs_excess_kurtosis": round(float(np.median(excess)), 6),
    }


def top_neighbors(xp: ModuleType, squared: NDArray[Any]) -> NDArray[Any]:
    result = np.empty((squared.shape[0], NEIGHBORS), dtype=RECORD)
    rows = xp.arange(squared.shape[0])
    to_host = np.asarray if xp is np else xp.asnumpy
    for rank in range(NEIGHBORS):
        index = xp.argmin(squared, axis=1)
        best = xp.sqrt(xp.maximum(squared[rows, index], 0) / HIDDEN)
        result["index"][:, rank] = to_host(index)
        result["distance"][:, rank] = to_host(best)
        squared[rows, index] = xp.inf
    return result


def squared_distances(
    a: NDArray[Any],
    b: NDArray[Any],
    norms_a: NDArray[Any],
    norms_b: NDArray[Any],
    excluded: NDArray[Any],
) -> NDArray[Any]:
    squared = a @ b.T
    squared *= -2
    squared += norms_a[:, None]
    squared += norms_b[None, :]
    squared[excluded] = np.inf
    return squared


def distance_references(
    xp: ModuleType, states: NDArray[Any], excluded: NDArray[Any], cross_nearest: NDArray[Any]
) -> tuple[list[float], list[list[float]]]:
    passes, count, hidden = states.shape
    to_host = np.asarray if xp is np else xp.asnumpy
    nearest = np.array(cross_nearest, copy=True)
    within = []
    for loop, values in enumerate(states):
        norms = xp.sum(values**2, axis=1)
        squared = squared_distances(values, values, norms, norms, excluded)
        distances = xp.sqrt(xp.maximum(xp.min(squared, axis=1), 0) / HIDDEN)
        nearest[loop, loop] = to_host(distances)
        within.append(round(float(nearest[loop, loop].mean(dtype=np.float64)), 6))

    # Each prompt contributes the same four query positions to each loop.
    grouped = states.reshape(passes, -1, POSITIONS, hidden).transpose(1, 0, 2, 3)
    grouped = grouped.reshape(-1, passes * POSITIONS, hidden)
    norms = xp.sum(grouped**2, axis=2)
    squared = grouped @ grouped.transpose(0, 2, 1)
    squared *= -2
    squared += norms[:, :, None]
    squared += norms[:, None, :]
    self_index = xp.arange(passes * POSITIONS)
    squared[:, self_index, self_index] = xp.inf
    squared = squared.reshape(-1, passes, POSITIONS, passes, POSITIONS).min(axis=-1)
    same = xp.sqrt(xp.maximum(squared, 0) / HIDDEN)
    same = same.transpose(1, 3, 0, 2).reshape(passes, passes, count)
    inclusive = np.minimum(nearest, to_host(same)).mean(axis=2, dtype=np.float64)
    return within, np.round(inclusive, 6).tolist()


def saved_cross_neighbors(
    directory: Path, layer: str, passes: int, count: int
) -> NDArray[np.float32]:
    nearest = np.zeros((passes, passes, count), dtype=np.float32)
    if passes > 1:
        for source in range(passes):
            targets = [target for target in range(passes) if target != source]
            records = np.fromfile(directory / f"{layer}-{source}.bin", dtype=RECORD)
            nearest[source, targets] = records.reshape(passes - 1, count, NEIGHBORS)["distance"][
                :, :, 0
            ]
    return nearest


def core_layer_states(
    xp: ModuleType, vectors: NDArray[np.float16], condition: Condition
) -> Iterator[tuple[str, NDArray[Any]]]:
    layout = visits(condition)
    for layer in range(6, 18):
        indices = [i for i, visit in enumerate(layout) if visit.layer == layer]
        yield str(layer), xp.asarray(vectors[indices], dtype=xp.float32)


def same_prompt(xp: ModuleType, count: int) -> NDArray[Any]:
    samples = xp.arange(count) // POSITIONS
    return samples[:, None] == samples[None, :]


def layer_neighbors(
    xp: ModuleType, states: NDArray[Any], excluded: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    passes, count, _ = states.shape
    results = np.empty((passes, passes, count, NEIGHBORS), dtype=RECORD)
    means = np.full((passes, passes), np.nan)
    squared_norms = xp.sum(states**2, axis=2)
    for source, target in combinations(range(passes), 2):
        squared = squared_distances(
            states[source], states[target], squared_norms[source], squared_norms[target], excluded
        )
        results[target, source] = top_neighbors(xp, squared.T.copy())
        results[source, target] = top_neighbors(xp, squared)
        for a, b in ((source, target), (target, source)):
            means[a, b] = results[a, b]["distance"][:, 0].mean(dtype=np.float64)
    return results, means


def build_condition(
    xp: ModuleType,
    vectors: NDArray[np.float16],
    condition: Condition,
    examples: list[dict[str, Any]],
    directory: Path,
    vocabulary: tuple[dict[int, bytes], dict[int, str]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261121)
    directions = rng.standard_normal((HIDDEN, PROJECTIONS)).astype(np.float32)
    count = len(examples) * POSITIONS
    normal = NormalDist()
    quantiles = np.array([normal.inv_cdf((i + 0.5) / count) for i in range(count)])
    baseline = float(np.mean(normal_scores(rng.standard_normal((count, 128)), quantiles)))
    layout = visits(condition)
    statistics = [
        {
            "layer": visit.layer,
            "loop": None if visit.pass_number is None else visit.pass_number - 1,
            **gaussianity(values.astype(np.float32), directions, quantiles),
        }
        for visit, values in zip(layout, vectors, strict=True)
    ]
    means = {}
    within = {}
    inclusive = {}
    excluded = same_prompt(xp, count)
    for layer, states in core_layer_states(xp, vectors, condition):
        neighbors, distances = layer_neighbors(xp, states, excluded)
        within[layer], inclusive[layer] = distance_references(
            xp, states, excluded, neighbors["distance"][:, :, :, 0]
        )
        if condition.passes == 1:
            continue
        for source in range(condition.passes):
            targets = [target for target in range(condition.passes) if source != target]
            path = directory / f"{layer}-{source}.bin"
            temp = path.with_suffix(".tmp")
            neighbors[source, targets].tofile(temp)
            temp.replace(path)
        means[layer] = [
            [
                None if source == target else round(float(value), 6)
                for target, value in enumerate(row)
            ]
            for source, row in enumerate(distances)
        ]
    for example in examples:
        example["tokens"] = decode_tokens(example.pop("ids"), *vocabulary)
    atomic_json(directory / "examples.json", examples)
    atomic_json(
        directory / "summary.json",
        {
            "count": count,
            "positions": POSITIONS,
            "neighbors": NEIGHBORS,
            "qq_reference": round(baseline, 6),
            "gaussianity": statistics,
            "mean_distances": means,
            "within_loop_distances": within,
            "including_same_prompt": inclusive,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--output", type=Path, default=Path("site/geometry/v1"))
    parser.add_argument("--cuda", action="store_true", help="Use optional CuPy for neighbor search")
    args = parser.parse_args()
    xp = importlib.import_module("cupy") if args.cuda else np
    vocab = token_decoder(args.tokenizer)
    items = load_items(args.cohort)
    if len(items) != len(SOURCES) * ITEMS_PER_SOURCE:
        raise ValueError("Geometry requires the complete frozen cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    catalog_path = args.output / "index.json"
    catalog = (
        json.loads(catalog_path.read_text())
        if catalog_path.exists()
        else {
            "conditions": [],
            "sources": [asdict(source) for source in SOURCES],
        }
    )
    order = [
        (LOCAL, Condition("variable", 16)),
        *((NDIF, condition) for condition in study_conditions(NDIF)),
        *((LOCAL, condition) for condition in study_conditions(LOCAL) if condition.passes != 16),
    ]
    for backend, condition in order:
        entry = {"backend": backend, "id": condition.name, **asdict(condition)}
        directory = args.output / backend / condition.name
        summary_path = directory / "summary.json"
        if entry in catalog["conditions"] and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if "including_same_prompt" not in summary:
                vectors, _ = load_condition(args.artifacts, backend, condition, items)
                if vectors.shape[1] != summary["count"]:
                    raise ValueError(
                        f"Geometry cohort size mismatch for {backend}/{condition.name}"
                    )
                excluded = same_prompt(xp, vectors.shape[1])
                within = {}
                inclusive = {}
                for layer, states in core_layer_states(xp, vectors, condition):
                    cross_nearest = saved_cross_neighbors(
                        directory, layer, condition.passes, vectors.shape[1]
                    )
                    within[layer], inclusive[layer] = distance_references(
                        xp, states, excluded, cross_nearest
                    )
                summary["within_loop_distances"] = within
                summary["including_same_prompt"] = inclusive
                atomic_json(summary_path, summary)
                del vectors
                print(
                    f"{backend}/{condition.name}: added diagonal and same-prompt comparisons",
                    flush=True,
                )
            else:
                print(f"{backend}/{condition.name}: already built", flush=True)
            continue
        vectors, examples = load_condition(args.artifacts, backend, condition, items)
        build_condition(xp, vectors, condition, examples, directory, vocab)
        del vectors
        catalog["conditions"].append(entry)
        atomic_json(catalog_path, catalog)
        print(f"{backend}/{condition.name}: published", flush=True)


if __name__ == "__main__":
    main()

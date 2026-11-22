"""Validate each forward and publish an atomic per-item completion marker."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from looped_mechanisms.design import HIDDEN, POSITIONS, Condition, visits
from looped_mechanisms.inputs import Item


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def result_dir(root: Path, backend: str, condition: Condition) -> Path:
    return root / backend / condition.name


def complete(directory: Path, item: Item, runtime: str) -> bool:
    marker = directory / f"{item.identifier}.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    vector_file = marker.with_suffix(".npz")
    if (
        record["input_sha256"] != item.digest
        or record["runtime"] != runtime
        or not vector_file.exists()
        or vector_file.stat().st_size != record["bytes"]
    ):
        raise ValueError(f"Result provenance or storage mismatch in {marker}")
    return True


def positions(length: int) -> list[int]:
    if length < POSITIONS:
        raise ValueError(f"Input has only {length} tokens")
    return [(length - 1) * i // POSITIONS for i in range(1, POSITIONS)] + [length - 1]


def save_result(
    directory: Path,
    condition: Condition,
    item: Item,
    runtime: str,
    token_ids: list[int],
    active: NDArray[Any],
    logits: list[float] | None,
) -> None:
    layout = visits(condition)
    expected = (len(layout), POSITIONS, HIDDEN)
    if active.shape != expected or not np.isfinite(active).all():
        raise ValueError(
            f"{item.identifier}: invalid activations {active.shape}, expected {expected}"
        )
    stored = active.astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError("Activation exceeds float16 storage range")
    if (logits is None) != (item.answer is None) or (
        logits is not None and (len(logits) != 4 or not np.isfinite(logits).all())
    ):
        raise ValueError("Invalid answer-choice logits")

    last = stored[:, -1, :].astype(np.float32)
    rms = np.sqrt(np.mean(last**2, axis=1))
    cosine: list[float | None] = [None] * len(layout)
    for index, visit in enumerate(layout):
        if visit.pass_number is not None and visit.pass_number > 1:
            prev = index - 12
            norm_product = float(np.linalg.norm(last[prev]) * np.linalg.norm(last[index]))
            if norm_product > 0:
                cosine[index] = float(
                    np.clip(np.dot(last[prev], last[index]) / norm_product, -1, 1)
                )

    directory.mkdir(parents=True, exist_ok=True)
    vectors = directory / f"{item.identifier}.npz"
    temp = directory / f"{item.identifier}.tmp.npz"
    np.savez(
        temp,
        active_block=stored,
        token_ids=np.asarray(token_ids),
        positions=np.asarray(positions(len(token_ids))),
    )
    temp.replace(vectors)
    prediction = int(np.argmax(logits)) if logits is not None else None
    atomic_json(
        directory / f"{item.identifier}.json",
        {
            "id": item.identifier,
            "source": item.source,
            "rank": item.rank,
            "input_sha256": item.digest,
            "runtime": runtime,
            "bytes": vectors.stat().st_size,
            "tokens": len(token_ids),
            "visits": len(layout),
            "prediction": prediction,
            "answer": item.answer,
            "choice_logits": logits,
            "rms": rms.tolist(),
            "cosine": cosine,
        },
    )

"""Run matched fixed and variable-loop conditions using llama.cpp and GGUF."""

import argparse
import ctypes as c
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from looped_mechanisms.design import HIDDEN, LABELS, LOCAL, POSITIONS, Condition, conditions, visits
from looped_mechanisms.inputs import SOURCES, Item, load_items
from looped_mechanisms.results import complete, result_dir, save_result

if TYPE_CHECKING:
    from collections.abc import Iterator
    from ctypes import Array

LLAMA_REVISION = "225a5345bbb0e0c70859af5441ba922beb666fcf"
MODELS = {
    "fixed": ("Inkling-2-Mini-IQ4_NL.gguf", "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9"),
    "variable": (
        "TML-d24-varloop-research-IQ4_NL.gguf",
        "7af7f5e17b57395455bd8a37d8cfefa98824ca6e",
    ),
}


def load_library(path: Path) -> c.CDLL:
    capture = c.CDLL(str(path.resolve()))
    capture.lm_init.argtypes = []
    capture.lm_init.restype = None
    capture.lm_open.argtypes = [c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int]
    capture.lm_open.restype = c.c_void_p
    capture.lm_close.argtypes = [c.c_void_p]
    capture.lm_close.restype = None
    capture.lm_tokenize.argtypes = [c.c_void_p, c.c_char_p, c.c_int, c.POINTER(c.c_int32), c.c_int]
    capture.lm_tokenize.restype = c.c_int
    capture.lm_forward.argtypes = [
        c.c_void_p,
        c.POINTER(c.c_int32),
        c.c_int,
        c.POINTER(c.c_int32),
        c.POINTER(c.c_float),
        c.POINTER(c.c_float),
        c.c_char_p,
        c.c_int,
    ]
    capture.lm_forward.restype = c.c_int
    capture.lm_init()
    return capture


@contextmanager
def load_model(lib: c.CDLL, path: Path, passes: int, threads: int) -> Iterator[int]:
    error = c.create_string_buffer(512)
    handle = lib.lm_open(str(path).encode(), passes, threads, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode())
    try:
        yield int(handle)
    finally:
        lib.lm_close(handle)


def tokenize(lib: c.CDLL, handle: int, text: str) -> tuple[Array[c.c_int32], int]:
    data = text.encode()
    buffer = (c.c_int32 * 1024)()
    length = int(lib.lm_tokenize(handle, data, len(data), buffer, len(buffer)))
    if length <= 0:
        raise ValueError(f"Tokenization failed with length {length}")
    return buffer, length


def run_condition(
    lib: c.CDLL,
    condition: Condition,
    items: list[Item],
    root: Path,
    model_dir: Path,
    threads: int,
) -> None:
    filename, revision = MODELS[condition.checkpoint]
    runtime = f"llama.cpp:{LLAMA_REVISION}:ggml-org:{filename}:{revision}"
    directory = result_dir(root, LOCAL, condition)
    todo = [item for item in items if not complete(directory, item, runtime)]
    if not todo:
        return
    with load_model(lib, model_dir / filename, condition.passes, threads) as handle:
        labels = []
        for letter in LABELS:
            label, length = tokenize(lib, handle, letter)
            if length != 1:
                raise ValueError(f"Choice label {letter} must be one token")
            labels.append(label[0])
        choices = (c.c_int32 * len(labels))(*labels)
        layout = visits(condition)
        order = [visit.slot for visit in layout]
        array = np.empty((len(layout), POSITIONS, HIDDEN), dtype=np.float32)
        pointer = array.ctypes.data_as(c.POINTER(c.c_float))
        error = c.create_string_buffer(512)
        logits = (c.c_float * len(labels))()
        for item in todo:
            ids, length = tokenize(lib, handle, item.text)
            score = item.answer is not None
            status = lib.lm_forward(
                handle, ids, length, choices if score else None, logits, pointer, error, len(error)
            )
            if status:
                raise RuntimeError(f"{item.identifier}: {error.value.decode()}")
            save_result(
                directory,
                condition,
                item,
                runtime,
                list(ids[:length]),
                array[order],
                list(logits) if score else None,
            )
            print(
                f"{condition.name} {item.source} rank={item.rank + 1} verified_visits={len(layout)}",
                flush=True,
            )


def verify_models(model_dir: Path, selected: list[Condition]) -> None:
    expected = {
        "fixed": "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4",
        "variable": "e9f70a167e54794bebce2d456b2118730b74fa7c35e23cb1abd4378f66b17bb2",
    }
    for model in sorted({condition.checkpoint for condition in selected}):
        path = model_dir / MODELS[model][0]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[model]:
            raise ValueError(f"SHA-256 mismatch for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--library", type=Path, default=Path("build/libloop_capture.so"))
    parser.add_argument("--per-source", type=int, choices=range(1, 65), default=64)
    parser.add_argument("--source", choices=[source.name for source in SOURCES], action="append")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--condition", choices=[condition.name for condition in conditions(LOCAL)])
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    items = [item for item in load_items(args.cohort, args.source) if item.rank < args.per_source]
    selected = [
        condition
        for condition in conditions(LOCAL)
        if args.condition is None or condition.name == args.condition
    ]
    if args.verify_weights:
        verify_models(args.models, selected)
    lib = load_library(args.library)
    for condition in selected:
        run_condition(lib, condition, items, args.artifacts, args.models, args.threads)


if __name__ == "__main__":
    main()

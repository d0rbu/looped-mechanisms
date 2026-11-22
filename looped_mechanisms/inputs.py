"""Freeze matched, seeded samples of public source material."""

import argparse
import hashlib
import json
import random
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from looped_mechanisms.design import ITEMS_PER_SOURCE, LABELS

SEED = 20261121
COHORT_VERSION = "inkling-text-v1"
SMOLTALK = "HuggingFaceTB/smoltalk"
SMOLTALK_REV = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"


@dataclass(frozen=True)
class Source:
    name: str
    category: str
    repo: str
    revision: str
    file: str
    split: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.file}"


SOURCES = (
    Source(
        "ToolACE",
        "Agent traces",
        "Team-ACE/ToolACE",
        "6bda777c88d21e5a204703c1ee45597a8fa4f734",
        "data.json",
        "train",
    ),
    Source(
        "Dolci Tool Use",
        "Agent traces",
        "allenai/Dolci-Think-SFT-Olmo-Hybrid-Tool-Use-SA",
        "9a418faa53d382c0f0a6cba79b7d1b05cbd07cd0",
        "data/train-00000-of-00001.parquet",
        "train",
    ),
    Source(
        "Everyday Conversations",
        "Conversation",
        SMOLTALK,
        SMOLTALK_REV,
        "data/everyday-conversations/test-00000-of-00001.parquet",
        "test",
    ),
    Source(
        "SystemChats",
        "Conversation",
        SMOLTALK,
        SMOLTALK_REV,
        "data/systemchats-30k/test-00000-of-00001.parquet",
        "test",
    ),
    Source(
        "ARC-Challenge",
        "Benchmark",
        "allenai/ai2_arc",
        "210d026faf9955653af8916fad021475a3f00453",
        "ARC-Challenge/test-00000-of-00001.parquet",
        "test",
    ),
    Source(
        "HellaSwag",
        "Benchmark",
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "data/validation-00000-of-00001.parquet",
        "validation",
    ),
)

ROLE_MARKERS = {"system": "system", "user": "user", "assistant": "model", "tool": "tool"}


@dataclass(frozen=True)
class Item:
    identifier: str
    source: str
    rank: int
    row: int
    text: str
    answer: int | None = None
    source_sha256: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def read_url(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urlopen(url, timeout=35) as response:
                return response.read()
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Download retry loop exhausted")


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render(messages: list[tuple[str, Any]], *, benchmark: bool = False) -> str:
    parts = ["<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"]
    for role, content in messages:
        text = content if isinstance(content, str) else json_text(content)
        if "<|" in text:
            raise ValueError("Source text has a reserved token marker")
        parts.append(
            f"<|message_{ROLE_MARKERS[role]}|><|content_text|>{text.strip()}<|end_message|>"
        )
        if role == "assistant":
            parts.append("<|content_model_end_sampling|>")
    if benchmark:
        parts.append("<|message_model|><|content_text|>")
    return "".join(parts)


def choice_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(
        f"{letter}. {answer}" for letter, answer in zip(LABELS, choices, strict=True)
    )
    return render(
        [("user", f"{question.strip()}\n\n{options}\n\nAnswer with one letter: A, B, C, or D.")],
        benchmark=True,
    )


def transform(source: Source, row: dict[str, Any]) -> tuple[str, int | None] | None:
    if source.name == "ARC-Challenge":
        choices = row["choices"]
        if len(choices["text"]) == 4:
            return choice_prompt(row["question"], choices["text"]), choices["label"].index(
                row["answerKey"]
            )
        return None
    if source.name == "HellaSwag":
        context = (row["ctx_a"] + " " + row["ctx_b"].capitalize()).strip()
        return choice_prompt(
            f"Choose the most plausible continuation.\n\n{context}", row["endings"]
        ), int(row["label"])
    if source.name == "ToolACE":
        mapping = {
            "user": "user",
            "assistant": "assistant",
            "tool": "tool",
            "tool_response": "tool",
            "function": "tool",
        }
        dialog = row["conversations"]
        if not any(message["from"] in {"tool", "tool_response", "function"} for message in dialog):
            return None
        policy, _, _ = row["system"].partition("Here is a list of functions in JSON format")
        messages = [("system", policy.strip())] if policy.strip() else []
        messages.extend((mapping[message["from"]], message["value"]) for message in dialog)
        return render(messages), None
    messages = [
        ("tool" if message["role"] == "environment" else message["role"], message["content"])
        for message in row["messages"]
    ]
    return render(messages), None


def sample_source(source: Source, raw: bytes) -> Iterator[Item]:
    source_hash = hashlib.sha256(raw).hexdigest()
    if source.file.endswith(".parquet"):
        from pyarrow import BufferReader
        from pyarrow.parquet import read_table

        rows = read_table(BufferReader(raw)).to_pylist()
    else:
        rows = json.loads(raw)
    order = list(range(len(rows)))
    random.Random(f"{SEED}:{source.name}").shuffle(order)
    seen = set()
    for row_index in order:
        try:
            transformed = transform(source, rows[row_index])
        except ValueError as error:
            if str(error) == "Source text has a reserved token marker":
                continue
            raise
        if transformed is None:
            continue
        text, answer = transformed
        # Conservative admission bound for a 1024-token window. The runtime also checks actual tokens.
        if not 50 <= len(text.encode()) <= 2400 or text in seen:
            continue
        yield Item(
            f"{SOURCES.index(source)}-{row_index:06d}",
            source.name,
            len(seen),
            row_index,
            text,
            answer,
            source_hash,
        )
        seen.add(text)


def freeze(source: Source, directory: Path) -> list[Item]:
    result = list(islice(sample_source(source, read_url(source.url)), ITEMS_PER_SOURCE))
    if len(result) != ITEMS_PER_SOURCE:
        raise RuntimeError(f"{source.name}: found only {len(result)} admissible rows")
    output = directory / f"{SOURCES.index(source)}.jsonl"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json_text(asdict(item)) + "\n" for item in result))
    temporary.replace(output)
    return result


def load_items(directory: Path, names: list[str] | None = None) -> list[Item]:
    items = []
    for index, source in enumerate(SOURCES):
        if names and source.name not in names:
            continue
        rows = [
            Item(**json.loads(line))
            for line in (directory / f"{index}.jsonl").read_text().splitlines()
        ]
        if len(rows) != ITEMS_PER_SOURCE or any(
            row.source != source.name or row.rank != i for i, row in enumerate(rows)
        ):
            raise ValueError(f"Invalid frozen cohort for {source.name}")
        hashes = {row.source_sha256 for row in rows}
        if len(hashes) != 1 or len(next(iter(hashes))) != 64:
            raise ValueError(f"Missing common source digest for {source.name}")
        if any(
            (row.answer in range(len(LABELS))) != (source.category == "Benchmark") for row in rows
        ):
            raise ValueError(f"Unexpected score target for {source.name}")
        items.extend(rows)
    return sorted(items, key=lambda item: (item.rank, item.identifier))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/cohort"))
    parser.add_argument("--source", choices=[source.name for source in SOURCES], action="append")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    selected = [source for source in SOURCES if not args.source or source.name in args.source]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(freeze, source, args.directory): source.name for source in selected}
        for future in as_completed(futures):
            print(f"{futures[future]}: froze {len(future.result())} items", flush=True)


if __name__ == "__main__":
    main()

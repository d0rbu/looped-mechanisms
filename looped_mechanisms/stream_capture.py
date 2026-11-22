"""Capture explicit token sequences from one resumable training shard."""

import argparse
import hashlib
import json
from pathlib import Path

from looped_mechanisms.inputs import SMOLTALK, Item
from looped_mechanisms.layer_pairs import collect
from looped_mechanisms.stream_corpus import CONTEXT_TOKENS, DATASET

STUDENTS = (6, 10, 17)


def load_shard(
    path: Path,
) -> tuple[list[Item], dict[str, list[int]], dict[str, list[bool]], int]:
    shard = json.loads(path.read_text())
    dataset = shard.get("dataset", DATASET)
    if dataset not in {DATASET, SMOLTALK}:
        raise ValueError(f"Unknown distillation dataset in {path}")
    items = []
    tokenized = {}
    masks = {}
    for example in shard["examples"]:
        identifier, tokens = example["id"], example["tokens"]
        if not identifier.isascii() or not identifier.isdigit() or identifier in tokenized:
            raise ValueError(f"Invalid or duplicate sequence identifier in {path}")
        if not 1 <= len(tokens) <= CONTEXT_TOKENS or not all(
            type(token) is int and token >= 0 for token in tokens
        ):
            raise ValueError(f"Invalid token sequence for {identifier}")
        digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        if dataset == SMOLTALK:
            targets = example["targets"]
            if (
                len(targets) != len(tokens)
                or not all(type(flag) is bool for flag in targets)
                or not any(targets)
            ):
                raise ValueError(f"Invalid assistant mask for {identifier}")
            masks[identifier] = targets
        items.append(Item(identifier, dataset, int(identifier), example["source_row"], digest))
        tokenized[identifier] = tokens
    total = (
        sum(sum(mask) for mask in masks.values()) if masks else sum(map(len, tokenized.values()))
    )
    if not total or total != shard["tokens"]:
        raise ValueError(f"Invalid token count in {path}")
    return items, tokenized, masks, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_file", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        parser.error("worker must be in [0, workers)")
    items, tokenized, _, _ = load_shard(args.shard_file)
    collect(
        items[args.worker :: args.workers],
        args.destination,
        False,
        args.batch_size,
        train_layers=STUDENTS,
        tokenized=tokenized,
    )


if __name__ == "__main__":
    main()

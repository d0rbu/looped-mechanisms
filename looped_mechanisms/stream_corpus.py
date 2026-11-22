"""Freeze a bounded, tokenized stream of new documents for layer distillation."""

import argparse
import hashlib
from pathlib import Path

from looped_mechanisms.results import atomic_json

DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TEACHER = "thinkingmachines/Inkling-2-Mini"
TARGET_TOKENS = 10_000_000
SHARD_TOKENS = 250_000
CONTEXT_TOKENS = 256
MILESTONES = (250_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000)


def main() -> None:
    from datasets import load_dataset  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/distillation-10m"))
    parser.add_argument("--tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    args = parser.parse_args()
    if min(args.tokens, args.shard_tokens) <= 0:
        parser.error("Token budgets must be positive")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    source = load_dataset(
        DATASET, name="sample-10BT", split="train", revision=REVISION, streaming=True
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    ordinal = 0
    seen: set[bytes] = set()
    batch: list[dict[str, object]] = []
    batch_tokens = 0
    for row_number, example in enumerate(source):
        description = example["text"]
        if "<|" in description:
            continue
        fingerprint = hashlib.sha256(description.strip().encode()).digest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        tokens = tokenizer.encode(description, add_special_tokens=False)
        cursor = 0
        while cursor < len(tokens) and completed < args.tokens:
            width = min(
                CONTEXT_TOKENS,
                len(tokens) - cursor,
                args.tokens - completed,
                args.shard_tokens - batch_tokens,
            )
            batch.append(
                {
                    "id": f"{ordinal:08d}",
                    "source_row": row_number,
                    "offset": cursor,
                    "tokens": tokens[cursor : cursor + width],
                }
            )
            cursor += width
            completed += width
            batch_tokens += width
            ordinal += 1
            if batch_tokens == args.shard_tokens or completed == args.tokens:
                shard = (completed - 1) // args.shard_tokens
                atomic_json(
                    output / f"{shard:04d}.json", {"tokens": batch_tokens, "examples": batch}
                )
                print(
                    f"Frozen shard {shard:04d}: {completed:,}/{args.tokens:,} token positions",
                    flush=True,
                )
                batch = []
                batch_tokens = 0
        if completed == args.tokens:
            break
    if completed != args.tokens:
        raise RuntimeError("The pinned source ended before the requested token budget")
    atomic_json(
        output / "manifest.json",
        {
            "dataset": DATASET,
            "revision": REVISION,
            "teacher": TEACHER,
            "tokens": completed,
            "context": CONTEXT_TOKENS,
            "shard_tokens": args.shard_tokens,
            "shards": (completed - 1) // args.shard_tokens + 1,
        },
    )


if __name__ == "__main__":
    main()

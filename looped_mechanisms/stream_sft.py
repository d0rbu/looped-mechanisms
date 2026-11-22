"""Freeze chat prompts with distinct assistant-token distillation targets."""

import argparse
import re
from pathlib import Path

from looped_mechanisms.inputs import SEED, SMOLTALK, SMOLTALK_REV, load_items, render
from looped_mechanisms.results import atomic_json
from looped_mechanisms.stream_corpus import CONTEXT_TOKENS, SHARD_TOKENS, TEACHER

ASSISTANT = re.compile(r"<\|message_model\|><\|content_text\|>(.*?)<\|end_message\|>", re.DOTALL)


def assistant_mask(text: str, offsets: list[tuple[int, int]]) -> list[bool]:
    spans = [match.span(1) for match in ASSISTANT.finditer(text)]
    return [
        start < end and any(start >= low and end <= high for low, high in spans)
        for start, end in offsets
    ]


def main() -> None:
    from datasets import load_dataset  # ty: ignore[unresolved-import]
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/distillation-sft-10m"))
    parser.add_argument("--tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("Token budget must be positive")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    source = load_dataset(
        SMOLTALK, name="all", split="train", revision=SMOLTALK_REV, streaming=True
    ).shuffle(seed=SEED, buffer_size=2048)
    seen = {item.text for item in load_items(Path("data/cohort"))}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    consumed = stored_context = ordinal = 0
    entries: list[dict[str, object]] = []
    shard_count = 0
    for row, example in enumerate(source):
        messages = [(part["role"], part["content"]) for part in example["messages"]]
        if any(
            role not in {"system", "user", "assistant"} or "<|" in content
            for role, content in messages
        ):
            continue
        text = render(messages)
        if text in seen:
            continue
        seen.add(text)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if len(ids) > CONTEXT_TOKENS:
            continue
        mask = assistant_mask(text, encoded["offset_mapping"])
        positions = [index for index, include in enumerate(mask) if include]
        offset = 0
        while offset < len(positions) and consumed < args.tokens:
            take = min(len(positions) - offset, SHARD_TOKENS - shard_count, args.tokens - consumed)
            selected = set(positions[offset : offset + take])
            entries.append(
                {
                    "id": f"{ordinal:08d}",
                    "source_row": row,
                    "tokens": ids,
                    "targets": [index in selected for index in range(len(ids))],
                }
            )
            offset += take
            consumed += take
            shard_count += take
            stored_context += len(ids)
            ordinal += 1
            if shard_count == SHARD_TOKENS or consumed == args.tokens:
                shard = (consumed - 1) // SHARD_TOKENS
                atomic_json(
                    output / f"{shard:04d}.json",
                    {
                        "dataset": SMOLTALK,
                        "tokens": shard_count,
                        "examples": entries,
                    },
                )
                print(
                    f"Frozen SFT shard {shard:04d}: {consumed:,}/{args.tokens:,} assistant tokens",
                    flush=True,
                )
                entries = []
                shard_count = 0
        if consumed == args.tokens:
            break
    if consumed != args.tokens:
        raise RuntimeError("The SFT source ended before the requested assistant-token budget")
    atomic_json(
        output / "manifest.json",
        {
            "dataset": SMOLTALK,
            "revision": SMOLTALK_REV,
            "teacher": TEACHER,
            "tokens": consumed,
            "context_tokens": stored_context,
            "context": CONTEXT_TOKENS,
            "shard_tokens": SHARD_TOKENS,
            "shards": (consumed - 1) // SHARD_TOKENS + 1,
        },
    )


if __name__ == "__main__":
    main()

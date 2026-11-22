"""Train one student on one captured shard with an atomic resumable frontier."""

import argparse
import hashlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch  # ty: ignore[unresolved-import]

from looped_mechanisms.design import HIDDEN
from looped_mechanisms.distill import (
    STUDENT_WIDTH,
    LayerStudent,
    files_for,
    read_sequence,
    save_weights,
    train,
)
from looped_mechanisms.inputs import SEED
from looped_mechanisms.stream_capture import STUDENTS, load_shard
from looped_mechanisms.stream_corpus import MILESTONES, REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_file", type=Path)
    parser.add_argument("captures", type=Path)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--ordinal", type=int)
    parser.add_argument("--save-snapshot", action="store_true")
    parser.add_argument("--phase", choices=("pretrain", "sft"), default="pretrain")
    parser.add_argument("--layer", type=int, choices=STUDENTS, required=True)
    args = parser.parse_args()
    shard = int(args.shard_file.stem) if args.ordinal is None else args.ordinal
    fingerprint = hashlib.sha256(args.shard_file.read_bytes()).hexdigest()
    items, _, masks, token_count = load_shard(args.shard_file)
    if bool(masks) != (args.phase == "sft"):
        raise ValueError("Assistant masks and training phase do not match")
    root = args.run_directory
    frontier = root / f"layer-{args.layer:02d}-frontier.pt"
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint = (
        torch.load(frontier, map_location="cpu", mmap=True, weights_only=True)
        if frontier.exists()
        else None
    )
    next_shard = checkpoint["next_shard"] if checkpoint else 0
    if checkpoint and (
        checkpoint["train_layer"] != args.layer
        or checkpoint["revision"] != REVISION
        or checkpoint["width"] != STUDENT_WIDTH
    ):
        raise ValueError("Checkpoint does not match this stream")
    consumed = checkpoint.get("shard_hashes", []) if checkpoint else []
    if len(consumed) != next_shard:
        raise ValueError("Checkpoint shard receipts do not match its frontier")
    if next_shard > shard:
        if consumed[shard] != fingerprint:
            raise ValueError("Completed shard content differs from the checkpoint")
        print(f"Layer {args.layer} already consumed shard {shard:04d}", flush=True)
        return
    if next_shard != shard:
        raise ValueError(f"Expected shard {next_shard:04d}, received {shard:04d}")
    paths = files_for(items, args.captures, list(STUDENTS))
    examples = []
    for item, path in zip(items, paths, strict=True):
        example = read_sequence(path, args.layer, list(STUDENTS))
        if item.identifier in masks:
            example = replace(example, target_mask=np.array(masks[item.identifier], dtype=bool))
        examples.append(example)
    if sum(example.training_tokens for example in examples) != token_count:
        raise ValueError("Captured tokens do not match this corpus shard")
    previous_history = checkpoint["history"] if checkpoint else []
    previous_prompts = checkpoint["train_prompts"] if checkpoint else 0
    copy_error = checkpoint["copy_error"] if checkpoint else 0.0
    if checkpoint:
        with torch.device("meta"):
            student = LayerStudent(STUDENT_WIDTH)
        student.load_state_dict(checkpoint["model"], assign=True)
        student.frequency = torch.exp(-math.log(10_000) * torch.arange(0, 64, 2) / 64)
        student.to(device)
    else:
        with torch.device(device):
            student = LayerStudent(STUDENT_WIDTH)
    rate = 0.005
    optimizer = torch.optim.Adafactor(student.parameters(), lr=rate, foreach=False)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint
    for example in examples:
        delta = example.target.astype(np.float32) - example.active
        supervised = delta if example.target_mask is None else delta[example.target_mask]
        copy_error += np.square(supervised).sum(dtype=np.float64)
    student.train()
    history = train(student, examples, 1, 384, device, rate, optimizer=optimizer, seed=SEED + shard)
    del examples
    combined = [*previous_history, {**history[0], "epoch": shard + 1, "phase": args.phase}]
    total = sum(row["tokens"] for row in combined)
    metadata = {
        "width": STUDENT_WIDTH,
        "train_layer": args.layer,
        "train_copy_rmse": math.sqrt(float(copy_error) / (total * HIDDEN)),
        "history": combined,
        "train_prompts": previous_prompts + len(items),
        "training_corpus": items[0].source,
        "revision": REVISION,
        "next_shard": shard + 1,
        "shard_hashes": [*consumed, fingerprint],
        "copy_error": float(copy_error),
    }
    if args.save_snapshot or (args.phase == "pretrain" and total in MILESTONES):
        snapshot = (
            root / "checkpoints" / f"stream-{args.phase}-{total:08d}-layer-{args.layer:02d}.pt"
        )
        save_weights(snapshot, student, metadata, compact=True)
    save_weights(frontier, student, {**metadata, "optimizer": optimizer.state_dict()})
    print(
        f"Layer {args.layer} consumed {total:,} distinct tokens through shard {shard:04d}",
        flush=True,
    )


if __name__ == "__main__":
    main()

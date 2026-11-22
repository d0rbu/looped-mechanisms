"""Prepare and capture the expanded 16-loop classifier cohort."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import numpy as np

from looped_mechanisms.design import HIDDEN, LOCAL, POSITIONS, Condition
from looped_mechanisms.inputs import (
    SEED,
    SOURCES,
    Item,
    Source,
    load_items,
    read_url,
    sample_source,
)
from looped_mechanisms.local_run import load_library, run_condition
from looped_mechanisms.probe_study import load_core, validation_ids
from looped_mechanisms.results import atomic_json

MEASURED = Condition("variable", 16)
BASELINE = Path("data/cohort")
COHORT = Path("data/loop-probe-16-expanded.json")
CAPTURES = Path("artifacts/loop-probe-16-expanded")
TRAIN_PROMPTS = 2_880
WIDTH = 159
EPOCHS = 20
BATCH = 2_048
BASELINE_SCORES = Path("site/loop-study/v1") / LOCAL / MEASURED.name / "scores.json"
REPORT = BASELINE_SCORES.with_name("expanded-scores.json")


def distillation_ready() -> bool:
    unit = "looped-mechanisms-10m-stream"
    if subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False
    ).returncode:
        return False
    root = Path("artifacts/distillation-stream")
    plan = json.loads((root / "plan.json").read_text())
    ordinal = len(list((root / "completed").glob("*.json")))
    staging = root / "staging" / f"{ordinal:04d}"
    if not staging.is_dir():
        return False
    pretrain_shards = plan["pretrain_tokens"] // 250_000
    if ordinal < pretrain_shards:
        source = Path("data/distillation-10m") / f"{ordinal:04d}.json"
    else:
        source = Path("data/distillation-sft-10m") / f"{ordinal - pretrain_shards:04d}.json"
    return all(
        (staging / f"{row['id']}.json").is_file()
        for row in json.loads(source.read_text())["examples"]
    )


def prepare(output: Path) -> None:
    base = load_items(BASELINE)
    held_out = validation_ids(base)
    train = [item for item in base if item.identifier not in held_out]
    seen = {item.text for item in base}

    def candidates(source: Source) -> deque[Item]:
        frozen = [item for item in base if item.source == source.name]
        pool = list(islice(sample_source(source, read_url(source.url)), TRAIN_PROMPTS))
        if pool[: len(frozen)] != frozen:
            raise ValueError(f"{source.name} no longer reproduces the original frozen sample")
        return deque(pool[len(frozen) :])

    with ThreadPoolExecutor(max_workers=3) as pool:
        queues = list(pool.map(candidates, SOURCES))
    extras = []
    while len(train) + len(extras) < TRAIN_PROMPTS and any(queues):
        for queue in queues:
            while queue:
                example = queue.popleft()
                if example.text not in seen:
                    extras.append(example)
                    seen.add(example.text)
                    break
            if len(train) + len(extras) == TRAIN_PROMPTS:
                break
    if len(train) + len(extras) != TRAIN_PROMPTS:
        raise ValueError("The pinned sources did not yield 2,880 distinct training prompts")
    atomic_json(output, [asdict(item) for item in extras])
    print(f"Added {len(extras):,} unique prompts: {dict(Counter(item.source for item in extras))}")


def train(
    cohort: Path,
    examples: list[Item],
    *,
    yield_to_distillation: bool = False,
    device_name: str = "cuda",
) -> None:
    import torch  # ty: ignore[unresolved-import]
    from torch import nn  # ty: ignore[unresolved-import]

    base = load_items(BASELINE)
    held_out = validation_ids(base)
    val = [item for item in base if item.identifier in held_out]
    original = [item for item in base if item.identifier not in held_out]
    if len(original) + len(examples) != TRAIN_PROMPTS:
        raise ValueError("Expected 2,880 training prompts")
    if {x.identifier for x in base} & {x.identifier for x in examples}:
        raise ValueError("Extra captures overlap the baseline cohort")
    if len({x.text for x in [*original, *examples]}) != TRAIN_PROMPTS or {
        x.text for x in [*original, *examples]
    } & {x.text for x in val}:
        raise ValueError("Duplicate training prompts or validation overlap")
    base_train = load_core(Path("artifacts"), LOCAL, MEASURED, original)
    extra_train = load_core(CAPTURES, LOCAL, MEASURED, examples)
    validation = load_core(Path("artifacts"), LOCAL, MEASURED, val)
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    reference = json.loads(BASELINE_SCORES.read_text())
    torch.set_float32_matmul_precision("high")
    device = torch.device(device_name)
    loops = MEASURED.passes
    loop_order = torch.arange(loops, device=device).repeat_interleave(POSITIONS)
    train_labels = loop_order.repeat(TRAIN_PROMPTS)
    val_labels = loop_order.repeat(len(val))
    output = CAPTURES / "heads"
    output.mkdir(parents=True, exist_ok=True)
    scores = []
    for offset in range(12):
        if yield_to_distillation and distillation_ready():
            return
        layer = 6 + offset
        weight = output / f"layer-{layer:02d}.pt"
        if weight.exists():
            prior = torch.load(weight, map_location="cpu", weights_only=True)
            if prior["cohort_sha256"] != digest:
                raise ValueError(f"Training cohort changed for layer {layer}")
            scores.append(prior["score"])
            continue
        train_rows = torch.from_numpy(
            np.concatenate((base_train[:, :, offset], extra_train[:, :, offset])).reshape(
                -1, HIDDEN
            )
        ).to(device=device, dtype=torch.float32)
        val_rows = torch.from_numpy(validation[:, :, offset].reshape(-1, HIDDEN)).to(
            device=device, dtype=torch.float32
        )
        scale, mean = torch.std_mean(train_rows, dim=0, correction=0)
        scale = scale.clamp_min(1e-6)
        train_rows.sub_(mean).div_(scale)
        val_rows.sub_(mean).div_(scale)
        torch.manual_seed(SEED + layer)
        model = nn.Sequential(nn.Linear(HIDDEN, WIDTH), nn.GELU(), nn.Linear(WIDTH, loops)).to(
            device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        for _ in range(EPOCHS):
            for indices in torch.randperm(len(train_rows), device=device).split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(
                    model(train_rows[indices]), train_labels[indices]
                )
                loss.backward()
                optimizer.step()
            schedule.step()
        model.eval()
        with torch.no_grad():
            predicted = torch.cat([model(batch).argmax(dim=1) for batch in val_rows.split(BATCH)])
            counts = torch.bincount(loops * val_labels + predicted, minlength=loops**2)
            confusion = counts.reshape(loops, loops)
            correct = int(confusion.diagonal().sum().item())
        score = {
            "layer": layer,
            "accuracy": correct / len(predicted),
            "baseline_accuracy": reference["layers"][offset]["accuracy"],
            "correct": correct,
            "examples": len(predicted),
            "confusion": confusion.tolist(),
        }
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != (HIDDEN + 1 + loops) * WIDTH + loops:
            raise ValueError("Unexpected classifier parameter count")
        tmp = weight.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "mean": mean,
                "scale": scale,
                "cohort_sha256": digest,
                "score": score,
            },
            tmp,
        )
        tmp.replace(weight)
        scores.append(score)
        del train_rows, val_rows
        print(f"Layer {layer}: {score['accuracy']:.2%} held-out top-1", flush=True)
    atomic_json(
        REPORT,
        {
            "head_dim": loops,
            "train_prompts": TRAIN_PROMPTS,
            "validation_prompts": len(val),
            "train_examples_per_layer": len(train_labels),
            "positions": POSITIONS,
            "parameters_per_classifier": (HIDDEN + 1 + loops) * WIDTH + loops,
            "baseline_parameters_per_classifier": (HIDDEN + 1) * loops,
            "train_by_source": dict(Counter(item.source for item in [*original, *examples])),
            "epochs": EPOCHS,
            "cohort_sha256": digest,
            "layers": scores,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "capture", "train"))
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--library", type=Path, default=Path("build/libloop_capture.so"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--yield-to-distillation", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.cohort)
        return
    examples = [Item(**row) for row in json.loads(args.cohort.read_text())]
    if args.action == "train":
        train(args.cohort, examples, yield_to_distillation=args.yield_to_distillation)
        return
    if args.models is None or args.offset < 0 or args.count <= 0:
        parser.error("capture requires --models, a nonnegative offset, and a positive count")
    run_condition(
        load_library(args.library),
        MEASURED,
        examples[args.offset : args.offset + args.count],
        CAPTURES,
        args.models,
        threads=4,
        should_yield=distillation_ready if args.yield_to_distillation else None,
    )


if __name__ == "__main__":
    main()

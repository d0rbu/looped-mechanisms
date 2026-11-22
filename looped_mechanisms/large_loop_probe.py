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
    SMOLTALK_REV,
    SOURCES,
    Item,
    Source,
    load_items,
    read_url,
    sample_source,
)
from looped_mechanisms.local_run import load_library, run_condition
from looped_mechanisms.loop_classifier import BASE_WIDTH, EPOCHS, fit, parameter_count
from looped_mechanisms.probe_study import load_core, validation_ids
from looped_mechanisms.results import atomic_json
from looped_mechanisms.stream_corpus import TEACHER

MEASURED = Condition("variable", 16)
BASELINE = Path("data/cohort")
COHORT = Path("data/loop-probe-16-expanded.json")
SUPPLEMENT = Path("data/loop-probe-16-supplement.json")
SFT_CORPUS = Path("data/distillation-sft-10m")
CAPTURES = Path("artifacts/loop-probe-16-expanded")
COMPACT_CAPTURES = Path("artifacts/loop-probe-16-layer-06")
BASE_TRAIN_PROMPTS = 2_880
TRAIN_PROMPTS = 28_800
WIDTH = 10 * BASE_WIDTH
TARGET_LAYER = 6
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
        pool = list(islice(sample_source(source, read_url(source.url)), BASE_TRAIN_PROMPTS))
        if pool[: len(frozen)] != frozen:
            raise ValueError(f"{source.name} no longer reproduces the original frozen sample")
        return deque(pool[len(frozen) :])

    with ThreadPoolExecutor(max_workers=3) as pool:
        queues = list(pool.map(candidates, SOURCES))
    extras = []
    while len(train) + len(extras) < BASE_TRAIN_PROMPTS and any(queues):
        for queue in queues:
            while queue:
                example = queue.popleft()
                if example.text not in seen:
                    extras.append(example)
                    seen.add(example.text)
                    break
            if len(train) + len(extras) == BASE_TRAIN_PROMPTS:
                break
    if len(train) + len(extras) != BASE_TRAIN_PROMPTS:
        raise ValueError("The pinned sources did not yield 2,880 distinct training prompts")
    atomic_json(output, [asdict(item) for item in extras])
    print(f"Added {len(extras):,} unique prompts: {dict(Counter(item.source for item in extras))}")


def load_examples(path: Path) -> list[Item]:
    return [Item(**row) for row in json.loads(path.read_text())]


def prepare_supplement(cohort: Path, output: Path) -> None:
    from transformers import AutoTokenizer  # ty: ignore[unresolved-import]

    base = [*load_items(BASELINE), *load_examples(cohort)]
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    base_tokens = tokenizer([item.text for item in base], add_special_tokens=False)["input_ids"]
    seen = {tuple(tokens) for tokens in base_tokens}
    manifest = json.loads((SFT_CORPUS / "manifest.json").read_text())
    if manifest["revision"] != SMOLTALK_REV or manifest["tokens"] != 10_000_000:
        raise ValueError("Unexpected pinned SmolTalk corpus")
    selected = []
    budget = TRAIN_PROMPTS - BASE_TRAIN_PROMPTS
    for index in range(manifest["shards"]):
        raw = (SFT_CORPUS / f"{index:04d}.json").read_bytes()
        source_digest = hashlib.sha256(raw).hexdigest()
        rows = json.loads(raw)["examples"]
        texts = tokenizer.batch_decode(
            [row["tokens"] for row in rows],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for row, text, ids in zip(rows, texts, encoded, strict=True):
            tokens = tuple(row["tokens"])
            if tuple(ids) != tokens or tokens in seen or len(tokens) < POSITIONS:
                continue
            seen.add(tokens)
            source_row = row["source_row"]
            selected.append(
                Item(
                    f"smol-{source_row:07d}",
                    "SmolTalk train",
                    len(selected),
                    source_row,
                    text,
                    source_sha256=source_digest,
                )
            )
            if len(selected) == budget:
                atomic_json(output, [asdict(item) for item in selected])
                print(f"Added {budget:,} distinct SmolTalk prompts")
                return
    raise ValueError(f"Only {len(selected):,} eligible supplemental SmolTalk prompts")


def train(
    cohort: Path,
    examples: list[Item],
    supplement: Path,
    *,
    yield_to_distillation: bool = False,
    device_name: str = "cuda",
) -> None:
    base = load_items(BASELINE)
    held_out = validation_ids(base)
    val = [item for item in base if item.identifier in held_out]
    original = [item for item in base if item.identifier not in held_out]
    supplemental = load_examples(supplement)
    training_items = [*original, *examples, *supplemental]
    if (
        len(original) + len(examples) != BASE_TRAIN_PROMPTS
        or len(training_items) != TRAIN_PROMPTS
        or len({item.identifier for item in [*training_items, *val]}) != TRAIN_PROMPTS + len(val)
        or len({item.text for item in [*training_items, *val]}) != TRAIN_PROMPTS + len(val)
    ):
        raise ValueError("Expected 28,800 unique training prompts and a disjoint original holdout")
    reference = json.loads(BASELINE_SCORES.read_text())
    baseline_digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    if (
        reference.get("hidden_units") != BASE_WIDTH
        or reference["train_prompts"] != BASE_TRAIN_PROMPTS
        or reference.get("extra_cohort_sha256") != baseline_digest
    ):
        raise ValueError("Publish the 2,880-prompt baseline before fitting the expanded probe")
    if yield_to_distillation and distillation_ready():
        return
    artifact_root = Path("artifacts")
    loops = MEASURED.passes
    training = np.empty((TRAIN_PROMPTS, loops, 1, POSITIONS, HIDDEN), dtype=np.float16)
    load_core(
        artifact_root, LOCAL, MEASURED, original, layer=TARGET_LAYER, out=training[: len(original)]
    )
    load_core(
        CAPTURES,
        LOCAL,
        MEASURED,
        examples,
        layer=TARGET_LAYER,
        out=training[len(original) : BASE_TRAIN_PROMPTS],
    )
    load_core(
        COMPACT_CAPTURES,
        LOCAL,
        MEASURED,
        supplemental,
        layer=TARGET_LAYER,
        capture_layer=TARGET_LAYER,
        out=training[BASE_TRAIN_PROMPTS:],
    )
    validation = load_core(artifact_root, LOCAL, MEASURED, val, layer=TARGET_LAYER)
    if yield_to_distillation and distillation_ready():
        return
    digest = hashlib.sha256(supplement.read_bytes()).hexdigest()
    lineage = {"extra_cohort_sha256": baseline_digest, "supplement_sha256": digest}
    output = CAPTURES / "heads"
    score = fit(
        training[:, :, 0],
        validation[:, :, 0],
        layer=TARGET_LAYER,
        width=WIDTH,
        batch_size=2_048,
        device_name=device_name,
        checkpoint=output / f"layer-{TARGET_LAYER:02d}-width-{WIDTH}-prompts-{TRAIN_PROMPTS}.pt",
        lineage={**lineage, "train_prompts": TRAIN_PROMPTS, "validation_ids": sorted(held_out)},
    )
    reference_score = next(row for row in reference["layers"] if row["layer"] == TARGET_LAYER)
    atomic_json(
        REPORT,
        {
            "head_dim": loops,
            "train_prompts": TRAIN_PROMPTS,
            "validation_prompts": len(val),
            "train_examples_per_layer": len(training) * loops * POSITIONS,
            "hidden_units": WIDTH,
            "positions": POSITIONS,
            "parameters_per_classifier": parameter_count(WIDTH, loops),
            "baseline_parameters_per_classifier": parameter_count(BASE_WIDTH, loops),
            "train_by_source": dict(Counter(item.source for item in training_items)),
            "epochs": EPOCHS,
            **lineage,
            "layers": [{**score, "baseline_accuracy": reference_score["accuracy"]}],
        },
    )
    print(f"Layer {TARGET_LAYER}: {score['accuracy']:.2%} held-out top-1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "prepare-supplement", "capture", "capture-hosted", "train")
    )
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--supplement", type=Path, default=SUPPLEMENT)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--library", type=Path, default=Path("build/libloop_capture.so"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--passes", type=int, choices=range(2, 17), default=16)
    parser.add_argument("--checkpoint", choices=("fixed", "variable"), default="variable")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--yield-to-distillation", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.cohort)
        return
    if args.action == "prepare-supplement":
        prepare_supplement(args.cohort, args.supplement)
        return
    examples = load_examples(args.cohort)
    if args.action == "capture-hosted":
        from looped_mechanisms.ndif_run import run_condition as run_hosted

        run_hosted(Condition(args.checkpoint, 3), examples, args.capture_root or CAPTURES, 8)
        return
    if args.action == "train":
        train(
            args.cohort, examples, args.supplement, yield_to_distillation=args.yield_to_distillation
        )
        return
    if args.models is None or args.offset < 0 or args.count <= 0:
        parser.error("capture requires --models, a nonnegative offset, and a positive count")
    if args.compact:
        if args.checkpoint != MEASURED.checkpoint or args.passes != MEASURED.passes:
            parser.error("compact capture is for the expanded 16-loop probe")
        examples = load_examples(args.supplement)
    run_condition(
        load_library(args.library),
        Condition(args.checkpoint, args.passes),
        examples[args.offset : args.offset + args.count],
        args.capture_root or (COMPACT_CAPTURES if args.compact else CAPTURES),
        args.models,
        threads=4,
        should_yield=distillation_ready if args.yield_to_distillation else None,
        capture_layer=TARGET_LAYER if args.compact else None,
    )


if __name__ == "__main__":
    main()

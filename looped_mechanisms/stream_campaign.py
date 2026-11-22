"""Capture and consume one bounded teacher shard at a time."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from looped_mechanisms.checkpoint_catalog import publish_checkpoints
from looped_mechanisms.inputs import SMOLTALK_REV
from looped_mechanisms.results import atomic_json
from looped_mechanisms.stream_capture import STUDENTS, load_shard
from looped_mechanisms.stream_corpus import MILESTONES, REVISION

MIN_FREE_BYTES = 50_000_000_000


def wait_for_unit(unit: str) -> None:
    while (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit], check=False
        ).returncode
        == 0
    ):
        time.sleep(15)


def wait_for_gpu() -> None:
    while True:
        status = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        free, utilization = (int(field) for field in status[0].split(","))
        if free >= 17_500 and utilization <= 15:
            return
        time.sleep(15)


def capture(teacher: Path, shard_file: Path, destination: Path, workers: int) -> None:
    items, _, _, _ = load_shard(shard_file)

    def worker(index: int) -> None:
        markers = [destination / f"{item.identifier}.json" for item in items[index::workers]]
        done = sum(marker.exists() for marker in markers)
        request = [
            "uv",
            "run",
            "--env-file",
            ".env",
            "--no-project",
            "--python",
            str(teacher),
            "--",
            "python",
            "-u",
            "-m",
            "looped_mechanisms.stream_capture",
            str(shard_file),
            str(destination),
            "--worker",
            str(index),
            "--workers",
            str(workers),
            "--batch-size",
            "4",
        ]
        failures = 0
        while failures < 8:
            if subprocess.run(request, check=False).returncode == 0:
                return
            current = sum(marker.exists() for marker in markers)
            failures = 1 if current > done else failures + 1
            done = current
            time.sleep(min(8 * failures, 60))
        raise RuntimeError(f"Teacher capture worker {index} failed repeatedly")

    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers)))


def planned_shards(
    phases: tuple[tuple[str, Path, str, int], ...],
) -> Iterator[tuple[str, Path, bool]]:
    for phase, corpus, expected_revision, budget in phases:
        manifest_path = corpus / "manifest.json"
        if phase == "sft" and not manifest_path.is_file():
            wait_for_unit("looped-mechanisms-sft-10m-corpus")
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != expected_revision or manifest["tokens"] < budget:
            raise ValueError(f"Unexpected frozen corpus for {phase}")
        if budget % manifest["shard_tokens"]:
            raise ValueError(f"The {phase} budget must end on a shard boundary")
        milestones = MILESTONES if phase == "pretrain" else {250_000, 1_000_000, 5_000_000}
        for index in range(budget // manifest["shard_tokens"]):
            boundary = (index + 1) * manifest["shard_tokens"]
            yield phase, corpus / f"{index:04d}.json", boundary == budget or boundary in milestones


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-python", type=Path, required=True)
    parser.add_argument("--student-python", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/distillation-10m"))
    parser.add_argument("--pretrain-tokens", type=int, default=10_000_000)
    parser.add_argument("--sft-corpus", type=Path, default=Path("data/distillation-sft-10m"))
    parser.add_argument("--sft-tokens", type=int, default=10_000_000)
    parser.add_argument("--run-directory", type=Path, default=Path("artifacts/distillation-stream"))
    parser.add_argument(
        "--publication", type=Path, default=Path("site/loop-study/v1/distillation-checkpoints")
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    if min(args.pretrain_tokens, args.sft_tokens) <= 0:
        parser.error("Both token budgets must be positive")
    phases = (
        ("pretrain", args.corpus, REVISION, args.pretrain_tokens),
        ("sft", args.sft_corpus, SMOLTALK_REV, args.sft_tokens),
    )
    root = args.run_directory
    receipts = root / "completed"
    receipts.mkdir(parents=True, exist_ok=True)
    args.publication.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    descriptor = {
        "pretrain_tokens": args.pretrain_tokens,
        "pretrain_revision": REVISION,
        "sft_tokens": args.sft_tokens,
        "sft_revision": SMOLTALK_REV,
    }
    if plan_path.exists() and json.loads(plan_path.read_text()) != descriptor:
        raise ValueError("The requested corpus allocation differs from this run's plan")
    atomic_json(plan_path, descriptor)
    wait_for_unit("looped-mechanisms-10m-capture-smoke")
    wait_for_unit("looped-mechanisms-additional-student-capture")
    consumed = 0
    target = args.pretrain_tokens + args.sft_tokens
    for ordinal, (phase, source, snapshot) in enumerate(planned_shards(phases)):
        name = f"{ordinal:04d}"
        destination = root / "staging" / name
        receipt = receipts / f"{name}.json"
        items, _, _, tokens = load_shard(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        consumed += tokens
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record != {
                "corpus_sha256": digest,
                "tokens_consumed": consumed,
                "layers": list(STUDENTS),
                "phase": phase,
            }:
                raise ValueError(f"Invalid consumption receipt for {name}")
        else:
            if shutil.disk_usage(root).free < MIN_FREE_BYTES:
                raise RuntimeError("Insufficient disk headroom for one shard and its checkpoints")
            print(f"Capturing {phase} shard {name}, {tokens:,} training positions", flush=True)
            capture(args.teacher_python, source, destination, args.workers)
            wait_for_unit("looped-mechanisms-additional-students")
            for layer in STUDENTS:
                wait_for_gpu()
                subprocess.run(
                    [
                        str(args.student_python),
                        "-u",
                        "-m",
                        "looped_mechanisms.stream_step",
                        str(source),
                        str(destination),
                        str(root),
                        "--layer",
                        str(layer),
                        "--ordinal",
                        str(ordinal),
                        "--phase",
                        phase,
                        *(["--save-snapshot"] if snapshot else []),
                    ],
                    check=True,
                )
                if snapshot:
                    checkpoint = f"stream-{phase}-{consumed:08d}-layer-{layer:02d}"
                    wait_for_gpu()
                    subprocess.run(
                        [
                            str(args.student_python),
                            "-u",
                            "-m",
                            "looped_mechanisms.distill",
                            "--phase",
                            "evaluate",
                            "--train-layer",
                            str(layer),
                            "--weights",
                            str(root / "checkpoints" / f"{checkpoint}.pt"),
                            "--output",
                            str(args.publication / f"{checkpoint}.json"),
                        ],
                        check=True,
                    )
                    publish_checkpoints(args.publication)
            atomic_json(
                receipt,
                {
                    "corpus_sha256": digest,
                    "tokens_consumed": consumed,
                    "layers": list(STUDENTS),
                    "phase": phase,
                },
            )
        # Only the capture files for this verified, fully consumed shard are disposable.
        if destination.exists():
            for item in items:
                for extension in (".npz", ".json"):
                    (destination / f"{item.identifier}{extension}").unlink(missing_ok=True)
            destination.rmdir()
        print(f"All students consumed {consumed:,}/{target:,} teacher tokens", flush=True)
    if consumed != target:
        raise ValueError("The completed shard receipts do not cover the corpus")
    publish_checkpoints(args.publication, completed=True)


if __name__ == "__main__":
    main()

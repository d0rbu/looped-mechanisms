"""Capture and consume one bounded teacher shard at a time."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from looped_mechanisms.checkpoint_catalog import publish_checkpoints
from looped_mechanisms.results import atomic_json
from looped_mechanisms.stream_capture import STUDENTS, load_shard
from looped_mechanisms.stream_corpus import MILESTONES, REVISION, TARGET_TOKENS

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
    items, _, _ = load_shard(shard_file)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-python", type=Path, required=True)
    parser.add_argument("--student-python", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/distillation-10m"))
    parser.add_argument("--run-directory", type=Path, default=Path("artifacts/distillation-stream"))
    parser.add_argument(
        "--publication", type=Path, default=Path("site/loop-study/v1/distillation-checkpoints")
    )
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["tokens"] != TARGET_TOKENS:
        raise ValueError("Expected the frozen 10M-token corpus")
    root = args.run_directory
    receipts = root / "completed"
    receipts.mkdir(parents=True, exist_ok=True)
    args.publication.mkdir(parents=True, exist_ok=True)
    wait_for_unit("looped-mechanisms-10m-capture-smoke")
    wait_for_unit("looped-mechanisms-additional-student-capture")
    consumed = 0
    for shard in range(manifest["shards"]):
        name = f"{shard:04d}"
        source = args.corpus / f"{name}.json"
        destination = root / "staging" / name
        receipt = receipts / f"{name}.json"
        items, _, tokens = load_shard(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        consumed += tokens
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record != {
                "corpus_sha256": digest,
                "tokens_consumed": consumed,
                "layers": list(STUDENTS),
            }:
                raise ValueError(f"Invalid consumption receipt for {name}")
        else:
            if shutil.disk_usage(root).free < MIN_FREE_BYTES:
                raise RuntimeError("Insufficient disk headroom for one shard and its checkpoints")
            print(f"Capturing training shard {name}, {tokens:,} token positions", flush=True)
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
                    ],
                    check=True,
                )
                if consumed in MILESTONES:
                    checkpoint = f"stream-{consumed:08d}-layer-{layer:02d}"
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
                },
            )
        # Only the capture files for this verified, fully consumed shard are disposable.
        if destination.exists():
            for item in items:
                for extension in (".npz", ".json"):
                    (destination / f"{item.identifier}{extension}").unlink(missing_ok=True)
            destination.rmdir()
        print(f"All students consumed {consumed:,}/{TARGET_TOKENS:,} teacher tokens", flush=True)
    if consumed != TARGET_TOKENS:
        raise ValueError("The completed shard receipts do not cover the corpus")


if __name__ == "__main__":
    main()

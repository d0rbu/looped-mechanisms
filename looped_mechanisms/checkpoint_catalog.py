"""Publish completed checkpoint evaluations for the dashboard slider."""

import argparse
import json
from pathlib import Path

from looped_mechanisms.results import atomic_json


def publish_checkpoints(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        report = json.loads(path.read_text())
        if len(report["layers"]) != 24:
            continue
        entries.append(
            {
                "file": path.name,
                "layer": report["train_layer"],
                "tokens": sum(epoch["tokens"] for epoch in report["history"]),
                "kind": "stream" if path.name.startswith("stream-") else "pilot",
            }
        )
    atomic_json(directory / "index.json", {"checkpoints": entries})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("site/loop-study/v1/distillation-checkpoints")
    )
    args = parser.parse_args()
    publish_checkpoints(args.directory)


if __name__ == "__main__":
    main()

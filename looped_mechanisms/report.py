"""Refresh the public data file from committed local result records."""

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from looped_mechanisms.design import ITEMS_PER_SOURCE, LOCAL, NDIF, Condition, conditions, visits
from looped_mechanisms.inputs import COHORT_VERSION, SEED, SOURCES
from looped_mechanisms.results import atomic_json, result_dir


def summarize(records: list[dict[str, Any]], condition: Condition) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    layout = visits(condition)
    scores = [record for record in records if record["answer"] is not None]
    rms = [statistics.fmean(record["rms"][i] for record in records) for i in range(len(layout))]
    cosine = []
    for i in range(len(layout)):
        values = [record["cosine"][i] for record in records if record["cosine"][i] is not None]
        cosine.append(round(statistics.fmean(values), 6) if values else None)
    return {
        "count": len(records),
        "evaluated": len(scores),
        "correct": sum(record["prediction"] == record["answer"] for record in scores),
        "rms": [round(value, 6) for value in rms],
        "cosine": cosine,
        "ranks": sorted(record["rank"] for record in records),
    }


def build_report(root: Path) -> dict[str, Any]:
    backends: dict[str, list[dict[str, Any]]] = {NDIF: [], LOCAL: []}
    for backend in backends:
        for condition in conditions(backend):
            layout = visits(condition)
            directory = result_dir(root, backend, condition)
            records = []
            for path in directory.glob("*.json"):
                record = json.loads(path.read_text())
                vectors = path.with_suffix(".npz")
                if (
                    record["visits"] != len(layout)
                    or not vectors.is_file()
                    or vectors.stat().st_size != record["bytes"]
                ):
                    raise ValueError(f"Invalid activation artifact for {path}")
                records.append(record)
            backends[backend].append(
                {
                    "id": condition.name,
                    "checkpoint": condition.checkpoint,
                    "passes": condition.passes,
                    "visits": [asdict(visit) for visit in layout],
                    "sources": {
                        source.name: summarize(
                            [record for record in records if record["source"] == source.name],
                            condition,
                        )
                        for source in SOURCES
                    },
                }
            )
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cohort": COHORT_VERSION,
        "seed": SEED,
        "target_per_source": ITEMS_PER_SOURCE,
        "sources": [asdict(source) for source in SOURCES],
        "backends": backends,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("site/results.json"))
    parser.add_argument("--interval", type=int)
    args = parser.parse_args()
    while True:
        atomic_json(args.output, build_report(args.artifacts))
        if args.interval is None:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

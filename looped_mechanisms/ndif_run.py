"""Read all 48 visits from the two three-pass NDIF deployments."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from looped_mechanisms.design import LABELS, NDIF, Condition, conditions
from looped_mechanisms.inputs import SOURCES, Item, load_items, read_url
from looped_mechanisms.results import complete, positions, result_dir, save_result

TASK = "image-text-to-text"


def deployment_status(condition: Condition) -> str:
    if condition.passes != 3:
        raise ValueError("The hosted configuration supports three passes")
    inventory = json.loads(read_url("https://api.ndif.us/status"))
    deployments = [
        deployment
        for deployment in inventory["deployments"].values()
        if deployment.get("repo_id") == condition.model_id
        and deployment.get("task") == TASK
        and deployment.get("application_state") == "RUNNING"
        and deployment.get("revision") is None
    ]
    if not deployments:
        raise RuntimeError(f"No running {TASK} deployment for {condition.model_id}")
    for deployment in deployments:
        config = json.loads(deployment["config"])["text_config"]
        if (
            config["loop_num_passes"] != condition.passes
            or config["num_hidden_layers"] != 24
            or deployment["dtype"] != "bfloat16"
        ):
            raise RuntimeError(f"Unexpected hosted pass or layer count for {condition.model_id}")
    nnsight_version = version("nnsight")
    transformers_version = version("transformers")
    return f"ndif:{condition.model_id}:{TASK}:bfloat16:nnsight-{nnsight_version}:transformers-{transformers_version}"


def run_condition(condition: Condition, items: list[Item], root: Path) -> None:
    import numpy as np
    import torch  # ty: ignore[unresolved-import]
    from nnsight import CONFIG, TransformersModel, save  # ty: ignore[unresolved-import]

    if not CONFIG.API.APIKEY:
        raise RuntimeError("NDIF_API_KEY is missing. Load the project's ignored .env.")
    runtime = deployment_status(condition)
    directory = result_dir(root, NDIF, condition)
    todo = [item for item in items if not complete(directory, item, runtime)]
    if not todo:
        return
    print(f"Preparing {condition.model_id} for {len(todo)} hosted traces", flush=True)
    model = TransformersModel(condition.model_id, task=TASK)
    llm = model.model.language_model
    tokenizer = model.tokenizer or model.processor.tokenizer
    label_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LABELS]
    if any(len(tokens) != 1 for tokens in label_ids):
        raise ValueError("Choice labels must be one token")
    choices = [tokens[0] for tokens in label_ids]
    for item in todo:
        token_ids = tokenizer.encode(item.text, add_special_tokens=False)
        if len(token_ids) > 1024:
            raise ValueError(f"{item.identifier} exceeds the study token limit")
        positions_to_save = positions(len(token_ids))
        score = item.answer is not None
        with model.trace(
            {"input_ids": torch.tensor([token_ids])}, remote=True, use_cache=False
        ) as tracer:
            layers = save([])
            for layer in range(6):
                layers.append(
                    llm.layers[layer].output[0, positions_to_save].detach().to(torch.float16).cpu()
                )
            for _ in tracer.iter[:3]:
                for layer in range(6, 18):
                    layers.append(
                        llm.layers[layer]
                        .output[0, positions_to_save]
                        .detach()
                        .to(torch.float16)
                        .cpu()
                    )
            for layer in range(18, 24):
                layers.append(
                    llm.layers[layer].output[0, positions_to_save].detach().to(torch.float16).cpu()
                )
            selected = (
                model.output.logits[0, -1, choices].float().cpu() if score else torch.empty(0)
            ).save()
        array = np.stack([tensor.numpy() for tensor in layers])
        save_result(
            directory,
            condition,
            item,
            runtime,
            token_ids,
            array,
            selected.tolist() if score else None,
        )
        print(
            f"{condition.name} {item.source} rank={item.rank + 1} verified_visits={array.shape[0]}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--per-source", type=int, choices=range(1, 65), default=64)
    parser.add_argument("--source", choices=[source.name for source in SOURCES], action="append")
    parser.add_argument("--condition", choices=[condition.name for condition in conditions(NDIF)])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    items = (
        [Item("smoke", "Smoke", 0, 0, "The capital of France is", None)]
        if args.smoke
        else [item for item in load_items(args.cohort, args.source) if item.rank < args.per_source]
    )
    for condition in conditions(NDIF):
        if args.condition is None or condition.name == args.condition:
            run_condition(condition, items, args.artifacts)


if __name__ == "__main__":
    main()

"""Capture the fixed Inkling decoder's real layer input and output pairs on NDIF."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from looped_mechanisms.design import HIDDEN, Condition
from looped_mechanisms.inputs import Item, load_items
from looped_mechanisms.ndif_run import TASK, deployment_status
from looped_mechanisms.probe_study import validation_ids
from looped_mechanisms.results import atomic_json

TRAIN_LAYER = 10
MAX_TOKENS = 128
# Completed blocks are shared until the next boundary. The first post-loop
# layer receives the final pass's blocks, distinct from those in pass zero.
BLOCK_LAYER = (0, *([1] * 6), *([7] * 6), *([13] * 5), 18, *([19] * 5))
BLOCK_COUNT = {0: 0, 1: 1, 7: 2, 13: 3, 18: 3, 19: 4}


def training_split(layers: tuple[int, ...]) -> str:
    if layers == (TRAIN_LAYER,):
        return "train"
    return "train-" + "-".join(f"{layer:02d}" for layer in layers)


def collect(
    items: list[Item],
    root: Path,
    validation: bool,
    batch_size: int,
    train_layers: tuple[int, ...] = (TRAIN_LAYER,),
) -> None:
    import torch  # ty: ignore[unresolved-import]
    from nnsight import CONFIG, TransformersModel, save  # ty: ignore[unresolved-import]

    if not CONFIG.API.APIKEY:
        raise RuntimeError("Load the project's ignored .env with NDIF_API_KEY")
    if not sys.stdout.isatty():
        CONFIG.APP.REMOTE_LOGGING = False
    condition = Condition("fixed", 3)
    runtime = deployment_status(condition)
    layers = tuple(range(24)) if validation else train_layers
    # Several layers can share the same completed block inputs.
    groups = {BLOCK_LAYER[layer]: layer for layer in layers}
    representatives = (18,) if validation else tuple(sorted(groups.values()))
    output_layers = (17, 23) if validation else layers
    root.mkdir(parents=True, exist_ok=True)
    todo = []
    for item in items:
        marker = root / f"{item.identifier}.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            if (
                record["digest"] != item.digest
                or record["layers"] != list(layers)
                or record["runtime"] != runtime
                or marker.with_suffix(".npz").stat().st_size != record["bytes"]
            ):
                raise ValueError(f"Layer capture provenance mismatch in {marker}")
        else:
            todo.append(item)
    if not todo:
        return

    model = TransformersModel(condition.model_id, task=TASK)
    llm = model.model.language_model
    tokenizer = model.tokenizer or model.processor.tokenizer
    for offset in range(0, len(todo), batch_size):
        batch = todo[offset : offset + batch_size]
        tokens = [
            tokenizer.encode(item.text, add_special_tokens=False)[:MAX_TOKENS] for item in batch
        ]
        with model.session(remote=True):
            returned = save([])
            for token_ids in tokens:
                with model.trace(
                    input_ids=torch.tensor([token_ids]), use_cache=False, logits_to_keep=1
                ) as tracer:
                    active_inputs, targets, completed = [], [], []
                    for _ in tracer.iter[0]:
                        for layer in layers:
                            args = llm.layers[layer].inputs[0]
                            active_inputs.append(args[0][0].detach().to(torch.float16).cpu())
                            if layer in representatives:
                                blocks = args[1][0, :, -1:] if layer == 18 else args[1][0]
                                completed.append(blocks.detach().to(torch.float16).cpu())
                            if layer in output_layers:
                                targets.append(
                                    llm.layers[layer].output[0].detach().to(torch.float16).cpu()
                                )
                    captured = (torch.stack(active_inputs), torch.stack(targets), completed)
                returned.append(captured)
        for item, token_ids, (active, target, completed) in zip(
            batch, tokens, returned, strict=True
        ):
            length = len(token_ids)
            active_array, target_array = active.numpy(), target.numpy()
            expected = (len(layers), length, HIDDEN)
            if active_array.shape != expected or target_array.shape != (
                len(output_layers),
                length,
                HIDDEN,
            ):
                raise ValueError(f"Wrong layer tensor shape for {item.identifier}")
            if validation:
                # A decoder layer's output is its successor's active input, until
                # the loop reset. Block boundaries append that boundary's active input.
                targets_array = np.concatenate(
                    [active_array[1:18], target_array[:1], active_array[19:24], target_array[1:]]
                )
                first, second, third = (active_array[i, :, None, :] for i in (0, 6, 12))
                final_block = completed[0].numpy()
                arrays = {
                    "active": active_array,
                    "target": targets_array,
                    "blocks_0": first[:, :0],
                    "blocks_1": first,
                    "blocks_7": np.concatenate([first, second], axis=1),
                    "blocks_13": np.concatenate([first, second, third], axis=1),
                    "blocks_18": np.concatenate([first, second, final_block], axis=1),
                    "blocks_19": np.concatenate(
                        [first, second, final_block, active_array[18, :, None]], axis=1
                    ),
                }
            else:
                arrays = {
                    "active": active_array,
                    "target": target_array,
                    **{
                        f"blocks_{BLOCK_LAYER[layer]}": blocks.numpy()
                        for layer, blocks in zip(representatives, completed, strict=True)
                    },
                }
            for representative in {BLOCK_LAYER[layer] for layer in layers}:
                if arrays[f"blocks_{representative}"].shape != (
                    length,
                    BLOCK_COUNT[representative],
                    HIDDEN,
                ):
                    raise ValueError(
                        f"Wrong block tensor shape at {item.identifier}/{representative}"
                    )
            if any(not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError(f"Nonfinite layer data for {item.identifier}")
            path = root / f"{item.identifier}.npz"
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, allow_pickle=False, **arrays)
            temporary.replace(path)
            atomic_json(
                path.with_suffix(".json"),
                {
                    "layers": list(layers),
                    "digest": item.digest,
                    "runtime": runtime,
                    "tokens": length,
                    "bytes": path.stat().st_size,
                },
            )
            print(
                f"{root.name} {item.identifier} {length} tokens ({len(layers)} layers)", flush=True
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "validation"))
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/layer-io-v1"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--train-layer", type=int, choices=range(6, 18), action="append")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must be in [0, shards)")
    if args.split == "validation" and args.train_layer:
        parser.error("--train-layer is only used for training captures")
    items = load_items(args.cohort)
    held_out = validation_ids(items)
    is_validation = args.split == "validation"
    selected = [item for item in items if (item.identifier in held_out) == is_validation]
    layers = tuple(sorted(set(args.train_layer))) if args.train_layer else (TRAIN_LAYER,)
    split = args.split if is_validation else training_split(layers)
    collect(
        selected[: args.limit][args.shard :: args.shards],
        args.artifacts / split,
        is_validation,
        args.batch_size,
        layers,
    )


if __name__ == "__main__":
    main()

"""Train a teacher-layer-sized causal student and measure transfer by physical layer."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch  # ty: ignore[unresolved-import]
from numpy.typing import NDArray
from torch import Tensor, nn  # ty: ignore[unresolved-import]
from torch.nn import functional as F  # ty: ignore[unresolved-import]

from looped_mechanisms.design import HIDDEN
from looped_mechanisms.inputs import SEED, Item, load_items
from looped_mechanisms.layer_pairs import BLOCK_LAYER, MAX_TOKENS, TRAIN_LAYER, training_split
from looped_mechanisms.probe_study import validation_ids
from looped_mechanisms.results import atomic_json

TEACHER_PARAMETERS = 1_048_818_048
FULL_ATTENTION_TEACHER_PARAMETERS = 1_048_974_720
STUDENT_WIDTH = 221_184
BLOCKS = 4


class LayerStudent(nn.Module):
    """Causal attention over token sequences followed by a wide dense feed-forward network."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.input = nn.Linear((1 + BLOCKS) * HIDDEN, HIDDEN, bias=False)
        self.norm = nn.RMSNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, 3 * 768, bias=False)
        self.attention_output = nn.Linear(768, HIDDEN, bias=False)
        self.mlp_norm = nn.RMSNorm(HIDDEN)
        self.up = nn.Linear(HIDDEN, width, bias=False)
        self.down = nn.Linear(width, HIDDEN, bias=False)
        nn.init.zeros_(self.down.weight)
        frequency = torch.exp(-math.log(10_000) * torch.arange(0, 64, 2) / 64)
        self.register_buffer("frequency", frequency, persistent=False)

    # torch.nn.Module.__call__ dispatches to this method.
    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:  # noqa: V105
        batch, length = inputs.shape[:2]
        states = F.rms_norm(inputs.float(), (HIDDEN,)).flatten(-2)
        states = self.input(states)
        q, k, v = self.qkv(self.norm(states)).reshape(batch, length, 3, 12, 64).unbind(dim=2)
        angles = torch.arange(length, device=inputs.device)[:, None] * self.frequency
        cosine, sine = (
            angles.cos().to(q.dtype)[None, :, None],
            angles.sin().to(q.dtype)[None, :, None],
        )
        q0, q1 = q.chunk(2, dim=-1)
        k0, k1 = k.chunk(2, dim=-1)
        q = torch.cat((q0 * cosine - q1 * sine, q1 * cosine + q0 * sine), dim=-1)
        k = torch.cat((k0 * cosine - k1 * sine, k1 * cosine + k0 * sine), dim=-1)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        states = states + self.attention_output(
            attended.transpose(1, 2).reshape(batch, length, 768)
        )
        update = self.down(F.gelu(self.up(self.mlp_norm(states[mask])), approximate="tanh"))
        return inputs[:, :, 0][mask].float() + update.float()


@dataclass(frozen=True)
class Sequence:
    active: NDArray[np.float16]
    blocks: NDArray[np.float16]
    target: NDArray[np.float16]

    @property
    def length(self) -> int:
        return len(self.target)


def read_sequence(path: Path, layer: int, captured_layers: list[int]) -> Sequence:
    index = captured_layers.index(layer)
    with np.load(path) as data:
        return Sequence(
            data["active"][index], data[f"blocks_{BLOCK_LAYER[layer]}"], data["target"][index]
        )


def files_for(items: list[Item], root: Path, captured_layers: list[int]) -> list[Path]:
    files = []
    for item in items:
        marker = root / f"{item.identifier}.json"
        path = marker.with_suffix(".npz")
        meta = json.loads(marker.read_text())
        if (
            meta["digest"] != item.digest
            or meta["layers"] != captured_layers
            or path.stat().st_size != meta["bytes"]
        ):
            raise ValueError(f"Invalid layer-pair capture for {item.identifier}")
        files.append(path)
    return files


def batch_sequences(
    examples: list[Sequence], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    length = max(example.length for example in examples)
    inputs = np.zeros((len(examples), length, 1 + BLOCKS, HIDDEN), dtype=np.float16)
    target = np.zeros((len(examples), length, HIDDEN), dtype=np.float16)
    mask = np.zeros((len(examples), length), dtype=bool)
    for index, example in enumerate(examples):
        tokens = example.length
        inputs[index, :tokens, 0] = example.active
        inputs[index, :tokens, 1 : 1 + example.blocks.shape[1]] = example.blocks
        target[index, :tokens] = example.target
        mask[index, :tokens] = True
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(mask).to(device),
    )


def groups(examples: list[Sequence], token_budget: int) -> list[list[Sequence]]:
    batches: list[list[Sequence]] = []
    batch: list[Sequence] = []
    for example in sorted(examples, key=lambda example: example.length, reverse=True):
        if not batch or batch[0].length * (len(batch) + 1) <= token_budget:
            batch.append(example)
        else:
            batches.append(batch)
            batch = [example]
    if batch:
        batches.append(batch)
    return batches


@torch.no_grad()
def evaluate(
    student: LayerStudent, examples: list[Sequence], token_budget: int, device: torch.device
) -> dict[str, float | int]:
    stats = torch.zeros(6, dtype=torch.float64, device=device)
    count = 0
    for batch in groups(examples, token_budget):
        inputs, targets, mask = batch_sequences(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted = student(inputs, mask)
        target = targets[mask].float()
        baseline = inputs[:, :, 0][mask].float()
        error = (predicted - target).square()
        copy_error = (baseline - target).square()
        stats += torch.stack(
            (
                error.sum(),
                copy_error.sum(),
                target.square().sum(),
                F.cosine_similarity(predicted, target, dim=-1).sum(),
                F.cosine_similarity(baseline, target, dim=-1).sum(),
                target.numel() * target.new_ones(()),
            )
        )
        count += sum(example.length for example in batch)
    error, copy_error, energy, cosine, copy_cosine, coordinates = stats.tolist()
    return {
        "tokens": count,
        "rmse": math.sqrt(error / coordinates),
        "copy_rmse": math.sqrt(copy_error / coordinates),
        "normalized_rmse": math.sqrt(error / energy),
        "copy_normalized_rmse": math.sqrt(copy_error / energy),
        "cosine": cosine / count,
        "copy_cosine": copy_cosine / count,
        "loss_ratio": error / max(copy_error, 1e-30),
    }


def train(
    student: LayerStudent,
    data: list[Sequence],
    epochs: int,
    token_budget: int,
    device: torch.device,
    rate: float,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.Adafactor(student.parameters(), lr=rate, foreach=False)
    rng = np.random.default_rng(SEED)
    batches = groups(data, token_budget)
    if not batches:
        raise ValueError("Distillation requires training sequences")
    epoch_tokens = sum(example.length for example in data)
    history = []
    for epoch in range(1, epochs + 1):
        epoch_error = 0.0
        for batch_index in rng.permutation(len(batches)):
            batch = batches[batch_index]
            inputs, targets, mask = batch_sequences(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predicted = student(inputs, mask)
                objective = F.mse_loss(predicted, targets[mask].float())
            objective.backward()
            optimizer.step()
            tokens = sum(example.length for example in batch)
            epoch_error += objective.item() * tokens
        rmse = math.sqrt(epoch_error / epoch_tokens)
        if not math.isfinite(rmse):
            raise ValueError("Student training diverged")
        history.append({"epoch": epoch, "rmse": rmse, "tokens": epoch_tokens})
        print(
            f"epoch {epoch}/{epochs}: training RMSE {rmse:.6f}, tokens {epoch_tokens:,}", flush=True
        )
    optimizer.zero_grad(set_to_none=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path("data/cohort"))
    parser.add_argument("--pairs", type=Path, default=Path("artifacts/layer-io-v1"))
    parser.add_argument("--train-layer", type=int, choices=range(6, 18), default=TRAIN_LAYER)
    parser.add_argument("--capture-layer", type=int, choices=range(6, 18), action="append")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("both", "train", "evaluate"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=384)
    parser.add_argument("--rate", type=float, default=0.005)
    args = parser.parse_args()
    if args.epochs < 1 or args.token_budget < 1 or args.rate <= 0:
        parser.error("epochs, token budget and learning rate must be positive")
    train_layer = args.train_layer
    captured_layers = tuple(sorted(set(args.capture_layer or [train_layer])))
    if train_layer not in captured_layers:
        parser.error("the training layer must appear in --capture-layer")
    suffix = "" if train_layer == TRAIN_LAYER else f"-{train_layer:02d}"
    weights = args.weights or Path(f"artifacts/distillation/student{suffix}.pt")
    output = args.output or Path(f"site/loop-study/v1/distillation{suffix}.json")
    teacher_parameters = (
        FULL_ATTENTION_TEACHER_PARAMETERS if train_layer % 6 == 5 else TEACHER_PARAMETERS
    )
    items = load_items(args.cohort)
    held_out = validation_ids(items)
    train_items = [item for item in items if item.identifier not in held_out]
    val_items = [item for item in items if item.identifier in held_out]
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if args.phase == "evaluate":
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        if checkpoint["width"] != STUDENT_WIDTH or checkpoint["train_layer"] != train_layer:
            raise ValueError("Checkpoint belongs to another distillation condition")
        with torch.device("meta"):
            student = LayerStudent(STUDENT_WIDTH)
        student.load_state_dict(checkpoint["model"], assign=True)
        # The rotary frequencies do not live in the serialization.
        frequency = torch.exp(-math.log(10_000) * torch.arange(0, 64, 2) / 64)
        student.frequency = frequency
        student.to(device)
        history = checkpoint["history"]
        train_baseline = checkpoint["train_copy_rmse"]
        del checkpoint
    else:
        train_files = files_for(
            train_items, args.pairs / training_split(captured_layers), list(captured_layers)
        )
        with torch.device(device):
            student = LayerStudent(STUDENT_WIDTH)
        train_data = [
            read_sequence(path, train_layer, list(captured_layers)) for path in train_files
        ]
        baseline_error = sum(
            np.square(sequence.target.astype(np.float32) - sequence.active).sum(dtype=np.float64)
            for sequence in train_data
        )
        train_baseline = math.sqrt(baseline_error / (HIDDEN * sum(s.length for s in train_data)))
        print(f"Training copy RMSE {train_baseline:.6f}", flush=True)
        student.train()
        history = train(student, train_data, args.epochs, args.token_budget, device, args.rate)
        del train_data
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_suffix(".tmp")
        torch.save(
            {
                "model": student.state_dict(),
                "width": STUDENT_WIDTH,
                "train_layer": train_layer,
                "history": history,
                "train_copy_rmse": train_baseline,
            },
            temporary,
        )
        temporary.replace(weights)
        print(f"Saved student to {weights}", flush=True)
        if args.phase == "train":
            return
    student_parameters = sum(parameter.numel() for parameter in student.parameters())
    if student_parameters < teacher_parameters:
        raise ValueError("Student capacity is smaller than the full teacher layer")
    student.eval()
    val_files = files_for(val_items, args.pairs / "validation", list(range(24)))
    loaded = []
    for path in val_files:
        with np.load(path) as saved:
            loaded.append({name: saved[name] for name in saved.files})
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for layer in range(24):
        data = [
            Sequence(
                saved["active"][layer],
                saved[f"blocks_{BLOCK_LAYER[layer]}"],
                saved["target"][layer],
            )
            for saved in loaded
        ]
        score = evaluate(student, data, args.token_budget, device)
        results.append({"layer": layer, **score})
        atomic_json(
            output,
            {
                "teacher_condition": "NDIF · Inkling 2 Mini · fixed 3 passes",
                "teacher_parameters": teacher_parameters,
                "student_parameters": student_parameters,
                "train_layer": train_layer,
                "max_prompt_tokens": MAX_TOKENS,
                "train_prompts": len(train_items),
                "validation_prompts": len(val_items),
                "train_copy_rmse": train_baseline,
                "history": history,
                "layers": results,
            },
        )
        print(
            f"layer {layer}: RMSE {score['rmse']:.5f} (copy {score['copy_rmse']:.5f})",
            flush=True,
        )


if __name__ == "__main__":
    main()

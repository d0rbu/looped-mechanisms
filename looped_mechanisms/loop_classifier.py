"""Train and evaluate a prompt-grouped neural loop classifier."""

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from looped_mechanisms.design import HIDDEN, POSITIONS
from looped_mechanisms.inputs import SEED

BASE_WIDTH = 512
EPOCHS = 20


# The functional form keeps Vulture from treating the JSON fields as unused variables.
LoopScore = TypedDict(  # noqa: UP013
    "LoopScore",
    {
        "layer": int,
        "accuracy": float,
        "correct": int,
        "examples": int,
        "confusion": list[list[int]],
    },
)


def parameter_count(width: int, loops: int) -> int:
    return (HIDDEN + 1 + loops) * width + loops


def fit(
    training: NDArray[np.float16],
    validation: NDArray[np.float16],
    *,
    layer: int,
    width: int,
    batch_size: int,
    device_name: str,
    checkpoint: Path,
    lineage: dict[str, object],
) -> LoopScore:
    import torch  # ty: ignore[unresolved-import]
    from torch import nn  # ty: ignore[unresolved-import]

    loops = training.shape[1]
    expected = (loops, POSITIONS, HIDDEN)
    if (
        not len(training)
        or not len(validation)
        or training.shape[1:] != expected
        or validation.shape[1:] != expected
    ):
        raise ValueError("Expected one [loop, token position, hidden] tensor per prompt")
    torch.set_float32_matmul_precision("high")
    device = torch.device(device_name)
    loop_order = torch.arange(loops, device=device).repeat_interleave(POSITIONS)
    train_labels = loop_order.repeat(len(training))
    val_labels = loop_order.repeat(len(validation))
    train_rows = torch.from_numpy(training.reshape(-1, HIDDEN)).to(
        device=device, dtype=torch.float32
    )
    val_rows = torch.from_numpy(validation.reshape(-1, HIDDEN)).to(
        device=device, dtype=torch.float32
    )
    scale, mean = torch.std_mean(train_rows, dim=0, correction=0)
    scale = scale.clamp_min(1e-6)
    train_rows.sub_(mean).div_(scale)
    val_rows.sub_(mean).div_(scale)
    torch.manual_seed(SEED + layer)
    model = nn.Sequential(nn.Linear(HIDDEN, width), nn.GELU(), nn.Linear(width, loops)).to(device)
    if sum(p.numel() for p in model.parameters()) != parameter_count(width, loops):
        raise ValueError("Unexpected classifier parameter count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    for _ in range(EPOCHS):
        for indices in torch.randperm(len(train_rows), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_rows[indices]), train_labels[indices])
            loss.backward()
            optimizer.step()
        schedule.step()
    if not all(parameter.isfinite().all() for parameter in model.parameters()):
        raise ValueError(f"Non-finite classifier parameters for layer {layer}")
    model.eval()
    with torch.no_grad():
        predicted = torch.cat([model(rows).argmax(dim=1) for rows in val_rows.split(batch_size)])
        confusion = torch.bincount(loops * val_labels + predicted, minlength=loops**2).reshape(
            loops, loops
        )
        correct = int(confusion.diagonal().sum().item())
    score: LoopScore = {
        "layer": layer,
        "accuracy": correct / len(predicted),
        "correct": correct,
        "examples": len(predicted),
        "confusion": confusion.tolist(),
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "hidden_units": width,
            "head_dim": loops,
            "epochs": EPOCHS,
            "score": score,
            **lineage,
        },
        temporary,
    )
    temporary.replace(checkpoint)
    return score

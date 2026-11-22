"""Study conditions and activation indexing shared by both backends."""

from dataclasses import dataclass

HIDDEN = 2304
POSITIONS = 4
ITEMS_PER_SOURCE = 64
NDIF = "ndif"
LOCAL = "llama.cpp-iq4"
LABELS = "ABCD"


@dataclass(frozen=True)
class Condition:
    checkpoint: str
    passes: int

    def __post_init__(self) -> None:
        if self.checkpoint not in {"fixed", "variable"}:
            raise ValueError(f"Unknown checkpoint {self.checkpoint}")
        if not 1 <= self.passes <= 16 or (self.checkpoint == "fixed" and self.passes != 3):
            raise ValueError(f"Pass count {self.passes} unavailable for {self.checkpoint}")

    @property
    def name(self) -> str:
        return f"{self.checkpoint}-{self.passes:02d}"

    @property
    def model_id(self) -> str:
        if self.checkpoint == "fixed":
            return "thinkingmachines/Inkling-2-Mini"
        return "thinkingmachines/TML-d24-varloop-research"


@dataclass(frozen=True)
class Visit:
    layer: int
    pass_number: int | None
    slot: int


def visits(condition: Condition) -> tuple[Visit, ...]:
    prelude = [Visit(layer, None, layer) for layer in range(6)]
    core = [
        Visit(
            layer,
            pass_number,
            layer if pass_number == 1 else 24 + 12 * (pass_number - 2) + layer - 6,
        )
        for pass_number in range(1, condition.passes + 1)
        for layer in range(6, 18)
    ]
    coda = [Visit(layer, None, layer) for layer in range(18, 24)]
    return tuple(prelude + core + coda)


def conditions(backend: str) -> list[Condition]:
    passes = [3] if backend == NDIF else [3, 1, 2, *range(4, 17)]
    return [Condition("fixed", 3), *(Condition("variable", p) for p in passes)]

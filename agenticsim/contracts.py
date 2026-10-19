"""Small executable subset of the proposed model contract, not an AAS schema."""
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Protocol


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str


@dataclass(frozen=True)
class Parameter:
    name: str
    semantic_id: str
    unit: str
    lower: float
    upper: float

    def validate(self, quantity: Quantity) -> None:
        if quantity.unit != self.unit:
            raise ValueError(f"{self.name}: expected {self.unit}, got {quantity.unit}; explicit conversion required")
        if not isfinite(quantity.value) or not self.lower <= quantity.value <= self.upper:
            raise ValueError(f"{self.name}: outside validated interval [{self.lower}, {self.upper}]")


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    version: str
    domain: str
    backend: str
    fidelity: str
    capabilities: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    assumptions: tuple[str, ...]

    def validate(self, values: dict[str, Quantity]) -> None:
        expected = {p.name for p in self.parameters}
        if set(values) != expected:
            raise ValueError(f"parameter names must be exactly {sorted(expected)}")
        for parameter in self.parameters:
            parameter.validate(values[parameter.name])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Request:
    parameters: dict[str, Quantity]
    times_s: tuple[float, ...]

    def validate_times(self) -> None:
        if not self.times_s or any(not isfinite(t) or t < 0 for t in self.times_s):
            raise ValueError("times must be finite, nonnegative and nonempty")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError("times must be strictly increasing")


@dataclass(frozen=True)
class Trajectory:
    times_s: tuple[float, ...]
    values: tuple[float, ...]
    observable_id: str
    unit: str


class SimulationBackend(Protocol):
    card: ModelCard

    def simulate(self, request: Request) -> Trajectory: ...


def require_capabilities(card: ModelCard, required: set[str]) -> None:
    missing = required - set(card.capabilities)
    if missing:
        raise ValueError(f"{card.model_id} lacks capabilities: {sorted(missing)}")

"""Instances, scenarios and decision spaces — layer L1/L3 of ``PLAN.md``.

An **instance** carries entities and parameters but no decisions and no solver.
A **scenario** is a named perturbation of one. A **decision space** says what
can be decided and how to move around in it — that is what lets a generic
search solver operate on a domain it knows nothing about.

Deliberately thin (architecture constraint A6): the shared layer defines only
*protocols*, never domain content. A shopfloor decision is a weight vector, an
intralogistics decision is a configuration record; nothing here tries to
unify their internals.
"""

from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable

__all__ = ["Instance", "Scenario", "DecisionSpace", "VectorSpace"]


@runtime_checkable
class Instance(Protocol):
    """One concrete problem instance."""

    def id(self) -> str:
        ...


@runtime_checkable
class Scenario(Protocol):
    """A named perturbation of a base instance."""

    name: str

    def apply(self, base: Any) -> Any:
        ...


@runtime_checkable
class DecisionSpace(Protocol):
    """What can be decided, and how to move within it.

    Two methods are enough for generic search: somewhere to start, and a
    neighbourhood move. Solvers that need more (gradients, algebraic
    structure) get it from the world model's capabilities instead.
    """

    def initial(self) -> Any:
        """A starting decision. Need not be feasible."""
        ...

    def perturb(self, decision: Any, rng: random.Random) -> Any:
        """A neighbour of ``decision``. Must return a new object."""
        ...


class VectorSpace:
    """Decision space for a fixed-length vector of reals.

    Covers any policy parameterised as weights — e.g. GradientShop's weight
    vector over named dispatching rules.
    """

    def __init__(
        self,
        dimension: int,
        *,
        start: float = 0.05,
        step: float = 0.25,
        names: list[str] | None = None,
    ) -> None:
        # NB: an all-zero start makes every score equal, which is a symmetric
        # saddle with zero gradient. Default to a small non-zero vector.
        self.dimension = dimension
        self.start = start
        self.step = step
        self.names = names or [f"x{i}" for i in range(dimension)]

    def initial(self) -> list[float]:
        return [self.start] * self.dimension

    def perturb(self, decision: list[float], rng: random.Random) -> list[float]:
        return [x + rng.gauss(0.0, self.step) for x in decision]

    def vectorize(self, decision: list[float]) -> list[float]:
        """Expose the numeric embedding required by surrogate optimisers."""
        if len(decision) != self.dimension:
            raise ValueError(
                f"expected a {self.dimension}-component decision, got {len(decision)}"
            )
        return [float(value) for value in decision]

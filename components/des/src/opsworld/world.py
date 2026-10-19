"""World models and what they can be asked for.

The framework's central abstraction (``PLAN.md`` Axiom 1): a **world model**
states how the world responds to a decision. It is *not* a solver. The same
world model admits many solvers; the same solver applies to many world models.

Rather than one fat interface, world models are graded by **capability**
(Axiom 4). A discrete-event simulation can hand out gradients but not an
algebraic form; a MILP can hand out an algebraic form and duals but no
trajectory. Forcing both through one interface would produce a wall of
``NotImplementedError`` and the interface would become a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Capability",
    "Sensitivity",
    "SensitivityKind",
    "Evaluation",
    "WorldModel",
    "DifferentiableWorldModel",
    "AlgebraicWorldModel",
    "SequentialWorldModel",
    "penalised_objective",
]


class Capability(StrEnum):
    """What a world model is able to answer. Keys match ``PLAN.md`` §1.6."""

    EVALUATE = "C0"            # evaluate(decision) -> Evaluation
    SEQUENTIAL = "C1"          # reset/step: a genuine MDP
    PATHWISE_GRADIENT = "C2a"  # d objective / d decision
    DUALS = "C2b"              # d objective / d constraint right-hand side
    ALGEBRAIC = "C3"           # an algebraic (e.g. MILP) form for exact solvers


class SensitivityKind(StrEnum):
    """Which first-order object this is. The distinction is not cosmetic.

    ``PATHWISE`` differentiates the objective with respect to the **decision
    variables** — it answers *"which decision should I change?"* and can be
    stepped along.

    ``DUAL`` differentiates with respect to a constraint's **right-hand side**
    — it answers *"which resource limit is binding, and what would relaxing it
    buy?"*. It lives in a different space and is **not** a search direction.

    Conflating the two is decision record D10 in ``PLAN.md``; here the
    distinction is enforced rather than merely documented.
    """

    PATHWISE = "pathwise"
    DUAL = "dual"


@dataclass
class Sensitivity:
    """First-order information attached to an :class:`Evaluation`."""

    kind: SensitivityKind
    names: list[str]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names/values length mismatch: {len(self.names)} vs {len(self.values)}"
            )

    def descent_direction(self) -> list[float]:
        """Direction that decreases the objective — pathwise gradients only."""
        if self.kind is not SensitivityKind.PATHWISE:
            raise TypeError(
                "a dual variable is a price on a constraint's right-hand side, "
                "not a direction in decision space; only PATHWISE sensitivities "
                "can be stepped along (PLAN.md D10)."
            )
        return [-v for v in self.values]

    def ranked(self) -> list[tuple[str, float]]:
        """Entries sorted most-negative first (most improving, for pathwise)."""
        return sorted(zip(self.names, self.values), key=lambda nv: nv[1])

    def as_text(self) -> str:
        """Verbalisation — the channel that grounds an LLM's next proposal."""
        if self.kind is SensitivityKind.PATHWISE:
            header = ("Sensitivity of the objective to each decision component "
                      "(negative = increasing it helps):")
            verb = lambda v: "LOWERS" if v < 0 else "RAISES"  # noqa: E731
            body = [f"  - {n:<28} d/dx={v:+.4f} -> increasing it {verb(v)} the objective"
                    for n, v in self.ranked()]
        else:
            header = ("Shadow prices — objective change per unit of relaxation of "
                      "each binding constraint:")
            body = [f"  - {n:<28} price={v:+.4f}"
                    for n, v in sorted(self.ranked(), key=lambda nv: -abs(nv[1]))]
        return "\n".join([header, *body])


@dataclass
class Evaluation:
    """What a world model reports about one decision.

    ``violations`` maps a constraint family to the **total violated magnitude**;
    empty or all-zero means feasible. Magnitudes rather than booleans, so the
    same evaluation can drive penalty-based search.

    ``provenance`` records which world model — and which projection of the
    dynamics — produced the numbers (architecture constraint A2). A static
    fluid model and a dynamic simulation of the same system are both valid
    world models and must be distinguishable in their output.
    """

    objective: float
    kpis: dict[str, float] = field(default_factory=dict)
    violations: dict[str, float] = field(default_factory=dict)
    sensitivity: Sensitivity | None = None
    provenance: str = "unknown"

    def total_violation(self) -> float:
        return sum(self.violations.values())

    def is_feasible(self, tol: float = 1e-6) -> bool:
        return self.total_violation() <= tol

    def violated(self, tol: float = 1e-6) -> dict[str, float]:
        return {k: v for k, v in self.violations.items() if v > tol}

    def summary(self) -> str:
        status = "feasible" if self.is_feasible() else "INFEASIBLE"
        lines = [f"[{self.provenance}] objective={self.objective:.6f}  ({status})"]
        for name, value in sorted(self.kpis.items()):
            lines.append(f"    kpi  {name:<24s} {value:12.4f}")
        for name, value in sorted(self.violated().items()):
            lines.append(f"    !!   {name:<24s} {value:12.6f}")
        return "\n".join(lines)

    def as_text(self) -> str:
        """Full verbalisation, including the first-order critique if present."""
        parts = [self.summary()]
        if self.sensitivity is not None:
            parts.append(self.sensitivity.as_text())
        return "\n".join(parts)


def penalised_objective(evaluation: Evaluation, penalty: float = 1e6) -> float:
    """Scalarise objective + constraint violation into one number to minimise.

    This is what lets a single generic solver drive both a constrained world
    model and an unconstrained one: with no violations the score *is* the
    objective.
    """
    return evaluation.objective + penalty * evaluation.total_violation()


# ----------------------------------------------------------------------
# protocols
# ----------------------------------------------------------------------
@runtime_checkable
class WorldModel(Protocol):
    """C0 — the minimum every world model must provide."""

    def evaluate(self, decision: Any) -> Evaluation:
        """Consequences of ``decision``. Must not mutate the decision."""
        ...

    def capabilities(self) -> set[Capability]:
        ...

    def decision_space(self) -> Any:
        """The :class:`~opsworld.problem.DecisionSpace` this model is defined over."""
        ...


@runtime_checkable
class DifferentiableWorldModel(WorldModel, Protocol):
    """C2a — exposes the gradient of the objective w.r.t. the decision."""

    def gradient(self, decision: Any) -> Sensitivity:
        ...


@runtime_checkable
class AlgebraicWorldModel(WorldModel, Protocol):
    """C3 (+C2b) — exposes an algebraic form, hence exact solvers and duals."""

    def solve_exact(self) -> tuple[Any, float]:
        ...

    def duals(self) -> Sensitivity:
        ...


@runtime_checkable
class SequentialWorldModel(WorldModel, Protocol):
    """C1 — a genuine MDP: state evolves and actions are taken repeatedly."""

    def reset(self) -> Any:
        ...

    def step(self, state: Any, action: Any) -> tuple[Any, float, bool]:
        ...

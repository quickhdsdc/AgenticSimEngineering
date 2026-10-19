"""Sizing rules — the intralogistics decision vocabulary.

The direct counterpart of :mod:`gradientshop.shopfloor.rules`. There, classic
dispatching rules (SPT, EDD, WSPT, ...) are stacked as columns of a feature
matrix, and a policy is a weight vector ``theta`` over those *named* rules:
``score = priority_matrix @ theta``. Here the same construction is applied to
material supply — each rule maps item attributes to a score meaning "give this
item a **larger** load carrier", and a policy blends them.

Why this matters for the joint work
-----------------------------------
The source dissertation's *Regelableitung* derives interpretable rules from a
family of MILP optima, with a Random Forest over item attributes
("Verbauhäufigkeit > 2.5/d, Wert < 5 EUR, Größe < 0.3 m³ -> ..."). In
reinforcement-learning terms that is behaviour cloning from an exact-solver
expert.

A weight vector over named rules is the same idea with two advantages:

* it is **differentiable**, so it can be fitted on the *actual cost* rather than
  on a classification of optimal labels — no expert solutions are needed at all,
  though it can imitate them if they exist;
* it stays **interpretable** in the same sense her rules are, because the
  weights sit on named heuristics rather than on tree splits.

And it yields the number her chapter needs: *how much more expensive is the
interpretable rule than the item-by-item optimum?*

What the policy is trained on, and why
-------------------------------------
The gradient comes from the **fluid model** (``moves = lambda / Q``, average
inventory ``Q/2 + safety stock``), not from backpropagating through the
simulation. That is a deliberate choice, and it is the one Phases 3 and 4
argue for:

* Phase 3 measured exactly when the fluid projection is trustworthy — it is
  *exact* at zero variability and degrades with the demand CV — so the
  surrogate's error budget is known rather than hoped for.
* Phase 4 measured what happens if you differentiate the simulation instead:
  the relaxed reorder trigger sits inside a feedback loop, and unless the
  trajectory stays fully away from the threshold, the gradient is destroyed
  (0.9% of near-threshold periods was enough to take the cosine against a
  finite difference from +1.00 to +0.09). See
  :func:`~intralogistics.differentiable.saturation_fraction`.

So: **train on the smooth surrogate, score on the simulation**, and report the
gap between them. :meth:`SizingPolicyWorld.simulated_cost` is the honest
scoring path; :meth:`SizingPolicyWorld.evaluate` is the trainable one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from opsworld.problem import VectorSpace
from opsworld.world import Capability, Evaluation, Sensitivity, SensitivityKind

from .differentiable import replenishment_cost

__all__ = [
    "Items",
    "SIZING_RULES",
    "rule_names",
    "sizing_score",
    "sizing_matrix",
    "capacity_from_scores",
    "SizingPolicyWorld",
]


@dataclass
class Items:
    """A batch of items competing for storage. All tensors are ``[N]``."""

    demand_rate: Tensor      # units consumed per period
    unit_value: Tensor       # EUR per unit -- inventory carrying cost driver
    unit_area: Tensor        # footprint per unit
    lead_time: Tensor        # replenishment lead time, periods
    demand_cv: Tensor        # coefficient of variation of demand

    def __len__(self) -> int:
        return int(self.demand_rate.shape[0])

    def safety_stock(self, service_factor: float = 1.65) -> Tensor:
        sigma = self.demand_cv * self.demand_rate
        return service_factor * sigma * torch.sqrt(self.lead_time)

    def reorder_point(self, service_factor: float = 1.65) -> Tensor:
        return self.demand_rate * self.lead_time + self.safety_stock(service_factor)


#: name -> (items -> score); a *larger* score means "give it a larger carrier".
SIZING_RULES: dict[str, Callable[[Items], Tensor]] = {
    # Fast movers are reordered constantly; bigger bins cut the move count.
    "HIGH_TURNOVER": lambda it: it.demand_rate,
    # Cheap stock is cheap to hold, so batching it up costs little.
    "LOW_VALUE": lambda it: -it.unit_value,
    # Small parts waste less floor space when batched.
    "SMALL_PART": lambda it: -it.unit_area,
    # Long resupply favours holding more per delivery.
    "LONG_LEAD": lambda it: it.lead_time,
    # Volatile demand needs a deeper buffer.
    "VOLATILE": lambda it: it.demand_cv,
    # Textbook economic order quantity, sqrt(2 * D * S / H) up to constants.
    "EOQ": lambda it: torch.sqrt(it.demand_rate / it.unit_value.clamp_min(1e-8)),
    # Value density -- the classic ABC axis: expensive per m2 stays small.
    "VALUE_DENSITY": lambda it: -(it.unit_value / it.unit_area.clamp_min(1e-8)),
    # Bias term, the analogue of FIFO: no attribute, just a level.
    "FLAT": lambda it: torch.ones_like(it.demand_rate),
}


def rule_names() -> list[str]:
    return list(SIZING_RULES)


def sizing_score(name: str, items: Items) -> Tensor:
    try:
        rule = SIZING_RULES[name]
    except KeyError:
        raise KeyError(f"unknown sizing rule {name!r}; known: {rule_names()}") from None
    return rule(items)


def sizing_matrix(
    items: Items, rules: list[str] | None = None, *, standardize: bool = True
) -> tuple[list[str], Tensor]:
    """Stack chosen rules into an ``[N, F]`` feature matrix, one column per rule.

    Columns are standardised so a policy weights rules on a comparable scale —
    same convention as :func:`gradientshop.shopfloor.rules.priority_matrix`.
    ``FLAT`` is left alone, since standardising a constant is undefined.
    """
    names = rules if rules is not None else rule_names()
    columns = [sizing_score(name, items) for name in names]
    matrix = torch.stack(columns, dim=1)
    if standardize:
        mean = matrix.mean(dim=0, keepdim=True)
        std = matrix.std(dim=0, keepdim=True)
        constant = std <= 1e-8
        matrix = torch.where(constant, matrix, (matrix - mean) / std.clamp_min(1e-8))
    return names, matrix


def capacity_from_scores(
    scores: Tensor, minimum: float, maximum: float, *, spread: float = 1.0
) -> Tensor:
    """Map policy scores onto the admissible carrier-capacity range, smoothly.

    A squashing map rather than a menu lookup: it is monotone (so the rule
    reads the way its name promises), bounded (so no policy proposes an
    impossible carrier), and differentiable (so the policy trains on cost).
    Discretising onto a real catalogue is a rounding step at the end, and is
    deliberately not part of the training signal.
    """
    if maximum <= minimum:
        raise ValueError("maximum capacity must exceed the minimum")
    return minimum + (maximum - minimum) * torch.sigmoid(scores / spread)


class SizingPolicyWorld:
    """World model whose *decision* is a weight vector over named sizing rules.

    Capabilities ``C0`` + ``C2a``: it evaluates a policy and hands back the
    gradient with respect to the rule weights, exactly as
    :class:`gradientshop.world.DispatchWorld` does for dispatching rules. The
    same gradient solver and the same LLM critique channel therefore drive both.
    """

    def __init__(
        self,
        items: Items,
        demand: Tensor,
        *,
        rules: list[str] | None = None,
        min_capacity: float = 5.0,
        max_capacity: float = 80.0,
        spread: float = 1.0,
        lead_time: int = 3,
        service_factor: float = 1.65,
        tau: float = 0.02,
        **cost_kwargs,
    ) -> None:
        if demand.shape[0] != len(items):
            raise ValueError("demand and items disagree on the number of items")
        self.items = items
        self.demand = demand
        self.rule_names, self.features = sizing_matrix(items, rules)
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.spread = spread
        self.lead_time = lead_time
        # Phase 4: use the SOFT path at small tau, never straight-through here.
        # With hard=True the backward pass carries sigma'/tau, which at small tau
        # is huge and meaningless -- following it *increases* the cost. The soft
        # path at small tau saturates instead, and what survives is the exact
        # pathwise gradient (verified there to a relative 4e-11).
        self.tau = tau
        self.service_factor = service_factor
        self.reorder = items.reorder_point(service_factor)
        self.cost_kwargs = cost_kwargs
        self.storage_rate = cost_kwargs.get("storage_rate", 1.0)
        self.area_fixed = cost_kwargs.get("area_fixed", 1.0 / 3.0)
        self.area_per_unit = cost_kwargs.get("area_per_unit", 1.0 / 15.0)
        self.move_cost = cost_kwargs.get("move_cost", 12.0)
        self._space = VectorSpace(
            len(self.rule_names), start=0.05, step=0.3, names=self.rule_names
        )

    @property
    def provenance(self) -> str:
        return f"intralogistics/sizing-policy ({len(self.rule_names)} named rules)"

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE, Capability.PATHWISE_GRADIENT}

    def decision_space(self) -> VectorSpace:
        return self._space

    def capacities(self, theta: list[float] | Tensor) -> Tensor:
        """The per-item carrier capacities this policy prescribes."""
        weights = torch.as_tensor(theta, dtype=self.features.dtype)
        return capacity_from_scores(
            self.features @ weights, self.min_capacity, self.max_capacity,
            spread=self.spread,
        )

    def fluid_cost(self, capacity: Tensor) -> Tensor:
        """Smooth surrogate cost — the training signal.

        ``moves = lambda / Q`` is flow conservation (constraint (17) of the
        MILP), verified against the simulation to 0.24% in Phase 3;
        ``avg_inventory = Q/2 + safety stock`` is the sawtooth mean, whose
        residual Phase 3 measured to be exactly constant in ``Q``.
        """
        avg_inventory = capacity / 2.0 + self.items.safety_stock(self.service_factor)
        moves = self.items.demand_rate / capacity
        carriers = avg_inventory / capacity
        area = self.area_fixed + self.area_per_unit * capacity
        return (self.storage_rate * carriers * area + self.move_cost * moves).sum()

    def _cost(self, weights: Tensor) -> Tensor:
        capacity = capacity_from_scores(
            self.features @ weights, self.min_capacity, self.max_capacity,
            spread=self.spread,
        )
        return self.fluid_cost(capacity)

    def simulated_cost(self, decision: list[float] | Tensor) -> float:
        """Cost of the policy's capacities under the **simulation** — the score.

        Training happens on :meth:`fluid_cost`; this is what the policy is
        actually worth. The difference between the two is the surrogate gap
        Phase 3 characterised.
        """
        return self.cost_of_capacities(self.capacities(decision))

    def evaluate(self, decision: list[float] | Tensor) -> Evaluation:
        weights = torch.as_tensor(
            decision, dtype=self.features.dtype
        ).clone().requires_grad_(True)
        cost = self._cost(weights)
        (gradient,) = torch.autograd.grad(cost, weights)
        capacity = self.capacities(weights.detach())
        return Evaluation(
            objective=cost.item(),
            kpis={
                "mean_capacity": capacity.mean().item(),
                "min_capacity": capacity.min().item(),
                "max_capacity": capacity.max().item(),
            },
            violations={},
            sensitivity=Sensitivity(
                kind=SensitivityKind.PATHWISE,
                names=list(self.rule_names),
                values=[float(g) for g in gradient],
            ),
            provenance=self.provenance,
        )

    def gradient(self, decision: list[float] | Tensor) -> Sensitivity:
        sensitivity = self.evaluate(decision).sensitivity
        assert sensitivity is not None
        return sensitivity

    # ------------------------------------------------------------------
    def cost_of_capacities(self, capacity: Tensor) -> float:
        """Cost of an arbitrary capacity vector — the yardstick for a policy.

        Lets the unconstrained per-item optimum be computed by search and the
        rule policy be scored against it: *how much does interpretability
        cost?*
        """
        with torch.no_grad():
            return replenishment_cost(
                capacity, self.demand, self.reorder,
                lead_time=self.lead_time, tau=self.tau, hard=False, **self.cost_kwargs,
            ).item()

    def explain(self, decision: list[float] | Tensor) -> str:
        """The policy in words — what an LLM or a planner reads."""
        weights = torch.as_tensor(decision, dtype=self.features.dtype)
        ranked = sorted(
            zip(self.rule_names, [float(w) for w in weights]),
            key=lambda nw: -abs(nw[1]),
        )
        lines = ["Carrier-sizing policy (larger weight -> larger carriers):"]
        for name, weight in ranked:
            direction = "larger" if weight > 0 else "smaller"
            lines.append(f"  {name:<15} {weight:+.4f}   -> {direction} carriers")
        capacity = self.capacities(weights)
        lines.append(
            f"  => capacities {capacity.min():.1f}-{capacity.max():.1f}, "
            f"mean {capacity.mean():.1f}"
        )
        return "\n".join(lines)

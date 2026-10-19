"""T1 — replenishment dynamics, and the two static projections of them.

Phase 3 of ``PLAN.md``. The static intralogistics model is a **time average**,
not a time-free description: constraint (17) is the stationarity condition
``dI/dt = 0`` and every parameter is a rate or a mean. The question this module
exists to answer is the one that decides whether the rest of the framework is
worth building:

    **When does that time average pick the same decision as the dynamics?**

The slice under test is the smallest one that can answer it: one item, one
warehouse, one decision — the **load-carrier capacity** ``Q`` — under two-bin
Kanban replenishment.

Three world models over that same decision, sharing one cost function so the
comparison is fair; they differ only in *where the cost function's inputs come
from*:

``StaticExogenousWorld``
    Average inventory is a **given parameter**, independent of ``Q``. This is
    what the source MILP does: constraint (8) fixes
    ``sum_l q_inventory == avg_inventory_total``.

``StaticSelfConsistentWorld``
    Average inventory is the textbook fluid expression ``Q/2 + safety stock``,
    so it at least *knows* that carrier size drives inventory. Included so the
    comparison is not a straw man: if the exogenous variant fails only because
    its parameter was stale, this variant will show that, and the fix is then
    a calibration rule rather than a simulation.

``DynamicWorld``
    Simulates the sawtooth. Average inventory, replenishment frequency, peak
    occupancy and **stockouts** are outputs, not inputs.

The structural point, stated precisely: the source model has the dependency
**backwards**. Constraint (11) ``q_inventory >= 2 * carrier_capacity`` reads
*"given the inventory you hold, a bin may be at most half of it"*, whereas the
physics is *"given the bin size you choose, your average inventory will be
about half a bin plus safety stock."* Same relationship — but one treats it as
a restriction on the decision, the other as a consequence of it. Only the
second closes the loop, because ``avg_inventory_total`` is otherwise a
parameter that depends on the decision it is supposed to inform.

.. note::
   Deliberately numpy, not tensors. Phase 4 tensorises this and relaxes the
   reorder trigger ``1[IP <= Q]`` with a sigmoid — the direct analogue of
   SoftSort for ``argsort``, and the second instance of the RQ1 phenomenon.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from opsworld.world import Capability, Evaluation

__all__ = [
    "ReplenishmentParams",
    "CarrierCostModel",
    "DynamicOutcome",
    "simulate_replenishment",
    "CapacityChoiceSpace",
    "StaticExogenousWorld",
    "StaticSelfConsistentWorld",
    "DynamicWorld",
]


@dataclass(frozen=True)
class ReplenishmentParams:
    """The environment: how demand arrives and how long resupply takes."""

    demand_rate: float = 10.0     # lambda, units per period
    demand_cv: float = 0.3        # coefficient of variation of per-period demand
    lead_time: int = 3            # periods between order and arrival
    service_factor: float = 1.65  # z in the safety-stock formula (~95%)
    horizon: int = 4000           # simulated periods (after warm-up)
    warmup: int = 400             # discarded periods, to shed the initial transient
    seed: int = 0

    @property
    def lead_time_demand(self) -> float:
        """Units consumed during one replenishment lead time."""
        return self.demand_rate * self.lead_time

    @property
    def safety_stock(self) -> float:
        """Textbook ``z * sigma * sqrt(L)``.

        Shared by the simulation and the self-consistent static model, so that
        any difference between them comes from the dynamics and not from two
        different stocking policies.
        """
        sigma = self.demand_cv * self.demand_rate
        return self.service_factor * sigma * math.sqrt(self.lead_time)

    @property
    def reorder_point(self) -> float:
        """``s`` — cover demand over the lead time, plus safety stock."""
        return self.lead_time_demand + self.safety_stock


@dataclass(frozen=True)
class CarrierCostModel:
    """Cost of a configuration, shared by all three world models.

    ``area(Q) = area_fixed + area_per_unit * Q`` — a carrier has a footprint
    overhead plus a part proportional to what it holds. The defaults are fitted
    to the two carriers of the source dataset (B1: capacity 10, area 1.0;
    B2: capacity 25, area 2.0).
    """

    storage_rate: float = 1.0        # cost per unit area per period
    area_fixed: float = 1.0 / 3.0
    area_per_unit: float = 1.0 / 15.0
    move_cost: float = 12.0          # cost per replenishment move
    stockout_penalty: float = 0.0    # per unit of unmet demand; 0 = invisible

    def area(self, capacity: float) -> float:
        return self.area_fixed + self.area_per_unit * capacity

    def cost(
        self,
        capacity: float,
        avg_inventory: float,
        moves_per_period: float,
        stockout_units: float = 0.0,
    ) -> dict[str, float]:
        """Cost components. Storage is priced per *carrier*, as in the MILP:
        constraint (21) makes the carrier count cover the inventory."""
        carriers = math.ceil(avg_inventory / capacity) if capacity > 0 else 0.0
        return {
            "cost_storage": self.storage_rate * carriers * self.area(capacity),
            "cost_reloc": self.move_cost * moves_per_period,
            "cost_stockout": self.stockout_penalty * stockout_units,
        }


@dataclass
class DynamicOutcome:
    """What the simulation reports — every field is an *output*, not an input."""

    avg_inventory: float
    max_inventory: float
    moves_per_period: float
    stockout_units_per_period: float
    service_level: float

    def summary(self) -> str:
        return (
            f"avg_inv={self.avg_inventory:7.2f}  peak={self.max_inventory:7.2f}  "
            f"moves/period={self.moves_per_period:.4f}  "
            f"service={100 * self.service_level:6.2f}%"
        )


def _demand_draws(params: ReplenishmentParams, n: int) -> np.ndarray:
    """Non-negative demand with the requested mean and coefficient of variation."""
    rng = np.random.default_rng(params.seed)
    if params.demand_cv <= 0:
        return np.full(n, params.demand_rate)
    # Gamma: mean = shape*scale, cv = 1/sqrt(shape).
    shape = 1.0 / params.demand_cv**2
    return rng.gamma(shape, params.demand_rate / shape, size=n)


def simulate_replenishment(
    capacity: float, params: ReplenishmentParams
) -> DynamicOutcome:
    """Simulate an ``(s, nQ)`` replenishment policy over ``capacity``-sized bins.

    Each period: receive arrivals, meet demand (shortfalls are backordered, as
    a starved line is served late rather than never), then check the inventory
    position ``on hand - backorders + on order``. When it has fallen to the
    reorder point ``s``, order **as many whole bins as needed** to get back
    above it.

    Ordering in multiples matters: with a single bin per period the system
    cannot keep up whenever ``demand_rate > capacity``, and the replenishment
    rate silently saturates — flow conservation then fails by a factor of
    three or more at small bin sizes. Two-bin Kanban is the special case of
    this policy in the regime ``lead-time demand <= capacity``, which is
    exactly the regime constraint (11) of the source model presumes.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    total = params.warmup + params.horizon
    demand = _demand_draws(params, total)
    reorder_point = params.reorder_point

    net = reorder_point + capacity    # start at the order-up-to level
    pipeline: deque[float] = deque([0.0] * max(1, params.lead_time))

    on_hand_trace = np.empty(params.horizon)
    moves = 0
    unmet = 0.0
    demand_total = 0.0

    for t in range(total):
        net += pipeline.popleft()

        available = max(net, 0.0)
        shortfall = max(demand[t] - available, 0.0)
        net -= demand[t]

        position = net + sum(pipeline)
        if position <= reorder_point:
            bins = math.ceil((reorder_point + capacity - position) / capacity)
            pipeline.append(bins * capacity)
        else:
            bins = 0
            pipeline.append(0.0)

        if t >= params.warmup:
            on_hand_trace[t - params.warmup] = max(net, 0.0)
            unmet += shortfall
            demand_total += demand[t]
            moves += bins

    return DynamicOutcome(
        avg_inventory=float(on_hand_trace.mean()),
        max_inventory=float(on_hand_trace.max()),
        moves_per_period=moves / params.horizon,
        stockout_units_per_period=unmet / params.horizon,
        service_level=1.0 - unmet / demand_total if demand_total else 1.0,
    )


class CapacityChoiceSpace:
    """The decision is a single load-carrier capacity, from a discrete menu."""

    def __init__(self, capacities: list[float]) -> None:
        if not capacities:
            raise ValueError("need at least one candidate capacity")
        self.capacities = sorted(capacities)

    def initial(self) -> float:
        return self.capacities[0]

    def perturb(self, decision: float, rng) -> float:
        index = self.capacities.index(decision) if decision in self.capacities else 0
        step = rng.choice((-1, 1))
        return self.capacities[max(0, min(len(self.capacities) - 1, index + step))]


class _CapacityWorld:
    """Shared plumbing: same decision space, same cost model, same report shape."""

    provenance = "unset"

    def __init__(
        self,
        params: ReplenishmentParams,
        costs: CarrierCostModel,
        capacities: list[float],
    ) -> None:
        self.params = params
        self.costs = costs
        self._space = CapacityChoiceSpace(capacities)

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE}

    def decision_space(self) -> CapacityChoiceSpace:
        return self._space

    def _report(self, components: dict[str, float], extra: dict[str, float]) -> Evaluation:
        return Evaluation(
            objective=sum(components.values()),
            kpis={**components, **extra},
            violations={},
            provenance=self.provenance,
        )


class StaticExogenousWorld(_CapacityWorld):
    """Average inventory is a fixed parameter — the source MILP's assumption.

    Constraint (11) of the source model, ``q_inventory >= 2 * capacity``, is
    carried over as the feasibility limit it is there: a bin may be at most
    half the inventory on hand.
    """

    provenance = "static/exogenous (T0, avg inventory given)"

    def __init__(
        self,
        params,
        costs,
        capacities,
        avg_inventory: float,
        *,
        kanban_constraint: bool = True,
    ) -> None:
        super().__init__(params, costs, capacities)
        self.avg_inventory = avg_inventory
        # Switchable so the experiment can separate two distinct failure modes:
        # the constraint wrongly shrinking the decision set, and the cost model
        # itself ranking decisions wrongly.
        self.kanban_constraint = kanban_constraint

    def evaluate(self, decision: float) -> Evaluation:
        capacity = float(decision)
        moves = self.params.demand_rate / capacity
        components = self.costs.cost(capacity, self.avg_inventory, moves)
        evaluation = self._report(
            components,
            {"avg_inventory": self.avg_inventory, "moves_per_period": moves},
        )
        # (11) two-bin Kanban, as a restriction on the decision
        if self.kanban_constraint and 2.0 * capacity > self.avg_inventory:
            evaluation.violations["11_kanban_two_carriers"] = (
                2.0 * capacity - self.avg_inventory
            )
        return evaluation


class StaticSelfConsistentWorld(_CapacityWorld):
    """Average inventory is the fluid expression ``Q/2 + safety stock``.

    The charitable static model: it does not simulate, but it does close the
    loop between the decision and the inventory the decision implies. Safety
    stock is the textbook ``z * sigma * sqrt(lead time)``.
    """

    provenance = "static/self-consistent (T0, avg inventory = Q/2 + ss)"

    def evaluate(self, decision: float) -> Evaluation:
        capacity = float(decision)
        # Same safety-stock rule the simulation stocks to, so any gap between
        # the two comes from the dynamics rather than from a different policy.
        avg_inventory = capacity / 2.0 + self.params.safety_stock
        moves = self.params.demand_rate / capacity
        components = self.costs.cost(capacity, avg_inventory, moves)
        return self._report(
            components,
            {"avg_inventory": avg_inventory, "moves_per_period": moves},
        )


class DynamicWorld(_CapacityWorld):
    """Simulated replenishment — inventory and moves are consequences.

    ``service_target`` turns the fill rate into a **constraint**. The static
    models cannot express it at all: a time average has no stockouts in it, so
    every configuration looks equally safe to them. Any configuration this
    model rejects and a static model accepts is a *false-feasible* — a plan an
    optimiser would happily return and a line would starve on.
    """

    provenance = "dynamic/DES (T1, simulated sawtooth)"

    def __init__(self, params, costs, capacities, *, service_target: float = 0.0) -> None:
        super().__init__(params, costs, capacities)
        self.service_target = service_target

    def evaluate(self, decision: float) -> Evaluation:
        capacity = float(decision)
        outcome = simulate_replenishment(capacity, self.params)
        components = self.costs.cost(
            capacity,
            outcome.avg_inventory,
            outcome.moves_per_period,
            outcome.stockout_units_per_period,
        )
        evaluation = self._report(
            components,
            {
                "avg_inventory": outcome.avg_inventory,
                "peak_inventory": outcome.max_inventory,
                "moves_per_period": outcome.moves_per_period,
                "service_level": outcome.service_level,
            },
        )
        if outcome.service_level < self.service_target:
            evaluation.violations["service_level"] = (
                self.service_target - outcome.service_level
            )
        return evaluation

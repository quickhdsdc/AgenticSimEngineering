"""T2 — the slow scale, where the problem is genuinely sequential (Phase 5.5).

``PLAN.md`` §1.4 splits the world into three time scales. T0 is the static
model, T1 the operating dynamics, and **T2 is the one nothing in this
repository has modelled yet**: months to years, where the *configuration itself*
changes. Two events drive it, both named in the source dissertation's slides as
open application scenarios:

    "Ich habe einen neuen Artikel im System."
    "Die Nachfrage nach meinem Artikel hat sich geändert."

Neither is a new optimisation problem. Both are transitions of an existing one,
and what makes them sequential is **Umräumkosten** — moving an item to a
different warehouse costs something *once*, and that cost depends on where it
was, which is the state. State, action, state-dependent cost: an MDP, and the
first ``C1`` world model on this side of the framework.

What this module is for
-----------------------
``PLAN.md`` asserts that re-solving the global model every period "is slow and
produces violently jumping configurations", and that reinforcement learning
therefore has a genuine edge here. That is an assumption, and the plan's own
rule is that assumptions get a falsifiable test before work is built on them.
So the module ships the environment and **three reference policies that bracket
the question**:

``FrozenPolicy``
    place newcomers, never move anything. Pays no reconfiguration cost and
    accumulates operating cost.

``MyopicReoptimise``
    re-solve from scratch each period, ignoring what it costs to get there.
    This is what re-running the MILP does.

``ReconfigurationAware``
    re-solve with the move cost charged inside the objective, so a change has
    to pay for itself out of the periods that remain.

If the third beats the other two, the sequential structure is real and worth a
learned policy. If it merely ties the second, the churn story is wrong and
Phase 5.5 should shrink rather than acquire an RL agent — see the measured
answer in ``PLAN.md``.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

from opsworld.world import Capability, Evaluation, penalised_objective

from .data import Instance
from .heuristic import GreedySupplySolver, PlanWorld, SupplyPlan, render
from .static_model import StaticWorldModel

__all__ = [
    "ReconfigurationCost",
    "Event",
    "Episode",
    "build_episode",
    "SupplyChainState",
    "SupplyChainMDP",
    "Policy",
    "FrozenPolicy",
    "MyopicReoptimise",
    "ReconfigurationAware",
    "run_episode",
    "EpisodeResult",
]

PROVENANCE = "reconfiguration/mdp (T2, slow scale)"


# ----------------------------------------------------------------------
# what a change costs
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ReconfigurationCost:
    """Umräumkosten — the one-off price of moving from one plan to another.

    Charged per *decision changed*, not per variable, because that is what a
    warehouse actually does: emptying a location, setting up a new one, and
    swapping a carrier type are each a job someone performs.

    ``open_location`` dominates on purpose: standing up stock in a warehouse
    that did not hold the item is the expensive move, and it is the one a
    myopic re-solve makes freely.
    """

    open_location: float = 40.0
    close_location: float = 15.0
    change_pick: float = 8.0
    change_carrier: float = 25.0

    def between(self, before: SupplyPlan, after: SupplyPlan, items) -> float:
        total = 0.0
        for item in items:
            was = set(before.stored_at(item))
            now = set(after.stored_at(item))
            total += self.open_location * len(now - was)
            total += self.close_location * len(was - now)
        for key, warehouse in after.pick.items():
            if before.pick.get(key) not in (None, warehouse):
                total += self.change_pick
        for key, carrier in after.carrier.items():
            previous = before.carrier.get(key)
            if previous is not None and previous != carrier:
                total += self.change_carrier
        return total


# ----------------------------------------------------------------------
# what happens between periods
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    """One slow-scale change to the world.

    ``kind`` is ``"new_item"`` or ``"demand_shift"``; ``payload`` carries the
    item name or the per-item multiplier. Deliberately a small closed set —
    these are the two scenarios the source work names, and inventing more
    would be modelling ahead of the evidence.
    """

    period: int
    kind: str
    payload: Any


@dataclass
class Episode:
    """A base instance plus the events that will happen to it."""

    base: Instance
    events: tuple[Event, ...]
    horizon: int

    def events_at(self, period: int) -> list[Event]:
        return [event for event in self.events if event.period == period]


def _with_new_item(instance: Instance, name: str, template: Any) -> Instance:
    """``instance`` plus one more item, cloned from ``template``'s parameters.

    A newcomer is a real item, so every indexed table it appears in has to gain
    an entry or :class:`~intralogistics.data.Instance` will reject it. Cloning
    an existing item keeps the newcomer physically plausible without inventing
    a second data generator.
    """
    data = copy.deepcopy(instance.data)
    data["items"] = [*instance.I, name]

    product = next(
        p for p in instance.P if template in instance.items_of_product(p)
    )
    data["items_of_product"] = {
        p: [*items, name] if p == product else list(items)
        for p, items in instance.data["items_of_product"].items()
    }
    data["qty_bom"][(product, name)] = instance.param("qty_bom", product, template)

    for r in instance.R:
        data["demand_item_resource"][(name, r)] = instance.param(
            "demand_item_resource", template, r
        )
    for key in ("avg_inventory_total", "item_value", "item_area", "incoming_quantity"):
        data[key][name] = instance.param(key, template)
    for b in instance.B:
        data["carrier_capacity"][(name, b)] = instance.param(
            "carrier_capacity", template, b
        )
    for l in instance.L_sys:
        data["min_inventory"][(name, l)] = instance.param("min_inventory", template, l)
    for l1, l2 in instance.U_edges:
        for b1 in instance.B:
            for b2 in instance.B:
                data["relocation_cost"][(name, l1, l2, b1, b2)] = instance.param(
                    "relocation_cost", template, l1, l2, b1, b2
                )
    return Instance(data)


def _with_demand_shift(instance: Instance, multipliers: dict[Any, float]) -> Instance:
    """``instance`` with per-item demand scaled.

    Average inventory scales with it. Leaving it fixed would quietly turn a
    demand shift into a change of *turnover ratio*, which is a different
    scenario and would confound the comparison.
    """
    data = copy.deepcopy(instance.data)
    for i in instance.I:
        factor = multipliers.get(i, 1.0)
        if factor == 1.0:
            continue
        for r in instance.R:
            data["demand_item_resource"][(i, r)] *= factor
        data["avg_inventory_total"][i] *= factor
        data["incoming_quantity"][i] *= factor
    return Instance(data)


def build_episode(
    base: Instance,
    *,
    horizon: int = 8,
    n_new_items: int = 2,
    shift_periods: tuple[int, ...] = (2, 5),
    shift_strength: float = 0.6,
    seed: int = 0,
) -> Episode:
    """A schedule of newcomers and demand shifts over ``horizon`` periods."""
    rng = random.Random(seed)
    events: list[Event] = []

    for k in range(n_new_items):
        period = rng.randrange(1, max(2, horizon))
        events.append(
            Event(period, "new_item", (f"NEW{k + 1}", rng.choice(base.I)))
        )
    for period in shift_periods:
        if period >= horizon:
            continue
        multipliers = {
            i: 1.0 + shift_strength * (rng.random() * 2.0 - 1.0) for i in base.I
        }
        events.append(Event(period, "demand_shift", multipliers))

    return Episode(base=base, events=tuple(sorted(events, key=lambda e: e.period)),
                   horizon=horizon)


# ----------------------------------------------------------------------
# the MDP
# ----------------------------------------------------------------------
@dataclass
class SupplyChainState:
    """Where the world is: the live instance, the standing plan, the clock."""

    period: int
    instance: Instance
    plan: SupplyPlan
    done: bool = False

    def copy(self) -> SupplyChainState:
        return SupplyChainState(self.period, self.instance, self.plan.copy(), self.done)


class SupplyChainMDP:
    """Configuration over the slow scale, as a ``C1`` sequential world model.

    ``step`` applies the period's events to the instance *first* — a newcomer
    or a demand shift is something that happens *to* the planner — then charges
    the action's reconfiguration cost and the resulting period's operating
    cost. That order matters: acting before the event would be clairvoyance,
    and it is exactly the advantage a lookahead policy is not entitled to.
    """

    def __init__(
        self,
        episode: Episode,
        *,
        costs: ReconfigurationCost = ReconfigurationCost(),
        penalty: float = 1e6,
    ) -> None:
        self.episode = episode
        self.costs = costs
        self.penalty = penalty
        self._model = StaticWorldModel()

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE, Capability.SEQUENTIAL}

    def decision_space(self):  # noqa: ANN201 - a SupplyPlanSpace
        """Plans over the *base* instance.

        A newcomer widens the space mid-episode, so this is the space as it
        stands at reset rather than a fixed description of the whole episode —
        which is exactly what makes the problem sequential rather than a large
        one-shot one.
        """
        return PlanWorld(self.episode.base).decision_space()

    # -- C0 -----------------------------------------------------------
    def evaluate(self, decision: tuple[Instance, SupplyPlan]) -> Evaluation:
        instance, plan = decision
        evaluation = self._model.evaluate(instance, render(instance, plan))
        evaluation.provenance = PROVENANCE
        return evaluation

    def operating_cost(self, instance: Instance, plan: SupplyPlan) -> float:
        return penalised_objective(self.evaluate((instance, plan)), self.penalty)

    # -- C1 -----------------------------------------------------------
    def reset(self) -> SupplyChainState:
        instance = self.episode.base
        plan = PlanWorld(instance).decision_space().initial()
        return SupplyChainState(period=0, instance=instance, plan=plan)

    def advance_world(self, state: SupplyChainState) -> Instance:
        """The instance as it will be *after* this period's events land."""
        instance = state.instance
        for event in self.episode.events_at(state.period + 1):
            if event.kind == "new_item":
                name, template = event.payload
                instance = _with_new_item(instance, name, template)
            elif event.kind == "demand_shift":
                instance = _with_demand_shift(instance, event.payload)
            else:  # pragma: no cover - the set is closed
                raise ValueError(f"unknown event kind {event.kind!r}")
        return instance

    def step(
        self, state: SupplyChainState, action: SupplyPlan
    ) -> tuple[SupplyChainState, float, bool]:
        """Apply the period's events, charge the move, then charge operations."""
        instance = self.advance_world(state)
        move = self.costs.between(state.plan, action, instance.I)
        operating = self.operating_cost(instance, action)
        period = state.period + 1
        done = period >= self.episode.horizon
        return (
            SupplyChainState(period, instance, action.copy(), done),
            -(move + operating),
            done,
        )


# ----------------------------------------------------------------------
# reference policies
# ----------------------------------------------------------------------
class Policy(Protocol):
    name: ClassVar[str]

    def act(self, mdp: SupplyChainMDP, state: SupplyChainState,
            instance: Instance) -> SupplyPlan:
        ...


def _place_newcomers(instance: Instance, plan: SupplyPlan) -> SupplyPlan:
    """Extend ``plan`` to cover items it does not mention yet.

    Every policy needs this: an item with no pick location leaves constraint
    (6) violated, so "do nothing" is not an available action for a newcomer.
    """
    from .heuristic import _assignment_cost, _cheapest_carrier

    extended = plan.copy()
    for i in instance.I:
        if any(item == i for item, _ in extended.pick):
            continue
        for r in instance.R:
            if float(instance.param("demand_item_resource", i, r)) <= 0:
                continue
            admissible = [
                l for l in instance.L_sys
                if instance.param("warehouse_assigned_to_resource", r, l)
            ]
            if not admissible:
                continue
            extended.pick[(i, r)] = min(
                admissible, key=lambda l: _assignment_cost(instance, i, r, l)
            )
        for l in extended.stored_at(i):
            if (i, l) not in extended.carrier:
                carrier = _cheapest_carrier(instance, i, l)
                if carrier is not None:
                    extended.carrier[(i, l)] = carrier
    return extended


class FrozenPolicy:
    """Place newcomers, never move anything that already works."""

    name: ClassVar[str] = "frozen"

    def act(self, mdp, state, instance):  # noqa: ANN001
        return _place_newcomers(instance, state.plan)


class MyopicReoptimise:
    """Re-solve from scratch every period, ignoring what the move costs.

    This is what re-running the exact model does, and it is the behaviour
    ``PLAN.md`` predicts will churn.
    """

    name: ClassVar[str] = "myopic"

    def __init__(self, *, budget: int = 4000, max_passes: int = 8) -> None:
        self.budget = budget
        self.max_passes = max_passes

    def act(self, mdp, state, instance):  # noqa: ANN001
        solver = GreedySupplySolver(max_passes=self.max_passes)
        return solver.solve(PlanWorld(instance), budget=self.budget).decision


class ReconfigurationAware:
    """Re-solve with the move cost charged inside the objective.

    A change has to pay for itself out of the periods that remain, so the
    horizon enters the decision. This is the cheapest possible way to be
    non-myopic — one step of lookahead, no learning — and it is the baseline a
    learned policy has to beat before it is worth building.
    """

    name: ClassVar[str] = "reconfiguration-aware"

    def __init__(self, *, budget: int = 4000, max_passes: int = 8) -> None:
        self.budget = budget
        self.max_passes = max_passes

    def act(self, mdp, state, instance):  # noqa: ANN001
        remaining = max(1, mdp.episode.horizon - state.period)
        world = PlanWorld(instance)
        space = world.decision_space()

        def score(plan: SupplyPlan) -> float:
            operating = world.evaluate(plan)
            move = mdp.costs.between(state.plan, plan, instance.I)
            return (
                remaining * penalised_objective(operating, mdp.penalty) + move
            )

        # Two starting points, not one. Warm-starting from the standing plan is
        # what makes this policy calm, but a standing plan that an event has
        # just made infeasible can trap the local search where a fresh
        # construction would not -- measured: at horizon 4 the warm start alone
        # left the configuration infeasible and the episode cost a penalty.
        # A policy must not lose to the one it is meant to beat merely because
        # of where it began, so the reconstruction competes on the same
        # move-aware objective and usually loses on its own merits.
        starts = [_place_newcomers(instance, state.plan)]
        rebuilt = GreedySupplySolver(max_passes=self.max_passes).solve(
            world, budget=self.budget
        ).decision
        starts.append(rebuilt)

        best = min(starts, key=score)
        best_score = score(best)
        used = 0
        for _ in range(self.max_passes):
            improved = False
            for candidate in space.moves(best):
                if used >= self.budget:
                    break
                used += 1
                value = score(candidate)
                if value < best_score - 1e-9:
                    best, best_score, improved = candidate, value, True
            if not improved or used >= self.budget:
                break
        return best


# ----------------------------------------------------------------------
# running one
# ----------------------------------------------------------------------
@dataclass
class EpisodeResult:
    """What a policy cost over an episode, split into its two parts."""

    policy: str
    operating: list[float] = field(default_factory=list)
    reconfiguration: list[float] = field(default_factory=list)
    infeasible_periods: int = 0
    locations_touched: int = 0

    @property
    def total(self) -> float:
        return sum(self.operating) + sum(self.reconfiguration)

    def summary(self) -> str:
        return (
            f"{self.policy:<22} total={self.total:12.2f}  "
            f"operating={sum(self.operating):11.2f}  "
            f"moves={sum(self.reconfiguration):9.2f}  "
            f"locations touched={self.locations_touched:3d}  "
            f"infeasible periods={self.infeasible_periods}"
        )


def run_episode(mdp: SupplyChainMDP, policy: Policy) -> EpisodeResult:
    """Roll the episode forward under ``policy``, accounting each period."""
    state = mdp.reset()
    result = EpisodeResult(policy=policy.name)

    while not state.done:
        upcoming = mdp.advance_world(state)
        action = policy.act(mdp, state, upcoming)

        move = mdp.costs.between(state.plan, action, upcoming.I)
        touched = sum(
            len(set(action.stored_at(i)) ^ set(state.plan.stored_at(i)))
            for i in upcoming.I
        )
        state, reward, _ = mdp.step(state, action)

        evaluation = mdp.evaluate((state.instance, state.plan))
        result.operating.append(-reward - move)
        result.reconfiguration.append(move)
        result.locations_touched += touched
        if not evaluation.is_feasible(tol=1e-6):
            result.infeasible_periods += 1

    return result

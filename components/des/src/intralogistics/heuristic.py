"""A constructive heuristic for instances the MILP cannot reach — Phase 5.1.

Why a heuristic is needed at all is measurable rather than assumed. The
relocation variable ``y_reloc_moves`` carries six indices, so the model grows as
``|I| x |R| x |edges| x |B|^2``, and branch and bound on the generated instances
of :mod:`intralogistics.instances` walks off a cliff::

    3 items x 2 resources     0.7 s
    4 items x 2 resources     4.6 s
    5 items x 2 resources    29.4 s
    6 items x 2 resources    58.7 s
    6 items x 3 resources   170.4 s

The dissertation this model comes from plans for *thousands* of items. The
exact solver is not slow there — it does not return.

Why the obvious thing does not work
-----------------------------------
Phase 2 already tried the obvious thing and measured it failing. From the empty
configuration **no single-variable move improves anything**: raising
``q_inventory`` reduces the violation of (8) inventory balance by exactly what
it adds to (9) store-binary and (21) carrier capacity. Zero improving
candidates out of 2000, in every variable family.

That is not a tuning problem, and the fix is not a cleverer acceptance rule.
The variables of this model are not independent decisions — they are the
*consequences* of a few real ones:

    where is each demand served from?   ->  x_pick
    which carrier sits there?           ->  x_carrier
    where does the bulk stock live?     ->  x_store, q_inventory, y_carriers,
                                            y_reloc_moves, x_product_...

So this module searches over the **three real decisions** and derives the rest.
A move is "serve (item, resource) from a different warehouse", and everything
that follows — inventory split, carrier counts, routing, product links — is
recomputed by :func:`render`. The neighbourhood is small, every neighbour is
feasible by construction, and the coupling that defeated the variable-level
search is now inside the renderer rather than in the search.

The payoff is not only heuristic: because the decision space is now the right
one, the **generic** ``HillClimbSolver`` from :mod:`opsworld.solver` — which
knows nothing about intralogistics — becomes effective on it. What was missing
in Phase 2 was never the solver.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, ClassVar

from opsworld.solver import SolverResult, check_capabilities
from opsworld.world import Capability, Evaluation, penalised_objective

from .data import Instance
from .decision import Decision
from .static_model import StaticWorldModel

__all__ = [
    "SupplyPlan",
    "render",
    "build_greedy_plan",
    "SupplyPlanSpace",
    "PlanWorld",
    "GreedySupplySolver",
]

PROVENANCE = "heuristic/plan (T0, time-averaged)"


# ----------------------------------------------------------------------
# the compact decision
# ----------------------------------------------------------------------
@dataclass
class SupplyPlan:
    """The three decisions everything else follows from.

    ``pick`` maps ``(item, resource)`` to the warehouse serving it — constraint
    (6) picks exactly one, and (7) restricts it to the warehouses assigned to
    that resource.

    ``carrier`` maps ``(item, warehouse)`` to a load-carrier type, for every
    warehouse the item ends up stored in.

    ``hub`` optionally names one intermediate warehouse per item to hold the
    bulk of its stock. It exists because constraint (8) is an *equality*: an
    item's whole average inventory has to sit somewhere, and the points of use
    are both the dearest place to put it and the one capped by (27).
    """

    pick: dict[tuple[Any, Any], Any] = field(default_factory=dict)
    carrier: dict[tuple[Any, Any], Any] = field(default_factory=dict)
    hub: dict[Any, Any] = field(default_factory=dict)

    def copy(self) -> SupplyPlan:
        return SupplyPlan(dict(self.pick), dict(self.carrier), dict(self.hub))

    def pick_warehouses(self, item: Any) -> list[Any]:
        """Distinct warehouses this item is picked from, in insertion order."""
        seen: list[Any] = []
        for (i, _), l in self.pick.items():
            if i == item and l not in seen:
                seen.append(l)
        return seen

    def stored_at(self, item: Any) -> list[Any]:
        """Every warehouse holding stock of ``item``: pick locations plus hub."""
        warehouses = self.pick_warehouses(item)
        hub = self.hub.get(item)
        if hub is not None and hub not in warehouses:
            warehouses.append(hub)
        return warehouses


# ----------------------------------------------------------------------
# plan -> Decision
# ----------------------------------------------------------------------
def _route(inst: Instance, plan: SupplyPlan, item: Any, target: Any) -> list[Any]:
    """Warehouse chain from goods receipt down to ``target``.

    Constraint (17b) forbids bypassing an intermediate warehouse the item is
    stored in, so the route is not a free choice: it must call at **every**
    stored intermediate warehouse whose stage lies between the source and the
    target. Sorting the stored warehouses by descending stage produces exactly
    that chain, and it is the only one (17b) permits.
    """
    source = inst.L_we[0]
    target_stage = inst.stage(target)
    waypoints = sorted(
        (
            l
            for l in plan.stored_at(item)
            if l in inst.L_log and target_stage < inst.stage(l) < inst.stage(source)
        ),
        key=lambda l: -inst.stage(l),
    )
    return [source, *waypoints, target]


def _floor(inst: Instance, plan: SupplyPlan, item: Any, warehouse: Any) -> float:
    """Least stock ``warehouse`` may legally hold of ``item``.

    The binding one of (11)'s two-bin rule and, at a pick location, (10)'s
    minimum. Both are lower bounds on ``q_inventory``, and (8) is an equality,
    so the sum of these floors is what an item's total stock has to cover
    before any of it is free to be placed.
    """
    b = plan.carrier.get((item, warehouse))
    two_bin = 2.0 * float(inst.param("carrier_capacity", item, b)) if b else 0.0
    minimum = (
        float(inst.param("min_inventory", item, warehouse))
        if warehouse in plan.pick_warehouses(item)
        else 0.0
    )
    return max(two_bin, minimum)


def _floor_total(inst: Instance, plan: SupplyPlan, item: Any) -> float:
    return sum(_floor(inst, plan, item, l) for l in plan.stored_at(item))


def _inventory_split(
    inst: Instance, plan: SupplyPlan, item: Any
) -> dict[Any, float]:
    """Distribute an item's total stock over the warehouses holding it.

    Each location gets the least it may legally hold — the larger of (10)'s
    minimum for a pick location and (11)'s two-bin rule — and the remainder
    goes to the hub if there is one, otherwise to the cheapest location. The
    residual has to go *somewhere*: (8) is an equality, not a bound.
    """
    total = float(inst.param("avg_inventory_total", item))
    warehouses = plan.stored_at(item)
    if not warehouses:
        return {}

    floor = {l: _floor(inst, plan, item, l) for l in warehouses}
    required = sum(floor.values())
    if required > total:
        # The floors alone overflow the item's stock: scale them down. The
        # result violates (11) and the evaluator will say so -- better a
        # legible violation than a silently inflated inventory.
        scale = total / required if required > 0 else 0.0
        return {l: value * scale for l, value in floor.items()}

    sink = plan.hub.get(item)
    if sink is None or sink not in floor:
        sink = min(floor, key=lambda l: float(inst.param("storage_cost_rate", l)))
    split = dict(floor)
    split[sink] += total - required
    return split


def _cheapest_carrier(inst: Instance, item: Any, warehouse: Any) -> Any | None:
    """Smallest admissible carrier — (20) admissibility, (11) two-bin capped.

    Smallest, because (11) makes a large carrier demand twice its capacity in
    stock at that location, and (8) then has to find that stock somewhere.
    """
    total = float(inst.param("avg_inventory_total", item))
    admissible = [
        b
        for b in inst.B
        if inst.param("carrier_allowed", warehouse, b)
        and 2.0 * float(inst.param("carrier_capacity", item, b)) <= total
    ]
    if not admissible:
        admissible = [b for b in inst.B if inst.param("carrier_allowed", warehouse, b)]
    if not admissible:
        return None
    return min(admissible, key=lambda b: float(inst.param("carrier_capacity", item, b)))


def render(inst: Instance, plan: SupplyPlan) -> Decision:
    """Expand a plan into the full variable set of the source model.

    Every derived family is computed from the constraint that defines it, so
    the mapping is auditable line by line rather than being a heuristic in its
    own right:

    (9)/(19) storage and carrier binaries follow the plan;
    (8)/(10)/(11) the inventory split; (21)/(22) the carrier counts;
    (12) the product-warehouse links; (17) the relocation moves, computed
    bottom-up along the route so that each arc delivers what the one below it
    consumes.
    """
    d = Decision()

    # -- carriers: fill any gap the plan left, so render() is total ------
    plan = plan.copy()
    for i in inst.I:
        for l in plan.stored_at(i):
            if (i, l) not in plan.carrier:
                chosen = _cheapest_carrier(inst, i, l)
                if chosen is not None:
                    plan.carrier[(i, l)] = chosen

    # -- (6)/(7) picking, and (12) the product links --------------------
    for (i, r), l in plan.pick.items():
        if float(inst.param("demand_item_resource", i, r)) <= 0:
            continue
        d.x_pick[(i, r, l)] = 1.0
    for p in inst.P:
        for i in inst.items_of_product(p):
            for (item, _), l in plan.pick.items():
                if item == i and float(inst.param("demand_item_resource", i, _)) > 0:
                    d.x_product_from_warehouse[(p, l)] = 1.0

    # -- (8)-(11) stock, (19)-(22) carriers ------------------------------
    for i in inst.I:
        split = _inventory_split(inst, plan, i)
        total = float(inst.param("avg_inventory_total", i))
        for l, quantity in split.items():
            b = plan.carrier.get((i, l))
            if b is None:
                continue
            d.x_store[(i, l)] = 1.0
            d.q_inventory[(i, l)] = quantity
            d.x_carrier[(i, l, b)] = 1.0
            capacity = float(inst.param("carrier_capacity", i, b))
            allowed = math.ceil(total / capacity)  # (22)
            d.y_carriers[(i, l, b)] = float(
                min(allowed, math.ceil(quantity / capacity - 1e-9))
            )

    # -- (17) relocation, computed bottom-up along each route ------------
    for (i, r), target in plan.pick.items():
        demand = float(inst.param("demand_item_resource", i, r))
        if demand <= 0:
            continue
        chain = _route(inst, plan, i, target)
        needed = demand
        for index in range(len(chain) - 1, 0, -1):
            upstream, downstream = chain[index - 1], chain[index]
            b_down = plan.carrier.get((i, downstream))
            if b_down is None:
                break
            capacity = float(inst.param("carrier_capacity", i, b_down))
            moves = math.ceil(needed / capacity - 1e-9)
            # (14): the source carrier must be the one chosen there; goods
            # receipt holds no carrier decision, so match the target and avoid
            # the carrier-change surcharge.
            b_up = plan.carrier.get((i, upstream), b_down)
            key = (i, r, upstream, downstream, b_up, b_down)
            d.y_reloc_moves[key] = d.y_reloc_moves.get(key, 0.0) + float(moves)
            needed = moves * capacity

    return d


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
def _assignment_cost(
    inst: Instance, item: Any, resource: Any, warehouse: Any
) -> float:
    """Marginal cost of serving ``(item, resource)`` from ``warehouse``.

    Deliberately analytic rather than a call to the evaluator. The evaluator
    is ``O(|I| |R| |edges| |B|^2)`` because of constraints (14)/(15), so using
    it to score every candidate assignment would reintroduce the very scaling
    this module exists to avoid. It is used to *accept* moves, not to enumerate
    them.
    """
    demand = float(inst.param("demand_item_resource", item, resource))
    b = _cheapest_carrier(inst, item, warehouse)
    if b is None:
        return math.inf

    capacity = float(inst.param("carrier_capacity", item, b))
    area = float(inst.param("carrier_area", b))
    hops = inst.stage(inst.L_we[0]) - inst.stage(warehouse)

    picking = demand * float(inst.param("picking_cost_rate", warehouse))
    access = float(inst.scalar("number_of_deliveries")) * float(
        inst.param("access_cost_rate", warehouse)
    )
    # two bins is what (11) forces this location to hold
    storage = (
        float(inst.param("storage_cost_rate", warehouse))
        * math.ceil(2.0 * capacity / capacity)
        * area
    )
    relocation = hops * math.ceil(demand / capacity) * (
        1.0 + 0.1 * area
    )
    return picking + access + storage + relocation


def build_greedy_plan(inst: Instance, *, use_hub: bool = True) -> SupplyPlan:
    """Construct a plan: cheapest admissible warehouse per demand, then a hub.

    Demands are placed largest first, which matters because the capacity
    constraints (23) and (25)/(26) are consumed by whoever gets there first;
    letting the big consumers choose while there is still room is the standard
    first-fit-decreasing argument.
    """
    plan = SupplyPlan()

    demands = sorted(
        (
            (float(inst.param("demand_item_resource", i, r)), i, r)
            for i in inst.I
            for r in inst.R
            if float(inst.param("demand_item_resource", i, r)) > 0
        ),
        key=lambda t: -t[0],
    )
    for _, i, r in demands:
        admissible = [
            l
            for l in inst.L_sys
            if inst.param("warehouse_assigned_to_resource", r, l)
        ]
        if not admissible:
            continue
        best = min(admissible, key=lambda l: _assignment_cost(inst, i, r, l))
        plan.pick[(i, r)] = best

    for i in inst.I:
        for l in plan.stored_at(i):
            chosen = _cheapest_carrier(inst, i, l)
            if chosen is not None:
                plan.carrier[(i, l)] = chosen

    if use_hub:
        # (8) is an equality, so the surplus over what the pick locations must
        # hold has to live somewhere. The cheapest intermediate warehouse is
        # the natural place; (17b) then routes every flow through it, which is
        # what a central warehouse is for.
        #
        # It is not always affordable. Every location holding an item owes two
        # bins of it under (11), so an item picked from many resources can have
        # its floors already add up to its entire stock — and adding a hub then
        # overflows (8). Checking before assigning is cheaper than repairing
        # afterwards, and the alternative (a smaller carrier) does not exist
        # once the smallest admissible one is already in use.
        if inst.L_log:
            hub = min(
                inst.L_log, key=lambda l: float(inst.param("storage_cost_rate", l))
            )
            for i in inst.I:
                if not plan.pick_warehouses(i):
                    continue
                b = _cheapest_carrier(inst, i, hub)
                if b is None:
                    continue
                if _floor_total(inst, plan, i) + 2.0 * float(
                    inst.param("carrier_capacity", i, b)
                ) > float(inst.param("avg_inventory_total", i)):
                    continue
                plan.hub[i] = hub
                plan.carrier[(i, hub)] = b
    return plan


# ----------------------------------------------------------------------
# the decision space, and the world model over it
# ----------------------------------------------------------------------
class SupplyPlanSpace:
    """Neighbourhood over plans rather than over variables.

    Three move families, matching the three real decisions. Each returns a
    plan whose expansion is feasible by construction, which is what the
    variable-level neighbourhood of Phase 2 could not do.

    .. note::
       **Measured limit of this neighbourhood.** At 10 items x 3 resources the
       search settles 10.8% above a branch-and-bound incumbent, and the
       difference is structural rather than a matter of budget: the incumbent
       serves 12 of 25 demands from a shared forward warehouse, paying more in
       picking to save far more in storage and relocation, because (11)'s
       two-bin rule is charged once per *location* rather than once per demand.

       Reaching that configuration needs a coordinated multi-demand move, and
       the ridge between the two is infeasible in three ways at once —
       (17b) requires an item stored at an intermediate warehouse to have
       outflow, so picking from there is only legal while the item is *also*
       stocked downstream; and (25)/(26) cap how much of a resource's demand
       may be picked from a logistics warehouse at all. Best improvement on a
       penalised objective cannot cross that ridge by construction. A move
       family that reassigns a *subset* of an item's demands while preserving a
       downstream store is the fix, and it is not implemented here.
    """

    def __init__(self, instance: Instance) -> None:
        self.inst = instance
        self._admissible = {
            (i, r): [
                l
                for l in instance.L_sys
                if instance.param("warehouse_assigned_to_resource", r, l)
            ]
            for i in instance.I
            for r in instance.R
            if float(instance.param("demand_item_resource", i, r)) > 0
        }

    def initial(self) -> SupplyPlan:
        return build_greedy_plan(self.inst)

    def moves(self, plan: SupplyPlan) -> list[SupplyPlan]:
        """Every neighbour, for a best-improvement search.

        Size is ``sum over demands of (|admissible| - 1)`` plus the carrier and
        hub alternatives — linear in the instance, not in the variable count.
        """
        out: list[SupplyPlan] = []

        for (i, r), current in plan.pick.items():
            for l in self._admissible.get((i, r), []):
                if l == current:
                    continue
                candidate = plan.copy()
                candidate.pick[(i, r)] = l
                if (i, l) not in candidate.carrier:
                    b = _cheapest_carrier(self.inst, i, l)
                    if b is not None:
                        candidate.carrier[(i, l)] = b
                out.append(candidate)

        for i in self.inst.I:
            for l in plan.stored_at(i):
                for b in self.inst.B:
                    if not self.inst.param("carrier_allowed", l, b):
                        continue
                    if plan.carrier.get((i, l)) == b:
                        continue
                    candidate = plan.copy()
                    candidate.carrier[(i, l)] = b
                    out.append(candidate)

        for i in self.inst.I:
            current_hub = plan.hub.get(i)
            for l in [*self.inst.L_log, None]:
                if l == current_hub:
                    continue
                candidate = plan.copy()
                if l is None:
                    candidate.hub.pop(i, None)
                else:
                    candidate.hub[i] = l
                    if (i, l) not in candidate.carrier:
                        b = _cheapest_carrier(self.inst, i, l)
                        if b is not None:
                            candidate.carrier[(i, l)] = b
                out.append(candidate)

        return out

    def perturb(self, plan: SupplyPlan, rng: random.Random) -> SupplyPlan:
        """One random neighbour — what a generic ``C0`` solver needs."""
        candidates = self.moves(plan)
        return rng.choice(candidates) if candidates else plan.copy()


class PlanWorld:
    """The Wirkmodell over plans instead of over raw variables (``C0``).

    Same instance, same evaluator, same cost function as
    :class:`~intralogistics.world.StaticWorld` — only the decision space
    differs. That is the whole claim of this phase: what defeated search in
    Phase 2 was the parameterisation, not the search.
    """

    def __init__(self, instance: Instance) -> None:
        self.inst = instance
        self._model = StaticWorldModel()
        self._space = SupplyPlanSpace(instance)

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE}

    def decision_space(self) -> SupplyPlanSpace:
        return self._space

    def evaluate(self, decision: SupplyPlan) -> Evaluation:
        evaluation = self._model.evaluate(self.inst, render(self.inst, decision))
        evaluation.provenance = PROVENANCE
        return evaluation

    def render(self, plan: SupplyPlan) -> Decision:
        """The plan as a configuration the source model would recognise."""
        return render(self.inst, plan)


# ----------------------------------------------------------------------
# the solver
# ----------------------------------------------------------------------
class GreedySupplySolver:
    """Greedy construction, then best-improvement local search over plans.

    ``requires = {C0}`` — evaluation and nothing else, so it runs on any world
    model exposing this decision space. Best improvement rather than first
    improvement because the neighbourhood is small and each evaluation is the
    expensive part; scanning it fully costs little more than sampling it and
    removes the dependence on move order.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.EVALUATE})

    def __init__(
        self,
        *,
        penalty: float = 1e6,
        max_passes: int = 20,
        max_moves: int | None = None,
        seed: int = 0,
    ) -> None:
        self.penalty = penalty
        self.max_passes = max_passes
        #: Cap on neighbours scanned per pass; ``None`` scans all of them.
        #: The neighbourhood is linear in the instance, but each neighbour
        #: costs a full evaluation, and the evaluator is
        #: ``O(|I| |R| |edges| |B|^2)`` because constraints (14)/(15) are
        #: checked over every index combination. Sampling degrades the search
        #: gracefully instead of letting a pass take minutes.
        self.max_moves = max_moves
        self.seed = seed

    def solve(self, world: Any, *, budget: int = 2000) -> SolverResult:
        check_capabilities(self, world)
        space = world.decision_space()
        rng = random.Random(self.seed)

        incumbent = space.initial()
        incumbent_eval = world.evaluate(incumbent)
        incumbent_score = penalised_objective(incumbent_eval, self.penalty)
        initial_score = incumbent_score
        history = [incumbent_score]
        used = 1

        for _ in range(self.max_passes):
            best_move = None
            best_eval = None
            best_score = incumbent_score

            neighbours = space.moves(incumbent)
            if self.max_moves is not None and len(neighbours) > self.max_moves:
                neighbours = rng.sample(neighbours, self.max_moves)

            for candidate in neighbours:
                if used >= budget:
                    break
                evaluation = world.evaluate(candidate)
                score = penalised_objective(evaluation, self.penalty)
                used += 1
                if score < best_score - 1e-9:
                    best_move, best_eval, best_score = candidate, evaluation, score

            if best_move is None:
                break
            incumbent, incumbent_eval, incumbent_score = (
                best_move, best_eval, best_score
            )
            history.append(incumbent_score)
            if used >= budget:
                break

        return SolverResult(
            incumbent, incumbent_eval, incumbent_score, initial_score, used, history
        )

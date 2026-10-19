"""Intralogistics as ``opsworld`` world models.

Binds an :class:`~intralogistics.data.Instance` to a model so the framework's
``evaluate(decision)`` signature is met, and supplies the decision space that
lets a generic search solver operate here at all.

Two world models over the *same* instance and the *same* decision space:

``StaticWorld``  the Wirkmodell, solver-free                            → C0
``MilpWorld``    the same algebra, exactly solvable, with duals   → C0+C2b+C3
"""

from __future__ import annotations

import random
from typing import Any

from opsworld.world import Capability, Evaluation, Sensitivity, SensitivityKind

from .data import Instance
from .decision import Decision
from .milp_model import MilpWorldModel
from .static_model import StaticWorldModel

__all__ = ["IntralogisticsDecisionSpace", "StaticWorld", "MilpWorld"]

_BINARY = ("x_store", "x_pick", "x_carrier", "x_product_from_warehouse")
_INTEGER = ("y_carriers", "y_reloc_moves")


class IntralogisticsDecisionSpace:
    """Configuration space: where to store, where to pick, which carrier, how many.

    ``initial`` returns a **feasible but over-provisioned** configuration, not
    the empty one. That is not a convenience — it is forced by the structure of
    the problem:

    From the empty configuration *no single move improves anything*, because
    raising ``q_inventory`` reduces the violation of (8) inventory balance by
    exactly the amount it adds to (9) store-binary and (21) carrier-capacity.
    Every neighbour is worse, in every variable family; measured, not guessed
    (0 improving candidates out of 2000). Local search from there cannot start
    — reaching feasibility needs coordinated multi-variable moves, i.e. a
    constructive heuristic, which is what this is.

    The construction deliberately over-provisions carriers and relocation
    moves, so a local search has real slack to remove and the improvement it
    achieves is genuine rather than rigged.
    """

    #: extra load carriers per (item, warehouse) beyond what inventory needs.
    CARRIER_SLACK = 2
    #: extra relocation moves per (item, resource) beyond what demand needs.
    RELOCATION_SLACK = 1

    def __init__(self, instance: Instance) -> None:
        self.inst = instance
        i_l = [(i, l) for i in instance.I for l in instance.L_sys]
        self.keys: dict[str, list[tuple[Any, ...]]] = {
            "x_store": i_l,
            "q_inventory": i_l,
            "x_pick": [(i, r, l) for i in instance.I for r in instance.R
                       for l in instance.L_sys],
            "x_carrier": [(i, l, b) for i, l in i_l for b in instance.B],
            "y_carriers": [(i, l, b) for i, l in i_l for b in instance.B],
            "x_product_from_warehouse": [(p, l) for p in instance.P
                                         for l in instance.L_sys],
            "y_reloc_moves": [(i, r, l1, l2, b1, b2)
                              for i in instance.I for r in instance.R
                              for l1, l2 in instance.U_edges
                              for b1 in instance.B for b2 in instance.B],
        }

    def empty(self) -> Decision:
        """The do-nothing configuration. Infeasible, and a dead end for local
        search — see the class docstring."""
        return Decision()

    def initial(self) -> Decision:
        """A feasible, deliberately over-provisioned configuration.

        Follows the forced structure of the problem: constraints (6)+(7) admit
        exactly one pick warehouse per (item, resource), so inventory, carriers
        and replenishment all have to sit there.
        """
        inst = self.inst
        d = Decision()
        import math

        # (6)+(7): the pick warehouse is dictated by the resource assignment.
        pick_warehouses: dict[Any, list[Any]] = {i: [] for i in inst.I}
        for i in inst.I:
            for r in inst.R:
                if inst.param("demand_item_resource", i, r) <= 0:
                    continue
                for l in inst.L_sys:
                    if inst.param("warehouse_assigned_to_resource", r, l):
                        d.x_pick[(i, r, l)] = 1.0
                        if l not in pick_warehouses[i]:
                            pick_warehouses[i].append(l)
                        break

        # (8): spread the item's total inventory over its pick warehouses.
        for i in inst.I:
            warehouses = pick_warehouses[i] or inst.L_sys[:1]
            share = inst.param("avg_inventory_total", i) / len(warehouses)
            for l in warehouses:
                d.q_inventory[(i, l)] = share
                d.x_store[(i, l)] = 1.0

                # (19)+(20)+(11): one allowed carrier whose double capacity the
                # inventory covers -- the Kanban two-bin rule binds the choice.
                allowed = [
                    b for b in inst.B
                    if inst.param("carrier_allowed", l, b)
                    and 2 * inst.param("carrier_capacity", i, b) <= share
                ]
                if not allowed:
                    continue
                b = min(allowed, key=lambda b: inst.param("carrier_capacity", i, b))
                d.x_carrier[(i, l, b)] = 1.0
                # (21)+(22): enough carriers to hold the inventory, plus slack.
                needed = math.ceil(share / inst.param("carrier_capacity", i, b))
                cap = math.ceil(
                    inst.param("avg_inventory_total", i)
                    / inst.param("carrier_capacity", i, b)
                )
                d.y_carriers[(i, l, b)] = float(min(needed + self.CARRIER_SLACK, cap))

        # (12): a product uses every warehouse any of its items is picked from.
        for p in inst.P:
            for i in inst.items_of_product(p):
                for l in pick_warehouses[i]:
                    d.x_product_from_warehouse[(p, l)] = 1.0

        # (17)+(18): replenish each pick warehouse straight from goods receipt.
        for i in inst.I:
            for r in inst.R:
                demand = inst.param("demand_item_resource", i, r)
                if demand <= 0:
                    continue
                for l in inst.L_sys:
                    if not d.x_pick.get((i, r, l)):
                        continue
                    carriers = [b for b in inst.B if d.x_carrier.get((i, l, b))]
                    if not carriers:
                        continue
                    b = carriers[0]
                    moves = math.ceil(demand / inst.param("carrier_capacity", i, b))
                    for l_we in inst.L_we:
                        if (l_we, l) in inst.U_edges:
                            d.y_reloc_moves[(i, r, l_we, l, b, b)] = float(
                                moves + self.RELOCATION_SLACK
                            )
        return d

    def perturb(self, decision: Decision, rng: random.Random, moves: int = 1) -> Decision:
        """Flip a binary, nudge an integer, or adjust an inventory level.

        One move by default. With several, a candidate that removes a surplus
        carrier (worth ~1e5 here) is almost always paired with a move that
        breaks feasibility, and the 1e6-per-unit penalty swamps the gain — so
        a multi-move neighbourhood rejects essentially everything.
        """
        new = Decision(**{name: dict(getattr(decision, name))
                          for name in Decision.VARIABLE_NAMES})
        for _ in range(moves):
            name = rng.choice(Decision.VARIABLE_NAMES)
            keys = self.keys[name]
            if not keys:
                continue
            key = rng.choice(keys)
            table = getattr(new, name)
            current = table.get(key, 0.0)
            if name in _BINARY:
                table[key] = 0.0 if current > 0.5 else 1.0
            elif name in _INTEGER:
                table[key] = max(0.0, current + rng.choice((-1.0, 1.0)))
            else:  # q_inventory
                table[key] = max(0.0, current + rng.gauss(0.0, 8.0))
        return new


class StaticWorld:
    """The Wirkmodell as an ``opsworld`` world model (C0)."""

    def __init__(self, instance: Instance) -> None:
        self.inst = instance
        self._model = StaticWorldModel()
        self._space = IntralogisticsDecisionSpace(instance)

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE}

    def decision_space(self) -> IntralogisticsDecisionSpace:
        return self._space

    def evaluate(self, decision: Decision) -> Evaluation:
        return self._model.evaluate(self.inst, decision)


class MilpWorld:
    """The Entscheidungsmodell as an ``opsworld`` world model (C0 + C2b + C3)."""

    def __init__(self, instance: Instance) -> None:
        self.inst = instance
        self._model = MilpWorldModel(instance)
        self._space = IntralogisticsDecisionSpace(instance)

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE, Capability.DUALS, Capability.ALGEBRAIC}

    def decision_space(self) -> IntralogisticsDecisionSpace:
        return self._space

    def evaluate(self, decision: Decision) -> Evaluation:
        return self._model.evaluate(self.inst, decision)

    def solve_exact(
        self, time_limit: float | None = None, solver: str | None = None
    ) -> tuple[Decision | None, float | None]:
        """Branch and bound. ``(None, None)`` when nothing feasible was found.

        The optional deadline is what makes this usable from a service: the
        model is exponential in the item count and stops returning at eight
        items, so an exact solve without a wall clock is a request that may
        never come back.
        """
        return self._model.solve(solver=solver, time_limit=time_limit)

    @property
    def solver_name(self) -> str | None:
        """MILP backend used by the last exact solve (for example HiGHS)."""
        return self._model.solver_name

    @property
    def termination(self) -> str | None:
        """Why the last exact solve stopped — ``optimal`` proves optimality."""
        return self._model.termination

    def duals(self) -> Sensitivity:
        """Shadow prices of the LP relaxation.

        Marked ``DUAL`` rather than ``PATHWISE`` on purpose: these price the
        *constraints*, not the decisions, and must not be stepped along
        (``PLAN.md`` D10). :meth:`Sensitivity.descent_direction` refuses them.
        """
        _, duals = self._model.lp_relaxation_duals()
        return Sensitivity(
            kind=SensitivityKind.DUAL,
            names=list(duals),
            values=[duals[k] for k in duals],
        )

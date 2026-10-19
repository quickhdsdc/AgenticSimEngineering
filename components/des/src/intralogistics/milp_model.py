"""MILP world model — the *Entscheidungsmodell*, ported to Pyomo.

Ported from Lena Langenfeld's ``gurobipy`` notebook. The algebra is unchanged;
only the modelling layer differs, so that the model is **solver-agnostic** and
runs with no commercial licence:

    HiGHS (open source, default) · SCIP · CBC · Gurobi

Capabilities: **C0** (evaluation), **C2b** (duals of the LP relaxation, i.e.
shadow prices), **C3** (algebraic structure) — see ``PLAN.md`` §1.6.

Constraint families carry the same names as in
:mod:`intralogistics.static_model`, so the two implementations can be compared
family by family.

.. note::
   ``evaluate()`` here computes cost and feasibility through **Pyomo's own
   expression algebra**, which is an implementation path independent of the
   numpy loops in :mod:`intralogistics.static_model`. The cross-validation in
   the test suite exploits that independence.
"""

from __future__ import annotations

import math
from typing import Any

import pyomo.environ as pyo

from .data import Instance
from .decision import Decision, Evaluation

PROVENANCE = "milp/pyomo (T0, time-averaged)"

DEFAULT_SOLVERS: tuple[str, ...] = ("appsi_highs", "highs", "scip", "cbc", "gurobi")

# Pyomo does not normalise native time-limit option names.  APPSI exposes a
# typed config entry; command-line plugins retain their solver-native spelling.
_TIME_LIMIT_OPTIONS: dict[str, str] = {
    "scip": "limits/time",
    "cbc": "seconds",
    "gurobi": "TimeLimit",
}

#: Constraint families, in source-model order.
FAMILIES: tuple[str, ...] = (
    "06_one_pick_location",
    "06_no_pick_if_no_demand",
    "07_pick_warehouse_assignment",
    "08_inventory_balance",
    "09_store_binary_upper",
    "10_min_inventory_if_picked",
    "11_kanban_two_carriers",
    "12_product_warehouse_link",
    "14_reloc_start_carrier",
    "15_reloc_target_carrier",
    "17_replenishment_covers_demand",
    "17b_intermediate_inflow",
    "17b_intermediate_outflow",
    "17b_no_bypass",
    "18_incoming_quantity_limit",
    "19_one_carrier_if_stored",
    "20_carrier_allowed",
    "21_carrier_capacity_covers_inventory",
    "22_carrier_count_only_if_selected",
    "23_warehouse_capacity",
    "24_transport_capacity",
    "25_staging_area_avg",
    "26_staging_area_max_lot",
    "27_value_limit_point_of_use",
)


def available_solvers(candidates: tuple[str, ...] = DEFAULT_SOLVERS) -> tuple[str, ...]:
    """Names of all usable MILP backends, in automatic-selection order."""
    usable: list[str] = []
    for name in candidates:
        try:
            opt = pyo.SolverFactory(name)
            if opt is not None and opt.available(exception_flag=False):
                usable.append(name)
        except Exception:  # noqa: BLE001 - probing, any failure means "unusable"
            continue
    return tuple(usable)


def available_solver(candidates: tuple[str, ...] = DEFAULT_SOLVERS) -> str | None:
    """Name of the first usable solver, or ``None`` if none is installed."""
    usable = available_solvers(candidates)
    return usable[0] if usable else None


class MilpWorldModel:
    """Pyomo/HiGHS formulation of the intralogistics network design problem."""

    def __init__(self, instance: Instance):
        self.inst = instance
        self.model: pyo.ConcreteModel | None = None
        self.solver_name: str | None = None
        self.termination: str | None = None

    def capabilities(self) -> set[str]:
        return {"C0", "C2b", "C3"}

    # ==================================================================
    # C3 — algebraic structure
    # ==================================================================
    def build(self) -> pyo.ConcreteModel:
        inst = self.inst
        m = pyo.ConcreteModel(name="intralogistics_network_design")

        IL = [(i, l) for i in inst.I for l in inst.L_sys]
        PL = [(p, l) for p in inst.P for l in inst.L_sys]
        IRL = [(i, r, l) for i in inst.I for r in inst.R for l in inst.L_sys]
        ILB = [(i, l, b) for i in inst.I for l in inst.L_sys for b in inst.B]
        IR = [(i, r) for i in inst.I for r in inst.R]
        IRUBB = [
            (i, r, l1, l2, b1, b2)
            for i in inst.I
            for r in inst.R
            for l1, l2 in inst.U_edges
            for b1 in inst.B
            for b2 in inst.B
        ]

        # ---- decision variables ----------------------------------------
        m.x_store = pyo.Var(IL, domain=pyo.Binary)
        m.q_inventory = pyo.Var(IL, domain=pyo.NonNegativeReals)
        m.x_product_from_warehouse = pyo.Var(PL, domain=pyo.Binary)
        m.y_reloc_moves = pyo.Var(IRUBB, domain=pyo.NonNegativeIntegers)
        m.x_pick = pyo.Var(IRL, domain=pyo.Binary)
        m.x_carrier = pyo.Var(ILB, domain=pyo.Binary)
        m.y_carriers = pyo.Var(ILB, domain=pyo.NonNegativeIntegers)

        # ---- cost variables — source constraints (2)-(5) are definitions -
        m.k_storage = pyo.Var(inst.L_sys, domain=pyo.NonNegativeReals)
        m.k_pick_var = pyo.Var(IR, domain=pyo.NonNegativeReals)
        m.k_pick_fix = pyo.Var(inst.P, domain=pyo.NonNegativeReals)
        m.k_reloc = pyo.Var(IR, domain=pyo.NonNegativeReals)

        # ---- objective ---------------------------------------------------
        m.obj = pyo.Objective(
            expr=(
                sum(m.k_storage[l] for l in inst.L_sys)
                + sum(m.k_pick_fix[p] for p in inst.P)
                + sum(m.k_pick_var[i, r] for i, r in IR)
                + sum(m.k_reloc[i, r] for i, r in IR)
            ),
            sense=pyo.minimize,
        )

        for family in ("02_cost_storage", "03_cost_pick_var",
                       "04_cost_pick_fix", "05_cost_reloc", *FAMILIES):
            setattr(m, family, pyo.ConstraintList())

        big_m = inst.scalar("big_m")
        n_deliveries = inst.scalar("number_of_deliveries")

        # ---- (2)-(5) cost definitions -----------------------------------
        for l in inst.L_sys:
            m.__getattribute__("02_cost_storage").add(
                m.k_storage[l]
                == inst.param("storage_cost_rate", l)
                * sum(
                    m.y_carriers[i, l, b] * inst.param("carrier_area", b)
                    for i in inst.I
                    for b in inst.B
                )
            )

        for i, r in IR:
            m.__getattribute__("03_cost_pick_var").add(
                m.k_pick_var[i, r]
                == sum(
                    m.x_pick[i, r, l]
                    * inst.param("demand_item_resource", i, r)
                    * inst.param("picking_cost_rate", l)
                    for l in inst.L_sys
                )
            )

        for p in inst.P:
            m.__getattribute__("04_cost_pick_fix").add(
                m.k_pick_fix[p]
                == n_deliveries
                * sum(
                    m.x_product_from_warehouse[p, l] * inst.param("access_cost_rate", l)
                    for l in inst.L_sys
                )
            )

        for i, r in IR:
            m.__getattribute__("05_cost_reloc").add(
                m.k_reloc[i, r]
                == sum(
                    m.y_reloc_moves[i, r, l1, l2, b1, b2]
                    * inst.param("relocation_cost", i, l1, l2, b1, b2)
                    for l1, l2 in inst.U_edges
                    for b1 in inst.B
                    for b2 in inst.B
                )
            )

        # ---- (6) genau ein Entnahmelager --------------------------------
        for i, r in IR:
            if inst.param("demand_item_resource", i, r) > 0:
                m.__getattribute__("06_one_pick_location").add(
                    sum(m.x_pick[i, r, l] for l in inst.L_sys) == 1
                )
            else:
                for l in inst.L_sys:
                    m.__getattribute__("06_no_pick_if_no_demand").add(
                        m.x_pick[i, r, l] == 0
                    )

        # ---- (7) Zuordnung Entnahmelager zur Ressource -------------------
        for i, r, l in IRL:
            m.__getattribute__("07_pick_warehouse_assignment").add(
                m.x_pick[i, r, l]
                <= inst.param("warehouse_assigned_to_resource", r, l)
            )

        # ---- (8) Bestandsverteilung --------------------------------------
        for i in inst.I:
            m.__getattribute__("08_inventory_balance").add(
                sum(m.q_inventory[i, l] for l in inst.L_sys)
                == inst.param("avg_inventory_total", i)
            )

        # ---- (9) Kopplung Menge / Binaervariable -------------------------
        for i, l in IL:
            m.__getattribute__("09_store_binary_upper").add(
                m.q_inventory[i, l]
                <= m.x_store[i, l] * inst.param("avg_inventory_total", i)
            )

        # ---- (10) Mindestbestand -----------------------------------------
        for i, l in IL:
            m.__getattribute__("10_min_inventory_if_picked").add(
                m.q_inventory[i, l]
                >= sum(
                    m.x_pick[i, r, l] * inst.param("min_inventory", i, l)
                    for r in inst.R
                )
            )

        # ---- (11) Kanban: zwei Ladungstraeger ----------------------------
        for i, l in IL:
            m.__getattribute__("11_kanban_two_carriers").add(
                m.q_inventory[i, l]
                >= 2
                * sum(
                    m.x_carrier[i, l, b] * inst.param("carrier_capacity", i, b)
                    for b in inst.B
                )
            )

        # ---- (12) Produkt-Lager-Zuordnung --------------------------------
        for p in inst.P:
            for i in inst.items_of_product(p):
                for r in inst.R:
                    for l in inst.L_sys:
                        m.__getattribute__("12_product_warehouse_link").add(
                            m.x_product_from_warehouse[p, l] >= m.x_pick[i, r, l]
                        )

        # ---- (13) structural: variables exist only on U_edges ------------

        # ---- (14)-(16) Ladungstraegerbindung der Umlagerung ---------------
        for i, r, l1, l2, b1, b2 in IRUBB:
            if l1 in inst.L_sys:
                m.__getattribute__("14_reloc_start_carrier").add(
                    m.y_reloc_moves[i, r, l1, l2, b1, b2]
                    <= big_m * m.x_carrier[i, l1, b1]
                )
            m.__getattribute__("15_reloc_target_carrier").add(
                m.y_reloc_moves[i, r, l1, l2, b1, b2]
                <= big_m * m.x_carrier[i, l2, b2]
            )

        # ---- (17) Steady-state Nachschubbilanz ----------------------------
        for i in inst.I:
            for r in inst.R:
                for l2 in inst.L_sys:
                    incoming = [e for e in inst.U_edges if e[1] == l2]
                    if not incoming:
                        continue
                    outgoing = [e for e in inst.U_edges if e[0] == l2]
                    m.__getattribute__("17_replenishment_covers_demand").add(
                        sum(
                            m.y_reloc_moves[i, r, l1, l2, b1, b2]
                            * inst.param("carrier_capacity", i, b2)
                            for l1, _ in incoming
                            for b1 in inst.B
                            for b2 in inst.B
                        )
                        >= inst.param("demand_item_resource", i, r) * m.x_pick[i, r, l2]
                        + sum(
                            m.y_reloc_moves[i, r, l2, l3, b1, b2]
                            * inst.param("carrier_capacity", i, b2)
                            for _, l3 in outgoing
                            for b1 in inst.B
                            for b2 in inst.B
                        )
                    )

        # ---- (17b) Zwischenlager: Zufluss, Abfluss, kein Bypass -----------
        for i in inst.I:
            for l_mid in inst.L_log:
                incoming = [e for e in inst.U_edges if e[1] == l_mid]
                outgoing = [e for e in inst.U_edges if e[0] == l_mid]

                if incoming:
                    m.__getattribute__("17b_intermediate_inflow").add(
                        sum(
                            m.y_reloc_moves[i, r, l1, l_mid, b1, b2]
                            for r in inst.R
                            for l1, _ in incoming
                            for b1 in inst.B
                            for b2 in inst.B
                        )
                        >= m.x_store[i, l_mid]
                    )
                if outgoing:
                    m.__getattribute__("17b_intermediate_outflow").add(
                        sum(
                            m.y_reloc_moves[i, r, l_mid, l2, b1, b2]
                            for r in inst.R
                            for _, l2 in outgoing
                            for b1 in inst.B
                            for b2 in inst.B
                        )
                        >= m.x_store[i, l_mid]
                    )

                stage_mid = inst.stage(l_mid)
                for l1, l2 in inst.U_edges:
                    if not (inst.stage(l1) > stage_mid > inst.stage(l2)):
                        continue
                    for r in inst.R:
                        m.__getattribute__("17b_no_bypass").add(
                            sum(
                                m.y_reloc_moves[i, r, l1, l2, b1, b2]
                                for b1 in inst.B
                                for b2 in inst.B
                            )
                            <= big_m * (1 - m.x_store[i, l_mid])
                        )

        # ---- (18) Wareneingangsmenge --------------------------------------
        for i in inst.I:
            for l_we in inst.L_we:
                outgoing = [e for e in inst.U_edges if e[0] == l_we]
                m.__getattribute__("18_incoming_quantity_limit").add(
                    sum(
                        m.y_reloc_moves[i, r, l_we, l2, b1, b2]
                        * inst.param("carrier_capacity", i, b2)
                        for r in inst.R
                        for _, l2 in outgoing
                        for b1 in inst.B
                        for b2 in inst.B
                    )
                    <= inst.param("incoming_quantity", i)
                )

        # ---- (19)-(22) Ladungstraeger --------------------------------------
        for i, l in IL:
            m.__getattribute__("19_one_carrier_if_stored").add(
                sum(m.x_carrier[i, l, b] for b in inst.B) == m.x_store[i, l]
            )
            m.__getattribute__("21_carrier_capacity_covers_inventory").add(
                m.q_inventory[i, l]
                <= sum(
                    m.y_carriers[i, l, b] * inst.param("carrier_capacity", i, b)
                    for b in inst.B
                )
            )
            for b in inst.B:
                m.__getattribute__("20_carrier_allowed").add(
                    m.x_carrier[i, l, b] <= inst.param("carrier_allowed", l, b)
                )
                max_carriers = math.ceil(
                    inst.param("avg_inventory_total", i)
                    / inst.param("carrier_capacity", i, b)
                )
                m.__getattribute__("22_carrier_count_only_if_selected").add(
                    m.y_carriers[i, l, b] <= m.x_carrier[i, l, b] * max_carriers
                )

        # ---- (23) Lagerkapazitaet ------------------------------------------
        for l in inst.L_sys:
            m.__getattribute__("23_warehouse_capacity").add(
                sum(
                    m.y_carriers[i, l, b] * inst.param("carrier_area", b)
                    for i in inst.I
                    for b in inst.B
                )
                <= inst.param("warehouse_capacity", l)
                * inst.param("planned_area_share", l)
            )

        # ---- (24) Transportkapazitaet ---------------------------------------
        m.__getattribute__("24_transport_capacity").add(
            sum(
                m.y_reloc_moves[i, r, l1, l2, b1, b2] * inst.param("carrier_area", b2)
                for i, r, l1, l2, b1, b2 in IRUBB
            )
            <= inst.scalar("transport_capacity")
        )

        # ---- (25)-(26) Bereitstellflaeche ------------------------------------
        for r in inst.R:
            m.__getattribute__("25_staging_area_avg").add(
                n_deliveries
                * sum(
                    inst.param("avg_lot_size", p)
                    * inst.param("qty_bom", p, i)
                    * inst.param("item_area", i)
                    * sum(m.x_pick[i, r, l] for l in inst.L_log)
                    for p in inst.P
                    for i in inst.items_of_product(p)
                )
                <= inst.param("staging_area_capacity", r)
            )
            m.__getattribute__("26_staging_area_max_lot").add(
                sum(
                    inst.param("max_lot_size", p)
                    * inst.param("qty_bom", p, i)
                    * inst.param("item_area", i)
                    * sum(m.x_pick[i, r, l] for l in inst.L_log)
                    for p in inst.P
                    for i in inst.items_of_product(p)
                )
                <= inst.param("staging_area_capacity", r)
            )

        # ---- (27) Wertrestriktion --------------------------------------------
        for l in inst.L_prod:
            m.__getattribute__("27_value_limit_point_of_use").add(
                sum(
                    m.q_inventory[i, l] * inst.param("item_value", i) for i in inst.I
                )
                <= inst.scalar("max_value_at_point_of_use")
            )

        self.model = m
        return m

    # ==================================================================
    # solving
    # ==================================================================
    def solve(
        self,
        solver: str | None = None,
        time_limit: float | None = None,
        tee: bool = False,
    ) -> tuple[Decision | None, float | None]:
        """Solve the MILP. Returns ``(decision, objective)``.

        Returns ``(None, None)`` when no feasible solution was found.
        Raises :class:`RuntimeError` if no solver is installed.
        """
        if self.model is None:
            self.build()
        m = self.model
        assert m is not None

        name = solver or available_solver()
        if name is None:
            raise RuntimeError(
                "No MILP solver available. Install one, e.g. `uv add highspy` "
                f"(tried: {', '.join(DEFAULT_SOLVERS)})."
            )
        self.solver_name = name

        opt = pyo.SolverFactory(name)
        if opt is None or not opt.available(exception_flag=False):
            raise RuntimeError(
                f"MILP solver {name!r} is not available; installed choices: "
                f"{', '.join(available_solvers()) or 'none'}"
            )
        solve_kwargs: dict[str, Any] = {}
        if time_limit is not None:
            if name in {"appsi_highs", "highs"}:
                # The APPSI legacy wrapper creates a fresh config inside
                # solve(), so assigning opt.config.time_limit beforehand is
                # silently discarded.  Its public compatibility argument is
                # the only reliable route.
                solve_kwargs["timelimit"] = time_limit
                # A deadline may arrive before the first incumbent.  APPSI's
                # default tries to load a nonexistent solution and raises
                # before we can report maxTimeLimit, so inspect first and load
                # explicitly only when a solution exists.
                solve_kwargs["load_solutions"] = False
            else:
                opt.options[_TIME_LIMIT_OPTIONS.get(name, "time_limit")] = time_limit

        results = opt.solve(m, tee=tee, **solve_kwargs)
        self.termination = str(results.solver.termination_condition)

        if self.termination not in {"optimal", "feasible", "maxTimeLimit"}:
            return None, None

        if solve_kwargs.get("load_solutions") is False:
            if len(results.solution) == 0:
                return None, None
            m.solutions.load_from(results)

        try:
            objective = float(pyo.value(m.obj))
        except Exception:  # noqa: BLE001 - no incumbent
            return None, None
        return self.extract_decision(), objective

    def extract_decision(self) -> Decision:
        """Read the current variable values back into a :class:`Decision`."""
        m = self.model
        assert m is not None

        def values(var: Any) -> dict[tuple[Any, ...], float]:
            out: dict[tuple[Any, ...], float] = {}
            for key in var:
                value = pyo.value(var[key], exception=False)
                if value is None:
                    continue
                out[key if isinstance(key, tuple) else (key,)] = float(value)
            return out

        return Decision(
            x_store=values(m.x_store),
            q_inventory=values(m.q_inventory),
            x_product_from_warehouse=values(m.x_product_from_warehouse),
            y_reloc_moves=values(m.y_reloc_moves),
            x_pick=values(m.x_pick),
            x_carrier=values(m.x_carrier),
            y_carriers=values(m.y_carriers),
        )

    # ==================================================================
    # C0 — evaluation through Pyomo's expression algebra
    # ==================================================================
    def evaluate(self, inst: Instance, d: Decision) -> Evaluation:
        """Cost and feasibility of ``d``, computed from the algebraic model.

        Independent of :mod:`intralogistics.static_model`: the numbers come
        out of Pyomo's expression evaluation, not the numpy loops.
        """
        if inst is not self.inst:
            raise ValueError("evaluate() expects the instance this model was built on.")
        if self.model is None:
            self.build()
        m = self.model
        assert m is not None

        self._load_decision(d)

        violations: dict[str, float] = {}
        for family in FAMILIES:
            container = getattr(m, family, None)
            if container is None:
                continue
            total = 0.0
            for index in container:
                con = container[index]
                body = float(pyo.value(con.body))
                if con.has_lb():
                    total += max(0.0, float(pyo.value(con.lower)) - body)
                if con.has_ub():
                    total += max(0.0, body - float(pyo.value(con.upper)))
            if total > 0.0:
                violations[family] = total

        kpis = {
            "cost_storage": sum(float(pyo.value(m.k_storage[l])) for l in self.inst.L_sys),
            "cost_pick_var": sum(float(pyo.value(v)) for v in m.k_pick_var.values()),
            "cost_pick_fix": sum(float(pyo.value(v)) for v in m.k_pick_fix.values()),
            "cost_reloc": sum(float(pyo.value(v)) for v in m.k_reloc.values()),
        }
        return Evaluation(
            objective=sum(kpis.values()),
            kpis=kpis,
            violations=violations,
            provenance=PROVENANCE,
        )

    def _load_decision(self, d: Decision) -> None:
        """Write ``d`` into the model, deriving the cost variables (2)-(5)."""
        inst = self.inst
        m = self.model
        assert m is not None

        for name in Decision.VARIABLE_NAMES:
            var = getattr(m, name)
            for key in var:
                tup = key if isinstance(key, tuple) else (key,)
                var[key].set_value(d.get(name, *tup), skip_validation=True)

        # (2)-(5) are definitions, so they are computed, never checked.
        for l in inst.L_sys:
            m.k_storage[l].set_value(
                inst.param("storage_cost_rate", l)
                * sum(
                    d.get("y_carriers", i, l, b) * inst.param("carrier_area", b)
                    for i in inst.I
                    for b in inst.B
                ),
                skip_validation=True,
            )
        for i in inst.I:
            for r in inst.R:
                m.k_pick_var[i, r].set_value(
                    sum(
                        d.get("x_pick", i, r, l)
                        * inst.param("demand_item_resource", i, r)
                        * inst.param("picking_cost_rate", l)
                        for l in inst.L_sys
                    ),
                    skip_validation=True,
                )
                m.k_reloc[i, r].set_value(
                    sum(
                        d.get("y_reloc_moves", i, r, l1, l2, b1, b2)
                        * inst.param("relocation_cost", i, l1, l2, b1, b2)
                        for l1, l2 in inst.U_edges
                        for b1 in inst.B
                        for b2 in inst.B
                    ),
                    skip_validation=True,
                )
        for p in inst.P:
            m.k_pick_fix[p].set_value(
                inst.scalar("number_of_deliveries")
                * sum(
                    d.get("x_product_from_warehouse", p, l)
                    * inst.param("access_cost_rate", l)
                    for l in inst.L_sys
                ),
                skip_validation=True,
            )

    # ==================================================================
    # C2b — shadow prices of the LP relaxation
    # ==================================================================
    def lp_relaxation_duals(
        self, solver: str | None = None
    ) -> tuple[float, dict[str, float]]:
        """Solve the LP relaxation and return ``(objective, duals-by-family)``.

        Duals answer *"which resource constraint is binding, and what would
        relaxing it buy?"* — they are **not** gradients with respect to the
        decision variables (see ``PLAN.md`` §1.6, decision record D10).
        Integrality is relaxed, so these are the duals of the relaxation, not
        of the MILP, which has no duals.
        """
        if self.model is None:
            self.build()
        m = self.model
        assert m is not None

        relaxed: list[Any] = []
        for name in ("x_store", "x_product_from_warehouse", "x_pick", "x_carrier"):
            var = getattr(m, name)
            for key in var:
                if var[key].domain is pyo.Binary:
                    var[key].domain = pyo.UnitInterval
                    relaxed.append(var[key])
        for name in ("y_reloc_moves", "y_carriers"):
            var = getattr(m, name)
            for key in var:
                if var[key].domain is pyo.NonNegativeIntegers:
                    var[key].domain = pyo.NonNegativeReals
                    relaxed.append(var[key])

        had_dual = hasattr(m, "dual")
        if not had_dual:
            m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        try:
            name = solver or available_solver()
            if name is None:
                raise RuntimeError("No LP solver available.")
            pyo.SolverFactory(name).solve(m)
            objective = float(pyo.value(m.obj))

            duals: dict[str, float] = {}
            for family in FAMILIES:
                container = getattr(m, family, None)
                if container is None:
                    continue
                total = 0.0
                for index in container:
                    total += float(m.dual.get(container[index], 0.0))
                if abs(total) > 1e-12:
                    duals[family] = total
            return objective, duals
        finally:
            for var in relaxed:
                var.domain = (
                    pyo.Binary
                    if var.domain is pyo.UnitInterval
                    else pyo.NonNegativeIntegers
                )
            # A lingering `dual` Suffix makes the next MILP solve fail: appsi
            # tries to load duals, which a mixed-integer model does not have.
            if not had_dual:
                m.del_component(m.dual)

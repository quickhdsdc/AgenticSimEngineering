"""Static world model — Lena Langenfeld's *Wirkmodell*, made executable.

This module is the point of PLAN.md Phase 1. In the source notebook the
cause-effect model and the solver are fused: every ``addConstr`` call is
simultaneously a statement about how the world works and an instruction about
what the solver should search. Here the two are separated — this file states
only how the world works, and can be called with no solver present:

    cost, feasibility = StaticWorldModel().evaluate(instance, decision)

Capability: **C0 (evaluation)** in the capability matrix of ``PLAN.md`` §1.6.

**Which projection is this?** (architecture constraint A2)
This is the *time-averaged / steady-state* view: every parameter is a rate or
a time average, and constraint (17) is the stationarity condition
``dI/dt = inflow - outflow = 0``. It is the ``T0`` layer of the time-scale
hierarchy — the fluid limit of the operational dynamics, not a time-free
description of them. The dynamic (``T1``) counterpart is the subject of
Phase 3.

Constraint numbering follows the source model verbatim, including the German
names, so the two implementations can be compared line by line.
"""

from __future__ import annotations

import math

from .data import Instance
from .decision import Decision, Evaluation

PROVENANCE = "static/fluid (T0, time-averaged)"


def _le(lhs: float, rhs: float) -> float:
    """Violation magnitude of ``lhs <= rhs``."""
    return max(0.0, lhs - rhs)


def _ge(lhs: float, rhs: float) -> float:
    """Violation magnitude of ``lhs >= rhs``."""
    return max(0.0, rhs - lhs)


def _eq(lhs: float, rhs: float) -> float:
    """Violation magnitude of ``lhs == rhs``."""
    return abs(lhs - rhs)


class StaticWorldModel:
    """Evaluate an intralogistics configuration without any solver."""

    def capabilities(self) -> set[str]:
        return {"C0"}

    # ==================================================================
    # cost model  — source constraints (2)-(5), which are *definitions*
    # ==================================================================
    def costs(self, inst: Instance, d: Decision) -> dict[str, float]:
        """Return the four cost components of the objective."""
        # (2) Lagerkosten — storage cost
        cost_storage = sum(
            inst.param("storage_cost_rate", l)
            * sum(
                d.get("y_carriers", i, l, b) * inst.param("carrier_area", b)
                for i in inst.I
                for b in inst.B
            )
            for l in inst.L_sys
        )

        # (3) Variable Entnahmekosten — variable picking cost
        cost_pick_var = sum(
            d.get("x_pick", i, r, l)
            * inst.param("demand_item_resource", i, r)
            * inst.param("picking_cost_rate", l)
            for i in inst.I
            for r in inst.R
            for l in inst.L_sys
        )

        # (4) Fixe Entnahmekosten — fixed picking cost
        cost_pick_fix = inst.scalar("number_of_deliveries") * sum(
            d.get("x_product_from_warehouse", p, l) * inst.param("access_cost_rate", l)
            for p in inst.P
            for l in inst.L_sys
        )

        # (5) Umlagerkosten — relocation cost
        cost_reloc = sum(
            d.get("y_reloc_moves", i, r, l1, l2, b1, b2)
            * inst.param("relocation_cost", i, l1, l2, b1, b2)
            for i in inst.I
            for r in inst.R
            for l1, l2 in inst.U_edges
            for b1 in inst.B
            for b2 in inst.B
        )

        return {
            "cost_storage": cost_storage,
            "cost_pick_var": cost_pick_var,
            "cost_pick_fix": cost_pick_fix,
            "cost_reloc": cost_reloc,
        }

    # ==================================================================
    # feasibility  — source constraints (6)-(27)
    # ==================================================================
    def violations(self, inst: Instance, d: Decision) -> dict[str, float]:
        v: dict[str, float] = {}

        def add(name: str, amount: float) -> None:
            if amount > 0.0:
                v[name] = v.get(name, 0.0) + amount

        big_m = inst.scalar("big_m")

        # (6) genau ein Entnahmelager je (Artikel, Ressource) mit Bedarf
        for i in inst.I:
            for r in inst.R:
                picks = sum(d.get("x_pick", i, r, l) for l in inst.L_sys)
                if inst.param("demand_item_resource", i, r) > 0:
                    add("06_one_pick_location", _eq(picks, 1.0))
                else:
                    add("06_no_pick_if_no_demand", abs(picks))

        # (7) Entnahmelager muss der Ressource zugeordnet sein
        for i in inst.I:
            for r in inst.R:
                for l in inst.L_sys:
                    add(
                        "07_pick_warehouse_assignment",
                        _le(
                            d.get("x_pick", i, r, l),
                            inst.param("warehouse_assigned_to_resource", r, l),
                        ),
                    )

        # (8) Gesamtbestand wird auf Lager verteilt
        for i in inst.I:
            total = sum(d.get("q_inventory", i, l) for l in inst.L_sys)
            add("08_inventory_balance", _eq(total, inst.param("avg_inventory_total", i)))

        # (9) Kopplung Lagermenge / Lager-Binaervariable
        for i in inst.I:
            for l in inst.L_sys:
                add(
                    "09_store_binary_upper",
                    _le(
                        d.get("q_inventory", i, l),
                        d.get("x_store", i, l) * inst.param("avg_inventory_total", i),
                    ),
                )

        # (10) Mindestbestand im Entnahmelager
        for i in inst.I:
            for l in inst.L_sys:
                required = sum(
                    d.get("x_pick", i, r, l) * inst.param("min_inventory", i, l)
                    for r in inst.R
                )
                add("10_min_inventory_if_picked", _ge(d.get("q_inventory", i, l), required))

        # (11) Kanban: Bestand deckt zwei Ladungstraeger
        for i in inst.I:
            for l in inst.L_sys:
                required = 2.0 * sum(
                    d.get("x_carrier", i, l, b) * inst.param("carrier_capacity", i, b)
                    for b in inst.B
                )
                add("11_kanban_two_carriers", _ge(d.get("q_inventory", i, l), required))

        # (12) Produkt-Lager-Zuordnung bei Entnahme
        for p in inst.P:
            for i in inst.items_of_product(p):
                for r in inst.R:
                    for l in inst.L_sys:
                        add(
                            "12_product_warehouse_link",
                            _ge(
                                d.get("x_product_from_warehouse", p, l),
                                d.get("x_pick", i, r, l),
                            ),
                        )

        # (13) Umlagerungen nur stromabwaerts.
        # In the MILP this is structural (variables exist only on U_edges);
        # here it must be checked, since a Decision may carry any key.
        admissible = set(inst.U_edges)
        for key, value in d.y_reloc_moves.items():
            if abs(value) <= 1e-9:
                continue
            if len(key) != 6 or (key[2], key[3]) not in admissible:
                add("13_downstream_only", abs(value))

        # (14)-(16) Umlagerung nur mit gewaehlten Ladungstraegern
        for i in inst.I:
            for r in inst.R:
                for l1, l2 in inst.U_edges:
                    for b1 in inst.B:
                        for b2 in inst.B:
                            moves = d.get("y_reloc_moves", i, r, l1, l2, b1, b2)
                            if l1 in inst.L_sys:
                                add(
                                    "14_reloc_start_carrier",
                                    _le(moves, big_m * d.get("x_carrier", i, l1, b1)),
                                )
                            add(
                                "15_reloc_target_carrier",
                                _le(moves, big_m * d.get("x_carrier", i, l2, b2)),
                            )

        # (17) Durchschnittlicher Nachschub deckt Bedarf und Weitergabe
        #      == Steady-state stationarity  dI/dt = 0
        for i in inst.I:
            for r in inst.R:
                for l2 in inst.L_sys:
                    incoming = [e for e in inst.U_edges if e[1] == l2]
                    if not incoming:
                        continue
                    inflow = sum(
                        d.get("y_reloc_moves", i, r, l1, l2, b1, b2)
                        * inst.param("carrier_capacity", i, b2)
                        for l1, _ in incoming
                        for b1 in inst.B
                        for b2 in inst.B
                    )
                    outflow = sum(
                        d.get("y_reloc_moves", i, r, l2, l3, b1, b2)
                        * inst.param("carrier_capacity", i, b2)
                        for _, l3 in (e for e in inst.U_edges if e[0] == l2)
                        for b1 in inst.B
                        for b2 in inst.B
                    )
                    demand = inst.param("demand_item_resource", i, r) * d.get(
                        "x_pick", i, r, l2
                    )
                    add("17_replenishment_covers_demand", _ge(inflow, demand + outflow))

        # (17b) Genutztes Zwischenlager braucht Zu- und Abfluss; kein Bypass
        for i in inst.I:
            for l_mid in inst.L_log:
                incoming = [e for e in inst.U_edges if e[1] == l_mid]
                outgoing = [e for e in inst.U_edges if e[0] == l_mid]
                stored = d.get("x_store", i, l_mid)

                if incoming:
                    inflow = sum(
                        d.get("y_reloc_moves", i, r, l1, l_mid, b1, b2)
                        for r in inst.R
                        for l1, _ in incoming
                        for b1 in inst.B
                        for b2 in inst.B
                    )
                    add("17b_intermediate_inflow", _ge(inflow, stored))
                if outgoing:
                    outflow = sum(
                        d.get("y_reloc_moves", i, r, l_mid, l2, b1, b2)
                        for r in inst.R
                        for _, l2 in outgoing
                        for b1 in inst.B
                        for b2 in inst.B
                    )
                    add("17b_intermediate_outflow", _ge(outflow, stored))

                stage_mid = inst.stage(l_mid)
                for l1, l2 in inst.U_edges:
                    if not (inst.stage(l1) > stage_mid > inst.stage(l2)):
                        continue
                    for r in inst.R:
                        bypass = sum(
                            d.get("y_reloc_moves", i, r, l1, l2, b1, b2)
                            for b1 in inst.B
                            for b2 in inst.B
                        )
                        add("17b_no_bypass", _le(bypass, big_m * (1.0 - stored)))

        # (18) Zugangsmenge aus dem Wareneingang begrenzt
        for i in inst.I:
            for l_we in inst.L_we:
                outgoing = [e for e in inst.U_edges if e[0] == l_we]
                shipped = sum(
                    d.get("y_reloc_moves", i, r, l_we, l2, b1, b2)
                    * inst.param("carrier_capacity", i, b2)
                    for r in inst.R
                    for _, l2 in outgoing
                    for b1 in inst.B
                    for b2 in inst.B
                )
                add(
                    "18_incoming_quantity_limit",
                    _le(shipped, inst.param("incoming_quantity", i)),
                )

        # (19) genau ein Ladungstraegertyp, wenn eingelagert
        for i in inst.I:
            for l in inst.L_sys:
                chosen = sum(d.get("x_carrier", i, l, b) for b in inst.B)
                add("19_one_carrier_if_stored", _eq(chosen, d.get("x_store", i, l)))

        # (20) Ladungstraegerwahl gemaess Lagerrestriktion
        for i in inst.I:
            for l in inst.L_sys:
                for b in inst.B:
                    add(
                        "20_carrier_allowed",
                        _le(
                            d.get("x_carrier", i, l, b),
                            inst.param("carrier_allowed", l, b),
                        ),
                    )

        # (21) Anzahl Ladungstraeger deckt den Bestand
        for i in inst.I:
            for l in inst.L_sys:
                capacity = sum(
                    d.get("y_carriers", i, l, b) * inst.param("carrier_capacity", i, b)
                    for b in inst.B
                )
                add(
                    "21_carrier_capacity_covers_inventory",
                    _le(d.get("q_inventory", i, l), capacity),
                )

        # (22) Behaelteranzahl nur fuer gewaehlten Traegertyp
        for i in inst.I:
            for l in inst.L_sys:
                for b in inst.B:
                    max_carriers = math.ceil(
                        inst.param("avg_inventory_total", i)
                        / inst.param("carrier_capacity", i, b)
                    )
                    add(
                        "22_carrier_count_only_if_selected",
                        _le(
                            d.get("y_carriers", i, l, b),
                            d.get("x_carrier", i, l, b) * max_carriers,
                        ),
                    )

        # (23) Lagerkapazitaet (Flaeche)
        for l in inst.L_sys:
            used = sum(
                d.get("y_carriers", i, l, b) * inst.param("carrier_area", b)
                for i in inst.I
                for b in inst.B
            )
            available = inst.param("warehouse_capacity", l) * inst.param(
                "planned_area_share", l
            )
            add("23_warehouse_capacity", _le(used, available))

        # (24) Transportkapazitaet
        transported = sum(
            d.get("y_reloc_moves", i, r, l1, l2, b1, b2) * inst.param("carrier_area", b2)
            for l1, l2 in inst.U_edges
            for i in inst.I
            for r in inst.R
            for b1 in inst.B
            for b2 in inst.B
        )
        add("24_transport_capacity", _le(transported, inst.scalar("transport_capacity")))

        # (25) Bereitstellflaeche, durchschnittliche Losgroesse
        for r in inst.R:
            used = inst.scalar("number_of_deliveries") * sum(
                inst.param("avg_lot_size", p)
                * inst.param("qty_bom", p, i)
                * inst.param("item_area", i)
                * sum(d.get("x_pick", i, r, l) for l in inst.L_log)
                for p in inst.P
                for i in inst.items_of_product(p)
            )
            add(
                "25_staging_area_avg",
                _le(used, inst.param("staging_area_capacity", r)),
            )

        # (26) Bereitstellflaeche, maximale Losgroesse
        for r in inst.R:
            used = sum(
                inst.param("max_lot_size", p)
                * inst.param("qty_bom", p, i)
                * inst.param("item_area", i)
                * sum(d.get("x_pick", i, r, l) for l in inst.L_log)
                for p in inst.P
                for i in inst.items_of_product(p)
            )
            add(
                "26_staging_area_max_lot",
                _le(used, inst.param("staging_area_capacity", r)),
            )

        # (27) Wertrestriktion am Verbauort
        for l in inst.L_prod:
            value = sum(
                d.get("q_inventory", i, l) * inst.param("item_value", i) for i in inst.I
            )
            add(
                "27_value_limit_point_of_use",
                _le(value, inst.scalar("max_value_at_point_of_use")),
            )

        return v

    # ==================================================================
    # C0 entry point
    # ==================================================================
    def evaluate(self, inst: Instance, d: Decision) -> Evaluation:
        """Cost and feasibility of ``d`` under ``inst`` — no solver involved."""
        costs = self.costs(inst, d)
        return Evaluation(
            objective=sum(costs.values()),
            kpis=costs,
            violations=self.violations(inst, d),
            provenance=PROVENANCE,
        )

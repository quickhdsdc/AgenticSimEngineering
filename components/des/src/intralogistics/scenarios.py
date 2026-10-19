"""Scenario technique — perturb an instance, re-solve, compare.

Scenarios are the robustness mechanism of the source dissertation: solve the
deterministic model once per demand/cost scenario, then derive interpretable
rules from the family of optima.

.. warning::
   Scenario studies only capture **level shifts** of the parameters, not
   **variability**. Safety stock and Kanban card counts exist because of
   variability, so a static scenario sweep cannot speak to them. That gap is
   what the ``T1`` dynamic world model addresses (``PLAN.md`` Phase 3).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .data import Instance
from .decision import Decision
from .milp_model import MilpWorldModel


@dataclass
class Scenario:
    """A named perturbation of a base instance's data."""

    name: str
    storage_cost_multiplier_by_stage: dict[Any, float] = field(default_factory=dict)
    storage_cost_multiplier_by_warehouse: dict[Any, float] = field(default_factory=dict)
    warehouse_capacity_multiplier: dict[Any, float] = field(default_factory=dict)
    carrier_capacity_multiplier: dict[Any, float] = field(default_factory=dict)
    avg_inventory_multiplier: dict[Any, float] = field(default_factory=dict)
    relocation_cost_multiplier: float | None = None
    transport_capacity: float | None = None
    set_values: dict[str, Any] = field(default_factory=dict)

    def apply(self, base: Instance) -> Instance:
        """Return a new :class:`Instance` with this scenario applied."""
        data = deepcopy(base.data)

        for stage, factor in self.storage_cost_multiplier_by_stage.items():
            for warehouse in data["warehouses"]:
                if (
                    data["stage_of_warehouse"][warehouse] == stage
                    and warehouse in data["storage_cost_rate"]
                ):
                    data["storage_cost_rate"][warehouse] *= factor

        for warehouse, factor in self.storage_cost_multiplier_by_warehouse.items():
            data["storage_cost_rate"][warehouse] *= factor

        for warehouse, factor in self.warehouse_capacity_multiplier.items():
            data["warehouse_capacity"][warehouse] *= factor

        # NOTE: the source notebook spelled this key "carrier_capacity_mul-tiplier"
        # (stray hyphen), so the scenario silently did nothing. Fixed here.
        for carrier, factor in self.carrier_capacity_multiplier.items():
            for item in data["items"]:
                data["carrier_capacity"][item, carrier] *= factor

        for item, factor in self.avg_inventory_multiplier.items():
            data["avg_inventory_total"][item] *= factor

        if self.transport_capacity is not None:
            data["transport_capacity"] = self.transport_capacity

        if self.relocation_cost_multiplier is not None:
            factor = self.relocation_cost_multiplier
            data["relocation_cost"] = {
                key: value * factor for key, value in data["relocation_cost"].items()
            }

        for parameter_name, updates in self.set_values.items():
            if isinstance(data[parameter_name], dict) and isinstance(updates, dict):
                data[parameter_name].update(updates)
            else:
                data[parameter_name] = updates

        return Instance(data)


def default_scenarios() -> list[Scenario]:
    """A starter set of cost and capacity scenarios (from the source notebook)."""
    return [
        Scenario("baseline"),
        Scenario("stage_0_storage_x2", storage_cost_multiplier_by_stage={0: 2.0}),
        Scenario("stage_0_storage_x5", storage_cost_multiplier_by_stage={0: 5.0}),
        Scenario("relocation_cost_x2", relocation_cost_multiplier=2.0),
        Scenario("L3_capacity_50_percent", warehouse_capacity_multiplier={"L3": 0.5}),
        Scenario("transport_capacity_50", transport_capacity=50.0),
    ]


@dataclass
class ScenarioResult:
    name: str
    termination: str | None
    objective: float | None
    kpis: dict[str, float]
    decision: Decision | None
    instance: Instance

    def used_warehouses(self, tol: float = 1e-6) -> list[Any]:
        if self.decision is None:
            return []
        return sorted({l for (_, l), v in self.decision.x_store.items() if abs(v) > tol})

    def used_edges(self, tol: float = 1e-6) -> list[tuple[Any, Any]]:
        if self.decision is None:
            return []
        return sorted(
            {
                (key[2], key[3])
                for key, v in self.decision.y_reloc_moves.items()
                if abs(v) > tol
            }
        )

    def carriers_moved(self, tol: float = 1e-6) -> float:
        if self.decision is None:
            return 0.0
        return sum(v for v in self.decision.y_reloc_moves.values() if abs(v) > tol)


def run_scenario_study(
    base: Instance,
    scenarios: list[Scenario] | None = None,
    *,
    solver: str | None = None,
    time_limit: float | None = None,
) -> list[ScenarioResult]:
    """Solve every scenario and return one :class:`ScenarioResult` per entry."""
    results: list[ScenarioResult] = []
    for scenario in scenarios if scenarios is not None else default_scenarios():
        instance = scenario.apply(base)
        model = MilpWorldModel(instance)
        decision, objective = model.solve(solver=solver, time_limit=time_limit)
        kpis = (
            model.evaluate(instance, decision).kpis if decision is not None else {}
        )
        results.append(
            ScenarioResult(
                name=scenario.name,
                termination=model.termination,
                objective=objective,
                kpis=kpis,
                decision=decision,
                instance=instance,
            )
        )
    return results


def summarize(results: list[ScenarioResult]) -> str:
    """A plain-text comparison table across scenarios."""
    header = (
        f"{'scenario':<24s}{'objective':>16s}{'storage':>14s}{'pick_var':>10s}"
        f"{'reloc':>9s}  warehouses"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        if r.objective is None:
            lines.append(f"{r.name:<24s}{'(no solution)':>16s}")
            continue
        lines.append(
            f"{r.name:<24s}{r.objective:>16.4f}"
            f"{r.kpis.get('cost_storage', 0.0):>14.1f}"
            f"{r.kpis.get('cost_pick_var', 0.0):>10.2f}"
            f"{r.kpis.get('cost_reloc', 0.0):>9.2f}"
            f"  {', '.join(map(str, r.used_warehouses()))}"
        )

    solved = [r.objective for r in results if r.objective is not None]
    if len(solved) > 1:
        lo, hi = min(solved), max(solved)
        spread = (hi - lo) / lo * 100 if lo else float("nan")
        lines += [
            "-" * len(header),
            f"objective spread across scenarios: {spread:.4f}% "
            f"(min {lo:.4f}, max {hi:.4f})",
        ]
    return "\n".join(lines)

"""Instance data and derived index sets for the intralogistics model.

This is layer L1 of the architecture (see ``PLAN.md``): a problem *instance*
carries entities, parameters and demand — but no decisions and no solver.

Both world models (:mod:`intralogistics.milp_model` and
:mod:`intralogistics.static_model`) consume the same :class:`Instance`, so the
cross-validation in the test suite compares *constraint logic*, not two
re-derivations of the index sets.

Ported from Lena Langenfeld's Gurobi notebook (``_others/run_model.ipynb``);
parameter names and semantics are kept verbatim so the German constraint
numbering in the source model still applies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REQUIRED_KEYS: tuple[str, ...] = (
    "products",
    "items",
    "items_of_product",
    "qty_bom",
    "resources",
    "warehouses",
    "load_carriers",
    "stage_of_warehouse",
    "demand_item_resource",
    "avg_inventory_total",
    "item_value",
    "item_area",
    "incoming_quantity",
    "carrier_capacity",
    "carrier_allowed",
    "carrier_area",
    "storage_cost_rate",
    "picking_cost_rate",
    "access_cost_rate",
    "relocation_cost",
    "warehouse_capacity",
    "transport_capacity",
    "staging_area_capacity",
    "max_value_at_point_of_use",
    "number_of_deliveries",
    "warehouse_assigned_to_resource",
    "min_inventory",
    "planned_area_share",
    "max_lot_size",
    "avg_lot_size",
    "product_demand",
    "big_m",
)


@dataclass
class Instance:
    """One concrete problem instance: entities, parameters, derived sets.

    Attributes named with single capitals follow the notation of the source
    model:

    ``P`` products · ``I`` items · ``R`` resources · ``L`` warehouses ·
    ``B`` load carriers.

    Derived warehouse subsets, by storage stage (``stage_of_warehouse``):

    ``L_we``   goods receipt — the highest stage (Wareneingang)
    ``L_sys``  every warehouse *below* the highest stage; the decision domain
    ``L_log``  strictly intermediate stages (Logistiklager)
    ``L_prod`` the lowest stage — point of use (Produktionslager)
    ``U_edges`` admissible relocation arcs, i.e. strictly downstream pairs.
    """

    data: dict[str, Any]

    P: list[Any] = field(default_factory=list)
    I: list[Any] = field(default_factory=list)
    R: list[Any] = field(default_factory=list)
    L: list[Any] = field(default_factory=list)
    B: list[Any] = field(default_factory=list)
    L_sys: list[Any] = field(default_factory=list)
    L_we: list[Any] = field(default_factory=list)
    L_log: list[Any] = field(default_factory=list)
    L_prod: list[Any] = field(default_factory=list)
    U_edges: list[tuple[Any, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate_keys()
        self._load_sets()
        self._validate_parameters()

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    def id(self) -> str:
        return (
            f"intralogistics(P={len(self.P)},I={len(self.I)},R={len(self.R)},"
            f"L={len(self.L)},B={len(self.B)})"
        )

    # ------------------------------------------------------------------
    # parameter access
    # ------------------------------------------------------------------
    def param(self, name: str, *key: Any) -> Any:
        """Look up ``name[key]``, accepting flat-tuple or nested mappings."""
        value = self.data[name]
        if not key:
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping for indexed access.")

        flat_key = key[0] if len(key) == 1 else tuple(key)
        if flat_key in value:
            return value[flat_key]

        current = value
        for part in key:
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(f"{name}[{','.join(map(str, key))}]")
            current = current[part]
        return current

    def has_param(self, name: str, *key: Any) -> bool:
        try:
            self.param(name, *key)
        except (KeyError, TypeError):
            return False
        return True

    def scalar(self, name: str) -> Any:
        value = self.data[name]
        if isinstance(value, Mapping):
            raise ValueError(f"{name} must be a scalar value, not a mapping.")
        return value

    def items_of_product(self, p: Any) -> list[Any]:
        value = self.param("items_of_product", p)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"items_of_product[{p}] must be a non-string sequence.")
        return list(value)

    def stage(self, warehouse: Any) -> Any:
        return self.param("stage_of_warehouse", warehouse)

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _as_list(self, key: str) -> list[Any]:
        value = self.data[key]
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{key} must be a non-string sequence.")
        if not value:
            raise ValueError(f"{key} must not be empty.")
        return list(value)

    def _validate_keys(self) -> None:
        if not isinstance(self.data, Mapping):
            raise ValueError("data must be a dictionary-like mapping.")
        missing = [key for key in REQUIRED_KEYS if key not in self.data]
        if missing:
            raise ValueError(f"Missing required data keys: {missing}")

    def _load_sets(self) -> None:
        self.P = self._as_list("products")
        self.I = self._as_list("items")
        self.R = self._as_list("resources")
        self.L = self._as_list("warehouses")
        self.B = self._as_list("load_carriers")

        stages = [self.stage(l) for l in self.L]
        min_stage, max_stage = min(stages), max(stages)

        self.L_sys = [l for l in self.L if self.stage(l) < max_stage]
        self.L_we = [l for l in self.L if self.stage(l) == max_stage]
        self.L_log = [l for l in self.L if min_stage < self.stage(l) < max_stage]
        self.L_prod = [l for l in self.L if self.stage(l) == min_stage]
        self.U_edges = [
            (l1, l2)
            for l1 in self.L
            for l2 in self.L
            if self.stage(l1) > self.stage(l2)
        ]

        if not self.L_sys:
            raise ValueError(
                "Derived set L_sys is empty. At least one warehouse must have "
                "stage_of_warehouse[l] < max(stage_of_warehouse.values())."
            )

    def _validate_parameters(self) -> None:
        checks: list[tuple[str, list[tuple[Any, ...]]]] = [
            ("stage_of_warehouse", [(l,) for l in self.L]),
            ("qty_bom", [(p, i) for p in self.P for i in self.items_of_product(p)]),
            ("demand_item_resource", [(i, r) for i in self.I for r in self.R]),
            ("avg_inventory_total", [(i,) for i in self.I]),
            ("item_value", [(i,) for i in self.I]),
            ("item_area", [(i,) for i in self.I]),
            ("incoming_quantity", [(i,) for i in self.I]),
            ("carrier_capacity", [(i, b) for i in self.I for b in self.B]),
            ("carrier_allowed", [(l, b) for l in self.L for b in self.B]),
            ("carrier_area", [(b,) for b in self.B]),
            ("storage_cost_rate", [(l,) for l in self.L_sys]),
            ("picking_cost_rate", [(l,) for l in self.L_sys]),
            ("access_cost_rate", [(l,) for l in self.L_sys]),
            (
                "relocation_cost",
                [
                    (i, l1, l2, b1, b2)
                    for i in self.I
                    for l1, l2 in self.U_edges
                    for b1 in self.B
                    for b2 in self.B
                ],
            ),
            ("warehouse_capacity", [(l,) for l in self.L_sys]),
            ("staging_area_capacity", [(r,) for r in self.R]),
            (
                "warehouse_assigned_to_resource",
                [(r, l) for r in self.R for l in self.L_sys],
            ),
            ("min_inventory", [(i, l) for i in self.I for l in self.L_sys]),
            ("planned_area_share", [(l,) for l in self.L_sys]),
            ("max_lot_size", [(p,) for p in self.P]),
            ("avg_lot_size", [(p,) for p in self.P]),
            ("product_demand", [(p,) for p in self.P]),
        ]

        for name, keys in checks:
            missing = [key for key in keys if not self.has_param(name, *key)]
            if missing:
                preview = missing[:10]
                extra = "" if len(missing) <= 10 else f" and {len(missing) - 10} more"
                raise ValueError(f"Missing entries for {name}: {preview}{extra}")

        for i in self.I:
            for b in self.B:
                if self.param("carrier_capacity", i, b) <= 0:
                    raise ValueError(f"carrier_capacity[{i},{b}] must be greater than 0.")

        for p in self.P:
            unknown = [i for i in self.items_of_product(p) if i not in self.I]
            if unknown:
                raise ValueError(f"items_of_product[{p}] contains unknown items: {unknown}")
            for i in self.items_of_product(p):
                if self.param("qty_bom", p, i) <= 0:
                    raise ValueError(f"qty_bom[{p},{i}] must be greater than 0.")


# ----------------------------------------------------------------------
# minimal test instance
# ----------------------------------------------------------------------
def build_minimal_data() -> dict[str, Any]:
    """A small, complete data instance for smoke-testing the model.

    One product, two items, two resources, six warehouses over four storage
    stages, two load-carrier types. Ported verbatim from the source notebook.
    """
    products = ["P1"]
    items = ["A1", "A2"]
    resources = ["R1", "R2"]
    warehouses = ["L6", "L4", "L5", "L3", "L2", "L1"]
    load_carriers = ["B2", "B1"]

    stage_of_warehouse = {"L6": 3, "L4": 2, "L5": 2, "L3": 1, "L2": 0, "L1": 0}
    items_of_product = {"P1": ["A1", "A2"]}
    qty_bom = {("P1", "A1"): 2, ("P1", "A2"): 1}

    demand_item_resource = {
        ("A1", "R1"): 20,
        ("A1", "R2"): 10,
        ("A2", "R1"): 10,
        ("A2", "R2"): 5,
    }
    carrier_capacity = {
        ("A1", "B2"): 25,
        ("A1", "B1"): 10,
        ("A2", "B2"): 25,
        ("A2", "B1"): 10,
    }
    carrier_area = {"B2": 2.0, "B1": 1.0}

    system_warehouses = ["L4", "L5", "L3", "L2", "L1"]
    resource_delivery_warehouse = {"R1": "L1", "R2": "L2"}

    relocation_cost: dict[tuple[Any, ...], float] = {}
    for i in items:
        for l1 in warehouses:
            for l2 in warehouses:
                if stage_of_warehouse[l1] <= stage_of_warehouse[l2]:
                    continue
                for b1 in load_carriers:
                    for b2 in load_carriers:
                        carrier_change_penalty = 0.0 if b1 == b2 else 0.5
                        relocation_cost[(i, l1, l2, b1, b2)] = (
                            1.0
                            + 0.2 * abs(stage_of_warehouse[l1] - stage_of_warehouse[l2])
                            + 0.1 * carrier_area[b2]
                            + carrier_change_penalty
                        )

    return {
        "products": products,
        "items": items,
        "items_of_product": items_of_product,
        "qty_bom": qty_bom,
        "resources": resources,
        "warehouses": warehouses,
        "load_carriers": load_carriers,
        "stage_of_warehouse": stage_of_warehouse,
        "demand_item_resource": demand_item_resource,
        "avg_inventory_total": {"A1": 60, "A2": 40},
        "item_value": {"A1": 5.0, "A2": 8.0},
        "item_area": {"A1": 0.2, "A2": 0.3},
        "incoming_quantity": {"A1": 100, "A2": 100},
        "carrier_capacity": carrier_capacity,
        "carrier_allowed": {
            ("L6", "B2"): 1, ("L6", "B1"): 1,
            ("L4", "B2"): 1, ("L4", "B1"): 0,
            ("L5", "B2"): 1, ("L5", "B1"): 0,
            ("L3", "B2"): 0, ("L3", "B1"): 1,
            ("L2", "B2"): 1, ("L2", "B1"): 1,
            ("L1", "B2"): 1, ("L1", "B1"): 1,
        },
        "carrier_area": carrier_area,
        "storage_cost_rate": {
            "L4": 1.0, "L5": 1.2, "L3": 100000, "L2": 100000.0, "L1": 100000.0,
        },
        "picking_cost_rate": {"L4": 5.0, "L5": 4.0, "L3": 2.0, "L2": 1.0, "L1": 1.0},
        "access_cost_rate": {"L4": 5.0, "L5": 4.0, "L3": 2.0, "L2": 1.0, "L1": 1.0},
        "relocation_cost": relocation_cost,
        "warehouse_capacity": {l: 100.0 for l in system_warehouses},
        "transport_capacity": 1000.0,
        "staging_area_capacity": {r: 1000.0 for r in resources},
        "max_value_at_point_of_use": 1000.0,
        "number_of_deliveries": 1,
        "warehouse_assigned_to_resource": {
            (r, l): 1 if resource_delivery_warehouse.get(r) == l else 0
            for r in resources
            for l in warehouses
        },
        "min_inventory": {(i, l): 1.0 for i in items for l in system_warehouses},
        "planned_area_share": {l: 1.0 for l in system_warehouses},
        "max_lot_size": {"P1": 10},
        "avg_lot_size": {"P1": 5},
        "product_demand": {"P1": 15},
        "big_m": 1000,
    }


def build_minimal_instance() -> Instance:
    """Convenience wrapper: :func:`build_minimal_data` as an :class:`Instance`."""
    return Instance(build_minimal_data())

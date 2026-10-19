"""Synthetic instances with genuine combinatorial freedom.

:func:`~intralogistics.data.build_minimal_data` is the notebook's smoke-test
instance and is kept verbatim, because Phase 1's cross-validation is anchored
to it. It is, however, structurally **rigid**: Phase 1 traced how constraints
(6)+(7) with a one-to-one ``warehouse_assigned_to_resource`` force the pick
warehouse, which forces inventory, which forces a carrier, which forces the
storage cost. There is essentially one configuration to find.

That rigidity is fatal for anything that studies *how the optimum moves*.
Measured on it, the argmin is unchanged over a ``1e-4``-to-``3`` range of
smoothing scales, so a perturbed gradient (Phase 5.3) can only ever report the
solver's tie-breaking. The instance below restores the choices the source model
is actually about:

* **where to pick from** — a resource may be served either from its own point
  of use or directly from a forward warehouse, so constraint (6) chooses
  instead of merely recording. That choice trades point-of-use storage cost
  against the staging-area limits (25)/(26);
* **which carrier size** — three sizes rather than two, with the large ones
  barred from the points of use, so constraint (11)'s two-bin rule bites
  differently in different places;
* **which route** — two parallel forward warehouses at the same stage, so
  relocation has alternatives rather than a single admissible arc;
* **capacity pressure** — warehouse and staging limits are set to bind for
  some items, which is what makes the choice non-separable.

Nothing here is new modelling: the algebra is exactly the source model's, and
:mod:`intralogistics.static_model` and :mod:`intralogistics.milp_model` both
consume it unchanged. Only the numbers are chosen to leave the decisions open.
"""

from __future__ import annotations

from typing import Any

from .data import Instance

__all__ = ["build_network_data", "build_network_instance"]

#: Load carriers, as ``(name, capacity multiplier, footprint)``.
#: Capacity is per item, scaled by that item's lot size, so "small" means the
#: same thing for a bolt and a battery module.
_CARRIERS: tuple[tuple[str, float, float], ...] = (
    ("KLT", 0.5, 0.6),    # small bin
    ("GLT", 1.0, 1.4),    # standard box
    ("PAL", 2.0, 2.4),    # pallet
)


def build_network_data(
    *,
    n_items: int = 4,
    n_resources: int = 2,
    n_forward: int = 2,
    seed: int = 0,
    storage_rate_point_of_use: float = 4.0,
    tighten: float = 1.0,
) -> dict[str, Any]:
    """A supply network with live choices at every decision family.

    Layout (stage in brackets)::

        WE [3]  goods receipt
          |
        ZL [2]  central warehouse
         / \\
      VL1  VL2 [1]  two parallel forward warehouses
         \\ /
        PL1 .. PLn [0]  one point of use per resource

    ``tighten`` scales the warehouse and staging capacities: ``1.0`` leaves
    them binding for some items, larger values relax them, smaller ones make
    the instance harder and eventually infeasible.

    ``seed`` varies demand, value and size across items deterministically —
    a small LCG rather than numpy, so an instance is reproducible without
    carrying a generator state around.
    """
    if n_items < 1 or n_resources < 1:
        raise ValueError("need at least one item and one resource")

    rand = _Lcg(seed)

    products = [f"P{k + 1}" for k in range((n_items + 1) // 2)]
    items = [f"A{k + 1}" for k in range(n_items)]
    resources = [f"R{k + 1}" for k in range(n_resources)]
    points_of_use = [f"PL{k + 1}" for k in range(n_resources)]
    forward = [f"VL{k + 1}" for k in range(n_forward)]
    warehouses = ["WE", "ZL", *forward, *points_of_use]
    carriers = [name for name, _, _ in _CARRIERS]

    stage_of_warehouse: dict[str, int] = {"WE": 3, "ZL": 2}
    stage_of_warehouse.update({l: 1 for l in forward})
    stage_of_warehouse.update({l: 0 for l in points_of_use})

    # -- bill of materials: items are dealt round-robin to products ------
    # There are ceil(n_items / 2) products, so a round robin always leaves every
    # product with at least one item -- the data model rejects an empty one.
    items_of_product: dict[str, list[str]] = {p: [] for p in products}
    for index, item in enumerate(items):
        items_of_product[products[index % len(products)]].append(item)
    qty_bom = {
        (p, i): 1 + (index % 3)
        for p, group in items_of_product.items()
        for index, i in enumerate(group)
    }

    # -- per-item character ----------------------------------------------
    # lot size spans an order of magnitude, so carrier sizing is a real
    # per-item decision rather than one global choice.
    lot_size = {i: 5.0 * (2 ** (rand.next() % 4)) for i in items}
    item_value = {i: round(0.5 + 12.0 * rand.uniform(), 2) for i in items}
    item_area = {i: round(0.05 + 0.35 * rand.uniform(), 3) for i in items}

    demand_item_resource: dict[tuple[str, str], float] = {}
    for i in items:
        # every item is consumed somewhere, but not everywhere
        served = [r for r in resources if rand.uniform() < 0.75] or [resources[0]]
        for r in resources:
            demand_item_resource[(i, r)] = (
                round(0.4 * lot_size[i] * (0.5 + rand.uniform()), 2)
                if r in served
                else 0.0
            )

    carrier_capacity = {
        (i, name): max(1.0, round(multiplier * lot_size[i], 2))
        for i in items
        for name, multiplier, _ in _CARRIERS
    }
    carrier_area = {name: area for name, _, area in _CARRIERS}

    # Total stock is enough for two-bin Kanban (11) on the standard box at up
    # to three locations -- so splitting inventory is possible but not free.
    avg_inventory_total = {i: round(6.0 * lot_size[i], 2) for i in items}
    incoming_quantity = {
        i: round(4.0 * sum(demand_item_resource[(i, r)] for r in resources) + 10.0, 2)
        for i in items
    }

    # -- the two rate ladders that pull against each other ----------------
    # Storage gets dearer towards the line, picking gets cheaper: the classic
    # forward-vs-central trade-off, and the reason (6) is a real decision.
    storage_cost_rate: dict[str, float] = {"ZL": 0.6}
    picking_cost_rate: dict[str, float] = {"ZL": 0.30}
    access_cost_rate: dict[str, float] = {"ZL": 6.0}
    for k, l in enumerate(forward):
        storage_cost_rate[l] = round(1.0 + 0.3 * k, 3)
        picking_cost_rate[l] = round(0.18 - 0.02 * k, 3)
        access_cost_rate[l] = round(3.0 + 0.5 * k, 3)
    storage_cost_rate.update({l: storage_rate_point_of_use for l in points_of_use})
    picking_cost_rate.update({l: 0.05 for l in points_of_use})
    access_cost_rate.update({l: 1.0 for l in points_of_use})

    # -- carrier admissibility -------------------------------------------
    # Pallets never reach the line; small bins are not worth a slot centrally.
    carrier_allowed: dict[tuple[str, str], int] = {}
    for l in warehouses:
        for name in carriers:
            if stage_of_warehouse[l] == 0:
                allowed = name != "PAL"
            elif l == "ZL":
                allowed = name != "KLT"
            else:
                allowed = True
            carrier_allowed[(l, name)] = int(allowed)

    # -- relocation cost: distance plus a carrier-change penalty -----------
    relocation_cost: dict[tuple[Any, ...], float] = {}
    for i in items:
        for l1 in warehouses:
            for l2 in warehouses:
                if stage_of_warehouse[l1] <= stage_of_warehouse[l2]:
                    continue
                hops = stage_of_warehouse[l1] - stage_of_warehouse[l2]
                detour = 0.15 * max(
                    (forward.index(l) for l in (l1, l2) if l in forward), default=0
                )
                for b1 in carriers:
                    for b2 in carriers:
                        relocation_cost[(i, l1, l2, b1, b2)] = round(
                            0.8
                            + 0.35 * hops
                            + 0.10 * carrier_area[b2]
                            + detour
                            + (0.0 if b1 == b2 else 0.45),
                            4,
                        )

    # -- who may serve whom ------------------------------------------------
    # Each resource has its own point of use *and* may be supplied straight
    # from either forward warehouse. That is the choice constraint (6) makes.
    warehouse_assigned_to_resource: dict[tuple[str, str], int] = {}
    for index, r in enumerate(resources):
        own = points_of_use[index]
        for l in warehouses:
            warehouse_assigned_to_resource[(r, l)] = int(l in (own, *forward))

    # -- capacities --------------------------------------------------------
    total_area = sum(
        avg_inventory_total[i] / carrier_capacity[(i, "GLT")] * carrier_area["GLT"]
        for i in items
    )
    warehouse_capacity = {"ZL": tighten * 0.9 * total_area}
    warehouse_capacity.update(
        {l: tighten * 0.9 / max(1, n_forward) * total_area for l in forward}
    )
    warehouse_capacity.update({l: tighten * 0.30 * total_area for l in points_of_use})

    # (14)/(15) and (17b) are big-M constraints. The source instance ships
    # ``big_m = 1000`` against move counts of order ten, which makes the LP
    # relaxation useless and the branch-and-bound tree enormous. A valid bound
    # is the largest move count any single (item, resource) can need: every
    # unit of that item's demand, carried in the smallest admissible bin, with
    # room to spare for multi-hop routing.
    max_moves = max(
        sum(demand_item_resource[(i, r)] for r in resources)
        / min(carrier_capacity[(i, name)] for name in carriers)
        for i in items
    )
    big_m = float(int(4 * max_moves) + 8)

    product_demand = {p: 20.0 for p in products}
    avg_lot_size = {p: 4.0 for p in products}
    max_lot_size = {p: 8.0 for p in products}

    # (25)/(26) price picking from a *logistics* warehouse in staging area, so
    # this is the limit that makes "just pick everything from VL1" impossible.
    staging_demand = sum(
        max_lot_size[p] * qty_bom[(p, i)] * item_area[i]
        for p, group in items_of_product.items()
        for i in group
    )
    staging_area_capacity = {r: tighten * 0.7 * staging_demand for r in resources}

    return {
        "products": products,
        "items": items,
        "items_of_product": items_of_product,
        "qty_bom": qty_bom,
        "resources": resources,
        "warehouses": warehouses,
        "load_carriers": carriers,
        "stage_of_warehouse": stage_of_warehouse,
        "demand_item_resource": demand_item_resource,
        "avg_inventory_total": avg_inventory_total,
        "item_value": item_value,
        "item_area": item_area,
        "incoming_quantity": incoming_quantity,
        "carrier_capacity": carrier_capacity,
        "carrier_allowed": carrier_allowed,
        "carrier_area": carrier_area,
        "storage_cost_rate": storage_cost_rate,
        "picking_cost_rate": picking_cost_rate,
        "access_cost_rate": access_cost_rate,
        "relocation_cost": relocation_cost,
        "warehouse_capacity": warehouse_capacity,
        "transport_capacity": 10_000.0,
        "staging_area_capacity": staging_area_capacity,
        "max_value_at_point_of_use": max(
            50.0,
            0.6 * sum(avg_inventory_total[i] * item_value[i] for i in items),
        ),
        "number_of_deliveries": 4,
        "warehouse_assigned_to_resource": warehouse_assigned_to_resource,
        "min_inventory": {
            (i, l): round(0.4 * lot_size[i], 2)
            for i in items
            for l in warehouses
            if stage_of_warehouse[l] < 3
        },
        "planned_area_share": {
            l: 1.0 for l in warehouses if stage_of_warehouse[l] < 3
        },
        "max_lot_size": max_lot_size,
        "avg_lot_size": avg_lot_size,
        "product_demand": product_demand,
        "big_m": big_m,
    }


def build_network_instance(**kwargs: Any) -> Instance:
    """:func:`build_network_data` as an :class:`~intralogistics.data.Instance`."""
    return Instance(build_network_data(**kwargs))


class _Lcg:
    """A tiny linear congruential generator.

    Deterministic across platforms and Python versions, which matters because
    instance numbers end up in the plan document as measured results.
    """

    def __init__(self, seed: int) -> None:
        self.state = (seed * 6364136223846793005 + 1442695040888963407) % (2**64)

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (2**64)
        return self.state >> 33

    def uniform(self) -> float:
        return self.next() / float(2**31)

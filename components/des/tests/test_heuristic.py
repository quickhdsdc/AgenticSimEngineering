"""Phase 5.1 — the constructive heuristic.

Three claims, and the third is the one that matters for the framework:

1. **construction produces feasible configurations**, on the source instance
   and on generated ones, without a repair phase;
2. **it is close to optimal where the optimum is computable** — the number
   Lena's *Regelableitung* chapter needs is "how much worse than optimal", and
   it cannot be quoted without checking it;
3. **the generic solver works once the decision space is right.** Phase 2
   measured ``HillClimbSolver`` making exactly zero progress on the raw
   variable space. The same solver, unchanged, improves on the plan space.
   What was missing was never the solver.
"""

from __future__ import annotations

import pytest

from intralogistics.data import build_minimal_instance
from intralogistics.heuristic import (
    GreedySupplySolver,
    PlanWorld,
    SupplyPlan,
    build_greedy_plan,
    render,
)
from intralogistics.instances import build_network_instance
from intralogistics.milp_model import MilpWorldModel
from intralogistics.static_model import StaticWorldModel
from intralogistics.world import StaticWorld
from opsworld.solver import (
    GradientDescentSolver,
    HillClimbSolver,
    RandomSearchSolver,
)
from opsworld.world import Capability


# ----------------------------------------------------------------------
# 1. construction
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n_items", "n_resources", "n_forward"),
    [(2, 2, 1), (3, 2, 1), (4, 2, 2), (8, 3, 2), (20, 4, 2)],
)
def test_greedy_construction_is_feasible(n_items, n_resources, n_forward):
    """No repair phase: the plan expands into a configuration that satisfies
    every constraint family, checked by the independent numpy evaluator."""
    instance = build_network_instance(
        n_items=n_items, n_resources=n_resources, n_forward=n_forward
    )
    evaluation = StaticWorldModel().evaluate(
        instance, render(instance, build_greedy_plan(instance))
    )
    assert evaluation.is_feasible(tol=1e-6), evaluation.violated()


def test_greedy_construction_is_feasible_on_the_source_instance():
    """The instance Phase 1 is anchored to, whose structure is forced."""
    instance = build_minimal_instance()
    evaluation = StaticWorldModel().evaluate(
        instance, render(instance, build_greedy_plan(instance))
    )
    assert evaluation.is_feasible(tol=1e-6), evaluation.violated()


def test_hub_is_skipped_when_the_two_bin_rule_leaves_no_room():
    """Every location holding an item owes two bins of it under (11), so an
    item picked from many resources can have its floors already exhaust its
    stock. Adding a hub then overflows the equality (8) — so it is not added.
    """
    instance = build_network_instance(n_items=4, n_resources=5, n_forward=1)
    plan = build_greedy_plan(instance)
    for i in instance.I:
        floors = 0.0
        for l in plan.stored_at(i):
            b = plan.carrier[(i, l)]
            floors += 2.0 * float(instance.param("carrier_capacity", i, b))
        assert floors <= float(instance.param("avg_inventory_total", i)) + 1e-9
    evaluation = StaticWorldModel().evaluate(instance, render(instance, plan))
    assert evaluation.is_feasible(tol=1e-6), evaluation.violated()


def test_render_serves_every_demand_exactly_once():
    instance = build_network_instance(n_items=3, n_resources=2, n_forward=2)
    decision = render(instance, build_greedy_plan(instance))
    for i in instance.I:
        for r in instance.R:
            picks = sum(decision.get("x_pick", i, r, l) for l in instance.L_sys)
            expected = (
                1.0 if float(instance.param("demand_item_resource", i, r)) > 0 else 0.0
            )
            assert picks == pytest.approx(expected)


def test_routes_call_at_every_stored_intermediate_warehouse():
    """Constraint (17b) forbids bypassing a stored intermediate warehouse, so
    the route is not a free choice. Rendering must honour that, not discover it
    through a violation."""
    instance = build_network_instance(n_items=3, n_resources=2, n_forward=2)
    plan = build_greedy_plan(instance)
    decision = render(instance, plan)

    for i in instance.I:
        staging = [l for l in plan.stored_at(i) if l in instance.L_log]
        for l_mid in staging:
            stage_mid = instance.stage(l_mid)
            bypass = sum(
                value
                for key, value in decision.y_reloc_moves.items()
                if key[0] == i
                and instance.stage(key[2]) > stage_mid > instance.stage(key[3])
            )
            assert bypass == 0.0


def test_render_is_total_even_for_a_plan_with_gaps():
    """A plan that names picks but no carriers still renders — the missing
    choices are filled in, so a move never has to construct a whole plan."""
    instance = build_network_instance(n_items=2, n_resources=2, n_forward=1)
    sparse = SupplyPlan(
        pick={
            (i, r): next(
                l for l in instance.L_sys
                if instance.param("warehouse_assigned_to_resource", r, l)
            )
            for i in instance.I
            for r in instance.R
            if float(instance.param("demand_item_resource", i, r)) > 0
        }
    )
    evaluation = StaticWorldModel().evaluate(instance, render(instance, sparse))
    assert evaluation.objective > 0.0


# ----------------------------------------------------------------------
# 2. quality against the exact solver
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n_items", "n_resources", "n_forward", "seed"),
    [(2, 2, 1, 0), (3, 2, 1, 0), (3, 2, 1, 3)],
)
def test_heuristic_reaches_the_optimum_on_small_instances(
    n_items, n_resources, n_forward, seed
):
    instance = build_network_instance(
        n_items=n_items, n_resources=n_resources, n_forward=n_forward, seed=seed
    )
    _, optimum = MilpWorldModel(instance).solve()
    assert optimum is not None

    result = GreedySupplySolver().solve(PlanWorld(instance), budget=100_000)
    assert result.evaluation.is_feasible(tol=1e-6)
    assert result.score <= optimum * (1 + 1e-6)


def test_local_search_repairs_an_infeasible_construction():
    """Tightening the capacities makes first-fit construction overshoot. The
    search has to recover feasibility, not just shave cost."""
    instance = build_network_instance(
        n_items=3, n_resources=2, n_forward=2, seed=0, tighten=0.55
    )
    world = PlanWorld(instance)
    start = world.evaluate(world.decision_space().initial())
    assert not start.is_feasible()

    result = GreedySupplySolver().solve(world, budget=100_000)
    assert result.evaluation.is_feasible(tol=1e-6), result.evaluation.violated()


def test_local_search_never_returns_worse_than_construction():
    instance = build_network_instance(n_items=4, n_resources=2, n_forward=2)
    result = GreedySupplySolver(max_passes=5).solve(PlanWorld(instance), budget=400)
    assert result.score <= result.initial_score + 1e-9
    assert result.history == sorted(result.history, reverse=True)


# ----------------------------------------------------------------------
# 3. the framework claim
# ----------------------------------------------------------------------
def test_the_generic_solver_works_on_the_plan_space_and_not_on_the_raw_one():
    """Phase 2's dead end, and its cause.

    ``HillClimbSolver`` knows nothing about intralogistics. On the raw variable
    space it cannot move, because the constraints are coupled so tightly that
    every single-variable change is a net loss. On the plan space — same
    instance, same evaluator, same objective — it improves. The obstacle was
    the parameterisation.
    """
    instance = build_network_instance(
        n_items=5, n_resources=2, n_forward=2, seed=1, tighten=0.7
    )
    solver = HillClimbSolver(seed=0)

    on_plans = solver.solve(PlanWorld(instance), budget=120)
    assert on_plans.improvement > 0.0
    assert on_plans.evaluation.is_feasible(tol=1e-6), on_plans.evaluation.violated()

    on_variables = solver.solve(StaticWorld(instance), budget=120)
    assert not on_variables.evaluation.is_feasible()
    assert on_plans.score < on_variables.score

    _, optimum = MilpWorldModel(instance).solve(time_limit=120)
    assert on_plans.score <= optimum * 1.05


def test_plan_world_offers_only_c0_and_the_gate_holds():
    instance = build_network_instance(n_items=2, n_resources=2, n_forward=1)
    world = PlanWorld(instance)
    assert world.capabilities() == {Capability.EVALUATE}

    RandomSearchSolver(seed=0).solve(world, budget=5)  # C0 only: fine
    with pytest.raises(TypeError, match="requires"):
        GradientDescentSolver().solve(world, budget=2)


def test_capping_the_neighbourhood_keeps_the_search_working():
    """Each neighbour costs a full evaluation, so at scale the neighbourhood
    has to be sampled. Doing so must degrade the result, not break it."""
    instance = build_network_instance(n_items=4, n_resources=2, n_forward=2)
    world = PlanWorld(instance)
    full = GreedySupplySolver(max_passes=5).solve(world, budget=100_000)
    capped = GreedySupplySolver(max_passes=5, max_moves=6, seed=1).solve(
        world, budget=100_000
    )
    assert capped.evaluation.is_feasible(tol=1e-6)
    assert capped.score <= capped.initial_score + 1e-9
    assert capped.evaluations_used < full.evaluations_used

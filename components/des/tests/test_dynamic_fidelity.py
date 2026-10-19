"""Phase 3 — when does the time average pick the same decision as the dynamics?

Two groups of tests:

**The simulator must be trustworthy before its verdict counts.** It is checked
against results that follow from theory rather than from the code: flow
conservation (``moves x Q = demand``, which is constraint (17) of the source
MILP holding in the dynamics), and the sawtooth mean ``Q/2 + const``.

**The verdict itself**, locked in as regressions: at zero variability the fluid
projection is *exact*, and away from it the static models lose the ranking,
reject good decisions, and accept configurations that starve the line.
"""

from __future__ import annotations

import pytest

from intralogistics.dynamic_model import (
    CarrierCostModel,
    DynamicWorld,
    ReplenishmentParams,
    StaticExogenousWorld,
    StaticSelfConsistentWorld,
    simulate_replenishment,
)
from opsworld import compare_world_models

CAPACITIES = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0]
INCUMBENT = 25.0


def params(cv: float, lead_time: int = 3, horizon: int = 3000) -> ReplenishmentParams:
    return ReplenishmentParams(
        demand_rate=10.0, demand_cv=cv, lead_time=lead_time, horizon=horizon, seed=1
    )


def best(world) -> float | None:
    feasible = [
        (world.evaluate(q).objective, q)
        for q in CAPACITIES
        if world.evaluate(q).is_feasible()
    ]
    return min(feasible)[1] if feasible else None


def worlds(cv: float, lead_time: int = 3, *, service_target: float = 0.0):
    p = params(cv, lead_time)
    costs = CarrierCostModel()
    dynamic = DynamicWorld(p, costs, CAPACITIES, service_target=service_target)
    calibrated = DynamicWorld(p, costs, CAPACITIES).evaluate(INCUMBENT)
    inventory = calibrated.kpis["avg_inventory"]
    return (
        dynamic,
        StaticExogenousWorld(p, costs, CAPACITIES, avg_inventory=inventory),
        StaticExogenousWorld(
            p, costs, CAPACITIES, avg_inventory=inventory, kanban_constraint=False
        ),
        StaticSelfConsistentWorld(p, costs, CAPACITIES),
    )


# ======================================================================
# is the simulator trustworthy?
# ======================================================================
@pytest.mark.parametrize("capacity", [5.0, 10.0, 20.0, 40.0])
def test_flow_conservation(capacity: float) -> None:
    """``moves x Q == demand`` — constraint (17) of the MILP, in the dynamics.

    This is what caught the first implementation: allowing only one bin per
    period silently capped the replenishment rate, and the ratio came out at
    0.33 instead of 1.0 for small bins.
    """
    outcome = simulate_replenishment(capacity, params(0.3, horizon=20000))
    assert outcome.moves_per_period * capacity == pytest.approx(10.0, rel=0.02)


def test_sawtooth_mean_is_half_a_bin_plus_a_constant() -> None:
    """With deterministic demand the on-hand mean is ``Q/2`` plus an offset
    that does not depend on ``Q`` — the classic sawtooth."""
    offsets = [
        simulate_replenishment(q, params(0.0, horizon=20000)).avg_inventory - q / 2
        for q in (10.0, 20.0, 40.0, 60.0)
    ]
    assert max(offsets) - min(offsets) < 1e-6, offsets


def test_variability_costs_service_and_inventory() -> None:
    previous_service, previous_inventory = 2.0, 0.0
    for cv in (0.0, 0.3, 0.6, 1.0):
        outcome = simulate_replenishment(20.0, params(cv, horizon=8000))
        assert outcome.service_level <= previous_service
        assert outcome.avg_inventory >= previous_inventory
        previous_service, previous_inventory = (
            outcome.service_level,
            outcome.avg_inventory,
        )


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        simulate_replenishment(0.0, params(0.3))


# ======================================================================
# the verdict
# ======================================================================
def test_fluid_projection_is_exact_without_variability() -> None:
    """The deterministic limit: the time average loses nothing.

    Also an internal check on the experiment — a disagreement *here* would
    mean the simulator, not the static model, is wrong.
    """
    dynamic, _, exogenous_unconstrained, self_consistent = worlds(cv=0.0)

    assert best(exogenous_unconstrained) == best(dynamic)
    assert best(self_consistent) == best(dynamic)

    report = compare_world_models(dynamic, self_consistent, CAPACITIES)
    assert report.rank_correlation == pytest.approx(1.0)
    assert report.preserves_ranking()


@pytest.mark.parametrize("cv", [0.25, 0.5, 0.75, 1.0])
def test_static_models_lose_the_ranking_under_variability(cv: float) -> None:
    """Away from the deterministic limit neither static model orders the
    candidate carrier sizes as the dynamics do."""
    dynamic, exogenous, _, self_consistent = worlds(cv=cv)
    for approximation in (exogenous, self_consistent):
        report = compare_world_models(dynamic, approximation, CAPACITIES)
        assert not report.preserves_ranking(0.9), report.summary()


def test_kanban_constraint_on_an_exogenous_parameter_rejects_good_decisions() -> None:
    """Constraint (11) is the dominant failure, and it is a *feasibility* one.

    ``q_inventory >= 2 * capacity`` with inventory held fixed caps the carrier
    size at half a number that should itself grow with the carrier size. Most
    of the menu is wrongly rejected, and the recommendation is far off.
    """
    dynamic, exogenous, exogenous_unconstrained, _ = worlds(cv=0.0)

    report = compare_world_models(dynamic, exogenous, CAPACITIES)
    assert report.false_infeasible >= 5, report.summary()

    optimum = dynamic.evaluate(best(dynamic)).objective
    constrained = dynamic.evaluate(best(exogenous)).objective
    unconstrained = dynamic.evaluate(best(exogenous_unconstrained)).objective

    assert constrained / optimum > 2.0, "expected a large regret from (11)"
    assert unconstrained == pytest.approx(optimum), "removing (11) should recover it"


def test_static_models_cannot_see_stockouts() -> None:
    """False-feasible: plans a static model accepts and the line starves on.

    A time average contains no stockouts, so service level is invisible to it.
    Under a 98% target at high variability the dynamics reject configurations
    the static model is happy with.
    """
    dynamic, _, exogenous_unconstrained, _ = worlds(
        cv=1.0, lead_time=6, service_target=0.98
    )
    report = compare_world_models(dynamic, exogenous_unconstrained, CAPACITIES)
    assert report.false_feasible >= 3, report.summary()
    assert report.misleads()


def test_no_false_feasible_when_variability_is_low() -> None:
    """The failure is caused by variability, not by the service target itself."""
    dynamic, _, exogenous_unconstrained, _ = worlds(
        cv=0.25, lead_time=3, service_target=0.98
    )
    report = compare_world_models(dynamic, exogenous_unconstrained, CAPACITIES)
    assert report.false_feasible == 0, report.summary()

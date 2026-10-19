"""Phase 1 acceptance tests — separating the Wirkmodell from the solver.

The central claim under test (``PLAN.md`` Axiom 1 / Phase 1.4) is that the
world model can be evaluated **without a solver**, and that when it is, it
reproduces the solver's own numbers exactly. Three independent implementation
paths must agree:

1. the MILP solver's objective value (HiGHS branch-and-bound),
2. Pyomo's expression algebra over the same model with variables fixed,
3. the standalone numpy evaluator in :mod:`intralogistics.static_model`.

Path 3 shares no code with paths 1 and 2 beyond the index sets, so agreement
is real cross-validation rather than a tautology.
"""

from __future__ import annotations

import random

import pytest

from intralogistics import (
    Decision,
    Instance,
    MilpWorldModel,
    StaticWorldModel,
    available_solver,
    build_minimal_data,
)
from intralogistics.scenarios import Scenario, default_scenarios, run_scenario_study

requires_solver = pytest.mark.skipif(
    available_solver() is None, reason="no MILP solver installed"
)


@pytest.fixture
def instance() -> Instance:
    return Instance(build_minimal_data())


@pytest.fixture
def solved(instance: Instance):
    model = MilpWorldModel(instance)
    decision, objective = model.solve()
    assert decision is not None, f"no feasible solution ({model.termination})"
    return model, decision, objective


# ----------------------------------------------------------------------
# instance / index sets
# ----------------------------------------------------------------------
def test_derived_sets(instance: Instance) -> None:
    assert instance.L_we == ["L6"]                       # highest stage
    assert instance.L_sys == ["L4", "L5", "L3", "L2", "L1"]
    assert instance.L_log == ["L4", "L5", "L3"]          # strictly intermediate
    assert instance.L_prod == ["L2", "L1"]               # point of use
    # relocation arcs are strictly downstream only
    assert len(instance.U_edges) == 13
    assert all(instance.stage(a) > instance.stage(b) for a, b in instance.U_edges)


def test_missing_key_is_rejected() -> None:
    data = build_minimal_data()
    del data["carrier_area"]
    with pytest.raises(ValueError, match="Missing required data keys"):
        Instance(data)


# ----------------------------------------------------------------------
# Phase 1.4 — cross-validation
# ----------------------------------------------------------------------
@requires_solver
def test_milp_solves_to_optimality(solved) -> None:
    model, _, objective = solved
    assert model.termination == "optimal"
    assert objective is not None and objective > 0


@requires_solver
def test_static_evaluator_reproduces_solver_objective(instance, solved) -> None:
    """1.4 — the solver-free evaluator recomputes the solver's objective."""
    _, decision, objective = solved
    static = StaticWorldModel().evaluate(instance, decision)
    assert static.objective == pytest.approx(objective, rel=1e-9, abs=1e-6)


@requires_solver
def test_static_evaluator_confirms_optimum_is_feasible(instance, solved) -> None:
    """1.4 — an independent feasibility check of the solver's answer."""
    _, decision, _ = solved
    static = StaticWorldModel().evaluate(instance, decision)
    assert static.violated() == {}, f"independent check found violations: {static.violated()}"


@requires_solver
def test_three_paths_agree_per_cost_component(instance, solved) -> None:
    """1.4 — agreement holds component by component, not just in total."""
    model, decision, objective = solved
    algebraic = model.evaluate(instance, decision)
    numeric = StaticWorldModel().evaluate(instance, decision)

    assert set(algebraic.kpis) == set(numeric.kpis)
    for name, value in numeric.kpis.items():
        assert algebraic.kpis[name] == pytest.approx(value, rel=1e-9, abs=1e-6), name
    assert algebraic.objective == pytest.approx(objective, rel=1e-9, abs=1e-6)


# ----------------------------------------------------------------------
# C2b — duals of the LP relaxation
# ----------------------------------------------------------------------
@requires_solver
def test_lp_relaxation_is_a_valid_lower_bound(instance, solved) -> None:
    model, _, objective = solved
    lp_objective, duals = model.lp_relaxation_duals()
    assert lp_objective <= objective + 1e-6, "LP relaxation must bound the MILP"
    assert duals, "expected at least one binding constraint"


@requires_solver
def test_model_still_solves_after_taking_duals(instance) -> None:
    """Regression: the `dual` Suffix must not outlive the LP relaxation.

    appsi tries to load duals on every solve once the Suffix exists, and a
    mixed-integer model has none — so leaving it attached broke the next
    ``solve()`` with "Solver does not currently have valid duals".
    """
    model = MilpWorldModel(instance)
    _, before = model.solve()
    model.lp_relaxation_duals()
    _, after = model.solve()
    assert after == pytest.approx(before, rel=1e-9)

    # integrality must also be restored
    assert all(v.domain is not None for v in model.model.x_store.values())
    decision, _ = model.solve()
    assert decision is not None
    assert all(
        value == pytest.approx(round(value), abs=1e-6)
        for value in decision.x_store.values()
    ), "binaries should be integral again after the relaxation round-trip"


# ----------------------------------------------------------------------
# Phase 1.5 — consistency on arbitrary (mostly infeasible) decisions
# ----------------------------------------------------------------------
def _random_decision(inst: Instance, rng: random.Random) -> Decision:
    """A random configuration; almost surely infeasible, which is the point."""
    d = Decision()
    for i in inst.I:
        for l in inst.L_sys:
            d.x_store[(i, l)] = float(rng.getrandbits(1))
            d.q_inventory[(i, l)] = float(rng.randrange(0, 60))
            for b in inst.B:
                d.x_carrier[(i, l, b)] = float(rng.getrandbits(1))
                d.y_carriers[(i, l, b)] = float(rng.randrange(0, 5))
        for r in inst.R:
            for l in inst.L_sys:
                d.x_pick[(i, r, l)] = float(rng.getrandbits(1))
            for l1, l2 in inst.U_edges:
                for b1 in inst.B:
                    for b2 in inst.B:
                        if rng.random() < 0.1:
                            d.y_reloc_moves[(i, r, l1, l2, b1, b2)] = float(
                                rng.randrange(0, 4)
                            )
    for p in inst.P:
        for l in inst.L_sys:
            d.x_product_from_warehouse[(p, l)] = float(rng.getrandbits(1))
    return d


@requires_solver
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_paths_agree_on_arbitrary_decisions(instance, seed) -> None:
    """1.5 — the two evaluators agree on cost *and* on every violated family."""
    model = MilpWorldModel(instance)
    model.build()
    decision = _random_decision(instance, random.Random(seed))

    algebraic = model.evaluate(instance, decision)
    numeric = StaticWorldModel().evaluate(instance, decision)

    assert algebraic.objective == pytest.approx(numeric.objective, rel=1e-9, abs=1e-6)
    assert algebraic.is_feasible() == numeric.is_feasible()
    assert set(algebraic.violated()) == set(numeric.violated())
    for family, amount in numeric.violated().items():
        assert algebraic.violations[family] == pytest.approx(
            amount, rel=1e-9, abs=1e-6
        ), family


def test_empty_decision_violates_the_mandatory_constraints(instance) -> None:
    """A do-nothing configuration must be reported infeasible, with zero cost."""
    evaluation = StaticWorldModel().evaluate(instance, Decision())
    assert evaluation.objective == 0.0
    assert not evaluation.is_feasible()
    # every item must still be stocked and every demand still picked
    assert "08_inventory_balance" in evaluation.violated()
    assert "06_one_pick_location" in evaluation.violated()


# ----------------------------------------------------------------------
# Phase 1.2 — the silently-dead scenario key
# ----------------------------------------------------------------------
def test_carrier_capacity_multiplier_takes_effect(instance) -> None:
    """Source notebook spelled this key 'carrier_capacity_mul-tiplier'.

    The stray hyphen meant the scenario was silently ignored.
    """
    scenario = Scenario("halve_B1", carrier_capacity_multiplier={"B1": 0.5})
    perturbed = scenario.apply(instance)
    for i in instance.I:
        assert perturbed.param("carrier_capacity", i, "B1") == pytest.approx(
            instance.param("carrier_capacity", i, "B1") * 0.5
        )
        assert perturbed.param("carrier_capacity", i, "B2") == pytest.approx(
            instance.param("carrier_capacity", i, "B2")
        )


def test_scenario_does_not_mutate_the_base_instance(instance) -> None:
    before = dict(instance.data["storage_cost_rate"])
    Scenario("x5", storage_cost_multiplier_by_stage={0: 5.0}).apply(instance)
    assert instance.data["storage_cost_rate"] == before


@requires_solver
def test_scenario_study_runs(instance) -> None:
    results = run_scenario_study(instance, default_scenarios())
    assert len(results) == len(default_scenarios())
    assert all(r.objective is not None for r in results)


# ----------------------------------------------------------------------
# a structural property of the source dataset, locked in as a regression
# ----------------------------------------------------------------------
@requires_solver
def test_point_of_use_storage_is_forced_not_chosen(instance) -> None:
    """Constraints (6)+(7) force picking — and hence storage — at L1/L2.

    Consequence: uniformly scaling ``storage_cost_rate`` at those warehouses
    changes the objective but **cannot** change the optimal decision. In the
    shipped dataset that rate is 1e5, so ~99.99% of the objective is a term
    the model has no freedom over. See PLAN.md Phase 1 findings.
    """
    outcomes = []
    for rate in (0.0, 2.0, 100000.0):
        data = build_minimal_data()
        for warehouse in ("L1", "L2", "L3"):
            data["storage_cost_rate"][warehouse] = rate
        inst = Instance(data)
        model = MilpWorldModel(inst)
        decision, objective = model.solve()
        assert decision is not None
        kpis = model.evaluate(inst, decision).kpis
        outcomes.append(
            (
                sorted(k for k, v in decision.x_pick.items() if v > 0.5),
                objective - kpis["cost_storage"],
            )
        )

    picks = [o[0] for o in outcomes]
    non_storage = [o[1] for o in outcomes]
    assert picks[0] == picks[1] == picks[2], "pick assignment should be rate-invariant"
    assert non_storage[0] == pytest.approx(non_storage[1], rel=1e-9)
    assert non_storage[1] == pytest.approx(non_storage[2], rel=1e-9)

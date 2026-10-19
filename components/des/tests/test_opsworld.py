"""Phase 2 acceptance tests — does the shared abstraction actually fit?

The decisive test is :func:`test_one_solver_instance_drives_both_domains`.
``PLAN.md`` names it the only hard evidence that the protocol is the right
shape, and sets the failure response explicitly: if the protocol needs a
special case per domain, **retreat to a thinner protocol rather than thickening
it** (risk R1).

Everything else here guards the properties that make the abstraction worth
having: capabilities gate solvers rather than degrading silently, and a dual
variable cannot be mistaken for a search direction (decision record D10).
"""

from __future__ import annotations

import random

import pytest
import torch

from gradientshop.world import DispatchWorld
from intralogistics import Instance, MilpWorldModel, build_minimal_data
from intralogistics.world import MilpWorld, StaticWorld
from opsworld import (
    Capability,
    GradientDescentSolver,
    HillClimbSolver,
    RandomSearchSolver,
    Sensitivity,
    SensitivityKind,
    WorldModel,
    compare_world_models,
    penalised_objective,
    spearman,
)
from opsworld.world import Evaluation

requires_solver = pytest.mark.skipif(
    __import__("intralogistics").available_solver() is None,
    reason="no MILP solver installed",
)


@pytest.fixture
def dispatch_world() -> DispatchWorld:
    generator = torch.Generator().manual_seed(7)
    n = 12
    return DispatchWorld(
        processing=torch.rand(n, generator=generator, dtype=torch.float64) * 9 + 1,
        weight=torch.rand(n, generator=generator, dtype=torch.float64) * 4 + 1,
        due=torch.rand(n, generator=generator, dtype=torch.float64) * 40 + 10,
    )


@pytest.fixture
def instance() -> Instance:
    return Instance(build_minimal_data())


@pytest.fixture
def static_world(instance: Instance) -> StaticWorld:
    return StaticWorld(instance)


# ======================================================================
# THE acceptance criterion
# ======================================================================
def test_one_solver_instance_drives_both_domains(dispatch_world, static_world) -> None:
    """One solver, two unrelated domains, no special-casing anywhere.

    The solver knows nothing about dispatching rules or load carriers. It uses
    ``C0`` and the decision space, and that is enough for a differentiable
    discrete-event model of a shopfloor *and* a constrained configuration
    problem in intralogistics.
    """
    solver = HillClimbSolver(seed=0)  # a single instance, reused

    dispatch_result = solver.solve(dispatch_world, budget=400)
    logistics_result = solver.solve(static_world, budget=400)

    assert dispatch_result.improvement > 0, "no progress on the dispatch domain"
    assert logistics_result.improvement > 0, "no progress on the intralogistics domain"
    # the intralogistics result must also remain a legal configuration
    assert logistics_result.evaluation.is_feasible()


@requires_solver
def test_generic_search_closes_part_of_the_gap_to_the_optimum(
    instance, static_world
) -> None:
    """A capability-C0-only solver gets meaningfully closer to the MILP optimum.

    Not a claim that hill climbing competes with branch and bound — it is the
    check that the world model is genuinely usable by search, i.e. that its
    objective responds to the moves a solver can make.
    """
    _, optimum = MilpWorldModel(instance).solve()
    assert optimum is not None

    result = HillClimbSolver(seed=0).solve(static_world, budget=400)
    start_gap = (result.initial_score - optimum) / optimum
    end_gap = (result.score - optimum) / optimum

    assert end_gap < start_gap / 2, f"gap {start_gap:.1%} -> {end_gap:.1%}"
    assert result.evaluation.is_feasible()


def test_random_search_also_runs_on_both(dispatch_world, static_world) -> None:
    solver = RandomSearchSolver(seed=1)
    for world in (dispatch_world, static_world):
        result = solver.solve(world, budget=50)
        assert result.evaluations_used == 50
        assert result.score <= result.initial_score


# ======================================================================
# capabilities gate solvers
# ======================================================================
def test_capability_gate_refuses_a_solver_the_world_cannot_serve(static_world) -> None:
    with pytest.raises(TypeError, match="C2a"):
        GradientDescentSolver().solve(static_world, budget=5)


def test_gradient_solver_runs_where_the_capability_is_present(dispatch_world) -> None:
    assert Capability.PATHWISE_GRADIENT in dispatch_world.capabilities()
    result = GradientDescentSolver(lr=0.3).solve(dispatch_world, budget=40)
    assert result.improvement > 0


def test_declared_capabilities(dispatch_world, static_world, instance) -> None:
    assert dispatch_world.capabilities() == {
        Capability.EVALUATE, Capability.PATHWISE_GRADIENT,
    }
    assert static_world.capabilities() == {Capability.EVALUATE}
    assert MilpWorld(instance).capabilities() == {
        Capability.EVALUATE, Capability.DUALS, Capability.ALGEBRAIC,
    }


def test_worlds_satisfy_the_protocol(dispatch_world, static_world) -> None:
    assert isinstance(dispatch_world, WorldModel)
    assert isinstance(static_world, WorldModel)


# ======================================================================
# decision record D10, enforced rather than documented
# ======================================================================
def test_dual_sensitivity_is_not_a_descent_direction() -> None:
    dual = Sensitivity(SensitivityKind.DUAL, ["23_warehouse_capacity"], [12.5])
    with pytest.raises(TypeError, match="right-hand side"):
        dual.descent_direction()


def test_pathwise_sensitivity_is_a_descent_direction() -> None:
    pathwise = Sensitivity(SensitivityKind.PATHWISE, ["SPT", "EDD"], [-2.0, 1.0])
    assert pathwise.descent_direction() == [2.0, -1.0]
    assert pathwise.ranked()[0][0] == "SPT"  # most improving first


@requires_solver
def test_milp_duals_are_marked_as_duals(instance) -> None:
    sensitivity = MilpWorld(instance).duals()
    assert sensitivity.kind is SensitivityKind.DUAL
    with pytest.raises(TypeError):
        sensitivity.descent_direction()


def test_sensitivity_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        Sensitivity(SensitivityKind.PATHWISE, ["a", "b"], [1.0])


# ======================================================================
# the empty configuration is a dead end — a structural property
# ======================================================================
def test_empty_configuration_admits_no_improving_move(static_world) -> None:
    """Why ``initial()`` constructs rather than starting from nothing.

    Raising ``q_inventory`` reduces the violation of (8) inventory balance by
    exactly what it adds to (9) and (21), so every single-variable neighbour of
    the empty configuration is worse. Local search cannot start there.
    """
    space = static_world.decision_space()
    empty = space.empty()
    base = penalised_objective(static_world.evaluate(empty))
    rng = random.Random(0)

    improving = sum(
        1
        for _ in range(400)
        if penalised_objective(static_world.evaluate(space.perturb(empty, rng))) < base
    )
    assert improving == 0, f"{improving}/400 neighbours improved; the dead end is gone"


def test_constructed_initial_is_feasible_but_suboptimal(static_world) -> None:
    evaluation = static_world.evaluate(static_world.decision_space().initial())
    assert evaluation.is_feasible(), evaluation.violated()
    assert evaluation.objective > 0


# ======================================================================
# fidelity between two world models
# ======================================================================
def test_spearman_on_known_values() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None  # undefined, not 0
    assert spearman([1], [2]) is None


@requires_solver
def test_static_and_milp_world_models_are_mutually_faithful(instance) -> None:
    """Phase 1's cross-validation, re-expressed through the Phase 2 API."""
    static, milp = StaticWorld(instance), MilpWorld(instance)
    space = static.decision_space()
    rng = random.Random(3)

    decisions = [space.initial()]
    for _ in range(9):
        decisions.append(space.perturb(decisions[-1], rng))

    report = compare_world_models(milp, static, decisions)
    assert report.n_decisions == 10
    assert report.is_faithful(atol=1e-6), report.summary()
    assert report.preserves_ranking(0.999)
    assert report.false_feasible == 0
    assert report.false_infeasible == 0
    assert not report.misleads()


def test_fidelity_report_flags_an_inverted_ranking() -> None:
    """A report must call out a relaxation that reverses the decision order."""

    class Flipped:
        def capabilities(self):
            return {Capability.EVALUATE}

        def decision_space(self):
            raise NotImplementedError

        def evaluate(self, decision):
            return Evaluation(objective=-float(decision), provenance="flipped")

    class Plain:
        def capabilities(self):
            return {Capability.EVALUATE}

        def decision_space(self):
            raise NotImplementedError

        def evaluate(self, decision):
            return Evaluation(objective=float(decision), provenance="plain")

    report = compare_world_models(Plain(), Flipped(), [1.0, 2.0, 3.0, 4.0])
    assert report.rank_correlation == pytest.approx(-1.0)
    assert not report.preserves_ranking()
    assert report.misleads()

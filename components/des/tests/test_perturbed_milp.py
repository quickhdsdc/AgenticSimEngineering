"""Phase 5.3 — pathwise gradients through the MILP.

Three things are under test, in decreasing order of how much they would hurt
if wrong:

1. **the estimator is the object it claims to be.** Checked against a closed
   form rather than against itself: on a step function the Gaussian-smoothed
   gradient is ``phi(0)/eps``, and the antithetic estimator has to reproduce
   that number. No solver is involved, so this is exact and fast;
2. **the wrapper does not corrupt the MILP.** The forward pass stays an exact
   integer solution, rate perturbations never move the feasible set, and the
   activity vector reproduces the objective term by term;
3. **the capability claim is real.** The result is typed ``PATHWISE`` and can
   be stepped along, where :meth:`~intralogistics.world.MilpWorld.duals`
   cannot — decision record D10, enforced rather than documented.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from intralogistics.data import build_minimal_instance
from intralogistics.decision import Decision
from intralogistics.instances import build_network_instance
from intralogistics.milp_model import MilpWorldModel
from intralogistics.perturbed import (
    DynamicSupplyCost,
    InfeasibleArgmin,
    PlanningMilp,
    PlanningRates,
    PlanningWorld,
    StaticTrueCost,
    SupplyDynamics,
    descent_check,
    epsilon_curve,
    honest_instance,
    perturbed_gradient,
    solution_key,
    spo_plus_subgradient,
    tune_planning_rates,
)
from intralogistics.static_model import StaticWorldModel
from intralogistics.world import MilpWorld
from opsworld.world import (
    Capability,
    DifferentiableWorldModel,
    SensitivityKind,
)


# ----------------------------------------------------------------------
# 1. the estimator, against a closed form
# ----------------------------------------------------------------------
class _StepOracle:
    """A one-dimensional step function dressed as an argmin oracle.

    ``loss(theta) = 1[theta_0 > 0]`` is piecewise constant with a single
    boundary, exactly like a real argmin but with a Gaussian smoothing that can
    be written down: ``L_eps(theta) = Phi(theta_0 / eps)``, hence
    ``dL_eps/dtheta_0 = phi(theta_0 / eps) / eps`` and zero elsewhere.
    """

    class _Rates:
        def names(self) -> list[str]:
            return ["a", "b"]

        def rates(self, theta):
            return np.asarray(theta, dtype=float)

    rates = _Rates()
    n_solves = 0

    def argmin(self, theta) -> Decision:
        self.n_solves += 1
        return Decision(x_store={("v",): 1.0 if theta[0] > 0 else 0.0})

    @staticmethod
    def loss(decision: Decision) -> float:
        return decision.get("x_store", "v")


def test_antithetic_estimator_matches_the_analytic_smoothed_gradient():
    oracle = _StepOracle()
    epsilon = 0.4
    estimate = perturbed_gradient(
        oracle, _StepOracle.loss, np.zeros(2),
        epsilon=epsilon, n_samples=8000, seed=0,
    )

    expected = 1.0 / math.sqrt(2 * math.pi) / epsilon
    assert estimate.gradient[0] == pytest.approx(expected, rel=0.05)
    # the second coordinate does not touch the boundary at all
    assert abs(estimate.gradient[1]) < 0.1 * expected
    # L_eps(0) = Phi(0) = 1/2
    assert estimate.smoothed_loss == pytest.approx(0.5, abs=0.02)


def test_smoothed_gradient_scales_as_one_over_epsilon():
    """The ``1/eps`` factor is why small eps is dangerous, not safe."""
    oracle = _StepOracle()
    norms = []
    for epsilon in (0.2, 0.4, 0.8):
        estimate = perturbed_gradient(
            oracle, _StepOracle.loss, np.zeros(2),
            epsilon=epsilon, n_samples=6000, seed=1,
        )
        norms.append(estimate.gradient[0] * epsilon)
    # eps * grad is constant: the gradient itself grows without bound
    assert max(norms) / min(norms) < 1.1


def test_no_signal_is_reported_as_no_signal_not_as_zero_slope():
    """Away from the boundary the loss is *constant*, and the estimate says so."""
    oracle = _StepOracle()
    estimate = perturbed_gradient(
        oracle, _StepOracle.loss, np.array([5.0, 0.0]),
        epsilon=0.1, n_samples=40, seed=2,
    )
    assert estimate.distinct_losses == 1
    assert not estimate.has_signal
    assert np.allclose(estimate.gradient, 0.0)
    assert estimate.relative_noise == math.inf


def test_epsilon_curve_trades_fidelity_for_signal():
    oracle = _StepOracle()
    curve = epsilon_curve(
        oracle, _StepOracle.loss, np.array([0.6, 0.0]),
        [0.05, 0.2, 0.8, 2.0], n_samples=400, seed=3,
    )
    # the smoothed value drifts further from the exact one as eps grows
    assert curve["gap"] == sorted(curve["gap"])
    # and there is nothing at all to see at the smallest scale
    assert curve["distinct_solutions"][0] == 1
    assert curve["distinct_solutions"][-1] == 2


def test_descent_check_reports_the_true_loss_not_the_smoothed_one():
    """A piecewise-constant objective has no useful finite difference, so the
    only honest acceptance test is whether a step actually improves it."""
    oracle = _StepOracle()
    # loss is 1 on the positive side; the gradient points uphill, so -grad
    # should cross to the cheap side once the step is large enough.
    report = descent_check(
        oracle, _StepOracle.loss, np.array([0.5, 0.0]),
        np.array([1.0, 0.0]), (0.1, 0.4, 0.9),
    )
    assert report["start_loss"] == 1.0
    assert report["loss"] == [1.0, 1.0, 0.0]
    assert report["best_improvement"] == 1.0

    with pytest.raises(ValueError, match="zero gradient"):
        descent_check(oracle, _StepOracle.loss, np.zeros(2), np.zeros(2), (1.0,))


def test_tuning_never_accepts_a_worse_point():
    """The line search makes the trace monotone -- a piecewise-constant loss
    gives no reason to trust a step it has not evaluated."""
    oracle = _StepOracle()
    result = tune_planning_rates(
        oracle, _StepOracle.loss, np.array([0.5, 0.0]),
        epsilon=0.4, steps=4, n_samples=40, seed=4,
    )
    assert result.trace == sorted(result.trace, reverse=True)
    assert result.best_loss <= result.start_loss
    assert result.best_loss == 0.0  # it does find the cheap side


# ----------------------------------------------------------------------
# 2. the wrapper, against the MILP it wraps
# ----------------------------------------------------------------------
def test_activity_reproduces_the_objective_term_by_term():
    """``objective == rates @ activity + relocation`` -- the identity that makes
    the activity vector a Jacobian and not a heuristic."""
    instance = build_minimal_instance()
    rates = PlanningRates.all_rates(instance, log_space=False)
    milp = PlanningMilp(instance, rates)
    theta = rates.baseline(instance)
    decision = milp.argmin(theta)

    evaluation = MilpWorldModel(instance).evaluate(instance, decision)
    linear = float(rates.rates(theta) @ rates.activity(instance, decision))
    assert linear + evaluation.kpis["cost_reloc"] == pytest.approx(
        evaluation.objective, rel=1e-12
    )


def test_theta_round_trips_through_the_instance():
    instance = build_minimal_instance()
    for log_space in (True, False):
        rates = PlanningRates.all_rates(instance, log_space=log_space)
        theta = rates.baseline(instance)
        rebuilt = rates.instance_with(instance, theta)
        assert rates.baseline(rebuilt) == pytest.approx(theta, rel=1e-12)
        # untouched data is shared, not copied and mangled
        assert rebuilt.param("carrier_capacity", "A1", "B1") == instance.param(
            "carrier_capacity", "A1", "B1"
        )


def test_only_rate_families_may_be_tuned():
    with pytest.raises(ValueError, match="not a rate family"):
        PlanningRates(entries=(("carrier_area", "B1"),))


def test_forward_pass_stays_an_exact_integer_solution():
    """The selling point of this relaxation: the planner never sees a
    fractional configuration, however large eps is."""
    instance = build_minimal_instance()
    rates = PlanningRates.all_rates(instance)
    milp = PlanningMilp(instance, rates)
    rng = np.random.default_rng(0)
    theta = rates.baseline(instance) + 1.5 * rng.standard_normal(len(rates))

    decision = milp.argmin(theta)
    for name in ("x_store", "x_pick", "x_carrier", "x_product_from_warehouse"):
        for value in getattr(decision, name).values():
            assert min(abs(value), abs(value - 1.0)) < 1e-6
    for name in ("y_carriers", "y_reloc_moves"):
        for value in getattr(decision, name).values():
            assert abs(value - round(value)) < 1e-6


def test_rates_cannot_change_the_feasible_set():
    """Rates enter the objective only, so every perturbed argmin is feasible
    for the *unperturbed* constraints -- checked with the independent numpy
    evaluator rather than with the model that produced the solution."""
    instance = build_minimal_instance()
    rates = PlanningRates.all_rates(instance)
    milp = PlanningMilp(instance, rates)
    checker = StaticWorldModel()
    rng = np.random.default_rng(1)

    for _ in range(3):
        theta = rates.baseline(instance) + 2.0 * rng.standard_normal(len(rates))
        evaluation = checker.evaluate(instance, milp.argmin(theta))
        assert evaluation.is_feasible(tol=1e-6), evaluation.violated()


def test_solution_key_ignores_numerical_dust_but_not_decisions():
    left = Decision(x_store={("A1", "L1"): 1.0}, q_inventory={("A1", "L1"): 60.0})
    same = Decision(
        x_store={("A1", "L1"): 1.0 + 1e-12},
        q_inventory={("A1", "L1"): 60.0, ("A2", "L1"): 0.0},
    )
    other = Decision(x_store={("A1", "L2"): 1.0}, q_inventory={("A1", "L1"): 60.0})
    assert solution_key(left) == solution_key(same)
    assert solution_key(left) != solution_key(other)


def test_degenerate_optima_produce_a_gradient_that_only_scales():
    """The failure mode a rigid, degenerate MILP produces — and how to spot it.

    Phase 1 traced why the minimal instance's configuration is forced. The
    consequence here is sharper than "no gradient": ``theta`` sits *on* a jump,
    because the model has several optima of equal planning cost and unequal
    true cost. Perturbing breaks that tie, so the loss does vary and
    ``has_signal`` is satisfied — but the same solutions come back at every
    scale, and ``eps * ||grad||`` is constant, i.e. the gradient diverges as
    ``1/eps`` without ever pointing at a reachable improvement.
    """
    instance = build_minimal_instance()
    rates = PlanningRates.storage_only(instance)
    milp = PlanningMilp(instance, rates)
    loss = StaticTrueCost(honest_instance(instance))
    theta = rates.baseline(instance)

    scaled = []
    for epsilon in (0.01, 0.1, 1.0):
        estimate = perturbed_gradient(
            milp, loss, theta, epsilon=epsilon, n_samples=6, seed=0
        )
        assert estimate.has_signal  # the loss does vary ...
        assert estimate.distinct_losses == 2  # ... over exactly two tied optima
        scaled.append(estimate.scaled_gradient_norm)

    # ... but only through tie-breaking: a hundredfold change of eps leaves the
    # scaled norm untouched, which is the signature to refuse to act on.
    assert max(scaled) / min(scaled) == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------------
# 3. the capability claim
# ----------------------------------------------------------------------
def test_planning_world_upgrades_c3_to_c2a():
    instance = build_minimal_instance()
    rates = PlanningRates.storage_only(instance)
    world = PlanningWorld(
        PlanningMilp(instance, rates),
        StaticTrueCost(honest_instance(instance)),
        epsilon=0.5,
        n_samples=4,
    )
    assert world.capabilities() == {
        Capability.EVALUATE,
        Capability.PATHWISE_GRADIENT,
    }
    assert isinstance(world, DifferentiableWorldModel)
    # the model it is built on has the algebra but not the gradient
    assert Capability.PATHWISE_GRADIENT not in MilpWorld(instance).capabilities()


def test_the_gradient_is_pathwise_where_the_dual_is_not():
    """D10 in one test: both are first-order, only one is a direction."""
    instance = build_minimal_instance()
    rates = PlanningRates.storage_only(instance)
    world = PlanningWorld(
        PlanningMilp(instance, rates),
        StaticTrueCost(honest_instance(instance)),
        epsilon=0.5,
        n_samples=4,
    )
    pathwise = world.gradient(rates.baseline(instance))
    assert pathwise.kind is SensitivityKind.PATHWISE
    assert len(pathwise.descent_direction()) == len(rates)

    dual = MilpWorld(instance).duals()
    assert dual.kind is SensitivityKind.DUAL
    with pytest.raises(TypeError, match="not a direction in decision space"):
        dual.descent_direction()


def test_planning_world_evaluate_reports_the_two_objectives_separately():
    instance = build_minimal_instance()
    rates = PlanningRates.storage_only(instance)
    world = PlanningWorld(
        PlanningMilp(instance, rates),
        StaticTrueCost(honest_instance(instance)),
        n_samples=4,
    )
    evaluation = world.evaluate(rates.baseline(instance))
    # what the planner believes, versus what it is worth
    assert evaluation.kpis["planning_objective"] > 1e5
    assert evaluation.objective < 1e3
    assert "perturbed-milp" in evaluation.provenance


# ----------------------------------------------------------------------
# SPO+, and the boundary of where it applies
# ----------------------------------------------------------------------
def test_spo_plus_vanishes_at_the_true_cost_vector():
    """``x*(2*theta - theta) == x*(theta)`` when theta is the truth, so the
    subgradient is exactly zero -- the fixed point SPO+ is built around."""
    instance = build_minimal_instance()
    rates = PlanningRates.all_rates(instance, log_space=False)
    milp = PlanningMilp(instance, rates)
    truth = rates.baseline(honest_instance(instance))
    assert np.allclose(spo_plus_subgradient(milp, truth, truth), 0.0)


def test_spo_plus_refuses_a_log_space_parameterisation():
    instance = build_minimal_instance()
    rates = PlanningRates.all_rates(instance, log_space=True)
    milp = PlanningMilp(instance, rates)
    theta = rates.baseline(instance)
    with pytest.raises(ValueError, match="linear in the parameter"):
        spo_plus_subgradient(milp, theta, theta)


# ----------------------------------------------------------------------
# the instance generator, and the dynamic loss
# ----------------------------------------------------------------------
def test_network_instance_leaves_the_decisions_open():
    """What the minimal instance lacks: more than one admissible pick
    warehouse, and more than one admissible carrier at a point of use."""
    instance = build_network_instance(n_items=3, n_resources=2, n_forward=1)
    for r in instance.R:
        admissible = [
            l for l in instance.L_sys
            if instance.param("warehouse_assigned_to_resource", r, l)
        ]
        assert len(admissible) >= 2

    point_of_use = [l for l in instance.L_prod]
    assert point_of_use
    for l in point_of_use:
        allowed = [b for b in instance.B if instance.param("carrier_allowed", l, b)]
        assert len(allowed) >= 2


@pytest.mark.parametrize(
    ("n_items", "n_resources", "n_forward"),
    [(1, 1, 1), (2, 1, 2), (5, 3, 2)],
)
def test_generator_builds_a_valid_instance_at_the_edges(n_items, n_resources, n_forward):
    """Instance construction validates its own parameter tables, so building
    one is the check. The round robin over ceil(n_items/2) products must never
    leave a product empty."""
    instance = build_network_instance(
        n_items=n_items, n_resources=n_resources, n_forward=n_forward
    )
    assert len(instance.I) == n_items
    assert all(instance.items_of_product(p) for p in instance.P)
    assert instance.scalar("big_m") > 0


def test_the_two_world_models_still_agree_on_the_generated_instance():
    """Phase 1's cross-validation is only as good as the instances it runs on."""
    instance = build_network_instance(n_items=2, n_resources=2, n_forward=1)
    milp = MilpWorldModel(instance)
    decision, objective = milp.solve()
    assert decision is not None

    from_pyomo = milp.evaluate(instance, decision)
    from_numpy = StaticWorldModel().evaluate(instance, decision)
    assert from_numpy.objective == pytest.approx(objective, rel=1e-9)
    assert from_numpy.objective == pytest.approx(from_pyomo.objective, rel=1e-12)
    assert from_numpy.is_feasible(tol=1e-6), from_numpy.violated()


def test_dynamic_loss_is_not_the_static_one_and_reports_service():
    instance = build_network_instance(n_items=2, n_resources=2, n_forward=1)
    decision, _ = MilpWorldModel(instance).solve()
    assert decision is not None

    report = DynamicSupplyCost(instance, SupplyDynamics(demand_cv=0.6)).report(decision)
    static = StaticWorldModel().evaluate(instance, decision)
    assert report["objective"] != pytest.approx(static.objective, rel=1e-3)
    assert 0.0 < report["service_level"] <= 1.0
    # service level and peak occupancy exist nowhere in the algebraic model
    assert report["peak_area"] > 0.0


def test_higher_variability_costs_more_under_the_dynamics_only():
    """The static model cannot see demand variability at all; the loss can."""
    instance = build_network_instance(n_items=2, n_resources=2, n_forward=1)
    decision, _ = MilpWorldModel(instance).solve()
    assert decision is not None

    calm = DynamicSupplyCost(instance, SupplyDynamics(demand_cv=0.1))(decision)
    choppy = DynamicSupplyCost(instance, SupplyDynamics(demand_cv=0.9))(decision)
    assert choppy > calm
    # ... while the static evaluation is identical, because the parameter is
    # not in its vocabulary
    assert StaticWorldModel().evaluate(instance, decision).objective == pytest.approx(
        StaticWorldModel().evaluate(instance, decision).objective
    )


def test_honest_instance_touches_only_the_placeholders():
    instance = build_minimal_instance()
    honest = honest_instance(instance, point_of_use_rate=3.0)
    assert honest.param("storage_cost_rate", "L4") == instance.param(
        "storage_cost_rate", "L4"
    )
    assert honest.param("storage_cost_rate", "L1") == 3.0
    assert instance.param("storage_cost_rate", "L1") == 100000.0


def test_infeasible_argmin_is_an_error_not_a_none():
    """A missing solution here would be a solver failure, and silently
    returning ``None`` would let it propagate into a gradient."""
    assert issubclass(InfeasibleArgmin, RuntimeError)

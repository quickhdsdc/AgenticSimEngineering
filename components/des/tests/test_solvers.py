"""The solver plugins themselves — Phase 6.2 needs them to be fair.

A sample-efficiency comparison is only worth reporting if none of the entrants
is crippled by its own implementation. Two were, and these tests pin the fixes:

* ``GradientDescentSolver`` returned its **last** iterate rather than its best,
  so a fixed step that overshot turned a good run into a bad number. Measured
  on the Phase 5.2 sizing policy: ``lr=0.05`` reached 38.40 while ``lr=0.3``
  ended at 39.98 — which would have made the comparison a statement about a
  hand-set constant;
* there was no strong derivative-free baseline at all, so "the gradient wins"
  would have been a claim against random search and a hill climb.
"""

from __future__ import annotations

import math

import pytest

from opsworld.problem import VectorSpace
from opsworld.solver import (
    BayesianOptimisationSolver,
    GradientDescentSolver,
    HillClimbSolver,
    RandomSearchSolver,
)
from opsworld.world import Capability, Evaluation, Sensitivity, SensitivityKind


class Quadratic:
    """``sum (x_i - target_i)^2`` — C0 + C2a, with an exact gradient."""

    def __init__(self, target: list[float], *, start: float = 0.0) -> None:
        self.target = target
        self._space = VectorSpace(len(target), start=start, step=0.4)
        self.calls = 0

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE, Capability.PATHWISE_GRADIENT}

    def decision_space(self) -> VectorSpace:
        return self._space

    def evaluate(self, decision) -> Evaluation:  # noqa: ANN001
        self.calls += 1
        value = sum((x - t) ** 2 for x, t in zip(decision, self.target))
        return Evaluation(objective=value, provenance="quadratic")

    def gradient(self, decision) -> Sensitivity:  # noqa: ANN001
        return Sensitivity(
            kind=SensitivityKind.PATHWISE,
            names=self._space.names,
            values=[2.0 * (x - t) for x, t in zip(decision, self.target)],
        )


# ----------------------------------------------------------------------
# gradient descent
# ----------------------------------------------------------------------
def test_gradient_descent_never_reports_worse_than_it_saw():
    """A solver that returns a point worse than one it already evaluated is
    not reporting a result. Provoked with a step size far too large."""
    world = Quadratic([1.0, -2.0, 0.5])
    result = GradientDescentSolver(lr=50.0).solve(world, budget=40)
    assert result.score <= min(result.history)
    assert result.score <= result.initial_score
    assert result.evaluation.objective == pytest.approx(result.score)


def test_backtracking_rescues_a_badly_chosen_step():
    """With backtracking the outcome stops depending on the constant."""
    target = [1.0, -2.0, 0.5]
    tuned = GradientDescentSolver(lr=0.4, backtrack=False).solve(
        Quadratic(target), budget=60
    )
    reckless_fixed = GradientDescentSolver(lr=50.0, backtrack=False).solve(
        Quadratic(target), budget=60
    )
    reckless_backtracking = GradientDescentSolver(lr=50.0).solve(
        Quadratic(target), budget=60
    )

    assert reckless_fixed.score > tuned.score          # the constant hurt
    assert reckless_backtracking.score < 1e-6          # backtracking recovered
    assert reckless_backtracking.score < reckless_fixed.score


def test_every_trial_step_is_charged_to_the_budget():
    """Otherwise the sample-efficiency comparison would be rigged: a line
    search would get free evaluations the other solvers do not."""
    world = Quadratic([1.0, -2.0, 0.5])
    result = GradientDescentSolver(lr=50.0).solve(world, budget=25)
    assert result.evaluations_used <= 25
    assert world.calls == result.evaluations_used


def test_gradient_history_is_monotone():
    world = Quadratic([1.0, -2.0, 0.5])
    result = GradientDescentSolver(lr=50.0).solve(world, budget=30)
    assert result.history == sorted(result.history, reverse=True)


# ----------------------------------------------------------------------
# bayesian optimisation
# ----------------------------------------------------------------------
def test_bayes_opt_needs_only_c0():
    assert BayesianOptimisationSolver.requires == frozenset({Capability.EVALUATE})


def test_bayes_opt_improves_and_respects_the_budget():
    world = Quadratic([1.0, -2.0, 0.5], start=3.0)
    result = BayesianOptimisationSolver(seed=0, n_initial=6).solve(world, budget=40)
    assert result.evaluations_used == 40
    assert world.calls == 40
    assert result.score < result.initial_score
    assert math.isfinite(result.score)


def test_bayes_opt_beats_random_search_on_the_same_budget():
    """It is meant to be the strong derivative-free entrant; if it does not
    beat random sampling it is not doing its job in the comparison."""
    target = [1.0, -2.0, 0.5]
    budget = 40
    bayes = BayesianOptimisationSolver(seed=0, n_initial=6).solve(
        Quadratic(target, start=3.0), budget=budget
    )
    random_search = RandomSearchSolver(seed=0).solve(
        Quadratic(target, start=3.0), budget=budget
    )
    assert bayes.score < random_search.score


def test_bayes_opt_refuses_a_decision_it_cannot_embed():
    """A GP needs numbers. Refusing loudly is how the framework treats any
    other mismatch between a solver and a world model."""

    class Wordy(Quadratic):
        def evaluate(self, decision) -> Evaluation:  # noqa: ANN001
            return Evaluation(objective=1.0, provenance="wordy")

        def decision_space(self):
            class Space:
                def initial(self):
                    return {"not": "a vector"}

                def perturb(self, decision, rng):
                    return dict(decision)

            return Space()

    with pytest.raises(TypeError, match="vector-valued"):
        BayesianOptimisationSolver(seed=0).solve(Wordy([1.0]), budget=4)


# ----------------------------------------------------------------------
# the comparison Phase 6.2 rests on
# ----------------------------------------------------------------------
def test_the_gradient_is_the_most_sample_efficient_on_a_smooth_problem():
    """The claim in its cleanest setting. On a problem where the gradient is
    exact it should dominate at every budget -- if it does not here, the
    measured result on the real world model would be about the solver."""
    target = [1.0, -2.0, 0.5, 2.0]
    for budget in (20, 60):
        scores = {
            "random": RandomSearchSolver(seed=0).solve(
                Quadratic(target, start=3.0), budget=budget
            ).score,
            "hill": HillClimbSolver(seed=0).solve(
                Quadratic(target, start=3.0), budget=budget
            ).score,
            "bayes": BayesianOptimisationSolver(seed=0, n_initial=6).solve(
                Quadratic(target, start=3.0), budget=budget
            ).score,
            "gradient": GradientDescentSolver(lr=0.3).solve(
                Quadratic(target, start=3.0), budget=budget
            ).score,
        }
        assert scores["gradient"] == min(scores.values()), (budget, scores)

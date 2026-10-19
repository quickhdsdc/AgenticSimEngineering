"""Phase 5.2 — a policy as a weight vector over *named* rules, in domain two.

The construction mirrors :mod:`gradientshop.shopfloor.rules`: stack named
heuristics as columns, let a policy weight them, differentiate the objective
w.r.t. those weights. What it buys is the number the source dissertation's
*Regelableitung* chapter needs — **how much more expensive is an interpretable
rule than the item-by-item optimum?**
"""

from __future__ import annotations

import pytest
import torch

from intralogistics.policy import (
    Items,
    SizingPolicyWorld,
    capacity_from_scores,
    rule_names,
    sizing_matrix,
)
from opsworld import Capability, GradientDescentSolver, HillClimbSolver

N_ITEMS = 24
HORIZON = 400
LEAD_TIME = 3


@pytest.fixture(scope="module")
def world() -> SizingPolicyWorld:
    torch.manual_seed(11)
    rate = torch.rand(N_ITEMS, dtype=torch.float64) * 4 + 0.5
    items = Items(
        demand_rate=rate,
        unit_value=torch.rand(N_ITEMS, dtype=torch.float64) * 20 + 1,
        unit_area=torch.rand(N_ITEMS, dtype=torch.float64) * 0.4 + 0.05,
        lead_time=torch.full((N_ITEMS,), float(LEAD_TIME), dtype=torch.float64),
        demand_cv=torch.rand(N_ITEMS, dtype=torch.float64) * 0.5 + 0.2,
    )
    shape = 1.0 / items.demand_cv**2
    demand = (
        torch.distributions.Gamma(shape.unsqueeze(1), (shape / rate).unsqueeze(1))
        .sample((HORIZON,))
        .squeeze(-1)
        .T.to(torch.float64)
    )
    return SizingPolicyWorld(items, demand, lead_time=LEAD_TIME)


@pytest.fixture(scope="module")
def per_item_optimum(world: SizingPolicyWorld):
    """Exact per-item optimum -- the cost is separable across items.

    Module-scoped: it costs a few thousand simulations, and three tests need it.
    """
    # 120 points, matching examples/phase5_interpretable_sizing_policy.py --
    # the baseline is itself a grid search, and a finer grid finds a better free
    # optimum, which makes the interpretable policy look worse. At 60 points the
    # gap reads 0.97%, at 120 it reads 1.17%. Test and example must resolve the
    # baseline identically or they are not measuring the same claim.
    grid = torch.linspace(5, 80, 120, dtype=torch.float64)
    base = torch.full((N_ITEMS,), 40.0, dtype=torch.float64)
    capacities = base.clone()
    for i in range(N_ITEMS):
        costs = []
        for q in grid:
            candidate = base.clone()
            candidate[i] = q
            costs.append(world.cost_of_capacities(candidate))
        capacities[i] = grid[int(torch.tensor(costs).argmin())]
    flat = min(
        world.cost_of_capacities(torch.full((N_ITEMS,), float(q)))
        for q in grid.tolist()
    )
    return capacities, world.cost_of_capacities(capacities), flat


@pytest.fixture(scope="module")
def trained(world: SizingPolicyWorld) -> torch.Tensor:
    return train(world)


def train(world: SizingPolicyWorld, steps: int = 800) -> torch.Tensor:
    theta = torch.tensor(
        world.decision_space().initial(), dtype=torch.float64, requires_grad=True
    )
    optimiser = torch.optim.Adam([theta], lr=0.05)
    for _ in range(steps):
        optimiser.zero_grad()
        world._cost(theta).backward()
        optimiser.step()
    return theta.detach()


# ======================================================================
# the rule basis
# ======================================================================
def test_rules_are_named_and_standardised(world: SizingPolicyWorld) -> None:
    """Varying columns are standardised; constant ones are left alone.

    Which columns are constant depends on the instance, not on the rule: with a
    uniform lead time, ``LONG_LEAD`` is as constant as ``FLAT``. Standardising
    it would divide by zero, so the matrix builder must skip it — this test
    checks the property, not a hardcoded list of names.
    """
    names, matrix = sizing_matrix(world.items)
    assert names == rule_names()
    assert matrix.shape == (N_ITEMS, len(names))

    raw = sizing_matrix(world.items, standardize=False)[1]
    varying = 0
    for index in range(len(names)):
        column = matrix[:, index]
        if raw[:, index].std() <= 1e-8:
            assert torch.allclose(column, raw[:, index]), names[index]
        else:
            varying += 1
            assert column.mean().abs() < 1e-9, names[index]
            assert column.std() == pytest.approx(1.0, abs=1e-9), names[index]
    assert varying >= 5, "most rules should discriminate between items"


def test_unknown_rule_is_rejected(world: SizingPolicyWorld) -> None:
    with pytest.raises(KeyError, match="unknown sizing rule"):
        sizing_matrix(world.items, ["NOT_A_RULE"])


def test_capacity_map_is_monotone_and_bounded() -> None:
    scores = torch.linspace(-8, 8, 50, dtype=torch.float64)
    capacity = capacity_from_scores(scores, 5.0, 80.0)
    assert torch.all(capacity[1:] > capacity[:-1])
    assert capacity.min() > 5.0 and capacity.max() < 80.0
    with pytest.raises(ValueError, match="must exceed"):
        capacity_from_scores(scores, 80.0, 5.0)


# ======================================================================
# the world model
# ======================================================================
def test_capabilities_match_the_dispatch_world(world: SizingPolicyWorld) -> None:
    """Same capabilities as GradientShop's policy world, so the same solvers fit."""
    assert world.capabilities() == {
        Capability.EVALUATE, Capability.PATHWISE_GRADIENT,
    }


def test_gradient_matches_finite_differences(world: SizingPolicyWorld) -> None:
    """The training signal is the smooth surrogate, so this must be exact."""
    theta = torch.tensor([0.05] * len(world.rule_names), dtype=torch.float64)
    reported = torch.tensor(
        world.evaluate(theta.tolist()).sensitivity.values, dtype=torch.float64
    )
    step = 1e-6
    reference = torch.zeros_like(reported)
    for i in range(len(theta)):
        up, down = theta.clone(), theta.clone()
        up[i] += step
        down[i] -= step
        reference[i] = (
            world.evaluate(up.tolist()).objective
            - world.evaluate(down.tolist()).objective
        ) / (2 * step)
    assert (reported - reference).norm() / reference.norm() < 1e-6


def test_generic_solvers_drive_this_world_too(world: SizingPolicyWorld) -> None:
    """A third world model, and the framework's solvers still just work.

    ``GradientDescentSolver`` takes a raw, unnormalised step, so its rate has to
    suit the gradient scale here (order 1e2 at the starting policy); at 0.05 it
    overshoots and the cost climbs. That is a property of plain gradient
    descent, not of the world model — Adam at 0.05 converges on the same
    objective.
    """
    for solver in (HillClimbSolver(seed=0), GradientDescentSolver(lr=1e-3)):
        result = solver.solve(world, budget=60)
        assert result.improvement > 0, type(solver).__name__


def test_explanation_names_every_rule(world: SizingPolicyWorld) -> None:
    text = world.explain([0.1] * len(world.rule_names))
    for name in world.rule_names:
        assert name in text


# ======================================================================
# THE number
# ======================================================================
def test_price_of_interpretability(world, per_item_optimum, trained) -> None:
    """Eight named weights land within a few percent of 24 free variables.

    Baselines, all scored on the **simulation** rather than on the surrogate
    the policy was trained on:

    * per-item optimum — 24 degrees of freedom, uninterpretable, exact because
      the cost is separable across items;
    * the rule policy — 8 interpretable weights;
    * one carrier size for everything — 1 degree of freedom.
    """
    _, optimum, flat = per_item_optimum
    policy = world.simulated_cost(trained)

    assert optimum < policy < flat, (optimum, policy, flat)
    assert (policy - optimum) / optimum < 0.05, "interpretability should cost little"


def test_policy_tracks_the_per_item_optimum(world, per_item_optimum, trained) -> None:
    """The rule policy must reproduce the *shape* of the optimum, not just its cost."""
    per_item, _, _ = per_item_optimum
    capacities = world.capacities(trained)
    correlation = torch.corrcoef(torch.stack([capacities, per_item]))[0, 1]
    assert correlation > 0.8, correlation


def test_surrogate_gap_is_reported_not_hidden(world, trained) -> None:
    """Training on the fluid model must be scored against the simulation."""
    theta = trained
    surrogate = world.evaluate(theta.tolist()).objective
    simulated = world.simulated_cost(theta)
    assert abs(surrogate - simulated) / simulated < 0.05
    assert surrogate != simulated, "the two paths should not be the same computation"


def test_the_documented_figures_are_the_ones_this_code_produces(
    world, per_item_optimum, trained
) -> None:
    """Guard the numbers that appear in PLAN.md, README.md and the memo.

    The other tests here assert the *shape* of the claim with loose bounds --
    "interpretability costs little", "the policy tracks the optimum" -- which is
    the right way to test a claim and the reason they did not catch this: the
    figures quoted in the documents had never been taken from a run of the
    shipped example. They were 56.92 / +1.44% / +0.945 against an actual
    57.0722 / +1.17% / +0.9561.

    So this test guards the documents rather than the model. A change that moves
    these numbers is legitimate; shipping it without updating the four places
    that quote them is not.
    """
    _, optimum, flat = per_item_optimum
    policy = world.simulated_cost(trained)
    per_item, _, _ = per_item_optimum
    correlation = torch.corrcoef(
        torch.stack([world.capacities(trained), per_item])
    )[0, 1].item()

    documented = {
        "per-item optimum": (optimum, 57.0722),
        "rule policy": (policy, 57.7414),
        "one size fits all": (flat, 59.2832),
    }
    for name, (measured, quoted) in documented.items():
        assert measured == pytest.approx(quoted, rel=5e-3), (
            f"{name}: code gives {measured:.4f}, documents say {quoted} — "
            "update PLAN.md 5.2, README.md, docs/memo-lena-folgebefunde.de.md"
        )

    assert 100 * (policy - optimum) / optimum == pytest.approx(1.17, abs=0.05)
    assert 100 * (flat - optimum) / optimum == pytest.approx(3.87, abs=0.05)
    assert correlation == pytest.approx(0.9561, abs=5e-3)

"""Phase 4 — relaxing a decision that sits *inside* a feedback loop.

M3 relaxes ``argsort`` once, at the end. This module relaxes a reorder
indicator that fires every period and whose output feeds back into the state.
The tests below establish that the two behave qualitatively differently, which
is the Phase 4 result.

Headline, **with its precondition**: at small ``tau`` the relaxed gradient
converges to the exact system's pathwise gradient to ~1e-11 — *provided the
trajectory never lingers near the reorder threshold*, so the sigmoid is
saturated everywhere. Then the relaxation contributes nothing and the useful
sensitivity flows through the *order size* (a continuous magnitude, Source A)
rather than the *order timing* (the discrete decision, Source B).

The precondition is sharp, and was found by overreaching without it: 0.90% of
near-threshold periods takes the cosine against a finite difference from
+1.0000 to +0.0856. :func:`~intralogistics.differentiable.saturation_fraction`
is the check to run before trusting one of these gradients.
"""

from __future__ import annotations

import pytest
import torch

from intralogistics.differentiable import (
    exact_replenishment_cost,
    fidelity_curve,
    gradient_growth_factor,
    gradient_norm,
    relaxed_replenishment,
    saturation_fraction,
    replenishment_cost,
    stability_curve,
)

LEAD_TIME = 3
DEMAND_CV = 0.4


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    rate = torch.tensor([2.0, 3.0, 2.5, 4.0, 3.5, 2.0], dtype=torch.float64)
    shape = 1.0 / DEMAND_CV**2
    demand = (
        torch.distributions.Gamma(shape, shape / rate.unsqueeze(1))
        .sample((800,))
        .squeeze(-1)
        .T.to(torch.float64)
    )
    reorder_point = rate * LEAD_TIME + 1.65 * DEMAND_CV * rate * LEAD_TIME**0.5
    capacity = torch.tensor([25.0, 30.0, 25.0, 35.0, 30.0, 25.0], dtype=torch.float64)
    # Two-bin Kanban is only valid while a lead time consumes less than a bin.
    assert bool(((rate * LEAD_TIME) < capacity).all())
    return capacity, demand, reorder_point


def finite_difference(capacity, demand, reorder_point, horizon, step=1e-3):
    gradient = torch.zeros_like(capacity)
    for i in range(capacity.shape[0]):
        up, down = capacity.clone(), capacity.clone()
        up[i] += step
        down[i] -= step
        gradient[i] = (
            exact_replenishment_cost(up, demand[:, :horizon], reorder_point,
                                     lead_time=LEAD_TIME)
            - exact_replenishment_cost(down, demand[:, :horizon], reorder_point,
                                       lead_time=LEAD_TIME)
        ) / (2 * step)
    return gradient


# ======================================================================
# the recursion itself
# ======================================================================
def test_flow_conservation_in_the_tensor_recursion(setup) -> None:
    """``moves x Q`` must equal the demand rate, as in the numpy simulation."""
    capacity, demand, reorder_point = setup
    trajectory = relaxed_replenishment(
        capacity, demand, reorder_point, lead_time=LEAD_TIME, tau=1e-3, hard=True
    )
    replenished = trajectory.moves_per_period * capacity
    assert torch.allclose(replenished, demand.mean(dim=1), rtol=0.05)


def test_shape_validation(setup) -> None:
    capacity, demand, reorder_point = setup
    with pytest.raises(ValueError, match=r"\(N,\)"):
        relaxed_replenishment(capacity.unsqueeze(0), demand, reorder_point)
    with pytest.raises(ValueError, match="number of items"):
        relaxed_replenishment(capacity[:3], demand, reorder_point)


def test_cost_is_differentiable_end_to_end(setup) -> None:
    capacity, demand, reorder_point = setup
    q = capacity.clone().requires_grad_(True)
    cost = replenishment_cost(q, demand[:, :50], reorder_point, tau=0.1,
                              lead_time=LEAD_TIME)
    (gradient,) = torch.autograd.grad(cost, q)
    assert gradient.shape == capacity.shape
    assert torch.isfinite(gradient).all()


# ======================================================================
# THE headline result
# ======================================================================
def test_small_tau_recovers_the_exact_gradient_when_fully_saturated(setup) -> None:
    """Autodiff equals a finite difference of the *exact* cost — **given
    saturation**.

    The condition is not decorative. What survives at small ``tau`` is the
    pathwise sensitivity through the order *size* (``order = Q * fires``,
    differentiable a.e.) rather than the order *timing*. That only holds while
    the trajectory stays away from the reorder threshold, so that the sigmoid's
    derivative vanishes everywhere. See the companion test for what happens
    when it does not.
    """
    capacity, demand, reorder_point = setup
    horizon = 10
    unsaturated = saturation_fraction(
        capacity, demand[:, :horizon], reorder_point, tau=0.02, lead_time=LEAD_TIME
    )
    assert unsaturated == 0.0, "precondition of this test"

    reference = finite_difference(capacity, demand, reorder_point, horizon)
    q = capacity.clone().requires_grad_(True)
    cost = replenishment_cost(
        q, demand[:, :horizon], reorder_point, tau=0.02, lead_time=LEAD_TIME
    )
    (gradient,) = torch.autograd.grad(cost, q)

    assert torch.allclose(gradient, reference, rtol=1e-6, atol=1e-9)
    assert (gradient - reference).norm() / reference.norm() < 1e-8


def test_a_trace_of_unsaturated_periods_destroys_the_gradient() -> None:
    """The condition above is sharp: under 1% near-threshold is already fatal.

    Guards against the overreach this test was written to correct — the
    recovery result was first stated as a property of small ``tau``, when it is
    a property of small ``tau`` **plus** a trajectory that never lingers near
    the threshold. A longer horizon makes such visits more likely, which is why
    the safe regime shrinks as the horizon grows.
    """
    torch.manual_seed(11)
    n, horizon = 24, 60
    rate = torch.rand(n, dtype=torch.float64) * 4 + 0.5
    cv = torch.rand(n, dtype=torch.float64) * 0.5 + 0.2
    shape = 1.0 / cv**2
    demand = (
        torch.distributions.Gamma(shape.unsqueeze(1), (shape / rate).unsqueeze(1))
        .sample((horizon,))
        .squeeze(-1)
        .T.to(torch.float64)
    )
    reorder_point = rate * LEAD_TIME + 1.65 * cv * rate * LEAD_TIME**0.5
    capacity = torch.full((n,), 30.0, dtype=torch.float64)

    unsaturated = saturation_fraction(
        capacity, demand, reorder_point, tau=0.02, lead_time=LEAD_TIME
    )
    assert 0.0 < unsaturated < 0.02, unsaturated

    reference = finite_difference(capacity, demand, reorder_point, horizon)
    q = capacity.clone().requires_grad_(True)
    cost = replenishment_cost(
        q, demand, reorder_point, tau=0.02, lead_time=LEAD_TIME
    )
    (gradient,) = torch.autograd.grad(cost, q)
    cosine = torch.nn.functional.cosine_similarity(gradient, reference, dim=0).item()

    assert cosine < 0.5, f"expected the gradient to be destroyed, cosine={cosine}"
    assert gradient.norm() > 100 * reference.norm()


def test_fidelity_improves_monotonically_as_tau_falls(setup) -> None:
    """Trajectory fidelity behaves as in M3: the gap closes as ``tau -> 0``."""
    capacity, demand, reorder_point = setup
    curve = fidelity_curve(
        capacity, demand[:, :10], reorder_point,
        [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02], lead_time=LEAD_TIME,
    )
    assert curve["gap"][0] > curve["gap"][-1]
    assert curve["gap"][-1] / curve["exact_cost"] < 1e-3


def test_gradient_quality_survives_small_tau_when_saturated(setup) -> None:
    """Unlike M3, small ``tau`` here *helps* the gradient rather than inverting it."""
    capacity, demand, reorder_point = setup
    horizon = 10
    reference = finite_difference(capacity, demand, reorder_point, horizon)

    def cosine(tau: float) -> float:
        q = capacity.clone().requires_grad_(True)
        cost = replenishment_cost(
            q, demand[:, :horizon], reorder_point, tau=tau, lead_time=LEAD_TIME
        )
        (gradient,) = torch.autograd.grad(cost, q)
        return torch.nn.functional.cosine_similarity(
            gradient, reference, dim=0
        ).item()

    assert cosine(0.05) > 0.99
    assert cosine(0.02) > 0.99


# ======================================================================
# the failure mode that has no analogue in M3
# ======================================================================
def test_relaxed_loop_is_unstable_at_intermediate_temperature(setup) -> None:
    """The relaxation makes the gradient grow exponentially in the horizon.

    The exact system cannot do this: a step function's derivative is zero
    almost everywhere, so there is no feedback term to compound.
    """
    capacity, demand, reorder_point = setup
    factor = gradient_growth_factor(
        capacity, demand, reorder_point, tau=1.0, lead_time=LEAD_TIME
    )
    assert factor > 1.2, f"expected an unstable loop, got {factor}"

    norms = [
        gradient_norm(capacity, demand, reorder_point, tau=1.0, horizon=h,
                      lead_time=LEAD_TIME)
        for h in (25, 50, 100, 200)
    ]
    assert all(b > 10 * a for a, b in zip(norms, norms[1:])), norms


def test_large_temperature_keeps_the_loop_stable(setup) -> None:
    capacity, demand, reorder_point = setup
    factor = gradient_growth_factor(
        capacity, demand, reorder_point, tau=2.0, lead_time=LEAD_TIME
    )
    assert factor == pytest.approx(1.0, abs=0.02), factor


def test_instability_peaks_at_intermediate_temperature(setup) -> None:
    """Non-monotone in ``tau`` — the danger zone is the middle, not the limit.

    This is the sharpest contrast with M3, where the danger zone is ``tau -> 0``.
    Very small ``tau`` saturates the sigmoid, so its derivative vanishes in
    almost every period and nothing compounds.
    """
    capacity, demand, reorder_point = setup
    curve = stability_curve(
        capacity, demand, reorder_point, [2.0, 1.0, 0.5, 0.2, 0.05],
        [25, 50], lead_time=LEAD_TIME,
    )
    factors = curve["growth_factor"]
    assert factors[0] < factors[1], "large tau should be the stable end"
    assert factors[1] > factors[2] > factors[3], "growth must fall again as tau shrinks"
    assert max(factors) == factors[1], factors

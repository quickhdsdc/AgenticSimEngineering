"""Phase 4 — is the RQ1 relaxation trade-off a general phenomenon?

Run with::

    uv run python examples/phase4_relaxation_across_domains.py

M3 relaxes ``argsort`` **once**. Replenishment relaxes a reorder indicator that
fires **every period** and whose output feeds back into the state. Both are
zero-gradient discrete decisions inside a simulation, relaxed at a temperature.

``PLAN.md`` Phase 4 set the acceptance either way: matching curves would make
the RQ1 finding a cross-domain phenomenon, differing curves would show it
depends on the form of the relaxation. They differ — and the difference is
sharper than a match would have been.
"""

from __future__ import annotations

import torch

from gradientshop.ddes.dispatch import fidelity_curve as dispatch_fidelity_curve

from intralogistics.differentiable import (
    exact_replenishment_cost,
    fidelity_curve,
    replenishment_cost,
    stability_curve,
)

LEAD_TIME = 3
DEMAND_CV = 0.4
TAUS = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
RULE = "=" * 88


def build() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    rate = torch.tensor([2.0, 3.0, 2.5, 4.0, 3.5, 2.0], dtype=torch.float64)
    shape = 1.0 / DEMAND_CV**2
    demand = (
        torch.distributions.Gamma(shape, shape / rate.unsqueeze(1))
        .sample((800,))
        .squeeze(-1)
        .T.to(torch.float64)
    )
    reorder = rate * LEAD_TIME + 1.65 * DEMAND_CV * rate * LEAD_TIME**0.5
    capacity = torch.tensor([25.0, 30.0, 25.0, 35.0, 30.0, 25.0], dtype=torch.float64)
    return capacity, demand, reorder


def finite_difference(capacity, demand, reorder, horizon, step=1e-3):
    gradient = torch.zeros_like(capacity)
    for i in range(capacity.shape[0]):
        up, down = capacity.clone(), capacity.clone()
        up[i] += step
        down[i] -= step
        gradient[i] = (
            exact_replenishment_cost(up, demand[:, :horizon], reorder,
                                     lead_time=LEAD_TIME)
            - exact_replenishment_cost(down, demand[:, :horizon], reorder,
                                       lead_time=LEAD_TIME)
        ) / (2 * step)
    return gradient


def main() -> None:
    capacity, demand, reorder = build()

    # ------------------------------------------------------------------
    print(RULE)
    print("A. M3 dispatch — argsort relaxed ONCE (the existing RQ1 result)")
    print(RULE)
    # The documented E3 instance (examples/m3_fidelity_study.py), reproduced
    # verbatim so this really is the published curve and not a fresh draw.
    generator = torch.Generator().manual_seed(7)
    n = 12
    processing = torch.rand(n, generator=generator, dtype=torch.float64) + 0.2
    weight = torch.rand(n, generator=generator, dtype=torch.float64) + 0.2
    scores = torch.randn(n, generator=generator, dtype=torch.float64)
    taus_m3 = TAUS + [0.01]
    curve = dispatch_fidelity_curve(processing, scores, weight, taus_m3)
    print(f"exact discrete cost = {curve['exact_cost']:.4f}\n")
    print(f"{'tau':>6} {'soft cost':>11} {'gap %':>9} {'grad cosine':>12}")
    for i, tau in enumerate(curve["tau"]):
        print(f"{tau:>6.2f} {curve['soft_cost'][i]:>11.4f} "
              f"{100 * curve['gap'][i] / curve['exact_cost']:>8.3f}% "
              f"{curve['grad_cosine'][i]:>+12.4f}")
    worst = min(curve["grad_cosine"])
    print(f"\n-> fidelity improves as tau falls; gradient agreement bottoms out at "
          f"{worst:+.4f}.")
    if worst < 0:
        print("   The smooth gradient ends up pointing the WRONG WAY.")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("B. Replenishment — reorder trigger relaxed EVERY PERIOD (short horizon)")
    print(RULE)
    horizon = 10
    reference = finite_difference(capacity, demand, reorder, horizon)
    curve = fidelity_curve(capacity, demand[:, :horizon], reorder, TAUS,
                           lead_time=LEAD_TIME)
    print(f"exact discrete cost = {curve['exact_cost']:.4f}   "
          f"||finite-difference gradient|| = {reference.norm():.4f}\n")
    print(f"{'tau':>6} {'soft cost':>11} {'gap %':>9} {'||grad||':>11} "
          f"{'cos(grad, fd)':>14}")
    for tau in TAUS:
        q = capacity.clone().requires_grad_(True)
        cost = replenishment_cost(q, demand[:, :horizon], reorder, tau=tau,
                                  lead_time=LEAD_TIME)
        (gradient,) = torch.autograd.grad(cost, q)
        cosine = torch.nn.functional.cosine_similarity(gradient, reference, dim=0)
        print(f"{tau:>6.2f} {cost.item():>11.4f} "
              f"{100 * abs(cost.item() - curve['exact_cost']) / curve['exact_cost']:>8.3f}% "
              f"{gradient.norm():>11.3e} {cosine:>+14.4f}")
    print("\n-> the OPPOSITE: as tau falls the gradient converges to the exact one.")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("C. Why: the relaxed decision sits inside a feedback loop")
    print(RULE)
    horizons = [25, 50, 100, 200, 400]
    curve = stability_curve(capacity, demand, reorder, [2.0, 1.0, 0.5, 0.2, 0.05],
                            horizons, lead_time=LEAD_TIME)
    header = f"{'tau':>6} " + " ".join(f"{'T=' + str(h):>12}" for h in horizons)
    print(header + f" {'growth/period':>14}")
    for i, tau in enumerate(curve["tau"]):
        row = " ".join(f"{v:>12.3e}" for v in curve["norm"][i])
        print(f"{tau:>6.2f} {row} {curve['growth_factor'][i]:>14.4f}")
    print(
        "\n-> the gradient grows exponentially in the horizon, worst at INTERMEDIATE\n"
        "   tau. The exact system cannot do this: a step function's derivative is\n"
        "   zero a.e., so there is no feedback term to compound. The instability is\n"
        "   created by the relaxation, not approximated by it."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("D. And at small tau the relaxation contributes nothing at all")
    print(RULE)
    q = capacity.clone().requires_grad_(True)
    cost = replenishment_cost(q, demand[:, :horizon], reorder, tau=0.02,
                              lead_time=LEAD_TIME)
    (gradient,) = torch.autograd.grad(cost, q)
    print("finite difference of the EXACT cost:",
          " ".join(f"{v:+.6f}" for v in reference.tolist()))
    print("autodiff through the relaxation    :",
          " ".join(f"{v:+.6f}" for v in gradient.tolist()))
    print(f"\nrelative difference = "
          f"{((gradient - reference).norm() / reference.norm()).item():.3e}")
    print(
        "\n-> with order = Q * 1[...], the sensitivity to Q flows through the order\n"
        "   MAGNITUDE (continuous, differentiable a.e. -- Source A), not through the\n"
        "   order TIMING (the discrete decision -- Source B). In M3 the reverse holds:\n"
        "   which job goes first IS the gradient, and nothing continuous carries it."
    )

    print(f"\n{RULE}")
    print("VERDICT")
    print(RULE)
    print(
        "'Relax the discrete decision' is not a uniform recipe. Two questions come\n"
        "first, and this domain answers both differently from M3:\n\n"
        "  1. Is the decision's sensitivity already carried by a continuous\n"
        "     magnitude?  If yes, relaxation is unnecessary -- and here it is.\n"
        "  2. Does the decision sit inside a feedback loop?  If yes, relaxation is\n"
        "     unstable, exponentially in the horizon.\n\n"
        "The RQ1 trade-off is therefore a property of ONE-SHOT relaxations, not of\n"
        "relaxed simulation in general. That is a sharper claim than a second\n"
        "instance of the same curve would have supported."
    )


if __name__ == "__main__":
    main()

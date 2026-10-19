"""Phase 5.3 — a pathwise gradient through the MILP, and where it stops working.

Run with::

    uv run python examples/phase5_milp_pathwise_gradient.py

(a few minutes: every Monte-Carlo draw is an exact mixed-integer solve)

This runs a **reduced sampling budget** so it finishes in one sitting: 16 draws
per gradient and 6 tuning steps, against the 32 and 8 behind the tables in
``PLAN.md`` §5.3 and ``docs/documentation.md`` §3.6c. The qualitative picture is
the same; the individual numbers are noisier, which is itself the point of
section C — this estimator buys a direction, not a precise one.

The capability table in ``PLAN.md`` §1.6 gives the MILP ``C0 + C2b + C3`` and
withholds ``C2a``. Decision record D10 insists the two first-order objects are
not interchangeable: a shadow price is a price on a right-hand side and cannot
be stepped along. This closes that gap by smoothing the *argmin* itself
(Berthet et al. 2020) — and then measures the three ways the result fails,
because the interesting content of Phase 5.3 is the failure map, not the
capability tick-box.
"""

from __future__ import annotations

import numpy as np

from intralogistics.data import build_minimal_instance
from intralogistics.instances import build_network_instance
from intralogistics.perturbed import (
    DynamicSupplyCost,
    PlanningMilp,
    PlanningRates,
    PlanningWorld,
    StaticTrueCost,
    SupplyDynamics,
    epsilon_curve,
    honest_instance,
    perturbed_gradient,
    spo_plus_subgradient,
    tune_planning_rates,
)
from intralogistics.world import MilpWorld

GEOMETRY = dict(n_items=3, n_resources=2, n_forward=1, seed=0)
HONEST_RATE = 4.0
DEMAND_CV = 0.5
RULE = "=" * 84


def main() -> None:
    evaluation = build_network_instance(**GEOMETRY,
                                        storage_rate_point_of_use=HONEST_RATE)
    loss = DynamicSupplyCost(evaluation, SupplyDynamics(demand_cv=DEMAND_CV))

    # ------------------------------------------------------------------
    print(RULE)
    print("A. Two first-order objects from the same MILP — only one is a direction")
    print(RULE)
    minimal = build_minimal_instance()
    duals = MilpWorld(minimal).duals()
    print("C2b, shadow prices of the LP relaxation:")
    print("   " + "\n   ".join(duals.as_text().splitlines()[1:4]))
    try:
        duals.descent_direction()
    except TypeError as exc:
        print(f"\n   descent_direction() refuses it: {exc}")

    rates = PlanningRates.storage_only(minimal)
    world = PlanningWorld(
        PlanningMilp(minimal, rates),
        StaticTrueCost(honest_instance(minimal)),
        epsilon=0.5,
        n_samples=8,
    )
    pathwise = world.gradient(rates.baseline(minimal))
    print("\nC2a, perturbed argmin gradient — capabilities "
          f"{sorted(c.value for c in world.capabilities())}")
    print(f"   descent_direction() accepts it: "
          f"{np.round(pathwise.descent_direction(), 3).tolist()}")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("B. Failure 1 — a degenerate MILP puts theta ON a jump")
    print(RULE)
    milp = PlanningMilp(minimal, rates)
    theta = rates.baseline(minimal)
    static_loss = StaticTrueCost(honest_instance(minimal))
    print(f"{'eps':>8} {'solutions':>11} {'values':>8} {'||grad||':>12} "
          f"{'eps*||grad||':>14}")
    for epsilon in (0.01, 0.05, 0.2, 1.0, 2.0):
        estimate = perturbed_gradient(milp, static_loss, theta, epsilon=epsilon,
                                      n_samples=8, seed=0)
        print(f"{epsilon:>8.2f} {estimate.distinct_solutions:>11d} "
              f"{estimate.distinct_losses:>8d} "
              f"{np.linalg.norm(estimate.gradient):>12.3f} "
              f"{estimate.scaled_gradient_norm:>14.4f}")
    print(
        "\n-> the same two tied optima come back over a 200-fold range of eps, so\n"
        "   eps*||grad|| is constant: the gradient diverges as 1/eps while pointing\n"
        "   at nothing reachable. The loss varies, so a 'did anything change?' check\n"
        "   passes -- this is why the scaling check is the one that matters."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("C. The eps sweep — the third structural type of relaxation")
    print(RULE)
    planning = build_network_instance(**GEOMETRY,
                                      storage_rate_point_of_use=HONEST_RATE)
    all_rates = PlanningRates.all_rates(planning)
    network = PlanningMilp(planning, all_rates)
    curve = epsilon_curve(
        network, loss, all_rates.baseline(planning),
        [0.02, 0.08, 0.2, 0.6, 2.0], n_samples=16, seed=3,
    )
    print(f"exact loss = {curve['exact_loss']:.4f}   "
          f"({network.n_solves} exact solves)\n")
    print(f"{'eps':>6} {'gap':>9} {'gap %':>8} {'||grad||':>10} "
          f"{'eps*||g||':>11} {'cos':>7} {'solutions':>10}")
    for k, epsilon in enumerate(curve["eps"]):
        print(f"{epsilon:>6.2f} {curve['gap'][k]:>9.4f} "
              f"{100 * curve['gap'][k] / curve['exact_loss']:>7.2f}% "
              f"{curve['grad_norm'][k]:>10.3f} "
              f"{curve['scaled_grad_norm'][k]:>11.3f} "
              f"{curve['grad_cosine'][k]:>+7.3f} "
              f"{curve['distinct_solutions'][k]:>10.0f}")
    print(
        "\n-> forward pass exact at every eps -- the planner never sees a fractional\n"
        "   configuration, which SoftSort and the sigmoid trigger cannot promise.\n"
        "   The price: at the smallest eps NOTHING is within reach and the gradient\n"
        "   is identically zero; widening it buys signal and spends fidelity, and\n"
        "   eps*||grad|| growing (unlike section B) confirms real boundaries."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("D. What it is for: the best planning rates are NOT the true costs")
    print(RULE)
    for start_rate in (1e5, 12.0, HONEST_RATE):
        instance = build_network_instance(**GEOMETRY,
                                          storage_rate_point_of_use=start_rate)
        storage_rates = PlanningRates.storage_only(instance)
        tuner = PlanningMilp(instance, storage_rates)
        result = tune_planning_rates(
            tuner, loss, storage_rates.baseline(instance),
            epsilon=0.8, steps=6, n_samples=16, seed=7,
        )
        tuned = ", ".join(f"{r:.2f}" for r in result.rates)
        print(f"  point-of-use rate starts at {start_rate:>8g}:  "
              f"{result.start_loss:7.3f} -> {result.best_loss:7.3f} "
              f"({result.improvement_pct:+6.2f}%)   tuned rates [{tuned}]")
    print(
        "\n-> starting from the honest costs, DISTORTING them still improves the\n"
        "   simulated outcome. That is the point of decision-focused tuning: the\n"
        "   planning model is time-averaged (Phase 3), so the rates that make its\n"
        "   optimum right are compensation for that bias, not prices. They are a\n"
        "   control knob and must not be read as costs.\n"
        "   Starting from the 1e5 placeholder (Phase 1) nothing moves: the nearest\n"
        "   boundary is four orders of magnitude away, and no local method reaches it."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("E. SPO+ — cheaper, deterministic, and not applicable to the loss we have")
    print(RULE)
    rate_space = PlanningRates.all_rates(minimal, log_space=False)
    spo_milp = PlanningMilp(minimal, rate_space)
    planning_theta = rate_space.baseline(minimal)
    true_theta = rate_space.baseline(honest_instance(minimal))
    subgradient = spo_plus_subgradient(spo_milp, planning_theta, true_theta)
    print("two solves, no variance; nonzero components at the placeholder rates:")
    for name, value in sorted(zip(rate_space.names(), subgradient),
                              key=lambda nv: nv[1]):
        if abs(value) > 1e-9:
            print(f"   {name:<28} {value:+.4f}")
    if not np.any(np.abs(subgradient) > 1e-9):
        print("   (all zero — the planning solution already matches the true one)")
    try:
        spo_plus_subgradient(PlanningMilp(minimal, PlanningRates.all_rates(minimal)),
                             planning_theta, true_theta)
    except ValueError as exc:
        print(f"\n   refused in log space: {exc}")
    print(
        "\n-> SPO+ is a convex surrogate for regret against a KNOWN cost vector, so it\n"
        "   needs the evaluation objective to live in the same linear family as the\n"
        "   planning one. A simulation has no cost vector, so SPO+ cannot be pointed\n"
        "   at DynamicSupplyCost at all. That is the trade: perturbation pays\n"
        "   n_samples solves and variance to buy independence from that assumption."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("VERDICT")
    print(RULE)
    print(
        "C3 -> C2a delivered, with the forward pass left exact. Placed against the\n"
        "other two relaxations in this repository, the failure mode is a third kind:\n\n"
        "  M3 dispatch (one-shot, deterministic)  bias      -- the direction is wrong\n"
        "  replenishment (in-loop, deterministic) instability -- exponential in T\n"
        "  MILP argmin (one-shot, stochastic)     reach     -- there is no local\n"
        "                                                     information at all\n\n"
        "'Reach' is the new one and it is a property of the INSTANCE, not the method:\n"
        "usable eps is set by the distance to the nearest decision boundary. Two\n"
        "diagnostics separate the regimes before any step is taken -- distinct loss\n"
        "values (is anything happening?) and eps*||grad|| (is it a nearby boundary,\n"
        "or a tie underfoot?). They are this phase's analogue of Phase 4's\n"
        "saturation_fraction, and the same rule applies: check first, then act."
    )


if __name__ == "__main__":
    main()

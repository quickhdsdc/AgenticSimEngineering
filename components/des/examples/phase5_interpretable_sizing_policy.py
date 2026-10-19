"""Phase 5.2 — what does an interpretable rule cost?

Run with::

    uv run python examples/phase5_interpretable_sizing_policy.py

The source dissertation derives interpretable rules from a family of MILP
optima with a Random Forest — behaviour cloning from an exact-solver expert.
This is the same idea as a **weight vector over named heuristics**, the
construction GradientShop already uses for dispatching rules, and it answers
the question that chapter has to answer: *how much more expensive is the
interpretable rule than the free per-item optimum?*

Everything is scored on the **simulation**, including the policy that was
trained on the smooth surrogate — the gap between the two is reported rather
than hidden.
"""

from __future__ import annotations

import torch

from intralogistics.policy import Items, SizingPolicyWorld

N_ITEMS = 24
HORIZON = 400
LEAD_TIME = 3
RULE = "=" * 78


def build() -> SizingPolicyWorld:
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


def main() -> None:
    world = build()
    grid = torch.linspace(5, 80, 120, dtype=torch.float64)
    base = torch.full((N_ITEMS,), 40.0, dtype=torch.float64)

    print(RULE)
    print(f"{N_ITEMS} items, {HORIZON} periods, decision = load-carrier capacity")
    print(RULE)
    print("named sizing rules:", ", ".join(world.rule_names))

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("1. Baselines (cost is separable across items, so per-item is exact)")
    print(RULE)
    per_item = base.clone()
    for i in range(N_ITEMS):
        costs = []
        for q in grid:
            candidate = base.clone()
            candidate[i] = q
            costs.append(world.cost_of_capacities(candidate))
        per_item[i] = grid[int(torch.tensor(costs).argmin())]
    optimum = world.cost_of_capacities(per_item)
    flat_cost, flat_q = min(
        (world.cost_of_capacities(torch.full((N_ITEMS,), float(q))), q)
        for q in grid.tolist()
    )

    # ------------------------------------------------------------------
    theta = torch.tensor(
        world.decision_space().initial(), dtype=torch.float64, requires_grad=True
    )
    optimiser = torch.optim.Adam([theta], lr=0.05)
    for _ in range(800):
        optimiser.zero_grad()
        world._cost(theta).backward()
        optimiser.step()
    theta = theta.detach()
    policy_cost = world.simulated_cost(theta)

    print(f"{'approach':<34}{'d.o.f.':>8}{'cost':>12}{'vs optimum':>13}")
    print("-" * 67)
    print(f"{'per-item optimum (free)':<34}{N_ITEMS:>8}{optimum:>12.4f}{'--':>13}")
    print(f"{'rule policy (named weights)':<34}{len(world.rule_names):>8}"
          f"{policy_cost:>12.4f}{100 * (policy_cost - optimum) / optimum:>12.2f}%")
    print(f"{f'one size fits all (Q={flat_q:.1f})':<34}{1:>8}"
          f"{flat_cost:>12.4f}{100 * (flat_cost - optimum) / optimum:>12.2f}%")

    correlation = torch.corrcoef(
        torch.stack([world.capacities(theta), per_item])
    )[0, 1]
    print(f"\ncorrelation of policy capacities with the per-item optimum: "
          f"{correlation:+.4f}")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("2. The trained policy, in words")
    print(RULE)
    print(world.explain(theta))

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("3. Trained on the surrogate, scored on the simulation")
    print(RULE)
    surrogate = world.evaluate(theta.tolist()).objective
    print(f"fluid surrogate (training signal) : {surrogate:.4f}")
    print(f"simulation (what it is worth)     : {policy_cost:.4f}")
    print(f"surrogate gap                     : "
          f"{100 * abs(surrogate - policy_cost) / policy_cost:.2f}%")
    print(
        "\nThe gradient comes from the fluid model on purpose. Phase 3 measured when\n"
        "that surrogate is trustworthy, and Phase 4 measured what backpropagating\n"
        "through the simulation costs instead -- unless the trajectory stays clear of\n"
        "the reorder threshold, the gradient is destroyed. Train on the surrogate,\n"
        "score on the simulation, report the gap."
    )

    print(f"\n{RULE}")
    print("VERDICT")
    print(RULE)
    print(
        f"Eight interpretable weights over named rules land "
        f"{100 * (policy_cost - optimum) / optimum:.2f}% above a free per-item\n"
        f"optimum with {N_ITEMS} degrees of freedom, and well inside the "
        f"{100 * (flat_cost - optimum) / optimum:.2f}% that a single\n"
        "global carrier size costs. That percentage is the price of\n"
        "interpretability -- and it is the number a rule-derivation chapter has to\n"
        "report. Unlike a fitted decision tree, this policy is trained on the actual\n"
        "cost, so it needs no expert solutions at all, though it can imitate them."
    )


if __name__ == "__main__":
    main()

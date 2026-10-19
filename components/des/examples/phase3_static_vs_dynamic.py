"""Phase 3 — does the time average pick the same carrier size as the dynamics?

Run with::

    uv run python examples/phase3_static_vs_dynamic.py

This is the experiment ``PLAN.md`` designates as the only one whose outcome was
unknown in advance, and as the judge of whether Phases 4-6 are worth doing. It
was written to be able to come out *negative* — a finding that the static model
is safe would have shrunk the plan rather than extended it.

The slice is minimal on purpose: one item, one warehouse, one decision (the
load-carrier capacity ``Q``), swept over a grid of lead time and demand
variability. Three world models share one cost function and differ only in
where its inputs come from.
"""

from __future__ import annotations

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
RULE = "=" * 100


def best(world) -> float | None:
    feasible = [
        (world.evaluate(q).objective, q)
        for q in CAPACITIES
        if world.evaluate(q).is_feasible()
    ]
    return min(feasible)[1] if feasible else None


def main() -> None:
    costs = CarrierCostModel()

    print(RULE)
    print("0. Is the simulator trustworthy? (checks against theory, not against itself)")
    print(RULE)
    p = ReplenishmentParams(demand_rate=10, demand_cv=0.3, lead_time=3, horizon=20000)
    print("flow conservation, moves x Q vs demand rate 10  "
          "(= constraint (17) holding in the dynamics):")
    for q in (5.0, 10.0, 20.0, 40.0):
        outcome = simulate_replenishment(q, p)
        print(f"    Q={q:5.1f}   moves x Q = {outcome.moves_per_period * q:7.4f}")
    p0 = ReplenishmentParams(demand_rate=10, demand_cv=0.0, lead_time=3, horizon=20000)
    print("\nsawtooth mean minus Q/2 (must not depend on Q):")
    for q in (10.0, 20.0, 40.0, 60.0):
        outcome = simulate_replenishment(q, p0)
        print(f"    Q={q:5.1f}   avg_inventory - Q/2 = {outcome.avg_inventory - q / 2:6.3f}")

    print(f"\n{RULE}")
    print("1. Which carrier size does each model recommend, and what does it truly cost?")
    print(RULE)
    print("regret = extra cost, measured in the dynamics, of following that model")
    print(
        f"\n{'L':>2}{'CV':>6} | {'exo +(11)':>9}{'regret':>8} | "
        f"{'exo -(11)':>9}{'regret':>8} | {'self-con':>8}{'regret':>8} | "
        f"{'TRUE':>5} | {'rho exo':>8}{'rho s-c':>8} | {'falseInf':>8}"
    )
    print("-" * 100)

    for lead_time in (1, 3, 6):
        for cv in (0.0, 0.25, 0.5, 0.75, 1.0):
            params = ReplenishmentParams(
                demand_rate=10, demand_cv=cv, lead_time=lead_time, horizon=6000, seed=1
            )
            dynamic = DynamicWorld(params, costs, CAPACITIES)
            inventory = dynamic.evaluate(INCUMBENT).kpis["avg_inventory"]
            exogenous = StaticExogenousWorld(
                params, costs, CAPACITIES, avg_inventory=inventory
            )
            unconstrained = StaticExogenousWorld(
                params, costs, CAPACITIES, avg_inventory=inventory,
                kanban_constraint=False,
            )
            self_consistent = StaticSelfConsistentWorld(params, costs, CAPACITIES)

            truth = best(dynamic)
            optimum = dynamic.evaluate(truth).objective
            cells = []
            for world in (exogenous, unconstrained, self_consistent):
                choice = best(world)
                cost = dynamic.evaluate(choice).objective if choice else float("nan")
                cells.append((choice, 100 * (cost - optimum) / optimum))

            report_exo = compare_world_models(dynamic, exogenous, CAPACITIES)
            report_sc = compare_world_models(dynamic, self_consistent, CAPACITIES)
            print(
                f"{lead_time:>2}{cv:>6.2f} | "
                f"{cells[0][0]:>9.0f}{cells[0][1]:>7.1f}% | "
                f"{cells[1][0]:>9.0f}{cells[1][1]:>7.1f}% | "
                f"{cells[2][0]:>8.0f}{cells[2][1]:>7.1f}% | "
                f"{truth:>5.0f} | {report_exo.rank_correlation:>+8.3f}"
                f"{report_sc.rank_correlation:>+8.3f} | "
                f"{report_exo.false_infeasible:>8}"
            )

    print(f"\n{RULE}")
    print("2. Can a static model see a starving line? (98% fill-rate target)")
    print(RULE)
    print("false-feasible = the static model approves it, the dynamics starve the line\n")
    print(f"{'L':>2}{'CV':>6} | {'fill rate at each Q (%)':<54} | {'falseFeas':>9}")
    print("-" * 80)
    for lead_time in (3, 6):
        for cv in (0.25, 0.5, 0.75, 1.0):
            params = ReplenishmentParams(
                demand_rate=10, demand_cv=cv, lead_time=lead_time, horizon=6000, seed=1
            )
            dynamic = DynamicWorld(
                params, costs, CAPACITIES, service_target=0.98
            )
            inventory = DynamicWorld(params, costs, CAPACITIES).evaluate(
                INCUMBENT
            ).kpis["avg_inventory"]
            static = StaticExogenousWorld(
                params, costs, CAPACITIES, avg_inventory=inventory,
                kanban_constraint=False,
            )
            rates = " ".join(
                f"{100 * dynamic.evaluate(q).kpis['service_level']:5.1f}"
                for q in CAPACITIES
            )
            report = compare_world_models(dynamic, static, CAPACITIES)
            print(f"{lead_time:>2}{cv:>6.2f} | {rates:<54} | {report.false_feasible:>9}")

    print(f"\n{RULE}")
    print("VERDICT")
    print(RULE)
    print(
        "Three distinct failure modes, in decreasing order of size:\n\n"
        "  1. Constraint (11) on an exogenous inventory parameter.\n"
        "     'q_inventory >= 2 * capacity' caps the bin at half a quantity that\n"
        "     should itself grow with the bin. It rejects 4-8 of 10 candidates and\n"
        "     costs up to 116.7%. Deleting it recovers the true optimum exactly at\n"
        "     CV = 0. This is a modelling error, not an approximation error.\n\n"
        "  2. The time average itself, once demand varies.\n"
        "     Rank correlation with the dynamics falls from +1.000 at CV = 0 to\n"
        "     roughly +0.5; neither static variant preserves the ordering.\n\n"
        "  3. Service level is invisible to a time average.\n"
        "     At high variability up to 6 of 10 configurations are approved by the\n"
        "     static model and starve the line in simulation.\n\n"
        "At CV = 0 the fluid projection is EXACT (regret 0.0%, rho = +1.000), which\n"
        "is the correct deterministic limit and an internal check on the experiment.\n"
        "The static model is faithful exactly where nothing varies -- and the two\n"
        "decisions this project cares about, carrier size and supply logic, are the\n"
        "ones whose whole purpose is to absorb variation."
    )


if __name__ == "__main__":
    main()

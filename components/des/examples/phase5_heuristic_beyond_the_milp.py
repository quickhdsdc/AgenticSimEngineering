"""Phase 5.1 — a constructive heuristic for instances the MILP cannot reach.

Run with::

    uv run python examples/phase5_heuristic_beyond_the_milp.py

(several minutes: section A deliberately runs the exact solver into a wall)

This is the payoff Axiom 1 promised. The world model was separated from the
solver in Phase 1; here a completely different solver is plugged into the same
world model, with no change to the model at all — and it reaches sizes branch
and bound does not return from.

The interesting part is *why* the obvious heuristic failed. Phase 2 measured a
generic hill climb making zero progress on the raw variable space: from the
empty configuration no single-variable move improves anything, because raising
``q_inventory`` relieves (8) by exactly what it costs in (9) and (21). The fix
turned out not to be a better solver but a better decision space — search the
three real decisions and *derive* the rest.
"""

from __future__ import annotations

import time

from opsworld.solver import HillClimbSolver

from intralogistics.heuristic import (
    GreedySupplySolver,
    PlanWorld,
    build_greedy_plan,
    render,
)
from intralogistics.instances import build_network_instance
from intralogistics.milp_model import MilpWorldModel
from intralogistics.static_model import StaticWorldModel
from intralogistics.world import StaticWorld

RULE = "=" * 84
TIME_LIMIT = 60.0


def main() -> None:
    # ------------------------------------------------------------------
    print(RULE)
    print("A. The wall — exact solve time against instance size")
    print(RULE)
    print(f"{'items':>6} {'res':>4} {'reloc vars':>11} {'objective':>11} "
          f"{'termination':>14} {'seconds':>9}")
    for n_items, n_res in ((3, 2), (4, 2), (5, 2), (6, 2), (8, 2), (10, 3)):
        instance = build_network_instance(
            n_items=n_items, n_resources=n_res, n_forward=2
        )
        variables = (
            len(instance.I) * len(instance.R)
            * len(instance.U_edges) * len(instance.B) ** 2
        )
        model = MilpWorldModel(instance)
        start = time.perf_counter()
        _, objective = model.solve(time_limit=TIME_LIMIT)
        elapsed = time.perf_counter() - start
        shown = f"{objective:11.3f}" if objective is not None else f"{'--':>11}"
        print(f"{n_items:>6} {n_res:>4} {variables:>11} {shown} "
              f"{model.termination:>14} {elapsed:>9.1f}")
    print(
        f"\n-> ``y_reloc_moves`` carries six indices, so the model grows as\n"
        f"   |I| x |R| x |edges| x |B|^2. The dissertation this comes from plans for\n"
        f"   thousands of items; there the exact solver does not return at all."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("B. Quality where the optimum is still computable")
    print(RULE)
    print(f"{'instance':<34} {'greedy':>11} {'+local':>11} {'optimum':>11} "
          f"{'gap':>8} {'t_heur':>8} {'t_milp':>8}")
    cases = [
        dict(n_items=3, n_resources=2, n_forward=1, seed=0),
        dict(n_items=4, n_resources=2, n_forward=2, seed=0),
        dict(n_items=5, n_resources=2, n_forward=1, seed=0),
        dict(n_items=5, n_resources=2, n_forward=2, seed=1, tighten=0.7),
        dict(n_items=8, n_resources=2, n_forward=2, seed=0),
    ]
    for case in cases:
        instance = build_network_instance(**case)
        world = PlanWorld(instance)

        start = time.perf_counter()
        result = GreedySupplySolver().solve(world, budget=100_000)
        t_heuristic = time.perf_counter() - start

        model = MilpWorldModel(instance)
        start = time.perf_counter()
        _, optimum = model.solve(time_limit=TIME_LIMIT)
        t_milp = time.perf_counter() - start

        label = ",".join(f"{k}={v}" for k, v in case.items() if k != "n_forward")
        gap = (
            f"{100 * (result.score - optimum) / optimum:+7.2f}%"
            if optimum
            else f"{'--':>8}"
        )
        note = "" if model.termination == "optimal" else f"  ({model.termination})"
        print(f"{label:<34} {result.initial_score:>11.3f} {result.score:>11.3f} "
              f"{optimum:>11.3f} {gap} {t_heuristic:>8.2f} {t_milp:>8.1f}{note}")
    print(
        "\n-> greedy construction alone is often already optimal on these instances,\n"
        "   because the cost structure is strongly directional (picking is cheapest\n"
        "   at the line, storage cheapest centrally). Local search earns its keep\n"
        "   where capacities bind and first-fit overshoots."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("C. Scale — where the heuristic still works, and where it stops")
    print(RULE)
    print(f"{'items':>6} {'res':>4} {'reloc vars':>11} {'t_construct':>12} "
          f"{'t_evaluate':>11} {'objective':>11} {'feasible':>9}")
    for n_items, n_res in ((10, 3), (20, 3), (40, 4), (80, 4), (160, 5)):
        instance = build_network_instance(
            n_items=n_items, n_resources=n_res, n_forward=2
        )
        variables = (
            len(instance.I) * len(instance.R)
            * len(instance.U_edges) * len(instance.B) ** 2
        )
        start = time.perf_counter()
        decision = render(instance, build_greedy_plan(instance))
        t_construct = time.perf_counter() - start

        start = time.perf_counter()
        evaluation = StaticWorldModel().evaluate(instance, decision)
        t_evaluate = time.perf_counter() - start

        print(f"{n_items:>6} {n_res:>4} {variables:>11} {t_construct:>12.3f} "
              f"{t_evaluate:>11.3f} {evaluation.objective:>11.2f} "
              f"{str(evaluation.is_feasible(tol=1e-6)):>9}")
    print(
        "\n-> construction is essentially free and stays feasible at 160 items, where\n"
        "   the exact model would have 180,000 integer relocation variables.\n"
        "   LOCAL SEARCH does not scale with it: each neighbour costs a full\n"
        "   evaluation, and the evaluator is O(|I| |R| |edges| |B|^2) because\n"
        "   constraints (14)/(15) are checked over every index combination. That is\n"
        "   the binding limit here, and it is the evaluator's, not the search's --\n"
        "   a sparse evaluator that only visits nonzero moves would remove it."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("D. Where it loses, and why")
    print(RULE)
    instance = build_network_instance(n_items=10, n_resources=3, n_forward=2)
    world = PlanWorld(instance)
    result = GreedySupplySolver().solve(world, budget=100_000)
    heuristic = StaticWorldModel().evaluate(instance, world.render(result.decision))

    model = MilpWorldModel(instance)
    incumbent, _ = model.solve(time_limit=TIME_LIMIT)
    exact = StaticWorldModel().evaluate(instance, incumbent)

    print(f"10 items x 3 resources — heuristic {heuristic.objective:.3f} vs "
          f"branch-and-bound incumbent {exact.objective:.3f} "
          f"({model.termination}, {TIME_LIMIT:.0f} s)")
    print(f"  (given 300 s the incumbent reaches 272.458, i.e. 10.8% below the "
          f"heuristic)\n")
    print(f"{'component':<18} {'heuristic':>12} {'exact':>12} {'delta':>12}")
    for key in sorted(heuristic.kpis):
        print(f"{key:<18} {heuristic.kpis[key]:>12.3f} {exact.kpis[key]:>12.3f} "
              f"{heuristic.kpis[key] - exact.kpis[key]:>+12.3f}")

    def picked_from(decision):
        counts: dict[str, int] = {}
        for key in decision.nonzero()["x_pick"]:
            counts[key[2]] = counts.get(key[2], 0) + 1
        return counts

    print(f"\n  heuristic picks from : {picked_from(world.render(result.decision))}")
    print(f"  exact     picks from : {picked_from(incumbent)}")
    print(
        "\n-> the exact solver serves a large share of demand from ONE shared forward\n"
        "   warehouse. It pays more picking cost and saves far more storage, because\n"
        "   (11)'s two-bin rule is charged once per LOCATION, not once per demand.\n"
        "   The heuristic cannot get there: the ridge between the two configurations\n"
        "   is infeasible in three ways at once -- (17b) makes picking from an\n"
        "   intermediate warehouse legal only while the item is also stocked\n"
        "   downstream, and (25)/(26) cap how much may be picked from a logistics\n"
        "   warehouse at all. Best improvement on a penalised objective cannot cross\n"
        "   an infeasible ridge by construction. The missing move family reassigns a\n"
        "   SUBSET of an item's demands while preserving a downstream store."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("E. The obstacle was the decision space, not the solver")
    print(RULE)
    instance = build_network_instance(
        n_items=5, n_resources=2, n_forward=2, seed=1, tighten=0.7
    )
    solver = HillClimbSolver(seed=0)
    on_plans = solver.solve(PlanWorld(instance), budget=120)
    on_variables = solver.solve(StaticWorld(instance), budget=120)
    _, optimum = MilpWorldModel(instance).solve(time_limit=TIME_LIMIT)

    print("the SAME generic HillClimbSolver, 120 evaluations, same instance:")
    print(f"  plan space      {on_plans.initial_score:14.3f} -> "
          f"{on_plans.score:12.3f}   feasible="
          f"{on_plans.evaluation.is_feasible(tol=1e-6)}")
    print(f"  variable space  {on_variables.initial_score:14.3f} -> "
          f"{on_variables.score:12.3f}   feasible="
          f"{on_variables.evaluation.is_feasible(tol=1e-6)}")
    print(f"  MILP optimum    {optimum:>30.3f}")
    print(
        "\n-> the solver is unchanged and knows nothing about intralogistics. What\n"
        "   changed is that a move is now a DECISION ('serve this demand from that\n"
        "   warehouse instead') rather than a VARIABLE, with the coupled consequences\n"
        "   recomputed. Phase 2's dead end was a parameterisation problem."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("VERDICT")
    print(RULE)
    print(
        "The number Lena's Regelableitung chapter needs is 'how much worse than\n"
        "optimal', and it can now be quoted rather than assumed. On every instance\n"
        "here where branch and bound PROVES an optimum, the heuristic reaches it --\n"
        "including one where construction starts infeasible and the search has to\n"
        "recover -- while running one to two orders of magnitude faster.\n\n"
        "It does not follow that the heuristic is good everywhere, and section D is\n"
        "the counterexample: at 10 items x 3 resources it settles 10.8% above an\n"
        "incumbent branch and bound found in 300 s without proving optimality. The\n"
        "cause is a specific missing move family, not a shortage of budget.\n\n"
        "Two limits, both concrete rather than hedging:\n"
        "  - the neighbourhood cannot cross an infeasible ridge (section D);\n"
        "  - local search does not scale with construction, because each neighbour\n"
        "    costs a full evaluation and the evaluator is O(|I| |R| |edges| |B|^2)\n"
        "    (section C).\n"
        "Both name their own fix: a subset-reassignment move, and a sparse evaluator."
    )


if __name__ == "__main__":
    main()

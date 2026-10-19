"""Phase 1 — the Wirkmodell, separated from the solver, and cross-validated.

Run with::

    uv run python examples/intralogistics_cross_validation.py

What it demonstrates:

1. The intralogistics MILP now solves with **no commercial licence**
   (Pyomo + HiGHS instead of gurobipy).
2. The cost-and-feasibility model can be evaluated with **no solver at all** —
   this is the *Wirkmodell* of the source dissertation, made executable.
3. Three independent implementation paths agree exactly, which is an
   independent verification of the solver's answer (V&V).
4. A structural property of the shipped test dataset that only becomes
   visible once you can evaluate configurations directly.
"""

from __future__ import annotations

from intralogistics import (
    Instance,
    MilpWorldModel,
    StaticWorldModel,
    available_solver,
    build_minimal_data,
)
from intralogistics.scenarios import default_scenarios, run_scenario_study, summarize

RULE = "=" * 78


def main() -> None:
    solver = available_solver()
    if solver is None:
        raise SystemExit("No MILP solver found. Install one with: uv add highspy")

    instance = Instance(build_minimal_data())
    print(RULE)
    print(f"Instance : {instance.id()}")
    print(f"Solver   : {solver}  (open source, no licence required)")
    print(
        f"Stages   : goods receipt {instance.L_we} -> intermediate "
        f"{instance.L_log} -> point of use {instance.L_prod}"
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}\n1. Solve the MILP (Entscheidungsmodell)\n{RULE}")
    model = MilpWorldModel(instance)
    decision, objective = model.solve()
    if decision is None:
        raise SystemExit(f"No feasible solution ({model.termination}).")
    print(f"termination      : {model.termination}")
    print(f"solver objective : {objective:.6f}")

    # ------------------------------------------------------------------
    print(f"\n{RULE}\n2. Evaluate the same decision WITHOUT a solver (Wirkmodell)\n{RULE}")
    static = StaticWorldModel().evaluate(instance, decision)
    print(static.summary())

    # ------------------------------------------------------------------
    print(f"\n{RULE}\n3. Cross-validation — three independent paths\n{RULE}")
    algebraic = model.evaluate(instance, decision)
    print(f"{'path':<34s}{'objective':>20s}")
    print(f"{'branch-and-bound (HiGHS)':<34s}{objective:>20.9f}")
    print(f"{'Pyomo expression algebra':<34s}{algebraic.objective:>20.9f}")
    print(f"{'standalone numpy evaluator':<34s}{static.objective:>20.9f}")
    print(f"\nmax pairwise difference : "
          f"{max(abs(objective - static.objective), abs(objective - algebraic.objective)):.3e}")
    print(f"independent feasibility : "
          f"{'FEASIBLE (0 violations)' if static.is_feasible() else static.violated()}")

    print("\ncost decomposition:")
    for name, value in sorted(static.kpis.items(), key=lambda kv: -kv[1]):
        print(f"   {name:<18s}{value:>16.4f}{100 * value / static.objective:>10.4f}%")

    # ------------------------------------------------------------------
    print(f"\n{RULE}\n4. Finding — point-of-use storage is forced, not chosen\n{RULE}")
    print(
        "Constraints (6)+(7) admit exactly one pick warehouse per resource, and\n"
        "(10)->(9)->(19)->(11)->(21) then force inventory and load carriers there.\n"
        "So `storage_cost_rate` at L1/L2 scales a term the model cannot avoid.\n"
    )
    print(f"{'rate(L1,L2,L3)':>16s}{'objective':>16s}{'storage':>14s}{'everything else':>18s}")
    picks = []
    for rate in (0.0, 2.0, 100000.0):
        data = build_minimal_data()
        for warehouse in ("L1", "L2", "L3"):
            data["storage_cost_rate"][warehouse] = rate
        probe = Instance(data)
        probe_model = MilpWorldModel(probe)
        probe_decision, probe_objective = probe_model.solve()
        assert probe_decision is not None and probe_objective is not None
        storage = probe_model.evaluate(probe, probe_decision).kpis["cost_storage"]
        picks.append(sorted(k for k, v in probe_decision.x_pick.items() if v > 0.5))
        print(
            f"{rate:>16.1f}{probe_objective:>16.2f}{storage:>14.2f}"
            f"{probe_objective - storage:>18.2f}"
        )
    print(
        f"\npick assignment identical across all rates : {picks[0] == picks[1] == picks[2]}"
        f"\n=> the rate changes the objective value but NOT the optimal decision."
        f"\n   At the shipped rate of 1e5 it accounts for "
        f"{100 * static.kpis['cost_storage'] / static.objective:.3f}% of the objective."
    )

    # ------------------------------------------------------------------
    print(f"\n{RULE}\n5. Scenario study\n{RULE}")
    print(summarize(run_scenario_study(instance, default_scenarios())))
    print(
        "\nNote: the spread is driven almost entirely by scaling the "
        "point-of-use\npenalty above, not by the storage/carrier/routing "
        "trade-offs the model\nis meant to study. Two scenarios have no effect at all."
    )


if __name__ == "__main__":
    main()

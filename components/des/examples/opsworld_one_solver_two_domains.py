"""Phase 2 — one solver, two unrelated domains.

Run with::

    uv run python examples/opsworld_one_solver_two_domains.py

The framework's claim is that the **world model is the shared asset and solvers
are plugins** (``PLAN.md`` Axiom 1), and that which solvers a domain admits is
decided by the **capabilities** its world model exposes (Axiom 4). This script
is the evidence:

1. A single ``HillClimbSolver`` instance optimises a differentiable
   discrete-event shopfloor model *and* a constrained intralogistics
   configuration problem, knowing nothing about either.
2. A gradient solver runs on the domain that offers ``C2a`` and is refused —
   loudly — by the one that does not.
3. A dual variable cannot be mistaken for a search direction.
"""

from __future__ import annotations

import torch

from gradientshop.world import DispatchWorld
from intralogistics import Instance, MilpWorldModel, build_minimal_data
from intralogistics.world import MilpWorld, StaticWorld
from opsworld import (
    GradientDescentSolver,
    HillClimbSolver,
    compare_world_models,
)

RULE = "=" * 78


def build_dispatch_world(n: int = 12) -> DispatchWorld:
    generator = torch.Generator().manual_seed(7)
    return DispatchWorld(
        processing=torch.rand(n, generator=generator, dtype=torch.float64) * 9 + 1,
        weight=torch.rand(n, generator=generator, dtype=torch.float64) * 4 + 1,
        due=torch.rand(n, generator=generator, dtype=torch.float64) * 40 + 10,
    )


def main() -> None:
    dispatch = build_dispatch_world()
    instance = Instance(build_minimal_data())
    static = StaticWorld(instance)

    print(RULE)
    print("1. ONE solver instance, TWO domains")
    print(RULE)
    solver = HillClimbSolver(seed=0)  # <- the same object drives both
    for label, world in (("dispatch (DDES)", dispatch), ("intralogistics", static)):
        capabilities = ", ".join(sorted(c.value for c in world.capabilities()))
        result = solver.solve(world, budget=400)
        print(f"\n{label}   [{capabilities}]")
        print(f"    {result.summary()}")
        print(f"    feasible: {result.evaluation.is_feasible()}")

    print(f"\n{RULE}")
    print("2. How close does a capability-C0-only solver get?")
    print(RULE)
    _, optimum = MilpWorldModel(instance).solve()
    result = HillClimbSolver(seed=0).solve(static, budget=400)
    print(f"constructed start : {result.initial_score:>14.2f}"
          f"   gap {100 * (result.initial_score - optimum) / optimum:6.2f}%")
    print(f"hill climb        : {result.score:>14.2f}"
          f"   gap {100 * (result.score - optimum) / optimum:6.2f}%")
    print(f"MILP optimum      : {optimum:>14.2f}   (branch and bound)")
    print("\nNot a claim that local search competes with branch and bound -- the\n"
          "point is that the world model responds to a solver that knows nothing\n"
          "about it.")

    print(f"\n{RULE}")
    print("3. Capabilities gate solvers -- loudly")
    print(RULE)
    print(f"gradient solver on dispatch      : "
          f"{GradientDescentSolver(lr=0.3).solve(dispatch, budget=40).summary()}")
    try:
        GradientDescentSolver().solve(static, budget=5)
    except TypeError as error:
        print(f"gradient solver on intralogistics: refused\n    {error}")

    print(f"\n{RULE}")
    print("4. A dual is not a descent direction (decision record D10)")
    print(RULE)
    duals = MilpWorld(instance).duals()
    print(f"kind = {duals.kind.value}")
    print(duals.as_text())
    try:
        duals.descent_direction()
    except TypeError as error:
        print(f"\nstepping along it is refused:\n    {error}")

    print(f"\n{RULE}")
    print("5. Fidelity between two world models of the same system")
    print(RULE)
    space = static.decision_space()
    import random

    rng = random.Random(3)
    decisions = [space.initial()]
    for _ in range(9):
        decisions.append(space.perturb(decisions[-1], rng))
    print(compare_world_models(MilpWorld(instance), static, decisions).summary())
    print("\nThe same comparison API will measure the relaxation gaps that are the\n"
          "shared scientific question: SoftSort temperature, and the time-average\n"
          "projection of the intralogistics dynamics (PLAN.md Phase 3).")


if __name__ == "__main__":
    main()

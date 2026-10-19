"""D2 / RQ2 proof of concept: an agent learns a dispatch policy from DDES
gradient feedback, closing the loop the proposal describes.

Run: ``uv run python examples/d2_llm_ddes_loop.py``

The DDES world model returns, after each policy proposal, a *verbalised*
gradient critique -- the structured observation an LLM would read. Here a
gradient-descent agent stands in for the LLM (so it runs with no API key), but
it consumes exactly the same feedback object a :class:`PromptedLLMAgent` would.
The printed observation is the concrete RQ2 artifact: "linguistic self-
evaluation grounded in simulation-backed state trajectories".
"""

from __future__ import annotations

import torch

from gradientshop.agent import (
    DispatchWorldModel,
    GradientDescentAgent,
    run_coupling_loop,
)
from gradientshop.ddes.dispatch import exact_dispatch_cost


def main() -> None:
    g = torch.Generator().manual_seed(4)
    n = 15
    processing = torch.rand(n, generator=g, dtype=torch.float64) + 0.3
    weight = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    due = torch.rand(n, generator=g, dtype=torch.float64) * processing.sum().item()

    world = DispatchWorldModel(processing, weight, due, tau=0.1, objective="wct")
    agent = GradientDescentAgent(world.n_features, lr=0.3)
    history = run_coupling_loop(world, agent, steps=80)

    wspt_opt = exact_dispatch_cost(processing, weight / processing, weight).item()

    print("=== Verbalised DDES feedback the agent reads on turn 0 ===\n")
    print(history[0].as_text())

    print("\n=== Closed-loop learning curve (true schedule cost) ===")
    for t in (0, 5, 20, 40, 79):
        print(f"  step {t:2d}: cost = {history[t].objective:.3f}")
    print(f"\n  WSPT optimum (lower bound) = {wspt_opt:.3f}")
    print(f"  start -> end cost          = {history[0].objective:.3f} "
          f"-> {history[-1].objective:.3f}")
    print("\nThe agent never re-sampled the shopfloor: every improvement came\n"
          "from one differentiable DDES replay per turn (proposal RQ2).")


if __name__ == "__main__":
    main()

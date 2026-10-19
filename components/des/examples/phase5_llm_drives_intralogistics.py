"""Phase 5.4 — RQ2 in a second domain, on the same harness.

Run with::

    uv run python examples/phase5_llm_drives_intralogistics.py

RQ2 asks whether a simulation replay can be embedded as a reasoning step inside
LLM inference. Shown in one domain, that leaves open whether the coupling was
tailored to the domain. Here the *same* ReAct harness, tool protocol and
coupling loop drive material supply instead of shopfloor dispatch.

Everything below runs offline. A scripted model stands in for a real one for
the tool-calling section; the coupling loop needs no model at all. To run the
ReAct part against a real endpoint, configure ``opsworld/agent/config.toml``.
"""

from __future__ import annotations

import asyncio
import json

import torch

from gradientshop.world import DispatchWorld
from intralogistics.agent import build_sizing_session, build_sizing_tools, load_persona
from intralogistics.policy import Items, SizingPolicyWorld
from opsworld.agent import Agent, AuditLog, GradientAgent, run_coupling_loop
from opsworld.agent.coupling import PromptedAgent
from opsworld.agent.llm import LLMResponse, ToolCall

N_ITEMS = 16
LEAD_TIME = 3
RULE = "=" * 78


class ScriptedLLM:
    def __init__(self, actions: list[dict]) -> None:
        self.actions, self.i = actions, 0

    async def ask_tool(self, messages, tools, tool_choice="auto") -> LLMResponse:
        action = self.actions[self.i]
        self.i += 1
        if action["type"] == "final":
            return LLMResponse(text=action["text"])
        return LLMResponse(tool_calls=[
            ToolCall(id=f"c{self.i}", name=action["name"],
                     arguments=action.get("args", {}))
        ])


def build_world() -> SizingPolicyWorld:
    torch.manual_seed(5)
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
        .sample((300,))
        .squeeze(-1)
        .T.to(torch.float64)
    )
    return SizingPolicyWorld(items, demand, lead_time=LEAD_TIME)


def main() -> None:
    world = build_world()
    dimension = len(world.rule_names)

    print(RULE)
    print("1. The world model, verbalised — what a model actually reads")
    print(RULE)
    evaluation = world.evaluate([0.05] * dimension)
    print(evaluation.as_text())

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("2. Closed loop: replay -> critique -> revised policy")
    print(RULE)
    agent = GradientAgent(dimension, lr=0.3)
    theta = agent.initial()
    print(f"{'turn':>6}{'surrogate cost':>18}{'simulated cost':>18}")
    for turn in range(61):
        evaluation = world.evaluate(theta)
        if turn in (0, 5, 20, 60):
            print(f"{turn:>6}{evaluation.objective:>18.4f}"
                  f"{world.simulated_cost(theta):>18.4f}")
        theta = agent.propose(evaluation)
    print("\nEvery improvement comes from one differentiable replay per turn -- no\n"
          "re-sampling of the shopfloor, and no expert solutions to imitate.")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("3. The same loop, driven through TEXT (prompt/parse path)")
    print(RULE)
    inner = GradientAgent(dimension, lr=0.3)
    prompts: list[str] = []

    def complete(messages: list[dict]) -> str:
        prompts.append(messages[-1]["content"])
        return json.dumps(inner.propose(world.evaluate(inner.initial())))

    text_trace = run_coupling_loop(world, PromptedAgent(complete, dimension), steps=15)
    print("first prompt the 'model' received:\n")
    print("\n".join("    " + line for line in prompts[0].splitlines()[:14]))
    print(f"\n    ... cost {text_trace[0].objective:.4f} -> "
          f"{text_trace[-1].objective:.4f} over {len(text_trace) - 1} turns")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("4. A tool-calling agent, through the real ReAct harness")
    print(RULE)
    theta = [0.05] * dimension
    session = build_sizing_session(world)
    actions = [
        {"type": "tool", "name": "describe_scenario"},
        {"type": "tool", "name": "evaluate_policy", "args": {"theta": theta}},
        {"type": "tool", "name": "check_gradient_health", "args": {"theta": theta}},
        {"type": "tool", "name": "compare_to_simulation", "args": {"theta": theta}},
        {"type": "tool", "name": "inspect_capacities", "args": {"theta": theta}},
        {"type": "final", "text": "Policy recorded."},
    ]
    agent = Agent(
        llm=ScriptedLLM(actions),
        tools=build_sizing_tools(),
        system_prompt=load_persona(),
        on_event=lambda event: print(f"    {event}"),
        audit=AuditLog(),
    )
    asyncio.run(agent.run("Find a good carrier-sizing policy.", session=session))
    print(f"\n    best recorded: cost={session['best']['objective']:.4f}")

    # ------------------------------------------------------------------
    print(f"\n{RULE}")
    print("5. One loop, two domains")
    print(RULE)
    generator = torch.Generator().manual_seed(7)
    n = 12
    dispatch = DispatchWorld(
        processing=torch.rand(n, generator=generator, dtype=torch.float64) * 9 + 1,
        weight=torch.rand(n, generator=generator, dtype=torch.float64) * 4 + 1,
        due=torch.rand(n, generator=generator, dtype=torch.float64) * 40 + 10,
    )
    for label, target in (("shopfloor dispatch", dispatch),
                          ("material supply", world)):
        size = len(target.decision_space().names)
        run = run_coupling_loop(target, GradientAgent(size, lr=0.3), steps=40)
        print(f"    {label:<22} {run[0].objective:>10.4f} -> {run[-1].objective:>10.4f}"
              f"   ({target.provenance})")

    print(f"\n{RULE}")
    print(
        "The harness, the tool protocol, the coupling loop and the audit format are\n"
        "imported, not reimplemented. Adding a domain to the agent story costs one\n"
        "module of tools and one persona -- which is the evidence that the RQ2\n"
        "coupling is a property of the interface rather than of the shopfloor."
    )


if __name__ == "__main__":
    main()

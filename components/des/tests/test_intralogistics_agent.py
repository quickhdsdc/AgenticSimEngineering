"""Phase 5.4 — RQ2 in a second domain, on the same harness.

RQ2 asks whether a simulation replay can be embedded as a reasoning step inside
LLM inference. Demonstrating that in one domain leaves open whether the
coupling was tailored to it. These tests drive the *same* ReAct harness, the
*same* tool protocol and the *same* coupling loop against material supply
instead of shopfloor dispatch — and the harness is imported, not reimplemented.

A scripted async model stands in for a real one, so the plumbing is exercised
end to end with no API key.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch

from intralogistics.agent import (
    build_sizing_session,
    build_sizing_tools,
    load_persona,
)
from intralogistics.policy import Items, SizingPolicyWorld
from opsworld.agent import Agent, AuditLog, GradientAgent, run_coupling_loop
from opsworld.agent.llm import LLMResponse, ToolCall
from opsworld.agent.coupling import PromptedAgent

N_ITEMS = 12
LEAD_TIME = 3


class ScriptedLLM:
    """Async LLM replaying a fixed list of tool calls, then a final message."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.i = 0

    async def ask_tool(self, messages, tools, tool_choice="auto") -> LLMResponse:
        action = self.actions[self.i]
        self.i += 1
        if action["type"] == "final":
            return LLMResponse(text=action["text"])
        return LLMResponse(tool_calls=[
            ToolCall(id=f"call{self.i}", name=action["name"],
                     arguments=action.get("args", {}))
        ])


@pytest.fixture
def world() -> SizingPolicyWorld:
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
        .sample((200,))
        .squeeze(-1)
        .T.to(torch.float64)
    )
    return SizingPolicyWorld(items, demand, lead_time=LEAD_TIME)


# ======================================================================
# the tools, through the real ReAct harness
# ======================================================================
def test_react_agent_drives_every_sizing_tool(world, tmp_path) -> None:
    session = build_sizing_session(world)
    theta = [0.05] * len(world.rule_names)

    actions = [
        {"type": "tool", "name": "describe_scenario"},
        {"type": "tool", "name": "evaluate_policy", "args": {"theta": theta}},
        {"type": "tool", "name": "inspect_capacities", "args": {"theta": theta}},
        {"type": "tool", "name": "compare_to_simulation", "args": {"theta": theta}},
        {"type": "tool", "name": "check_gradient_health", "args": {"theta": theta}},
        {"type": "final", "text": "Policy recorded."},
    ]
    agent = Agent(
        llm=ScriptedLLM(actions),
        tools=build_sizing_tools(),
        system_prompt=load_persona(),
        audit=AuditLog(log_dir=tmp_path),
    )
    answer = asyncio.run(
        agent.run("Find a good carrier-sizing policy.", session=session)
    )

    assert "Policy recorded" in str(answer)
    assert session["best"]["theta"] == theta
    assert session["best"]["objective"] > 0
    assert len(list(tmp_path.iterdir())) == 1, "the run should be audited"


def test_tools_reject_a_wrong_length_policy(world) -> None:
    tools = build_sizing_tools()
    session = build_sizing_session(world)
    result = asyncio.run(
        tools.execute("evaluate_policy", {"theta": [0.1, 0.2]}, session)
    )
    assert result.error is not None
    assert "expected" in result.error


def test_tools_report_an_empty_session_back_to_the_model() -> None:
    """The harness turns tool exceptions into results rather than raising.

    That is deliberate — a model that calls a tool wrongly should be able to
    read the complaint and recover, not have the run torn down. The tool still
    has to say something actionable.
    """
    tools = build_sizing_tools()
    result = asyncio.run(tools.execute("describe_scenario", {}, {}))
    assert result.error is not None
    assert "build_sizing_session" in result.error


def test_unknown_tool_is_reported_not_raised() -> None:
    tools = build_sizing_tools()
    result = asyncio.run(tools.execute("no_such_tool", {}, {}))
    assert result.error is not None and "unknown tool" in result.error


def test_persona_names_the_rules_and_the_tools() -> None:
    persona = load_persona()
    for token in ("HIGH_TURNOVER", "evaluate_policy", "compare_to_simulation",
                  "check_gradient_health"):
        assert token in persona
    with pytest.raises(FileNotFoundError):
        load_persona("no_such_persona")


def test_gradient_health_tool_reports_a_verdict(world) -> None:
    session = build_sizing_session(world)
    tools = build_sizing_tools()
    result = asyncio.run(
        tools.execute("check_gradient_health",
                      {"theta": [0.05] * len(world.rule_names)}, session)
    )
    assert result.error is None
    assert "unsaturated periods" in result.output


# ======================================================================
# the coupling loop, now shared between domains
# ======================================================================
def test_gradient_agent_learns_a_sizing_policy(world) -> None:
    agent = GradientAgent(len(world.rule_names), lr=0.3)
    trace = run_coupling_loop(world, agent, steps=40)
    assert trace[-1].objective < trace[0].objective
    assert all(e.sensitivity is not None for e in trace)


def test_prompted_agent_closes_the_loop_through_text(world) -> None:
    """The channel between simulator and model is language plus gradient.

    A fake ``complete`` exercises the real prompt/parse path: the observation is
    verbalised, a reply is parsed back into weights, and the loop closes.
    """
    seen: list[str] = []
    inner = GradientAgent(len(world.rule_names), lr=0.3)

    def complete(messages: list[dict]) -> str:
        """A stand-in model: reads the verbalised critique, replies with weights.

        It delegates the actual arithmetic to a gradient agent, because what is
        under test is the prompt/parse channel, not the model's judgement.
        """
        seen.append(messages[-1]["content"])
        return json.dumps(inner.propose(world.evaluate(inner.initial())))

    prompted = PromptedAgent(complete, len(world.rule_names))
    trace = run_coupling_loop(world, prompted, steps=12)

    assert trace[-1].objective < trace[0].objective
    assert any("Sensitivity of the objective" in text for text in seen)
    assert any("HIGH_TURNOVER" in text for text in seen)


def test_the_same_loop_drives_the_other_domain(world) -> None:
    """The loop is imported, not reimplemented — check it on dispatch too."""
    from gradientshop.world import DispatchWorld

    generator = torch.Generator().manual_seed(7)
    n = 12
    dispatch = DispatchWorld(
        processing=torch.rand(n, generator=generator, dtype=torch.float64) * 9 + 1,
        weight=torch.rand(n, generator=generator, dtype=torch.float64) * 4 + 1,
        due=torch.rand(n, generator=generator, dtype=torch.float64) * 40 + 10,
    )
    for target in (world, dispatch):
        dimension = len(target.decision_space().names)
        trace = run_coupling_loop(target, GradientAgent(dimension, lr=0.3), steps=30)
        assert trace[-1].objective < trace[0].objective, target.provenance

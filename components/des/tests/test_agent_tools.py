"""The DDES agent tools driven end-to-end through the real ReAct harness, with a
scripted async LLM standing in for a real model (no API key needed).

Proves the tool-calling plumbing: the agent invokes each DDES tool, results feed
back, the world model records the best policy, and the run is audited.
"""

from __future__ import annotations

import asyncio
import json

import torch

from opsworld.agent import Agent, AuditLog
from opsworld.agent.llm import LLMResponse, ToolCall
from gradientshop.agent.tools import build_dispatch_session, build_dispatch_tools


class ScriptedLLM:
    """Async LLM that replays a fixed list of tool calls / final text."""

    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.i = 0

    async def ask_tool(self, messages, tools, tool_choice="auto") -> LLMResponse:
        act = self.actions[self.i]
        self.i += 1
        if act["type"] == "final":
            return LLMResponse(text=act["text"])
        return LLMResponse(tool_calls=[
            ToolCall(id=f"call{self.i}", name=act["name"], arguments=act.get("args", {}))])


def _instance(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, generator=g, dtype=torch.float64) + 0.3
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    d = torch.rand(n, generator=g, dtype=torch.float64) * p.sum().item()
    return p, w, d


def test_react_agent_drives_all_ddes_tools(tmp_path):
    session = build_dispatch_session(*_instance(), tau=0.1, objective="wct")
    n = session["world"].n_features
    theta = [0.05] * n

    actions = [
        {"type": "tool", "name": "describe_scenario"},
        {"type": "tool", "name": "validate_timing_fidelity"},
        {"type": "tool", "name": "evaluate_policy", "args": {"theta": theta}},
        {"type": "tool", "name": "compare_to_reference", "args": {"theta": theta}},
        {"type": "final", "text": "Converged; best policy recorded."},
    ]
    agent = Agent(
        llm=ScriptedLLM(actions),
        tools=build_dispatch_tools(),
        system_prompt="test",
        audit=AuditLog(run_id="pytest", log_dir=tmp_path),
    )

    result = asyncio.run(agent.run("optimise the dispatch policy", session))

    assert result["final"] == "Converged; best policy recorded."
    # evaluate_policy populated the best-so-far record.
    assert "best" in session and session["best"]["theta"] == theta
    # The whole run was audited.
    lines = (tmp_path / "run_pytest.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(x)["event"] for x in lines]
    assert "run_start" in events and "final" in events
    assert events.count("tool_result") == 4


def test_evaluate_policy_rejects_wrong_length_theta(tmp_path):
    session = build_dispatch_session(*_instance(), objective="wct")
    actions = [
        {"type": "tool", "name": "evaluate_policy", "args": {"theta": [1.0, 2.0]}},
        {"type": "final", "text": "done"},
    ]
    agent = Agent(llm=ScriptedLLM(actions), tools=build_dispatch_tools(),
                  system_prompt="test", audit=AuditLog(run_id="pytest2", log_dir=tmp_path))
    asyncio.run(agent.run("go", session))
    # A too-short theta is reported as a tool error, not a crash; no best recorded.
    assert "best" not in session

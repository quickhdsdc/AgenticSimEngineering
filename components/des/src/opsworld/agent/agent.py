"""Async ReAct agent loop -- the harness core (mirrors MoldDesign).

``run(goal)``: assemble system prompt (persona) + history -> ``llm.ask_tool`` ->
execute tool calls on the shared ``session`` blackboard -> feed results back ->
loop until the model emits text only. Every step is audited; high-risk tools pass
through a human-in-the-loop approver.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from .audit import AuditLog
from .llm import LLMClient
from .tool import ToolCollection

PERSONA_DIR = Path(__file__).resolve().parents[1] / "personas"

# HITL approval callback: returns (approved, note). Default auto-approves.
Approver = Callable[[str, dict], Awaitable[tuple[bool, str]]]


async def auto_approve(tool_name: str, args: dict) -> tuple[bool, str]:
    return True, "auto-approved"


def load_persona(name: str) -> str:
    return (PERSONA_DIR / f"{name}.md").read_text(encoding="utf-8")


class Agent:
    def __init__(self, llm: LLMClient, tools: ToolCollection, system_prompt: str,
                 max_steps: int = 12, approver: Approver = auto_approve,
                 on_event: Callable[[str], None] | None = None,
                 audit: AuditLog | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.approver = approver
        self.on_event = on_event or (lambda s: None)
        self.audit = audit or AuditLog()

    async def run(self, goal: str, session: dict | None = None) -> dict:
        session = session if session is not None else {}
        self.audit.record("run_start", goal=goal)
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": goal},
        ]
        for step in range(self.max_steps):
            resp = await self.llm.ask_tool(messages, self.tools.to_params())

            if not resp.tool_calls:
                self.on_event(f"[final] {resp.text}")
                self.audit.record("final", text=resp.text)
                session["final"] = resp.text
                session["audit_log"] = str(self.audit.path)
                return session

            messages.append({
                "role": "assistant", "content": resp.text or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name,
                                  "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in resp.tool_calls
                ],
            })

            for tc in resp.tool_calls:
                tool = self.tools.get(tc.name)
                if tool is not None and tool.hitl_required:
                    approved, note = await self.approver(tc.name, tc.arguments)
                    self.audit.record("hitl", step=step, tool=tc.name,
                                      approved=approved, note=note)
                    if not approved:
                        msg = f"human rejected: {note}"
                        self.on_event(f"[hitl] {tc.name} rejected: {note}")
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "name": tc.name, "content": msg})
                        continue

                self.on_event(f"[tool] {tc.name}({tc.arguments})")
                self.audit.record("tool_call", step=step, tool=tc.name, args=tc.arguments,
                                  deterministic=(tool.deterministic if tool else None))
                result = await self.tools.execute(tc.name, tc.arguments, session)
                text = str(result)
                self.on_event(f"   -> {text}")
                self.audit.record("tool_result", step=step, tool=tc.name,
                                  ok=result.error is None, output=text)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "name": tc.name, "content": text})

        self.audit.record("stop", reason="max_steps")
        session["final"] = "reached step limit without converging."
        session["audit_log"] = str(self.audit.path)
        return session

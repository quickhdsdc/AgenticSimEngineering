"""Agent 循环(async ReAct)—— harness 的核。

think(): 组装 system prompt(人设)+ 历史 → llm.ask_tool(messages, tools)
act():  执行工具(hitl_required 前挂人审)→ ToolResult 回灌 → 循环,直到只出文本。
全程写审计日志(P3);HITL 关卡(P5)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from .audit import AuditLog
from .llm import LLMClient
from .tool import ToolCollection

PERSONA_DIR = Path(__file__).resolve().parents[3] / "personas"

# HITL 审批回调:返回 (approved, note)。默认自动批(Phase 1);Phase 2 接真人/UI。
Approver = Callable[[str, dict], Awaitable[tuple[bool, str]]]


async def auto_approve(tool_name: str, args: dict) -> tuple[bool, str]:
    return True, "auto-approved (Phase1)"


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
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in resp.tool_calls
                ],
            })

            for tc in resp.tool_calls:
                tool = self.tools.get(tc.name)
                # HITL:高风险工具执行前挂人审(P5)
                if tool is not None and tool.hitl_required:
                    approved, note = await self.approver(tc.name, tc.arguments)
                    self.audit.record("hitl", step=step, tool=tc.name, approved=approved, note=note)
                    if not approved:
                        result_str = f"人工驳回:{note}"
                        self.on_event(f"[hitl] {tc.name} 被驳回:{note}")
                        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result_str})
                        continue

                self.on_event(f"[tool] {tc.name}({tc.arguments})")
                self.audit.record("tool_call", step=step, tool=tc.name, args=tc.arguments,
                                  deterministic=(tool.deterministic if tool else None))
                result = await self.tools.execute(tc.name, tc.arguments, session)
                text = str(result)
                self.on_event(f"   → {text}")
                self.audit.record("tool_result", step=step, tool=tc.name,
                                  ok=result.error is None, output=text)
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": text})

        self.audit.record("stop", reason="max_steps")
        session["final"] = "达到步数上限,未收敛。"
        session["audit_log"] = str(self.audit.path)
        return session

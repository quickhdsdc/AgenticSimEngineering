"""Shared LLM agent harness — OpenManus-style async ReAct (mirrors MoldDesign).

Domain-agnostic by construction: it knows about messages, tools and audit
logs, never about dispatching rules or load carriers. A domain plugs in by
exposing its world model as tools (see :mod:`gradientshop.agent.tools`), so
every domain in the framework gets an LLM agent without a second harness.
"""

from __future__ import annotations

from .agent import Agent, auto_approve, load_persona
from .audit import AuditLog
from .config import LLMSettings, load_llm
from .llm import LLMClient, LLMResponse, OpenAICompatLLM, ToolCall, build_llm
from .coupling import GradientAgent, PolicyAgent, PromptedAgent, run_coupling_loop
from .tool import BaseTool, ToolCollection, ToolResult

__all__ = [
    "Agent", "auto_approve", "load_persona",
    "AuditLog",
    "LLMSettings", "load_llm",
    "LLMClient", "LLMResponse", "OpenAICompatLLM", "ToolCall", "build_llm",
    "BaseTool", "ToolCollection", "ToolResult",
    "GradientAgent", "PolicyAgent", "PromptedAgent", "run_coupling_loop",
]

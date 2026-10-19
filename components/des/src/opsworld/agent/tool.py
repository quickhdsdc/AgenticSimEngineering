"""Tool base classes (OpenManus-aligned, mirrors the MoldDesign harness).

* ``BaseTool``   -- name/description/parameters + async ``execute(session, **kw)``
  returning a ``ToolResult``; ``to_param()`` emits the OpenAI function schema.
* ``ToolResult`` -- output/error envelope.
* ``ToolCollection`` -- registry passed to the model and used to dispatch calls.

Enterprise metadata (``deterministic`` / ``hitl_required`` / ``writes``) lets the
audit log and any human-in-the-loop gate reason about a tool before it runs. In
GradientShop the important flag is ``deterministic``: the DDES replay tools are
pure numerics, distinguishing them from tools that themselves call an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any = None
    error: str | None = None

    def __bool__(self) -> bool:
        return self.output is not None or self.error is not None

    def __str__(self) -> str:
        return f"ERROR: {self.error}" if self.error else str(self.output)


class BaseTool(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict | None = None
    deterministic: bool = True        # False = the tool internally calls an LLM
    hitl_required: bool = False       # True = needs human approval before running
    writes: list[str] = Field(default_factory=list)  # declared side effects

    async def __call__(self, session: dict, **kwargs) -> ToolResult:
        return await self.execute(session, **kwargs)

    @abstractmethod
    async def execute(self, session: dict, **kwargs) -> ToolResult:
        """Run the tool. ``session`` is the shared blackboard across tools
        (carries big objects like the world model and the current scenario)."""

    def to_param(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ToolCollection:
    def __init__(self, *tools: BaseTool) -> None:
        self.tools: tuple[BaseTool, ...] = tools
        self.tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> list[dict]:
        return [t.to_param() for t in self.tools]

    def get(self, name: str) -> BaseTool | None:
        return self.tool_map.get(name)

    async def execute(self, name: str, args: dict, session: dict) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolResult(error=f"unknown tool {name}")
        try:
            return await tool(session, **(args or {}))
        except Exception as e:  # feed the error back to the model so it can recover
            return ToolResult(error=f"{type(e).__name__}: {e}")

    def add(self, *tools: BaseTool) -> "ToolCollection":
        self.tools += tools
        for t in tools:
            self.tool_map[t.name] = t
        return self

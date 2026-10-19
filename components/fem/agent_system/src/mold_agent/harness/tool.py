"""工具基类 —— 对齐 OpenManus 约定,mold-agent 技能可与 EmbodiedTwin backend 互相 drop。

- BaseTool: name/description/parameters + async execute() -> ToolResult, to_param() 出 OpenAI function 格式。
- ToolResult: output/error(与 OpenManus 同构)。
- ToolCollection: 管理多工具,to_params() 喂给模型,execute(name, args) 分发。
- 额外(企业级):deterministic / hitl_required / writes 元数据,供审计与权限用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    output: Any = None
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self) -> bool:
        return self.output is not None or self.error is not None

    def __str__(self) -> str:
        return f"ERROR: {self.error}" if self.error else str(self.output)


class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: dict | None = None
    # 企业级元数据(P1/P3/P5)
    deterministic: bool = True        # True=纯确定性(判定/计算);False=内部调 LLM
    hitl_required: bool = False       # True=执行前需人审(高风险动作)
    writes: list[str] = Field(default_factory=list)  # 声明副作用(产出文件/外部写)

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, session: dict, **kwargs) -> ToolResult:
        return await self.execute(session, **kwargs)

    @abstractmethod
    async def execute(self, session: dict, **kwargs) -> ToolResult:
        """执行工具。session 为跨工具共享的黑板(传大对象 SimResult/ProjectSpec)。"""

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
            return ToolResult(error=f"未知工具 {name}")
        try:
            return await tool(session, **(args or {}))
        except Exception as e:  # 错误回灌给模型,让其纠偏
            return ToolResult(error=f"{type(e).__name__}: {e}")

    def add(self, *tools: BaseTool) -> "ToolCollection":
        self.tools += tools
        for t in tools:
            self.tool_map[t.name] = t
        return self

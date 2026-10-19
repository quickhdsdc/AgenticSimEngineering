"""模型网关 —— 模型无关的 LLMClient(async ask_tool,对齐 OpenManus)。

只连真实端点(OpenAI 兼容 / Azure);无离线/仿真档。未配置好就报错,不静默造假。
生产: 自托管 vLLM Qwen(MOLD_CONFIG=config/config.vllm.toml)、DeepSeek、Azure 等。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .config import LLMSettings, load_llm


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    async def ask_tool(self, messages: list[dict], tools: list[dict],
                       tool_choice: str = "auto") -> LLMResponse: ...


def _is_reasoning_model(model: str) -> bool:
    """推理模型(gpt-5 / o1 / o3 / o4…)在 Azure/OpenAI 上要求 max_completion_tokens
    且不接受自定义 temperature。对齐 EmbodiedTwin backend 的 REASONING_MODELS 处理。"""
    m = (model or "").lower()
    return any(k in m for k in ("gpt-5", "o1", "o3", "o4-mini", "o1-mini", "o3-mini"))


class OpenAICompatLLM:
    """任意 OpenAI 兼容端点(async)。api_type = openai / azure。"""

    def __init__(self, cfg: LLMSettings) -> None:
        self.cfg = cfg
        if cfg.api_type == "azure":
            from openai import AsyncAzureOpenAI
            self._client = AsyncAzureOpenAI(
                api_key=cfg.api_key, api_version=cfg.api_version or "2024-10-21",
                azure_endpoint=cfg.base_url)
        else:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY")

    async def ask_tool(self, messages, tools, tool_choice="auto") -> LLMResponse:
        params: dict = {
            "model": self.cfg.model, "messages": messages,
            "tools": tools or None, "tool_choice": tool_choice if tools else None,
        }
        if _is_reasoning_model(self.cfg.model):
            params["max_completion_tokens"] = self.cfg.max_tokens   # 推理模型:不发 temperature
        else:
            params["max_tokens"] = self.cfg.max_tokens
            params["temperature"] = self.cfg.temperature
        resp = await self._client.chat.completions.create(**params)
        msg = resp.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(text=msg.content, tool_calls=calls)


def build_llm(profile: str | None = None) -> LLMClient:
    """按 config.toml profile 装配真实 LLM 客户端。"""
    return OpenAICompatLLM(load_llm(profile))

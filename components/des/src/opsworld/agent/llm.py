"""Model gateway -- a model-agnostic async ``LLMClient`` (mirrors MoldDesign).

Connects to real OpenAI-compatible / Azure endpoints only; no offline/mock
backend. Reasoning models (gpt-5 / o1 / o3 / o4-mini) require
``max_completion_tokens`` and reject a custom temperature, handled here.
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
    m = (model or "").lower()
    return any(k in m for k in ("gpt-5", "o1", "o3", "o4-mini", "o1-mini", "o3-mini"))


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint (async). ``api_type`` = openai | azure."""

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
            params["max_completion_tokens"] = self.cfg.max_tokens
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
    """Assemble a real LLM client from the selected config profile."""
    return OpenAICompatLLM(load_llm(profile))

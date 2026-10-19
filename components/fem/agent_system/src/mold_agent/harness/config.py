"""配置加载 —— 读 config/config.toml 的 LLM profiles(对齐 OpenManus 语义)。

- [llm]        = 默认档;[llm.<name>] = 具名档(继承默认并覆盖)。
- 选择: 环境变量 MOLD_LLM_PROFILE(默认用 [llm])。
- 配置文件: 环境变量 MOLD_CONFIG(默认 config/config.toml;自托管 vLLM 指向 config/config.vllm.toml)。
- 无任何离线/仿真档:未配置好真实端点就直接报错(不静默造假)。
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.toml"


class LLMSettings(BaseModel):
    api_type: str = "openai"          # openai | azure
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_version: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0


def _config_path() -> Path:
    p = os.environ.get("MOLD_CONFIG")
    return Path(p) if p else DEFAULT_CONFIG


def _resolve_secret(val) -> str:
    if isinstance(val, str) and val.startswith("env:"):
        return os.environ.get(val[4:], "")
    return val


def _raw_llm() -> dict:
    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(f"未找到配置 {path}。请配置真实 LLM 端点(见 config/config.toml)。")
    with path.open("rb") as f:
        return tomllib.load(f).get("llm", {})


def default_profile() -> str | None:
    return os.environ.get("MOLD_LLM_PROFILE")  # None → 用 [llm] 默认档


def load_llm(profile: str | None = None) -> LLMSettings:
    profile = profile or default_profile()
    raw = _raw_llm()
    base = {k: v for k, v in raw.items() if not isinstance(v, dict)}   # [llm] 标量 = 默认档
    if profile:
        named = raw.get(profile)
        if not isinstance(named, dict):
            avail = [k for k, v in raw.items() if isinstance(v, dict)]
            raise KeyError(f"未知 LLM profile '{profile}'。可用: {avail} 或留空用默认档。")
        data = {**base, **named}
    else:
        data = dict(base)
    if not data.get("model") or not data.get("base_url"):
        raise ValueError(f"LLM 配置不完整(profile={profile or 'default'}):需 model 与 base_url。")
    data["api_type"] = str(data.get("api_type", "openai")).lower()   # 'Openai' → 'openai'
    if "api_key" in data:
        data["api_key"] = _resolve_secret(data["api_key"])
    return LLMSettings(**{k: v for k, v in data.items() if k in LLMSettings.model_fields})

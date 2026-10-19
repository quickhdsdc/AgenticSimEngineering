"""LLM configuration loader (TOML profiles, mirrors the MoldDesign harness).

* ``[llm]``        = default profile; ``[llm.<name>]`` = named profile (inherits
  the default and overrides it).
* Profile selected by env var ``OPSWORLD_LLM_PROFILE`` (unset -> the ``[llm]``
  default). ``GS_LLM_PROFILE`` is still honoured, from before the harness was
  shared between domains.
* Config file located by env var ``OPSWORLD_CONFIG`` (or ``GS_CONFIG``);
  default ``opsworld/agent/config.toml``.
* Secrets may be written ``"env:VAR"`` to read from the environment (never commit
  keys). There is no offline/mock backend: an unconfigured endpoint is an error,
  not a silent fake.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = AGENT_DIR / "config.toml"


class LLMSettings(BaseModel):
    api_type: str = "openai"          # openai | azure
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_version: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0


def _config_path() -> Path:
    p = os.environ.get("OPSWORLD_CONFIG") or os.environ.get("GS_CONFIG")
    return Path(p) if p else DEFAULT_CONFIG


def _resolve_secret(val):
    if isinstance(val, str) and val.startswith("env:"):
        return os.environ.get(val[4:], "")
    return val


def _raw_llm() -> dict:
    path = _config_path()
    if not path.exists():
        raise FileNotFoundError(
            f"config not found at {path}. Copy {DEFAULT_CONFIG.name}.example "
            f"to {path} and point it at a real LLM endpoint "
            f"(or set OPSWORLD_CONFIG).")
    with path.open("rb") as f:
        return tomllib.load(f).get("llm", {})


def default_profile() -> str | None:
    # None -> use the [llm] default profile.
    return (os.environ.get("OPSWORLD_LLM_PROFILE")
            or os.environ.get("GS_LLM_PROFILE"))


def load_llm(profile: str | None = None) -> LLMSettings:
    profile = profile or default_profile()
    raw = _raw_llm()
    base = {k: v for k, v in raw.items() if not isinstance(v, dict)}  # scalar [llm] = default
    if profile:
        named = raw.get(profile)
        if not isinstance(named, dict):
            avail = [k for k, v in raw.items() if isinstance(v, dict)]
            raise KeyError(f"unknown LLM profile {profile!r}; available: {avail} "
                           f"or leave unset for the default.")
        data = {**base, **named}
    else:
        data = dict(base)
    if not data.get("model") or not data.get("base_url"):
        raise ValueError(f"incomplete LLM config (profile={profile or 'default'}): "
                         f"need both 'model' and 'base_url'.")
    data["api_type"] = str(data.get("api_type", "openai")).lower()
    if "api_key" in data:
        data["api_key"] = _resolve_secret(data["api_key"])
    return LLMSettings(**{k: v for k, v in data.items() if k in LLMSettings.model_fields})

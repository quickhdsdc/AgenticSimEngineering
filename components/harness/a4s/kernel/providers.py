"""Model access behind one interface.

The deployment already spans Azure OpenAI, a local vLLM serving the Qwen
family, and Anthropic models for comparison. The differences that matter are
narrow -- how tool calls are shaped, how errors report themselves -- so the
interface is narrow too.

Three backends, for two different reasons. The plain OpenAI path covers vLLM,
OpenRouter and OpenAI itself. Azure needs its own because it is *not*
OpenAI-compatible at the URL: a request goes to
``/openai/deployments/{deployment}/chat/completions?api-version=...`` and
``model`` names a deployment rather than a model, so pointing the ordinary
client at ``AZURE_OPENAI_ENDPOINT`` produces a 404, not a completion.
Anthropic needs its own because its tool-call and system-prompt shapes
genuinely differ.

Constrained decoding is a first-class parameter rather than a wrapper. The
whole point of the AAS-schema-constrained work is that the grammar comes
from live topology, so it has to reach the request, and vLLM takes it as
``extra_body``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Completion:
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0


DEFAULT_AZURE_API_VERSION = "2024-10-21"

# Which variables each provider reads, in precedence order. Kept as data so
# `a4s doctor` can report exactly what a profile looked for without the two
# lists drifting apart.
ENV_KEYS: Dict[str, Dict[str, tuple]] = {
    "azure": {
        "api_key": ("A4S_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"),
        "base_url": ("A4S_BASE_URL", "AZURE_OPENAI_ENDPOINT"),
        "model": ("A4S_MODEL", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME"),
        "api_version": ("A4S_AZURE_API_VERSION", "AZURE_OPENAI_API_VERSION"),
    },
    "anthropic": {
        "api_key": ("A4S_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        "base_url": ("A4S_BASE_URL", "ANTHROPIC_BASE_URL"),
        "model": ("A4S_MODEL",),
    },
    "openai": {
        "api_key": ("A4S_API_KEY", "OPENAI_API_KEY"),
        "base_url": ("A4S_BASE_URL", "OPENAI_BASE_URL"),
        "model": ("A4S_MODEL",),
    },
}

DEFAULT_MODEL = {
    "azure": "gpt-4o",           # a deployment name, not a model name
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}


def resolve_provider() -> str:
    """Pick a provider from what is configured, when nothing says which.

    Explicit ``A4S_PROVIDER`` always wins. Otherwise Azure is preferred over
    Anthropic because an Azure endpoint is deployment-specific: if one is
    configured on this host, it is the one someone meant.
    """
    from .config import first

    explicit = first("A4S_PROVIDER")
    if explicit:
        return explicit.lower()
    if first(*ENV_KEYS["azure"]["base_url"]):
        return "azure"
    if first(*ENV_KEYS["anthropic"]["api_key"][1:]):
        return "anthropic"
    return "openai"


def json_schema(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAI Structured Outputs `response_format` for one schema."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def ask_json(
    provider: Any,
    messages: List[Dict[str, Any]],
    config: ModelConfig,
    *,
    schema: Optional[Dict[str, Any]] = None,
    name: str = "result",
) -> Completion:
    """Ask for JSON, constrained by a schema where that is available.

    Returns the whole `Completion`, not its text: `finish_reason` is the only
    thing that separates "the page held nothing" from "the answer was cut off",
    and the two need different fixes from whoever reads the error.

    **The schema is an improvement on the parse, not a replacement for it.**
    Every caller still reads the reply with its tolerant parser, and that is
    deliberate: §3.16 lost five pages of a datasheet to replies that were empty
    rather than malformed, and a schema does not help with an empty reply. What
    it removes is the other failure -- a fenced or prefaced object, a missing
    field, a number sent as a string -- which the parsers currently absorb one
    special case at a time.

    Degrades in three steps, because a provider that cannot do this must not
    turn into an error:

    * `TypeError` -- a provider or a test double written before the parameter
      existed. Retry without it.
    * a 400 naming `response_format` or `json_schema` -- a real endpoint that
      does not implement it, which includes most vLLM builds and older Azure
      API versions. Retry without it.
    * anything else is a real failure and is raised.
    """
    if schema is not None:
        try:
            return provider.complete(
                messages, [], config, response_format=json_schema(name, schema)
            )
        except TypeError:
            pass
        except Exception as exc:  # noqa: BLE001 - narrowed by the message test
            text = str(exc).lower()
            if not any(
                token in text
                for token in ("response_format", "json_schema", "structured output")
            ):
                raise
    return provider.complete(messages, [], config)


@dataclass
class ModelConfig:
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    provider: str = "openai"
    temperature: float = 0.0
    max_tokens: int = 4096
    context_window: int = 128_000
    # Azure routes by deployment and pins a dated API surface; ignored elsewhere.
    api_version: Optional[str] = None
    # Passed through to vLLM as extra_body. Carries the guided-decoding
    # grammar when a run is in a constrained mode.
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "ModelConfig":
        from .config import first

        name = (provider or resolve_provider()).lower()
        keys = ENV_KEYS.get(name, ENV_KEYS["openai"])
        return cls(
            model=first(*keys["model"]) or DEFAULT_MODEL.get(name, "gpt-4o-mini"),
            base_url=first(*keys["base_url"]),
            api_key=first(*keys["api_key"]),
            provider=name,
            temperature=float(os.getenv("A4S_TEMPERATURE", "0")),
            max_tokens=int(os.getenv("A4S_MAX_TOKENS", "4096")),
            context_window=int(os.getenv("A4S_CONTEXT_WINDOW", "128000")),
            api_version=(
                first(*keys["api_version"]) or DEFAULT_AZURE_API_VERSION
                if name == "azure"
                else None
            ),
        )

    def missing(self) -> List[str]:
        """Variables this profile needs and did not find. Empty means runnable."""
        keys = ENV_KEYS.get(self.provider, ENV_KEYS["openai"])
        gaps: List[str] = []
        if not self.api_key and self.provider != "openai":
            gaps.append(" or ".join(keys["api_key"]))
        if self.provider == "azure" and not self.base_url:
            gaps.append(" or ".join(keys["base_url"]))
        return gaps


class TransientModelError(RuntimeError):
    """Rate limit, timeout, or a 5xx -- worth retrying as-is."""


class ContextOverflowError(RuntimeError):
    """The request exceeded the window -- retrying unchanged cannot help."""


class Provider(Protocol):
    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        config: ModelConfig,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Completion:
        ...


# Reasoning-family names, as a first guess only. On Azure a deployment may be
# called anything at all, so this saves a round trip when the name is
# conventional and is corrected by `Quirks.learn` when it is not.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class Quirks:
    """Per-endpoint parameter differences, learned from the endpoint's own 400s.

    The reasoning models renamed ``max_tokens`` to ``max_completion_tokens``
    and accept only the default temperature. Hard-coding a model list ages
    badly and does not survive Azure's arbitrary deployment names, so the
    first rejection teaches the client and every later call is correct.
    """

    def __init__(self, model: str) -> None:
        reasoning = any(model.lower().startswith(p) for p in _REASONING_PREFIXES)
        self.max_completion_tokens = reasoning
        self.fixed_temperature = reasoning

    def kwargs(self, config: ModelConfig) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.max_completion_tokens:
            out["max_completion_tokens"] = config.max_tokens
        else:
            out["max_tokens"] = config.max_tokens
        if not self.fixed_temperature:
            out["temperature"] = config.temperature
        return out

    def learn(self, message: str) -> bool:
        """Adjust from one rejection. Returns whether anything changed."""
        lowered = message.lower()
        if "max_completion_tokens" in lowered and not self.max_completion_tokens:
            self.max_completion_tokens = True
            return True
        if "temperature" in lowered and "support" in lowered and not self.fixed_temperature:
            self.fixed_temperature = True
            return True
        return False


class OpenAICompatible:
    """OpenAI, vLLM, OpenRouter -- anything speaking ``/v1/chat/completions``.

    Azure is deliberately not in that list; see :class:`AzureOpenAICompatible`.
    """

    def __init__(self, config: ModelConfig) -> None:
        from openai import OpenAI

        self._quirks = Quirks(config.model)
        self._client = OpenAI(
            api_key=config.api_key or "not-needed-for-local",
            base_url=config.base_url,
        )

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        config: ModelConfig,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Completion:
        import openai

        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=config.model,
                    messages=_for_openai(messages),
                    tools=tools or None,
                    extra_body=config.extra_body or None,
                    **({"response_format": response_format} if response_format else {}),
                    **self._quirks.kwargs(config),
                )
                break
            except openai.BadRequestError as exc:
                # The reasoning families reject two parameters the chat models
                # require. Which family a name belongs to is not knowable from
                # the name -- an Azure deployment can be called anything -- so
                # the endpoint is asked, once, and the answer is remembered for
                # the life of the client.
                attempt += 1
                if attempt <= 2 and self._quirks.learn(str(exc)):
                    continue
                if "context" in str(exc).lower() or "maximum" in str(exc).lower():
                    raise ContextOverflowError(str(exc)) from exc
                raise
            except (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError) as exc:
                raise TransientModelError(str(exc)) from exc

        choice = response.choices[0]
        calls: List[ToolCall] = []
        for raw in choice.message.tool_calls or []:
            calls.append(
                ToolCall(
                    id=raw.id,
                    name=raw.function.name,
                    arguments=_loads(raw.function.arguments),
                )
            )
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            tool_calls=calls,
            finish_reason=choice.finish_reason or "stop",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class AzureOpenAICompatible(OpenAICompatible):
    """Azure OpenAI: same request body, different routing.

    Everything after the client construction is identical to the OpenAI path,
    so only the client is replaced. ``config.model`` is the *deployment* name
    here -- Azure resolves the model from the deployment, and a model id in
    that slot returns a 404 with a message that does not say so.
    """

    def __init__(self, config: ModelConfig) -> None:
        from openai import AzureOpenAI

        # Azure v1 uses the standard client, with the deployment in model.
        # Retain Azure credential resolution and the legacy dated route.
        if config.base_url and config.base_url.rstrip('/').endswith('/openai/v1'):
            super().__init__(config)
            return

        if not config.base_url:
            raise RuntimeError(
                "Azure needs an endpoint. Set AZURE_OPENAI_ENDPOINT "
                "(e.g. https://<resource>.openai.azure.com)."
            )
        self._quirks = Quirks(config.model)
        self._client = AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.base_url,
            api_version=config.api_version or DEFAULT_AZURE_API_VERSION,
        )


class Anthropic:
    """Claude's Messages API -- a different tool-call and system-prompt shape."""

    def __init__(self, config: ModelConfig) -> None:
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=config.api_key,
            **({"base_url": config.base_url} if config.base_url else {}),
        )

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        config: ModelConfig,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Completion:
        # Accepted and ignored: the Messages API has no `response_format`, and
        # its equivalent is a tool definition, which this path does not build.
        # `ask_json` falls back to the prompt-stated shape, which is what every
        # caller parsed before schemas existed.
        del response_format
        import anthropic

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        converted = _to_anthropic(m for m in messages if m["role"] != "system")
        try:
            response = self._client.messages.create(
                model=config.model,
                system=system or None,
                messages=converted,
                tools=[_tool_to_anthropic(t) for t in tools] or None,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        except anthropic.RateLimitError as exc:
            raise TransientModelError(str(exc)) from exc
        except anthropic.BadRequestError as exc:
            if "token" in str(exc).lower():
                raise ContextOverflowError(str(exc)) from exc
            raise

        text_parts: List[str] = []
        calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return Completion(
            text="".join(text_parts),
            tool_calls=calls,
            finish_reason=response.stop_reason or "stop",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def build(config: ModelConfig) -> Provider:
    gaps = config.missing()
    if gaps:
        raise RuntimeError(
            f"provider {config.provider!r} is not configured; set "
            + ", ".join(gaps)
            + ". Run 'a4s doctor' to see what was found."
        )
    if config.provider == "anthropic":
        try:
            return Anthropic(config)
        except ImportError as exc:  # pragma: no cover - install-time path
            raise RuntimeError(
                "the anthropic provider needs its SDK: pip install 'a4s-harness[anthropic]'"
            ) from exc
    if config.provider == "azure":
        return AzureOpenAICompatible(config)
    return OpenAICompatible(config)


# ---------------------------------------------------------------------------

def _loads(raw: str) -> Dict[str, Any]:
    """Parse tool arguments, tolerating the shapes small models emit.

    A local 4B model returns malformed JSON often enough that failing the
    turn on it wastes a step the budget cannot spare. Returning the raw text
    lets the tool report a specific error the model can correct from.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def _image_blocks(content: Any) -> List[Dict[str, Any]]:
    """OpenAI-shaped content parts, in Anthropic's block shape.

    The loop writes attachments once, in the format the default provider takes.
    Converting here rather than asking the loop to branch keeps the knowledge
    of each API's shape inside its own adapter, which is where it stays correct.
    """
    blocks: List[Dict[str, Any]] = []
    for part in content or []:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            _, _, data = url.partition("base64,")
            if data:
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png",
                               "data": data},
                })
    return blocks


def _for_openai(messages: Any) -> List[Dict[str, Any]]:
    """Drop the loop's private keys before the wire.

    `_attachments` marks the message the Anthropic adapter folds into a
    tool_result. This API has no use for it and rejects unknown fields, so it
    is stripped here rather than never being written -- the marker has to
    survive in `self.messages`, because the conversation is replayed to
    whichever provider is configured on the *next* call, not only this one.
    """
    cleaned: List[Dict[str, Any]] = []
    for message in messages:
        if any(key.startswith("_") for key in message):
            cleaned.append({k: v for k, v in message.items() if not k.startswith("_")})
        else:
            cleaned.append(message)
    return cleaned


def _to_anthropic(messages: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pending = list(messages)
    for index, message in enumerate(pending):
        role = message["role"]
        if message.get("_attachments"):
            # Already folded into the tool_result below it.
            continue
        if role == "tool":
            # Anthropic takes images *inside* a tool_result, which OpenAI does
            # not -- so the attachment message the loop appended after this one
            # is merged back in here. Sending it as its own user turn would put
            # a bare message between a tool_use and its result, which the API
            # rejects.
            body: Any = message["content"]
            following = pending[index + 1] if index + 1 < len(pending) else None
            if following is not None and following.get("_attachments"):
                body = [{"type": "text", "text": message["content"]}]
                body.extend(_image_blocks(following.get("content")))
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message["tool_call_id"],
                            "content": body,
                        }
                    ],
                }
            )
        elif role == "assistant" and message.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": _loads(call["function"]["arguments"]),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            content = message.get("content")
            if isinstance(content, list):
                out.append({"role": role, "content": _image_blocks(content)})
            else:
                out.append({"role": role, "content": content or ""})
    return out


def _tool_to_anthropic(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool["function"]
    return {"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]}

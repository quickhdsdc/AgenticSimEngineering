"""Provider resolution and the parameter quirks the endpoints impose.

None of these tests reach the network. What they pin down is the part that
silently misroutes: which variable a profile reads, and which request shape an
endpoint will accept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from a4s.kernel import config as cfg
from a4s.kernel.providers import (
    DEFAULT_AZURE_API_VERSION,
    ModelConfig,
    Quirks,
    build,
    resolve_provider,
)

AZURE_VARS = (
    "A4S_PROVIDER", "A4S_API_KEY", "A4S_BASE_URL", "A4S_MODEL",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION",
    "A4S_AZURE_API_VERSION", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY",
    "ANTHROPIC_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own keys must not decide what these tests assert."""
    for name in AZURE_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# which provider, and from which variables
# ---------------------------------------------------------------------------

def test_explicit_provider_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com")
    monkeypatch.setenv("A4S_PROVIDER", "anthropic")
    assert resolve_provider() == "anthropic"


def test_azure_is_inferred_from_its_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Azure endpoint is resource-specific; if one is configured it was meant."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-x")
    assert resolve_provider() == "azure"


def test_anthropic_is_inferred_from_a_key_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-x")
    assert resolve_provider() == "anthropic"


def test_openai_is_the_fallback() -> None:
    assert resolve_provider() == "openai"


def test_anthropic_accepts_either_key_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLAUDE_API_KEY is what the deployment already uses; ANTHROPIC_API_KEY is the SDK's."""
    monkeypatch.setenv("CLAUDE_API_KEY", "from-claude-var")
    assert ModelConfig.from_env("anthropic").api_key == "from-claude-var"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-anthropic-var")
    assert ModelConfig.from_env("anthropic").api_key == "from-anthropic-var"


def test_a4s_api_key_overrides_the_provider_specific_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-var")
    monkeypatch.setenv("A4S_API_KEY", "override")
    assert ModelConfig.from_env("azure").api_key == "override"


def test_azure_reads_a_deployment_not_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-gpt5-deployment")
    assert ModelConfig.from_env("azure").model == "my-gpt5-deployment"


def test_azure_api_version_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ModelConfig.from_env("azure").api_version == DEFAULT_AZURE_API_VERSION
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    assert ModelConfig.from_env("azure").api_version == "2025-01-01-preview"


def test_missing_reports_what_to_set() -> None:
    gaps = ModelConfig.from_env("azure").missing()
    assert any("AZURE_OPENAI_API_KEY" in g for g in gaps)
    assert any("AZURE_OPENAI_ENDPOINT" in g for g in gaps)


def test_build_refuses_an_unconfigured_provider_before_the_network() -> None:
    with pytest.raises(RuntimeError, match="not configured"):
        build(ModelConfig.from_env("anthropic"))


def test_a_configured_provider_reports_no_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert ModelConfig.from_env("azure").missing() == []


# ---------------------------------------------------------------------------
# quirks
# ---------------------------------------------------------------------------

def test_chat_models_send_max_tokens() -> None:
    kwargs = Quirks("gpt-4o").kwargs(ModelConfig(model="gpt-4o", temperature=0.0))
    assert kwargs == {"max_tokens": 4096, "temperature": 0.0}


@pytest.mark.parametrize("model", ["gpt-5-2025-08-07", "o1-preview", "o3-mini", "o4-mini"])
def test_reasoning_models_are_guessed_from_the_name(model: str) -> None:
    kwargs = Quirks(model).kwargs(ModelConfig(model=model))
    assert "max_completion_tokens" in kwargs
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_an_arbitrary_deployment_name_is_learned_from_the_rejection() -> None:
    """Azure deployments can be called anything, so the name is not evidence."""
    quirks = Quirks("prod-eu-west")
    assert "max_tokens" in quirks.kwargs(ModelConfig(model="prod-eu-west"))

    assert quirks.learn(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    assert "max_completion_tokens" in quirks.kwargs(ModelConfig(model="prod-eu-west"))

    assert quirks.learn(
        "Unsupported value: 'temperature' does not support 0 with this model."
    )
    assert "temperature" not in quirks.kwargs(ModelConfig(model="prod-eu-west"))


def test_learning_the_same_thing_twice_reports_no_change() -> None:
    """Otherwise the retry loop would spin on an error it cannot actually fix."""
    quirks = Quirks("gpt-4o")
    message = "Use 'max_completion_tokens' instead."
    assert quirks.learn(message)
    assert not quirks.learn(message)


def test_an_unrelated_bad_request_teaches_nothing() -> None:
    assert not Quirks("gpt-4o").learn("Invalid value for 'tools': expected an array")


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

def test_env_file_fills_gaps_but_never_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's real environment is the operator's decision; the file is a default."""
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "export AZURE_OPENAI_ENDPOINT=https://from-file.openai.azure.com\n"
        'AZURE_OPENAI_API_KEY="quoted-key"\n'
        "CLAUDE_API_KEY='single-quoted'\n"
        "MALFORMED_LINE_WITHOUT_EQUALS\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://from-shell.openai.azure.com")

    assert cfg.load_env(str(env)) == env
    import os

    assert os.environ["AZURE_OPENAI_ENDPOINT"] == "https://from-shell.openai.azure.com"
    assert os.environ["AZURE_OPENAI_API_KEY"] == "quoted-key"
    assert os.environ["CLAUDE_API_KEY"] == "single-quoted"


def test_load_env_is_silent_when_there_is_no_file(tmp_path: Path) -> None:
    assert cfg.load_env(str(tmp_path / "absent.env")) is None


def test_describe_reports_presence_and_length_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key must be diagnosable without ever being printed."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "abcdef")
    rows = cfg.describe("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY")
    assert rows == [
        {"name": "AZURE_OPENAI_API_KEY", "set": True, "length": 6},
        {"name": "OPENAI_API_KEY", "set": False, "length": 0},
    ]
    assert "abcdef" not in cfg.redact("abcdef")


def test_first_skips_blank_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A4S_API_KEY", "   ")
    monkeypatch.setenv("OPENAI_API_KEY", "real")
    assert cfg.first("A4S_API_KEY", "OPENAI_API_KEY") == "real"

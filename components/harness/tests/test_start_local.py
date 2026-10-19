"""Launcher checks are offline: never start services, Docker, browsers or models."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/start-local.py"
    spec = importlib.util.spec_from_file_location("start_local_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("url,allowed", [
    ("http://localhost:8081", True), ("http://127.0.0.1:8081/", True),
    ("http://example.com:8081", False), ("http://localhost:9999", False),
    ("http://user:pass@localhost:8081", False), ("http://localhost:8081/custom", False),
    ("http://localhost:8081?target=other", False),
])
def test_only_exact_local_deployment_can_be_started(launcher, url, allowed):
    assert launcher.local_endpoint(url, 8081) is allowed


def test_readonly_check_never_starts_anything(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "configure", lambda: None)
    monkeypatch.setenv("A4S_AAS_ENDPOINT", "http://localhost:8081")
    monkeypatch.setattr(launcher, "aas_ready", lambda e: True)
    monkeypatch.setattr(launcher, "mes_ready", lambda e: True)
    monkeypatch.setattr(launcher, "backend_ready", lambda e: True)
    monkeypatch.setattr(launcher, "frontend_ready", lambda: True)
    for name in ("launch", "start_compose"):
        monkeypatch.setattr(launcher, name, lambda *a, **k: pytest.fail("Read-only check mutated state"))
    assert launcher.main(["--check"]) == 0
    monkeypatch.setattr(launcher, "backend_ready", lambda e: False)
    assert launcher.main(["--check"]) == 1


def test_backend_probe_requires_expected_repository(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "fetch", lambda url: SimpleNamespace(
        status_code=200, json=lambda: {"ok": True, "repository": "http://other:8081"}))
    assert not launcher.backend_ready("http://localhost:8081")


def test_dotenv_is_loaded_without_overriding_process_or_changing_file(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("A4S_ENV_FILE", raising=False)
    monkeypatch.setenv("A4S_AAS_ENDPOINT", "http://explicit:8081")
    monkeypatch.delenv("A4S_LAUNCH_TEST", raising=False)
    monkeypatch.setenv("PATH", launcher.os.environ.get("PATH", ""))
    monkeypatch.setenv("PYTHONUTF8", "1")
    config = tmp_path / ".env"
    original = "A4S_AAS_ENDPOINT=http://file:8081\nA4S_LAUNCH_TEST=loaded\n"
    config.write_text(original, encoding="utf-8")
    launcher.configure()
    assert launcher.os.environ["A4S_AAS_ENDPOINT"] == "http://explicit:8081"
    assert launcher.os.environ["A4S_LAUNCH_TEST"] == "loaded"
    assert config.read_text(encoding="utf-8") == original
    launcher.os.environ.pop("A4S_LAUNCH_TEST")


def test_agent_mode_forwards_task_without_starting_services(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "configure", lambda: None)
    seen = []
    monkeypatch.setattr("a4s.cli.main", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(launcher, "fetch", lambda *a: pytest.fail("CLI mode must not start the web stack"))
    assert launcher.main(["--agent", "List the instances"]) == 0
    assert seen == [["run", "List the instances"]]


def test_missing_deployment_never_calls_compose_start(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "docker")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"[]")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="not initialized"):
        launcher.start_compose("BaSyx", [], tmp_path)
    assert len(calls) == 1 and "start" not in calls[0]

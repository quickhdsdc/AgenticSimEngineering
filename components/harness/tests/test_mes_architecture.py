"""Shared command contracts and execution correlation, without a live MES."""

import argparse
import json
import shlex

import httpx
import pytest

from a4s.mesctl.actions import ActionSpec
from a4s.mesctl.adapter import PreparedRequest
from a4s.mesctl.client import Config, MesClient, MesError
from a4s.mesctl.commands import build_parser, command_info, describe_actions
from a4s.mesctl.execution import ExecutionContext
from a4s.mesctl.executor import execute


def test_catalog_covers_existing_surface_and_declares_all_writes():
    catalog = describe_actions()["actions"]
    assert len(catalog) == 45
    assert len({a["name"] for a in catalog}) == 45
    for action in catalog:
        if action["method"] != "GET":
            assert action["effects"]
            assert action["risk"] == "business"
            assert action["verification"]["resource_read_back"]
            assert action["dry_run"]
            assert "no automatic replay" in action["retry"]
    resolve = describe_actions(["issue", "resolve"])["actions"][0]
    assert resolve["verification"]["expected_status"] == "RESOLVED"
    assert resolve["verification"]["related_order_read_back"]
    assert next(a for a in resolve["arguments"] if a["name"] == "resolution_notes")["required"]


def test_mutation_cannot_be_registered_without_effects():
    with pytest.raises(ValueError, match="effects"):
        ActionSpec("unsafe", "/api/v1/work-orders", "Missing contract", "POST", "work-orders")


def test_discovery_is_local_and_needs_no_credentials(monkeypatch, capsys):
    from a4s.mesctl.__main__ import main

    monkeypatch.setattr("a4s.kernel.config.load_env", lambda: None)
    monkeypatch.setattr(
        Config, "from_env", lambda: pytest.fail("Discovery must not load MES configuration")
    )
    assert main(["describe", "schedule", "move"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["actions"][0]["agent_policy_checks"]
    assert command_info("mes describe schedule move")["mutates"] is False
    with pytest.raises(MesError, match="No matching"):
        describe_actions(["not-a-domain"])


@pytest.mark.parametrize("value", ["", "x\r\nHeader: injected", "a b", "x" * 129, "中文"])
def test_correlation_ids_are_header_safe(value):
    with pytest.raises(argparse.ArgumentTypeError):
        ExecutionContext(value, "attempt-1")


def test_trace_groups_attempts_without_reusing_execution_ids(monkeypatch):
    monkeypatch.delenv("A4S_MES_EXECUTION_ID", raising=False)
    monkeypatch.delenv("A4S_MES_TRACE_ID", raising=False)
    first = ExecutionContext.from_env("demo-fault-1")
    second = ExecutionContext.from_env("demo-fault-1")
    assert first.trace_id == second.trace_id
    assert first.execution_id != second.execution_id


def test_wire_headers_correlate_write_and_read_back_without_changing_body():
    context = ExecutionContext("demo-fault-1", "attempt-1")
    calls = []

    def handler(req):
        calls.append(req)
        assert req.headers["X-Correlation-ID"] == context.trace_id
        assert req.headers["X-A4S-Execution-ID"] == context.execution_id
        if req.method == "POST":
            assert json.loads(req.content) == {}
        return httpx.Response(200, json={"data": {"id": 42, "status": "PAUSED"}})

    args = build_parser().parse_args(["order", "pause", "42"])
    with MesClient(
        Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)
    ) as client:
        result = execute(args, client, context=context)
        assert client.execution_context is None
    assert len(calls) == 2
    assert result["execution"]["outcome"] == "verified"
    assert result["execution"]["request_fingerprint"] == result["mutation"]["request_fingerprint"]
    assert result["affected_resources"] == ["mes://work-orders/42"]


def test_preview_and_commit_have_same_request_fingerprint():
    seen = []

    def handler(req):
        seen.append(req.method)
        return httpx.Response(200, json={"data": {"id": 42, "status": "PAUSED"}})

    with MesClient(
        Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)
    ) as client:
        preview = execute(build_parser().parse_args(["order", "pause", "42", "--dry-run"]), client)
        assert not seen
        actual = execute(build_parser().parse_args(["order", "pause", "42"]), client)
    assert preview["request_fingerprint"] == actual["mutation"]["request_fingerprint"]
    assert preview["execution"]["execution_id"] != actual["execution"]["execution_id"]
    assert preview["execution"]["outcome"] == "previewed"
    a = PreparedRequest(
        "order update", "PATCH", "http://mes.test", "/api/v1/work-orders/42", {"priority": 1}
    )
    b = PreparedRequest(
        "order update", "PATCH", "http://mes.test", "/api/v1/work-orders/42", {"priority": 2}
    )
    assert a.fingerprint != b.fingerprint


@pytest.mark.parametrize(
    "failure,outcome",
    [(422, "rejected"), (500, "unknown"), (408, "unknown"), ("timeout", "unknown")],
)
def test_mutation_failure_is_correlated_and_never_replayed(failure, outcome):
    calls = []

    def handler(req):
        calls.append(req)
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=req)
        return httpx.Response(failure, json={"message": "Not accepted or outcome unknown"})

    with MesClient(
        Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(MesError) as err:
            execute(
                build_parser().parse_args(["order", "pause", "42"]),
                client,
                context=ExecutionContext("trace", "attempt"),
            )
        assert client.execution_context is None
    assert len(calls) == 1
    assert err.value.execution["outcome"] == outcome
    assert err.value.execution["execution_id"] == "attempt"


def test_bad_readback_is_accepted_but_unverified_not_replayed():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(
            200, json={"data": {"id": 42}} if req.method == "POST" else {"data": []}
        )

    with MesClient(
        Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)
    ) as client:
        result = execute(build_parser().parse_args(["order", "pause", "42"]), client)
    assert result["execution"]["outcome"] == "accepted_unverified"
    assert result["verified"] is False
    assert len(calls) == 2


def test_preflight_failure_makes_no_mutation():
    with MesClient(
        Config("http://mes.test", "secret"),
        transport=httpx.MockTransport(lambda req: pytest.fail("No HTTP expected")),
    ) as client:
        with pytest.raises(MesError) as err:
            execute(build_parser().parse_args(["order", "update", "42"]), client)
    assert err.value.execution["outcome"] == "preflight_failed"


def test_runner_failure_retains_started_attempt(tmp_path):
    from a4s.kernel.approval import Gate, Policy
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    class BrokenEnvironment:
        def run(self, command, **kwargs):
            raise OSError("runner interrupted")

    ledger = Ledger(tmp_path / "ledger.db", "interrupted", actor="actor")
    try:
        gate = Gate(Policy("machine"), ledger)
        registry = ToolRegistry(BrokenEnvironment(), ledger, tmp_path, gate=gate)
        command = "mes order pause 42"
        gate.preapprove(command)
        assert "runner interrupted" in registry.dispatch("shell", {"command": command})
        events = [
            e for e in ledger.read(session_id="interrupted", limit=100) if e.kind == "mes.execution"
        ]
        assert [e.payload["stage"] for e in events] == ["started", "execution_error"]
        assert (
            events[0].payload["execution"]["execution_id"]
            == events[1].payload["execution"]["execution_id"]
        )
        assert events[1].payload["execution"]["outcome"] == "unknown"
    finally:
        ledger.close()


def test_gate_and_ledger_share_cli_execution_ids(tmp_path):
    from a4s.environments.base import ExecResult
    from a4s.kernel.approval import Gate, Policy
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    class FakeEnvironment:
        def run(self, command, **kwargs):
            execution_env = kwargs["env"]
            context = ExecutionContext(
                execution_env["A4S_MES_TRACE_ID"], execution_env["A4S_MES_EXECUTION_ID"]
            )
            args = build_parser().parse_args(shlex.split(command)[1:])

            def handler(req):
                return httpx.Response(200, json={"data": {"id": 42, "status": "PAUSED"}})

            with MesClient(
                Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)
            ) as client:
                output = execute(args, client, context=context)
            return ExecResult(0, json.dumps(output), "", False, 0, "test")

    ledger = Ledger(tmp_path / "ledger.db", "task-session", actor="actor")
    try:
        gate = Gate(Policy("machine"), ledger)
        registry = ToolRegistry(FakeEnvironment(), ledger, tmp_path, gate=gate)
        command = "mes order pause 42 --trace-id fault-1"
        assert "refused" in registry.dispatch("shell", {"command": command})
        assert not [
            e
            for e in ledger.read(session_id="task-session", limit=100)
            if e.kind == "mes.execution"
        ]
        registry.begin_turn()
        gate.preapprove(command)
        result = json.loads(registry.dispatch("shell", {"command": command}))
        eid = result["execution"]["execution_id"]
        events = list(ledger.read(session_id="task-session", limit=100))
        approval = next(e for e in events if e.kind == "approval" and e.payload["allowed"])
        assert approval.payload["detail"]["execution"]["execution_id"] == eid
        assert approval.payload["detail"]["effects"]
        write = next(e for e in events if e.kind == "mes.write")
        assert write.payload["execution"]["execution_id"] == eid
        assert write.payload["execution"]["trace_id"] == "fault-1"
        assert write.payload["outcome"] == "verified"
        assert write.payload["execution"]["request_fingerprint"]
        stages = [e.payload["stage"] for e in events if e.kind == "mes.execution"]
        assert stages == ["started", "finished"]
    finally:
        ledger.close()

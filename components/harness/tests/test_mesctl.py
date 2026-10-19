"""OpenMES wire contracts and the agent's approval/execution boundary."""

import json
import shlex

import httpx
import pytest

from a4s.mesctl.commands import build_parser, command_info
from a4s.mesctl.executor import execute
from a4s.mesctl.domains.scheduling import schedule_time
from a4s.mesctl.client import Config, MesClient, MesError
from a4s.kernel.approval import BUSINESS, READ, Gate, Policy, Scopes, classify


def run(command, handler):
    args = build_parser().parse_args(shlex.split(command))
    with MesClient(Config("http://mes.test", "secret-token"),
                   transport=httpx.MockTransport(handler)) as client:
        return execute(args, client)


def test_order_create_wire_contract_and_read_back():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Bearer secret-token"
        if request.method == "POST":
            assert json.loads(request.content) == {
                "order_no": "WO-4711", "planned_qty": 50, "line_id": 7,
                "product_type_id": 3, "priority": 0,
            }
            return httpx.Response(201, json={"data": {"id": 42, "status": "PENDING"}})
        return httpx.Response(200, json={"data": {"id": 42, "status": "PENDING",
            "order_no": "WO-4711", "planned_qty": "50.00", "line_id": 7, "product_type_id": 3, "priority": 0}})

    result = run("order create --order-no WO-4711 --planned-qty 50 --line 7 --product-type-id 3 --priority 0", handler)
    assert result["verified"]
    assert calls == [("POST", "/api/v1/work-orders"), ("GET", "/api/v1/work-orders/42")]


def test_dry_run_never_sends_a_mutation():
    result = run("order create --order-no X --planned-qty 3 --dry-run", lambda req: pytest.fail("No HTTP expected"))
    assert result["dry_run"] and result["body"]["planned_qty"] == 3


@pytest.mark.parametrize("count", [0, 2])
def test_current_order_refuses_ambiguity(count):
    def handler(req):
        assert req.url.params["status"] == "IN_PROGRESS"
        return httpx.Response(200, json={"data": [{"id": i} for i in range(count)]})
    with pytest.raises(MesError, match="exactly one"):
        run("order current --line 7", handler)


def test_exact_code_resolution_and_nested_pagination():
    def handler(req):
        page = int(req.url.params["page"])
        return httpx.Response(200, json={"data": {"data": [{"id": page, "code": f"L{page}"}], "last_page": 2}})
    with MesClient(Config("http://mes.test", "secret"), transport=httpx.MockTransport(handler)) as client:
        assert client.resolve("lines", "L2", "code") == 2
        with pytest.raises(MesError, match="found 0"):
            client.resolve("lines", "L", "code")


def test_issue_requires_title_and_resolution_notes():
    for args in ("issue create --order 42 --type TOOL_FAILURE", "issue resolve 19"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(shlex.split(args))


def test_issue_type_code_resolution_and_automatic_order_effect():
    def handler(req):
        path = req.url.path
        if path == "/api/v1/issue-types":
            return httpx.Response(200, json={"data": [{"id": 3, "code": "TOOL_FAILURE"}]})
        if path == "/api/v1/issue-types/3":
            return httpx.Response(200, json={"data": {"id": 3, "is_blocking": True}})
        if path == "/api/v1/issues":
            assert json.loads(req.content)["issue_type_id"] == 3
            return httpx.Response(201, json={"data": {"id": 19, "work_order_id": 42}})
        if path == "/api/v1/issues/19":
            return httpx.Response(200, json={"data": {"id": 19, "work_order_id": 42,
                "issue_type_id": 3, "title": "Festo fault", "status": "OPEN"}})
        assert path == "/api/v1/work-orders/42"
        return httpx.Response(200, json={"data": {"id": 42, "status": "BLOCKED"}})
    result = run('issue create --order 42 --type TOOL_FAILURE --title "Festo fault"', handler)
    assert result["verified"]
    assert result["order_after"]["data"]["status"] == "BLOCKED"


@pytest.mark.parametrize("failure", ["incompatible", "in_progress", "routing", "conflict"])
def test_schedule_preflight_and_server_conflict(failure):
    writes = []
    def handler(req):
        if req.method != "GET":
            writes.append(req)
            assert failure == "conflict"
            return httpx.Response(409, json={"conflict": True, "message": "Overlap"})
        path = req.url.path
        if path == "/api/v1/work-orders/42":
            return httpx.Response(200, json={"data": {"id": 42, "product_type_id": 3,
                "status": "IN_PROGRESS" if failure == "in_progress" else "PENDING",
                "process_snapshot": {"steps": [{"workstation_id": 999}]} if failure == "routing" else {}}})
        if path.endswith("/product-types"):
            return httpx.Response(200, json={"data": [] if failure == "incompatible" else [{"id": 3}]})
        if path.endswith("/workstations"):
            return httpx.Response(200, json={"data": [{"id": 4}]})
        return httpx.Response(200, json={"grid": {}})
    with pytest.raises(MesError) as error:
        run("schedule move 42 --line 9 --start 2026-09-11T08:00:00Z --end 2026-09-11T09:00:00Z", handler)
    assert len(writes) == (1 if failure == "conflict" else 0)
    if failure == "conflict":
        assert error.value.status == 409


def test_failed_read_back_retains_successful_mutation():
    writes = []
    def handler(req):
        if req.method == "POST":
            writes.append(req)
            return httpx.Response(200, json={"data": {"id": 42, "status": "PAUSED"}})
        return httpx.Response(503, json={"message": "Unavailable"})
    result = run("order pause 42", handler)
    assert not result["verified"]
    assert result["response"]["data"]["status"] == "PAUSED"
    assert len(writes) == 1


def test_read_back_must_confirm_requested_status():
    result = run("order pause 42", lambda req: httpx.Response(200,
        json={"data": {"id": 42, "status": "PAUSED" if req.method == "POST" else "IN_PROGRESS"}}))
    assert result["verified"] is False
    assert "status" in result["verification_error"]


@pytest.mark.parametrize("status", [401, 403, 422, 429, 500])
def test_http_errors_are_structured_and_credentials_redacted(status):
    def handler(req):
        return httpx.Response(status, json={"message": "echo secret-token", "token": "anything", "errors": {"title": ["required"]}})
    with pytest.raises(MesError) as error:
        run("line ls", handler)
    assert error.value.status == status
    assert "secret-token" not in str(error.value)
    assert error.value.details["token"] == "[REDACTED]"
    assert error.value.details["errors"]["title"] == ["required"]


def test_redirects_are_never_followed_with_credentials():
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(302, headers={"Location": "https://other.test"}, json={"message": "redirect"})
    with pytest.raises(MesError):
        run("line ls", handler)
    assert len(calls) == 1


def test_health_does_not_send_token_and_config_is_not_repr_secret():
    config = Config("http://mes.test", "hidden-token")
    assert "hidden-token" not in repr(config)
    def handler(req):
        assert "Authorization" not in req.headers
        return httpx.Response(200, json={"status": "ok"})
    with MesClient(config, transport=httpx.MockTransport(handler)) as client:
        client.get("/api/health")


@pytest.mark.parametrize("endpoint", ["", "file:///tmp", "http://user:pass@mes.test", "http://mes.test/?token=x"])
def test_config_rejects_invalid_or_credential_urls(endpoint):
    with pytest.raises(MesError):
        Config(endpoint)


@pytest.mark.parametrize("command,risk", [
    ("mes order show 42", READ),
    ("mes order pause 42", BUSINESS),
    ("mes order pause 42 --dry-run", READ),
    ("mes order pause 42 --invalid", BUSINESS),
    ("python -m a4s.mesctl order pause 42", BUSINESS),
    ('"C:/venv/Scripts/mes.exe" issue resolve 19 --notes repaired', BUSINESS),
    ("mes order show 42 && mes order pause 42", BUSINESS),
    ("mes --help", READ),
])
def test_mes_commands_are_gated_by_the_cli_parser(command, risk):
    action = classify("shell", {"command": command})
    assert action.risk == risk
    assert action.subject.startswith("mes://")
    assert Policy("machine").decide(action) == ("ask" if risk == BUSINESS else "allow")


def test_scope_and_one_shot_approval():
    gate = Gate(Policy("machine"), scopes=Scopes(configured=True, entries={"actor": ["mes://work-orders/42"]}), actor="actor")
    command = "mes order pause 42"
    assert not gate.check("shell", {"command": command}).allowed
    gate.preapprove(command)
    assert gate.check("shell", {"command": command}).allowed
    assert not gate.check("shell", {"command": command}).allowed
    assert not gate.check("shell", {"command": "mes order pause 43"}).allowed


def test_prompt_advertises_mes_only_when_configured(monkeypatch, tmp_path):
    from a4s.kernel.prompt import build_system_prompt
    monkeypatch.delenv("A4S_MES_ENDPOINT", raising=False)
    assert "The `mes` command surface" not in build_system_prompt(tmp_path)
    monkeypatch.setenv("A4S_MES_ENDPOINT", "http://mes.test")
    monkeypatch.setenv("A4S_MES_TOKEN", "must-not-appear")
    prompt = build_system_prompt(tmp_path)
    assert "mes schedule move" in prompt
    assert "must-not-appear" not in prompt


def test_tool_registry_blocks_then_executes_and_records_mes_write(tmp_path):
    from a4s.environments.base import ExecResult
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry
    class FakeEnvironment:
        count = 0
        def run(self, command, **kwargs):
            self.count += 1
            return ExecResult(exit_code=0, stdout='{"verified":true}', stderr="",
                              truncated=False, duration_s=0, environment="test")
    env = FakeEnvironment()
    ledger = Ledger(tmp_path / "ledger.db", "mes-test", actor="actor")
    try:
        gate = Gate(Policy("machine"), ledger, approver=None)
        registry = ToolRegistry(env, ledger, tmp_path, gate=gate)
        command = "mes order pause 42"
        assert "refused" in registry.dispatch("shell", {"command": command})
        assert env.count == 0
        registry.begin_turn()
        gate.preapprove(command)
        registry.dispatch("shell", {"command": command})
        assert env.count == 1
        events = list(ledger.read(session_id="mes-test", limit=100))
        writes = [e for e in events if e.kind == "mes.write"]
        assert len(writes) == 1
        assert writes[0].subject == "mes://work-orders/42"
    finally:
        ledger.close()


def test_non_mes_command_is_not_claimed():
    assert command_info("aas get X Y") is None


def test_schedule_timezone_must_be_known():
    with pytest.raises(MesError, match="A4S_MES_TIMEZONE"):
        schedule_time("2026-09-11T08:00:00Z", "")
    assert schedule_time("2026-09-11T10:00:00+02:00", "UTC") == "2026-09-11T08:00:00+00:00"


def test_schedule_converts_to_server_local_and_refuses_dst_ambiguity():
    pytest.importorskip("tzdata")
    assert schedule_time("2026-09-11T08:00:00Z", "Europe/Berlin") == "2026-09-11T10:00:00+02:00"
    with pytest.raises(MesError, match="Ambiguous DST"):
        schedule_time("2026-10-25T00:30:00Z", "Europe/Berlin")


def test_compound_mes_commands_cannot_hide_an_out_of_scope_target():
    gate = Gate(Policy("off"), scopes=Scopes(configured=True, entries={"actor": ["mes://work-orders/42"]}), actor="actor")
    decision = gate.check("shell", {"command": "mes order pause 42 && mes order pause 43"})
    assert not decision.allowed
    assert decision.source == "command-shape"
    assert "separate shell call" in gate.refusal(decision)

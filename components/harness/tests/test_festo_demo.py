"""Offline regression of discovery, sourced knowledge and operator demo safety."""

import copy
import json
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from a4s.aasctl.catalog import LEGACY_BLUEPRINTS, shells_in_view, submodels_in_view
from a4s.aasctl.__main__ import build_parser
from a4s.machines.festo_cp_lab import demo_monitor, knowledge, maintenance
from a4s.server import festo_demo_routes as routes


def shell(identifier, kind="Instance", refs=()):
    return {"id": identifier, "assetInformation": {"assetKind": kind}, "submodels": [
        {"keys": [{"type": "Submodel", "value": ref}]} for ref in refs]}


def test_instance_discovery_uses_metadata_and_explicit_legacy_configuration(monkeypatch):
    monkeypatch.setenv("A4S_BLUEPRINT_IDS", "configured")
    legacy = next(iter(LEGACY_BLUEPRINTS))
    rows = [shell("type", "Type"), shell(legacy), shell("configured"),
            shell("InstanceNamedTemplate"), shell("unknown", "NotApplicable")]
    original = copy.deepcopy(rows)
    assert [s["id"] for s in shells_in_view(rows)] == ["InstanceNamedTemplate"]
    assert {s["id"] for s in shells_in_view(rows, "Type")} == {"type", legacy, "configured"}
    assert len(shells_in_view(rows, "all")) == 5
    assert rows == original  # discovery never rewrites deployed asset kinds


def test_submodel_kind_is_independent_and_shared_instance_is_retained():
    shells = [shell("type", "Type", ["blueprint", "shared"]),
              shell("real", refs=["shared", "real", "template"])]
    rows = [{"id": identifier, "kind": kind} for identifier, kind in [
        ("blueprint", "Instance"), ("shared", "Instance"),
        ("real", "Instance"), ("template", "Template")]]
    assert [s["id"] for s in submodels_in_view(rows, shells)] == ["shared", "real"]
    assert [s["id"] for s in submodels_in_view(rows, shells, "Template")] == ["template"]
    assert len(submodels_in_view(rows, shells, "all")) == 4


@pytest.mark.parametrize("words,attribute,expected", [
    (["shell", "list"], "asset_kind", "Instance"),
    (["shell", "list", "--asset-kind", "all"], "asset_kind", "all"),
    (["sm", "list"], "kind", "Instance"),
    (["sm", "list", "--kind", "Template"], "kind", "Template"),
    (["index", "--all"], "kind", "Instance"),
    (["find", "pressure", "--across"], "kind", "Instance"),
])
def test_operational_cli_defaults(words, attribute, expected):
    assert getattr(build_parser().parse_args(words), attribute) == expected


def test_knowledge_separates_reference_from_fiction_and_supports_chinese():
    pytest.importorskip("pypdfium2")
    results = knowledge.search("气压泄漏维护", 7)
    assert any(r["id"] == "pneumatic-leak-scenario" and r["origin"] == "DEMO_FICTION" for r in results)
    hardware = next(r for r in knowledge.search("turning", 7) if r["id"] == "hardware-layout")
    assert hardware["origin"] == "REFERENCE"
    assert "not lathe machining" in hardware["text"]
    assert "slides" in hardware["source"]
    assert knowledge.search("zzzzunmatchedterm") == []
    assert len(knowledge.search("", 2)) == 2


@pytest.mark.parametrize("seconds,status", [(0, "NORMAL"), (8, "NORMAL"), (15, "WARNING"), (21, "FAULT")])
def test_monitor_deterministic_fiction(seconds, status):
    row = demo_monitor.signal_at(seconds)
    assert row["status"] == status
    assert row["turning_ready"] is True
    if status == "FAULT":
        assert row["press_ready"] is False
        assert row["collective_fault"] is True
        assert row["conveyor_speed_m_min"] == 0


def test_monitor_arm_latch_history_and_disarm(monkeypatch):
    monkeypatch.setattr(demo_monitor, "STATE", {"run_id": None, "started_at": None, "active": False})
    now = [100.0]
    monkeypatch.setattr(demo_monitor.time, "time", lambda: now[0])
    first = demo_monitor.start()
    assert demo_monitor.start()["run_id"] == first["run_id"]
    now[0] += 21
    alarm = demo_monitor.observe()
    assert alarm["mode"] == "SIMULATION" and alarm["alarm_id"]
    now[0] += 200
    assert demo_monitor.observe()["alarm_id"] == alarm["alarm_id"]
    assert len(demo_monitor.observe()["history"]) == 60
    assert demo_monitor.stop()["status"] == "DISARMED"
    assert demo_monitor.observe()["alarm_id"] is None
    assert demo_monitor.start()["run_id"] != first["run_id"]


def test_maintenance_profile_reuses_semantics_without_claiming_complete_template(tmp_path):
    def collection(name, values):
        return {"modelType": "SubmodelElementCollection", "idShort": name, "value": values}

    def prop(name):
        return {"modelType": "MultiLanguageProperty", "idShort": name,
                "value": [{"language": "en", "text": "example"}],
                "semanticId": {"type": "ExternalReference", "keys": [
                    {"type": "GlobalReference", "value": "urn:test:" + name}]}}

    template = {"modelType": "Submodel", "kind": "Template", "id": "urn:test:02018",
                "idShort": "MaintenanceInstructions", "submodelElements": [
                    collection("MaintenanceInstructionsForSpecificInterval__00__", [
                        collection("BasicMaintenanceInformation", [prop(n) for n in (
                            "MaintenanceID", "NameOfMaintenance", "SourceOfMaintenanceInstructions",
                            "SafetyRegulationsToBeObserved")]),
                        collection("ListMaintenanceSteps", [collection("MaintenanceStep__00__", [
                            prop(n) for n in ("MaintenanceStepID", "MaintenanceStepName", "InstructionMaintenanceStep")])])])]}
    path = tmp_path / "template.json"
    path.write_text(json.dumps({"submodels": [template]}), encoding="utf-8")
    result = maintenance.build(path)
    assert result["id"] == maintenance.MAINTENANCE_ID and result["kind"] == "Instance"
    assert result["semanticId"]["keys"][0]["value"] == maintenance.PROFILE
    steps = result["submodelElements"][0]["value"][1]["value"]
    assert len(steps) == 3
    assert steps[0]["value"][0]["semanticId"]["keys"][0]["value"] == "urn:test:MaintenanceStepID"
    assert "DEMO FICTION" in json.dumps(result)
    assert "Not a claim of complete IDTA" in json.dumps(result)


def test_production_context_holds_references_not_a_second_business_ledger():
    result = maintenance.production_context("A4S-FESTO", "A4S-BACKUP", "A4S-DEMO-PART", "http://mes.test")
    fields = routes._fields(result)
    assert fields["MESLineCode"] == "A4S-FESTO" and fields["EvidenceMode"] == "SIMULATION"
    assert "AffectedOrderId" not in fields
    assert not any("status" in name.lower() or "stock" in name.lower() for name in fields)


@pytest.fixture
def demo(monkeypatch, tmp_path):
    state = {"asset": {"id": maintenance.SHELL_ID}, "context": {"MESBackupLineCode": "A4S-BACKUP"},
             "maintenance": {"id": "urn:test:maintenance"}, "order_a": {"id": 11, "status": "ACCEPTED"},
             "order_b": {"id": 12, "line_id": 1}, "issues": [], "line": {"id": 1}, "backup": {"id": 2}}
    monkeypatch.setattr(routes, "PLANS", {})
    monkeypatch.setattr(routes, "AUDIT_PATH", tmp_path / "ledger.db")
    monkeypatch.setattr(routes, "snapshot", lambda: copy.deepcopy(state))
    monkeypatch.setattr(routes.Config, "from_env", lambda: None)
    monkeypatch.setattr(routes, "MesClient", lambda config: nullcontext(object()))
    monkeypatch.setattr(routes, "_run", lambda client, words: {"request_fingerprint": "fp"})
    calls = []

    def execute(args, client, context):
        calls.append(args)
        return {"verified": True, "execution": {"outcome": "verified", **context.as_dict()}}

    monkeypatch.setattr(routes, "execute", execute)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, state, calls


def new_plan(client):
    response = client.post("/api/demo/festo/preview", json={"scenario": "simulated-maintenance", "action": "create_issue"})
    assert response.status_code == 200, response.text
    return response.json()


def test_operator_requires_approval_consumes_once_and_records_readback(demo):
    client, state, calls = demo
    plan = new_plan(client)
    assert not calls and "state_digest" not in plan and "words" not in plan
    body = {"plan_id": plan["id"], "approved": False}
    assert client.post("/api/demo/festo/execute", json=body).status_code == 403
    body["approved"] = True
    assert client.post("/api/demo/festo/execute", json=body).json()["result"]["verified"] is True
    assert client.post("/api/demo/festo/execute", json=body).status_code == 409
    assert len(calls) == 1
    kinds = [e["kind"] for e in client.get("/api/demo/festo/audit").json()["events"]]
    assert "approval" in kinds and "mes.write" in kinds


@pytest.mark.parametrize("reason", ["stale", "expired", "changed_request", "concurrent"])
def test_stale_expired_changed_or_concurrent_approval_never_writes(demo, monkeypatch, reason):
    client, state, calls = demo
    plan = new_plan(client)
    if reason == "stale":
        state["order_a"]["status"] = "BLOCKED"
    elif reason == "expired":
        routes.PLANS[plan["id"]]["expires_at"] = 0
    elif reason == "changed_request":
        monkeypatch.setattr(routes, "_run", lambda c, w: {"request_fingerprint": "changed"})
    else:
        routes.COMMIT_LOCK.acquire()
    try:
        assert client.post("/api/demo/festo/execute", json={"plan_id": plan["id"], "approved": True}).status_code == 409
        assert not calls
    finally:
        if reason == "concurrent":
            routes.COMMIT_LOCK.release()


def test_recovery_actions_reject_duplicate_issue_wrong_state_and_arbitrary_action(demo):
    client, state, calls = demo
    state["issues"] = [{"id": 9, "status": "OPEN"}]
    for action in ("create_issue", "resolve_issue", "close_issue", "plc_reset"):
        with pytest.raises(HTTPException):
            routes.action_words(action, state)
    assert routes.action_words("acknowledge_issue", state) == ["issue", "acknowledge", "9"]
    state["issues"][0]["status"] = "ACKNOWLEDGED"
    assert "SIMULATED" in routes.action_words("resolve_issue", state)[-1]
    assert client.post("/api/demo/festo/monitor/start", json={"mode": "LIVE"}).status_code == 422
    assert not calls


def test_audit_returns_latest_events_in_chronological_order(demo):
    client, _, _ = demo
    with routes.Ledger(routes.AUDIT_PATH, "tail-test", actor="test") as ledger:
        for number in range(125):
            ledger.append("test", {"number": number})
        assert ledger.read(limit=1)[0].payload["number"] == 0
    events = client.get("/api/demo/festo/audit").json()["events"]
    assert len(events) == 100
    assert events[0]["payload"]["number"] == 25
    assert events[-1]["payload"]["number"] == 124


@pytest.mark.parametrize("failure", ["unverified", "timeout"])
def test_uncertain_execution_is_not_reported_verified_or_replayed(demo, monkeypatch, failure):
    client, _, calls = demo

    def execute(args, mes, context):
        calls.append(args)
        if failure == "timeout":
            raise routes.MesError("Write outcome unknown; inspect before retry")
        return {"verified": False, "execution": {"outcome": "accepted_unverified", **context.as_dict()}}

    monkeypatch.setattr(routes, "execute", execute)
    plan = new_plan(client)
    body = {"plan_id": plan["id"], "approved": True}
    response = client.post("/api/demo/festo/execute", json=body)
    if failure == "unverified":
        assert response.status_code == 200
        assert response.json()["result"]["verified"] is False
    else:
        assert response.status_code == 502
    assert client.post("/api/demo/festo/execute", json=body).status_code == 409
    assert len(calls) == 1

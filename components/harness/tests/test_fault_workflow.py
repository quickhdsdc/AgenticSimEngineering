from pathlib import Path

from a4s.kernel.prompt import load_skills, select_skill
from a4s.machines.festo_cp_lab.fault_profile import models
from a4s.machines.festo_cp_lab.opc_simulator import ENDPOINT, signal_values
from a4s.machines.festo_cp_lab import knowledge
from a4s.server.live_monitor_routes import investigation_request
from a4s.server.opc_routes import _walk


def test_uniform_notification_selects_maintenance_skill_without_test_instructions():
    for source in ("live", "simulator"):
        event = {"id": "event-1", "at": 2, "target": {"asset_id": "urn:test:current",
                 "endpoint": "http://localhost:8081", "source": source},
                 "before": {"source": source, "values": []}, "after": {"values": []}}
        request = investigation_request(event)
        assert "urn:test:current" in request
        skills = load_skills(Path("a4s/skills"))
        assert select_skill(skills, request).name == "fault-maintenance"
        assert ("SIMULATED OPC" in request) == (source == "simulator")


def test_profile_only_exposes_loopback_reads_and_business_references():
    context = {"id": "old", "idShort": "ProductionContext", "submodelElements": []}
    built = models(context, "urn:test:current")
    found = []
    _walk(built[0], "", found)
    assert len(found) == 8
    assert all(s["endpoint"].rstrip("/") == ENDPOINT.rstrip("/") for s in found)
    assert all(s["description"] for s in found)
    assert context["id"] == "old"
    assert "Operation\"" not in str(built)


def test_leak_and_emergency_stop_have_distinct_evidence():
    leak = signal_values("fault")
    assert leak["Press.PressureBar"] < 4.8 and leak["Press.FlowLMin"] > 30
    assert not leak["Safety.EmergencyStop"] and not leak["Turning.Fault"]
    assert signal_values("estop")["Safety.EmergencyStop"]
    assert not signal_values("normal")["Press.Fault"]


def test_retrieval_contains_real_pdf_diagnostic_and_planning_passages():
    hits = knowledge.search("pneumatic pressure flow Fault01 emergency stop", 4)
    assert any(hit["id"] == "pneumatic-diagnostic-matrix" for hit in hits)
    hits = knowledge.search("maintenance inspection planning allowance", 7)
    assert any(hit["id"] == "pneumatic-maintenance-plan" and hit["page"] == 2 for hit in hits)

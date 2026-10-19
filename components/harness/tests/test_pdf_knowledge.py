"""Actual PDF extraction and synthetic fault controls, without external services."""

import hashlib

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from a4s.machines.festo_cp_lab import pdf_knowledge, knowledge, demo_monitor
from a4s.server import festo_demo_routes as routes


@pytest.fixture(autouse=True)
def pdf_dependency():
    pytest.importorskip("pypdfium2")


def test_rehearsal_actions_require_closed_issue_and_correct_order_state(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException

    monkeypatch.setattr(routes.Config, "from_env", lambda: SimpleNamespace(server_timezone="UTC"))
    state = {"context": {"MESLineCode": "A4S-FESTO", "MESBackupLineCode": "A4S-BACKUP"},
             "line": {"id": 1}, "backup": {"id": 2},
             "order_a": {"id": 5, "status": "PENDING"},
             "order_b": {"id": 6, "line_id": 2,
                         "planned_start_at": "2026-09-13T09:00:00Z",
                         "planned_end_at": "2026-09-13T09:30:00Z"},
             "issues": [{"id": 9, "status": "OPEN"}]}
    for action in ("return_upcoming", "accept_affected"):
        with pytest.raises(HTTPException) as exc:
            routes.action_words(action, state)
        assert exc.value.status_code == 409
    state["issues"][0]["status"] = "CLOSED"
    assert routes.action_words("accept_affected", state) == ["order", "accept", "5"]
    words = routes.action_words("return_upcoming", state)
    assert words[:5] == ["schedule", "move", "6", "--line", "1"]
    assert words[6] == "2026-09-13T09:00:00+00:00"
    state["order_b"]["line_id"] = 1
    with pytest.raises(HTTPException):
        routes.action_words("return_upcoming", state)
    state["order_a"]["status"] = "ACCEPTED"
    with pytest.raises(HTTPException):
        routes.action_words("accept_affected", state)


def test_pdf_corpus_has_page_citations_and_real_file_hashes():
    rows = pdf_knowledge.chunks()
    assert len(rows) == 9
    assert len({r["id"] for r in rows}) == 9
    assert {r["origin"] for r in rows} == {"REFERENCE", "DEMO_FICTION", "PROJECT_CONTRACT", "ENGINEERING_SCENARIO"}
    for row in rows:
        data = pdf_knowledge.document_path(row["document_id"]).read_bytes()
        assert data.startswith(b"%PDF-")
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert f"p. {row['page']}" in row["citation"]
        assert "CONTENT" not in row["text"]
    assert sum(d["pages"] for d in pdf_knowledge.catalog()) == 9


def test_retrieval_reads_modified_pdf_not_authoring_constants(monkeypatch, tmp_path):
    for identifier, filename in pdf_knowledge.FILES.items():
        data = pdf_knowledge.document_path(identifier).read_bytes()
        if identifier == "maintenance":
            # Equal-byte-length edit keeps the PDF xref intact while proving
            # text and provenance are derived from the file actually ingested.
            assert b"6.0 bar" in data
            data = data.replace(b"6.0 bar", b"9.9 bar")
        (tmp_path / filename).write_bytes(data)
    monkeypatch.setattr(pdf_knowledge, "DIRECTORY", tmp_path)
    row = next(r for r in knowledge.search("pressure leakage", 7) if r["id"] == "pneumatic-leak-scenario")
    assert "9.9 bar" in row["text"]
    assert "6.0 bar" not in row["text"]
    assert row["sha256"] == hashlib.sha256((tmp_path / row["filename"]).read_bytes()).hexdigest()


def test_missing_pdf_fails_without_silent_text_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_knowledge, "DIRECTORY", tmp_path)
    with pytest.raises(FileNotFoundError, match="PDFs missing"):
        knowledge.search("pressure")
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.get("/api/demo/festo/knowledge").status_code == 503
        assert client.get("/api/demo/festo/knowledge/documents").status_code == 503


def test_pdf_http_catalog_download_and_unknown_id():
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        catalog = client.get("/api/demo/festo/knowledge/documents")
        assert catalog.status_code == 200 and len(catalog.json()["documents"]) == 4
        pdf = client.get("/api/demo/festo/knowledge/documents/technical")
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")
        assert pdf.headers["content-type"] == "application/pdf"
        assert client.get("/api/demo/festo/knowledge/documents/not-allowed").status_code == 404
        result = client.get("/api/demo/festo/knowledge", params={"query": "pressure leakage"}).json()
        assert result["retrieval"].startswith("PDF")
        assert all("page" in c and "sha256" in c for c in result["chunks"])


def test_operator_injects_and_recovers_without_rewriting_signal_history(monkeypatch):
    monkeypatch.setattr(demo_monitor, "STATE", {"run_id": None, "started_at": None, "active": False})
    now = [100.0]
    monkeypatch.setattr(demo_monitor.time, "time", lambda: now[0])
    with pytest.raises(ValueError, match="Start simulation"):
        demo_monitor.inject()
    demo_monitor.start(normal_only=True)
    now[0] += 50
    assert demo_monitor.observe()["status"] == "NORMAL"
    demo_monitor.inject()
    now[0] += 25
    alarm = demo_monitor.observe()
    assert alarm["status"] == "FAULT"
    assert demo_monitor.inject()["alarm_id"] == alarm["alarm_id"]
    recovered = demo_monitor.recover()
    assert recovered["status"] == "NORMAL" and recovered["alarm_id"] is None
    assert any(r["status"] == "FAULT" for r in recovered["history"])
    demo_monitor.inject()
    now[0] += 25
    assert demo_monitor.observe()["alarm_id"] != alarm["alarm_id"]


def test_new_signal_controls_require_explicit_simulation(monkeypatch):
    monkeypatch.setattr(demo_monitor, "STATE", {"run_id": None, "started_at": None, "active": False})
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        for action in ("inject", "recover"):
            assert client.post(f"/api/demo/festo/monitor/{action}", json={"mode": "LIVE"}).status_code == 422
        assert client.post("/api/demo/festo/monitor/inject", json={"mode": "SIMULATION"}).status_code == 409
        state = client.post("/api/demo/festo/monitor/start", json={"mode": "SIMULATION", "normal_only": True}).json()
        assert state["active"] and state["leak_started_at"] is None

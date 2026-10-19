"""No PLC/network: exercise physical-start gates, evidence and failure handling."""
import asyncio
from contextlib import asynccontextmanager
import copy
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from a4s.server import festo_start_routes as start

ORIGIN = {"origin": "http://localhost:5173"}


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(start, "DIRECTORY", tmp_path)
    monkeypatch.setattr(start, "bindings", lambda: "binding")
    app = FastAPI()
    app.include_router(start.router)
    with TestClient(app) as client:
        yield client


def normal():
    return {name: name in ("Ready", "ManualEnabled") for name in start.SIGNALS}


def test_confirmation_origin_single_use_and_global_reservation(setup, monkeypatch):
    client = setup
    launched = []

    class Thread:
        def __init__(self, **kwargs):
            launched.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(start.threading, "Thread", Thread)
    assert client.post("/api/festo/start/preview").status_code == 403
    plan = client.post("/api/festo/start/preview", headers=ORIGIN).json()
    second = client.post("/api/festo/start/preview", headers=ORIGIN).json()
    body = {"id": plan["id"], "approved": True, "onsite_safe": True}
    assert client.post("/api/festo/start/execute", headers=ORIGIN, json={"id":plan["id"]}).status_code == 403
    assert client.post("/api/festo/start/execute", headers=ORIGIN, json=body).status_code == 200
    assert client.post("/api/festo/start/execute", headers=ORIGIN, json=body).status_code == 409
    assert client.post("/api/festo/start/execute", headers=ORIGIN, json={**body, "id":second["id"]}).status_code == 409
    assert len(launched) == 1
    assert client.post("/api/festo/start/review", headers=ORIGIN, json={"id":plan["id"], "onsite_reviewed":True}).status_code == 409


def reserve():
    run = {"id":"test-run", "status":"running", "owner":start.OWNER, "binding":"binding",
           "events":[], "commands":{}, "results":{}, "dispatch_deadline":time.time()+30}
    start.save(run)
    with start.database() as db:
        db.execute("UPDATE guard SET run=? WHERE id=1", (run["id"],))
    return run


def run_fake(monkeypatch, run, *, fault=False, fail=False, bad_read=False):
    monkeypatch.setattr(start, "WINDOW", .4)
    monkeypatch.setattr(start, "INTERVAL", .001)
    monkeypatch.setattr(start, "Session", lambda client: client)
    calls = []

    class Fake:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.started = False
            self.count = 0

        async def sample(self):
            if bad_read and self.started:
                raise TimeoutError()
            row = normal()
            row["Fault"] = fault
            if self.started:
                self.count += 1
                row["ConveyorForward"] = self.count <= 2
                row["Fault"] = self.count > 2
            return row

        async def start(self):
            assert len([e for e in run["events"] if e["kind"] == "sample" and e["phase"] == "baseline"]) == 4
            calls.append(self.endpoint)
            self.started = True
            if fail and self.endpoint.endswith("4041"):
                raise TimeoutError()  # write may have reached PLC despite timeout

    @asynccontextmanager
    async def factory(endpoint):
        yield Fake(endpoint)

    asyncio.run(start.conduct(run, factory))
    return calls


def test_existing_fault_records_baseline_but_never_writes(setup, monkeypatch):
    run = reserve()
    assert run_fake(monkeypatch, run, fault=True) == []
    assert run["blockers"]
    assert start.classify(run, "Press") == "BLOCKED_BEFORE_WRITE"
    assert start.fetch(run["id"])["events"]
    assert (start.DIRECTORY / "test-run.ledger.db").is_file()


def test_records_before_concurrent_writes_and_detects_motion_fault_stop(setup, monkeypatch):
    run = reserve()
    assert len(run_fake(monkeypatch, run)) == 2
    assert run["commands"] == {"Press":"accepted", "Turning":"accepted"}
    assert all(start.classify(run, s) == "MOTION_THEN_FAULT_STOP" for s in start.STATIONS)


@pytest.mark.parametrize("fail,bad_read", [(True,False), (False,True)])
def test_partial_write_or_lost_observation_never_retries(setup, monkeypatch, fail, bad_read):
    run = reserve()
    assert len(run_fake(monkeypatch, run, fail=fail, bad_read=bad_read)) == 2
    assert start.classify(run, "Turning") == "EVIDENCE_INCOMPLETE"


def test_restart_keeps_guard_until_explicit_late_onsite_review(setup):
    run = reserve()
    run["owner"] = "old-process"
    start.save(run)
    assert setup.get("/api/festo/start/latest").json()["run"]["status"] == "interrupted"
    assert setup.post("/api/festo/start/preview", headers=ORIGIN).status_code == 409
    body = {"id":run["id"], "onsite_reviewed":True}
    assert setup.post("/api/festo/start/review", headers=ORIGIN, json=body).status_code == 409
    run["dispatch_deadline"] = time.time()-61
    start.save(run)
    assert setup.post("/api/festo/start/review", headers=ORIGIN, json=body).status_code == 200
    assert setup.post("/api/festo/start/preview", headers=ORIGIN).status_code == 200


def test_fault_and_readiness_and_request_are_independent():
    for field, value in [("Fault",True),("Ready",False),("StartRequested",True),
                         ("ConveyorForward",True),("ConveyorReverse",True),("Error",None),
                         ("Auto",True),("ManualEnabled",False)]:
        values = {"Press":normal(), "Turning":normal()}
        values["Press"][field] = value
        assert start.blockers(values)


def test_expired_preview_cannot_dispatch(setup):
    plan = setup.post("/api/festo/start/preview", headers=ORIGIN).json()
    plan["expires_at"] = 0
    start.save(plan)
    assert setup.post("/api/festo/start/execute", headers=ORIGIN,
                      json={"id":plan["id"], "approved":True, "onsite_safe":True}).status_code == 409


def test_binding_change_before_worker_prevents_connection(setup, monkeypatch):
    run = reserve()
    monkeypatch.setattr(start, "bindings", lambda: "changed")
    async def forbidden(*args, **kwargs):
        pytest.fail("Must not connect or write with changed bindings")
    monkeypatch.setattr(start, "conduct", forbidden)
    start.worker(run)
    assert start.fetch(run["id"])["status"] == "needs_review"
    assert not run["commands"]


def test_expired_dispatch_reservation_never_writes(setup, monkeypatch):
    run = reserve()
    run["dispatch_deadline"] = 0
    with pytest.raises(ValueError, match="reservation expired"):
        run_fake(monkeypatch, run)
    assert not run["commands"]


def test_audit_failure_prevents_connection(setup, monkeypatch):
    run = reserve()
    class BrokenLedger:
        def __init__(self, *args, **kwargs):
            raise OSError("disk unavailable")
    monkeypatch.setattr(start, "Ledger", BrokenLedger)
    start.worker(run)
    assert start.fetch(run["id"])["status"] == "needs_review"
    assert not run["commands"]


def test_session_checks_quality_and_writes_only_boolean_true():
    ua = pytest.importorskip("asyncua").ua
    writes = []
    class Node:
        async def read_data_type_as_variant_type(self):
            return ua.VariantType.Boolean
        async def write_value(self, value):
            writes.append(value.Value.Value)
    class Client:
        bad = False
        def get_node(self, node):
            return Node()
        async def read_attributes(self, nodes):
            return [ua.DataValue(ua.Variant(False, ua.VariantType.Boolean),
                                StatusCode=ua.StatusCode(ua.StatusCodes.BadNotConnected if self.bad else ua.StatusCodes.Good))
                    for _ in nodes]
    async def check():
        client = Client()
        session = start.Session(client)
        assert len(await session.sample()) == len(start.SIGNALS)
        client.bad = True
        with pytest.raises(Exception):
            await session.sample()
        await session.start()
    asyncio.run(check())
    assert writes == [True]


@pytest.mark.parametrize("result,rows", [
    ("NO_MOTION_OBSERVED", [{}]),
    ("NO_MOTION_OBSERVED_WITH_FAULT", [{"Fault":True}]),
    ("MOTION_OUTPUT_ACTIVE_AT_WINDOW_END", [{"ConveyorForward":True}]),
    ("MOTION_THEN_STOP_CAUSE_UNCONFIRMED", [{"ConveyorForward":True},{}]),
    ("START_REQUEST_STILL_HIGH_REVIEW_REQUIRED", [{"StartRequested":True}]),
])
def test_result_labels_do_not_claim_physical_success(result, rows):
    run = {"commands":{"Press":"accepted"}, "events":[{"kind":"command_intent", "station":"Press", "elapsed":0}]}
    for i, row in enumerate(rows):
        run["events"].append({"kind":"sample", "phase":"observation", "station":"Press",
                              "elapsed":i*.25, "values":{**normal(), **row}})
    assert start.classify(run, "Press") == result
    broken = copy.deepcopy(run)
    broken["events"].append({"kind":"read_error"})
    assert start.classify(broken, "Press") == "EVIDENCE_INCOMPLETE"

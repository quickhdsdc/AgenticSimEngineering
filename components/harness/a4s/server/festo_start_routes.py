"""Local, confirmed two-station start with durable evidence and no automatic retry.

Not a safety PLC or a production authorization service. No acknowledge, reset,
mode switch, stop or invented pulse is issued. Existing approved remote Start
semantics are a single Boolean true write; a latched request prevents another run.
"""
import asyncio
from contextlib import AsyncExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from fastapi import APIRouter, Body, HTTPException, Request

from ..aasctl import opc
from ..aasctl.client import BasyxClient, Endpoint
from ..ledger import Ledger
from .festo_assets import current_instance, instance_submodels, required_submodel
from . import opc_routes

router = APIRouter(prefix="/api/festo/start", tags=["Confirmed physical start"])
DIRECTORY = Path(__file__).resolve().parents[2] / "temp" / "festo-start"
OWNER = uuid.uuid4().hex
WINDOW = 15.0
INTERVAL = 0.25
TIMEOUT = 3.0
STATIONS = {"Press": "opc.tcp://192.0.2.10:4042",
            "Turning": "opc.tcp://192.0.2.10:4041"}
START_NODE = 'ns=3;s="dbVar"."RemoteControlAutomatic_Start"'
SIGNALS = {
    "Ready": 'ns=3;s="ApplicationModule_DB"."xReady"',
    "Fault": 'ns=3;s="dbVar"."Err"."xSF"',
    "Fault01": 'ns=3;s="dbVar"."Err"."xSF01"',
    "Fault0": 'ns=3;s="dbVar"."Err"."xSF0"',
    "Error": 'ns=3;s="ApplicationModule_DB"."xError"',
    "ConveyorForward": 'ns=3;s="xQA1_A1"',
    "ConveyorReverse": 'ns=3;s="xQA1_A2"',
    "Auto": 'ns=3;s="ApplicationModule_DB"."xAuto"',
    "ManualEnabled": 'ns=3;s="dbVar"."OpMode"."Man"."xEn"',
    "StartRequested": START_NODE,
}


@contextmanager
def database():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DIRECTORY / "runs.sqlite", timeout=5)
    db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS guard (id INTEGER PRIMARY KEY CHECK(id=1), run TEXT)")
    db.execute("INSERT OR IGNORE INTO guard VALUES (1, NULL)")
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def save(run):
    with database() as db:
        db.execute("INSERT OR REPLACE INTO records VALUES (?, ?)", (run["id"], json.dumps(run)))


def fetch(identifier):
    with database() as db:
        row = db.execute("SELECT payload FROM records WHERE id=?", (identifier,)).fetchone()
    if not row:
        raise HTTPException(404, "Start run not found")
    return json.loads(row[0])


def local_operator(request):
    # JSON + an explicit local Origin required for writes. This is a local-only
    # operator surface, not authentication. Never expose it via a reverse proxy.
    if request.client.host not in ("127.0.0.1", "::1", "testclient"):
        raise HTTPException(403, "Physical start is restricted to the local operator")
    if request.headers.get("origin") not in ("http://localhost:5173", "http://127.0.0.1:5173"):
        raise HTTPException(403, "Local operations UI origin required")


def bindings():
    """Resolve instance references, but require the commissioned station allowlist."""
    with BasyxClient(Endpoint.from_env()) as client:
        shell = current_instance(client)
        models = instance_submodels(client, shell)
        operational = required_submodel(models, shell, "OperationalData")["id"]
        services = required_submodel(models, shell, "services_delegated")["id"]
    ops = opc_routes.operations(services, None)["operations"]
    sources = opc_routes._sources_of(operational, None)
    for station, endpoint in STATIONS.items():
        action = next(o for o in ops if o["path"] == "DelegatedServices.Start" + station)
        if (action["endpoint"], action["node"], action["requiresSafetyApproval"]) != (endpoint, START_NODE, True):
            raise ValueError("Start delegation or approval declaration changed")
        for signal in ("Ready", "Fault", "ConveyorForward", "StartRequested"):
            src = next(s for s in sources if s["path"] == station + "." + signal)
            if (src["endpoint"], src["node"]) != (endpoint, SIGNALS[signal]):
                raise ValueError("Operational evidence mapping changed")
    contract = {"instance": shell["id"], "stations": STATIONS, "signals": SIGNALS,
                "operations": [o for o in ops if o["path"] in
                               ("DelegatedServices.StartPress", "DelegatedServices.StartTurning")]}
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def blockers(values):
    reasons = []
    for station, row in values.items():
        for signal in SIGNALS:
            if type(row.get(signal)) is not bool:
                reasons.append(f"{station}.{signal}: missing or non-Boolean evidence")
        if row.get("Fault") is not False:
            reasons.append(f"{station}: pre-existing collective fault")
        if row.get("Ready") is not True:
            reasons.append(f"{station}: module not ready")
        if row.get("ConveyorForward") is not False or row.get("ConveyorReverse") is not False:
            reasons.append(f"{station}: conveyor output already active or unknown")
        if row.get("StartRequested") is not False:
            reasons.append(f"{station}: Start request already high or unknown; no automatic reset")
        if row.get("Auto") is not False or row.get("ManualEnabled") is not True:
            reasons.append(f"{station}: not in the observed idle/manual baseline; inspect HMI")
    return reasons


class Session:
    def __init__(self, client):
        self.client = client

    async def sample(self):
        nodes = [self.client.get_node(n) for n in SIGNALS.values()]
        # Batch read with StatusCode validation; wall receipt time is not PLC time.
        data = await self.client.read_attributes(nodes)
        result = {}
        for name, value in zip(SIGNALS, data):
            value.StatusCode.check()
            result[name] = value.Value.Value
        if len(result) != len(SIGNALS):
            raise ValueError("Incomplete OPC read")
        return result

    async def start(self):
        from asyncua import ua
        node = self.client.get_node(START_NODE)
        if await node.read_data_type_as_variant_type() != ua.VariantType.Boolean:
            raise ValueError("Start is not Boolean")
        await node.write_value(ua.DataValue(ua.Variant(True, ua.VariantType.Boolean)))


def classify(run, station):
    samples = [e for e in run["events"] if e["kind"] == "sample" and e.get("station") == station]
    intent = next((e["elapsed"] for e in run["events"] if e["kind"] == "command_intent"
                   and e.get("station") == station), float("inf"))
    after = [e for e in samples if e["phase"] == "observation" and e["elapsed"] >= intent]
    command = run["commands"].get(station, "not_sent")
    if run.get("blockers"):
        return "BLOCKED_BEFORE_WRITE"
    if command != "accepted" or not after or any(e["kind"] == "read_error" for e in run["events"]):
        return "EVIDENCE_INCOMPLETE"
    if any(b["elapsed"] - a["elapsed"] > 1.0 for a, b in zip(samples, samples[1:])):
        return "EVIDENCE_INCOMPLETE"
    moved = False
    fault_after_motion = False
    for e in after:
        v = e["values"]
        if v["Fault"] and moved:
            fault_after_motion = True
        moved = moved or v["ConveyorForward"] or v["ConveyorReverse"]
    final = after[-1]["values"]
    if final["StartRequested"]:
        return "START_REQUEST_STILL_HIGH_REVIEW_REQUIRED"
    if fault_after_motion and not (final["ConveyorForward"] or final["ConveyorReverse"]):
        return "MOTION_THEN_FAULT_STOP"
    if final["ConveyorForward"] or final["ConveyorReverse"]:
        return "MOTION_OUTPUT_ACTIVE_AT_WINDOW_END"
    if not moved:
        return "NO_MOTION_OBSERVED_WITH_FAULT" if final["Fault"] else "NO_MOTION_OBSERVED"
    return "MOTION_THEN_STOP_CAUSE_UNCONFIRMED"


async def conduct(run, factory=None):
    """Persistent sessions; both baselines must pass before either write task exists."""
    factory = factory or (lambda endpoint: opc._connect(endpoint, TIMEOUT))
    began = time.monotonic()
    with Ledger(DIRECTORY / (run["id"] + ".ledger.db"), run["id"], actor="local-web-operator") as ledger:
        def event(kind, **payload):
            row = {"kind": kind, "at": time.time(), "elapsed": round(time.monotonic()-began, 4), **payload}
            ledger.append("festo.start", row, subject="festo-start:" + run["binding"])
            run["events"].append(row)
            save(run)

        event("approval", targets=STATIONS, value=True, window=WINDOW,
              protocol="single true write; no automatic reset/stop/retry; non-atomic pair")
        async with AsyncExitStack() as stack:
            sessions = {}
            for station, endpoint in STATIONS.items():
                sessions[station] = Session(await asyncio.wait_for(
                    stack.enter_async_context(factory(endpoint)), TIMEOUT + 2))

            async def sample(station, phase):
                try:
                    values = await asyncio.wait_for(sessions[station].sample(), TIMEOUT)
                    event("sample", station=station, phase=phase, values=values)
                    return values
                except Exception as exc:
                    event("read_error", station=station, phase=phase, error=type(exc).__name__)
                    return {}

            # Two snapshots are recorded before dispatch. No root-cause guesses.
            for _ in range(2):
                rows = await asyncio.gather(*(sample(s, "baseline") for s in STATIONS))
                run["blockers"] = blockers(dict(zip(STATIONS, rows)))
                if run["blockers"]:
                    event("blocked", reasons=run["blockers"])
                    return
                await asyncio.sleep(INTERVAL)

            async def dispatch(station):
                with database() as db:
                    active = db.execute("SELECT run FROM guard WHERE id=1").fetchone()[0]
                if active != run["id"] or time.time() > run["dispatch_deadline"]:
                    raise ValueError("Start reservation expired; no write permitted")
                event("command_intent", station=station, node=START_NODE, value=True)
                run["commands"][station] = "unknown"
                save(run)  # crash after this point must not permit a replay
                try:
                    await asyncio.wait_for(sessions[station].start(), TIMEOUT)
                    run["commands"][station] = "accepted"
                    event("command_ack", station=station, note="OPC write accepted, not proof of motion")
                except Exception as exc:
                    event("command_uncertain", station=station, error=type(exc).__name__)

            # Start observation before dispatch; it continues while writes settle.
            async def observe():
                deadline = time.monotonic() + WINDOW
                while time.monotonic() < deadline:
                    await asyncio.gather(*(sample(s, "observation") for s in STATIONS))
                    await asyncio.sleep(INTERVAL)

            await asyncio.gather(observe(), *(dispatch(s) for s in STATIONS))


def worker(run):
    try:
        if bindings() != run["binding"]:
            raise ValueError("AAS bindings changed; nothing dispatched")
        async def bounded():
            await asyncio.wait_for(conduct(run), 60)
        asyncio.run(bounded())
        run["results"] = {s: classify(run, s) for s in STATIONS}
        accepted = sum(v == "accepted" for v in run["commands"].values())
        run["pair_outcome"] = ("BOTH_WRITES_ACKNOWLEDGED_NOT_PROOF_OF_MOTION" if accepted == 2 else
                               "PARTIAL_WRITE_ACKNOWLEDGEMENT_REVIEW_REQUIRED" if accepted == 1 else
                               "NO_WRITES_SENT" if not run["commands"] else "WRITE_OUTCOMES_UNKNOWN")
        uncertain = any(v in ("EVIDENCE_INCOMPLETE", "START_REQUEST_STILL_HIGH_REVIEW_REQUIRED")
                        for v in run["results"].values())
        run["status"] = "needs_review" if uncertain else "finished"
    except Exception as exc:
        run["status"] = "needs_review"
        run["error"] = type(exc).__name__ + ": observation or audit interrupted; inspect the HMI before any further action"
    finally:
        try:
            with Ledger(DIRECTORY / (run["id"] + ".ledger.db"), run["id"], actor="local-web-operator") as ledger:
                ledger.append("festo.start.result", {"status": run["status"], "results": run["results"],
                                                     "commands": run["commands"]}, subject="festo-start:" + run["binding"])
        except Exception:
            run["status"] = "needs_review"
            run["error"] = "Final audit unavailable; no automatic retry"
        save(run)
        if run["status"] == "finished":
            with database() as db:
                db.execute("UPDATE guard SET run=NULL WHERE run=?", (run["id"],))


@router.post("/preview")
def preview(request: Request):
    local_operator(request)
    try:
        binding = bindings()
    except Exception as exc:
        raise HTTPException(409, "Cannot validate commissioned AAS Start/evidence mappings") from exc
    with database() as db:
        if db.execute("SELECT run FROM guard WHERE id=1").fetchone()[0]:
            raise HTTPException(409, "A start is active or requires review; inspect the last run")
    run = {"id": uuid.uuid4().hex, "owner": OWNER, "created_at": time.time(),
           "expires_at": time.time()+120, "status": "preview", "binding": binding,
           "targets": STATIONS, "node": START_NODE, "events": [], "commands": {},
           "results": {}, "window_seconds": WINDOW,
           "warning": "PHYSICAL MOTION: both stations, non-atomic single true writes. No automatic fault reset, stop or retry. Observation ending does not stop equipment."}
    save(run)
    return run


@router.post("/execute")
def execute(request: Request, body: dict = Body(...)):
    local_operator(request)
    if body.get("approved") is not True or body.get("onsite_safe") is not True:
        raise HTTPException(403, "Explicit approval and onsite safety confirmation required")
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT payload FROM records WHERE id=?", (str(body.get("id", "")),)).fetchone()
        run = json.loads(row[0]) if row else None
        if not run or run["status"] != "preview" or run["expires_at"] < time.time() or run["owner"] != OWNER:
            raise HTTPException(409, "Preview consumed, expired or server restarted; never replay")
        if db.execute("SELECT run FROM guard WHERE id=1").fetchone()[0]:
            raise HTTPException(409, "Another start is active or needs review")
        run["status"] = "running"
        run["dispatch_deadline"] = time.time() + 30
        run["confirmation"] = {"approved": True, "onsite_safe": True, "at": time.time()}
        db.execute("UPDATE records SET payload=? WHERE id=?", (json.dumps(run), run["id"]))
        db.execute("UPDATE guard SET run=? WHERE id=1", (run["id"],))
    try:
        threading.Thread(target=worker, args=(run,), daemon=True).start()
    except Exception as exc:
        run["status"] = "needs_review"
        run["error"] = "Start worker could not be launched; inspect latest run before any new attempt"
        save(run)
        raise HTTPException(503, run["error"]) from exc
    return {"id": run["id"], "status": "running"}


@router.get("/latest")
def latest():
    with database() as db:
        active = db.execute("SELECT run FROM guard WHERE id=1").fetchone()[0]
        row = db.execute("SELECT payload FROM records WHERE json_extract(payload, '$.status') != 'preview' ORDER BY rowid DESC LIMIT 1").fetchone()
    run = fetch(active) if active else json.loads(row[0]) if row else None
    if run and run["status"] == "running" and run["owner"] != OWNER:
        run["status"] = "interrupted"
    return {"run": run}


@router.post("/review")
def review(request: Request, body: dict = Body(...)):
    local_operator(request)
    run = fetch(str(body.get("id", "")))
    if body.get("onsite_reviewed") is not True or not (run["status"] == "needs_review" or
            (run["status"] == "running" and run["owner"] != OWNER
             and time.time() > run.get("dispatch_deadline", float("inf")) + 60)):
        raise HTTPException(409, "Only an interrupted/uncertain run can be acknowledged after onsite inspection")
    with Ledger(DIRECTORY / (run["id"] + ".ledger.db"), run["id"], actor="local-web-operator") as ledger:
        ledger.append("festo.start.review", {"onsite_reviewed": True, "no_plc_write": True}, subject="festo-start:" + run["binding"])
    run["status"] = "reviewed"
    save(run)
    with database() as db:
        db.execute("UPDATE guard SET run=NULL WHERE run=?", (run["id"],))
    return {"run": run}

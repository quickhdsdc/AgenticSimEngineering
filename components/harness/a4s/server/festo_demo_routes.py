"""Local, operator-driven recovery demo using the existing MES action contracts.

Read-only analysis and explicit, single-use approval of server-generated actions.
This does not run an LLM, open OPC, or add a second business database. Agent chat
continues to use shell -> mes -> its own Harness gate. These unauthenticated local
UI routes are not a production authorization service.
"""

import hashlib
import json
from pathlib import Path
import shlex
import threading
import time
import uuid

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from ..aasctl.client import BasyxClient, Endpoint
from ..ledger import Ledger
from .festo_assets import current_instance, instance_submodels, required_submodel
from ..machines.festo_cp_lab import demo_monitor, knowledge
from ..machines.festo_cp_lab import pdf_knowledge
from ..machines.festo_cp_lab.mes_binding import current_demo_orders, resolve_lines
from ..mesctl.client import Config, MesClient, MesError
from ..mesctl.commands import build_parser
from ..mesctl.execution import ExecutionContext
from ..mesctl.executor import execute

router = APIRouter(prefix="/api/demo/festo", tags=["Festo maintenance recovery"])
AUDIT_PATH = Path(__file__).resolve().parents[2] / "temp" / "festo-demo" / "ledger.db"
LOCK = threading.RLock()
PLANS = {}
COMMIT_LOCK = threading.Lock()
TITLE = "A4S-381 simulated maintenance review"
ALLOWED = {"create_issue", "move_upcoming", "acknowledge_issue", "resolve_issue", "close_issue",
           "return_upcoming", "accept_affected"}


def _run(client, words):
    return execute(build_parser().parse_args(words), client)


def _fields(submodel):
    return {el["idShort"]: el.get("value") for el in submodel.get("submodelElements", [])
            if el.get("modelType") == "Property"}


def snapshot():
    with BasyxClient(Endpoint.from_env()) as aas:
        shell = current_instance(aas)
        models = instance_submodels(aas, shell)
        context = _fields(required_submodel(models, shell, "ProductionContext"))
        maintenance = required_submodel(models, shell, "MaintenanceInstructions")
    config = Config.from_env()
    # Reject an untrusted AAS URL before opening an authenticated MES client.
    if context.get("MESBaseUrl", "").rstrip("/") != config.endpoint:
        raise HTTPException(409, "AAS MES reference differs from the configured trusted MES service")
    with MesClient(config) as mes:
        line, backup = resolve_lines(context, mes)
        order_a, order_b = current_demo_orders(mes, line, backup)
        # Resolve latest details from MES, since the list may omit fields.
        order_a = _run(mes, ["order", "show", str(order_a["id"])])["data"]
        order_b = _run(mes, ["order", "show", str(order_b["id"])])["data"]
        issues = mes.rows("/api/v1/issues", work_order_id=order_a["id"])
        issues = [issue for issue in issues if issue.get("title") == TITLE]
    return {
        "asset": {"id": shell["id"], "idShort": shell["idShort"], "assetKind": "Instance"},
        "operational_id": models.get("OperationalData", {}).get("id"),
        "maintenance": maintenance, "context": context, "order_a": order_a, "order_b": order_b,
        "issues": issues, "line": line, "backup": backup, "mes_url": config.endpoint,
        "observed_at": time.time(), "evidence_mode": "SIMULATION", "plc_writes": False,
        "limitations": ["Scenario fault and backup resource are simulated, not live OPC evidence.",
                        "Orders are accepted/planned, not a demonstrated running production batch.",
                        "No validated physical capacity, RUL prediction or automatic repair."],
    }


@router.get("/asset")
def read_asset():
    try:
        with BasyxClient(Endpoint.from_env()) as aas:
            shell = current_instance(aas)
            models = instance_submodels(aas, shell)
        return {"asset": shell, "endpoint": Endpoint.from_env().base_url,
                "operational_id": models.get("OperationalData", {}).get("id")}
    except Exception as exc:
        raise _fail(exc) from exc


@router.get("/knowledge")
def retrieve_knowledge(query: str = "pressure leakage maintenance"):
    try:
        return {"retrieval": "PDF text / local lexical", "chunks": knowledge.search(query[:500], 4)}
    except (ImportError, FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(503, "PDF retrieval unavailable; check documents dependencies and regenerate demo PDFs") from exc


@router.get("/knowledge/documents")
def knowledge_documents():
    try:
        return {"documents": pdf_knowledge.catalog()}
    except (ImportError, FileNotFoundError, ValueError, IndexError) as exc:
        raise HTTPException(503, "PDF library unavailable; check documents dependencies and regenerate demo PDFs") from exc


@router.get("/knowledge/documents/{document_id}")
def knowledge_document(document_id: str):
    try:
        path = pdf_knowledge.document_path(document_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, "Document not available") from exc
    return FileResponse(path, media_type="application/pdf")


@router.get("/monitor")
def read_monitor():
    return demo_monitor.observe()


@router.post("/monitor/start")
def start_monitor(body: dict = Body(...)):
    if body.get("mode") != "SIMULATION":
        raise HTTPException(422, "This generator supports explicit SIMULATION only")
    return demo_monitor.start(normal_only=body.get("normal_only") is True)


@router.post("/monitor/inject")
def inject_monitor(body: dict = Body(...)):
    if body.get("mode") != "SIMULATION":
        raise HTTPException(422, "Simulation confirmation required")
    try:
        return demo_monitor.inject()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/monitor/recover")
def recover_monitor(body: dict = Body(...)):
    if body.get("mode") != "SIMULATION":
        raise HTTPException(422, "Simulation confirmation required")
    return demo_monitor.recover()


@router.post("/monitor/stop")
def stop_monitor():
    return demo_monitor.stop()


def _state_digest(state):
    # Conservative approval freshness. Volatile observation timestamps are not
    # part of the comparison; current MES/AAS resources and references are.
    fields = {key: state[key] for key in
              ("asset", "context", "maintenance", "order_a", "order_b", "issues", "line", "backup")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def action_words(action, state):
    if action not in ALLOWED:
        raise HTTPException(422, "Unknown recovery action")
    a, b = state["order_a"], state["order_b"]
    open_issues = [i for i in state["issues"] if i["status"] != "CLOSED"]
    if action == "accept_affected":
        if open_issues or a["status"] != "PENDING":
            raise HTTPException(409, "Close the simulated issue and verify a PENDING order first")
        return ["order", "accept", str(a["id"])]
    if action == "create_issue":
        if open_issues:
            raise HTTPException(409, "A demo issue already exists; read it instead of duplicating it")
        if a["status"] not in ("PENDING", "ACCEPTED", "IN_PROGRESS", "PAUSED", "BLOCKED"):
            raise HTTPException(409, "Affected order is not eligible for this demo")
        return ["issue", "create", "--order", str(a["id"]), "--type", "TOOL_FAILURE",
                "--title", TITLE, "--description",
                "SIMULATION: operator-triggered maintenance scenario, not a PLC fault observation. "
                f"Asset: {state['asset']['id']}. Maintenance profile: {state['maintenance']['id']}. No physical repair performed."]
    if action in ("move_upcoming", "return_upcoming"):
        returning = action == "return_upcoming"
        if returning and open_issues:
            raise HTTPException(409, "Close the simulated issue before returning the planned order")
        if not returning and not open_issues:
            raise HTTPException(409, "Create the maintenance issue first; then reassess production impact")
        target = str(state["line"]["id"] if returning else state["backup"]["id"])
        if b.get("line_id") == int(target):
            raise HTTPException(409, "Upcoming order is already on the target line; no duplicate move needed")
        if not b.get("planned_start_at") or not b.get("planned_end_at"):
            raise HTTPException(409, "Upcoming order has no explicit time window")
        # Normalize MES wall-time values using its configured timezone. Do not
        # guess that an offset-free upstream value is UTC.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        zone = Config.from_env().server_timezone
        if not zone:
            raise HTTPException(409, "Configure A4S_MES_TIMEZONE first")

        def timestamp(value):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo(zone))
            return parsed.isoformat()

        return ["schedule", "move", str(b["id"]), "--line", target,
                "--start", timestamp(b["planned_start_at"]), "--end", timestamp(b["planned_end_at"])]
    if len(open_issues) != 1:
        raise HTTPException(409, "Expected exactly one open demo issue; inspect MES")
    issue = open_issues[0]
    required = {"acknowledge_issue": "OPEN", "resolve_issue": "ACKNOWLEDGED", "close_issue": "RESOLVED"}
    if issue["status"] != required[action]:
        raise HTTPException(409, f"Expected issue status {required[action]}; refresh current state")
    verb = action.removesuffix("_issue")
    words = ["issue", verb, str(issue["id"])]
    if action == "resolve_issue":
        words += ["--notes", "Operator ended the SIMULATED maintenance scenario. "
                  "No physical repair is claimed; no PLC command was executed. Read order_after."]
    return words


def _fail(exc):
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, MesError):
        return HTTPException(exc.status or 502, str(exc))
    # Avoid exposing credential-bearing exception bodies / arbitrary documents.
    return HTTPException(502, f"Cannot read demo dependencies ({type(exc).__name__}); check setup and service health")


@router.get("")
def read_demo():
    try:
        return snapshot()
    except Exception as exc:
        raise _fail(exc) from exc


@router.post("/preview")
def preview(body: dict = Body(...)):
    if body.get("scenario") != "simulated-maintenance":
        raise HTTPException(422, "Select the explicitly simulated maintenance scenario")
    try:
        state = snapshot()
        words = action_words(body.get("action"), state)
        with MesClient(Config.from_env()) as client:
            result = _run(client, words + ["--dry-run"])
        plan_id = uuid.uuid4().hex
        plan = {"id": plan_id, "action": body["action"], "words": words,
                "command": "mes " + shlex.join(words), "state_digest": _state_digest(state),
                "asset_id": state["asset"]["id"], "preview": result, "expires_at": time.time() + 300,
                "note": "One operator-approved MES action; no LLM execution or PLC write."}
        with LOCK:
            for key in list(PLANS):
                if PLANS[key]["expires_at"] < time.time():
                    del PLANS[key]
            if len(PLANS) >= 100:
                raise HTTPException(429, "Too many pending previews; wait for expiry")
            PLANS[plan_id] = plan
        return {key: value for key, value in plan.items() if key not in ("words", "state_digest")}
    except Exception as exc:
        raise _fail(exc) from exc


@router.post("/execute")
def commit(body: dict = Body(...)):
    if not COMMIT_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another demo execution is in progress; refresh before trying again")
    try:
        return _commit(body)
    finally:
        COMMIT_LOCK.release()


def _commit(body):
    if body.get("approved") is not True:
        raise HTTPException(403, "Explicit operator approval is required")
    # Atomically consume the preview even if execution later fails. No automatic
    # replay on timeout, browser retry, double click or read-back failure.
    with LOCK:
        plan = PLANS.pop(str(body.get("plan_id", "")), None)
    if not plan or plan["expires_at"] < time.time():
        raise HTTPException(409, "Preview expired, consumed or server restarted; refresh and preview again")
    session = "festo-demo-" + plan["id"]
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Ledger(AUDIT_PATH, session, actor="web-operator") as ledger:
            ledger.append("approval", {"approved": True, "plan_id": plan["id"],
                                       "command": plan["command"], "mode": "SIMULATION"}, subject=plan["asset_id"])
            state = snapshot()
            if _state_digest(state) != plan["state_digest"]:
                ledger.append("mes.execution", {"phase": "preflight_failed", "reason": "State changed"})
                raise HTTPException(409, "MES/AAS state changed since preview; reassess and approve a fresh plan")
            with MesClient(Config.from_env()) as client:
                check = _run(client, plan["words"] + ["--dry-run"])
                if check["request_fingerprint"] != plan["preview"]["request_fingerprint"]:
                    raise HTTPException(409, "Prepared request changed; preview again")
                context = ExecutionContext(session, str(uuid.uuid4()))
                ledger.append("mes.execution", {"phase": "started", **context.as_dict(),
                                                "command": plan["command"]})
                try:
                    result = execute(build_parser().parse_args(plan["words"]), client, context=context)
                except MesError as exc:
                    ledger.append("mes.write", {"error": str(exc), "execution": getattr(exc, "execution", {})})
                    raise
                ledger.append("mes.execution", {"phase": "finished", **result["execution"]})
                ledger.append("mes.write", result, subject=build_parser().parse_args(plan["words"]).action.subject(
                    build_parser().parse_args(plan["words"])))
                return {"result": result, "audit_session": session,
                        "note": "MES read-back only; does not verify physical recovery."}
    except Exception as exc:
        raise _fail(exc) from exc


@router.get("/audit")
def audit():
    if not AUDIT_PATH.exists():
        return {"events": []}
    with Ledger(AUDIT_PATH, "festo-demo-read", actor="web-operator") as ledger:
        rows = list(reversed(ledger.read(limit=100, newest_first=True)))
    return {"events": [{"seq": row.seq, "ts": row.ts, "session": row.session_id,
                        "kind": row.kind, "subject": row.subject, "payload": row.payload} for row in rows]}

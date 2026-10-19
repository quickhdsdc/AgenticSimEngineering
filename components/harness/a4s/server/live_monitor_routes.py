"""Explicitly armed backend observation; browser handoff to the existing Agent."""

import os
import json
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from ..aasctl import opc
from ..aasctl.client import AasHttpError, BasyxClient, Endpoint
from ..machines.festo_cp_lab.live_monitor import Monitor
from ..machines.festo_cp_lab.opc_simulator import controller as simulator
from .festo_assets import identity, is_festo
from .opc_routes import _walk

router = APIRouter(prefix="/api/festo/live-monitor")


@router.get("/simulator")
def simulator_status():
    return simulator.status()


@router.post("/simulator/start")
def simulator_start():
    try:
        return simulator.start()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/simulator/state")
def simulator_state(body: dict = Body(...)):
    try:
        return simulator.set_mode(body.get("mode"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/simulator/stop")
def simulator_stop():
    return simulator.stop()


def read_target(target):
    with BasyxClient(Endpoint.from_env(target["endpoint"])) as aas:
        shell = aas.get_shell(target["asset_id"])
        if not is_festo(shell):
            raise ValueError("Target must be an existing Festo Instance")
        models = []
        for ref in ([] if target.get("source") == "simulator" else shell.get("submodels", [])):
            for key in ref.get("keys", []):
                if key.get("type") != "Submodel":
                    continue
                try:
                    models.append(aas.get_submodel(key["value"]))
                except AasHttpError as exc:
                    if exc.status != 404:
                        raise
    if target.get("source") == "simulator":
        from ..machines.festo_cp_lab.fault_profile import models as simulation_models

        models = simulation_models({"submodelElements": []}, target["asset_id"])[:1]
    models = [
        m for m in models if m.get("idShort") == "OperationalData" and m.get("kind") != "Template"
    ]
    if len(models) != 1:
        raise ValueError("Expected exactly one OperationalData instance")
    sources = []
    _walk(models[0], "", sources)
    if target.get("source") == "simulator":
        from ..machines.festo_cp_lab.opc_simulator import ENDPOINT
        if not sources or any(item["endpoint"].rstrip("/") != ENDPOINT.rstrip("/") for item in sources):
            raise ValueError("Test profile must bind only the isolated loopback OPC server")
    groups = {}
    for item in sources:
        groups.setdefault(item["endpoint"], []).append(item)
    values, failures = [], []
    for endpoint, items in groups.items():
        try:
            evidence = opc.read_evidence(endpoint, [i["node"] for i in items], timeout=8)
            values.extend({**item, **value} for item, value in zip(items, evidence))
        except opc.OpcError:
            failures.append({"endpoint": endpoint, "error": "OPC read unavailable"})
    return {"submodel": models[0]["id"], "source": target.get("source", "live"),
            "binding_origin": "runtime simulation profile" if target.get("source") == "simulator" else "AAS",
            "values": values, "failed": failures}


service = Monitor(Path(os.getenv("A4S_LIVE_MONITOR_DB", "temp/festo-live/monitor.db")), read_target)


def target(asset_id, endpoint, source="live"):
    if source not in ("live", "simulator"):
        raise HTTPException(422, "Unknown monitoring source")
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise HTTPException(422, "Explicit asset_id is required")
    selected = identity(endpoint, asset_id)
    return {**selected, "source": source} if source == "simulator" else selected


@router.get("")
def status(asset_id: str, endpoint: str, source: str = "live"):
    return service.status(target(asset_id, endpoint, source))


@router.post("/start")
def start(body: dict = Body(...)):
    selection = target(body.get("asset_id"), body.get("endpoint"), body.get("source", "live"))
    try:
        # Validate identity without reading MES or sending a PLC command.
        with BasyxClient(Endpoint.from_env(selection["endpoint"])) as aas:
            shell = aas.get_shell(selection["asset_id"])
            if not is_festo(shell):
                raise ValueError("Select a Festo Instance")
        return service.start(selection)
    except Exception as exc:
        raise HTTPException(
            409, f"Cannot arm monitor ({type(exc).__name__}); check target or active readers"
        ) from exc


@router.post("/stop")
def stop(body: dict = Body(...)):
    return service.stop(target(body.get("asset_id"), body.get("endpoint"), body.get("source", "live")))


@router.post("/claim")
def claim(body: dict = Body(...)):
    event = service.claim(target(body.get("asset_id"), body.get("endpoint"), body.get("source", "live")), body.get("event_id"))
    if not event:
        raise HTTPException(
            409, "Event already reserved, not a fresh transition, stale or no longer active"
        )
    return {"event": event, "request": investigation_request(event), "delivery": "reserved_not_completed"}


def investigation_request(event):
    """Same operational notification for every signal source; procedure lives in a skill."""
    def frame(sample):
        if not sample:
            return None
        return {"observed_at": sample.get("observed_at"), "values": [
            {key: row.get(key) for key in ("path", "value", "dataType", "status", "source_timestamp")}
            for row in sample.get("values", [])]}
    target = event["target"]
    mode = target.get("source", "live")
    evidence = {"source": mode, "event_id": event["id"], "first_observed_at": event.get("first_observed_at"),
                "confirmed_at": event["at"], "before": frame(event.get("before")),
                "after": frame(event.get("after"))}
    return ("Maintenance investigation: fault transition detected.\n"
            f"Asset: {target['asset_id']}\nAAS repository: {target['endpoint']}\n" +
            ("SIMULATED OPC evidence for this instance. Use its AAS for MES business references; "
             "the physical OperationalData mappings do not define these simulated signals. "
             "Retrieve the pneumatic simulation guide for signal meanings. No physical fault is asserted.\n"
             if mode == "simulator" else "") +
            "Investigate this event and propose a maintenance plan and production impact assessment.\n"
            "Evidence: " + json.dumps(evidence))

"""Provision only the named Festo instance and uniquely labelled MES demo orders.

Default is read-only planning; --apply is an explicit operator provisioning step.
Existing submodels/orders are never overwritten. No PLC or cloud model calls.
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import uuid

from a4s.aasctl.client import AasHttpError, BasyxClient, Endpoint
from a4s.kernel.config import load_env
from a4s.ledger import Ledger
from a4s.machines.festo_cp_lab import maintenance as model
from a4s.mesctl.client import Config, MesClient, MesError
from a4s.mesctl.commands import build_parser
from a4s.mesctl.executor import execute


def attach_once(client, shell_id, submodel, ledger):
    try:
        existing = client.get_submodel(submodel["id"])
    except AasHttpError as exc:
        if exc.status != 404:
            raise
        client.create_submodel(submodel)
        ledger.append("aas.create", {"id": submodel["id"]}, subject=submodel["id"])
    else:
        if existing != submodel:
            raise RuntimeError(f"Existing submodel differs; refusing to overwrite {submodel['id']}")
    refs = client.list_submodel_refs(shell_id)
    if not any(key.get("value") == submodel["id"] for ref in refs for key in ref.get("keys", [])):
        client.add_submodel_ref(shell_id, {"type": "ModelReference", "keys": [
            {"type": "Submodel", "value": submodel["id"]}]})
        ledger.append("aas.write", {"attached": submodel["id"]}, subject=shell_id)
    return client.get_submodel(submodel["id"])


def main():
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--template", type=Path, default=Path(
        "docs/IDTA_02018_Template_MaintenanceInstructions.json"))
    args = parser.parse_args()
    profile = model.build(args.template)
    session = "festo-provision-" + uuid.uuid4().hex[:12]
    with BasyxClient(Endpoint.from_env()) as aas, MesClient(Config.from_env()) as mes:
        shell = aas.get_shell(model.SHELL_ID)
        if shell.get("assetInformation", {}).get("assetKind") != "Instance":
            raise RuntimeError("Target must be an Instance; no blueprint is modified")
        line = mes.resolve("lines", "A4S-FESTO", "code")
        backup = mes.resolve("lines", "A4S-BACKUP", "code")
        products = mes.rows(f"/api/v1/lines/{line}/product-types")
        product = next(p["id"] for p in products if p["code"] == "A4S-DEMO-PART")
        try:
            existing_context = aas.get_submodel(model.CONTEXT_ID)
        except AasHttpError as exc:
            if exc.status != 404:
                raise
            existing_context = None
        if existing_context:
            print(json.dumps({"already_provisioned": True, "context": model.CONTEXT_ID,
                              "note": "No new records created; inspect the existing demo."}))
            return
        plan = {"shell": model.SHELL_ID, "maintenance": model.MAINTENANCE_ID,
                "line_id": line, "backup_line_id": backup, "product_type_id": product,
                "creates": "Two accepted, not running orders; two scheduled placements; two AAS submodels",
                "mode": "SIMULATION", "plc_writes": False}
        if not args.apply:
            print(json.dumps({"dry_run": True, **plan}, indent=2))
            return
        evidence_dir = Path("temp/festo-demo") / session
        evidence_dir.mkdir(parents=True, exist_ok=False)
        with Ledger(evidence_dir / "ledger.db", session, actor="operator-provisioning") as ledger:
            ledger.append("approval", {"explicit_apply": True, "plan": plan}, subject=model.SHELL_ID)

            def run(words):
                parsed = build_parser().parse_args(words)
                ledger.append("mes.execution", {"phase": "started", "command": words})
                try:
                    result = execute(parsed, mes)
                except MesError as exc:
                    ledger.append("mes.write", {"error": str(exc), "execution": getattr(exc, "execution", {})})
                    raise
                ledger.append("mes.write", result, subject=parsed.action.subject(parsed))
                if parsed.action.mutates and not result.get("verified"):
                    raise RuntimeError("Write accepted but unverified. Inspect MES; do not replay.")
                return result

            created = []
            start = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
                hour=8, minute=0, second=0, microsecond=0)
            for index, suffix in enumerate(("A", "B")):
                order_no = "A4S-381-" + session.rsplit("-", 1)[-1].upper() + "-" + suffix
                result = run(["order", "create", "--order-no", order_no, "--line", str(line),
                              "--product-type-id", str(product), "--planned-qty", "10",
                              "--description", "Smartphone housing stamping + turning-over DEMO. "
                              "Fictional pneumatic leak and backup; no running batch or physical production claim."])
                order_id = result["read_back"]["data"]["id"]
                created.append(order_id)
                run(["order", "accept", str(order_id)])
                begin = start + timedelta(hours=index)
                run(["schedule", "move", str(order_id), "--line", str(line),
                     "--start", begin.isoformat(), "--end", (begin + timedelta(minutes=30)).isoformat()])
            attach_once(aas, model.SHELL_ID, profile, ledger)
            context = model.production_context("A4S-FESTO", "A4S-BACKUP", "A4S-DEMO-PART", mes.config.endpoint)
            attach_once(aas, model.SHELL_ID, context, ledger)
            print(json.dumps({**plan, "order_ids": created, "ledger": str(evidence_dir / "ledger.db"),
                              "verified": True}, indent=2))


if __name__ == "__main__":
    main()

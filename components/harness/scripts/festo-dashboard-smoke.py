"""Explicitly exercise the provisioned MES demo, then restore its planned baseline.

Creates a retained, closed simulated issue and audit evidence. Never touches OPC
or a model provider. Default is a read-only readiness check; --apply approves this
bounded test workflow on the two AAS-linked A4S-381 demo orders only.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

import httpx

from a4s.kernel.config import load_env
from a4s.ledger import Ledger
from a4s.mesctl.client import Config, MesClient
from a4s.mesctl.commands import build_parser
from a4s.mesctl.executor import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    load_env()
    with httpx.Client(base_url="http://localhost:8000", timeout=60) as client:
        def call(method, path="", **kwargs):
            response = client.request(method, "/api/demo/festo" + path, **kwargs)
            response.raise_for_status()
            return response.json()

        initial = call("GET")
        a, b = initial["order_a"], initial["order_b"]
        if not all(o["order_no"].startswith("A4S-381-") and o["status"] == "ACCEPTED" for o in (a, b)):
            raise RuntimeError("Expected two accepted, explicitly provisioned demo orders")
        if any(i["status"] != "CLOSED" for i in initial["issues"]):
            raise RuntimeError("An incident is already active; inspect it instead of running smoke")
        if b["line_id"] == initial["backup"]["id"]:
            raise RuntimeError("Upcoming order already moved; do not overwrite that decision")
        if not args.apply:
            print(json.dumps({"ready": True, "order_ids": [a["id"], b["id"]], "writes": False}))
            return
        for action in ("create_issue", "move_upcoming", "acknowledge_issue", "resolve_issue", "close_issue"):
            plan = call("POST", "/preview", json={"scenario": "simulated-maintenance", "action": action})
            result = call("POST", "/execute", json={"plan_id": plan["id"], "approved": True})
            if not result["result"]["verified"]:
                raise RuntimeError("Accepted but not verified; inspect before any further action")
            state = call("GET")
            print(json.dumps({"action": action, "verified": True, "audit_session": result["audit_session"],
                              "order_a": state["order_a"]["status"], "order_b_line": state["order_b"]["line_id"]}), flush=True)
        # Only restore the deliberate test changes. Retain the closed issue and
        # its audit; this is not deleting evidence or claiming physical recovery.
        state = call("GET")
        if state["order_a"]["status"] != "PENDING" or state["order_b"]["line_id"] != initial["backup"]["id"]:
            raise RuntimeError("Unexpected final state; do not blindly restore")
        config = Config.from_env()

        def timestamp(value):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return (dt.replace(tzinfo=ZoneInfo(config.server_timezone)) if dt.tzinfo is None else dt).isoformat()

        session = "festo-smoke-restore-" + uuid.uuid4().hex[:12]
        with MesClient(config) as mes, Ledger(Path("temp/festo-demo/ledger.db"), session,
                                              actor="operator-smoke") as ledger:
            for words in (["schedule", "move", str(b["id"]), "--line", str(b["line_id"]),
                           "--start", timestamp(b["planned_start_at"]), "--end", timestamp(b["planned_end_at"])],
                          ["order", "accept", str(a["id"]) ]):
                preview = execute(build_parser().parse_args(words + ["--dry-run"]), mes)
                ledger.append("approval", {"explicit_smoke_apply": True, "preview": preview})
                result = execute(build_parser().parse_args(words), mes)
                ledger.append("mes.write", result)
                if not result["verified"]:
                    raise RuntimeError("Restoration unverified; do not replay")
        final = call("GET")
        assert final["order_a"]["status"] == "ACCEPTED"
        assert final["order_b"]["line_id"] == b["line_id"]
        print(json.dumps({"baseline_restored": True, "closed_issue_retained": True, "plc_writes": False}))


if __name__ == "__main__":
    main()

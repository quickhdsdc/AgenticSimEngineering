"""Live CLI smoke test against the isolated local demo; leaves labelled records.

Uses real subprocess commands, then the harness shell/approval/Ledger path.
No LLM or PLC is involved. Each run creates a fresh A4S-SMOKE order.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

from a4s.kernel.config import load_env
from a4s.mesctl.client import Config

ROOT = Path(__file__).resolve().parents[1]
TRACE_ID = "smoke-" + uuid.uuid4().hex
ATTEMPTS = set()


def cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "a4s.mesctl", *map(str, args), "--trace-id", TRACE_ID],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    if result.returncode:
        raise RuntimeError(f"mes {' '.join(map(str, args))}: {result.stderr or result.stdout}")
    data = json.loads(result.stdout)
    assert data["execution"]["trace_id"] == TRACE_ID
    assert data["execution"]["execution_id"] not in ATTEMPTS
    ATTEMPTS.add(data["execution"]["execution_id"])
    print(f"PASS mes {' '.join(map(str, args))}", flush=True)
    return data


def main():
    load_env(str(ROOT / ".env"))
    if Config.from_env().endpoint != "http://localhost:8092":
        raise SystemExit("This mutating smoke test is restricted to the local demo endpoint")
    cli("health")
    assert cli("auth", "whoami")["data"]["username"] == "a4s-agent"
    product = next(p for p in cli("product", "ls")["data"] if p["code"] == "A4S-DEMO-PART")
    number = "A4S-SMOKE-" + uuid.uuid4().hex[:10].upper()
    create = (
        "order",
        "create",
        "--order-no",
        number,
        "--line",
        "A4S-FESTO",
        "--product-type-id",
        product["id"],
        "--planned-qty",
        10,
    )
    assert cli(*create, "--dry-run")["dry_run"]
    order = cli(*create)["read_back"]["data"]
    assert order["status"] == "PENDING"
    assert cli("order", "accept", number)["read_back"]["data"]["status"] == "ACCEPTED"
    issue = cli(
        "issue",
        "create",
        "--order",
        number,
        "--type",
        "TOOL_FAILURE",
        "--title",
        "Simulated equipment fault (CLI integration smoke test)",
    )
    issue_id = issue["read_back"]["data"]["id"]
    assert issue["order_after"]["data"]["status"] == "BLOCKED"
    cli("issue", "acknowledge", issue_id)
    resolved = cli(
        "issue",
        "resolve",
        issue_id,
        "--notes",
        "Simulation cleared; no physical machine was operated",
    )
    # Upstream recomputes status from batches; it does NOT restore ACCEPTED.
    assert resolved["order_after"]["data"]["status"] == "PENDING"
    assert cli("issue", "close", issue_id)["read_back"]["data"]["status"] == "CLOSED"
    start = (datetime.now(timezone.utc) + timedelta(days=2)).replace(second=0, microsecond=0)
    for previous in cli("order", "ls", "--line", "A4S-BACKUP")["data"]:
        if previous.get("planned_end_at"):
            start = max(
                start,
                datetime.fromisoformat(previous["planned_end_at"].replace("Z", "+00:00"))
                + timedelta(hours=1),
            )
    end = start + timedelta(minutes=15)
    move = (
        "schedule",
        "move",
        number,
        "--line",
        "A4S-BACKUP",
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
    )
    assert cli(*move, "--dry-run")["dry_run"]
    moved = cli(*move)["read_back"]["data"]
    assert moved["line"]["code"] == "A4S-BACKUP"
    cli("schedule", "board")
    cli("schedule", "capacity")
    cli("schedule", "changes")
    cli("machine", "monitor")
    cli("downtime", "reasons")
    cli("downtime", "ls")
    cli("oee", "show")

    from a4s.environments.local import LocalEnvironment
    from a4s.kernel.approval import Gate, Policy
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    ledger_path = ROOT / "temp" / "openmes-smoke" / TRACE_ID / "ledger.db"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(ledger_path, TRACE_ID, actor="operator-smoke")
    try:
        gate = Gate(Policy("machine"), ledger)
        registry = ToolRegistry(LocalEnvironment(), ledger, ROOT, gate=gate)
        command = f"mes order update {order['id']} --priority 2 --trace-id {TRACE_ID}"
        assert "refused" in registry.dispatch("shell", {"command": command})
        registry.begin_turn()
        gate.preapprove(command)
        output = registry.dispatch("shell", {"command": command})
        verified = json.loads(output)
        assert verified["verified"], output
        execution_id = verified["execution"]["execution_id"]
        assert verified["execution"]["trace_id"] == TRACE_ID
        events = list(ledger.read(session_id=TRACE_ID, limit=100))
        write = next(event for event in events if event.kind == "mes.write")
        assert write.payload["execution"]["execution_id"] == execution_id
        assert write.payload["outcome"] == "verified"
        approval = next(
            event for event in events if event.kind == "approval" and event.payload["allowed"]
        )
        assert approval.payload["detail"]["execution"]["execution_id"] == execution_id
        assert (
            write.payload["execution"]["request_fingerprint"]
            == verified["mutation"]["request_fingerprint"]
        )
        print("PASS harness shell -> approval -> mes -> live API -> mes.write Ledger", flush=True)
    finally:
        ledger.close()
    print(
        json.dumps(
            {
                "order_no": number,
                "order_id": order["id"],
                "issue_id": issue_id,
                "result": "passed",
                "records_retained": True,
                "trace_id": TRACE_ID,
                "ledger": str(ledger_path),
            }
        )
    )


if __name__ == "__main__":
    main()

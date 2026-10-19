"""Read-only operator projection of the existing workspace ledger."""
from contextlib import closing
import json
import re
import sqlite3

from .chat import _home

KINDS = ("tool.call", "tool.result", "approval", "mes.execution", "mes.write",
         "aas.write", "aas.create", "aas.delete", "op.invoke", "error")


def ledger_activity(since, limit=500):
    path = _home() / "ledger.db"
    if not path.exists():
        return {"activity": [], "ledger_available": False, "ledger_truncated": False}
    # Do not use Ledger(): its constructor creates schema and changes journal mode.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            f"SELECT * FROM events WHERE ts >= ? AND kind IN ({','.join('?' for _ in KINDS)}) "
            "ORDER BY seq DESC LIMIT ?", (since, *KINDS, limit + 1)).fetchall()
        truncated = len(rows) > limit
        activity = []
        for row in rows[:limit]:
            payload = json.loads(row["payload"])
            kind = row["kind"]
            call_id = payload.get("call_id")
            if kind == "tool.call" and call_id:
                # Show one tool result, or a pending call when no result exists yet.
                completed = db.execute(
                    "SELECT 1 FROM events WHERE session_id=? AND kind='tool.result' "
                    "AND json_extract(payload, '$.call_id')=? LIMIT 1",
                    (row["session_id"], call_id)).fetchone()
                if completed:
                    continue
            tool = payload.get("tool", "tool")
            execution = payload.get("execution") or {}
            outcome = execution.get("outcome")
            status = "Recorded"
            title = kind
            if kind == "tool.call":
                title, status = f"Agent · {tool}", "Awaiting result"
            elif kind == "tool.result":
                output = str(payload.get("output", ""))
                status = "Refused" if output.startswith("refused:") else "Error" if output.startswith("error:") else "Result recorded"
                title = "Agent task finished" if tool == "finish" else f"Agent · {tool} result"
                if tool == "shell" and call_id:
                    call = db.execute(
                        "SELECT payload FROM events WHERE session_id=? AND kind='tool.call' "
                        "AND json_extract(payload, '$.call_id')=? ORDER BY seq DESC LIMIT 1",
                        (row["session_id"], call_id)).fetchone()
                    if call:
                        command = json.loads(call['payload']).get('arguments', {}).get('command', '')
                        # Only expose the command family and verb, not arguments or output.
                        action = re.search(r'\b(mes|aas|a4s\.mesctl|a4s\.aasctl)\s+([a-z-]+)\s+([a-z-]+)', command)
                        if action:
                            family = 'MES' if 'mes' in action[1] else 'AAS'
                            title = f"Agent · {family} {action[2]} {action[3]}"
            elif kind == "approval":
                status = "Allowed" if payload.get("allowed") is True else "Not allowed"
                title = "Approval decision"
            elif kind == "mes.execution":
                title = "Agent · MES execution"
                status = outcome or payload.get("stage", "Recorded")
            else:
                title = {"mes.write": "MES write recorded", "aas.write": "AAS write recorded",
                         "aas.create": "AAS created", "aas.delete": "AAS deleted",
                         "op.invoke": "Operation invoked", "error": "Agent error"}.get(kind, kind)
            # Deliberately project metadata, not raw prompts, command arguments,
            # tool output or credentials from the full audit payload.
            evidence = {"seq": row["seq"], "session_id": row["session_id"],
                        "actor": row["actor"], "kind": kind, "call_id": call_id,
                        "hash": row["hash"], "status": status}
            for key in ("execution_id", "trace_id", "outcome", "request_fingerprint", "action"):
                if key in execution:
                    evidence[key] = execution[key]
            if kind == "approval":
                evidence.update({k: payload.get(k) for k in ("allowed", "source", "risk")})
            activity.append({"id": f"ledger:{row['seq']}", "at": row["ts"],
                             "type": "Agent ledger", "title": title,
                             "detail": f"{status} · {row['session_id']}", "evidence": evidence})
    return {"activity": activity, "ledger_available": True, "ledger_truncated": truncated}

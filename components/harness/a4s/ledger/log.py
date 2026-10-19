"""Append-only, hash-chained event log.

Adapted from Buzz's premise that every action -- by a human or a process --
is one signed event in one log, and the log is the single source of truth.
Buzz reaches that over Nostr relays because it needs federation between
untrusted parties. A shopfloor does not: the parties are known, the
deployment is one site, and a relay would be infrastructure without a
matching requirement. What survives the translation is the shape:

* **One log, every actor.** Operator actions, agent tool calls, AAS
  mutations and service invocations are the same kind of record. There is
  no separate "agent log" to reconcile against the process record.
* **Tamper evidence.** Each entry commits to its predecessor, so a
  retrospective edit is detectable rather than merely discouraged --
  the property an audit of an autonomous system actually needs.
* **Identity per actor, not per permission flag.** Each agent signs with
  its own key, so "who changed this property" is answerable after the fact
  without trusting an application-level field.

Two consumers, one structure. Compliance reads it as provenance for every
change to an asset's shell. Training reads it as trajectories -- the log
already holds the observation/action pairs that a fine-tuning or RL
pipeline would otherwise have to reconstruct from scraped console output.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL    NOT NULL,
    session_id TEXT    NOT NULL,
    actor      TEXT    NOT NULL,
    kind       TEXT    NOT NULL,
    subject    TEXT,
    payload    TEXT    NOT NULL,
    prev_hash  TEXT    NOT NULL,
    hash       TEXT    NOT NULL UNIQUE,
    signature  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject);
CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind);
"""


# Event kinds. Strings rather than Nostr's integers: the log is read by
# people during an audit far more often than it is parsed, and a grep for
# "aas.write" should work without a lookup table.
class Kind:
    TURN_USER = "turn.user"
    TURN_ASSISTANT = "turn.assistant"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    CMD_EXEC = "cmd.exec"
    AAS_WRITE = "aas.write"
    MES_WRITE = "mes.write"
    MES_EXECUTION = "mes.execution"
    AAS_CREATE = "aas.create"
    AAS_DELETE = "aas.delete"
    OP_INVOKE = "op.invoke"
    # Every gate decision, including the automatic ones: "who approved
    # changing this machine's temperature, and when" has to be answerable.
    APPROVAL = "approval"
    DELEGATE = "agent.delegate"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    session_id: str
    actor: str
    kind: str
    subject: Optional[str]
    payload: Dict[str, Any]
    prev_hash: str
    hash: str
    signature: Optional[str]


def _canonical(
    ts: float,
    session_id: str,
    actor: str,
    kind: str,
    subject: Optional[str],
    payload: Dict[str, Any],
    prev_hash: str,
) -> bytes:
    """Deterministic serialisation of the fields the hash commits to.

    Sorted keys and no whitespace: two processes must derive byte-identical
    input, or the chain fails to verify for a reason that has nothing to do
    with tampering.
    """
    return json.dumps(
        {
            "ts": round(ts, 6),
            "session_id": session_id,
            "actor": actor,
            "kind": kind,
            "subject": subject,
            "payload": payload,
            "prev": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class Ledger:
    def __init__(self, path: str | Path, session_id: str, actor: str = "operator") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.actor = actor
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        # WAL so a reader (a dashboard, an export job) never blocks the
        # agent mid-turn.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._signer = None

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def use_identity(self, signer: "Identity") -> None:
        self._signer = signer
        self.actor = signer.public_id

    # -- writing ---------------------------------------------------------

    def head(self) -> str:
        row = self._db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return row["hash"] if row else GENESIS

    def append(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        subject: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> Event:
        ts = time.time()
        who = actor or self.actor
        prev = self.head()
        body = _canonical(ts, self.session_id, who, kind, subject, payload, prev)
        digest = hashlib.sha256(body).hexdigest()
        signature = self._signer.sign(body) if self._signer else None

        cursor = self._db.execute(
            "INSERT INTO events (ts, session_id, actor, kind, subject, payload,"
            " prev_hash, hash, signature) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                ts,
                self.session_id,
                who,
                kind,
                subject,
                json.dumps(payload, ensure_ascii=False),
                prev,
                digest,
                signature,
            ),
        )
        self._db.commit()
        return Event(
            seq=cursor.lastrowid,
            ts=ts,
            session_id=self.session_id,
            actor=who,
            kind=kind,
            subject=subject,
            payload=payload,
            prev_hash=prev,
            hash=digest,
            signature=signature,
        )

    # -- reading ---------------------------------------------------------

    def read(
        self,
        *,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        subject: Optional[str] = None,
        limit: int = 500,
        newest_first: bool = False,
    ) -> List[Event]:
        clauses: List[str] = []
        params: List[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        order = "DESC" if newest_first else "ASC"
        rows = self._db.execute(
            f"SELECT * FROM events {where} ORDER BY seq {order} LIMIT ?", params
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def provenance(self, subject: str) -> List[Event]:
        """Every recorded change to one AAS path, oldest first.

        This is the question an auditor actually asks -- "who set this
        property, when, and on whose instruction" -- and it is a single
        indexed lookup because subject is the AAS path.
        """
        return self.read(subject=subject, limit=10_000)

    def verify(self) -> tuple[bool, Optional[int]]:
        """Recompute the chain. Returns (ok, first_bad_seq)."""
        prev = GENESIS
        for row in self._db.execute("SELECT * FROM events ORDER BY seq ASC"):
            body = _canonical(
                row["ts"],
                row["session_id"],
                row["actor"],
                row["kind"],
                row["subject"],
                json.loads(row["payload"]),
                prev,
            )
            if row["prev_hash"] != prev or hashlib.sha256(body).hexdigest() != row["hash"]:
                return False, row["seq"]
            prev = row["hash"]
        return True, None

    # -- replay ----------------------------------------------------------

    def replay(self, session_id: str, system_prompt: str) -> List[Dict[str, Any]]:
        """Rebuild a session's conversation, exactly as the model last saw it.

        A run that stops for an operator has to be continuable, and the ledger
        is the only copy: the process is gone. There is no separate checkpoint
        file because a second store of the same conversation is a second store
        that can disagree with the audit record -- the decision log's reason for
        having one append-only log at all.

        The system prompt is passed in rather than replayed. It is generated
        from the skills on disk and the CLI's own verb table, both of which may
        legitimately have changed; pinning a stale copy would resume an agent
        into a world that no longer exists.
        """
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for event in self.read(session_id=session_id, limit=100_000):
            if event.kind == Kind.TURN_USER:
                messages.append({"role": "user", "content": event.payload.get("text", "")})
            elif event.kind == Kind.TURN_ASSISTANT:
                message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": event.payload.get("text") or "",
                }
                calls = event.payload.get("tool_calls") or []
                if calls:
                    message["tool_calls"] = calls
                messages.append(message)
            elif event.kind == Kind.TOOL_RESULT:
                call_id = event.payload.get("call_id")
                if not call_id:
                    # Written before call ids were recorded. Attaching the
                    # result to the wrong call would corrupt the history in a
                    # way the model cannot see, so the session is not resumable
                    # and says so rather than resuming into nonsense.
                    raise ValueError(
                        f"session {session_id!r} was recorded before tool call ids "
                        "were logged and cannot be replayed"
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": event.payload.get("output", ""),
                    }
                )
        return messages

    def first_instruction(self, session_id: str) -> Optional[str]:
        """What the operator originally asked, for rebuilding the prompt.

        The system prompt is regenerated on resume rather than replayed, and
        generating it needs the instruction: that is what selects the skill.
        Without it a resumed run lost the procedure the original run had --
        and `select_skill`'s own measurement says an agent that has to find a
        skill by itself does not, three times out of three.

        The *first* user turn, not the last: later ones are the operator
        answering a question, and answering "yes" is not the task.
        """
        for event in self.read(session_id=session_id, limit=100_000):
            if event.kind == Kind.TURN_USER:
                text = str(event.payload.get("text") or "").strip()
                if text:
                    return text
        return None

    def last_refusal(self, session_id: str) -> Optional[Event]:
        """The most recent action the gate stopped.

        Not simply the last approval event: reads are recorded too, and by the
        time a run halts the agent has usually called `ask_user` after being
        refused, so the last approval is that harmless call rather than the one
        the operator is being asked about.
        """
        for event in reversed(self.read(session_id=session_id, kind=Kind.APPROVAL, limit=10_000)):
            if not event.payload.get("allowed"):
                return event
        return None

    # -- export ----------------------------------------------------------

    def export_trajectory(self, session_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Yield one JSONL record per turn, shaped for SFT/RL curation.

        The log is already the trajectory: a tool call and its result are
        adjacent events sharing a subject. Emitting them in this shape is
        what removes the separate scrape-and-reconstruct step that
        trajectory datasets usually need, and it means training data and
        audit evidence can never disagree -- they are the same rows.
        """
        events = self.read(session_id=session_id or self.session_id, limit=100_000)
        pending: Optional[Event] = None
        for event in events:
            if event.kind == Kind.TOOL_CALL:
                pending = event
            elif event.kind == Kind.TOOL_RESULT and pending is not None:
                yield {
                    "session": event.session_id,
                    "actor": event.actor,
                    "call": pending.payload,
                    "result": event.payload,
                    "subject": pending.subject,
                    "latency_s": round(event.ts - pending.ts, 4),
                }
                pending = None
            elif event.kind in (Kind.TURN_USER, Kind.TURN_ASSISTANT):
                yield {"session": event.session_id, "role": event.kind, "content": event.payload}


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        seq=row["seq"],
        ts=row["ts"],
        session_id=row["session_id"],
        actor=row["actor"],
        kind=row["kind"],
        subject=row["subject"],
        payload=json.loads(row["payload"]),
        prev_hash=row["prev_hash"],
        hash=row["hash"],
        signature=row["signature"],
    )


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

class Identity:
    """An Ed25519 keypair identifying one actor.

    Every agent -- the orchestrator, each asset agent, each delegated child
    -- gets its own. Scoping by identity rather than by a permission flag is
    the property worth importing from Buzz: an asset agent's authority is
    the set of shells its key may write, which is checkable from the log
    afterwards and does not depend on the agent process behaving.

    Signing is optional. Without ``cryptography`` installed the hash chain
    still gives tamper evidence; signatures add attribution, which matters
    once more than one agent writes to the same repository.
    """

    def __init__(self, private_key: Any, public_id: str) -> None:
        self._key = private_key
        self.public_id = public_id

    @classmethod
    def generate(cls) -> "Identity":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        return cls(key, _public_id(key))

    @classmethod
    def load(cls, path: str | Path) -> "Identity":
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        raw = Path(path).read_bytes()
        key = Ed25519PrivateKey.from_private_bytes(raw)
        return cls(key, _public_id(key))

    @classmethod
    def load_or_create(cls, path: str | Path) -> "Identity":
        target = Path(path)
        if target.exists():
            return cls.load(target)
        identity = cls.generate()
        target.parent.mkdir(parents=True, exist_ok=True)
        from cryptography.hazmat.primitives import serialization

        target.write_bytes(
            identity._key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        os.chmod(target, 0o600)
        return identity

    def sign(self, body: bytes) -> str:
        return self._key.sign(body).hex()


def _public_id(key: Any) -> str:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()

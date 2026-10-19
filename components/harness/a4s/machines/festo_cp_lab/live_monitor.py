"""Persistent read-only fault episodes. No MES state, PLC writes or LLM calls."""

import hashlib
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

CRITICAL = ("Press.Fault", "Turning.Fault")
MAX_GAP = 60


def reduce_sample(state, sample, now):
    """Two consecutive complete Good Boolean frames confirm a transition.

    A gap invalidates continuity, not the last confirmed equipment state.
    All timestamps are host observations; a polling interval can miss short faults.
    """
    s = dict(state)
    events = []
    previous = s.get("last_good")
    rows = {v["path"]: v for v in sample.get("values", [])}
    good = all(
        sum(v.get("path") == p for v in sample.get("values", [])) == 1
        and p in rows
        and rows[p].get("status") == "Good"
        and rows[p].get("dataType") == "Boolean"
        and type(rows[p].get("value")) is bool
        for p in CRITICAL
    )
    binding = hashlib.sha256(
        json.dumps(
            [(p, rows.get(p, {}).get("endpoint"), rows.get(p, {}).get("node")) for p in CRITICAL],
            sort_keys=True,
        ).encode()
    ).hexdigest()

    def event(kind, before, after):
        events.append(
            {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "at": now,
                "episode_id": s.get("episode_id"),
                "before": before,
                "after": after,
            }
        )

    if not good:
        if s.get("communication") != "UNKNOWN":
            event("COMMUNICATION_LOST", previous, sample)
        s.update(communication="UNKNOWN", gap=True, candidate=None, count=0, latest=sample)
        return s, events
    if s.get("communication") == "UNKNOWN":
        event("COMMUNICATION_RESTORED", previous, sample)
    if s.get("binding") and s["binding"] != binding:
        event("BINDING_CHANGED", previous, sample)
        s.update(stable=None, episode_id=None, candidate=None, count=0, gap=True)
    if previous and now - previous["observed_at"] > MAX_GAP:
        s.update(gap=True, candidate=None, count=0)
    fault = any(rows[p]["value"] for p in CRITICAL)
    if s.get("candidate") != fault:
        s.update(candidate=fault, count=1, candidate_before=previous, candidate_first=sample)
    else:
        s["count"] = s.get("count", 0) + 1
    if s["count"] >= 2:
        stable = s.get("stable")
        if stable is None:
            if fault:
                s["episode_id"] = uuid.uuid4().hex
                event("BASELINE_FAULT", s.get("candidate_before"), sample)
        elif stable != fault:
            if fault:
                s["episode_id"] = uuid.uuid4().hex
                event(
                    "FAULT_OBSERVED_AFTER_GAP" if s.get("gap") else "FAULT_RAISED",
                    s.get("candidate_before"),
                    sample,
                )
            else:
                event("RECOVERY_OBSERVED", s.get("candidate_before"), sample)
                s["episode_id"] = None
        s.update(stable=fault, gap=False)
    s.update(communication="GOOD", binding=binding, last_good=sample, latest=sample)
    return s, events


class Monitor:
    """Single-process demo service. SQLite holds evidence and one-shot handoffs.

    Monitoring is explicitly re-armed after restart; old fault episodes survive.
    No automatic retry of an uncertain Agent handoff. Not a multi-worker service.
    """

    def __init__(self, path, reader, interval=3):
        self.path = Path(path)
        self.reader = reader
        self.interval = interval
        self.lock = threading.RLock()
        self.workers = {}

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE IF NOT EXISTS states (scope TEXT PRIMARY KEY, body TEXT NOT NULL)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "scope TEXT NOT NULL, id TEXT UNIQUE, body TEXT NOT NULL, claimed INTEGER DEFAULT 0)"
        )
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def key(target):
        return hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()

    def _load(self, db, key):
        row = db.execute("SELECT body FROM states WHERE scope=?", (key,)).fetchone()
        return json.loads(row[0]) if row else {}

    def _save(self, db, key, state):
        db.execute("INSERT OR REPLACE INTO states VALUES (?,?)", (key, json.dumps(state)))

    def ingest(self, target, sample):
        key = self.key(target)
        with self.lock, self.db() as db:
            state, events = reduce_sample(self._load(db, key), sample, sample["observed_at"])
            self._save(db, key, state)
            for event in events:
                event["target"] = target
                event["first_observed_at"] = (
                    (state.get("candidate_first") or sample)["observed_at"]
                    if event["kind"]
                    in (
                        "BASELINE_FAULT",
                        "FAULT_RAISED",
                        "FAULT_OBSERVED_AFTER_GAP",
                        "RECOVERY_OBSERVED",
                    )
                    else sample["observed_at"]
                )
                db.execute(
                    "INSERT INTO events(scope,id,body) VALUES (?,?,?)",
                    (key, event["id"], json.dumps(event)),
                )
        return events

    def status(self, target):
        key = self.key(target)
        with self.lock, self.db() as db:
            state = self._load(db, key)
            events = [
                {**json.loads(row["body"]), "seq": row["seq"], "claimed": bool(row["claimed"])}
                for row in db.execute(
                    "SELECT * FROM events WHERE scope=? ORDER BY seq DESC LIMIT 100", (key,)
                )
            ]
            worker = self.workers.get(key)
            running = bool(worker and worker[1].is_alive() and not worker[0].is_set())
        at = (state.get("latest") or {}).get("observed_at", 0)
        return {
            "target": target,
            "running": running,
            "state": state,
            "events": list(reversed(events)),
            "stale": not running or time.time() - at > MAX_GAP,
            "note": "Host timestamps; two Good Boolean samples confirm transitions. No PLC writes.",
        }

    def start(self, target):
        key = self.key(target)
        with self.lock:
            old = self.workers.get(key)
            if old and old[1].is_alive():
                if old[0].is_set():
                    raise ValueError("Previous read is still stopping; wait before re-arming")
                return self.status(target)
            if sum(thread.is_alive() for _, thread in self.workers.values()) >= 8:
                raise ValueError("Eight active monitors maximum")
            with self.db() as db:
                state = self._load(db, key)
                state.update(gap=True, candidate=None, count=0)
                self._save(db, key, state)
            stop = threading.Event()
            thread = threading.Thread(target=self._loop, args=(target, stop), daemon=True)
            self.workers[key] = (stop, thread)
            thread.start()
        return self.status(target)

    def stop(self, target):
        with self.lock:
            worker = self.workers.get(self.key(target))
            if worker:
                worker[0].set()
        return self.status(target)

    def close(self):
        with self.lock:
            for stop, _ in self.workers.values():
                stop.set()

    def _loop(self, target, stop):
        while not stop.is_set():
            try:
                sample = self.reader(target)
            except Exception as exc:
                sample = {"values": [], "error": type(exc).__name__}
            sample["observed_at"] = time.time()
            with self.lock:
                if stop.is_set():
                    return
                self.ingest(target, sample)
            stop.wait(self.interval)

    def claim(self, target, event_id):
        """At-most-once dispatch reservation, NOT a completed Agent task."""
        with self.lock, self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM events WHERE scope=? AND id=?", (self.key(target), event_id)
            ).fetchone()
            if not row or row["claimed"]:
                return None
            event = json.loads(row["body"])
            state = self._load(db, self.key(target))
            if (
                event["kind"] != "FAULT_RAISED"
                or event["episode_id"] != state.get("episode_id")
                or state.get("count", 0) < 2
                or time.time() - event["at"] > 300
                or state.get("communication") != "GOOD"
                or time.time() - (state.get("latest") or {}).get("observed_at", 0) > MAX_GAP
            ):
                return None
            db.execute("UPDATE events SET claimed=1 WHERE id=?", (event_id,))
            return event

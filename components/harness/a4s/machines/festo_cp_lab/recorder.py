"""Record the cell's OPC UA signals to SQLite, so a test leaves evidence.

Four start attempts have been made against this cell and none of them moved
it. What is known about each is what somebody happened to read afterwards,
which is why the fault bits in `dbVar.Err` went unseen for a day: nobody was
looking at them at the moment that mattered. A recorder running across the
attempt turns "it did not start" into a timeline.

**Changes, not samples.** A row is written when a value differs from the last
one recorded, plus a heartbeat so a long flat stretch is distinguishable from a
recorder that died. The cell is mostly still -- 30 seconds of idle pressure has
a standard deviation of 2 counts -- so change-only recording is both small and
exactly what a diagnosis reads: `xBusy` went true at 13:24:06.2, and here is
everything else that moved in that second.

**The node list is wider than the AAS.** The model carries what an operator
needs; a diagnosis needs the safety inputs, the fault words, the mode block and
the analogue channels the PLCs have of their own. Those are listed here rather
than modelled, because most of them are only interesting while something is
wrong -- and a model that carried every symbol in the data block would be a
copy of the PLC, not a description of the asset.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .model import SERVERS

APPLICATION_DB = 'ns=3;s="ApplicationModule_DB"'
REMOTE_DB = 'ns=3;s="dbVar"'

# What to watch on a station's PLC. Grouped by where the answer to a different
# question lives, because that is how a person reads the timeline back.
PLC_NODES: List[Tuple[str, str]] = [
    # the application module's own interface
    *[(f"app.{n}", f'{APPLICATION_DB}."{n}"') for n in (
        "xStart", "xInit", "xAuto", "xManual", "xReset",
        "xReady", "xBusy", "xError", "xInitPos", "iRetCode",
    )],
    # the collective fault word. `xSF` is the one that was true while
    # `app.xError` stayed false, which is the whole reason this list exists.
    *[(f"err.{n}", f'{REMOTE_DB}."Err"."{n}"') for n in (
        "xSF", "xSF0", "xSF1", "xSF2", "xSF01", "xSF01_ext", "xQuit", "xHint",
    )],
    # the remote-control surface
    ("remote.Start", f'{REMOTE_DB}."RemoteControlAutomatic_Start"'),
    ("remote.Stop", f'{REMOTE_DB}."RemoteControlAutomatic_Stop"'),
    ("mode.Man.xEn", f'{REMOTE_DB}."OpMode"."Man"."xEn"'),
    ("mode.Man.byStatus", f'{REMOTE_DB}."OpMode"."Man"."byStatus"'),
    ("mode.xInit", f'{REMOTE_DB}."OpMode"."xInit"'),
    ("mode.xInitPos", f'{REMOTE_DB}."OpMode"."xInitPos"'),
    # the process image: real switches and real actuators, named from the
    # configuration the MSB demo used
    ("in.xSF5_estop", 'ns=3;s="xSF5"'),
    ("in.xSF2", 'ns=3;s="xSF2"'),
    ("in.xSF3", 'ns=3;s="xSF3"'),
    ("in.xG1_KG1_BG7", 'ns=3;s="xG1_KG1_BG7"'),
    ("in.xBG1_carrier_at_module", 'ns=3;s="xBG1"'),
    ("in.xBG5_carrier_at_start", 'ns=3;s="xBG5"'),
    ("in.xBG6_carrier_at_end", 'ns=3;s="xBG6"'),
    ("in.byAnaIn_8Bit", 'ns=3;s="byAnaIn_8Bit"'),
    ("in.byAnaIn2_8Bit", 'ns=3;s="byAnaIn2_8Bit"'),
    ("out.xQA1_A1_belt_fwd", 'ns=3;s="xQA1_A1"'),
    ("out.xQA1_A2_belt_rev", 'ns=3;s="xQA1_A2"'),
    ("rfid.iCode", 'ns=3;s="dbRfidData"."ID1"."iCode"'),
    ("rfid.iCarrierID", 'ns=3;s="dbRfidData"."ID1"."iCarrierID"'),
]

REVPI_NODES: List[Tuple[str, str]] = [
    ("sensor.Pressure_1", "ns=1;s=Pressure_1"),
    ("sensor.Pressure_2", "ns=1;s=Pressure_2"),
    ("sensor.Flow_1", "ns=1;s=Flow_1"),
    ("sensor.Flow_2", "ns=1;s=Flow_2"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts        REAL NOT NULL,
    station   TEXT NOT NULL,
    signal    TEXT NOT NULL,
    node      TEXT NOT NULL,
    value     TEXT,
    kind      TEXT NOT NULL      -- 'change' | 'heartbeat' | 'error'
);
CREATE INDEX IF NOT EXISTS readings_ts     ON readings(ts);
CREATE INDEX IF NOT EXISTS readings_signal ON readings(station, signal, ts);

CREATE TABLE IF NOT EXISTS notes (
    ts   REAL NOT NULL,
    text TEXT NOT NULL
);
"""


@dataclass
class Target:
    station: str
    endpoint: str
    signals: List[Tuple[str, str]]


def targets() -> List[Target]:
    """One entry per server, with the signals worth watching on it."""
    out: List[Target] = []
    for name, spec in SERVERS.items():
        signals = REVPI_NODES if spec["device"] == "RevolutionPi" else PLC_NODES
        out.append(Target(station=name, endpoint=spec["endpoint"],
                          signals=list(signals)))
    return out


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path))
    db.executescript(SCHEMA)
    # A recorder that loses the last few seconds to a power cut has lost the
    # part of the timeline it was running for. Durability over throughput: this
    # writes a handful of rows a second, not thousands.
    db.execute("PRAGMA journal_mode=WAL")
    db.commit()
    return db


def note(db: sqlite3.Connection, text: str) -> None:
    """Mark the timeline. `--note 'about to press start'` is worth more than a
    guess at which second an operator acted."""
    db.execute("INSERT INTO notes (ts, text) VALUES (?, ?)", (time.time(), text))
    db.commit()


async def _watch(target: Target, db: sqlite3.Connection, *, interval: float,
                 heartbeat: float, deadline: float) -> None:
    from asyncua import Client

    last: Dict[str, Any] = {}
    last_beat = 0.0

    while time.monotonic() < deadline:
        try:
            async with Client(url=target.endpoint, timeout=8) as client:
                nodes = {name: client.get_node(nid) for name, nid in target.signals}
                while time.monotonic() < deadline:
                    now = time.time()
                    beat = (time.monotonic() - last_beat) >= heartbeat
                    rows = []
                    for name, node in nodes.items():
                        try:
                            value = await node.read_value()
                        except Exception as exc:  # noqa: BLE001
                            rows.append((now, target.station, name,
                                         dict(target.signals)[name],
                                         f"{type(exc).__name__}", "error"))
                            continue
                        text = str(value)
                        if last.get(name) != text:
                            rows.append((now, target.station, name,
                                         dict(target.signals)[name], text, "change"))
                            last[name] = text
                        elif beat:
                            rows.append((now, target.station, name,
                                         dict(target.signals)[name], text,
                                         "heartbeat"))
                    if beat:
                        last_beat = time.monotonic()
                    if rows:
                        db.executemany(
                            "INSERT INTO readings (ts, station, signal, node, value, kind)"
                            " VALUES (?, ?, ?, ?, ?, ?)", rows)
                        db.commit()
                    await asyncio.sleep(interval)
        except Exception as exc:  # noqa: BLE001 - a dropped server is not the run
            # Recorded, so a gap in the timeline is explained rather than blank.
            db.execute(
                "INSERT INTO readings (ts, station, signal, node, value, kind)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), target.station, "_connection", target.endpoint,
                 f"{type(exc).__name__}: {exc}", "error"))
            db.commit()
            await asyncio.sleep(2.0)


async def record(db: sqlite3.Connection, *, interval: float, heartbeat: float,
                 duration: float) -> None:
    deadline = time.monotonic() + duration
    await asyncio.gather(*(
        _watch(t, db, interval=interval, heartbeat=heartbeat, deadline=deadline)
        for t in targets()
    ))


# ---------------------------------------------------------------------------
# reading it back
# ---------------------------------------------------------------------------

def timeline(db: sqlite3.Connection, *, since: Optional[float] = None,
             limit: int = 200) -> List[sqlite3.Row]:
    db.row_factory = sqlite3.Row
    where = "WHERE kind = 'change'"
    args: List[Any] = []
    if since is not None:
        where += " AND ts >= ?"
        args.append(since)
    return list(db.execute(
        f"SELECT * FROM readings {where} ORDER BY ts LIMIT ?", (*args, limit)))


def summary(db: sqlite3.Connection) -> Dict[str, Any]:
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT COUNT(*) n, MIN(ts) a, MAX(ts) b FROM readings").fetchone()
    changes = db.execute(
        "SELECT station, signal, COUNT(*) n FROM readings WHERE kind='change'"
        " GROUP BY station, signal ORDER BY n DESC LIMIT 20").fetchall()
    errors = db.execute(
        "SELECT COUNT(*) n FROM readings WHERE kind='error'").fetchone()["n"]
    return {
        "rows": row["n"],
        "from": row["a"],
        "to": row["b"],
        "seconds": (row["b"] - row["a"]) if row["a"] else 0,
        "errors": errors,
        "most_active": [dict(r) for r in changes],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        prog="a4s-festo-record",
        description="Record the Festo cell's OPC UA signals to SQLite.")
    parser.add_argument("--db", type=Path,
                        default=Path.home() / ".a4s" / "festo.db")
    parser.add_argument("--interval", type=float, default=0.25,
                        help="seconds between polls (default 0.25)")
    parser.add_argument("--heartbeat", type=float, default=30.0,
                        help="seconds between full samples, so a flat stretch "
                             "is distinguishable from a dead recorder")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="seconds to record (default 600)")
    parser.add_argument("--note", help="write a note into the timeline and exit")
    parser.add_argument("--timeline", action="store_true",
                        help="print what changed, most recent run")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    db = open_db(args.db)

    if args.note:
        note(db, args.note)
        print(f"noted at {datetime.now():%H:%M:%S}")
        return 0

    if args.summary:
        s = summary(db)
        print(f"{s['rows']} rows over {s['seconds']:.0f}s, {s['errors']} errors")
        for r in s["most_active"]:
            print(f"  {r['n']:6}  {r['station']:12} {r['signal']}")
        return 0

    if args.timeline:
        rows = timeline(db)
        for r in rows:
            t = datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S.%f")[:-3]
            print(f"  {t}  {r['station']:12} {r['signal']:28} = {r['value']}")
        print(f"  ({len(rows)} changes)")
        return 0

    print(f"recording {sum(len(t.signals) for t in targets())} signals from "
          f"{len(targets())} servers into {args.db}")
    print(f"every {args.interval}s for {args.duration:.0f}s; "
          f"changes only, heartbeat every {args.heartbeat:.0f}s")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(record(db, interval=args.interval, heartbeat=args.heartbeat,
                           duration=args.duration))
    except KeyboardInterrupt:
        print("\nstopped")
    s = summary(db)
    print(f"{s['rows']} rows, {s['errors']} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Job entry points that exist so the queue can be tested honestly.

A job is resolved by dotted path **inside a fresh process**, so a test target
has to live in an importable module — a function defined in a test file would
be re-imported by ``spawn`` along with the whole test session. These three are
deliberately trivial and import nothing: a worker running them costs
milliseconds, which keeps the queue's own tests from paying torch's import
cost to prove that a timeout works.
"""

from __future__ import annotations

import time

__all__ = ["echo", "boom", "sleep_forever"]


def echo(payload: dict) -> dict:
    """Return the payload with a marker — the success path."""
    return {"echoed": payload, "ok": True}


def boom(payload: dict) -> dict:
    """Raise, so the failure path carries a real traceback back to the parent."""
    raise ValueError(payload.get("message", "deliberate failure"))


def sleep_forever(payload: dict) -> dict:
    """Sleep past any sane timeout, so cancellation has something to cancel."""
    time.sleep(float(payload.get("seconds", 600)))
    return {"finished": True}

"""Deterministic, explicitly synthetic signal stream, advanced on polling.

No background PLC client, no claim of unattended agent monitoring. The open UI
polls and can wake chat once per simulated alarm after the operator arms it.
"""

import threading
import time
import uuid

LOCK = threading.RLock()
STATE = {"run_id": None, "started_at": None, "active": False}


def start(normal_only=False):
    with LOCK:
        if STATE["active"]:
            return observe()
        now = time.time()
        STATE.update(run_id=uuid.uuid4().hex, started_at=now, active=True,
                     leak_started_at=None if normal_only else now,
                     alarm_sequence=0 if normal_only else 1,
                     transitions=[{"at": now, "leak": None if normal_only else now}])
        return observe()


def inject():
    with LOCK:
        if not STATE["active"]:
            raise ValueError("Start simulation monitoring first")
        if STATE.get("leak_started_at") is None:
            now = time.time()
            STATE.update(leak_started_at=now, alarm_sequence=STATE.get("alarm_sequence", 0) + 1)
            STATE["transitions"].append({"at": now, "leak": now})
        return observe()


def recover():
    with LOCK:
        if STATE.get("leak_started_at") is not None:
            STATE["transitions"].append({"at": time.time(), "leak": None})
            STATE["leak_started_at"] = None
        return observe()


def stop():
    with LOCK:
        STATE["active"] = False
        return observe()


def signal_at(elapsed):
    pressure = max(3.8, 6.0 - max(0, elapsed - 8) * .13)
    below_since = 8 + (6.0 - 4.8) / .13
    fault = elapsed > below_since + 3
    status = "FAULT" if fault else "WARNING" if pressure < 5.2 else "NORMAL"
    return {"elapsed": round(elapsed, 1), "pressure_bar": round(pressure, 2),
            "status": status, "press_ready": pressure >= 4.8 and not fault,
            "turning_ready": True, "conveyor_speed_m_min": 0 if fault else 4.0,
            "collective_fault": fault}


def observe():
    with LOCK:
        now = time.time()
        elapsed = max(0, now - STATE["started_at"]) if STATE["active"] else 0
        def sample(second):
            sample_time = (STATE["started_at"] or now) + second
            transitions = [t for t in STATE.get("transitions", []) if t["at"] <= sample_time]
            leak = transitions[-1]["leak"] if transitions and STATE["active"] else None
            at = max(0, sample_time - leak) if leak is not None else 0
            return {**signal_at(at), "elapsed": round(second, 1)}
        current = sample(elapsed)
        history = [sample(second) for second in range(max(0, int(elapsed) - 59), int(elapsed) + 1)]
        return {**STATE, **current, "status": current["status"] if STATE["active"] else "DISARMED",
                "mode": "SIMULATION", "observed_at": now, "history": history,
                "thresholds": {"warning_bar": 5.2, "fault_bar": 4.8, "sustain_seconds": 3},
                "alarm_id": f"{STATE['run_id']}:{STATE.get('alarm_sequence', 0)}:pressure-low" if STATE["active"] and current["collective_fault"] else None,
                "hypothesis": "Fictional pneumatic connector leakage; not a measured root cause",
                "note": "Synthetic engineering units, never converted from live OPC raw values. "
                        "The open browser polls this stream; not an unattended monitoring service."}

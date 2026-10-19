import time
import threading
import pytest

from a4s.machines.festo_cp_lab.live_monitor import Monitor, reduce_sample


def sample(fault=False, at=None, status="Good", missing=False):
    return {
        "observed_at": at or time.time(),
        "values": []
        if missing
        else [
            {
                "path": p,
                "value": fault if p.startswith("Press") else False,
                "dataType": "Boolean",
                "status": status,
                "endpoint": "opc.tcp://test",
                "node": p,
                "source_timestamp": "2026-09-23T10:00:00Z",
            }
            for p in ("Press.Fault", "Turning.Fault")
        ],
    }


def feed(state, *frames):
    events = []
    for frame in frames:
        state, new = reduce_sample(state, frame, frame["observed_at"])
        events.extend(new)
    return state, events


def test_baseline_existing_fault_is_not_new_and_repeats_are_merged():
    state, events = feed({}, *[sample(True) for _ in range(10)])
    assert [e["kind"] for e in events] == ["BASELINE_FAULT"]
    assert state["stable"] is True


def test_transition_evidence_debounce_recovery_and_new_episode():
    s, _ = feed({}, sample(), sample())
    s, events = feed(s, sample(True), sample())
    assert not events and s["stable"] is False
    s, events = feed(s, sample(True), sample(True), sample(True))
    event = events[0]
    assert event["kind"] == "FAULT_RAISED"
    assert event["before"]["values"][0]["value"] is False
    assert event["after"]["values"][0]["value"] is True
    s, recovery = feed(s, sample(), sample())
    assert recovery[0]["kind"] == "RECOVERY_OBSERVED"
    s, events = feed(s, sample(True), sample(True))
    assert events[0]["episode_id"] != event["episode_id"]


def test_unknown_never_clears_fault_and_never_creates_new_fault():
    s, _ = feed({}, sample(), sample())
    s, events = feed(s, sample(missing=True), sample(missing=True), sample(True), sample(True))
    assert [e["kind"] for e in events] == [
        "COMMUNICATION_LOST",
        "COMMUNICATION_RESTORED",
        "FAULT_OBSERVED_AFTER_GAP",
    ]
    s, events = feed(s, sample(status="Uncertain"), sample(missing=True))
    assert s["stable"] is True and s["communication"] == "UNKNOWN"
    assert [e["kind"] for e in events] == ["COMMUNICATION_LOST"]
    s, events = feed(s, sample(True), sample(True))
    assert [e["kind"] for e in events] == ["COMMUNICATION_RESTORED"]


def test_long_gap_and_changed_binding_require_rebaseline():
    s, _ = feed({}, sample(at=100), sample(at=101))
    s, events = feed(s, sample(True, at=200), sample(True, at=201))
    assert events[0]["kind"] == "FAULT_OBSERVED_AFTER_GAP"
    a = sample(True, at=202)
    a["values"][0]["node"] = "replacement"
    s, events = feed(s, a, {**a, "observed_at": 203})
    assert [e["kind"] for e in events] == ["BINDING_CHANGED", "BASELINE_FAULT"]


def test_persistence_scope_and_atomic_handoff(tmp_path):
    path = tmp_path / "events.db"
    target = {"endpoint": "http://aas", "asset_id": "urn:one"}
    monitor = Monitor(path, lambda _: sample())
    for frame in (sample(), sample(), sample(True), sample(True)):
        monitor.ingest(target, frame)
    event = monitor.status(target)["events"][0]
    restored = Monitor(path, lambda _: sample())
    assert restored.status(target)["events"][0]["id"] == event["id"]
    assert not restored.status(target)["running"]
    assert not restored.status({**target, "endpoint": "http://other"})["events"]
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(restored.claim(target, event["id"])))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(e is not None for e in results) == 1


def test_worker_duplicate_start_and_stop(tmp_path):
    ready = threading.Event()

    def read(_):
        ready.set()
        return sample()

    monitor = Monitor(tmp_path / "worker.db", read, interval=0.01)
    target = {"endpoint": "http://aas", "asset_id": "urn:one"}
    try:
        monitor.start(target)
        assert ready.wait(2)
        worker = monitor.workers[monitor.key(target)][1]
        monitor.start(target)
        assert monitor.workers[monitor.key(target)][1] is worker
        monitor.stop(target)
        worker.join(2)
        assert not monitor.status(target)["running"]
    finally:
        monitor.close()


@pytest.mark.parametrize(
    "value,data_type,status",
    [
        ("true", "Boolean", "Good"),
        (1, "Int16", "Good"),
        (True, "Boolean", "Bad"),
        (False, "Boolean", "Uncertain"),
    ],
)
def test_invalid_critical_evidence_is_unknown(value, data_type, status):
    s, _ = feed({}, sample(), sample())
    bad = sample()
    bad["values"][0].update(value=value, dataType=data_type, status=status)
    s, events = feed(s, bad, bad)
    assert s["stable"] is False and s["communication"] == "UNKNOWN"
    assert [e["kind"] for e in events] == ["COMMUNICATION_LOST"]


def test_rearm_does_not_replay_existing_episode(tmp_path):
    monitor = Monitor(tmp_path / "events.db", lambda _: sample(True))
    target = {"asset_id": "urn:one", "endpoint": "http://aas"}
    for frame in (sample(), sample(), sample(True), sample(True)):
        monitor.ingest(target, frame)
    event = monitor.status(target)["events"][0]
    assert monitor.claim({**target, "asset_id": "urn:other"}, event["id"]) is None
    assert monitor.claim(target, event["id"])
    monitor = Monitor(tmp_path / "events.db", lambda _: sample(True))
    for frame in (sample(True), sample(True)):
        monitor.ingest(target, frame)
    assert len(monitor.status(target)["events"]) == 1
    assert monitor.claim(target, event["id"]) is None

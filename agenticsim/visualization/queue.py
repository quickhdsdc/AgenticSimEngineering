"""Explicit FIFO Study-to-playback mapping, with source and consistency checks."""
import hashlib
import json
import math
from pathlib import Path

from agenticsim.adapters.des import SingleServer
from agenticsim.domain import EngineeringTask, Evaluation, assess, require


def digest(payload):
    canonical = json.dumps(payload, sort_keys=True, allow_nan=False, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def queue_replay(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(record, dict) and isinstance(record.get("payload"), dict), "Study envelope required")
    study = record["payload"]
    require(record.get("sha256") == digest(study), "Study payload checksum mismatch")
    require(study.get("schema_version") == "1.0", "unsupported Study version")
    require(study.get("status") in ("completed", "completed_with_errors"), "Study must be terminal before playback")
    task = EngineeringTask(study["task"])
    adapter = SingleServer()
    adapter.validate(task)
    require(len(task.document["inputs"]["service"]["values"]) <= 250, "FIFO viewer supports at most 250 jobs")
    candidates = {c["id"]: c for c in task.document["candidates"]}
    runs = study.get("runs")
    require(isinstance(runs, list) and len(runs) == len(candidates), "one run per candidate required")
    require(all(isinstance(r, dict) for r in runs), "run records must be objects")
    require({r.get("candidate_id") for r in runs} == set(candidates), "run/candidate mapping mismatch")
    require(all(isinstance(r.get("run_id"), str) and r["run_id"] for r in runs)
            and len({r["run_id"] for r in runs}) == len(runs), "unique run IDs required")
    result = {
        "schema_version": "1.0", "mapping": "fifo-single-server-playback/1",
        "study_id": study["study_id"], "task_id": task.document["task_id"],
        "study_sha256": record["sha256"], "time_unit": "s",
        "input_source": task.document["context"]["source"],
        "description": task.document["context"]["description"],
        "decision": study["decision"], "model": task.document["model"],
        "scene": {"id": "fifo-schematic", "geometry_source": "synthetic_fixture",
                  "length_unit": "m", "frame": "right-handed-Y-up",
                  "placement": "schematic; positions are not measured or simulated trajectories",
                  "objects": [{"id": "server", "kind": "single-server", "capacity": 1}]},
        "runs": [],
    }
    for run in runs:
        result["runs"].append(_run(task, run, candidates[run["candidate_id"]], adapter))
    accepted = [r for r in result["runs"] if r.get("assessment", {}).get("status") == "accepted"]
    failed = any(r["status"] != "succeeded" for r in result["runs"])
    direction = task.document["objective"]["direction"]
    best = (min if direction == "minimize" else max)(accepted, key=lambda r: r["assessment"]["objective"]) if accepted else None
    require(study["status"] == ("completed_with_errors" if failed else "completed"), "Study/run status mismatch")
    require(study["decision"] == {
        "status": "inconclusive" if failed or not best else "candidate_selected",
        "best_observed_candidate": best["candidate_id"] if best else None,
        "scope": "supplied_candidates_and_model_constraints_only",
        "physical_validation": "not_established",
    }, "decision does not match candidate assessments")
    return result


def _run(task, run, candidate, adapter):
    mapped = {"run_id": run["run_id"], "candidate_id": run["candidate_id"], "status": run["status"]}
    if run["status"] != "succeeded":
        require(run["status"] in ("failed", "timed_out", "cancelled"), "unsupported run status")
        require("result" not in run, "failed run must not carry successful playback data")
        mapped.update(error=run.get("error", {}), jobs=[], events=[], event_times=[], duration=None)
        return mapped
    evaluation = Evaluation.from_dict(run["result"])
    evaluation.validate(adapter.model)
    require(evaluation.provenance.get("input_source") == task.document["context"]["source"], "input provenance mismatch")
    require(run.get("assessment") == assess(task, evaluation), "assessment does not match observations")
    observations = {o.semantic_id: o for o in evaluation.observations}
    departures = observations["urn:ase:queue:departure"]
    services = task.document["inputs"]["service"]["values"]
    require(departures.axis == tuple(range(len(services))) and departures.axis_unit == "job_index",
            "departure axis must enumerate all jobs")
    arrival, previous = 0.0, 0.0
    jobs, events = [], []
    scale = candidate["parameters"]["service_scale"]["value"]
    for i, (inter, service, depart) in enumerate(zip(task.document["inputs"]["interarrival"]["values"],
                                                  services, departures.values)):
        arrival += inter
        start = max(arrival, previous)
        require(math.isclose(depart, start + service * scale, rel_tol=1e-10, abs_tol=1e-10),
                "departure times contradict FIFO model and candidate parameters")
        jobs.append({"id": f"job-{i}", "index": i, "arrival": arrival, "start": start, "departure": depart})
        events.extend({"time": time, "job_id": f"job-{i}", "kind": kind}
                      for kind, time in (("arrived", arrival), ("started", start), ("departed", depart)))
        previous = depart
    makespan = observations["urn:ase:queue:makespan"].values[0]
    require(math.isclose(makespan, previous, rel_tol=1e-10, abs_tol=1e-10), "makespan/departure mismatch")
    # Stable presentation order only; original engine callback ordering is not inferred.
    order = {"departed": 0, "arrived": 1, "started": 2}
    events.sort(key=lambda e: (e["time"], order[e["kind"]], e["job_id"]))
    mapped.update(source=evaluation.source, assessment=run["assessment"], provenance=evaluation.provenance,
                  service_scale=scale, jobs=jobs, events=events,
                  event_times=sorted({0.0, *(e["time"] for e in events)}), duration=makespan)
    return mapped

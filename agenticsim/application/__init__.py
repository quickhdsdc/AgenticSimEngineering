"""Deterministic finite-candidate study service, independent of agents and solvers."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

from agenticsim.domain import EngineeringTask, assess
from agenticsim.ports import Evaluator, Executor, StudyStore


def now():
    return datetime.now(timezone.utc).isoformat()


def compare(document: dict, evaluator: Evaluator, executor: Executor, store: StudyStore) -> dict:
    task = EngineeringTask(deepcopy(document))
    evaluator.validate(task)  # Validate every candidate before any worker starts.
    study = store.create(task.document)
    study["model_definition"] = asdict(evaluator.model)
    study["backend_descriptor"] = asdict(evaluator.backend)
    # Store the same JSON-compatible representation returned to callers.
    import json
    study = json.loads(json.dumps(study, allow_nan=False))
    study["status"] = "running"
    store.save(study)
    for candidate in task.document["candidates"]:
        run = {"run_id": str(uuid4()), "attempt": 1, "candidate_id": candidate["id"],
               "status": "running", "started_at": now()}
        study["runs"].append(run)
        store.save(study)
        started = monotonic()
        try:
            result = executor.evaluate(task, candidate)
            result.validate(evaluator.model)
            assessment = assess(task, result)
            run.update(status="succeeded", result=result.to_dict(), assessment=assessment)
        except TimeoutError as exc:
            run.update(status="timed_out", error={"type": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            run.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        run["finished_at"] = now()
        run["wall_seconds"] = monotonic() - started
        store.save(study)
    accepted = [r for r in study["runs"] if r.get("assessment", {}).get("status") == "accepted"]
    failed = any(r["status"] != "succeeded" for r in study["runs"])
    direction = task.document["objective"]["direction"]
    best = (min if direction == "minimize" else max)(accepted, key=lambda r: r["assessment"]["objective"]) if accepted else None
    study.update(status="completed_with_errors" if failed else "completed", finished_at=now(),
                 decision={"status": "inconclusive" if failed or not best else "candidate_selected",
                           "best_observed_candidate": best["candidate_id"] if best else None,
                           "scope": "supplied_candidates_and_model_constraints_only",
                           "physical_validation": "not_established"})
    store.save(study)
    return study

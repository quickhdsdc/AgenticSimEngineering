"""The HTTP surface — the boundary where the framework's rules must survive.

Three things are under test, and the first two are the ones that would fail
silently:

* **the codec is reversible.** Decisions in this framework are indexed by
  tuples and JSON keys are strings, so every crossing is a lossy conversion
  unless someone checks. Measured before the service existed: ``Decision`` and
  ``SupplyPlan`` both raise on ``json.dumps``;
* **the capability gate survives the trip.** ``check_capabilities`` protects a
  call; the registry has to protect a *menu*, one step earlier, or the UI will
  offer a control that cannot work. Including the requirement capabilities do
  not express — a solver that enumerates a neighbourhood needs a decision space
  that has one;
* **a job can actually be stopped.** A timeout that cannot interrupt
  branch-and-bound is decoration.
"""

from __future__ import annotations

import json
import math
import time

import pytest

from opsworld.service import codec
from opsworld.service.jobs import JobRunner
from opsworld.service.registry import (
    ParamSpec,
    Registry,
    ScenarioSpec,
    SolverSpec,
    WorldSpec,
)
from opsworld.world import Capability, Evaluation, Sensitivity, SensitivityKind


# ----------------------------------------------------------------------
# codec
# ----------------------------------------------------------------------
@pytest.mark.parametrize("key", [
    ("A1", "L1"),
    ("A1", "R1", "L6", "L1", "B1", "B1"),
    ("L4",),
    ("item with, comma", 'and "quotes"'),
    ("含中文的名字", "L1"),
])
def test_index_keys_round_trip(key):
    assert codec.decode_key(codec.encode_key(key)) == key


def test_a_scalar_key_round_trips_as_a_one_tuple():
    """The decoder never has to guess arity, so scalars are wrapped."""
    assert codec.decode_key(codec.encode_key("L1")) == ("L1",)


def test_encoded_keys_are_valid_json_object_keys():
    table = {("A1", "L1"): 1.0, ("A2", "L2"): 2.0}
    assert json.loads(json.dumps(codec.encode_table(table)))


def test_a_delimiter_join_would_have_been_ambiguous():
    """Why the key is a JSON array and not ``"|".join(...)``."""
    left = codec.encode_key(("a|b", "c"))
    right = codec.encode_key(("a", "b|c"))
    assert left != right
    assert codec.decode_key(left) != codec.decode_key(right)


def test_index_map_preserves_non_numeric_values():
    """A supply plan maps an index to a *warehouse name*; coercing to float
    would destroy it, which is why the numeric table is the specialisation."""
    plan_like = {("A1", "R1"): "PL1", ("A1", "PL1"): "KLT"}
    assert codec.decode_index_map(codec.encode_index_map(plan_like)) == plan_like
    with pytest.raises(ValueError):
        codec.decode_table(codec.encode_index_map(plan_like))


def test_decode_key_refuses_something_it_did_not_write():
    with pytest.raises(ValueError, match="not an encoded index"):
        codec.decode_key("A1|L1")


def test_non_finite_numbers_travel_as_null():
    """``inf`` is a real output here — an all-zero gradient reports infinite
    relative noise — and bare ``Infinity`` is not valid JSON."""
    assert codec.jsonable(math.inf) is None
    assert codec.jsonable(math.nan) is None
    assert codec.jsonable({"a": [1.0, math.inf]}) == {"a": [1.0, None]}
    json.dumps(codec.jsonable({"noise": math.inf}))  # must not raise


def test_encoded_evaluation_keeps_the_violation_magnitudes():
    evaluation = Evaluation(
        objective=12.5,
        kpis={"cost_storage": 4.0},
        violations={"11_kanban_two_carriers": 3.25},
        provenance="test",
    )
    payload = codec.encode_evaluation(evaluation)
    assert payload["feasible"] is False
    assert payload["violations"]["11_kanban_two_carriers"] == 3.25
    assert payload["total_violation"] == 3.25
    json.dumps(payload)


def test_encoded_sensitivity_keeps_its_kind():
    """D10: a client must be able to tell a dual from a pathwise gradient."""
    dual = Sensitivity(kind=SensitivityKind.DUAL, names=["c1"], values=[2.0])
    assert codec.encode_sensitivity(dual)["kind"] == "dual"
    assert codec.encode_sensitivity(None) is None


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def test_numeric_params_clamp_and_enums_refuse():
    """A slider dragged past its limit means 'as far as it goes'; a value that
    is not on an enum is a genuine mismatch."""
    n = ParamSpec("n", "int", 4, "n", minimum=1, maximum=10)
    assert n.coerce(99) == 10
    assert n.coerce(-5) == 1
    assert n.coerce("7") == 7
    with pytest.raises(ValueError, match="expected a int"):
        n.coerce("many")

    kind = ParamSpec("kind", "enum", "wct", "kind", choices=("wct", "tardiness"))
    assert kind.coerce("tardiness") == "tardiness"
    with pytest.raises(ValueError, match="not one of"):
        kind.coerce("makespan")


def test_unknown_parameters_are_refused_rather_than_ignored():
    spec = ScenarioSpec("s", "s", "", (ParamSpec("a", "int", 1, "a"),),
                        build=lambda a: a, summarise=lambda i: {})
    with pytest.raises(ValueError, match="unknown parameter"):
        spec.make({"b": 2})


def _tiny_registry():
    registry = Registry()
    registry.add_scenario(ScenarioSpec(
        "s", "s", "", (ParamSpec("n", "int", 2, "n", minimum=1, maximum=5),),
        build=lambda n: {"n": n}, summarise=lambda i: dict(i)))
    return registry


def test_a_world_cannot_name_a_scenario_that_is_not_there():
    registry = _tiny_registry()
    with pytest.raises(KeyError, match="not registered"):
        registry.add_world(WorldSpec(
            "w", "w", "", "missing", build=lambda i: i,
            encode_decision=lambda w, d: d, decode_decision=lambda w, d: d))


def test_duplicate_registration_is_refused():
    registry = _tiny_registry()
    with pytest.raises(KeyError, match="already registered"):
        registry.add_scenario(ScenarioSpec("s", "s", "", (), build=dict,
                                           summarise=lambda i: {}))


class _Space:
    def initial(self):
        return [0.0]

    def perturb(self, decision, rng):
        return list(decision)


class _SpaceWithMoves(_Space):
    def moves(self, decision):
        return [list(decision)]


def test_solvers_are_filtered_by_capability():
    registry = _tiny_registry()
    registry.add_solver(SolverSpec("c0", "c0", "", frozenset({Capability.EVALUATE}),
                                   build=lambda: object()))
    registry.add_solver(SolverSpec(
        "grad", "grad", "",
        frozenset({Capability.EVALUATE, Capability.PATHWISE_GRADIENT}),
        build=lambda: object()))

    offered = registry.solvers_for({Capability.EVALUATE}, _Space())
    assert [s.id for s in offered] == ["c0"]
    offered = registry.solvers_for(
        {Capability.EVALUATE, Capability.PATHWISE_GRADIENT}, _Space())
    assert {s.id for s in offered} == {"c0", "grad"}


def test_solvers_are_also_filtered_by_what_the_decision_space_supports():
    """The gap the HTTP surface exposed.

    A neighbourhood-enumerating solver truthfully declares only ``C0`` — it
    evaluates and nothing more — yet it calls ``space.moves()``. Offered
    against a weight vector it would clear the capability gate and then fail
    with an ``AttributeError``, so the requirement has to be named.
    """
    registry = _tiny_registry()
    registry.add_solver(SolverSpec(
        "enumerating", "enumerating", "", frozenset({Capability.EVALUATE}),
        build=lambda: object(), requires_space=frozenset({"moves"})))

    assert registry.solvers_for({Capability.EVALUATE}, _Space()) == []
    assert [s.id for s in registry.solvers_for({Capability.EVALUATE},
                                               _SpaceWithMoves())] == ["enumerating"]


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------
def _await(runner: JobRunner, job, limit: float = 60.0):
    deadline = time.time() + limit
    while time.time() < deadline:
        current = runner.get(job.id)
        if current.status in ("done", "failed", "cancelled"):
            return current
        time.sleep(0.05)
    raise AssertionError(f"job {job.id} still {runner.get(job.id).status}")


ECHO = "opsworld.service._selftest.echo"
BOOM = "opsworld.service._selftest.boom"
FOREVER = "opsworld.service._selftest.sleep_forever"


@pytest.mark.slow
def test_a_job_runs_in_another_process_and_comes_back_as_json():
    runner = JobRunner(max_concurrent=1)
    try:
        done = _await(runner, runner.submit("echo", ECHO, {"a": 1}))
        assert done.status == "done"
        assert done.result == {"echoed": {"a": 1}, "ok": True}
        json.dumps(done.as_json())
    finally:
        runner.shutdown()


@pytest.mark.slow
def test_a_failing_job_carries_the_worker_traceback_back():
    runner = JobRunner(max_concurrent=1)
    try:
        failed = _await(runner, runner.submit("boom", BOOM, {"message": "nope"}))
        assert failed.status == "failed"
        assert "ValueError: nope" in failed.error
        assert "Traceback" in failed.error
    finally:
        runner.shutdown()


@pytest.mark.slow
def test_the_timeout_is_enforced_from_outside_the_worker():
    """Branch-and-bound does not poll a flag, so the deadline cannot be
    cooperative — it has to terminate the process."""
    runner = JobRunner(max_concurrent=1)
    try:
        job = runner.submit("forever", FOREVER, {"seconds": 600}, timeout=3.0)
        finished = _await(runner, job, limit=45)
        assert finished.status == "failed"
        assert "timed out" in finished.error
        assert finished.elapsed < 30
    finally:
        runner.shutdown()


@pytest.mark.slow
def test_a_running_job_can_actually_be_cancelled():
    runner = JobRunner(max_concurrent=1)
    try:
        job = runner.submit("forever", FOREVER, {"seconds": 600}, timeout=600)
        deadline = time.time() + 45
        while runner.get(job.id).status != "running" and time.time() < deadline:
            time.sleep(0.05)
        assert runner.get(job.id).status == "running"
        cancelled = runner.cancel(job.id)
        assert cancelled.status == "cancelled"
        assert "cancelled" in (cancelled.error or "")
    finally:
        runner.shutdown()


def test_cancelling_an_unknown_job_says_so():
    runner = JobRunner(max_concurrent=1)
    try:
        with pytest.raises(KeyError, match="unknown job"):
            runner.get("nope")
    finally:
        runner.shutdown()


# ----------------------------------------------------------------------
# the app itself — only where the optional dependency is installed
# ----------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi", reason="uv sync --group service")


@pytest.fixture(scope="module")
def client():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from fastapi.testclient import TestClient
    from serve_api import build_app

    with TestClient(build_app(max_concurrent=1, timeout=60)) as test_client:
        yield test_client


def test_schema_computes_the_solver_menu_per_world(client):
    """The contract a front end is written against: which solvers a world model
    can serve is a fact the server derives, not a list someone maintains."""
    schema = client.get("/api/schema").json()
    worlds = {w["id"]: w for w in schema["worlds"]}

    assert "C2a" in worlds["intralogistics.sizing"]["capabilities"]
    assert "gradient" in worlds["intralogistics.sizing"]["solvers"]

    assert worlds["intralogistics.plan"]["capabilities"] == ["C0"]
    assert "gradient" not in worlds["intralogistics.plan"]["solvers"]
    assert "bayes" not in worlds["intralogistics.plan"]["solvers"]
    assert "bayes" in worlds["intralogistics.sizing"]["solvers"]
    # ... and the decision-space requirement, which capabilities do not express
    assert "greedy-supply" in worlds["intralogistics.plan"]["solvers"]
    assert "greedy-supply" not in worlds["gradientshop.dispatch"]["solvers"]

    for world in schema["worlds"]:
        assert "error" not in world, world


def test_a_plan_survives_the_round_trip(client):
    """The decision that could not be serialised at all before the codec."""
    scenario = {"n_items": 4, "n_resources": 2}
    first = client.post("/api/initial", json={"world": "intralogistics.plan",
                                             "scenario": scenario}).json()
    again = client.post("/api/evaluate", json={"world": "intralogistics.plan",
                                               "scenario": scenario,
                                               "decision": first["decision"]}).json()
    assert again["evaluation"]["objective"] == pytest.approx(
        first["evaluation"]["objective"])
    assert again["evaluation"]["feasible"] == first["evaluation"]["feasible"]


def test_the_gradient_endpoint_refuses_a_world_without_one(client):
    response = client.post("/api/gradient", json={"world": "intralogistics.plan"})
    assert response.status_code == 400
    assert "C2a" in response.json()["detail"]


def test_the_gradient_endpoint_marks_its_result_pathwise(client):
    payload = client.post("/api/gradient",
                          json={"world": "intralogistics.sizing"}).json()
    assert payload["sensitivity"]["kind"] == "pathwise"
    assert len(payload["descent_direction"]) == len(payload["sensitivity"]["names"])


def test_a_small_run_on_a_fast_world_answers_inline(client):
    """Spawning a worker costs ~3.8 s here against ~20 ms of work, so 'always a
    job' would make the fast world models two orders of magnitude slower."""
    started = time.perf_counter()
    payload = client.post("/api/solve", json={"world": "intralogistics.sizing",
                                              "solver": "gradient",
                                              "budget": 40}).json()
    assert payload["mode"] == "inline"
    assert payload["status"] == "done"
    assert payload["result"]["score"] < payload["result"]["initial_score"]
    assert len(payload["result"]["history"]) > 1
    assert time.perf_counter() - started < 3.0


def test_a_slow_world_is_queued_with_a_readable_reason(client):
    payload = client.post("/api/solve", json={"world": "intralogistics.plan",
                                              "solver": "greedy-supply",
                                              "budget": 100000}).json()
    assert payload["mode"] == "job"
    assert "inline limit" in payload["reason"]
    client.delete(f"/api/jobs/{payload['id']}")


def test_the_tau_sweep_returns_the_disagreement_in_one_payload(client):
    """The cosine and the operational verdict contradict each other (D15);
    a UI can only show that if both arrive together."""
    payload = client.post("/api/relaxation/tau",
                          json={"scenario": {"n_jobs": 12, "seed": 7}}).json()
    rows = {round(r["tau"], 3): r for r in payload["rows"]}
    assert rows[2.0]["grad_cosine"] > 0.99 and rows[2.0]["descends"] is False
    assert rows[0.01]["grad_cosine"] < -0.9 and rows[0.01]["descends"] is True


def test_the_verdict_is_answerable_for_a_world_with_no_gradient(client):
    """200 with ``applicable: false``, not a 400.

    Asking for a gradient that does not exist is a client error; asking whether
    stepping helps has an answer for every world model — "there is nothing to
    step along". A client that receives an error has to invent a display, and
    inventing one is how a stale verdict from a previously selected world ends
    up on screen.
    """
    response = client.post("/api/verdict", json={"world": "intralogistics.plan",
                                                 "scenario": {"n_items": 4}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["applicable"] is False
    assert payload["capabilities"] == ["C0"]
    assert "C2a" in payload["reason"]
    assert "descends" not in payload  # nothing to report, so nothing reported

    # ... while the gradient endpoint still refuses, because that IS an error
    assert client.post("/api/gradient",
                       json={"world": "intralogistics.plan"}).status_code == 400


def test_duals_do_not_count_as_a_direction(client):
    """D10 at the boundary: the MILP has C2b and that is not steppable."""
    payload = client.post("/api/verdict",
                          json={"world": "intralogistics.milp",
                                "scenario": {"n_items": 3}}).json()
    assert payload["applicable"] is False
    assert "C2b" in payload["capabilities"]


def test_the_verdict_carries_a_trace_and_names_what_it_stepped(client):
    payload = client.post("/api/verdict",
                          json={"world": "intralogistics.sizing"}).json()
    assert payload["applicable"] is True
    assert payload["descends"] is True
    assert payload["best_value"] < payload["start"]
    assert len(payload["steps"]) == len(payload["values"])
    # the space is named because this verdict and the tau sweep's disagree
    assert payload["space"]["names"]
    assert "different question" in payload["space"]["note"]


def test_an_unknown_world_is_a_404_not_a_500(client):
    response = client.post("/api/solve", json={"world": "nope", "solver": "random"})
    assert response.status_code == 404


def test_the_algebraic_world_is_offered_its_exact_solver(client):
    """C3's whole purpose is to unlock an exact solver, and for a while nothing
    consumed it: ``solve_exact()`` is a method on the world model while the menu
    is built from ``Solver`` objects, so the registry offered random search on
    the one model that can *prove* optimality. The adapter closes that."""
    worlds = {w["id"]: w for w in client.get("/api/schema").json()["worlds"]}

    assert "C3" in worlds["intralogistics.milp"]["capabilities"]
    assert "exact" in worlds["intralogistics.milp"]["solvers"]
    # ... and it is offered nowhere else, because nothing else has the algebra
    for world_id, world in worlds.items():
        if world_id != "intralogistics.milp":
            assert "exact" not in world["solvers"], world_id


def test_the_exact_solver_reports_whether_it_proved_optimality(client):
    """A time-limited run returns a feasible incumbent with no proof attached.
    Calling that "the optimum" is the mistake the wrapper exists to make hard,
    so the termination status travels with the result."""
    submitted = client.post("/api/solve", json={
        "world": "intralogistics.milp", "solver": "exact",
        "scenario": {"n_items": 2, "n_resources": 2},
        "solver_params": {"time_limit": 60.0}, "timeout": 180,
    }).json()
    assert submitted["mode"] == "job"

    for _ in range(600):
        status = client.get(f"/api/jobs/{submitted['id']}").json()
        if status["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.25)

    assert status["status"] == "done", status.get("error")
    provenance = status["result"]["evaluation"]["provenance"]
    assert "optimal" in provenance, provenance
    assert status["result"]["evaluation"]["feasible"]
    assert status["result"]["evaluations_used"] == 2


def test_the_exact_solver_refuses_a_world_without_the_algebra(client):
    """The capability gate, on the path a client would actually hit."""
    submitted = client.post("/api/solve", json={
        "world": "intralogistics.plan", "solver": "exact", "budget": 1,
    }).json()
    for _ in range(200):
        status = client.get(f"/api/jobs/{submitted['id']}").json()
        if status["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert status["status"] == "failed"
    assert "C3" in status["error"]

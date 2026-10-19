"""The HTTP surface. Small on purpose.

Every endpoint is either a *fast* call answered in the request, or a *job*
submitted to :mod:`opsworld.service.jobs`. Which one a given world model gets
is not a judgement call in the front end — the registry says so, from measured
latencies, and the client reads it off ``/api/schema``.

The one design decision worth stating: **the server holds no session state.**
An instance is a pure function of ``(scenario id, params)``, so a request that
needs one rebuilds it — a few milliseconds — rather than referring to a
server-side handle. That makes every response reproducible outside the server,
lets the process restart without losing a client's work, and means the
slow-scale environment can be stepped by sending the state back rather than by
the server remembering whose episode is whose.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from .jobs import JobRunner
from .registry import Registry

__all__ = ["create_app", "run_solver_job"]


# ----------------------------------------------------------------------
# request bodies
# ----------------------------------------------------------------------
# Module level, not inside create_app(). With `from __future__ import
# annotations` every annotation is a string, and FastAPI resolves them against
# the *module* namespace -- a model defined in a local scope is invisible
# there, and every endpoint silently degrades to expecting a query parameter
# (a 422 on a perfectly good body). Pydantic is already a hard dependency;
# only FastAPI and uvicorn are optional, and they stay inside the function.
class ScenarioBody(BaseModel):
    scenario: str
    params: dict[str, Any] = Field(default_factory=dict)


class WorldBody(BaseModel):
    world: str
    scenario: dict[str, Any] = Field(default_factory=dict)


class DecisionBody(WorldBody):
    decision: Any = None


class SolveBody(WorldBody):
    solver: str
    budget: int = 100
    solver_params: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    #: Queue it even if it would have run inline — for testing the job path,
    #: and for a client that wants to keep its request short.
    force_job: bool = False


class VerdictBody(WorldBody):
    decision: Any = None
    step_sizes: list[float] = Field(
        default_factory=lambda: [0.02, 0.05, 0.1, 0.25, 0.5, 1.0])


class SweepBody(BaseModel):
    scenario: dict[str, Any] = Field(default_factory=dict)
    taus: list[float] = Field(
        default_factory=lambda: [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])


# ----------------------------------------------------------------------
# the job entry point — resolved and executed inside the worker process
# ----------------------------------------------------------------------
def run_solver_job(payload: dict) -> dict:
    """Rebuild the world from its recipe, run the solver, return plain JSON.

    Imported by name in a fresh process, so everything it needs arrives as
    strings and numbers. Rebuilding costs milliseconds and buys the property
    that a job is reproducible from its payload alone.
    """
    from opsworld.service.codec import encode_solver_result

    registry = _registry_from(payload["domains"])
    spec = registry.world(payload["world"])
    world, instance, _ = registry.build_world(payload["world"], payload.get("scenario"))
    solver_spec = registry.solver(payload["solver"])
    solver, _ = solver_spec.make(payload.get("solver_params"))

    missing = set(solver_spec.requires) - set(world.capabilities())
    if missing:
        raise TypeError(
            f"{solver_spec.id} requires {sorted(c.value for c in missing)}, which "
            f"{spec.id} does not provide"
        )
    if not solver_spec.serves(set(world.capabilities()), world.decision_space()):
        raise TypeError(
            f"{solver_spec.id} needs a decision space providing "
            f"{sorted(solver_spec.requires_space)}; {spec.id}'s does not"
        )

    started = time.perf_counter()
    result = solver.solve(world, budget=int(payload.get("budget", 100)))
    encoded = encode_solver_result(
        result, decision=spec.encode_decision(world, result.decision)
    )
    encoded["seconds"] = round(time.perf_counter() - started, 3)
    return encoded


def _registry_from(domains: list[str]) -> Registry:
    registry = Registry()
    for name in domains:
        module = __import__(name, fromlist=["register"])
        module.register(registry)
    return registry


# ----------------------------------------------------------------------
# the app
# ----------------------------------------------------------------------
def create_app(registry: Registry, *, domains: list[str],
               runner: JobRunner | None = None):
    """Build the FastAPI app over a registry the caller has already populated.

    ``domains`` is the list of module paths whose ``register()`` built the
    registry; a job worker replays them to reconstruct it. Passing them
    explicitly (rather than inferring) keeps this module free of any domain
    knowledge, which is architecture constraint A6.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ModuleNotFoundError as exc:  # pragma: no cover - install guidance
        raise ModuleNotFoundError(
            "the HTTP surface needs the optional service dependencies: "
            "`uv sync --group service`"
        ) from exc

    runner = runner or JobRunner()

    app = FastAPI(
        title="opsworld",
        version="0.1.0",
        description=(
            "世界模型是共享资产,求解器是插件。哪些求解器能用于某个世界模型,"
            "是服务器算出来的,不是前端硬编码的。"
        ),
    )
    # A local front end is developed on a different port; this is a research
    # server bound to localhost, not a public endpoint.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    def _fail(exc: Exception, code: int = 400):
        return HTTPException(status_code=code, detail=f"{type(exc).__name__}: {exc}")

    # -- the contract -------------------------------------------------
    @app.get("/api/schema")
    def schema() -> dict:
        """Everything a front end needs to render itself.

        Includes, per world model, the solver ids it can actually serve — the
        capability gate applied at menu-build time rather than at call time.
        """
        payload = registry.as_json()
        payload["limits"] = {
            "max_concurrent_jobs": runner.max_concurrent,
            "default_timeout_seconds": runner.default_timeout,
        }
        return payload

    # -- fast paths ---------------------------------------------------
    @app.post("/api/scenario")
    def build_scenario(body: ScenarioBody) -> dict:
        try:
            spec = registry.scenario(body.scenario)
            instance, resolved = spec.make(body.params)
            return {"scenario": spec.id, "params": resolved,
                    "summary": spec.summarise(instance)}
        except (KeyError, ValueError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/initial")
    def initial(body: WorldBody) -> dict:
        """A starting decision, and what it is worth."""
        from .codec import encode_evaluation

        try:
            spec = registry.world(body.world)
            world, instance, params = registry.build_world(body.world, body.scenario)
            decision = world.decision_space().initial()
            payload = {
                "world": spec.id,
                "scenario_params": params,
                "capabilities": sorted(c.value for c in world.capabilities()),
                "decision": spec.encode_decision(world, decision),
                "evaluation": encode_evaluation(world.evaluate(decision)),
            }
            if spec.describe is not None:
                payload["describe"] = spec.describe(world, instance)
            return payload
        except (KeyError, ValueError, TypeError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/evaluate")
    def evaluate(body: DecisionBody) -> dict:
        """One decision, one verdict. The call a slider fires on every drag."""
        from .codec import encode_evaluation

        try:
            spec = registry.world(body.world)
            world, _, _ = registry.build_world(body.world, body.scenario)
            decision = (
                world.decision_space().initial() if body.decision is None
                else spec.decode_decision(world, body.decision)
            )
            started = time.perf_counter()
            evaluation = world.evaluate(decision)
            return {"evaluation": encode_evaluation(evaluation),
                    "seconds": round(time.perf_counter() - started, 4)}
        except (KeyError, ValueError, TypeError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/gradient")
    def gradient(body: DecisionBody) -> dict:
        """The pathwise gradient — refused, loudly, where the model has none."""
        from .codec import encode_sensitivity

        try:
            spec = registry.world(body.world)
            world, _, _ = registry.build_world(body.world, body.scenario)
            if not hasattr(world, "gradient"):
                raise TypeError(
                    f"{spec.id} does not expose C2a; it offers "
                    f"{sorted(c.value for c in world.capabilities())}"
                )
            decision = (
                world.decision_space().initial() if body.decision is None
                else spec.decode_decision(world, body.decision)
            )
            sensitivity = world.gradient(decision)
            return {"sensitivity": encode_sensitivity(sensitivity),
                    "descent_direction": sensitivity.descent_direction()}
        except (KeyError, ValueError, TypeError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/verdict")
    def verdict(body: VerdictBody) -> dict:
        """Would a step along the gradient actually help? Always answerable.

        Deliberately **200 with ``applicable: false``** where a world model has
        no gradient, rather than the 400 ``/api/gradient`` returns. The two
        endpoints answer different questions: asking for a gradient that does
        not exist is a client error, but asking *whether stepping helps* has a
        well-defined answer for every world model — "there is nothing to step
        along". A client that gets an error has to invent a display; one that
        gets an answer does not, and inventing one is how a stale verdict from
        a previously selected world ends up on screen.

        The check is operational because nothing else works here: the true
        objective is piecewise constant in the decision, so a finite-difference
        reference does not exist and the cosine in common use disagrees with
        the answer in both directions (decision record D15).

        **This is not the same verdict as the one in ``/api/relaxation/tau``,
        and the payload says so.** That one steps the raw priority *scores* and
        asks whether the relaxation's gradient is usable at a temperature —
        the RQ1 question. This one steps the *decision the client is editing*
        and asks whether the next move improves it — the optimisation
        question. They can and do disagree: on the documented instance at
        ``tau=2.0`` a step in score space finds nothing while a step in policy
        space improves the objective by a third. Both are true of different
        moves, so each response carries a ``space`` naming what it stepped.
        """
        from opsworld.diagnostics import descent_check
        from opsworld.world import Capability, penalised_objective

        from .codec import encode_sensitivity, jsonable

        try:
            spec = registry.world(body.world)
            world, _, _ = registry.build_world(body.world, body.scenario)
            capabilities = sorted(c.value for c in world.capabilities())

            if (Capability.PATHWISE_GRADIENT not in world.capabilities()
                    or not hasattr(world, "gradient")):
                return {
                    "world": spec.id,
                    "applicable": False,
                    "capabilities": capabilities,
                    "reason": (
                        f"{spec.id} declares {capabilities}; a descent check "
                        "needs C2a. There is no direction to step along, which "
                        "is an answer, not a failure."
                    ),
                }

            decision = (
                world.decision_space().initial() if body.decision is None
                else spec.decode_decision(world, body.decision)
            )
            sensitivity = world.gradient(decision)
            score = lambda point: penalised_objective(  # noqa: E731
                world.evaluate([float(v) for v in point]))

            try:
                report = descent_check(score, list(decision), sensitivity.values,
                                       body.step_sizes)
            except ValueError:
                # A zero gradient is not "no improvement available" -- it is no
                # local information at all, which Phase 5.3 had to distinguish
                # sharply. Say which one it is.
                return {
                    "world": spec.id, "applicable": True, "descends": False,
                    "zero_gradient": True, "capabilities": capabilities,
                    "start": jsonable(float(score(list(decision)))),
                    "reason": ("the gradient is exactly zero here — no decision "
                               "boundary is within reach, so no step size helps"),
                    "sensitivity": encode_sensitivity(sensitivity),
                }

            space = world.decision_space()
            return {
                "world": spec.id,
                "applicable": True,
                "descends": report.descends,
                "zero_gradient": False,
                "capabilities": capabilities,
                "space": {
                    "what": "the decision this client is editing",
                    "names": list(getattr(space, "names", [])),
                    "note": ("a step here moves the decision; the verdict in "
                             "/api/relaxation/tau moves the raw scores and "
                             "answers a different question — the two can "
                             "disagree at the same temperature"),
                },
                "start": jsonable(report.start),
                "best_step": report.best_step,
                "best_value": jsonable(report.best_value),
                "improvement": jsonable(report.improvement),
                "steps": [jsonable(v) for v in report.steps],
                "values": [jsonable(v) for v in report.values],
                "text": report.as_text(),
                "sensitivity": encode_sensitivity(sensitivity),
            }
        except (KeyError, ValueError, TypeError) as exc:
            raise _fail(exc) from exc

    @app.post("/api/relaxation/tau")
    def relaxation_tau(body: SweepBody) -> dict:
        """The temperature sweep, with the operational verdict beside the cosine."""
        try:
            import gradientshop.service as gs

            spec = registry.scenario("gradientshop.dispatch")
            scenario, resolved = spec.make(body.scenario)
            started = time.perf_counter()
            out = gs.tau_sweep(scenario, body.taus)
            out["params"] = resolved
            out["seconds"] = round(time.perf_counter() - started, 4)
            return out
        except (KeyError, ValueError, TypeError) as exc:
            raise _fail(exc) from exc

    # -- jobs ---------------------------------------------------------
    @app.post("/api/solve")
    def solve(body: SolveBody) -> dict:
        """Run a solver — inline when that is genuinely faster, else as a job.

        Not two code paths for the sake of it. Spawning a worker costs about
        3.8 s here (``spawn`` re-imports the entry module and torch with it),
        measured against a gradient run whose real work was 18 ms. Sending the
        fast world models through the queue would make them 200x slower, so
        each world model declares the budget it is still worth answering
        inline; ``mode`` in the response says which path was taken, and
        ``force_job`` overrides.
        """
        try:
            world_spec = registry.world(body.world)
            solver_spec = registry.solver(body.solver)
        except KeyError as exc:
            raise _fail(exc, 404) from exc

        payload = {
            "domains": list(domains),
            "world": world_spec.id,
            "solver": solver_spec.id,
            "scenario": body.scenario,
            "solver_params": body.solver_params,
            "budget": int(body.budget),
        }

        inline = (
            not body.force_job
            and 0 < int(body.budget) <= world_spec.inline_max_budget
        )
        if inline:
            try:
                result = run_solver_job(payload)
            except (KeyError, ValueError, TypeError) as exc:
                raise _fail(exc) from exc
            return {"mode": "inline", "status": "done", "result": result,
                    "kind": f"solve:{world_spec.id}:{solver_spec.id}"}

        job = runner.submit(
            kind=f"solve:{world_spec.id}:{solver_spec.id}",
            entry="opsworld.service.app.run_solver_job",
            payload=payload,
            timeout=body.timeout,
        )
        out = job.as_json()
        out["mode"] = "job"
        out["reason"] = (
            "forced by the client" if body.force_job
            else f"budget {body.budget} exceeds {world_spec.id}'s inline limit "
                 f"of {world_spec.inline_max_budget}"
        )
        return out

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [j.as_json() for j in runner.list()]}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        try:
            return runner.get(job_id).as_json()
        except KeyError as exc:
            raise _fail(exc, 404) from exc

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        try:
            return runner.cancel(job_id).as_json()
        except KeyError as exc:
            raise _fail(exc, 404) from exc

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "worlds": len(registry.worlds),
                "solvers": len(registry.solvers), "jobs": len(runner.list())}

    app.state.registry = registry
    app.state.runner = runner
    return app

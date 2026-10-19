"""DDES exposed as agent tools (RQ2).

These make the differentiable simulator a set of callable tools an LLM can invoke
inside the ReAct loop, operating on a shared ``session`` blackboard:

* ``describe_scenario``          -- the jobs, the rule basis, and the best policy so far.
* ``evaluate_policy``            -- one DDES replay: objective + gradient + verbalised critique.
* ``compare_to_reference``       -- relaxation fidelity: the differentiable (soft) schedule vs
                                    the exact discrete schedule for this policy (RQ1 for dispatch).
* ``validate_timing_fidelity``   -- engine fidelity: the DDES tandem line vs the SimPy ground truth.

All tools are ``deterministic`` (pure numerics; none call an LLM). Build the
session with :func:`build_dispatch_session` and the toolset with
:func:`build_dispatch_tools`.
"""

from __future__ import annotations

import torch

from ...ddes.dispatch import exact_dispatch_cost
from ...shopfloor import battery
from ...validation import compare_tandem_line
from opsworld.agent.tool import BaseTool, ToolCollection, ToolResult
from ..worldmodel import DispatchWorldModel

_THETA_PARAM = {
    "type": "object",
    "properties": {"theta": {"type": "array", "items": {"type": "number"},
                             "description": "one weight per dispatching rule, in the listed order"}},
    "required": ["theta"],
}


def _world(session: dict) -> DispatchWorldModel:
    world = session.get("world")
    if world is None:
        raise RuntimeError("no scenario in session; build it with build_dispatch_session().")
    return world


class DescribeScenario(BaseTool):
    name: str = "describe_scenario"
    description: str = ("Summarise the scheduling scenario: number of jobs, the objective, "
                        "the dispatching-rule basis theta weights, and the best policy found so far.")
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, session: dict) -> ToolResult:
        w = _world(session)
        best = session.get("best")
        best_txt = (f"best so far: objective={best['objective']:.3f} theta={best['theta']}"
                    if best else "no policy evaluated yet")
        return ToolResult(output=(
            f"{w.processing.shape[0]} jobs, objective = {w.objective_label}, "
            f"softsort tau = {w.tau}. Dispatching rules (theta order): {w.feature_names}. "
            f"{best_txt}. Call evaluate_policy with a theta to get a gradient critique."))


class EvaluatePolicy(BaseTool):
    name: str = "evaluate_policy"
    description: str = ("Replay the differentiable DES for a dispatch policy (weights over the "
                        "dispatching rules) and return the objective, its gradient w.r.t. each "
                        "weight, and a critique of which weights to raise or lower.")
    parameters: dict = _THETA_PARAM

    async def execute(self, session: dict, theta: list | None = None) -> ToolResult:
        w = _world(session)
        if theta is None or len(theta) != w.n_features:
            return ToolResult(error=f"theta must have {w.n_features} numbers "
                                    f"for rules {w.feature_names}.")
        step = session.get("eval_count", 0)
        obs = w.evaluate(theta, step=step)
        session["eval_count"] = step + 1
        best = session.get("best")
        if best is None or obs.objective < best["objective"]:
            session["best"] = {"objective": obs.objective, "theta": list(obs.theta)}
        return ToolResult(output=obs.as_text())


class CompareToReference(BaseTool):
    name: str = "compare_to_reference"
    description: str = ("Check relaxation fidelity: compare the differentiable (soft) schedule "
                        "objective against the EXACT discrete schedule for this policy. Tells you "
                        "whether the smooth objective can be trusted at the current temperature.")
    parameters: dict = _THETA_PARAM

    async def execute(self, session: dict, theta: list | None = None) -> ToolResult:
        w = _world(session)
        if theta is None or len(theta) != w.n_features:
            return ToolResult(error=f"theta must have {w.n_features} numbers.")
        obs = w.evaluate(theta)
        exact = obs.objective
        soft = obs.soft_objective
        gap = abs(soft - exact)
        rel = gap / exact if exact else 0.0
        verdict = ("trustworthy" if rel < 0.02 else
                   "usable" if rel < 0.05 else "biased -- lower tau or anneal")
        return ToolResult(output=(
            f"relaxation fidelity: exact discrete = {exact:.3f}, relaxed = {soft:.3f}, "
            f"gap = {gap:.3f} ({rel:.1%}) -> {verdict}."))


class ValidateTimingFidelity(BaseTool):
    name: str = "validate_timing_fidelity"
    description: str = ("Check engine fidelity: run the battery-line DDES and the independent "
                        "SimPy reference on the same sample path and report their discrepancy. "
                        "A faithful DDES matches SimPy to floating-point tolerance.")
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, session: dict) -> ToolResult:
        line = session.get("line")
        if line is None:
            return ToolResult(error="no line scenario in session.")
        report = compare_tandem_line(line["interarrival"], line["service"])
        return ToolResult(output=report.summary())


def build_dispatch_session(processing: torch.Tensor, weight: torch.Tensor,
                           due: torch.Tensor, *, tau: float = 0.1,
                           objective: str = "wct", seed: int = 0) -> dict:
    """Assemble the shared blackboard: the dispatch world model plus a battery
    tandem-line sample path for the engine-fidelity check."""
    world = DispatchWorldModel(processing, weight, due, tau=tau, objective=objective)
    cfg = battery.default_config()
    g = torch.Generator().manual_seed(seed)
    base = battery.sample_base_service(cfg, processing.shape[0], generator=g)
    n = processing.shape[0]
    interarrival = torch.full((n,), 1.0 / cfg.arrival_rate, dtype=torch.float64)
    return {"world": world, "line": {"interarrival": interarrival, "service": base}}


def build_dispatch_tools() -> ToolCollection:
    """The DDES toolset for the scheduling agent."""
    return ToolCollection(DescribeScenario(), EvaluatePolicy(),
                          CompareToReference(), ValidateTimingFidelity())

"""Intralogistics exposed as agent tools — RQ2 in a second domain.

The shared ReAct harness (:mod:`opsworld.agent`) already knows how to run a
model against a toolset; a domain joins by wrapping its world model in tools.
That is the whole cost of adding a domain to the agent story, and it is what
this module demonstrates: no second harness, no second persona framework, no
second audit format.

Tools, deliberately mirroring :mod:`gradientshop.agent.tools.ddes_tools` so the
two domains present the same shape of interface to a model:

* ``describe_scenario``      -- the items, the named sizing rules, the best policy so far
* ``evaluate_policy``        -- one replay: cost + gradient + a verbalised critique
* ``inspect_capacities``     -- what capacities the policy actually prescribes, per item
* ``compare_to_simulation``  -- the surrogate gap: fluid training signal vs simulation
* ``check_gradient_health``  -- saturation diagnostic; is the gradient trustworthy here?

All tools are deterministic; none call an LLM. The last one exists because of
the Phase 4 correction: a gradient from this domain is only meaningful while
the trajectory stays clear of the reorder threshold, and an agent that is going
to act on gradients should be able to ask whether they are sound.
"""

from __future__ import annotations

import torch

from opsworld.agent.tool import BaseTool, ToolCollection, ToolResult

from ..differentiable import saturation_fraction
from ..policy import SizingPolicyWorld

__all__ = ["build_sizing_session", "build_sizing_tools"]

_THETA_PARAM = {
    "type": "object",
    "properties": {
        "theta": {
            "type": "array",
            "items": {"type": "number"},
            "description": "one weight per named sizing rule, in the listed order",
        }
    },
    "required": ["theta"],
}


def _world(session: dict) -> SizingPolicyWorld:
    world = session.get("world")
    if world is None:
        raise RuntimeError(
            "no scenario in session; build it with build_sizing_session()."
        )
    return world


class DescribeScenario(BaseTool):
    name: str = "describe_scenario"
    description: str = (
        "Summarise the material-supply scenario: how many items, the named "
        "carrier-sizing rules that the policy weights, the admissible capacity "
        "range, and the best policy found so far."
    )
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, session: dict) -> ToolResult:
        world = _world(session)
        best = session.get("best")
        best_text = (
            f"best so far: cost={best['objective']:.4f} theta={best['theta']}"
            if best
            else "no policy evaluated yet"
        )
        return ToolResult(output=(
            f"{len(world.items)} items competing for storage. Decision: one load-carrier "
            f"capacity per item, between {world.min_capacity:.0f} and "
            f"{world.max_capacity:.0f} units, set by a weighted blend of named rules. "
            f"Rules (theta order): {world.rule_names}. A positive weight pushes toward "
            f"LARGER carriers. {best_text}. Call evaluate_policy with a theta to get a "
            f"gradient critique."
        ))


class EvaluatePolicy(BaseTool):
    name: str = "evaluate_policy"
    description: str = (
        "Replay the material supply under a sizing policy. Returns the total cost, "
        "the gradient of that cost with respect to each rule weight, and a written "
        "critique naming which weights to raise or lower. Negative gradient on a "
        "rule means increasing that weight LOWERS cost."
    )
    parameters: dict = _THETA_PARAM

    async def execute(self, session: dict, theta: list[float]) -> ToolResult:
        world = _world(session)
        if len(theta) != len(world.rule_names):
            return ToolResult(error=(
                f"expected {len(world.rule_names)} weights for {world.rule_names}, "
                f"got {len(theta)}"
            ))
        evaluation = world.evaluate(theta)
        best = session.get("best")
        if best is None or evaluation.objective < best["objective"]:
            session["best"] = {"objective": evaluation.objective, "theta": list(theta)}
        session.setdefault("history", []).append(evaluation.objective)
        return ToolResult(output=evaluation.as_text())


class InspectCapacities(BaseTool):
    name: str = "inspect_capacities"
    description: str = (
        "Show the load-carrier capacity this policy prescribes for each item, "
        "alongside the item attributes, so the rule can be sanity-checked "
        "against physical intuition."
    )
    parameters: dict = _THETA_PARAM

    async def execute(self, session: dict, theta: list[float]) -> ToolResult:
        world = _world(session)
        capacity = world.capacities(theta)
        items = world.items
        order = torch.argsort(capacity, descending=True)
        lines = [world.explain(theta), "", "per item (largest carrier first):",
                 f"  {'item':>5}{'capacity':>10}{'demand/period':>15}"
                 f"{'value':>9}{'area':>8}"]
        for rank in order[:8].tolist():
            lines.append(
                f"  {rank:>5}{capacity[rank]:>10.1f}{items.demand_rate[rank]:>15.2f}"
                f"{items.unit_value[rank]:>9.2f}{items.unit_area[rank]:>8.3f}"
            )
        if len(order) > 8:
            lines.append(f"  ... and {len(order) - 8} more items")
        return ToolResult(output="\n".join(lines))


class CompareToSimulation(BaseTool):
    name: str = "compare_to_simulation"
    description: str = (
        "Check the surrogate gap. The gradient comes from a smooth fluid model; "
        "this scores the same policy on the full replenishment simulation and "
        "reports how far apart they are. A large gap means the training signal "
        "is misleading for this policy."
    )
    parameters: dict = _THETA_PARAM

    async def execute(self, session: dict, theta: list[float]) -> ToolResult:
        world = _world(session)
        surrogate = world.evaluate(theta).objective
        simulated = world.simulated_cost(theta)
        relative = abs(surrogate - simulated) / simulated if simulated else 0.0
        verdict = (
            "trustworthy" if relative < 0.02
            else "usable" if relative < 0.05
            else "biased -- do not trust the gradient at this policy"
        )
        return ToolResult(output=(
            f"surrogate (training signal) = {surrogate:.4f}, "
            f"simulation (true cost) = {simulated:.4f}, "
            f"gap = {abs(surrogate - simulated):.4f} ({relative:.2%}) -> {verdict}."
        ))


class CheckGradientHealth(BaseTool):
    name: str = "check_gradient_health"
    description: str = (
        "Diagnose whether gradients are meaningful for this policy. Reports the "
        "share of periods in which the reorder trigger sits near its threshold; "
        "above about 0.5% the gradient through the simulation is unreliable."
    )
    parameters: dict = _THETA_PARAM

    async def execute(self, session: dict, theta: list[float]) -> ToolResult:
        world = _world(session)
        unsaturated = saturation_fraction(
            world.capacities(theta), world.demand, world.reorder,
            tau=world.tau, lead_time=world.lead_time,
        )
        verdict = (
            "sound -- the trigger stays clear of its threshold"
            if unsaturated < 0.005
            else "UNRELIABLE -- the trajectory lingers near the reorder threshold; "
                 "gradients through the simulation are not trustworthy here"
        )
        return ToolResult(output=(
            f"unsaturated periods = {unsaturated:.3%} -> {verdict}. "
            f"(The training gradient comes from the fluid surrogate, which is "
            f"unaffected; use compare_to_simulation to check that instead.)"
        ))


def build_sizing_session(world: SizingPolicyWorld) -> dict:
    """Assemble the shared blackboard for the material-supply agent."""
    return {"world": world}


def build_sizing_tools() -> ToolCollection:
    """The intralogistics toolset — same harness, second domain."""
    return ToolCollection(
        DescribeScenario(),
        EvaluatePolicy(),
        InspectCapacities(),
        CompareToSimulation(),
        CheckGradientHealth(),
    )

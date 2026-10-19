"""The closed DDES <-> policy-agent loop (D2 / RQ2, offline).

One turn: the agent proposes ``theta`` -> the DDES world model replays the
schedule and returns a gradient-bearing :class:`Observation` -> the agent reads
it and proposes again. This is "rapid closed-loop learning" with the simulator
replay standing in for costly shopfloor trials. For the LLM-in-the-driver's-seat
variant, see :mod:`opsworld.agent`.
"""

from __future__ import annotations

from .agents import PolicyAgent
from .worldmodel import DispatchWorldModel, Observation

__all__ = ["run_coupling_loop"]


def run_coupling_loop(world: DispatchWorldModel, agent: PolicyAgent,
                      steps: int) -> list[Observation]:
    """Run ``steps`` turns of the agent/DDES loop; return the observation trace.

    The last element is the replay of the agent's final proposal, so
    ``history[-1].objective`` is the achieved schedule cost.
    """
    theta = agent.initial_theta()
    history: list[Observation] = []
    for t in range(steps):
        obs = world.evaluate(theta, step=t)
        history.append(obs)
        theta = agent.propose(obs)
    history.append(world.evaluate(theta, step=steps))  # replay final proposal
    return history

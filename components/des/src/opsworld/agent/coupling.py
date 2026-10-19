"""The closed world-model <-> agent loop, for any domain.

One turn: the agent proposes a decision, the world model replays it and returns
an :class:`~opsworld.world.Evaluation` carrying an objective, a gradient and a
verbalisation of that gradient; the agent reads the verbalisation and proposes
again. The simulator replay stands in for a costly real-world trial.

Domain-agnostic by construction — it needs only ``evaluate`` and, for the
gradient agent, a ``PATHWISE`` sensitivity. The same loop therefore drives a
differentiable shopfloor model and a material-supply policy, which is what
turns RQ2 from a single demonstration into a claim about the coupling itself.

The LLM-in-the-driver's-seat variant, where the model *chooses* which tools to
call rather than being handed an observation, is :class:`opsworld.agent.Agent`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ..world import Evaluation, SensitivityKind, WorldModel

__all__ = ["PolicyAgent", "GradientAgent", "PromptedAgent", "run_coupling_loop"]


@runtime_checkable
class PolicyAgent(Protocol):
    def initial(self) -> Any:
        """The opening proposal."""
        ...

    def propose(self, evaluation: Evaluation) -> Any:
        """A revised proposal, given the latest replay."""
        ...


class GradientAgent:
    """Steps along the returned descent direction — the grounded update, bare.

    Refuses a ``DUAL`` sensitivity: a shadow price is not a direction in
    decision space (``PLAN.md`` D10). :meth:`Sensitivity.descent_direction`
    raises, and that is deliberate.
    """

    def __init__(
        self,
        dimension: int,
        *,
        lr: float = 0.3,
        initial: list[float] | None = None,
        normalise: bool = True,
    ) -> None:
        # NB: all-equal weights are a symmetric saddle in both domains -- every
        # job scores the same, every item gets the same carrier. Start non-zero.
        self._theta = list(initial) if initial is not None else [0.05] * dimension
        self.lr = lr
        self.normalise = normalise

    def initial(self) -> list[float]:
        return list(self._theta)

    def propose(self, evaluation: Evaluation) -> list[float]:
        sensitivity = evaluation.sensitivity
        if sensitivity is None:
            raise ValueError("GradientAgent needs a world model that reports a gradient")
        direction = sensitivity.descent_direction()
        if self.normalise:
            norm = sum(d * d for d in direction) ** 0.5
            if norm > 0:
                direction = [d / norm for d in direction]
        self._theta = [t + self.lr * d for t, d in zip(self._theta, direction)]
        return list(self._theta)


class PromptedAgent:
    """Verbalises the replay into a prompt and parses the model's reply.

    ``complete`` maps chat messages to reply text, so any backend — or a fake,
    in tests — can drive the loop. The point is that the channel between
    simulator and model is *language plus gradient*, not a numeric API.
    """

    SYSTEM = (
        "You are tuning a policy expressed as a weight vector over named rules. "
        "After each proposal a simulation replays the system and reports the "
        "objective together with its gradient with respect to each weight. A "
        "negative gradient on a rule means increasing that weight lowers the "
        "objective. Use the critique to propose better weights. Reply with ONLY "
        "a JSON list of numbers, one per rule, in the listed order."
    )

    def __init__(
        self,
        complete: Callable[[list[dict]], str],
        dimension: int,
        *,
        initial: list[float] | None = None,
    ) -> None:
        self.complete = complete
        self.dimension = dimension
        self._initial = initial if initial is not None else [0.05] * dimension

    def initial(self) -> list[float]:
        return list(self._initial)

    def build_messages(self, evaluation: Evaluation) -> list[dict]:
        names = evaluation.sensitivity.names if evaluation.sensitivity else []
        user = (
            f"Rule order: {names}\n\n{evaluation.as_text()}\n\n"
            "Return the revised weights as a JSON list."
        )
        return [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": user},
        ]

    def _parse(self, reply: str) -> list[float]:
        match = re.search(r"\[[^\]]*\]", reply, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON list found in reply: {reply!r}")
        theta = [float(x) for x in json.loads(match.group(0))]
        if len(theta) != self.dimension:
            raise ValueError(f"expected {self.dimension} weights, got {len(theta)}")
        return theta

    def propose(self, evaluation: Evaluation) -> list[float]:
        return self._parse(self.complete(self.build_messages(evaluation)))


def run_coupling_loop(
    world: WorldModel, agent: PolicyAgent, steps: int
) -> list[Evaluation]:
    """Run ``steps`` turns and return the trace, final proposal included.

    ``trace[-1].objective`` is what the agent actually achieved.
    """
    decision = agent.initial()
    trace: list[Evaluation] = []
    for _ in range(steps):
        evaluation = world.evaluate(decision)
        trace.append(evaluation)
        decision = agent.propose(evaluation)
    trace.append(world.evaluate(decision))
    return trace

"""Programmatic policy agents that consume DDES gradient feedback (D2).

A :class:`PolicyAgent` proposes a rule-weight vector ``theta``, then revises it
from the :class:`~gradientshop.agent.worldmodel.Observation` a DDES replay
returns. These run *without* an LLM endpoint, so the RQ2 loop is exercisable
offline and serves as the optimisation baseline:

* :class:`GradientDescentAgent` -- descends the returned gradient directly.
* :class:`PromptedLLMAgent`     -- verbalises the observation into a prompt and
  parses revised weights from a model reply; takes a ``complete`` callable so any
  backend (or a fake, in tests) can drive it.

The full ReAct loop where an LLM *chooses* to call DDES tools lives in
:mod:`opsworld.agent` + :mod:`gradientshop.agent.tools`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol

from .worldmodel import Observation

__all__ = ["PolicyAgent", "GradientDescentAgent", "PromptedLLMAgent"]


class PolicyAgent(Protocol):
    def initial_theta(self) -> list[float]:
        """Starting policy weights."""
        ...

    def propose(self, observation: Observation) -> list[float]:
        """Revised policy weights given the latest DDES feedback."""
        ...


class GradientDescentAgent:
    """Descends the DDES gradient directly -- the grounded-update loop in its
    purest form (the 'gradient grounds the next proposal' of RQ2)."""

    def __init__(self, n_features: int, *, lr: float = 0.3,
                 initial: list[float] | None = None) -> None:
        # NB: all-zero weights make every job score equal -- a symmetric saddle
        # of the relaxed objective with zero gradient. Start from a small
        # non-zero policy so the very first replay yields a usable direction.
        self._theta = list(initial) if initial is not None else [0.05] * n_features
        self.lr = lr

    def initial_theta(self) -> list[float]:
        return list(self._theta)

    def propose(self, observation: Observation) -> list[float]:
        self._theta = [t + self.lr * d
                       for t, d in zip(observation.theta,
                                       observation.suggested_direction())]
        return list(self._theta)


class PromptedLLMAgent:
    """Drives a real LLM with the verbalised DDES observation (RQ2 proof).

    ``complete`` maps chat messages ``[{"role", "content"}, ...]`` to the model's
    reply text; the agent asks for revised weights as a JSON list and parses them.
    A fake ``complete`` makes the coupling loop unit-testable offline.
    """

    SYSTEM = (
        "You are a production-scheduling agent tuning a dispatch policy. The "
        "policy scores jobs by a weighted sum of named dispatching rules; higher "
        "score = dispatched earlier. After each proposal a differentiable "
        "discrete-event simulation replays the schedule and reports the objective "
        "and its gradient w.r.t. each weight. Use that feedback to propose better "
        "weights. Reply with ONLY a JSON list of numbers, one weight per rule, in "
        "the listed order."
    )

    def __init__(self, complete: Callable[[list[dict]], str], n_features: int,
                 *, initial: list[float] | None = None) -> None:
        self.complete = complete
        self.n_features = n_features
        self._initial = initial if initial is not None else [0.0] * n_features

    def initial_theta(self) -> list[float]:
        return list(self._initial)

    def build_messages(self, observation: Observation) -> list[dict]:
        user = (f"Rule order: {observation.feature_names}\n\n"
                f"{observation.as_text()}\n\n"
                "Return the revised weights as a JSON list.")
        return [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}]

    def _parse(self, reply: str) -> list[float]:
        match = re.search(r"\[[^\]]*\]", reply, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON list found in LLM reply: {reply!r}")
        theta = [float(x) for x in json.loads(match.group(0))]
        if len(theta) != self.n_features:
            raise ValueError(f"expected {self.n_features} weights, got {len(theta)}")
        return theta

    def propose(self, observation: Observation) -> list[float]:
        reply = self.complete(self.build_messages(observation))
        return self._parse(reply)

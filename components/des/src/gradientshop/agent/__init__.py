"""D2 -- LLM <-> DDES coupling (proposal RQ2).

Two entry points:

* **Programmatic loop** (offline, no API key): a :class:`PolicyAgent`
  (:class:`GradientDescentAgent` or :class:`PromptedLLMAgent`) revises a dispatch
  policy from DDES gradient feedback via :func:`run_coupling_loop`.
* **ReAct agent** (needs an LLM endpoint): the :mod:`opsworld.agent` runs a real model
  that chooses to call the DDES :mod:`.tools` (replay + gradient, compare to the
  discrete and SimPy references) inside a tool-use loop.
"""

from __future__ import annotations

from . import agents, loop, tools, worldmodel
from .agents import GradientDescentAgent, PolicyAgent, PromptedLLMAgent
from .loop import run_coupling_loop
from .worldmodel import DispatchWorldModel, Observation

__all__ = [
    "agents", "loop", "tools", "worldmodel",
    "DispatchWorldModel", "Observation",
    "PolicyAgent", "GradientDescentAgent", "PromptedLLMAgent",
    "run_coupling_loop",
]

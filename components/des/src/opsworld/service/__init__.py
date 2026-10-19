"""An HTTP surface over the world models — the composition layer, kept thin.

Three modules with no third-party dependency and no domain import:

``codec``     tuple-indexed decisions <-> JSON, plus the ``opsworld`` types
``registry``  what can be built, and which solvers a world model can serve
``jobs``      one process per long run, with real cancellation and a timeout

and one that needs FastAPI, imported only when you ask for it:

``app``       ``create_app(registry, runner)``

The split is deliberate. Architecture constraint A6 says the shared layer stays
thin, so a web framework is not allowed to become a requirement of the research
packages — ``uv sync --group service`` opts in. And nothing here knows a domain:
domains register themselves, and the composition root
(``examples/serve_api.py``) is the single place that imports both sides.
"""

from __future__ import annotations

from .codec import (
    decode_key,
    decode_table,
    encode_evaluation,
    encode_key,
    encode_sensitivity,
    encode_solver_result,
    encode_table,
    jsonable,
)
from .jobs import Job, JobError, JobRunner
from .registry import ParamSpec, Registry, ScenarioSpec, SolverSpec, WorldSpec

__all__ = [
    "encode_key", "decode_key", "encode_table", "decode_table",
    "encode_evaluation", "encode_sensitivity", "encode_solver_result", "jsonable",
    "ParamSpec", "ScenarioSpec", "WorldSpec", "SolverSpec", "Registry",
    "Job", "JobError", "JobRunner",
]

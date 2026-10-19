"""Intralogistics agent layer — the second domain on the shared harness (RQ2).

Adding a domain to the agent story costs exactly one module of tools and one
persona: the ReAct loop, the LLM client, the tool protocol, the config loader
and the audit log all come from :mod:`opsworld.agent` unchanged.

That is the point being demonstrated. RQ2 asks whether a simulation replay can
be embedded as a reasoning step inside LLM inference; showing it in a single
domain leaves open whether the coupling was tailored to that domain. Here the
same harness drives a differentiable discrete-event shopfloor model and a
constrained material-supply problem, through the same
``objective + gradient + verbalised critique`` channel.
"""

from .tools import build_sizing_session, build_sizing_tools

__all__ = ["build_sizing_session", "build_sizing_tools", "load_persona"]


def load_persona(name: str = "supply_planner") -> str:
    """Read a persona system prompt shipped with this package."""
    from pathlib import Path

    path = Path(__file__).resolve().parent / "personas" / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"no persona {name!r} at {path}")
    return path.read_text(encoding="utf-8")

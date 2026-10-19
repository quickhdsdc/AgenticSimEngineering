"""Where a command runs.

Hermes carries seven of these (local, Docker, SSH, Singularity, Modal,
Daytona, Vercel) behind one interface, and that indirection is what lets the
same agent work on a laptop and on a cluster. The industrial case needs the
same seam for a different reason: a command may belong on the orchestrator
host, on a machine's edge node beside its OPC UA server, or inside a
container pinned to a simulation image. Which one is a deployment fact, not
an agent decision, so it lives here rather than in the agent's prompt.

Two are implemented now -- local and SSH -- because those are the two the
current topology needs. The interface exists so the third does not require
touching the loop.
"""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

# Output that goes into a model's context has to be bounded. A single
# `aas sm tree` on a large product submodel can otherwise cost more tokens
# than the whole rest of the turn.
MAX_OUTPUT_CHARS = 12_000


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    duration_s: float
    environment: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def for_model(self) -> str:
        """Render for the model: outcome first, then whatever output fits."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.rstrip()}")
        if not parts:
            parts.append("(no output)")
        if self.truncated:
            parts.append(f"[output truncated at {MAX_OUTPUT_CHARS} chars]")
        if self.exit_code != 0:
            parts.insert(0, f"[exit {self.exit_code}]")
        return "\n".join(parts)


def clamp(text: str) -> tuple[str, bool]:
    """Keep the head and the tail; the middle of a long listing is the least informative part."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    head = text[: MAX_OUTPUT_CHARS // 2]
    tail = text[-MAX_OUTPUT_CHARS // 2 :]
    return f"{head}\n...\n{tail}", True


class Environment(ABC):
    """A place commands run."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 120.0,
    ) -> ExecResult:
        ...

    def close(self) -> None:  # pragma: no cover - most backends need nothing
        return None

    @staticmethod
    def quote(value: str) -> str:
        return shlex.quote(value)

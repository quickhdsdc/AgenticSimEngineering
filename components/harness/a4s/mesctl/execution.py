"""Correlation metadata only: neither authorization nor business state.

An execution ID identifies one attempt; a trace ID links attempts in a task.
Neither identifier supplies server-side idempotency or proves authorization.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import os
import re
from uuid import uuid4


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value):
        raise argparse.ArgumentTypeError(
            "Correlation ID must be 1-128 ASCII letters, digits, '.', '_', ':' or '-'"
        )
    return value


@dataclass(frozen=True)
class ExecutionContext:
    trace_id: str
    execution_id: str

    def __post_init__(self):
        identifier(self.trace_id)
        identifier(self.execution_id)

    @classmethod
    def for_tool(cls, session_id, trace_id=None):
        trace = trace_id or "session-" + hashlib.sha256(session_id.encode()).hexdigest()[:24]
        return cls(trace, str(uuid4()))

    @classmethod
    def from_env(cls, trace_id=None):
        # Harness-supplied identity wins; these are correlation, not auth claims.
        return cls(
            os.getenv("A4S_MES_TRACE_ID") or trace_id or str(uuid4()),
            os.getenv("A4S_MES_EXECUTION_ID") or str(uuid4()),
        )

    def as_dict(self):
        return asdict(self)

    def environment(self):
        return {"A4S_MES_TRACE_ID": self.trace_id, "A4S_MES_EXECUTION_ID": self.execution_id}

    def headers(self):
        return {"X-Correlation-ID": self.trace_id, "X-A4S-Execution-ID": self.execution_id}

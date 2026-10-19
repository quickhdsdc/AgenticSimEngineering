"""Audit log -- every run is reproducible and inspectable (mirrors MoldDesign).

One JSONL file per run: LLM decisions, tool calls with args and results, HITL
gates, and the final answer. For a domain whose tools wrap a world model, this
doubles as the record of the agent's search grounded in simulation replays.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

# <repo>/logs  (src/opsworld/agent/audit.py -> parents[3] == repo root)
LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


class AuditLog:
    def __init__(self, run_id: str | None = None, log_dir: Path | None = None) -> None:
        self.dir = log_dir or LOG_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"run_{self.run_id}.jsonl"
        self._t0 = time.time()

    def record(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time() - self._t0, 3), "run_id": self.run_id,
               "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

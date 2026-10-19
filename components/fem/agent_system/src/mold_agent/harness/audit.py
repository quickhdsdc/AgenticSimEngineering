"""审计日志(P3:一切可审计可复现)。

每次 run 一个 JSONL 文件,逐条记录:LLM 决策 / 工具调用+入参+结果 / HITL / 终稿。
工程签字、责任追溯、事后复盘的依据。
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


class AuditLog:
    def __init__(self, run_id: str | None = None) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"run_{self.run_id}.jsonl"
        self._t0 = time.time()

    def record(self, event: str, **fields) -> None:
        rec = {
            "ts": round(time.time() - self._t0, 3),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

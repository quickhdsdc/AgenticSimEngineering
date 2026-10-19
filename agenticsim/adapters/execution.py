"""Local synchronous isolated execution and atomic study snapshots.

No retry, resume, cancellation API or exactly-once claim in this first slice.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4
from uuid import UUID

from agenticsim.application import now
from agenticsim.domain import Evaluation
from agenticsim.adapters.process_tree import run_worker

ROOT = Path(__file__).resolve().parents[2]


def worker_environment(directory):
    """A compute worker does not need Agent credentials or plant destinations."""
    excluded = ("A4S_", "AZURE_OPENAI", "OPENAI_", "ANTHROPIC_", "CLAUDE_")
    return dict({k: v for k, v in os.environ.items() if not k.upper().startswith(excluded)},
                ASE_RUN_DIRECTORY=str(directory))


class LocalWorker:
    def __init__(self, python=None):
        self.python = str(python or sys.executable)

    def evaluate(self, task, candidate):
        payload = json.dumps({"task": task.document, "candidate": candidate}, allow_nan=False)
        directory = ROOT / "work/runs" / "workers" / uuid4().hex
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(payload, encoding="utf-8")
        env = worker_environment(directory)
        try:
            result = run_worker([self.python, "-m", "agenticsim.runtime.worker"], payload=payload,
                                cwd=ROOT, env=env, timeout=task.document["budget"]["timeout_s"])
        except subprocess.TimeoutExpired as exc:
            (directory / "stdout.log").write_text(exc.output or "", encoding="utf-8")
            (directory / "stderr.log").write_text(exc.stderr or "", encoding="utf-8")
            raise TimeoutError(f"evaluation worker exceeded timeout; evidence: {directory}") from exc
        (directory / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{result.stderr[-4000:]}\nevidence: {directory}")
        return Evaluation.from_dict(json.loads(result.stdout))


class FileStudyStore:
    def __init__(self, root):
        self.root = Path(root)

    def create(self, task):
        study = {"schema_version": "1.0", "study_id": str(uuid4()), "task": task,
                 "created_at": now(), "status": "submitted", "runs": []}
        (self.root / study["study_id"]).mkdir(parents=True, exist_ok=False)
        self.save(study)
        return study

    def save(self, study):
        # A store reference is not a general filesystem path. Validate it
        # before opening files, including when save is called independently.
        study_id = study.get("study_id")
        if not isinstance(study_id, str) or str(UUID(study_id)) != study_id:
            raise ValueError("study_id must be a canonical UUID")
        canonical = json.dumps(study, sort_keys=True, allow_nan=False, ensure_ascii=False).encode("utf-8")
        envelope = {"payload": study, "sha256": hashlib.sha256(canonical).hexdigest(),
                    "hash_scope": "UTF-8 json.dumps(payload, sort_keys=True, allow_nan=False, ensure_ascii=False)"}
        directory = (self.root / study_id).resolve()
        if not directory.is_relative_to(self.root.resolve()):
            raise ValueError("study directory leaves store root")
        target = directory / "study.json"
        temporary = None
        try:
            # Unique temporary files avoid collisions between snapshot writes.
            # This remains a single-writer snapshot store, not a task scheduler.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix="study-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(envelope, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, study_id):
        if not isinstance(study_id, str) or str(UUID(study_id)) != study_id:
            raise ValueError("study_id must be a canonical UUID")
        directory = (self.root / study_id).resolve()
        if not directory.is_relative_to(self.root.resolve()):
            raise ValueError("study directory leaves store root")
        envelope = json.loads((directory / "study.json").read_text(encoding="utf-8"))
        payload = envelope["payload"]
        canonical = json.dumps(payload, sort_keys=True, allow_nan=False, ensure_ascii=False).encode("utf-8")
        if payload.get("study_id") != study_id or hashlib.sha256(canonical).hexdigest() != envelope["sha256"]:
            raise ValueError("Study identity or digest mismatch")
        return payload

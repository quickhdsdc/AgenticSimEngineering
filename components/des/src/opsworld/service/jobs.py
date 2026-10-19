"""Work that outlives a request.

The latencies in this framework span four and a half orders of magnitude,
measured on the instances the UI will actually offer::

    policy evaluate (cost + gradient)        0.2 ms
    tau sweep, 8 temperatures                8.7 ms
    static evaluate, 40 items                 87 ms
    ------------------------------------------------ the line
    heuristic solve, 6 items x 3 resources   2.4 s
    MILP solve, 4 items x 2 resources        4.6 s
    slow-scale episode, 8 periods            9.2 s

Everything above the line answers inside a request. Everything below it needs a
job, and a job needs three things a thread pool does not give:

**Real cancellation.** ``ProcessPoolExecutor`` cannot interrupt a future that
has started, and branch-and-bound in HiGHS does not poll a flag. Each job
therefore gets its own process, which can actually be terminated.

**A wall clock the work cannot argue with.** A user who dials in ten items and
three resources has asked for something that may not finish; the timeout is the
answer, and it is enforced from outside the worker.

**Nothing unpicklable crossing the boundary.** A built world model holds torch
tensors and, for the MILP, a live Pyomo model. So a job never ships an object —
it ships the *recipe* (``domain modules``, ``ids``, ``plain params``) and the
worker rebuilds. That costs a few milliseconds and buys determinism: the same
request reconstructs the same instance, which is also what makes a result
reproducible outside the server.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Job", "JobRunner", "JobError"]


class JobError(RuntimeError):
    """The worker raised. The message carries the worker's traceback."""


@dataclass
class Job:
    id: str
    kind: str
    payload: dict
    status: str = "queued"          # queued | running | done | failed | cancelled
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    timeout: float = 300.0

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at

    def as_json(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "elapsed_seconds": round(self.elapsed, 3),
            "queued_seconds": round((self.started_at or time.time()) - self.submitted_at, 3),
            "timeout_seconds": self.timeout,
            "result": self.result, "error": self.error,
        }


def _worker(entry: str, payload: dict, pipe) -> None:  # pragma: no cover - child process
    """Run one job in a fresh process and send back plain JSON.

    ``entry`` is a dotted path resolved *inside* the child, so nothing but
    strings and numbers crosses the process boundary.
    """
    try:
        module_name, _, attr = entry.rpartition(".")
        module = __import__(module_name, fromlist=[attr])
        result = getattr(module, attr)(payload)
        pipe.send(("ok", result))
    except BaseException as exc:  # noqa: BLE001 - the parent decides what to do
        pipe.send(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        pipe.close()


class JobRunner:
    """A small, honest job queue: one process per job, bounded concurrency."""

    def __init__(self, *, max_concurrent: int = 2, default_timeout: float = 300.0,
                 retain: int = 200) -> None:
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.retain = retain
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max_concurrent)
        self._ctx = mp.get_context("spawn")

    # -- submission ---------------------------------------------------
    def submit(self, kind: str, entry: str, payload: dict,
               timeout: float | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, payload=payload,
                  timeout=float(timeout or self.default_timeout))
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        threading.Thread(target=self._run, args=(job, entry), daemon=True).start()
        return job

    def _run(self, job: Job, entry: str) -> None:
        self._slots.acquire()
        try:
            with self._lock:
                if job.status == "cancelled":
                    return
                job.status = "running"
                job.started_at = time.time()

            parent, child = self._ctx.Pipe(duplex=False)
            proc = self._ctx.Process(target=_worker, args=(entry, job.payload, child),
                                     daemon=True)
            proc.start()
            child.close()
            with self._lock:
                self._procs[job.id] = proc

            got = parent.poll(job.timeout)
            if not got:
                proc.terminate()
                proc.join(5)
                self._finish(job, "failed",
                             error=f"timed out after {job.timeout:g}s — "
                                   f"reduce the instance size or the budget")
                return

            try:
                tag, value = parent.recv()
            except EOFError:
                self._finish(job, "failed", error="worker died without a result")
                return
            finally:
                parent.close()
            proc.join(5)

            with self._lock:
                if job.status == "cancelled":
                    return
            if tag == "ok":
                self._finish(job, "done", result=value)
            else:
                self._finish(job, "failed", error=value)
        except Exception as exc:  # noqa: BLE001
            self._finish(job, "failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._procs.pop(job.id, None)
            self._slots.release()

    def _finish(self, job: Job, status: str, *, result: Any = None,
                error: str | None = None) -> None:
        with self._lock:
            job.status = status
            job.result = result
            job.error = error
            job.finished_at = time.time()

    # -- control ------------------------------------------------------
    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._lock:
            proc = self._procs.get(job_id)
            if job.status in ("done", "failed", "cancelled"):
                return job
            job.status = "cancelled"
            job.finished_at = time.time()
            job.error = "cancelled by the client"
        if proc is not None and proc.is_alive():
            proc.terminate()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(f"unknown job {job_id!r}") from None

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.submitted_at)

    def _evict(self) -> None:
        """Drop the oldest finished jobs once the table is over budget."""
        if len(self._jobs) <= self.retain:
            return
        finished = sorted(
            (j for j in self._jobs.values()
             if j.status in ("done", "failed", "cancelled")),
            key=lambda j: j.finished_at or 0.0,
        )
        for job in finished[: len(self._jobs) - self.retain]:
            self._jobs.pop(job.id, None)

    def shutdown(self) -> None:
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            if proc.is_alive():
                proc.terminate()

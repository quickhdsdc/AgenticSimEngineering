"""M1 -- The single-server FIFO queue as a differentiable DES.

This is the smallest discrete-event system that still exhibits the two features
that make DES interesting: a resource contended by a stream of jobs, and a
state (the workload) that evolves through event-driven ``max`` logic.

For a first-come-first-served single server the departure times obey the
**Lindley recursion** (cf. proposal ref [14], Che, Dong & Namkoong,
"Differentiable discrete-event simulation for queuing network control")::

    start_i     = max(depart_{i-1}, arrive_i)      # server free AND job present
    depart_i    = start_i + service_i
    wait_i      = start_i - arrive_i               # time queued before service
    sojourn_i   = depart_i - arrive_i              # total time in system

Every operation here (``cumsum``, ``max``, ``+``, ``-``) is differentiable
almost everywhere. The trajectory is a *piecewise-smooth* function of the
service/arrival parameters, non-differentiable only on the measure-zero set
where the event order flips (two events exactly tie). Hence PyTorch autodiff
yields exact pathwise gradients of any downstream metric w.r.t. those
parameters -- the "smoothed approximations of event-time dynamics" of the
proposal's source A. No relaxation is needed yet; discrete *decisions*
(dispatch/routing, source B) enter only from M3 onward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["QueueTrajectory", "single_server", "lindley_from_arrivals"]


def lindley_from_arrivals(arrive: Tensor, service: Tensor) -> tuple[Tensor, Tensor]:
    """FIFO single-server recursion from *absolute* arrival times.

    Shared core of both the single queue (M1) and each station of a tandem line
    (M2). ``arrive`` must be non-decreasing (FIFO preserves order, so a station
    fed by an upstream station's departures satisfies this automatically).

    Returns ``(start, depart)`` tensors, each ``[N]``.
    """
    starts: list[Tensor] = []
    departs: list[Tensor] = []
    prev_depart = arrive.new_zeros(())  # server idle before the first job
    for i in range(arrive.shape[0]):
        start_i = torch.maximum(prev_depart, arrive[i])
        depart_i = start_i + service[i]
        starts.append(start_i)
        departs.append(depart_i)
        prev_depart = depart_i
    return torch.stack(starts), torch.stack(departs)


@dataclass
class QueueTrajectory:
    """Per-job trajectory of a single-server queue. All tensors share ``[N]``.

    Attributes are differentiable w.r.t. any leaf tensor feeding
    ``interarrival``/``service``.
    """

    arrive: Tensor        # arrival time of each job (cumulative inter-arrivals)
    start: Tensor         # service-start time
    depart: Tensor        # departure time
    wait: Tensor          # time spent queued before service (>= 0)
    sojourn: Tensor       # total time in system = wait + service

    @property
    def n(self) -> int:
        return self.arrive.shape[0]

    def mean_wait(self, *, warmup: int = 0) -> Tensor:
        """Mean queueing delay, discarding the first ``warmup`` jobs (transient)."""
        return self.wait[warmup:].mean()

    def mean_sojourn(self, *, warmup: int = 0) -> Tensor:
        return self.sojourn[warmup:].mean()

    def throughput(self, *, warmup: int = 0) -> Tensor:
        """Completed jobs per unit time over the measured window."""
        w = self.arrive.new_tensor(float(warmup))  # noqa: F841 (kept for symmetry)
        completed = self.depart[warmup:]
        n = completed.shape[0]
        span = completed[-1] - self.arrive[warmup]
        return completed.new_tensor(float(n)) / span


def single_server(interarrival: Tensor, service: Tensor) -> QueueTrajectory:
    """Simulate a FIFO single-server queue from event-time increments.

    Args:
        interarrival: ``[N]`` gaps between successive arrivals (job 0's gap is
            measured from time 0). Must be non-negative.
        service: ``[N]`` service requirement of each job. Must be non-negative.

    Returns:
        A :class:`QueueTrajectory`. The forward pass runs an explicit ``O(N)``
        Lindley recursion; the sequential ``max`` dependency prevents a simple
        vectorization, but it keeps the autograd graph exact and cheap for the
        small ``N`` used in M1/M2 validation.
    """
    if interarrival.shape != service.shape:
        raise ValueError(
            f"interarrival {tuple(interarrival.shape)} and service "
            f"{tuple(service.shape)} must have equal shape"
        )

    arrive = torch.cumsum(interarrival, dim=0)
    start, depart = lindley_from_arrivals(arrive, service)
    wait = start - arrive
    sojourn = depart - arrive
    return QueueTrajectory(
        arrive=arrive, start=start, depart=depart, wait=wait, sojourn=sojourn
    )

"""M2 -- Multi-station tandem (flow) line as a differentiable DES.

Generalises the single server (M1) to ``M`` stations in series with **fixed
routing**: every job visits stations ``0, 1, ..., M-1`` in order, each a FIFO
single server. A job cannot start at station ``k`` until it has both departed
station ``k-1`` *and* the station-``k`` server has finished the previous job::

    depart[i, k] = max(depart[i-1, k], depart[i, k-1]) + service[i, k]

with boundaries ``depart[i, -1] = arrive[i]`` (external arrival) and
``depart[-1, k] = 0`` (stations idle initially). This is the max-plus recursion
for a tandem queue; computed station-by-station it is exactly the M1 Lindley
recursion with each station fed by the previous station's departures. Still no
discrete decisions -- routing is fixed -- so plain autodiff gives exact
pathwise gradients of makespan / throughput / WIP w.r.t. per-station rates.
Dispatch *decisions* (which job next, source B) enter in M3 (:mod:`.dispatch`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .queue import lindley_from_arrivals

__all__ = ["LineTrajectory", "tandem_line"]


@dataclass
class LineTrajectory:
    """Trajectory of a tandem line. ``depart``/``start`` are ``[N, M]``."""

    arrive: Tensor   # [N] external arrival time of each job
    start: Tensor    # [N, M] service-start time at each station
    depart: Tensor   # [N, M] departure time from each station

    @property
    def n_jobs(self) -> int:
        return self.arrive.shape[0]

    @property
    def n_stations(self) -> int:
        return self.depart.shape[1]

    def makespan(self) -> Tensor:
        """Completion time of the last job at the last station (relative to the
        first arrival). The canonical objective to minimise."""
        return self.depart[-1, -1] - self.arrive[0]

    def throughput(self, *, warmup: int = 0) -> Tensor:
        exit_times = self.depart[warmup:, -1]
        n = exit_times.shape[0]
        span = exit_times[-1] - self.arrive[warmup]
        return exit_times.new_tensor(float(n)) / span

    def station_sojourn(self, *, warmup: int = 0) -> Tensor:
        """Mean total time-in-system per job (last departure - arrival)."""
        return (self.depart[warmup:, -1] - self.arrive[warmup:]).mean()


def tandem_line(interarrival: Tensor, service: Tensor) -> LineTrajectory:
    """Simulate an ``M``-station FIFO tandem line.

    Args:
        interarrival: ``[N]`` gaps between external arrivals.
        service: ``[N, M]`` service requirement of job ``i`` at station ``k``.

    Returns:
        A :class:`LineTrajectory`. Runs one M1 Lindley pass per station, feeding
        station ``k`` the absolute departure times of station ``k-1``.
    """
    if service.ndim != 2:
        raise ValueError(f"service must be [N, M], got {tuple(service.shape)}")
    if service.shape[0] != interarrival.shape[0]:
        raise ValueError(
            f"job count mismatch: interarrival {tuple(interarrival.shape)} vs "
            f"service {tuple(service.shape)}"
        )

    arrive = torch.cumsum(interarrival, dim=0)
    n_stations = service.shape[1]

    starts: list[Tensor] = []
    departs: list[Tensor] = []
    upstream = arrive  # arrivals to station 0 are the external arrivals
    for k in range(n_stations):
        start_k, depart_k = lindley_from_arrivals(upstream, service[:, k])
        starts.append(start_k)
        departs.append(depart_k)
        upstream = depart_k  # station k's departures feed station k+1
    return LineTrajectory(
        arrive=arrive, start=torch.stack(starts, dim=1),
        depart=torch.stack(departs, dim=1),
    )

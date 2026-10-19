"""M4 -- Battery-remanufacturing line as a differentiable DES scenario.

Instantiates the proposal's D1 battery-remanufacturing process on the DDES core.
End-of-life automotive packs flow through four stages::

    Disassembly -> Testing/Cycling -> Sorting/Grading -> Reassembly

The **Testing/Cycling** stage is the bottleneck: cell cycling on a BaSyTec CTS
cycler takes hours, far longer than the mechanical stages. Its parallelism (the
cycler's 8 channels) is modelled as a *continuous capacity* -- a fluid
relaxation of ``c`` parallel servers into an effective service rate
``service / c``. Because ``c`` enters differentiably, "how many channels to
give each stage under a fixed budget" becomes a gradient problem: differentiate
makespan w.r.t. the capacity vector and let the gradient find the bottleneck.

Service times are log-normal (a realistic heavy-tailed model for manual/testing
work), sampled via the reparameterized primitives so both the stage parameters
and the capacities carry gradients. Stage means/CVs are placeholders to be
replaced by calibration data from the IFF testing module (proposal D1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..ddes import samplers
from ..ddes.line import tandem_line

__all__ = ["BatteryLineConfig", "default_config", "sample_base_service",
           "effective_service", "makespan"]


@dataclass
class BatteryLineConfig:
    """Parameters of the remanufacturing line. Times are in hours."""

    stage_names: list[str]
    mean_service: Tensor   # [M] mean single-server service time per stage
    cv: Tensor             # [M] coefficient of variation (log-normal spread)
    arrival_rate: float    # packs arriving per hour

    n_stages: int = field(init=False)

    def __post_init__(self) -> None:
        self.mean_service = torch.as_tensor(self.mean_service, dtype=torch.float64)
        self.cv = torch.as_tensor(self.cv, dtype=torch.float64)
        self.n_stages = self.mean_service.shape[0]
        if self.cv.shape[0] != self.n_stages or len(self.stage_names) != self.n_stages:
            raise ValueError("stage_names, mean_service, cv must share length M")


def default_config() -> BatteryLineConfig:
    """Placeholder calibration: cycling dominates the line (~8x the next stage)."""
    return BatteryLineConfig(
        stage_names=["Disassembly", "Testing/Cycling", "Sorting", "Reassembly"],
        mean_service=[0.5, 4.0, 0.3, 1.0],
        cv=[0.3, 0.2, 0.4, 0.3],
        arrival_rate=0.6,
    )


def _lognormal_params(mean: Tensor, cv: Tensor) -> tuple[Tensor, Tensor]:
    """(mu, sigma) of a log-normal with the given mean and coefficient of variation."""
    sigma2 = torch.log1p(cv ** 2)
    sigma = torch.sqrt(sigma2)
    mu = torch.log(mean) - 0.5 * sigma2
    return mu, sigma


def sample_base_service(config: BatteryLineConfig, n_jobs: int, *,
                        generator: torch.Generator | None = None) -> Tensor:
    """Sample the ``[N, M]`` single-server (capacity-1) service times.

    Uses common random numbers so the same sample path can be reused across
    capacity settings for low-variance gradients.
    """
    mu, sigma = _lognormal_params(config.mean_service, config.cv)
    cols = []
    for k in range(config.n_stages):
        z = samplers.normal_noise(n_jobs, generator=generator)
        cols.append(samplers.lognormal(mu[k], sigma[k], z))
    return torch.stack(cols, dim=1)


def effective_service(base_service: Tensor, capacity: Tensor) -> Tensor:
    """Fluid relaxation of ``c`` parallel servers: divide service by capacity.

    ``capacity`` is ``[M]`` (continuous, > 0) and broadcasts across jobs.
    """
    return base_service / capacity.clamp_min(1e-6)


def makespan(config: BatteryLineConfig, capacity: Tensor, base_service: Tensor,
             *, interarrival: Tensor | None = None) -> Tensor:
    """Makespan of the line for a given per-stage capacity allocation.

    Differentiable w.r.t. ``capacity`` (and ``base_service``). If
    ``interarrival`` is omitted, packs are released deterministically at the
    configured arrival rate.
    """
    n = base_service.shape[0]
    if interarrival is None:
        interarrival = torch.full((n,), 1.0 / config.arrival_rate,
                                  dtype=base_service.dtype)
    service = effective_service(base_service, capacity)
    return tandem_line(interarrival, service).makespan()

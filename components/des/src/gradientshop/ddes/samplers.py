"""M0 -- Reparameterizable stochastic primitives for DDES.

A discrete-event simulation is driven by random primitives: inter-arrival
times, processing/service times, etc. To get *pathwise* (reparameterization)
gradients of downstream metrics w.r.t. distribution parameters, we must sample
via a differentiable transform of parameter-free noise::

    S = g(theta, U),     U ~ base noise independent of theta

so that ``dS/dtheta`` exists and autodiff can propagate it. This is the
inverse-CDF / reparameterization trick (cf. proposal ref [16], Mohamed et al.,
"Monte Carlo gradient estimation"). Every sampler here takes an explicit noise
tensor ``u`` so that *common random numbers* can be reused across parameter
settings -- essential for low-variance gradients and for validating autodiff
against finite differences on a fixed sample path.

Convention: ``rate``/``mu``/``sigma`` are torch tensors that may carry
``requires_grad=True``; ``u`` is drawn once (parameter-free) and held fixed.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "uniform_noise",
    "normal_noise",
    "exponential",
    "lognormal",
]


def uniform_noise(n: int, *, generator: torch.Generator | None = None,
                  dtype: torch.dtype = torch.float64) -> Tensor:
    """Parameter-free U(0, 1) noise, open on 0 to keep ``log(u)`` finite."""
    u = torch.rand(n, generator=generator, dtype=dtype)
    # Map (0, 1] -> avoid exactly 0 so inverse-CDF transforms stay finite.
    return u.clamp_min(torch.finfo(dtype).tiny)


def normal_noise(n: int, *, generator: torch.Generator | None = None,
                 dtype: torch.dtype = torch.float64) -> Tensor:
    """Parameter-free standard-normal noise."""
    return torch.randn(n, generator=generator, dtype=dtype)


def exponential(rate: Tensor, u: Tensor) -> Tensor:
    """Exponential samples via inverse CDF: ``S = -log(U) / rate``.

    ``rate`` broadcasts against ``u``. Because ``U`` and ``1 - U`` are equal in
    distribution we use ``-log(U)`` directly. The pathwise derivative is
    ``dS/drate = -S / rate``, giving the exact ``d E[S]/drate = -1/rate**2``.
    """
    return -torch.log(u) / rate


def lognormal(mu: Tensor, sigma: Tensor, z: Tensor) -> Tensor:
    """Log-normal samples via ``S = exp(mu + sigma * Z)``, ``Z`` standard normal.

    A common heavy-tailed model for machine processing times. ``E[S] =
    exp(mu + sigma**2 / 2)``.
    """
    return torch.exp(mu + sigma * z)

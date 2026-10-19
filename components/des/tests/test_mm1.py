"""M1 validation: gradients through the single-server Lindley recursion.

Two independent checks:

1. **Autodiff == finite differences** on a *fixed sample path* (common random
   numbers). This is the decisive test that the pathwise gradient is correct:
   it certifies the autograd graph, independent of any queueing theory.

2. **Statistical agreement with M/M/1 steady-state theory**: the mean queueing
   delay Wq = rho / (mu - lambda) and dWq/dmu have closed forms, which the
   simulated estimates should approach on a long run.
"""

from __future__ import annotations

import torch

from gradientshop.ddes import samplers
from gradientshop.ddes.queue import single_server


def _simulate_mean_wait(mu: torch.Tensor, lam: float, u_arr, u_srv, warmup):
    interarrival = samplers.exponential(torch.tensor(lam, dtype=torch.float64), u_arr)
    service = samplers.exponential(mu, u_srv)
    traj = single_server(interarrival, service)
    return traj.mean_wait(warmup=warmup)


def test_autodiff_matches_finite_difference_on_fixed_path():
    """Pathwise dWq/dmu from autograd matches central finite differences."""
    n, lam, warmup = 4000, 0.7, 500
    g = torch.Generator().manual_seed(123)
    u_arr = samplers.uniform_noise(n, generator=g)
    u_srv = samplers.uniform_noise(n, generator=g)

    mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    wq = _simulate_mean_wait(mu, lam, u_arr, u_srv, warmup)
    (grad_ad,) = torch.autograd.grad(wq, mu)

    # Central difference on the SAME random draws (u_arr, u_srv fixed).
    eps = 1e-6
    with torch.no_grad():
        wp = _simulate_mean_wait(torch.tensor(1.0 + eps, dtype=torch.float64),
                                 lam, u_arr, u_srv, warmup)
        wm = _simulate_mean_wait(torch.tensor(1.0 - eps, dtype=torch.float64),
                                 lam, u_arr, u_srv, warmup)
    grad_fd = (wp - wm) / (2 * eps)

    assert grad_ad.item() < 0  # faster service -> less waiting
    assert abs(grad_ad.item() - grad_fd.item()) < 1e-4


def test_matches_mm1_steady_state_theory():
    """Simulated Wq and dWq/dmu approach the M/M/1 closed forms on a long run."""
    n, lam, warmup = 60_000, 0.6, 5000
    g = torch.Generator().manual_seed(2024)
    u_arr = samplers.uniform_noise(n, generator=g)
    u_srv = samplers.uniform_noise(n, generator=g)

    mu = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    wq = _simulate_mean_wait(mu, lam, u_arr, u_srv, warmup)
    (grad_ad,) = torch.autograd.grad(wq, mu)

    muf, rho = 1.0, lam / 1.0
    wq_theory = rho / (muf - lam)                     # = lam / (mu (mu - lam))
    # d/dmu [ lam / (mu (mu - lam)) ] = -lam (2 mu - lam) / (mu (mu - lam))**2
    dwq_theory = -lam * (2 * muf - lam) / (muf * (muf - lam)) ** 2

    assert abs(wq.item() - wq_theory) / wq_theory < 0.15
    assert abs(grad_ad.item() - dwq_theory) / abs(dwq_theory) < 0.25

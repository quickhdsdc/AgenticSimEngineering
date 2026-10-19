"""M2 validation: gradients through the tandem-line max-plus recursion."""

from __future__ import annotations

import torch

from gradientshop.ddes import samplers
from gradientshop.ddes.line import tandem_line
from gradientshop.ddes.queue import single_server


def test_single_station_line_matches_m1_queue():
    """A 1-station line must reproduce the M1 single-server queue exactly."""
    g = torch.Generator().manual_seed(3)
    n = 500
    inter = samplers.exponential(torch.tensor(0.7, dtype=torch.float64),
                                 samplers.uniform_noise(n, generator=g))
    srv = samplers.exponential(torch.tensor(1.0, dtype=torch.float64),
                               samplers.uniform_noise(n, generator=g))
    q = single_server(inter, srv)
    line = tandem_line(inter, srv.unsqueeze(1))
    assert torch.allclose(q.depart, line.depart[:, 0], atol=1e-12)


def test_makespan_gradient_matches_finite_difference():
    """d makespan / d rate_k from autodiff matches central differences,
    on a fixed sample path, for every station."""
    n, m = 300, 3
    g = torch.Generator().manual_seed(11)
    u_arr = samplers.uniform_noise(n, generator=g)
    u_srv = [samplers.uniform_noise(n, generator=g) for _ in range(m)]
    inter = samplers.exponential(torch.tensor(0.5, dtype=torch.float64), u_arr)

    def makespan(rates: torch.Tensor) -> torch.Tensor:
        cols = [samplers.exponential(rates[k], u_srv[k]) for k in range(m)]
        service = torch.stack(cols, dim=1)
        return tandem_line(inter, service).makespan()

    rates = torch.tensor([1.2, 0.9, 1.5], dtype=torch.float64, requires_grad=True)
    ms = makespan(rates)
    (grad_ad,) = torch.autograd.grad(ms, rates)

    eps = 1e-6
    grad_fd = torch.empty(m, dtype=torch.float64)
    with torch.no_grad():
        for k in range(m):
            rp = rates.detach().clone(); rp[k] += eps
            rm = rates.detach().clone(); rm[k] -= eps
            grad_fd[k] = (makespan(rp) - makespan(rm)) / (2 * eps)

    assert torch.allclose(grad_ad, grad_fd, atol=1e-4)
    # Speeding up any station cannot increase makespan: every partial <= 0.
    assert (grad_ad <= 1e-9).all()


def test_bottleneck_optimization_reduces_makespan():
    """Gradient descent on station rates lowers makespan monotonically."""
    n, m = 200, 3
    g = torch.Generator().manual_seed(5)
    u_arr = samplers.uniform_noise(n, generator=g)
    u_srv = [samplers.uniform_noise(n, generator=g) for _ in range(m)]
    inter = samplers.exponential(torch.tensor(0.5, dtype=torch.float64), u_arr)

    rates = torch.tensor([1.0, 0.8, 1.2], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([rates], lr=0.02)
    first = last = None
    for step in range(30):
        opt.zero_grad()
        cols = [samplers.exponential(rates[k], u_srv[k]) for k in range(m)]
        ms = tandem_line(inter, torch.stack(cols, dim=1)).makespan()
        ms.backward()
        opt.step()
        with torch.no_grad():
            rates.clamp_(min=0.3)
        if step == 0:
            first = ms.item()
        last = ms.item()
    assert last < first

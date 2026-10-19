"""M0 validation: pathwise gradients of reparameterized samplers.

We check that autodiff through the inverse-CDF transform recovers the analytic
derivative of the distribution mean w.r.t. its parameter.
"""

from __future__ import annotations

import torch

from gradientshop.ddes import samplers

torch.manual_seed(0)


def test_exponential_mean_gradient_matches_analytic():
    """d E[S]/d rate = -1/rate**2 for S ~ Exp(rate)."""
    g = torch.Generator().manual_seed(42)
    u = samplers.uniform_noise(200_000, generator=g)

    rate = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    s = samplers.exponential(rate, u)
    mean = s.mean()
    mean.backward()

    # Monte-Carlo mean approx 1/rate; its parameter-gradient approx -1/rate**2.
    assert abs(mean.item() - 1.0 / 2.5) < 5e-3
    analytic = -1.0 / 2.5**2
    assert abs(rate.grad.item() - analytic) < 5e-3


def test_exponential_pathwise_derivative_identity():
    """Per-sample dS/drate = -S/rate exactly (elementwise)."""
    g = torch.Generator().manual_seed(1)
    u = samplers.uniform_noise(1000, generator=g)
    rate = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    s = samplers.exponential(rate, u)
    grad = torch.autograd.grad(s.sum(), rate)[0]
    expected = (-s / rate).sum()
    assert torch.allclose(grad, expected.detach(), rtol=1e-10, atol=1e-10)


def test_lognormal_mean_gradient_matches_analytic():
    """d E[S]/d mu = E[S] for S ~ LogNormal(mu, sigma)."""
    g = torch.Generator().manual_seed(7)
    z = samplers.normal_noise(500_000, generator=g)
    mu = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(0.5, dtype=torch.float64)
    s = samplers.lognormal(mu, sigma, z)
    mean = s.mean()
    mean.backward()

    analytic_mean = torch.exp(mu + sigma**2 / 2).item()
    assert abs(mean.item() - analytic_mean) < 1e-2
    # d/dmu exp(mu + sigma^2/2) = exp(mu + sigma^2/2) = E[S].
    assert abs(mu.grad.item() - analytic_mean) < 1e-2

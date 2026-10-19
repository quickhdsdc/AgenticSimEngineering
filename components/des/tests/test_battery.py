"""M4 validation: the battery-remanufacturing scenario is differentiable and
its capacity-allocation gradient identifies the cycling bottleneck."""

from __future__ import annotations

import torch

from gradientshop.shopfloor import battery


def test_makespan_gradient_prefers_the_bottleneck():
    """At equal capacity, makespan is most sensitive to the cycling stage, so
    its capacity gradient has the largest magnitude."""
    cfg = battery.default_config()
    g = torch.Generator().manual_seed(0)
    base = battery.sample_base_service(cfg, 60, generator=g)

    capacity = torch.ones(cfg.n_stages, dtype=torch.float64, requires_grad=True)
    ms = battery.makespan(cfg, capacity, base)
    (grad,) = torch.autograd.grad(ms, capacity)

    cycling = cfg.stage_names.index("Testing/Cycling")
    assert grad[cycling] < 0                      # more channels -> less makespan
    assert grad.abs().argmax().item() == cycling  # it dominates sensitivity


def test_capacity_allocation_reduces_makespan_under_budget():
    """Projected-gradient reallocation of a fixed capacity budget lowers makespan
    and pours most capacity into the cycling stage."""
    cfg = battery.default_config()
    g = torch.Generator().manual_seed(1)
    base = battery.sample_base_service(cfg, 80, generator=g)
    budget = float(cfg.n_stages)  # start from all-ones

    logits = torch.zeros(cfg.n_stages, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)

    def capacity_from(logits: torch.Tensor) -> torch.Tensor:
        # Simplex allocation of the fixed budget across stages.
        return budget * torch.softmax(logits, dim=0)

    ms0 = battery.makespan(cfg, capacity_from(logits).detach(), base).item()
    for _ in range(200):
        opt.zero_grad()
        ms = battery.makespan(cfg, capacity_from(logits), base)
        ms.backward()
        opt.step()
    cap = capacity_from(logits).detach()
    ms1 = battery.makespan(cfg, cap, base).item()

    assert ms1 < ms0
    assert abs(cap.sum().item() - budget) < 1e-6           # budget respected
    cycling = cfg.stage_names.index("Testing/Cycling")
    assert cap.argmax().item() == cycling                  # bottleneck gets the most

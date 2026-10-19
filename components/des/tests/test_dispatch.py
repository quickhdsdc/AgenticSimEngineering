"""M3 validation: the relaxed dispatch decision is faithful and its gradient
is useful for optimising the true (discrete) objective."""

from __future__ import annotations

import torch

from gradientshop.ddes import dispatch

torch.manual_seed(0)


def _instance(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    processing = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    weight = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    return processing, weight


def test_soft_sort_is_row_stochastic():
    scores = torch.randn(10, dtype=torch.float64)
    p = dispatch.soft_sort(scores, tau=0.3)
    assert torch.allclose(p.sum(dim=1), torch.ones(10, dtype=torch.float64), atol=1e-10)


def test_soft_sort_converges_to_hard_permutation():
    """As tau -> 0 the soft matrix approaches the exact argsort permutation."""
    scores = torch.tensor([0.3, 0.9, 0.1, 0.5], dtype=torch.float64)
    idx = torch.argsort(scores, descending=True)
    p = dispatch.soft_sort(scores, tau=1e-3)
    assert torch.equal(p.argmax(dim=1), idx)
    assert p.max(dim=1).values.min() > 0.99  # nearly one-hot rows


def test_straight_through_forward_is_exact():
    """hard=True must reproduce the exact discrete objective on the forward pass."""
    processing, weight = _instance()
    scores = torch.randn(processing.shape[0], dtype=torch.float64)
    relaxed = dispatch.dispatch_cost(processing, scores, weight, tau=0.5, hard=True)
    exact = dispatch.exact_dispatch_cost(processing, scores, weight)
    assert abs(relaxed.item() - exact.item()) < 1e-9


def test_fidelity_gap_shrinks_with_temperature():
    """Smaller tau -> smaller gap between relaxed and exact objective (RQ1)."""
    processing, weight = _instance(n=10, seed=2)
    scores = torch.randn(10, dtype=torch.float64)
    curve = dispatch.fidelity_curve(processing, scores, weight,
                                    taus=[1.0, 0.3, 0.1, 0.03])
    gaps = curve["gap"]
    assert gaps[-1] < gaps[0]                    # monotone-ish improvement
    assert gaps[-1] < 0.05 * curve["exact_cost"]  # tight at small tau


def test_relaxed_gradient_optimises_true_objective():
    """Gradient descent on scores through the RELAXED cost lowers the EXACT
    weighted completion time -- i.e. the relaxation preserves usable structure.

    Ground truth: 1||sum w_j C_j is optimised by WSPT (order by w/p desc)."""
    processing, weight = _instance(n=12, seed=3)
    # Generic (non-optimal) starting order. NB: all-equal scores are a symmetric
    # saddle where the relaxed gradient vanishes, so start from a random order.
    g = torch.Generator().manual_seed(99)
    scores = (0.5 * torch.randn(12, generator=g, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.Adam([scores], lr=0.2)

    exact_start = dispatch.exact_dispatch_cost(processing, scores, weight).item()
    steps = 400
    for t in range(steps):
        # Anneal temperature: smooth (explore) -> sharp (commit to a schedule).
        tau = 0.4 * (0.02 / 0.4) ** (t / (steps - 1))
        opt.zero_grad()
        cost = dispatch.dispatch_cost(processing, scores, weight, tau=tau)
        cost.backward()
        opt.step()
    exact_end = dispatch.exact_dispatch_cost(processing, scores, weight).item()

    # Optimal cost achievable by the WSPT ordering, as the target lower bound.
    wspt_scores = weight / processing
    opt_cost = dispatch.exact_dispatch_cost(processing, wspt_scores, weight).item()

    assert exact_end < exact_start
    # The relaxed gradient drives the true objective to within a few percent of
    # the WSPT optimum. It does NOT reach it exactly -- residual relaxation bias
    # is precisely the RQ1 phenomenon this workstream is meant to characterise.
    rel_gap = (exact_end - opt_cost) / opt_cost
    assert 0.0 <= rel_gap < 0.05

"""Shopfloor dispatching-rule vocabulary."""

from __future__ import annotations

import torch

from gradientshop.shopfloor.rules import Jobs, priority, priority_matrix, rule_names


def _jobs(seed=0, n=6):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    w = torch.rand(n, generator=g, dtype=torch.float64) + 0.2
    d = torch.rand(n, generator=g, dtype=torch.float64) * p.sum().item()
    return Jobs(p, w, d)


def test_spt_prefers_shorter_jobs():
    jobs = _jobs()
    scores = priority("SPT", jobs)
    # Highest SPT score must be the shortest processing time.
    assert scores.argmax().item() == jobs.processing.argmin().item()


def test_wspt_matches_weight_over_time_order():
    jobs = _jobs(seed=2)
    scores = priority("WSPT", jobs)
    expected = jobs.weight / jobs.processing
    assert torch.equal(scores.argsort(descending=True), expected.argsort(descending=True))


def test_priority_matrix_shape_and_standardisation():
    jobs = _jobs(seed=3, n=10)
    names, mat = priority_matrix(jobs)
    assert names == rule_names()
    assert mat.shape == (10, len(names))
    # Standardised columns: ~zero mean, ~unit std.
    assert torch.allclose(mat.mean(dim=0), torch.zeros(len(names), dtype=torch.float64), atol=1e-9)
    assert torch.allclose(mat.std(dim=0), torch.ones(len(names), dtype=torch.float64), atol=1e-6)


def test_subset_of_rules_selectable():
    jobs = _jobs()
    names, mat = priority_matrix(jobs, ["SPT", "EDD"])
    assert names == ["SPT", "EDD"]
    assert mat.shape[1] == 2

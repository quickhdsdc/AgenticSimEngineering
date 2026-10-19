"""DDES-vs-reference fidelity via the first-class validation API.

If these pass, the differentiable model is a faithful surrogate of an
independent event-driven engine -- the premise the whole approach rests on.
"""

from __future__ import annotations

import numpy as np
import torch

from gradientshop.validation import (
    FidelityReport,
    compare_single_server,
    compare_tandem_line,
    compare_trajectories,
)


def test_single_server_is_faithful_to_simpy():
    rng = np.random.default_rng(0)
    n = 400
    inter = torch.tensor(rng.exponential(1 / 0.7, size=n))
    srv = torch.tensor(rng.exponential(1 / 1.0, size=n))

    report = compare_single_server(inter, srv)
    assert isinstance(report, FidelityReport)
    assert report.is_faithful(atol=1e-9)
    assert report.makespan_rel_error < 1e-12


def test_tandem_line_is_faithful_to_simpy():
    rng = np.random.default_rng(1)
    n, m = 250, 4
    inter = torch.tensor(rng.exponential(1 / 0.5, size=n))
    srv = torch.tensor(rng.exponential(1.0, size=(n, m)))

    report = compare_tandem_line(inter, srv)
    assert report.is_faithful(atol=1e-9)
    assert "FAITHFUL" in report.summary()


def test_report_flags_divergence():
    """A perturbed 'DDES' trajectory must be reported as divergent."""
    ddes = torch.arange(10, dtype=torch.float64)
    ref = ddes.clone()
    ref[3] += 0.5
    report = compare_trajectories(ddes, ref)
    assert not report.is_faithful()
    assert abs(report.max_abs_error - 0.5) < 1e-12

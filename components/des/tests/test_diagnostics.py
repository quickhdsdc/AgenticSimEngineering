"""The shared pre-flight check, and the M3 cell it fills in.

Phase 6's judgement table wants, for each relaxation, one quantity a
practitioner can compute on *their* instance before acting on a gradient. Two
cells were filled by earlier phases. This is the third, and it turned out not
to be a new statistic: ``argsort`` dispatch and a mixed-integer ``argmin`` both
approximate an objective that is piecewise constant in the parameter, so
neither admits a finite-difference reference and both need the same operational
answer — does a step actually help?

The negative result is tested too. ``unresolved_fraction`` is the obvious
analogue of ``saturation_fraction`` and it does *not* predict usability; a test
pins that so the statistic is not quietly promoted later.
"""

from __future__ import annotations

import pytest
import torch

from gradientshop.ddes.dispatch import (
    dispatch_cost,
    dispatch_descent_check,
    exact_dispatch_cost,
    soft_sort,
    unresolved_fraction,
)
from opsworld.diagnostics import DescentReport, descent_check


def _instance(seed: int, n: int = 12):
    generator = torch.Generator().manual_seed(seed)
    processing = torch.rand(n, generator=generator, dtype=torch.float64) + 0.2
    weight = torch.rand(n, generator=generator, dtype=torch.float64) + 0.2
    scores = torch.randn(n, generator=generator, dtype=torch.float64)
    return processing, weight, scores


# ----------------------------------------------------------------------
# the generic check
# ----------------------------------------------------------------------
def test_descent_check_finds_the_step_that_helps():
    report = descent_check(
        lambda p: (p[0] - 1.0) ** 2, [0.0], [-2.0], (0.1, 0.5, 1.0)
    )
    assert isinstance(report, DescentReport)
    assert report.start == pytest.approx(1.0)
    assert report.descends
    assert report.best_step == 1.0
    assert report.best_value == pytest.approx(0.0)
    assert report.improvement == pytest.approx(1.0)


def test_descent_check_says_so_when_nothing_helps():
    report = descent_check(lambda p: p[0], [0.0], [-1.0], (0.1, 1.0))
    assert not report.descends
    assert report.best_step is None
    assert "do not act on this gradient" in report.as_text()


def test_descent_check_refuses_a_zero_gradient():
    """No direction is not the same as no improvement available -- Phase 5.3
    had to make that distinction sharply, and it is enforced here."""
    with pytest.raises(ValueError, match="zero gradient"):
        descent_check(lambda p: p[0], [0.0, 0.0], [0.0, 0.0], (1.0,))


def test_descent_check_rejects_a_mismatched_gradient():
    with pytest.raises(ValueError, match="components"):
        descent_check(lambda p: p[0], [0.0, 0.0], [1.0], (1.0,))


def test_normalise_makes_step_sizes_mean_distance():
    """Without normalisation the step is scaled by a gradient magnitude that
    varies by orders of magnitude with the temperature."""
    scaled = descent_check(
        lambda p: abs(p[0] - 1.0), [0.0], [-100.0], (1.0,), normalise=True
    )
    raw = descent_check(
        lambda p: abs(p[0] - 1.0), [0.0], [-100.0], (1.0,), normalise=False
    )
    assert scaled.values[0] == pytest.approx(0.0)
    assert raw.values[0] == pytest.approx(99.0)


# ----------------------------------------------------------------------
# the M3 cell
# ----------------------------------------------------------------------
def test_dispatch_descent_check_scores_the_exact_schedule():
    """The report must be about the discrete cost, not the relaxed surrogate:
    confirming the surrogate is differentiable was never in doubt."""
    processing, weight, scores = _instance(10)
    report = dispatch_descent_check(processing, weight=weight, scores=scores, tau=0.1)
    assert report.start == pytest.approx(
        exact_dispatch_cost(processing, scores, weight).item()
    )
    for value in report.values:
        assert value == pytest.approx(round(value, 10))  # a real schedule cost


def test_the_relaxed_gradient_is_usable_in_the_documented_window():
    """E3 puts the usable window at tau ~ 0.1-0.2. Checked operationally rather
    than through the cosine, over several instances."""
    helped = 0
    for seed in (7, 8, 9, 10, 11, 12):
        processing, weight, scores = _instance(seed)
        report = dispatch_descent_check(
            processing, weight=weight, scores=scores, tau=0.1
        )
        helped += int(report.descends)
    assert helped >= 5


def test_unresolved_fraction_describes_the_forward_pass():
    processing, weight, scores = _instance(7)
    # a large temperature commits to nothing ...
    assert unresolved_fraction(scores, tau=5.0) == pytest.approx(1.0)
    # ... and the rows really are diffuse
    assert soft_sort(scores, 5.0).max(dim=1).values.max().item() < 0.9
    # a small one commits to almost everything
    assert unresolved_fraction(scores, tau=0.01) < 0.3


def test_unresolved_fraction_does_not_predict_whether_the_gradient_helps():
    """The recorded negative result.

    If this statistic predicted usability it would be the M3 analogue of
    ``saturation_fraction``. It does not: the same value accompanies steps that
    help and steps that hurt, so the pre-flight check has to be the operational
    one. Pinned so the statistic is not promoted later by accident.
    """
    outcomes: dict[float, set[bool]] = {}
    for seed in (7, 8, 9, 10, 11, 12):
        processing, weight, scores = _instance(seed)
        for tau in (5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
            fraction = round(unresolved_fraction(scores, tau), 3)
            report = dispatch_descent_check(
                processing, weight=weight, scores=scores, tau=tau
            )
            outcomes.setdefault(fraction, set()).add(report.descends)

    ambiguous = [value for value, seen in outcomes.items() if len(seen) > 1]
    assert ambiguous, (
        "unresolved_fraction separated helpful from harmful gradients on this "
        "sample -- re-examine, it may be predictive after all"
    )


def test_the_cosine_metric_disagrees_with_what_actually_helps():
    """Why the cosine in ``fidelity_curve`` is a proxy and not a verdict.

    Both gradients it compares share the soft Jacobian, so they agree exactly
    when they are both about to vanish. There exists a temperature where the
    cosine is strongly negative and the step nevertheless improves the exact
    cost.
    """
    processing, weight, scores = _instance(7)
    tau = 0.01

    left = scores.detach().clone().requires_grad_(True)
    (g_soft,) = torch.autograd.grad(
        dispatch_cost(processing, left, weight, tau=tau), left
    )
    right = scores.detach().clone().requires_grad_(True)
    (g_hard,) = torch.autograd.grad(
        dispatch_cost(processing, right, weight, tau=tau, hard=True), right
    )
    cosine = torch.nn.functional.cosine_similarity(g_soft, g_hard, dim=0).item()

    report = dispatch_descent_check(processing, weight=weight, scores=scores, tau=tau)
    assert cosine < -0.9
    assert report.descends and report.improvement > 0.0

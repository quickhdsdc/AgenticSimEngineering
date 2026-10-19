"""M3 -- Differentiable dispatch: continuous relaxation of the sequencing
decision (proposal source B, and the heart of RQ1).

Event *timing* is differentiable a.e. (M0-M2). A *dispatch decision* -- which
queued job to process next -- is a genuine ``argmax`` over priority scores and
carries no gradient. We relax it with **SoftSort** (Prillo & Eisenberger,
ICML 2020), a temperature-controlled surrogate of ``argsort``:

    P = softmax( -|sort_desc(s)[:, None] - s[None, :]| / tau ,  dim=1)

Row ``k`` of the row-stochastic matrix ``P`` is a soft one-hot for "which job
sits at rank ``k``". As ``tau -> 0`` it approaches the hard permutation. A
straight-through variant (``hard=True``) runs the *exact* discrete schedule on
the forward pass (a valid, unbiased objective) while routing the soft gradient
backward -- the pragmatic estimator flagged by proposal ref [15] (Gumbel-Softmax
family).

Given the permutation we schedule a single machine and score it with standard
objectives (weighted completion time, weighted tardiness). ``tau`` is the exact
knob RQ1 studies: large ``tau`` = smooth but biased; small ``tau`` = faithful
but with vanishing/spiky gradients. :func:`fidelity_curve` sweeps it.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "soft_sort",
    "schedule_completions",
    "weighted_completion_time",
    "weighted_tardiness",
    "dispatch_cost",
    "exact_dispatch_cost",
    "fidelity_curve",
    "unresolved_fraction",
    "dispatch_descent_check",
]


def soft_sort(scores: Tensor, tau: float = 0.1, *, hard: bool = False) -> Tensor:
    """Soft permutation matrix sorting ``scores`` in *descending* order.

    Args:
        scores: ``[N]`` priority score per job (higher = dispatched earlier).
        tau: relaxation temperature; smaller is closer to a hard permutation.
        hard: if True, straight-through -- forward is the exact permutation,
            backward uses the soft Jacobian.

    Returns:
        ``[N, N]`` row-stochastic matrix ``P``; ``P[k, j]`` ~ P(job ``j`` at
        rank ``k``).
    """
    s_sorted, _ = torch.sort(scores, descending=True)
    pairwise = (s_sorted.unsqueeze(1) - scores.unsqueeze(0)).abs()
    p_soft = torch.softmax(-pairwise / tau, dim=1)
    if not hard:
        return p_soft
    idx = torch.argsort(scores, descending=True)
    p_hard = torch.zeros_like(p_soft)
    p_hard[torch.arange(p_soft.shape[0], device=scores.device), idx] = 1.0
    return (p_hard - p_soft).detach() + p_soft  # straight-through estimator


def schedule_completions(processing: Tensor, perm: Tensor) -> Tensor:
    """Per-*job* completion times on a single machine under permutation ``perm``.

    ``processing`` is ``[N]``; ``perm`` is a (soft) ``[N, N]`` permutation whose
    row ``k`` is the job at rank ``k``. Jobs run back-to-back in rank order.
    """
    p_rank = perm @ processing              # expected processing time at each rank
    c_rank = torch.cumsum(p_rank, dim=0)    # completion time at each rank
    return perm.transpose(0, 1) @ c_rank    # scatter completions back to jobs


def weighted_completion_time(completion: Tensor, weight: Tensor) -> Tensor:
    """Sum of ``weight_j * C_j`` (minimised by the WSPT rule)."""
    return (weight * completion).sum()


def weighted_tardiness(completion: Tensor, due: Tensor, weight: Tensor) -> Tensor:
    """Sum of ``weight_j * max(C_j - due_j, 0)`` -- the classic 1||sum w_j T_j."""
    return (weight * torch.clamp(completion - due, min=0.0)).sum()


def dispatch_cost(processing: Tensor, scores: Tensor, weight: Tensor,
                  *, due: Tensor | None = None, tau: float = 0.1,
                  hard: bool = False) -> Tensor:
    """End-to-end differentiable dispatch objective as a function of ``scores``."""
    perm = soft_sort(scores, tau, hard=hard)
    completion = schedule_completions(processing, perm)
    if due is None:
        return weighted_completion_time(completion, weight)
    return weighted_tardiness(completion, due, weight)


def exact_dispatch_cost(processing: Tensor, scores: Tensor, weight: Tensor,
                        *, due: Tensor | None = None) -> Tensor:
    """The true (non-relaxed) objective under the hard argsort schedule.

    Used to measure how far the relaxed surrogate deviates (RQ1 fidelity).
    """
    idx = torch.argsort(scores, descending=True)
    c_sorted = torch.cumsum(processing[idx], dim=0)
    completion = torch.empty_like(c_sorted)
    completion[idx] = c_sorted
    if due is None:
        return weighted_completion_time(completion, weight)
    return weighted_tardiness(completion, due, weight)


def fidelity_curve(processing: Tensor, scores: Tensor, weight: Tensor,
                   taus: list[float], *, due: Tensor | None = None) -> dict:
    """RQ1 fidelity probe: sweep ``tau`` and report where the relaxation holds.

    For each ``tau`` returns the relaxed objective, its absolute gap to the
    exact objective, and the cosine similarity between the relaxed gradient
    (w.r.t. ``scores``) and the straight-through gradient -- a proxy for whether
    the smooth gradient still points the right way.

    Returns a dict of parallel lists keyed ``tau``, ``soft_cost``, ``gap``,
    ``grad_cosine``.
    """
    exact = exact_dispatch_cost(processing, scores, weight, due=due).item()

    def grad_of(tau: float, hard: bool) -> tuple[float, Tensor]:
        s = scores.detach().clone().requires_grad_(True)
        cost = dispatch_cost(processing, s, weight, due=due, tau=tau, hard=hard)
        (g,) = torch.autograd.grad(cost, s)
        return cost.item(), g

    out: dict[str, list[float]] = {"tau": [], "soft_cost": [], "gap": [],
                                   "grad_cosine": []}
    for tau in taus:
        soft_cost, g_soft = grad_of(tau, hard=False)
        _, g_st = grad_of(tau, hard=True)
        cos = torch.nn.functional.cosine_similarity(
            g_soft.flatten(), g_st.flatten(), dim=0).item()
        out["tau"].append(tau)
        out["soft_cost"].append(soft_cost)
        out["gap"].append(abs(soft_cost - exact))
        out["grad_cosine"].append(cos)
    out["exact_cost"] = exact  # type: ignore[assignment]
    return out


def unresolved_fraction(scores: Tensor, tau: float = 0.1,
                        threshold: float = 0.9) -> float:
    """Share of ranks the relaxation has *not* committed to a single job.

    A row of ``P`` whose largest entry is below ``threshold`` is still spread
    over several jobs, i.e. the relaxation has not decided who sits at that
    rank. The statistic is cheap and it describes the forward pass exactly.

    .. warning::
       **It does not predict whether the gradient is usable**, and it was added
       to record that. It is the obvious analogue of
       :func:`~intralogistics.differentiable.saturation_fraction`, which *does*
       predict quality for the in-loop replenishment relaxation — but the two
       relaxations differ in kind. Measured over six instances and nine
       temperatures: fully unresolved (``1.000``) rows accompany both the best
       descent steps of one instance and steps that actively worsen another,
       and fully resolved rows do the same.

       Use :func:`dispatch_descent_check` to decide whether to act on the
       gradient. This one is for describing *what the relaxation did*, not for
       trusting it.
    """
    p = soft_sort(scores, tau)
    return (p.max(dim=1).values < threshold).to(torch.float64).mean().item()


def dispatch_descent_check(processing: Tensor, scores: Tensor, weight: Tensor,
                           *, due: Tensor | None = None, tau: float = 0.1,
                           step_sizes: tuple[float, ...] = (0.02, 0.05, 0.1, 0.3, 1.0)):
    """Does a step along the relaxed gradient lower the **exact** cost?

    The pre-flight check for M3, and it is deliberately the same one
    :mod:`intralogistics.perturbed` uses for the mixed-integer ``argmin``: both
    relaxations approximate an objective that is piecewise constant in the
    parameter, so neither admits a finite-difference reference and both have to
    be judged operationally. See :mod:`opsworld.diagnostics`.

    Returns an :class:`~opsworld.diagnostics.DescentReport`.
    """
    from opsworld.diagnostics import descent_check

    s = scores.detach().clone().requires_grad_(True)
    cost = dispatch_cost(processing, s, weight, due=due, tau=tau)
    (gradient,) = torch.autograd.grad(cost, s)

    def exact(point) -> float:
        candidate = torch.tensor(list(point), dtype=scores.dtype,
                                 device=scores.device)
        return exact_dispatch_cost(processing, candidate, weight, due=due).item()

    return descent_check(
        exact,
        scores.detach().tolist(),
        gradient.tolist(),
        step_sizes,
    )

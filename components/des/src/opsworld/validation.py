"""Fidelity between two world models — a cross-cutting concern.

This layer sits across the architecture rather than inside it, because its
input is **two world models** (``PLAN.md`` A4). It is a first-class module and
not a test helper, because "when is my tractable relaxation faithful, and when
does it mislead?" is the shared scientific question of the whole framework
(Axiom 3):

============================  ============================  ==================
reference                     approximation                 relaxation
============================  ============================  ==================
exact discrete schedule       SoftSort relaxation           temperature tau
event-driven simulation       differentiable recursion      — (should be exact)
dynamic replenishment (T1)    time-averaged fluid (T0)      the projection
integer programme             LP relaxation                 integrality
============================  ============================  ==================

The decisive number is **not** the numeric error. An approximation exists to
*pick the right decision*; getting the wrong number while preserving the
ordering of candidates is a usable approximation, and getting the right number
while inverting the ordering is not. Hence ``rank_correlation`` and
``false_feasible`` sit alongside the error norms.

.. note::
   Trajectory-level comparison (DDES departure times vs a SimPy reference)
   stays in :mod:`gradientshop.validation`: it compares *trajectories*, not
   *decisions*, which is a genuinely different object.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .world import Evaluation, WorldModel

__all__ = ["FidelityReport", "compare_world_models", "spearman"]


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks, ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation. ``None`` when undefined (constant input)."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / math.sqrt(var_a * var_b)


@dataclass
class FidelityReport:
    """How well an approximate world model stands in for a reference one."""

    n_decisions: int
    max_abs_error: float
    rmse: float
    max_rel_error: float
    rank_correlation: float | None
    false_feasible: int
    false_infeasible: int
    reference_provenance: str
    approximation_provenance: str

    def is_faithful(self, atol: float = 1e-6) -> bool:
        """Numeric agreement — the right question only in the timing regime."""
        return self.max_abs_error <= atol

    def preserves_ranking(self, threshold: float = 0.9) -> bool:
        """Whether the approximation orders decisions like the reference.

        This is the question that matters when the approximation is *meant* to
        differ from the reference — a relaxation, or a time average.
        """
        return self.rank_correlation is not None and self.rank_correlation >= threshold

    def misleads(self) -> bool:
        """Actively harmful: the ordering is inverted, or feasibility is wrong."""
        inverted = self.rank_correlation is not None and self.rank_correlation < 0
        return inverted or self.false_feasible > 0

    def summary(self) -> str:
        rho = "n/a" if self.rank_correlation is None else f"{self.rank_correlation:+.4f}"
        return (
            f"[{self.approximation_provenance}] vs [{self.reference_provenance}] "
            f"over {self.n_decisions} decisions\n"
            f"    max|err| = {self.max_abs_error:.6e}   rmse = {self.rmse:.6e}   "
            f"max rel = {self.max_rel_error:.6e}\n"
            f"    rank correlation = {rho}   "
            f"false-feasible = {self.false_feasible}   "
            f"false-infeasible = {self.false_infeasible}"
        )


def compare_world_models(
    reference: WorldModel,
    approximation: WorldModel,
    decisions: Sequence[Any],
    *,
    tol: float = 1e-6,
) -> FidelityReport:
    """Evaluate the same decisions under both models and report the gap.

    ``false_feasible`` counts decisions the approximation calls feasible but
    the reference rejects — the dangerous direction, since those are exactly
    the plans an optimiser would happily return.
    """
    if not decisions:
        raise ValueError("need at least one decision to compare")

    ref: list[Evaluation] = [reference.evaluate(d) for d in decisions]
    approx: list[Evaluation] = [approximation.evaluate(d) for d in decisions]

    errors = [abs(a.objective - r.objective) for r, a in zip(ref, approx)]
    rel_errors = [
        abs(a.objective - r.objective) / abs(r.objective) if r.objective else 0.0
        for r, a in zip(ref, approx)
    ]
    false_feasible = sum(
        1 for r, a in zip(ref, approx)
        if a.is_feasible(tol) and not r.is_feasible(tol)
    )
    false_infeasible = sum(
        1 for r, a in zip(ref, approx)
        if r.is_feasible(tol) and not a.is_feasible(tol)
    )

    return FidelityReport(
        n_decisions=len(decisions),
        max_abs_error=max(errors),
        rmse=math.sqrt(sum(e * e for e in errors) / len(errors)),
        max_rel_error=max(rel_errors),
        rank_correlation=spearman([r.objective for r in ref],
                                  [a.objective for a in approx]),
        false_feasible=false_feasible,
        false_infeasible=false_infeasible,
        reference_provenance=ref[0].provenance,
        approximation_provenance=approx[0].provenance,
    )

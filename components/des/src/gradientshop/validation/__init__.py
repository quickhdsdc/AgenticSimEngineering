"""DDES-vs-reference fidelity validation (RQ1).

Turns the "is the differentiable surrogate faithful?" question into a reusable
API over the :mod:`gradientshop.ddes` core and the :mod:`gradientshop.reference`
ground truth.
"""

from __future__ import annotations

from .compare import (
    FidelityReport,
    compare_single_server,
    compare_tandem_line,
    compare_trajectories,
    dispatch_fidelity,
)

__all__ = [
    "FidelityReport", "compare_trajectories",
    "compare_single_server", "compare_tandem_line", "dispatch_fidelity",
]

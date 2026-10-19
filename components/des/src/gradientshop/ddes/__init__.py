"""GradientShop DDES core -- a differentiable discrete-event simulator.

Workstream Task 2.1 of the GradientShop proposal, built as a milestone ladder:

* **M0** (:mod:`.samplers`)  -- reparameterizable stochastic primitives.
* **M1** (:mod:`.queue`)     -- single-server FIFO Lindley recursion.
* **M2** (:mod:`.line`)      -- multi-station tandem line (fixed routing).
* **M3** (:mod:`.dispatch`)  -- differentiable dispatch via SoftSort relaxation.

Ground-truth SimPy engines live in :mod:`gradientshop.reference`; the
DDES-vs-reference comparison in :mod:`gradientshop.validation`. The shopfloor
domain (dispatch rules, battery scenario) is :mod:`gradientshop.shopfloor`; the
LLM coupling (D2) is :mod:`gradientshop.agent`.
"""

from __future__ import annotations

from . import dispatch, line, queue, samplers
from .dispatch import soft_sort, weighted_completion_time, weighted_tardiness
from .line import LineTrajectory, tandem_line
from .queue import QueueTrajectory, single_server

__all__ = [
    "samplers", "queue", "line", "dispatch",
    "QueueTrajectory", "single_server",
    "LineTrajectory", "tandem_line",
    "soft_sort", "weighted_completion_time", "weighted_tardiness",
]

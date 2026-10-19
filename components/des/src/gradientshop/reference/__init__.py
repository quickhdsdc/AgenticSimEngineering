"""Non-differentiable ground-truth DES engines (proposal D1 open-source reference).

The DDES core (:mod:`gradientshop.ddes`) re-expresses shopfloor models as tensor
recursions so gradients flow. These SimPy engines are the *independent oracle*
the DDES is validated against -- a real process-based event heap with no notion
of differentiation. The comparison itself lives in :mod:`gradientshop.validation`.
"""

from __future__ import annotations

from .simpy_engine import simpy_single_server, simpy_tandem_line

__all__ = ["simpy_single_server", "simpy_tandem_line"]

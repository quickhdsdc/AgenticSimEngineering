"""First-class DDES <-> reference fidelity comparison (RQ1 infrastructure).

The central scientific question of GradientShop is whether the *differentiable*
surrogate stays faithful to a real discrete-event simulation. This module makes
that comparison a reusable API rather than a one-off test: run the DDES and the
SimPy ground truth on an identical sample path and report where -- and by how
much -- they diverge.

Two regimes:
* **Timing fidelity** (M1/M2): with fixed routing the two engines should agree to
  floating-point tolerance; any gap is a modelling bug. :func:`compare_single_server`,
  :func:`compare_tandem_line`.
* **Relaxation fidelity** (M3): the soft dispatch *deliberately* departs from the
  discrete schedule; here the gap is the object of study, swept over temperature.
  :func:`dispatch_fidelity` (thin re-export of the dispatch fidelity curve).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..ddes.line import tandem_line
from ..ddes.queue import single_server
from ..reference import simpy_single_server, simpy_tandem_line

__all__ = ["FidelityReport", "compare_trajectories", "compare_single_server",
           "compare_tandem_line", "dispatch_fidelity"]


@dataclass
class FidelityReport:
    """Discrepancy between a DDES trajectory and its reference.

    ``max_abs_error`` is the decisive number for timing fidelity (should be ~0);
    the others summarise the distribution and a headline KPI (makespan).
    """

    n_events: int
    max_abs_error: float
    rmse: float
    makespan_ddes: float
    makespan_reference: float

    @property
    def makespan_rel_error(self) -> float:
        ref = self.makespan_reference
        return abs(self.makespan_ddes - ref) / ref if ref else 0.0

    def is_faithful(self, atol: float = 1e-6) -> bool:
        """True when the DDES reproduces the reference to ``atol`` (timing regime)."""
        return self.max_abs_error <= atol

    def summary(self) -> str:
        flag = "FAITHFUL" if self.is_faithful() else "DIVERGENT"
        return (f"[{flag}] events={self.n_events} max|err|={self.max_abs_error:.2e} "
                f"rmse={self.rmse:.2e} makespan ddes={self.makespan_ddes:.4f} "
                f"ref={self.makespan_reference:.4f} "
                f"(rel {self.makespan_rel_error:.2e})")


def compare_trajectories(ddes_depart: Tensor, reference_depart) -> FidelityReport:
    """Compare two departure-time arrays (any shape, matched elementwise)."""
    ddes = ddes_depart.detach().to(torch.float64)
    ref = torch.as_tensor(reference_depart, dtype=torch.float64)
    if ddes.shape != ref.shape:
        raise ValueError(f"shape mismatch: DDES {tuple(ddes.shape)} vs "
                         f"reference {tuple(ref.shape)}")
    err = (ddes - ref).abs()
    # Makespan = last exit time along the last axis (flattened last column).
    ms_ddes = ddes.reshape(ddes.shape[0], -1)[:, -1].max().item()
    ms_ref = ref.reshape(ref.shape[0], -1)[:, -1].max().item()
    return FidelityReport(
        n_events=err.numel(),
        max_abs_error=err.max().item(),
        rmse=torch.sqrt((err ** 2).mean()).item(),
        makespan_ddes=ms_ddes,
        makespan_reference=ms_ref,
    )


def compare_single_server(interarrival: Tensor, service: Tensor) -> FidelityReport:
    """Run the M1 queue on both engines and report the discrepancy."""
    ddes = single_server(interarrival, service).depart
    ref = simpy_single_server(interarrival.detach().tolist(),
                              service.detach().tolist())
    return compare_trajectories(ddes, ref)


def compare_tandem_line(interarrival: Tensor, service: Tensor) -> FidelityReport:
    """Run the M2 tandem line on both engines and report the discrepancy."""
    ddes = tandem_line(interarrival, service).depart
    ref = simpy_tandem_line(interarrival.detach().tolist(),
                            service.detach().numpy())
    return compare_trajectories(ddes, ref)


def dispatch_fidelity(processing: Tensor, scores: Tensor, weight: Tensor,
                      taus: list[float], *, due: Tensor | None = None) -> dict:
    """Relaxation-fidelity sweep for the M3 dispatch decision (see
    :func:`gradientshop.ddes.dispatch.fidelity_curve`)."""
    from ..ddes.dispatch import fidelity_curve
    return fidelity_curve(processing, scores, weight, taus, due=due)

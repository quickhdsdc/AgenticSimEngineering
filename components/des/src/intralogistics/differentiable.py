"""Phase 4 — the replenishment dynamics as a differentiable recursion.

Phase 3 established that the time average picks the wrong carrier size. This
module asks the *methodological* question behind that: the reorder trigger is a
discrete decision with no gradient, so what does relaxing it cost?

    ``order_t = Q * 1[position_t <= s]``

The indicator is mathematically the same kind of object as the ``argsort`` in
M3 dispatch: a zero-gradient discrete decision inside a simulation. Relaxing it
with a sigmoid at temperature ``tau`` is the direct analogue of SoftSort, so
the two domains can be swept the same way and put side by side.

That comparison is the point. GradientShop's central RQ1 result
(``docs/documentation.md`` §3.4) is a *negative* one: as ``tau -> 0`` the
SoftSort schedule fidelity improves monotonically while the gradient's
agreement with the true gradient collapses and then **inverts**, leaving only a
narrow usable window. If the same shape appears here — a different relaxation
(threshold, not ranking), a different domain, a different objective — then the
finding is a property of relaxed simulation rather than of one example. If it
does *not*, that is equally informative: it would mean the phenomenon depends
on the specific form of the relaxation, and the open question becomes which
forms are affected.

:func:`fidelity_curve` deliberately mirrors
:func:`gradientshop.ddes.dispatch.fidelity_curve` — same arguments, same output
keys — so the two curves are directly comparable.

**Regime.** One bin is ordered per trigger, i.e. two-bin Kanban, which is valid
while lead-time demand stays below the bin size. That is the regime constraint
(11) of the source MILP presumes, and Phase 3 showed what happens outside it
(the replenishment rate saturates silently).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "ReplenishmentTrajectory",
    "relaxed_replenishment",
    "replenishment_cost",
    "exact_replenishment_cost",
    "fidelity_curve",
    "gradient_norm",
    "gradient_growth_factor",
    "stability_curve",
    "saturation_fraction",
]


class ReplenishmentTrajectory(torch.nn.Module):  # container only; no parameters
    """Aggregates of one replay: mean on-hand stock and orders per period."""

    def __init__(self, avg_inventory: Tensor, moves_per_period: Tensor) -> None:
        super().__init__()
        self.avg_inventory = avg_inventory
        self.moves_per_period = moves_per_period


def _trigger(gap: Tensor, tau: float, hard: bool) -> Tensor:
    """Relaxed reorder indicator ``1[gap >= 0]`` at temperature ``tau``.

    ``hard=True`` is the straight-through variant: the forward pass runs the
    exact discrete trajectory, the backward pass uses the sigmoid's Jacobian.
    Same convention as the dispatch relaxation, so the two are comparable.
    """
    soft = torch.sigmoid(gap / tau)
    if not hard:
        return soft
    exact = (gap >= 0).to(soft.dtype)
    return soft + (exact - soft).detach()


def relaxed_replenishment(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    *,
    lead_time: int = 3,
    tau: float = 0.1,
    hard: bool = False,
) -> ReplenishmentTrajectory:
    """Differentiable ``(s, Q)`` replenishment over ``N`` items and ``T`` periods.

    ``capacity`` ``(N,)`` is the decision; ``demand`` ``(N, T)`` is fixed noise,
    held constant across parameter settings so autodiff is comparable to a
    finite difference on a single sample path — the common-random-numbers
    discipline used throughout the DDES core.

    Every operation is differentiable a.e. except the reorder indicator, which
    is exactly what ``tau`` relaxes.
    """
    if capacity.ndim != 1 or demand.ndim != 2:
        raise ValueError("capacity must be (N,) and demand (N, T)")
    if capacity.shape[0] != demand.shape[0]:
        raise ValueError("capacity and demand disagree on the number of items")

    horizon = demand.shape[1]
    net = reorder_point + capacity           # start at the order-up-to level
    pipeline = [torch.zeros_like(capacity) for _ in range(max(1, lead_time))]

    on_hand_total = torch.zeros_like(capacity)
    orders_total = torch.zeros_like(capacity)

    for t in range(horizon):
        net = net + pipeline.pop(0)
        net = net - demand[:, t]
        on_hand_total = on_hand_total + torch.relu(net)

        position = net + torch.stack(pipeline).sum(dim=0)
        fires = _trigger(reorder_point - position, tau, hard)
        pipeline.append(capacity * fires)
        orders_total = orders_total + fires

    return ReplenishmentTrajectory(
        avg_inventory=on_hand_total / horizon,
        moves_per_period=orders_total / horizon,
    )


def replenishment_cost(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    *,
    lead_time: int = 3,
    tau: float = 0.1,
    hard: bool = False,
    storage_rate: float = 1.0,
    area_fixed: float = 1.0 / 3.0,
    area_per_unit: float = 1.0 / 15.0,
    move_cost: float = 12.0,
) -> Tensor:
    """Total storage + replenishment cost of a vector of carrier capacities.

    The carrier count is ``avg_inventory / capacity``, the smooth counterpart of
    the MILP's ``ceil`` in constraint (21): rounding is a second, unrelated
    source of non-differentiability and is deliberately kept out of this
    experiment so the sweep isolates the reorder trigger.
    """
    trajectory = relaxed_replenishment(
        capacity, demand, reorder_point, lead_time=lead_time, tau=tau, hard=hard
    )
    carriers = trajectory.avg_inventory / capacity
    area = area_fixed + area_per_unit * capacity
    storage = storage_rate * carriers * area
    relocation = move_cost * trajectory.moves_per_period
    return (storage + relocation).sum()


def exact_replenishment_cost(
    capacity: Tensor, demand: Tensor, reorder_point: Tensor, **kwargs
) -> Tensor:
    """Cost of the exact discrete trajectory (hard indicator, no relaxation)."""
    with torch.no_grad():
        return replenishment_cost(
            capacity.detach(), demand, reorder_point, tau=1e-12, hard=True, **kwargs
        )


def fidelity_curve(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    taus: list[float],
    **kwargs,
) -> dict:
    """RQ1 fidelity probe for the reorder trigger — the analogue of M3's sweep.

    For each ``tau``: the relaxed objective, its gap to the exact discrete
    objective, and the cosine similarity between the relaxed gradient w.r.t.
    ``capacity`` and the straight-through gradient.

    Returns parallel lists keyed ``tau``, ``soft_cost``, ``gap``,
    ``grad_cosine``, plus ``exact_cost`` — the same keys as
    :func:`gradientshop.ddes.dispatch.fidelity_curve`.
    """
    exact = exact_replenishment_cost(capacity, demand, reorder_point, **kwargs).item()

    def probe(tau: float, hard: bool) -> tuple[float, Tensor]:
        q = capacity.detach().clone().requires_grad_(True)
        cost = replenishment_cost(
            q, demand, reorder_point, tau=tau, hard=hard, **kwargs
        )
        (gradient,) = torch.autograd.grad(cost, q)
        return cost.item(), gradient

    out: dict[str, list[float]] = {
        "tau": [], "soft_cost": [], "gap": [], "grad_cosine": [],
    }
    for tau in taus:
        soft_cost, g_soft = probe(tau, hard=False)
        _, g_straight_through = probe(tau, hard=True)
        cosine = torch.nn.functional.cosine_similarity(
            g_soft.flatten(), g_straight_through.flatten(), dim=0
        ).item()
        out["tau"].append(tau)
        out["soft_cost"].append(soft_cost)
        out["gap"].append(abs(soft_cost - exact))
        out["grad_cosine"].append(cosine)
    out["exact_cost"] = exact  # type: ignore[assignment]
    return out


# ----------------------------------------------------------------------
# a diagnostic the dispatch relaxation does not need
# ----------------------------------------------------------------------
def gradient_norm(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    *,
    tau: float,
    horizon: int | None = None,
    **kwargs,
) -> float:
    """Norm of the relaxed gradient w.r.t. ``capacity`` over ``horizon`` periods."""
    window = demand if horizon is None else demand[:, :horizon]
    q = capacity.detach().clone().requires_grad_(True)
    cost = replenishment_cost(q, window, reorder_point, tau=tau, hard=False, **kwargs)
    (gradient,) = torch.autograd.grad(cost, q)
    return gradient.norm().item()


def gradient_growth_factor(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    *,
    tau: float,
    short: int = 75,
    long: int = 100,
    **kwargs,
) -> float:
    """Per-period growth of the gradient norm — the loop-gain diagnostic.

    The relaxed reorder decision sits **inside a feedback loop**: an order
    changes the inventory position, which changes the next trigger. The
    Jacobian is therefore a product of one factor per period, and a per-period
    factor above 1 means the gradient grows exponentially in the horizon.

    The exact system has no such term — a step function's derivative is zero
    almost everywhere — so this instability is created by the relaxation, not
    approximated by it. There is no analogue in the one-shot dispatch
    relaxation, where the relaxed decision is applied once.

    Returns ``(||g(long)|| / ||g(short)||) ** (1 / (long - short))``; ``1.0``
    means a stable loop.
    """
    if long <= short:
        raise ValueError("long horizon must exceed the short one")
    a = gradient_norm(capacity, demand, reorder_point, tau=tau, horizon=short, **kwargs)
    b = gradient_norm(capacity, demand, reorder_point, tau=tau, horizon=long, **kwargs)
    if a <= 0.0:
        return float("nan")
    return (b / a) ** (1.0 / (long - short))


def saturation_fraction(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    *,
    tau: float,
    lead_time: int = 3,
    cutoff: float = 6.0,
) -> float:
    """Share of ``(item, period)`` pairs where the sigmoid is **not** saturated.

    This is the quantity that decides whether the relaxed gradient is usable,
    and it should be checked before trusting one.

    When the inventory position sits far from the reorder point relative to
    ``tau``, the sigmoid saturates, its derivative vanishes, and what remains is
    the exact pathwise gradient of the discrete system. When the trajectory
    *lingers near the threshold*, each such period injects a factor of order
    ``Q / (4 * tau)`` into the Jacobian, and those factors compound along the
    horizon.

    The effect is brutally sharp: measured across instances, **0.00%
    unsaturated gives a cosine of +1.0000 against a finite difference of the
    exact cost, while 0.90% already drops it to +0.09**. It is therefore not
    enough to make ``tau`` small — one must check that the trajectory stays
    away from the threshold, which is a property of the instance *and* of the
    horizon, since a longer horizon means more chances to pass near it.
    """
    net = reorder_point + capacity
    pipeline = [torch.zeros_like(capacity) for _ in range(max(1, lead_time))]
    near = 0
    total = 0
    with torch.no_grad():
        for t in range(demand.shape[1]):
            net = net + pipeline.pop(0) - demand[:, t]
            position = net + torch.stack(pipeline).sum(dim=0)
            x = (reorder_point - position) / tau
            near += int((x.abs() < cutoff).sum())
            total += capacity.numel()
            pipeline.append(capacity * torch.sigmoid(x))
    return near / total if total else 0.0


def stability_curve(
    capacity: Tensor,
    demand: Tensor,
    reorder_point: Tensor,
    taus: list[float],
    horizons: list[int],
    **kwargs,
) -> dict:
    """Gradient norm over a ``tau`` x ``horizon`` grid, plus the growth factor.

    Returns ``{"tau": [...], "horizons": [...], "norm": [[...]],
    "growth_factor": [...]}`` with ``norm[i][j]`` for ``taus[i]``,
    ``horizons[j]``.
    """
    norms = [
        [
            gradient_norm(capacity, demand, reorder_point, tau=tau, horizon=h, **kwargs)
            for h in horizons
        ]
        for tau in taus
    ]
    growth = [
        gradient_growth_factor(capacity, demand, reorder_point, tau=tau, **kwargs)
        for tau in taus
    ]
    return {
        "tau": list(taus),
        "horizons": list(horizons),
        "norm": norms,
        "growth_factor": growth,
    }

"""Pathwise gradients *through* the MILP — Phase 5.3 of ``PLAN.md``.

The capability table (§1.6) lists the MILP as ``C0 + C2b + C3``: it evaluates,
it hands out shadow prices, it has algebraic structure. What it does **not**
have is ``C2a``, a pathwise gradient — and decision record D10 insists the two
first-order objects are not interchangeable. A shadow price answers *"which
constraint binds?"*; it is a price on a right-hand side and
:meth:`~opsworld.world.Sensitivity.descent_direction` refuses to step along it.
This module supplies the object that *can* be stepped along, and thereby turns
``C3`` into ``C2a``.

The question it answers
-----------------------
The MILP is solved with **planning parameters** — cost rates that a planner
types in. Phase 1 found one of them is not a cost at all: the ``1e5`` storage
rate at the points of use is a placeholder meant to discourage stocking there,
and on the minimal instance it makes up 99.994% of the objective while changing
no decision whatsoever. So the honest question is not "what does the MILP
cost?" but:

    **Which planning rates make the MILP return a configuration that is
    actually cheap once the dynamics are simulated?**

Written out, that is a bilevel problem::

    theta  ->  x*(theta) = argmin_x  c(theta)^T x   s.t. x in X     (the MILP)
           ->  L(theta)  = loss(x*(theta))                          (the truth)

``L`` is piecewise constant in ``theta``: the argmin sits on a vertex until a
cost boundary is crossed, then jumps. Its derivative is zero almost everywhere
and undefined on the boundaries — exactly the same pathology as ``argsort`` in
:mod:`gradientshop.ddes.dispatch` and the reorder indicator in
:mod:`intralogistics.differentiable`, and it needs the same treatment: a
relaxation.

The relaxation, and why it is a *third* structural type
------------------------------------------------------
Following Berthet et al. (2020), smooth the argmin by **perturbing its inputs**
rather than softening its formula::

    L_eps(theta) = E_Z[ loss(x*(theta + eps Z)) ],   Z ~ N(0, I)
    grad L_eps   = (1/eps) E_Z[ loss(x*(theta + eps Z)) Z ]

Phase 6's matrix is indexed by the *structure* of the relaxation, not by the
domain (that rewrite is Phase 4's conclusion). This is the third entry, and it
differs from both existing ones in a way that matters:

===================  ==================  ====================  ==================
                     M3 dispatch         replenishment         MILP argmin
                     (one-shot,          (in-loop,             (one-shot,
                     deterministic)      deterministic)        stochastic)
===================  ==================  ====================  ==================
forward pass         *approximate*       *approximate*         **exact** — a real
                     (SoftSort output    (sigmoid trigger)     integer solution
                     is not a permutation)
what breaks          direction (bias)    magnitude             **reach** — there
as tau/eps -> 0                          (explosion)           is no local
                                                               information at all
===================  ==================  ====================  ==================

The forward pass staying exact is the practical selling point: the planner
never has to act on a fractional configuration. The price is subtler than
sampling noise. As ``eps -> 0`` the perturbation stops flipping the argmin, so
every draw returns the same solution, the estimator is **identically zero**,
and no number of samples recovers anything — the loss really is constant on
that ball. Widening ``eps`` buys signal and spends fidelity. So the usable
range is bounded on both sides, and its lower bound is the distance from
``theta`` to the nearest decision boundary: **a property of the instance, not
of the method.** Same trade-off shape as M3, an entirely different mechanism.

There is a second, quieter failure. Mixed-integer programs are routinely
degenerate, and when several optima tie, ``theta`` sits *on* a jump of the
loss rather than near one. Perturbing then merely breaks the tie, the loss does
vary, and every "did anything change?" check passes — while the gradient
diverges as ``1/eps`` and points at nothing reachable. The signature is that
``eps * ||grad||`` stays constant as ``eps`` moves; see :func:`epsilon_curve`.

Why the parameters are held in log space
----------------------------------------
The rates of the source instance span ``1.0`` to ``1e5``. An additive
perturbation that is meaningful for one is either invisible or catastrophic for
the other, and it can drive a rate negative. Smoothing ``log(rate)`` instead
makes ``eps`` mean *"a factor of e^eps"* uniformly and keeps every rate
positive by construction. :func:`spo_plus_subgradient` is the one thing that
cannot live in log space — it is derived from the objective being *linear* in
the perturbed parameter — and it says so rather than silently returning
something wrong.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from opsworld.world import (
    Capability,
    Evaluation,
    Sensitivity,
    SensitivityKind,
)

from .data import Instance
from .decision import Decision
from .dynamic_model import ReplenishmentParams, simulate_replenishment
from .milp_model import MilpWorldModel, available_solver
from .static_model import StaticWorldModel

__all__ = [
    "RATE_FAMILIES",
    "PlanningRates",
    "PlanningMilp",
    "InfeasibleArgmin",
    "solution_key",
    "honest_instance",
    "StaticTrueCost",
    "SupplyDynamics",
    "DynamicSupplyCost",
    "PerturbedGradient",
    "perturbed_gradient",
    "spo_plus_subgradient",
    "epsilon_curve",
    "descent_check",
    "TuningResult",
    "tune_planning_rates",
    "PlanningWorld",
]

PROVENANCE = "perturbed-milp (T0 argmin, Gaussian-smoothed)"

#: Instance parameters that enter the objective **linearly through a rate**.
#: Restricted on purpose: for each of these the per-solution activity
#: ``d(objective)/d(rate)`` is available in closed form (see
#: :meth:`PlanningRates.activity`), which is what SPO+ needs and what makes the
#: gradient readable as "cost per unit of this rate".
RATE_FAMILIES: tuple[str, ...] = (
    "storage_cost_rate",
    "picking_cost_rate",
    "access_cost_rate",
)


class InfeasibleArgmin(RuntimeError):
    """The MILP had no feasible solution at these planning parameters.

    Cannot happen from perturbing rates alone — they enter only the objective,
    never the feasible set — so this signals a solver failure, not a modelling
    one, and is not swallowed.
    """


# ----------------------------------------------------------------------
# theta: which planning parameters are being tuned
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PlanningRates:
    """The parameterisation ``theta -> instance``.

    ``entries`` names the ``(family, warehouse)`` pairs under control. In log
    space (the default) ``theta[k] = log(rate_k)``; see the module docstring
    for why.
    """

    entries: tuple[tuple[str, Any], ...]
    log_space: bool = True

    def __post_init__(self) -> None:
        for family, _ in self.entries:
            if family not in RATE_FAMILIES:
                raise ValueError(
                    f"{family!r} is not a rate family; the activity "
                    f"d(objective)/d(rate) is only defined for {RATE_FAMILIES}."
                )

    # -- construction --------------------------------------------------
    @classmethod
    def all_rates(
        cls,
        instance: Instance,
        families: Sequence[str] = RATE_FAMILIES,
        *,
        log_space: bool = True,
    ) -> PlanningRates:
        """Every rate of ``families`` over every decision warehouse."""
        return cls(
            entries=tuple(
                (family, warehouse)
                for family in families
                for warehouse in instance.L_sys
            ),
            log_space=log_space,
        )

    @classmethod
    def storage_only(
        cls, instance: Instance, *, log_space: bool = True
    ) -> PlanningRates:
        """Just the storage rates — the family Phase 1 found a placeholder in."""
        return cls.all_rates(instance, ("storage_cost_rate",), log_space=log_space)

    # -- the map itself ------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> list[str]:
        prefix = "log " if self.log_space else ""
        return [f"{prefix}{family}[{key}]" for family, key in self.entries]

    def baseline(self, instance: Instance) -> np.ndarray:
        """``theta`` of the instance as given."""
        rates = np.array(
            [float(instance.param(family, key)) for family, key in self.entries]
        )
        return np.log(rates) if self.log_space else rates

    def rates(self, theta: np.ndarray) -> np.ndarray:
        """``theta`` back in rate units."""
        theta = np.asarray(theta, dtype=float)
        return np.exp(theta) if self.log_space else theta

    def instance_with(self, base: Instance, theta: np.ndarray) -> Instance:
        """``base`` with the controlled rates replaced by ``theta``.

        Only the touched rate tables are copied; the rest of the data — the
        208-entry relocation-cost table included — is shared, because this runs
        once per Monte-Carlo draw.
        """
        values = self.rates(theta)
        data = dict(base.data)
        touched: dict[str, dict[Any, float]] = {}
        for (family, key), value in zip(self.entries, values):
            table = touched.get(family)
            if table is None:
                table = touched[family] = dict(data[family])
            table[key] = float(value)
        data.update(touched)
        return Instance(data)

    # -- the Jacobian handle -------------------------------------------
    def activity(self, instance: Instance, decision: Decision) -> np.ndarray:
        """``d(MILP objective) / d(rate_k)`` at a fixed solution.

        Read straight off the cost definitions (2)-(4) of the source model, so
        that ``objective(theta, x) == rates(theta) @ activity(x) + reloc(x)``
        — an identity the test suite checks against
        :meth:`~intralogistics.milp_model.MilpWorldModel.evaluate`.
        """
        inst = instance
        out = np.zeros(len(self.entries))
        n_deliveries = inst.scalar("number_of_deliveries")
        for index, (family, l) in enumerate(self.entries):
            if family == "storage_cost_rate":
                out[index] = sum(
                    decision.get("y_carriers", i, l, b) * inst.param("carrier_area", b)
                    for i in inst.I
                    for b in inst.B
                )
            elif family == "picking_cost_rate":
                out[index] = sum(
                    decision.get("x_pick", i, r, l)
                    * inst.param("demand_item_resource", i, r)
                    for i in inst.I
                    for r in inst.R
                )
            else:  # access_cost_rate
                out[index] = n_deliveries * sum(
                    decision.get("x_product_from_warehouse", p, l) for p in inst.P
                )
        return out


# ----------------------------------------------------------------------
# the argmin oracle
# ----------------------------------------------------------------------
def solution_key(decision: Decision) -> tuple:
    """Canonical hashable form of a solution — the cache key for the loss.

    The argmin is piecewise constant, so most Monte-Carlo draws return a
    solution that has already been scored. That is what makes an expensive
    honest loss (a simulation) affordable inside the estimator.
    """
    return tuple(
        (name, tuple(sorted((k, round(v, 9)) for k, v in table.items() if abs(v) > 1e-9)))
        for name, table in decision.as_dict().items()
    )


class PlanningMilp:
    """``argmin`` over the MILP as a function of the planning parameters.

    Every call is an *exact* mixed-integer solve — the forward pass is never
    relaxed. Only the gradient is smoothed.
    """

    def __init__(
        self,
        base: Instance,
        rates: PlanningRates,
        *,
        solver: str | None = None,
    ) -> None:
        self.base = base
        self.rates = rates
        self.solver = solver or available_solver()
        if self.solver is None:
            raise RuntimeError(
                "No MILP solver available; install one, e.g. `uv add highspy`."
            )
        self.n_solves = 0
        self._cache: dict[tuple, Decision] = {}

    def argmin(self, theta: np.ndarray) -> Decision:
        instance = self.rates.instance_with(self.base, theta)
        model = MilpWorldModel(instance)
        decision, objective = model.solve(solver=self.solver)
        self.n_solves += 1
        if decision is None or objective is None:
            raise InfeasibleArgmin(
                f"solver reported {model.termination!r}; rates enter only the "
                "objective, so the feasible set cannot have changed."
            )
        return decision

    def planning_objective(self, theta: np.ndarray, decision: Decision) -> float:
        """``c(theta)^T x`` — what the MILP believes the solution costs."""
        instance = self.rates.instance_with(self.base, theta)
        return float(MilpWorldModel(instance).evaluate(instance, decision).objective)


# ----------------------------------------------------------------------
# the outer loss: what the configuration is actually worth
# ----------------------------------------------------------------------
def honest_instance(
    base: Instance,
    *,
    point_of_use_rate: float = 3.0,
    placeholder_above: float = 1e4,
) -> Instance:
    """``base`` with the placeholder storage rates replaced by real ones.

    Phase 1 established that the ``1e5`` rates at the points of use are not
    costs: they are a deterrent against stocking there, aimed at a choice the
    model does not actually offer, and they contribute 99.994% of the objective
    without moving a single decision. A loss function that keeps them measures
    the placeholder, not the plant — so evaluation gets honest rates while
    ``theta`` keeps the planning ones. That gap *is* the object of study here.
    """
    table = base.data["storage_cost_rate"]
    data = dict(base.data)
    data["storage_cost_rate"] = {
        l: (point_of_use_rate if float(v) >= placeholder_above else float(v))
        for l, v in table.items()
    }
    return Instance(data)


class StaticTrueCost:
    """Honest cost under the *same* time-averaged algebra (T0).

    A useful control rather than the main event: because it lives in the same
    linear family as the planning objective, the answer is known in advance —
    the best planning rates are the honest ones. Its job is to confirm the
    estimator recovers a known optimum, and to be the one loss for which SPO+
    is even defined.
    """

    name = "static_true_cost"

    def __init__(self, instance: Instance, *, penalty: float = 1e6) -> None:
        self.inst = instance
        self.penalty = penalty
        self._model = StaticWorldModel()

    def __call__(self, decision: Decision) -> float:
        evaluation = self._model.evaluate(self.inst, decision)
        return evaluation.objective + self.penalty * evaluation.total_violation()

    def cost_vector_theta(self, rates: PlanningRates) -> np.ndarray:
        """The honest rates as a ``theta`` — the target SPO+ regresses towards."""
        return rates.baseline(self.inst)


@dataclass(frozen=True)
class SupplyDynamics:
    """The T1 parameters the static model has no place to put.

    Lead time and demand variability appear nowhere in the source MILP; they
    are the reason safety stock and Kanban card counts exist at all (§3.2).
    Phase 3 measured how the time average degrades as ``demand_cv`` rises, so
    these are the knobs that decide how far the planning model can be trusted.
    """

    lead_time: int = 3
    demand_cv: float = 0.4
    service_factor: float = 1.65
    horizon: int = 1500
    warmup: int = 200
    stockout_penalty: float = 0.0
    seed: int = 0


@dataclass
class _Flow:
    """What one ``(item, warehouse)`` slice of a configuration looks like."""

    capacity: float        # bin size Q, from the chosen load carrier
    area: float            # footprint of one such carrier
    throughput: float      # units per period leaving this warehouse -- eq. (17)
    moves: float           # replenishment moves the MILP planned
    per_move_cost: float   # realised cost of one inbound move on the chosen route
    static_storage: float  # what the static model charges, as a fallback


def _slice_seed(item: Any, warehouse: Any) -> int:
    """Demand stream identifier for one ``(item, warehouse)`` slice.

    Derived from the slice's **identity**, never from its position in the
    configuration. Two reasons, and the first is a validity threat rather than
    a nicety:

    * a seed taken from an enumeration index changes for every slice downstream
      of any structural change, so two configurations would be compared on
      different random draws. A tuner could then "improve" the loss by finding
      a lucky realisation;
    * holding the stream fixed per slice makes every comparison a **common
      random numbers** comparison, which cancels most of the simulation noise
      out of the loss *differences* — and differences are all the perturbed
      estimator ever looks at.

    ``crc32`` rather than ``hash``: string hashing is salted per process, and
    these numbers end up in the plan document.
    """
    return zlib.crc32(repr((item, warehouse)).encode()) % 100_000


def _flows(instance: Instance, decision: Decision) -> dict[tuple[Any, Any], _Flow]:
    """Decompose a configuration into per-``(item, warehouse)`` supply slices.

    ``throughput`` is the right-hand side of constraint (17) — picking demand
    plus onward relocation — so the slice handed to the simulator is the same
    one Phase 3 validated, not a new modelling assumption.
    """
    inst = instance
    flows: dict[tuple[Any, Any], _Flow] = {}
    for i in inst.I:
        for l in inst.L_sys:
            if decision.get("x_store", i, l) < 0.5:
                continue
            carriers = [b for b in inst.B if decision.get("x_carrier", i, l, b) > 0.5]
            if not carriers:
                continue
            b = carriers[0]

            picked = sum(
                decision.get("x_pick", i, r, l)
                * inst.param("demand_item_resource", i, r)
                for r in inst.R
            )
            onward = sum(
                decision.get("y_reloc_moves", i, r, l, l3, b1, b2)
                * inst.param("carrier_capacity", i, b2)
                for r in inst.R
                for start, l3 in inst.U_edges
                if start == l
                for b1 in inst.B
                for b2 in inst.B
            )
            inbound = [
                (r, l1, b1, b2)
                for r in inst.R
                for l1, target in inst.U_edges
                if target == l
                for b1 in inst.B
                for b2 in inst.B
            ]
            moves = sum(
                decision.get("y_reloc_moves", i, r, l1, l, b1, b2)
                for r, l1, b1, b2 in inbound
            )
            spend = sum(
                decision.get("y_reloc_moves", i, r, l1, l, b1, b2)
                * inst.param("relocation_cost", i, l1, l, b1, b2)
                for r, l1, b1, b2 in inbound
            )
            flows[(i, l)] = _Flow(
                capacity=float(inst.param("carrier_capacity", i, b)),
                area=float(inst.param("carrier_area", b)),
                throughput=float(picked + onward),
                moves=float(moves),
                per_move_cost=float(spend / moves) if moves > 0 else 0.0,
                static_storage=float(
                    inst.param("storage_cost_rate", l)
                    * decision.get("y_carriers", i, l, b)
                    * inst.param("carrier_area", b)
                ),
            )
    return flows


class DynamicSupplyCost:
    """Score a MILP configuration by **simulating** it (T1).

    The split is deliberate and is the whole reason this loss is interesting:

    * the MILP decides the *structure* — which warehouse, which carrier, which
      route. That is kept exactly as solved;
    * the simulation decides the *frequency and the level* — how often a bin
      actually has to be moved, how much stock that leaves standing, whether
      the line ever starves. Those are outputs here and inputs there.

    So storage and relocation are re-priced from
    :func:`~intralogistics.dynamic_model.simulate_replenishment`, while picking
    and access costs — which are pure structure, with no time behaviour — are
    taken from the static algebra unchanged.

    Nothing about this is expressible inside the MILP, which is the point: were
    it expressible, tuning planning rates would be pointless and one would just
    solve the right model. This is the non-degenerate case that
    :class:`StaticTrueCost` is not.
    """

    name = "dynamic_supply_cost"

    def __init__(
        self,
        instance: Instance,
        dynamics: SupplyDynamics = SupplyDynamics(),
        *,
        penalty: float = 1e6,
    ) -> None:
        self.inst = instance
        self.dynamics = dynamics
        self.penalty = penalty
        self._static = StaticWorldModel()

    def __call__(self, decision: Decision) -> float:
        return self.report(decision)["objective"]

    def report(self, decision: Decision) -> dict[str, float]:
        """Full breakdown — the same keys as the static KPIs, plus service."""
        inst = self.inst
        static = self._static.evaluate(inst, decision)
        dyn = self.dynamics

        cost_storage = 0.0
        cost_reloc = 0.0
        cost_stockout = 0.0
        worst_service = 1.0
        peak_area = 0.0

        for (i, l), flow in sorted(_flows(inst, decision).items()):
            if flow.throughput <= 0.0:
                # Stored but shipping nothing: there is no sawtooth to simulate,
                # so the static charge stands.
                cost_storage += flow.static_storage
                continue

            params = ReplenishmentParams(
                demand_rate=flow.throughput,
                demand_cv=dyn.demand_cv,
                lead_time=dyn.lead_time,
                service_factor=dyn.service_factor,
                horizon=dyn.horizon,
                warmup=dyn.warmup,
                seed=dyn.seed + _slice_seed(i, l),
            )
            outcome = simulate_replenishment(flow.capacity, params)

            carriers = math.ceil(outcome.avg_inventory / flow.capacity)
            cost_storage += (
                float(inst.param("storage_cost_rate", l)) * carriers * flow.area
            )
            cost_reloc += flow.per_move_cost * outcome.moves_per_period
            cost_stockout += dyn.stockout_penalty * outcome.stockout_units_per_period
            worst_service = min(worst_service, outcome.service_level)
            peak_area += math.ceil(outcome.max_inventory / flow.capacity) * flow.area

        objective = (
            cost_storage
            + cost_reloc
            + cost_stockout
            + static.kpis["cost_pick_var"]
            + static.kpis["cost_pick_fix"]
            + self.penalty * static.total_violation()
        )
        return {
            "objective": objective,
            "cost_storage": cost_storage,
            "cost_reloc": cost_reloc,
            "cost_stockout": cost_stockout,
            "cost_pick_var": static.kpis["cost_pick_var"],
            "cost_pick_fix": static.kpis["cost_pick_fix"],
            "service_level": worst_service,
            "peak_area": peak_area,
        }


# ----------------------------------------------------------------------
# the estimator
# ----------------------------------------------------------------------
@dataclass
class PerturbedGradient:
    """One Monte-Carlo estimate of ``grad L_eps(theta)``."""

    theta: np.ndarray
    names: list[str]
    epsilon: float
    gradient: np.ndarray
    std_error: np.ndarray
    smoothed_loss: float
    exact_loss: float
    n_samples: int
    distinct_solutions: int
    distinct_losses: int

    @property
    def gap(self) -> float:
        """``|L_eps - L|`` — how far smoothing moved the *value*."""
        return abs(self.smoothed_loss - self.exact_loss)

    @property
    def has_signal(self) -> bool:
        """Whether any perturbation actually changed what the decision is worth.

        Keyed on ``distinct_losses``, not on ``distinct_solutions``: the MILP
        is degenerate and a perturbation routinely returns a *structurally*
        different solution of identical value — a different tie broken a
        different way. Those draws carry no information, and counting them
        would make an estimator of pure tie-breaking look informative.

        When the loss is constant over the whole ball of radius ``eps``, the
        estimator is identically zero and there is no direction to report. That
        is not a failure of the method: the objective really is flat there, and
        the only remedy is a larger ``eps``, which costs fidelity (see
        :func:`epsilon_curve`).

        This is the counterpart of
        :func:`~intralogistics.differentiable.saturation_fraction`: a cheap
        precondition to check *before* acting on the gradient. It is
        **necessary but not sufficient** — a loss that varies because ``theta``
        sits *exactly on* a jump also passes, and that case is diagnosed by the
        ``scaled_grad_norm`` column of :func:`epsilon_curve`.
        """
        return self.distinct_losses > 1 and bool(np.any(self.gradient))

    @property
    def scaled_gradient_norm(self) -> float:
        """``eps * ||gradient||``.

        Constant across ``eps`` means the variation comes from a jump *at*
        ``theta`` rather than from a boundary near it — see
        :func:`epsilon_curve`.
        """
        return self.epsilon * float(np.linalg.norm(self.gradient))

    @property
    def relative_noise(self) -> float:
        """``||standard error|| / ||gradient||``.

        Above ~1 the sampling error is as large as the signal, and more samples
        are the only cure. Infinite when the gradient is exactly zero, i.e.
        when nothing was in reach.

        Note this is *not* the quantity that separates the usable ``eps`` here
        — measured, it stays around 0.6–0.9 across two decades of ``eps``,
        because widening the ball adds both signal and spread. Reach
        (:attr:`has_signal`) and the scaling test
        (:attr:`scaled_gradient_norm`) are what discriminate.
        """
        norm = float(np.linalg.norm(self.gradient))
        return float(np.linalg.norm(self.std_error) / norm) if norm > 0 else math.inf

    def sensitivity(self) -> Sensitivity:
        """As an ``opsworld`` first-order object — ``PATHWISE``, not ``DUAL``.

        The type is the claim: unlike
        :meth:`~intralogistics.world.MilpWorld.duals`, this one may be stepped
        along, and :meth:`~opsworld.world.Sensitivity.descent_direction`
        accepts it.
        """
        return Sensitivity(
            kind=SensitivityKind.PATHWISE,
            names=list(self.names),
            values=[float(v) for v in self.gradient],
        )

    def as_text(self) -> str:
        lines = [
            f"perturbed argmin gradient  eps={self.epsilon:g}  n={self.n_samples}  "
            f"distinct solutions={self.distinct_solutions} "
            f"(distinct values={self.distinct_losses})",
            f"  L(theta)={self.exact_loss:.4f}  "
            f"L_eps(theta)={self.smoothed_loss:.4f}  gap={self.gap:.4f}",
            f"  relative noise={self.relative_noise:.3f}"
            + ("   <-- estimate is noise, not direction" if self.relative_noise > 1 else ""),
        ]
        return "\n".join([*lines, self.sensitivity().as_text()])


def perturbed_gradient(
    milp: PlanningMilp,
    loss: Callable[[Decision], float],
    theta: np.ndarray,
    *,
    epsilon: float,
    n_samples: int = 32,
    seed: int = 0,
    loss_cache: dict[tuple, float] | None = None,
) -> PerturbedGradient:
    """Gaussian-smoothed gradient of ``loss(argmin(theta))``.

    Antithetic pairs ``(Z, -Z)`` are used rather than the plain estimator
    ``(1/eps) E[f Z]``. Both are unbiased for ``grad L_eps``; the antithetic
    form cancels the term proportional to the *mean* loss, which is the term
    that dominates here — the loss varies by a few percent across draws while
    its level is large, and the plain estimator spends all its samples fighting
    that offset.

    ``n_samples`` is rounded down to an even number: one solve per half-pair.
    """
    theta = np.asarray(theta, dtype=float)
    rng = np.random.default_rng(seed)
    cache = loss_cache if loss_cache is not None else {}

    def scored(candidate: np.ndarray) -> float:
        decision = milp.argmin(candidate)
        key = solution_key(decision)
        if key not in cache:
            cache[key] = float(loss(decision))
        return cache[key]

    exact_loss = scored(theta)
    n_pairs = max(1, n_samples // 2)

    estimates = np.empty((n_pairs, theta.size))
    values: list[float] = []
    seen: set[tuple] = set()
    for pair in range(n_pairs):
        z = rng.standard_normal(theta.size)
        forward = milp.argmin(theta + epsilon * z)
        backward = milp.argmin(theta - epsilon * z)
        for decision in (forward, backward):
            key = solution_key(decision)
            seen.add(key)
            if key not in cache:
                cache[key] = float(loss(decision))
            values.append(cache[key])
        estimates[pair] = 0.5 * (values[-2] - values[-1]) / epsilon * z

    gradient = estimates.mean(axis=0)
    std_error = (
        estimates.std(axis=0, ddof=1) / math.sqrt(n_pairs)
        if n_pairs > 1
        else np.full(theta.size, math.inf)
    )
    return PerturbedGradient(
        theta=theta,
        names=milp.rates.names(),
        epsilon=epsilon,
        gradient=gradient,
        std_error=std_error,
        smoothed_loss=float(np.mean(values)),
        exact_loss=exact_loss,
        n_samples=2 * n_pairs,
        distinct_solutions=len(seen),
        distinct_losses=len({round(value, 9) for value in values}),
    )


def spo_plus_subgradient(
    milp: PlanningMilp,
    theta: np.ndarray,
    true_theta: np.ndarray,
) -> np.ndarray:
    """SPO+ subgradient (Elmachtoub & Grigas, 2022) — the deterministic rival.

    Two solves instead of ``n_samples``, and no variance at all::

        grad = 2 * [ activity(x*(true)) - activity(x*(2*theta - true)) ]

    The catch is structural, and it is worth stating because it is what makes
    the perturbed estimator the general tool: SPO+ is a convex surrogate for
    *regret against a known linear cost vector*. It needs the evaluation
    objective to live in the **same linear family** as the planning objective.
    That holds for :class:`StaticTrueCost` and fails for
    :class:`DynamicSupplyCost` — a simulation has no cost vector — so SPO+
    simply cannot be pointed at the loss that matters here.

    Raises when the parameterisation is in log space: the derivation assumes
    ``c^T x`` is linear in the perturbed parameter, and ``exp`` is not.
    """
    if milp.rates.log_space:
        raise ValueError(
            "SPO+ requires the objective to be linear in the parameter, so it "
            "needs a rate-space PlanningRates (log_space=False). The perturbed "
            "estimator has no such requirement."
        )
    theta = np.asarray(theta, dtype=float)
    true_theta = np.asarray(true_theta, dtype=float)

    optimal = milp.argmin(true_theta)
    spo_point = milp.argmin(2.0 * theta - true_theta)
    reference = milp.rates.instance_with(milp.base, true_theta)
    return 2.0 * (
        milp.rates.activity(reference, optimal)
        - milp.rates.activity(reference, spo_point)
    )


# ----------------------------------------------------------------------
# the RQ1 probe for this relaxation
# ----------------------------------------------------------------------
def epsilon_curve(
    milp: PlanningMilp,
    loss: Callable[[Decision], float],
    theta: np.ndarray,
    epsilons: Sequence[float],
    *,
    n_samples: int = 32,
    seed: int = 0,
) -> dict:
    """Sweep the smoothing scale — the counterpart of ``fidelity_curve``.

    Keys are chosen to line up with
    :func:`gradientshop.ddes.dispatch.fidelity_curve` and
    :func:`intralogistics.differentiable.fidelity_curve` so the three
    relaxations can be tabulated side by side: ``gap`` is again
    ``|relaxed value - exact value|``, and ``grad_cosine`` is again "can this
    gradient be trusted".

    What ``grad_cosine`` measures differs, because the failure mode differs.
    There is no straight-through reference to compare against — the forward
    pass is already exact — so the estimate is compared with **an independent
    replicate of itself**. Two draws of a real direction agree; two draws of
    noise are orthogonal in expectation. It needs no ground truth, which is
    precisely the situation a practitioner is in.

    ``scaled_grad_norm`` (``eps * ||grad||``) separates the two ways the loss
    can vary under perturbation, and the distinction decides whether the
    gradient means anything:

    * **flat across ``eps``** — ``theta`` lies *on* a jump of the loss. Mixed
      integer programs are routinely degenerate, so a perturbation merely
      breaks a tie between optima the planner already considers equal, and the
      smoothed gradient diverges as ``1/eps`` while pointing nowhere. Measured
      on the source's minimal instance: identical solution set and identical
      loss values over a 400-fold range of ``eps``;
    * **growing with ``eps``** — genuine boundaries are being discovered, more
      of them as the ball widens. That is the regime the estimator is for.
    """
    theta = np.asarray(theta, dtype=float)
    cache: dict[tuple, float] = {}
    out: dict[str, list[float]] = {
        "eps": [],
        "smoothed_loss": [],
        "gap": [],
        "grad_norm": [],
        "scaled_grad_norm": [],
        "grad_cosine": [],
        "relative_noise": [],
        "distinct_solutions": [],
        "distinct_losses": [],
    }
    exact = math.nan
    for epsilon in epsilons:
        first = perturbed_gradient(
            milp, loss, theta, epsilon=epsilon, n_samples=n_samples,
            seed=seed, loss_cache=cache,
        )
        second = perturbed_gradient(
            milp, loss, theta, epsilon=epsilon, n_samples=n_samples,
            seed=seed + 9973, loss_cache=cache,
        )
        exact = first.exact_loss
        norms = np.linalg.norm(first.gradient) * np.linalg.norm(second.gradient)
        cosine = (
            float(first.gradient @ second.gradient / norms) if norms > 0 else 0.0
        )
        out["eps"].append(epsilon)
        out["smoothed_loss"].append(0.5 * (first.smoothed_loss + second.smoothed_loss))
        out["gap"].append(0.5 * (first.gap + second.gap))
        out["grad_norm"].append(float(np.linalg.norm(first.gradient)))
        out["scaled_grad_norm"].append(first.scaled_gradient_norm)
        out["grad_cosine"].append(cosine)
        out["relative_noise"].append(first.relative_noise)
        out["distinct_solutions"].append(
            max(first.distinct_solutions, second.distinct_solutions)
        )
        out["distinct_losses"].append(
            max(first.distinct_losses, second.distinct_losses)
        )
    out["exact_loss"] = exact  # type: ignore[assignment]
    return out


def descent_check(
    milp: PlanningMilp,
    loss: Callable[[Decision], float],
    theta: np.ndarray,
    gradient: np.ndarray,
    step_sizes: Sequence[float],
) -> dict:
    """Does stepping along ``-gradient`` lower the **true**, unsmoothed loss?

    The only acceptance test for a piecewise-constant objective. A cosine
    against a finite difference is meaningless when the finite difference is
    zero almost everywhere, so the question has to be asked operationally.

    Delegates to :func:`opsworld.diagnostics.descent_check`, which asks exactly
    this of the relaxed ``argsort`` in :mod:`gradientshop.ddes.dispatch` too.
    The two relaxations differ in form and are identical in what makes their
    gradients hard to trust.
    """
    from opsworld.diagnostics import descent_check as shared

    report = shared(
        lambda point: loss(milp.argmin(np.asarray(point, dtype=float))),
        np.asarray(theta, dtype=float).tolist(),
        np.asarray(gradient, dtype=float).tolist(),
        step_sizes,
    )
    return {
        "step": report.steps,
        "loss": report.values,
        "improvement": [report.start - value for value in report.values],
        "start_loss": report.start,  # type: ignore[dict-item]
        "best_improvement": report.improvement,  # type: ignore[dict-item]
    }


@dataclass
class TuningResult:
    """Outcome of :func:`tune_planning_rates`."""

    theta: np.ndarray
    names: list[str]
    rates: np.ndarray
    start_loss: float
    best_loss: float
    trace: list[float]
    accepted: int
    widened: int
    n_solves: int

    @property
    def improvement(self) -> float:
        return self.start_loss - self.best_loss

    @property
    def improvement_pct(self) -> float:
        return 100.0 * self.improvement / self.start_loss if self.start_loss else 0.0

    def as_text(self) -> str:
        lines = [
            f"planning rates tuned on {len(self.names)} parameters: "
            f"{self.start_loss:.4f} -> {self.best_loss:.4f} "
            f"({self.improvement_pct:+.2f}%)",
            f"  accepted {self.accepted} steps, widened eps {self.widened} times, "
            f"{self.n_solves} exact solves",
        ]
        lines += [
            f"    {name:<32} {rate:12.4f}"
            for name, rate in zip(self.names, self.rates)
        ]
        return "\n".join(lines)


def tune_planning_rates(
    milp: PlanningMilp,
    loss: Callable[[Decision], float],
    theta: np.ndarray,
    *,
    epsilon: float = 0.8,
    steps: int = 8,
    n_samples: int = 24,
    step_sizes: Sequence[float] = (2.0, 0.8, 0.3),
    epsilon_max: float = 4.0,
    seed: int = 0,
) -> TuningResult:
    """Descend the true loss using perturbed gradients, with a line search.

    Two departures from plain gradient descent, both forced by the loss being
    piecewise constant rather than merely non-convex:

    * **steps are accepted only if they improve.** The candidate costs one
      exact solve, which is cheap next to the ``n_samples`` the gradient
      already spent, and a step that lands in a worse cell has no small
      version that lands in a better one;
    * **``eps`` widens when there is nothing to see.** A zero gradient here
      does not mean a stationary point — it means no decision boundary lies
      within ``eps``. Shrinking the step is the wrong response; enlarging the
      search radius is the right one, at a known cost in fidelity.
    """
    theta = np.asarray(theta, dtype=float).copy()
    cache: dict[tuple, float] = {}

    def evaluate(candidate: np.ndarray) -> float:
        decision = milp.argmin(candidate)
        key = solution_key(decision)
        if key not in cache:
            cache[key] = float(loss(decision))
        return cache[key]

    start = evaluate(theta)
    best = start
    trace = [start]
    accepted = 0
    widened = 0
    solves_before = milp.n_solves

    for step in range(steps):
        estimate = perturbed_gradient(
            milp, loss, theta, epsilon=epsilon, n_samples=n_samples,
            seed=seed + 31 * step, loss_cache=cache,
        )
        if not estimate.has_signal:
            epsilon = min(epsilon * 2.0, epsilon_max)
            widened += 1
            trace.append(best)
            continue

        direction = estimate.gradient / float(np.linalg.norm(estimate.gradient))
        moved = False
        for size in step_sizes:
            candidate = theta - size * direction
            value = evaluate(candidate)
            if value < best - 1e-9:
                theta, best, moved = candidate, value, True
                accepted += 1
                break
        if not moved:
            epsilon = min(epsilon * 1.5, epsilon_max)
            widened += 1
        trace.append(best)

    return TuningResult(
        theta=theta,
        names=milp.rates.names(),
        rates=milp.rates.rates(theta),
        start_loss=start,
        best_loss=best,
        trace=trace,
        accepted=accepted,
        widened=widened,
        n_solves=milp.n_solves - solves_before,
    )


# ----------------------------------------------------------------------
# as an opsworld world model
# ----------------------------------------------------------------------
class PlanningWorld:
    """Planning parameters as the decision — ``C0 + C2a`` on top of a ``C3`` model.

    The capability upgrade promised by Phase 5.3, stated in the type system:
    :class:`~intralogistics.world.MilpWorld` reports
    ``{EVALUATE, DUALS, ALGEBRAIC}`` and cannot be handed to a gradient solver;
    this reports ``{EVALUATE, PATHWISE_GRADIENT}`` over a different decision
    space — the rates fed *to* that model rather than the configuration coming
    *out* of it — and can.
    """

    def __init__(
        self,
        milp: PlanningMilp,
        loss: Callable[[Decision], float],
        *,
        epsilon: float = 0.5,
        n_samples: int = 32,
        seed: int = 0,
    ) -> None:
        self.milp = milp
        self.loss = loss
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.seed = seed
        self._cache: dict[tuple, float] = {}

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE, Capability.PATHWISE_GRADIENT}

    def decision_space(self):  # noqa: ANN201 - a VectorSpace over theta
        from opsworld.problem import VectorSpace

        return VectorSpace(len(self.milp.rates), names=self.milp.rates.names())

    def evaluate(self, decision: Sequence[float]) -> Evaluation:
        theta = np.asarray(decision, dtype=float)
        solution = self.milp.argmin(theta)
        key = solution_key(solution)
        if key not in self._cache:
            self._cache[key] = float(self.loss(solution))
        rates = self.milp.rates.rates(theta)
        return Evaluation(
            objective=self._cache[key],
            kpis={
                "planning_objective": self.milp.planning_objective(theta, solution),
                **{name: float(value)
                   for name, value in zip(self.milp.rates.names(), rates)},
            },
            violations={},
            provenance=PROVENANCE,
        )

    def gradient(self, decision: Sequence[float]) -> Sensitivity:
        return self.estimate(decision).sensitivity()

    def estimate(self, decision: Sequence[float]) -> PerturbedGradient:
        return perturbed_gradient(
            self.milp,
            self.loss,
            np.asarray(decision, dtype=float),
            epsilon=self.epsilon,
            n_samples=self.n_samples,
            seed=self.seed,
            loss_cache=self._cache,
        )

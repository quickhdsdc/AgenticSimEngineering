"""Is this gradient usable? — the pre-flight check, shared across domains.

Every relaxation in this repository approximates an objective that is
**piecewise constant** in the parameter being differentiated:

* ``argsort`` dispatch — the exact schedule cost is constant until two priority
  scores cross, then jumps;
* a mixed-integer ``argmin`` — the solution sits on a vertex until a cost
  boundary is crossed, then jumps;

and for those a finite difference is not a reference at all: it is zero almost
everywhere and undefined on the jumps. Comparing a relaxed gradient against
another relaxed gradient is no better — two estimators sharing a Jacobian agree
with each other exactly when they are both about to vanish. Measured on the M3
dispatch instance: at ``tau = 0.01`` the cosine between the soft and
straight-through gradients is ``-1.000`` while a step along the soft gradient
*lowers* the exact cost by 0.49; at ``tau = 0.02`` on another instance the
cosine is ``+1.000`` while the same step makes the cost slightly worse.

So the question has to be asked operationally, and it is the same question in
both domains: **does stepping along the negative gradient reduce the true,
unrelaxed objective?** That is what this module answers. It needs nothing but a
way to score a point, so it applies to any relaxation, present or future.

The replenishment recursion of Phase 4 is the instructive exception: there the
sensitivity flows through a continuous order *quantity* rather than through the
discrete order *timing*, the exact cost really is differentiable a.e., and a
finite difference is available. That is precisely why its gradient recovers as
``tau -> 0`` while the other two do not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["DescentReport", "descent_check"]


@dataclass
class DescentReport:
    """What a trial step along ``-gradient`` actually did."""

    start: float
    steps: list[float]
    values: list[float]
    #: Step size that helped most, or ``None`` if none of them did.
    best_step: float | None
    best_value: float

    @property
    def improvement(self) -> float:
        return self.start - self.best_value

    @property
    def descends(self) -> bool:
        """Whether *any* trial step improved the true objective."""
        return self.best_step is not None

    def as_text(self) -> str:
        head = (
            f"descent check: start={self.start:.6f} "
            + (
                f"-> {self.best_value:.6f} at step {self.best_step:g} "
                f"({self.improvement:+.6f})"
                if self.descends
                else "-> no trial step improved it; do not act on this gradient"
            )
        )
        rows = [
            f"    step {step:<10g} {value:.6f}"
            for step, value in zip(self.steps, self.values)
        ]
        return "\n".join([head, *rows])


def descent_check(
    objective: Callable[[Sequence[float]], float],
    point: Sequence[float],
    gradient: Sequence[float],
    step_sizes: Sequence[float] = (0.02, 0.05, 0.1, 0.3, 1.0),
    *,
    normalise: bool = True,
) -> DescentReport:
    """Try steps along ``-gradient`` and report what the true objective did.

    ``objective`` must evaluate the **unrelaxed** quantity — the exact discrete
    schedule cost, the simulated cost of an integer solution — not the relaxed
    surrogate the gradient came from. Scoring the surrogate would only confirm
    that the surrogate is differentiable, which was never in doubt.

    ``normalise`` divides the direction by its norm, so ``step_sizes`` are
    distances in parameter space rather than multiples of a gradient magnitude
    that varies by orders of magnitude with the temperature. Pass ``False`` to
    keep the raw scaling, e.g. when comparing step rules.

    A zero gradient raises: there is no direction, and silently reporting "no
    improvement" would confuse *no information* with *no improvement available*
    — a distinction Phase 5.3 had to make sharply.
    """
    values_point = [float(x) for x in point]
    direction = [float(g) for g in gradient]
    if len(direction) != len(values_point):
        raise ValueError(
            f"gradient has {len(direction)} components for a point of "
            f"{len(values_point)}"
        )

    norm = sum(g * g for g in direction) ** 0.5
    if norm == 0.0:
        raise ValueError(
            "zero gradient: there is no direction to step along. That is not "
            "the same as a stationary point -- for a piecewise-constant "
            "objective it usually means nothing lies within reach."
        )
    if normalise:
        direction = [g / norm for g in direction]

    start = float(objective(values_point))
    steps: list[float] = []
    values: list[float] = []
    best_step: float | None = None
    best_value = start

    for size in step_sizes:
        candidate = [x - size * g for x, g in zip(values_point, direction)]
        value = float(objective(candidate))
        steps.append(float(size))
        values.append(value)
        if value < best_value - 1e-12:
            best_step, best_value = float(size), value

    return DescentReport(
        start=start,
        steps=steps,
        values=values,
        best_step=best_step,
        best_value=best_value,
    )

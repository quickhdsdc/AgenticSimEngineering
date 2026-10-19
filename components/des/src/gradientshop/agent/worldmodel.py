"""D2 -- The DDES as a world model with a gradient feedback channel (RQ2).

RQ2 asks whether a DDES forward *replay* can be embedded as a structured
reasoning step inside LLM inference, "grounding linguistic self-evaluation in
simulation-backed state trajectories". This is the coupling substrate.

A dispatch policy is a weight vector ``theta`` over the named dispatching rules
of :mod:`gradientshop.shopfloor.rules` (SPT, EDD, WSPT, ...). The world model
scores jobs with ``s = rule_features @ theta``, runs the differentiable schedule
(M3), and returns not just the scalar objective but the *gradient*
``d objective/d theta`` and a verbalisation of it. That verbalised gradient is
what grounds the agent's next proposal -- a literal step for a gradient agent,
injected context for an LLM (see :mod:`.agents` and :mod:`.tools`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..ddes.dispatch import (
    exact_dispatch_cost,
    schedule_completions,
    soft_sort,
    weighted_completion_time,
    weighted_tardiness,
)
from ..shopfloor.rules import Jobs, priority_matrix

__all__ = ["Observation", "DispatchWorldModel"]


@dataclass
class Observation:
    """Structured, verbalisable feedback returned by one DDES replay."""

    step: int
    objective: float                 # exact (discrete) schedule cost
    soft_objective: float            # relaxed cost actually differentiated
    theta: list[float]               # current policy weights (one per rule)
    gradient: list[float]            # d soft_objective / d theta
    feature_names: list[str]         # the dispatching-rule names theta weights
    objective_label: str = "objective"

    def suggested_direction(self) -> list[float]:
        """Descent direction on the policy weights (negated gradient)."""
        return [-g for g in self.gradient]

    def as_text(self) -> str:
        """Render the gradient as a structured observation for an LLM context --
        a natural-language, simulation-grounded critique of the current policy."""
        lines = [
            f"[DDES replay @ step {self.step}]",
            f"{self.objective_label} = {self.objective:.3f} "
            f"(relaxed {self.soft_objective:.3f})",
            "Sensitivity of the objective to each dispatching-rule weight "
            "(negative = increasing this weight helps):",
        ]
        ranked = sorted(zip(self.feature_names, self.theta, self.gradient),
                        key=lambda r: r[2])
        for name, th, g in ranked:
            verb = "LOWERS cost" if g < 0 else "RAISES cost"
            lines.append(f"  - {name:<10} weight={th:+.3f}  d/dw={g:+.4f}  "
                         f"-> increasing it {verb}")
        lines.append(f"Recommended: raise the weight on '{ranked[0][0]}' the most.")
        return "\n".join(lines)


class DispatchWorldModel:
    """Differentiable single-machine dispatch world model over a rule policy."""

    def __init__(self, processing: Tensor, weight: Tensor, due: Tensor,
                 *, tau: float = 0.1, objective: str = "wct",
                 rules: list[str] | None = None) -> None:
        if objective not in ("wct", "tardiness"):
            raise ValueError("objective must be 'wct' or 'tardiness'")
        self.processing = processing.to(torch.float64)
        self.weight = weight.to(torch.float64)
        self.due = due.to(torch.float64)
        self.tau = tau
        # 'wct' = sum w_j C_j (smooth, reliably optimisable by the relaxation);
        # 'tardiness' = sum w_j T_j (nonconvex -- a documented RQ1 failure mode).
        self.objective = objective
        jobs = Jobs(self.processing, self.weight, self.due)
        self.feature_names, self.features = priority_matrix(jobs, rules)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    @property
    def objective_label(self) -> str:
        return "weighted completion time" if self.objective == "wct" \
            else "weighted tardiness"

    def _cost(self, completion: Tensor) -> Tensor:
        if self.objective == "wct":
            return weighted_completion_time(completion, self.weight)
        return weighted_tardiness(completion, self.due, self.weight)

    def evaluate(self, theta: list[float] | Tensor, step: int = 0) -> Observation:
        """Forward DDES replay: cost + gradient of the policy ``theta``."""
        th = torch.as_tensor(theta, dtype=torch.float64).clone().requires_grad_(True)
        scores = self.features @ th
        completion = schedule_completions(self.processing, soft_sort(scores, self.tau))
        soft_cost = self._cost(completion)
        (grad,) = torch.autograd.grad(soft_cost, th)
        exact_due = self.due if self.objective == "tardiness" else None
        exact = exact_dispatch_cost(self.processing, scores.detach(),
                                    self.weight, due=exact_due).item()
        return Observation(
            step=step,
            objective=exact,
            soft_objective=soft_cost.item(),
            theta=[float(x) for x in th.detach()],
            gradient=[float(x) for x in grad],
            feature_names=self.feature_names,
            objective_label=self.objective_label,
        )

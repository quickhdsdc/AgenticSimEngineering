"""GradientShop as an ``opsworld`` world model.

A thin adapter over :class:`~gradientshop.agent.worldmodel.DispatchWorldModel`.
Adapter rather than direct conformance on purpose: the DDES core and its 27
tests are the frozen ``v0.1.0-ddes-core`` baseline, and the shared layer is
supposed to stay thin (``PLAN.md`` A6). Nothing in ``ddes/`` changes; only the
call signature is translated.

Capabilities: ``C0`` (evaluation) and ``C2a`` (pathwise gradient). Dispatch is
an *unconstrained* problem, so ``violations`` is always empty — which is the
point of scoring via :func:`~opsworld.world.penalised_objective`: with no
constraints the score is just the objective, so the same solver drives this
world model and the constrained intralogistics one without special-casing.
"""

from __future__ import annotations

from torch import Tensor

from opsworld.problem import VectorSpace
from opsworld.world import Capability, Evaluation, Sensitivity, SensitivityKind

from .agent.worldmodel import DispatchWorldModel, Observation

__all__ = ["DispatchWorld", "observation_to_evaluation"]


def observation_to_evaluation(observation: Observation, provenance: str) -> Evaluation:
    """Re-express a DDES replay as the framework's :class:`Evaluation`.

    ``objective`` is the *exact discrete* schedule cost; the relaxed cost that
    was actually differentiated is kept as a KPI, so the relaxation gap stays
    visible instead of being silently averaged away.
    """
    return Evaluation(
        objective=observation.objective,
        kpis={
            "soft_objective": observation.soft_objective,
            "relaxation_gap": observation.soft_objective - observation.objective,
        },
        violations={},  # single-machine dispatch is unconstrained
        sensitivity=Sensitivity(
            kind=SensitivityKind.PATHWISE,
            names=observation.feature_names,
            values=observation.gradient,
        ),
        provenance=provenance,
    )


class DispatchWorld:
    """Differentiable single-machine dispatch over a dispatching-rule policy."""

    def __init__(
        self,
        processing: Tensor,
        weight: Tensor,
        due: Tensor,
        *,
        tau: float = 0.1,
        objective: str = "wct",
        rules: list[str] | None = None,
        start: float = 0.05,
        step: float = 0.25,
    ) -> None:
        self._model = DispatchWorldModel(
            processing, weight, due, tau=tau, objective=objective, rules=rules
        )
        # NB: a non-zero start matters here -- all-equal scores are a symmetric
        # saddle of the relaxed objective with exactly zero gradient.
        self._space = VectorSpace(
            self._model.n_features,
            start=start,
            step=step,
            names=self._model.feature_names,
        )
        self._steps = 0

    @property
    def provenance(self) -> str:
        return (f"ddes/dispatch (relaxed, tau={self._model.tau}, "
                f"{self._model.objective})")

    def capabilities(self) -> set[Capability]:
        return {Capability.EVALUATE, Capability.PATHWISE_GRADIENT}

    def decision_space(self) -> VectorSpace:
        return self._space

    def evaluate(self, decision: list[float]) -> Evaluation:
        self._steps += 1
        observation = self._model.evaluate(decision, step=self._steps)
        return observation_to_evaluation(observation, self.provenance)

    def gradient(self, decision: list[float]) -> Sensitivity:
        sensitivity = self.evaluate(decision).sensitivity
        assert sensitivity is not None
        return sensitivity

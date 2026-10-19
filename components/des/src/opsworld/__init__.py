"""opsworld — a capability-graded framework for operations decision problems.

Every decision problem is the triple ``(decision space, world model, objective)``.
The **world model** states how the world responds to a decision; everything
else — MILP, gradient descent, reinforcement learning, an LLM agent — is
machinery for *searching* the decision space, not part of the problem.

Keeping those two apart (``PLAN.md`` Axiom 1) is what lets one world model
serve many solvers and one solver serve many domains. Which solvers a world
model admits is decided by the **capabilities** it exposes, not by taste:

===========  =============================================  ======================
``C0``       ``evaluate(decision) -> Evaluation``           heuristics, BO, LLM
``C1``       ``reset`` / ``step``                           RL, MPC
``C2a``      pathwise gradient  ``dJ/d decision``           gradient descent
``C2b``      duals  ``dJ/d right-hand side``                sensitivity analysis
``C3``       algebraic form                                 exact solvers
===========  =============================================  ======================

Two domains are implemented against it: :mod:`gradientshop` (differentiable
discrete-event shopfloor models) and :mod:`intralogistics` (multi-stage
material supply configuration).

This package is deliberately thin. It defines protocols, never domain content
(architecture constraint A6).
"""

from .problem import DecisionSpace, Instance, Scenario, VectorSpace
from .solver import (
    GradientDescentSolver,
    HillClimbSolver,
    RandomSearchSolver,
    Solver,
    SolverResult,
    check_capabilities,
)
from .validation import FidelityReport, compare_world_models, spearman
from .world import (
    AlgebraicWorldModel,
    Capability,
    DifferentiableWorldModel,
    Evaluation,
    SequentialWorldModel,
    Sensitivity,
    SensitivityKind,
    WorldModel,
    penalised_objective,
)

__all__ = [
    "AlgebraicWorldModel",
    "Capability",
    "DecisionSpace",
    "DifferentiableWorldModel",
    "Evaluation",
    "FidelityReport",
    "GradientDescentSolver",
    "HillClimbSolver",
    "Instance",
    "RandomSearchSolver",
    "Scenario",
    "SensitivityKind",
    "Sensitivity",
    "SequentialWorldModel",
    "Solver",
    "SolverResult",
    "VectorSpace",
    "WorldModel",
    "check_capabilities",
    "compare_world_models",
    "penalised_objective",
    "spearman",
]

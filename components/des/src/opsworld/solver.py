"""Solvers — pluggable search machinery over a world model.

Each solver declares the **capabilities** it needs (``PLAN.md`` A5). That
declaration is what makes the framework's central claim operational: *the
world model is the shared asset, solvers are plugins*. Whether a domain can
use reinforcement learning, gradient descent or an exact solver is not a
matter of taste — it is determined by what its world model exposes.

The two solvers here need only ``C0``, i.e. the ability to evaluate a decision.
They therefore run against **any** world model in the framework, which is
exactly the acceptance criterion for Phase 2.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from .world import Capability, Evaluation, WorldModel, penalised_objective

__all__ = ["SolverResult", "Solver", "RandomSearchSolver", "HillClimbSolver",
           "GradientDescentSolver", "BayesianOptimisationSolver",
           "ExactSolver", "check_capabilities"]


@dataclass
class SolverResult:
    """Best decision found, plus enough trace to see whether it worked."""

    decision: Any
    evaluation: Evaluation
    score: float
    initial_score: float
    evaluations_used: int
    history: list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.initial_score - self.score

    def summary(self) -> str:
        return (
            f"score {self.initial_score:.6f} -> {self.score:.6f} "
            f"(improved {self.improvement:.6f}) in {self.evaluations_used} evaluations"
        )


@runtime_checkable
class Solver(Protocol):
    requires: ClassVar[frozenset[Capability]]

    def solve(self, world: WorldModel, *, budget: int = 100) -> SolverResult:
        ...


def check_capabilities(solver: Any, world: WorldModel) -> None:
    """Raise if ``world`` cannot serve ``solver``. Fail loudly, not subtly."""
    missing = set(solver.requires) - set(world.capabilities())
    if missing:
        raise TypeError(
            f"{type(solver).__name__} requires {sorted(missing)}, which "
            f"{type(world).__name__} does not provide "
            f"(it offers {sorted(world.capabilities())})."
        )


class RandomSearchSolver:
    """Sample the decision space; keep the best. The honest baseline."""

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.EVALUATE})

    def __init__(self, *, seed: int = 0, penalty: float = 1e6) -> None:
        self.seed = seed
        self.penalty = penalty

    def solve(self, world: WorldModel, *, budget: int = 100) -> SolverResult:
        check_capabilities(self, world)
        rng = random.Random(self.seed)
        space = world.decision_space()

        best = space.initial()
        best_eval = world.evaluate(best)
        best_score = penalised_objective(best_eval, self.penalty)
        initial_score = best_score
        history = [best_score]

        for _ in range(max(0, budget - 1)):
            candidate = space.perturb(space.initial(), rng)
            evaluation = world.evaluate(candidate)
            score = penalised_objective(evaluation, self.penalty)
            if score < best_score:
                best, best_eval, best_score = candidate, evaluation, score
            history.append(best_score)

        return SolverResult(best, best_eval, best_score, initial_score,
                            budget, history)


class HillClimbSolver:
    """Perturb the incumbent; accept strict improvements.

    Needs nothing but ``evaluate`` and a neighbourhood move, so it is the
    cheapest possible check that a world model is genuinely usable by search
    rather than only by its original solver.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.EVALUATE})

    def __init__(self, *, seed: int = 0, penalty: float = 1e6) -> None:
        self.seed = seed
        self.penalty = penalty

    def solve(self, world: WorldModel, *, budget: int = 100) -> SolverResult:
        check_capabilities(self, world)
        rng = random.Random(self.seed)
        space = world.decision_space()

        incumbent = space.initial()
        incumbent_eval = world.evaluate(incumbent)
        incumbent_score = penalised_objective(incumbent_eval, self.penalty)
        initial_score = incumbent_score
        history = [incumbent_score]

        for _ in range(max(0, budget - 1)):
            candidate = space.perturb(incumbent, rng)
            evaluation = world.evaluate(candidate)
            score = penalised_objective(evaluation, self.penalty)
            if score < incumbent_score:
                incumbent, incumbent_eval, incumbent_score = (
                    candidate, evaluation, score
                )
            history.append(incumbent_score)

        return SolverResult(incumbent, incumbent_eval, incumbent_score,
                            initial_score, budget, history)


class BayesianOptimisationSolver:
    """Gaussian-process surrogate with expected improvement — ``C0`` only.

    The strongest derivative-free baseline worth comparing a gradient method
    against: random search shows what no modelling buys, hill climbing shows
    what local structure buys, and this shows what a *global* surrogate buys
    from the same evaluation budget. Without it, "the gradient is more sample
    efficient" would be a claim against straw men.

    Deliberately compact and dependency-free — a zero-mean GP with an RBF
    kernel, lengthscale set by the median pairwise distance, and expected
    improvement maximised over a random candidate pool. It is not a
    contribution and does not pretend to be tuned; it is a baseline, and a
    tuned one would only strengthen the comparison it is losing.

    Requires vector-valued decisions. Raises on anything else rather than
    coercing, in keeping with how the framework treats a mismatch.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.EVALUATE})

    def __init__(
        self,
        *,
        seed: int = 0,
        penalty: float = 1e6,
        n_initial: int = 8,
        n_candidates: int = 64,
        noise: float = 1e-6,
    ) -> None:
        self.seed = seed
        self.penalty = penalty
        self.n_initial = n_initial
        self.n_candidates = n_candidates
        self.noise = noise

    # -- the surrogate -------------------------------------------------
    @staticmethod
    def _kernel(a: Any, b: Any, lengthscale: float) -> Any:
        import numpy as np

        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-0.5 * d2 / (lengthscale**2))

    def _posterior(self, x: Any, y: Any, candidates: Any) -> tuple[Any, Any]:
        import numpy as np

        pairwise = np.sqrt(
            ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
        )
        upper = pairwise[np.triu_indices(len(x), k=1)]
        lengthscale = float(np.median(upper)) if upper.size else 1.0
        if not lengthscale > 0:
            lengthscale = 1.0

        mean, spread = y.mean(), y.std()
        spread = spread if spread > 0 else 1.0
        z = (y - mean) / spread

        k = self._kernel(x, x, lengthscale) + self.noise * np.eye(len(x))
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate designs
            chol = np.linalg.cholesky(k + 1e-4 * np.eye(len(x)))
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, z))

        ks = self._kernel(candidates, x, lengthscale)
        mu = ks @ alpha
        v = np.linalg.solve(chol, ks.T)
        var = np.clip(1.0 - (v**2).sum(axis=0), 1e-12, None)
        return mu * spread + mean, np.sqrt(var) * spread

    def solve(self, world: WorldModel, *, budget: int = 100) -> SolverResult:
        import math

        import numpy as np

        check_capabilities(self, world)
        rng = random.Random(self.seed)
        space = world.decision_space()

        def as_vector(decision: Any) -> list[float]:
            try:
                return [float(v) for v in decision]
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{type(self).__name__} needs vector-valued decisions; "
                    f"{type(decision).__name__} is not one."
                ) from exc

        start = space.initial()
        observed: list[list[float]] = []
        scores: list[float] = []
        best, best_eval, best_score = None, None, math.inf

        def record(decision: Any) -> float:
            nonlocal best, best_eval, best_score
            evaluation = world.evaluate(decision)
            score = penalised_objective(evaluation, self.penalty)
            observed.append(as_vector(decision))
            scores.append(score)
            if score < best_score:
                best, best_eval, best_score = decision, evaluation, score
            return score

        initial_score = record(start)
        history = [best_score]
        for _ in range(min(self.n_initial, budget) - 1):
            record(space.perturb(start, rng))
            history.append(best_score)

        while len(scores) < budget:
            pool = [space.perturb(best, rng) for _ in range(self.n_candidates)]
            pool += [space.perturb(start, rng) for _ in range(self.n_candidates // 2)]
            candidates = np.array([as_vector(p) for p in pool], dtype=float)
            mu, sigma = self._posterior(
                np.array(observed, dtype=float),
                np.array(scores, dtype=float),
                candidates,
            )
            # expected improvement for a MINIMISATION problem
            gap = best_score - mu
            z = gap / sigma
            normal = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
            density = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
            ei = gap * normal + sigma * density
            record(pool[int(np.argmax(ei))])
            history.append(best_score)

        assert best is not None and best_eval is not None
        return SolverResult(best, best_eval, best_score, initial_score,
                            len(scores), history)


class ExactSolver:
    """Branch and bound through the world model's algebraic form — needs ``C3``.

    The capability table says ``C3`` unlocks "exact solvers, dual bounds,
    optimality proofs", and until this class existed nothing consumed it: the
    registry offered random search on the one world model that can *prove* a
    solution optimal, because ``solve_exact()`` is a method on the world model
    while the menu is built from ``Solver`` objects. This is the adapter that
    was missing.

    It is the odd one out among the solvers here, and the differences are the
    point rather than an inconvenience:

    * ``budget`` is meaningless. Branch and bound does not consume evaluations
      one at a time; it is a single call that either returns a proof or runs
      out of clock. The parameter is accepted to satisfy the protocol and
      ignored, and ``evaluations_used`` reports the two evaluations actually
      made (the incumbent it started from, and the solution it returned);
    * it is the only solver that can answer **"is this optimal?"**. The answer
      is in ``termination`` on the result's evaluation provenance — a
      time-limited run returns a feasible incumbent with no proof attached,
      and calling that "the optimum" is the mistake this wrapper exists to
      make hard;
    * a deadline is not optional in practice. The model is exponential in the
      instance size, so a request without one may never come back.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.EVALUATE, Capability.ALGEBRAIC}
    )

    def __init__(self, *, time_limit: float | None = 60.0,
                 backend: str | None = None,
                 penalty: float = 1e6) -> None:
        self.time_limit = time_limit
        self.backend = backend
        self.penalty = penalty

    def solve(self, world: Any, *, budget: int = 1) -> SolverResult:
        check_capabilities(self, world)

        # A starting point, only so the report can say what was improved on.
        start = world.decision_space().initial()
        start_eval = world.evaluate(start)
        initial_score = penalised_objective(start_eval, self.penalty)

        if self.backend is not None:
            decision, objective = world.solve_exact(
                time_limit=self.time_limit, solver=self.backend
            )
        else:
            try:
                decision, objective = world.solve_exact(time_limit=self.time_limit)
            except TypeError:
                # An algebraic world model that predates the deadline argument.
                decision, objective = world.solve_exact()

        if decision is None or objective is None:
            # No incumbent. Reporting the starting point as "the result" would
            # be a lie of the worst kind here -- this solver's whole claim is
            # that its answer is exact.
            raise RuntimeError(
                "the exact solver returned no feasible solution"
                + (f" (termination: {world.termination})"
                   if hasattr(world, "termination") else "")
                + "; raise the time limit or shrink the instance"
            )

        evaluation = world.evaluate(decision)
        score = penalised_objective(evaluation, self.penalty)
        termination = getattr(world, "termination", None)
        backend = getattr(world, "solver_name", None)
        evaluation.provenance = (
            f"{evaluation.provenance} "
            f"[{backend or 'unknown backend'}: {termination or 'unknown'}]"
        )
        return SolverResult(decision, evaluation, score, initial_score,
                            evaluations_used=2, history=[initial_score, score])


class GradientDescentSolver:
    """Step along the pathwise gradient — requires ``C2a``.

    Included to make the capability gate visible: applied to a world model
    that only offers ``C0``, :func:`check_capabilities` refuses rather than
    silently degrading.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.EVALUATE, Capability.PATHWISE_GRADIENT}
    )

    def __init__(
        self,
        *,
        lr: float = 0.3,
        penalty: float = 1e6,
        backtrack: bool = True,
        shrink: float = 0.5,
        max_backtracks: int = 6,
    ) -> None:
        self.lr = lr
        self.penalty = penalty
        #: Halve the step until it improves, rather than trusting ``lr``.
        #: Without it this solver is at the mercy of a hand-set constant: on
        #: the sizing policy of Phase 5.2, ``lr=0.05`` reaches 38.40 while
        #: ``lr=0.3`` oscillates and ends at 39.98 -- which would have made a
        #: sample-efficiency comparison a statement about the constant rather
        #: than about the gradient. Every trial step is charged to the budget,
        #: so the comparison stays honest.
        self.backtrack = backtrack
        self.shrink = shrink
        self.max_backtracks = max_backtracks

    def solve(self, world: Any, *, budget: int = 100) -> SolverResult:
        check_capabilities(self, world)
        space = world.decision_space()

        decision = space.initial()
        evaluation = world.evaluate(decision)
        score = penalised_objective(evaluation, self.penalty)
        initial_score = score
        used = 1

        # Track the best point seen, not the last one reached: a fixed step can
        # walk away from a good iterate, and a solver that returns something
        # worse than what it has already evaluated is not reporting a result.
        best, best_eval, best_score = decision, evaluation, score
        history = [best_score]
        step = self.lr

        while used < budget:
            direction = world.gradient(decision).descent_direction()
            trials = self.max_backtracks if self.backtrack else 1
            moved = False

            for attempt in range(trials):
                if used >= budget:
                    break
                size = step * self.shrink**attempt
                candidate = [x + size * d for x, d in zip(decision, direction)]
                candidate_eval = world.evaluate(candidate)
                candidate_score = penalised_objective(candidate_eval, self.penalty)
                used += 1
                if candidate_score < score or not self.backtrack:
                    decision, evaluation, score = (
                        candidate, candidate_eval, candidate_score
                    )
                    step = size
                    moved = True
                    if candidate_score < best_score:
                        best, best_eval, best_score = (
                            candidate, candidate_eval, candidate_score
                        )
                    break

            history.append(best_score)
            if self.backtrack and not moved:
                step *= self.shrink
                if step < 1e-12:
                    break

        return SolverResult(best, best_eval, best_score, initial_score,
                            used, history)

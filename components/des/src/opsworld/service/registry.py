"""What the server is allowed to build, and what a client may ask it for.

The registry is the contract a front end is written against. It exists so the
UI is **generated from the framework rather than hard-coded against it** — and
the thing that makes that possible is already in the architecture:

    A world model declares its capabilities. A solver declares what it
    requires. Which solvers are offerable for a chosen world model is
    therefore a fact the server can compute, not a menu someone maintains.

That is architecture constraint A5 doing a second job. Point a gradient solver
at a ``C0``-only world model and :func:`~opsworld.solver.check_capabilities`
refuses; the registry surfaces the same rule one step earlier, so the control
is never offered in the first place.

Nothing here imports a domain (A6). Domains register themselves —
``intralogistics.service.register(registry)`` — and the composition root
(``examples/serve_api.py``) is the only place that knows both sides.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from opsworld.world import Capability

__all__ = ["ParamSpec", "ScenarioSpec", "WorldSpec", "SolverSpec", "Registry"]


@dataclass(frozen=True)
class ParamSpec:
    """One control in the UI, described well enough to render itself."""

    name: str
    kind: str  # "int" | "float" | "bool" | "enum"
    default: Any
    label: str
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] | None = None

    def coerce(self, value: Any) -> Any:
        """Validate and convert one client-supplied value.

        Clamping rather than rejecting is deliberate for the numeric kinds: a
        slider that has been dragged past its limit means "as far as it goes",
        and refusing the request would be a worse answer than the extreme.
        Anything not on the enum is a genuine mismatch and is refused.
        """
        if self.kind == "bool":
            return bool(value)
        if self.kind == "enum":
            if value not in (self.choices or ()):
                raise ValueError(
                    f"{self.name}: {value!r} is not one of {list(self.choices or ())}"
                )
            return value
        try:
            number = int(value) if self.kind == "int" else float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: expected a {self.kind}, got {value!r}") from exc
        if self.minimum is not None:
            number = max(number, int(self.minimum) if self.kind == "int" else self.minimum)
        if self.maximum is not None:
            number = min(number, int(self.maximum) if self.kind == "int" else self.maximum)
        return number

    def as_json(self) -> dict:
        out = {
            "name": self.name, "kind": self.kind, "default": self.default,
            "label": self.label, "help": self.help,
        }
        for key, value in (("min", self.minimum), ("max", self.maximum),
                           ("step", self.step)):
            if value is not None:
                out[key] = value
        if self.choices is not None:
            out["choices"] = list(self.choices)
        return out


def _coerce(params: Sequence[ParamSpec], supplied: dict | None) -> dict:
    supplied = supplied or {}
    unknown = set(supplied) - {p.name for p in params}
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    return {p.name: p.coerce(supplied.get(p.name, p.default)) for p in params}


@dataclass(frozen=True)
class ScenarioSpec:
    """A family of problem instances the client can dial in."""

    id: str
    label: str
    description: str
    params: tuple[ParamSpec, ...]
    build: Callable[..., Any]
    summarise: Callable[[Any], dict]

    def make(self, params: dict | None = None) -> tuple[Any, dict]:
        resolved = _coerce(self.params, params)
        instance = self.build(**resolved)
        return instance, resolved

    def as_json(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "params": [p.as_json() for p in self.params]}


@dataclass(frozen=True)
class WorldSpec:
    """A world model, plus the codec for the decisions it takes."""

    id: str
    label: str
    description: str
    scenario: str
    build: Callable[[Any], Any]
    encode_decision: Callable[[Any, Any], Any]
    decode_decision: Callable[[Any, Any], Any]
    #: Whether a plain evaluate() is fast enough to answer inside a request.
    #: Measured, not guessed.
    interactive: bool = True
    #: Largest solver budget still worth running *inline*; 0 means never.
    #:
    #: Spawning a worker costs about 3.8 s on Windows, because ``spawn``
    #: re-imports the entry module and torch with it — measured against a
    #: gradient run whose actual work was 18 ms. So "always a job" is the wrong
    #: default: it would make the fast world models 200× slower than they are.
    #: The limit is per world model because the cost per evaluation is a
    #: property of the model (0.2 ms for a policy weight vector; 8.8 ms for a
    #: 6x3 configuration and 87 ms for a 40x4 one, which is why the
    #: configuration worlds set this to 0 and always queue).
    inline_max_budget: int = 0
    #: Optional per-world extras a client can render (e.g. rule names).
    describe: Callable[[Any, Any], dict] | None = None

    def capabilities(self, world: Any) -> set[Capability]:
        return set(world.capabilities())

    def as_json(self, capabilities: Sequence[str] | None = None) -> dict:
        return {
            "id": self.id, "label": self.label, "description": self.description,
            "scenario": self.scenario, "interactive": self.interactive,
            "inline_max_budget": self.inline_max_budget,
            "capabilities": list(capabilities or []),
        }


@dataclass(frozen=True)
class SolverSpec:
    """A search strategy, with the capabilities it needs to run at all."""

    id: str
    label: str
    description: str
    requires: frozenset[Capability]
    build: Callable[..., Any]
    params: tuple[ParamSpec, ...] = ()
    #: Methods the *decision space* must provide, beyond ``initial``/``perturb``.
    #:
    #: A gap the HTTP surface exposed: capabilities describe what a **world
    #: model** can answer, and say nothing about what its **decision space**
    #: supports. ``GreedySupplySolver`` declares only ``C0`` — truthfully, it
    #: evaluates and nothing more — yet it enumerates a neighbourhood via
    #: ``space.moves()``, which a ``VectorSpace`` does not have. Offered against
    #: a weight-vector world it would pass the capability gate and then fail
    #: with an ``AttributeError``. Naming the requirement keeps the menu honest.
    requires_space: frozenset[str] = frozenset()
    #: Rough cost per evaluation, for the client to warn before a long run.
    #: ``None`` means "depends on the world model", which is the honest answer
    #: for the generic solvers.
    seconds_per_evaluation: float | None = None

    def make(self, params: dict | None = None) -> tuple[Any, dict]:
        resolved = _coerce(self.params, params)
        return self.build(**resolved), resolved

    def serves(self, capabilities: set[Capability], space: Any) -> bool:
        if not self.requires <= capabilities:
            return False
        return all(callable(getattr(space, name, None)) for name in self.requires_space)

    def as_json(self) -> dict:
        return {
            "id": self.id, "label": self.label, "description": self.description,
            "requires": sorted(c.value for c in self.requires),
            "requires_space": sorted(self.requires_space),
            "params": [p.as_json() for p in self.params],
            "seconds_per_evaluation": self.seconds_per_evaluation,
        }


class Registry:
    """Everything the server can build, and the rules connecting them."""

    def __init__(self) -> None:
        self.scenarios: dict[str, ScenarioSpec] = {}
        self.worlds: dict[str, WorldSpec] = {}
        self.solvers: dict[str, SolverSpec] = {}

    # -- registration -------------------------------------------------
    def add_scenario(self, spec: ScenarioSpec) -> None:
        self._reject_duplicate(self.scenarios, spec.id, "scenario")
        self.scenarios[spec.id] = spec

    def add_world(self, spec: WorldSpec) -> None:
        self._reject_duplicate(self.worlds, spec.id, "world")
        if spec.scenario not in self.scenarios:
            raise KeyError(
                f"world {spec.id!r} names scenario {spec.scenario!r}, which is "
                f"not registered (have: {sorted(self.scenarios)})"
            )
        self.worlds[spec.id] = spec

    def add_solver(self, spec: SolverSpec) -> None:
        self._reject_duplicate(self.solvers, spec.id, "solver")
        self.solvers[spec.id] = spec

    @staticmethod
    def _reject_duplicate(table: dict, key: str, kind: str) -> None:
        if key in table:
            raise KeyError(f"{kind} {key!r} is already registered")

    # -- lookup -------------------------------------------------------
    def scenario(self, id: str) -> ScenarioSpec:
        return self._get(self.scenarios, id, "scenario")

    def world(self, id: str) -> WorldSpec:
        return self._get(self.worlds, id, "world")

    def solver(self, id: str) -> SolverSpec:
        return self._get(self.solvers, id, "solver")

    @staticmethod
    def _get(table: dict, key: str, kind: str):
        try:
            return table[key]
        except KeyError:
            raise KeyError(
                f"unknown {kind} {key!r}; available: {sorted(table)}"
            ) from None

    # -- the capability gate, one step earlier ------------------------
    def solvers_for(self, capabilities: set[Capability],
                    space: Any = None) -> list[SolverSpec]:
        """Solvers this world model can actually serve.

        The rule ``check_capabilities`` enforces at call time, applied at
        menu-build time so an impossible combination is never offered — plus
        the decision-space requirement that capabilities alone do not express
        (see :attr:`SolverSpec.requires_space`).
        """
        return [s for s in self.solvers.values() if s.serves(capabilities, space)]

    def build_world(self, world_id: str, scenario_params: dict | None = None):
        """``(world, instance, resolved scenario params)`` for one request."""
        spec = self.world(world_id)
        instance, resolved = self.scenario(spec.scenario).make(scenario_params)
        return spec.build(instance), instance, resolved

    # -- the contract a front end is written against ------------------
    def as_json(self) -> dict:
        worlds = []
        for spec in self.worlds.values():
            try:
                world, _, _ = self.build_world(spec.id)
                capabilities = spec.capabilities(world)
                space = world.decision_space()
                caps = sorted(c.value for c in capabilities)
                offerable = [s.id for s in self.solvers_for(capabilities, space)]
                entry = spec.as_json(caps)
                entry["solvers"] = offerable
            except Exception as exc:  # noqa: BLE001 - a broken spec must be visible
                entry = spec.as_json([])
                entry["solvers"] = []
                entry["error"] = f"{type(exc).__name__}: {exc}"
            worlds.append(entry)
        return {
            "scenarios": [s.as_json() for s in self.scenarios.values()],
            "worlds": worlds,
            "solvers": [s.as_json() for s in self.solvers.values()],
            "capabilities": {
                c.value: d for c, d in [
                    (Capability.EVALUATE, "评估一个决策"),
                    (Capability.SEQUENTIAL, "序贯环境:reset / step"),
                    (Capability.PATHWISE_GRADIENT, "对决策变量的通路梯度"),
                    (Capability.DUALS, "约束右端的影子价格 — 不是搜索方向"),
                    (Capability.ALGEBRAIC, "代数结构:可精确求解"),
                ]
            },
        }

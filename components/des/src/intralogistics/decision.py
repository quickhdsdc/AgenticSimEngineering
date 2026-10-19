"""Decision and evaluation types for the intralogistics world models.

A :class:`Decision` holds only the *primary* decision variables. The cost
variables of the source model (``k_storage``, ``k_pick_var``, ``k_pick_fix``,
``k_reloc``) are **not** part of a decision: constraints (2)-(5) of the source
MILP are *definitions*, not restrictions, and therefore belong to the world
model, which computes them. Keeping that split is what makes the world model
usable without a solver (see ``PLAN.md`` Axiom 1).

:class:`Evaluation` was promoted into ``opsworld`` in Phase 2 and is
re-exported here so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opsworld.world import Evaluation

__all__ = ["Decision", "Evaluation", "Key"]

Key = tuple[Any, ...]


@dataclass
class Decision:
    """A candidate intralogistics configuration.

    ``x_*`` are binary in the MILP, ``y_*`` integer, ``q_inventory``
    continuous. Values are stored as floats so that relaxed or heuristic
    proposals can be evaluated without conversion.

    Missing keys read as zero, so a sparse decision is legal input.
    """

    x_store: dict[Key, float] = field(default_factory=dict)
    q_inventory: dict[Key, float] = field(default_factory=dict)
    x_product_from_warehouse: dict[Key, float] = field(default_factory=dict)
    y_reloc_moves: dict[Key, float] = field(default_factory=dict)
    x_pick: dict[Key, float] = field(default_factory=dict)
    x_carrier: dict[Key, float] = field(default_factory=dict)
    y_carriers: dict[Key, float] = field(default_factory=dict)

    VARIABLE_NAMES: tuple[str, ...] = (
        "x_store",
        "q_inventory",
        "x_product_from_warehouse",
        "y_reloc_moves",
        "x_pick",
        "x_carrier",
        "y_carriers",
    )

    def get(self, variable: str, *key: Any) -> float:
        """Value of ``variable[key]``, or 0.0 if unset."""
        return float(getattr(self, variable).get(key, 0.0))

    def as_dict(self) -> dict[str, dict[Key, float]]:
        return {name: getattr(self, name) for name in self.VARIABLE_NAMES}

    def nonzero(self, tol: float = 1e-6) -> dict[str, dict[Key, float]]:
        """The support of the decision — useful for reporting and diffing."""
        return {
            name: {k: v for k, v in getattr(self, name).items() if abs(v) > tol}
            for name in self.VARIABLE_NAMES
        }


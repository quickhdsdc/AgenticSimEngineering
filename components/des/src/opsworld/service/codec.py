"""JSON on the wire — and the one place the domain types resist it.

A world model in this framework indexes its decisions the way the source
models do: by tuples. ``q_inventory[("A1", "L1")]``, ``x_reloc[(i, r, l1, l2,
b1, b2)]``. That is the right shape for the algebra and the wrong shape for
JSON, whose object keys must be strings — measured rather than assumed::

    Evaluation    -> serialises
    SolverResult  -> serialises (``history`` is already a per-iteration trace)
    Decision      -> TypeError: keys must be str, int, float, bool or None
    SupplyPlan    -> TypeError: same

So the crossing needs an explicit codec, and this is it. Two rules it follows:

**The key encoding is reversible and unambiguous.** A tuple becomes the JSON
array that spells it — ``("A1", "L1")`` → ``'["A1", "L1"]'`` — rather than a
delimiter join, which would break the moment an entity name contained the
delimiter. Round-tripping is a test, not a hope.

**Nothing here imports a domain.** Architecture constraint A6 keeps the shared
layer thin, and a shared layer that imports ``intralogistics`` to serialise a
``Decision`` would invert the dependency and stop being shared. What lives here
are the generic table/`Evaluation`/`SolverResult` codecs; each domain supplies
the codec for its own decision type and hands it to the registry.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = [
    "encode_key",
    "decode_key",
    "encode_index_map",
    "decode_index_map",
    "encode_table",
    "decode_table",
    "encode_evaluation",
    "encode_sensitivity",
    "encode_solver_result",
    "jsonable",
]


# ----------------------------------------------------------------------
# tuple keys
# ----------------------------------------------------------------------
def encode_key(key: Any) -> str:
    """A tuple index as a JSON-object key.

    ``("A1", "L1")`` becomes ``'["A1", "L1"]'``. A scalar key is wrapped so the
    decoder never has to guess arity: ``"L1"`` becomes ``'["L1"]'``.
    """
    parts = list(key) if isinstance(key, tuple) else [key]
    return json.dumps(parts, ensure_ascii=False, separators=(", ", ": "))


def decode_key(key: str) -> tuple:
    """Inverse of :func:`encode_key`. Raises on anything it did not write."""
    try:
        parts = json.loads(key)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{key!r} is not an encoded index; expected a JSON array such as "
            '\'["A1", "L1"]\''
        ) from exc
    if not isinstance(parts, list):
        raise ValueError(f"{key!r} decodes to {type(parts).__name__}, not a list")
    return tuple(parts)


def encode_index_map(table: dict[Any, Any]) -> dict[str, Any]:
    """A tuple-indexed map as a JSON object, values untouched.

    Not every indexed table holds numbers. A supply plan maps
    ``(item, resource)`` to a *warehouse name* and ``(item, warehouse)`` to a
    *carrier name*; coercing those to float destroys them, so the numeric
    variant below is the specialisation and this is the general case.
    """
    return {encode_key(k): v for k, v in table.items()}


def decode_index_map(table: dict[str, Any]) -> dict[tuple, Any]:
    """Inverse of :func:`encode_index_map`."""
    return {decode_key(k): v for k, v in table.items()}


def encode_table(table: dict[Any, float]) -> dict[str, float]:
    """A tuple-indexed *numeric* variable table as a JSON object."""
    return {encode_key(k): float(v) for k, v in table.items()}


def decode_table(table: dict[str, Any]) -> dict[tuple, float]:
    """Inverse of :func:`encode_table`."""
    return {decode_key(k): float(v) for k, v in table.items()}


# ----------------------------------------------------------------------
# opsworld types
# ----------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    """Coerce a number to something ``json.dumps`` accepts.

    ``inf`` and ``nan`` are real outputs here — an all-zero gradient reports
    infinite relative noise — and ``json.dumps`` emits bare ``Infinity``, which
    is not valid JSON and which browsers reject. They travel as ``null`` with
    the meaning carried by the field's own documentation.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def encode_sensitivity(sensitivity: Any) -> dict | None:
    """A :class:`~opsworld.world.Sensitivity`, with its kind preserved.

    ``kind`` is not decoration: decision record D10 makes a dual variable
    illegal as a search direction, so the client has to be able to tell which
    object it received.
    """
    if sensitivity is None:
        return None
    return {
        "kind": str(sensitivity.kind),
        "names": list(sensitivity.names),
        "values": [jsonable(float(v)) for v in sensitivity.values],
        "text": sensitivity.as_text(),
    }


def encode_evaluation(evaluation: Any) -> dict:
    """A :class:`~opsworld.world.Evaluation` as the UI needs to read it.

    ``violations`` is carried verbatim rather than collapsed to a boolean: the
    magnitude per constraint family is what lets a client show *which* limit is
    breached and by how much, which is most of the value of the static model.
    """
    return {
        "objective": jsonable(float(evaluation.objective)),
        "kpis": {k: jsonable(float(v)) for k, v in evaluation.kpis.items()},
        "violations": {k: jsonable(float(v)) for k, v in evaluation.violations.items()},
        "feasible": bool(evaluation.is_feasible()),
        "total_violation": jsonable(float(evaluation.total_violation())),
        "provenance": evaluation.provenance,
        "sensitivity": encode_sensitivity(evaluation.sensitivity),
        "text": evaluation.as_text(),
    }


def encode_solver_result(result: Any, *, decision: Any = None) -> dict:
    """A :class:`~opsworld.solver.SolverResult`.

    ``history`` is the best-so-far score after each step and is already the
    shape a convergence chart wants; it is returned whole rather than streamed,
    so a client animates a finished trace instead of holding a socket open.
    """
    return {
        "score": jsonable(float(result.score)),
        "initial_score": jsonable(float(result.initial_score)),
        "improvement": jsonable(float(result.improvement)),
        "evaluations_used": int(result.evaluations_used),
        "history": [jsonable(float(v)) for v in result.history],
        "evaluation": encode_evaluation(result.evaluation),
        "decision": decision,
        "summary": result.summary(),
    }

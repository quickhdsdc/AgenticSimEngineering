"""Submodel-template instantiation.

An IDTA submodel template is not a blank form. It carries the slot structure,
the semantic ids that make a slot mean something, and -- crucially -- a
cardinality declaration per slot saying whether the slot is mandatory. Turning
one into an instance is therefore a transformation, not a copy, and three parts
of it are easy to get wrong:

**Cardinality is written four different ways.** Across the 39 templates in the
reference repository the same idea appears as ``TemplateQualifier/SMT/Cardinality``
(the current IDTA spelling), ``ConceptQualifier/Cardinality``,
``ConceptQualifier/Multiplicity``, and ``ConceptQualifier/SMT/Cardinality``, with
values drawn from two vocabularies (``One``/``ZeroToOne``/... and ``1``/``0..1``/...).
A reader that understands only one of them silently treats optional slots as
mandatory, or vice versa. :func:`cardinality` normalises all four.

**Template qualifiers may not survive into an instance.** AASd-129 says a
qualifier with ``kind = TemplateQualifier`` may only appear inside a submodel
whose ``kind`` is ``Template``. Copying a template verbatim and flipping
``kind`` to ``Instance`` -- which is what the tool this replaces did -- produces
a submodel that violates the metamodel. The cardinality is not lost, though: it
is reported in the instantiation summary, where the caller can act on it.

**Example values are not data.** ``ConceptQualifier/ExampleValue`` exists so a
human reading the template knows what the slot looks like. Seeding it into an
instance manufactures a value that traces to no source, which is precisely the
failure a passport is supposed to prevent. Opt in with ``seed_examples`` if you
want a demo fixture; the default is an empty instance.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Qualifier ``type`` values that carry a cardinality, in every spelling seen in
# the wild. Compared case-sensitively -- these are spec strings, not free text.
CARDINALITY_TYPES = ("SMT/Cardinality", "Cardinality", "Multiplicity")

# Qualifier ``type`` values that describe the template itself: authoring hints
# for form editors and the cardinality declarations above. None of them mean
# anything on an instance, so they are dropped unless explicitly kept.
TEMPLATE_ONLY_TYPES = frozenset(
    {
        *CARDINALITY_TYPES,
        "ExampleValue",
        "EditIdShort",
        "PresetIdShort",
        "FormTitle",
        "FormInfo",
        "FormChoices",
        "Required",
        "Offered",
        "NotAssigned",
        "LifeCycleQual",
        "LoadStatus",
    }
)

_CARDINALITY_WORDS: Dict[str, Tuple[int, Optional[int]]] = {
    "One": (1, 1),
    "ZeroToOne": (0, 1),
    "ZeroToMany": (0, None),
    "OneToMany": (1, None),
}

_RANGE_RE = re.compile(r"^\s*(\d+)\s*\.\.\s*(\d+|\*|n|N)\s*$")

# Element kinds whose ``value`` is asserted data about the asset rather than
# structure. These are the slots ``aas populate`` fills, so instantiation
# empties them; everything else keeps whatever the template declared.
_DATA_BEARING = frozenset({"Property", "MultiLanguageProperty", "File", "Blob"})

_CHILD_KEYS = ("submodelElements", "value", "statements", "annotations")


class SmtError(ValueError):
    """The template cannot be instantiated as asked."""


def cardinality(node: Dict[str, Any]) -> Tuple[int, Optional[int]]:
    """Return ``(minimum, maximum)`` for one template node; ``None`` means unbounded.

    An absent declaration is read as ``One``. That follows the SMT spec, and it
    is also the safe direction: an unannotated slot is part of the template's
    structure and dropping it would quietly shrink the instance.
    """
    for qual in node.get("qualifiers") or []:
        if qual.get("type") not in CARDINALITY_TYPES:
            continue
        raw = (qual.get("value") or "").strip()
        if raw in _CARDINALITY_WORDS:
            return _CARDINALITY_WORDS[raw]
        match = _RANGE_RE.match(raw)
        if match:
            low, high = match.group(1), match.group(2)
            return int(low), None if high in ("*", "n", "N") else int(high)
        if raw.isdigit():
            return int(raw), int(raw)
        # An unrecognised spelling is reported by the caller rather than
        # guessed at; treating it as mandatory keeps the slot.
        return 1, 1
    return 1, 1


def is_optional(node: Dict[str, Any]) -> bool:
    return cardinality(node)[0] == 0


def cardinality_label(node: Dict[str, Any]) -> str:
    low, high = cardinality(node)
    return f"{low}..{'*' if high is None else high}"


@dataclass
class Instantiation:
    """The result of a transformation, with enough detail to review it."""

    submodel: Dict[str, Any]
    kept: List[str] = field(default_factory=list)
    dropped_optional: List[str] = field(default_factory=list)
    stripped_qualifiers: int = 0
    seeded: List[str] = field(default_factory=list)
    unknown_cardinality: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.submodel.get("id"),
            "idShort": self.submodel.get("idShort"),
            "kind": self.submodel.get("kind"),
            "elements": len(self.submodel.get("submodelElements") or []),
            "kept": len(self.kept),
            "dropped_optional": len(self.dropped_optional),
            "stripped_qualifiers": self.stripped_qualifiers,
            "seeded": len(self.seeded),
            "unknown_cardinality": self.unknown_cardinality,
        }


def instantiate(
    template: Dict[str, Any],
    *,
    submodel_id: str,
    id_short: Optional[str] = None,
    include_optional: bool = False,
    keep_values: bool = False,
    seed_examples: bool = False,
    keep_template_qualifiers: bool = False,
    require_template: bool = True,
) -> Instantiation:
    """Transform a ``kind: Template`` submodel into a ``kind: Instance`` one.

    ``include_optional`` decides the shape of the result: by default only slots
    with a minimum cardinality of 1 survive, which is the smallest submodel that
    still satisfies the template. Pass it when the caller intends to fill
    optional slots too -- an empty optional slot is not an error, but neither is
    it evidence of anything, so it is not created speculatively.
    """
    if require_template and template.get("kind") != "Template":
        raise SmtError(
            f"{template.get('id')!r} has kind={template.get('kind')!r}, not 'Template'. "
            "Pass --force to instantiate from it anyway."
        )

    result = Instantiation(submodel=copy.deepcopy(template))
    sm = result.submodel
    sm["kind"] = "Instance"
    sm["id"] = submodel_id
    if id_short:
        sm["idShort"] = id_short

    _transform(sm, "", result, include_optional, keep_values, seed_examples,
               keep_template_qualifiers)
    return result


def _transform(
    node: Dict[str, Any],
    path: str,
    result: Instantiation,
    include_optional: bool,
    keep_values: bool,
    seed_examples: bool,
    keep_template_qualifiers: bool,
) -> None:
    example = _example_value(node)
    if not keep_template_qualifiers:
        result.stripped_qualifiers += _strip_qualifiers(node)

    if node.get("modelType") in _DATA_BEARING:
        if seed_examples and example is not None:
            _set_value(node, example)
            result.seeded.append(path)
        elif not keep_values:
            _clear_value(node)

    for key in _CHILD_KEYS:
        children = node.get(key)
        if not isinstance(children, list):
            continue
        # A SubmodelElementList addresses its children by index, so a path step
        # for them is [i] rather than an idShort -- and a template list is
        # instantiated empty, since its items are data the caller has yet to add.
        in_list = node.get("modelType") == "SubmodelElementList"
        surviving: List[Any] = []
        for index, child in enumerate(children):
            if not isinstance(child, dict) or "modelType" not in child:
                surviving.append(child)
                continue
            step = f"[{index}]" if in_list else str(child.get("idShort") or f"[{index}]")
            child_path = f"{path}{step}" if in_list else (f"{path}.{step}" if path else step)

            _note_unknown(child, child_path, result)
            if in_list or (not include_optional and is_optional(child)):
                if not in_list:
                    result.dropped_optional.append(child_path)
                # List items are template exemplars; dropping them is not an
                # "optional slot dropped" event, so it is not reported as one.
                continue

            result.kept.append(child_path)
            _transform(child, child_path, result, include_optional, keep_values,
                       seed_examples, keep_template_qualifiers)
            surviving.append(child)
        node[key] = surviving


def _note_unknown(node: Dict[str, Any], path: str, result: Instantiation) -> None:
    for qual in node.get("qualifiers") or []:
        if qual.get("type") not in CARDINALITY_TYPES:
            continue
        raw = (qual.get("value") or "").strip()
        if raw not in _CARDINALITY_WORDS and not _RANGE_RE.match(raw) and not raw.isdigit():
            result.unknown_cardinality.append(f"{path}: {raw!r}")
        return


def _strip_qualifiers(node: Dict[str, Any]) -> int:
    """Remove template-authoring qualifiers; keep the ones that still do work.

    ``invocationDelegation`` is the case that matters: it is what makes an
    Operation callable, so it is functional metadata and must survive.
    """
    qualifiers = node.get("qualifiers")
    if not isinstance(qualifiers, list):
        return 0
    kept = [
        q
        for q in qualifiers
        if q.get("kind") != "TemplateQualifier" and q.get("type") not in TEMPLATE_ONLY_TYPES
    ]
    removed = len(qualifiers) - len(kept)
    if kept:
        node["qualifiers"] = kept
    else:
        node.pop("qualifiers", None)
    return removed


def _example_value(node: Dict[str, Any]) -> Optional[str]:
    for qual in node.get("qualifiers") or []:
        if qual.get("type") == "ExampleValue":
            return qual.get("value")
    return None


def _clear_value(node: Dict[str, Any]) -> None:
    model_type = node.get("modelType")
    if model_type == "MultiLanguageProperty":
        node["value"] = []
    elif model_type == "Blob":
        node.pop("value", None)
    else:
        node["value"] = None


def _set_value(node: Dict[str, Any], raw: str) -> None:
    """Write an example into a slot, honouring the ``text@lang`` form templates use."""
    if node.get("modelType") == "MultiLanguageProperty":
        text, _, lang = raw.rpartition("@")
        node["value"] = [{"language": lang or "en", "text": text or raw}]
    else:
        node["value"] = raw


def outline(template: Dict[str, Any], max_rows: int = 60) -> List[str]:
    """One ``cardinality  path  type`` row per slot, for deciding what to fill."""
    rows: List[str] = []

    def walk(node: Dict[str, Any], path: str, level: int) -> None:
        for key in _CHILD_KEYS:
            children = node.get(key)
            if not isinstance(children, list):
                continue
            in_list = node.get("modelType") == "SubmodelElementList"
            for index, child in enumerate(children):
                if not isinstance(child, dict) or "modelType" not in child:
                    continue
                if len(rows) >= max_rows:
                    return
                step = f"[{index}]" if in_list else str(child.get("idShort") or f"[{index}]")
                child_path = f"{path}{step}" if in_list else (f"{path}.{step}" if path else step)
                rows.append(
                    f"{cardinality_label(child):<6} {'  ' * level}{child_path:<44} "
                    f"{child.get('modelType')}"
                )
                walk(child, child_path, level + 1)

    walk(template, "", 0)
    return rows


# ---------------------------------------------------------------------------
# conformance: what the template asked for that the instance does not have
# ---------------------------------------------------------------------------

# A slot's value counts as absent when it is any of these. `False` and `0` are
# deliberately not in the set -- they are answers, not gaps.
def _is_empty(node: Dict[str, Any]) -> bool:
    model_type = node.get("modelType")
    if model_type in ("SubmodelElementCollection", "SubmodelElementList"):
        return not (node.get("value") or [])
    if model_type == "MultiLanguageProperty":
        return not (node.get("value") or [])
    if model_type == "Range":
        return node.get("min") in (None, "") and node.get("max") in (None, "")
    if model_type in ("Operation", "Capability", "RelationshipElement", "Entity"):
        # Structural elements are not filled in; their presence is the point.
        return False
    value = node.get("value")
    return value is None or value == ""


@dataclass
class Gap:
    path: str
    model_type: str
    cardinality: str
    status: str          # "missing" | "empty"
    semantic_id: Optional[str] = None

    def line(self) -> str:
        return f"{self.status:<8} {self.cardinality:<6} {self.path:<44} {self.model_type}"


def _index(node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in _CHILD_KEYS:
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict) and child.get("idShort"):
                    out[child["idShort"]] = child
    return out


def gaps(
    template: Dict[str, Any],
    instance: Dict[str, Any],
    *,
    include_optional: bool = False,
) -> List[Gap]:
    """Slots the template requires that the instance has not answered.

    Which slots are mandatory is knowable only from the template -- an
    instantiated submodel has had its cardinality qualifiers stripped, because
    AASd-129 forbids them there. So completeness is a comparison between two
    submodels, not a property of one, and it cannot live in ``aas validate``.

    A slot is a gap when its element is absent (``missing``) or present with no
    value (``empty``). The distinction matters: ``missing`` needs
    ``aas el add``, ``empty`` needs ``aas set``.
    """
    found: List[Gap] = []

    def walk(t_node: Dict[str, Any], i_node: Optional[Dict[str, Any]], path: str) -> None:
        present = _index(i_node) if i_node else {}
        for key in _CHILD_KEYS:
            children = t_node.get(key)
            if not isinstance(children, list):
                continue
            if t_node.get("modelType") == "SubmodelElementList":
                # List items are exemplars, not slots; nothing to compare.
                continue
            for child in children:
                if not isinstance(child, dict) or "modelType" not in child:
                    continue
                id_short = child.get("idShort")
                if not id_short:
                    continue
                optional = is_optional(child)
                if optional and not include_optional:
                    continue
                child_path = f"{path}.{id_short}" if path else id_short
                match = present.get(id_short)
                keys = (child.get("semanticId") or {}).get("keys") or []
                semantic = keys[0].get("value") if keys else None
                if match is None:
                    found.append(
                        Gap(child_path, child.get("modelType", "?"),
                            cardinality_label(child), "missing", semantic)
                    )
                    continue
                if _is_empty(match):
                    found.append(
                        Gap(child_path, child.get("modelType", "?"),
                            cardinality_label(child), "empty", semantic)
                    )
                walk(child, match, child_path)

    walk(template, instance, "")
    return found

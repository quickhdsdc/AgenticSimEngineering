"""Turning a document into a submodel *template*, not into values.

`extract.py` answers "what does this document say", and produces flat facts:
`{entity, value, unit, source}`. That is the right shape for `aas match`, which
writes them into a template someone already wrote.

This answers a different question -- "what shape of submodel would hold what
this document describes" -- and the answer is a tree. The gap it closes is real:
`aas smt instantiate` requires a `kind: Template` submodel to already exist, so
a product class nobody has written a template for is unreachable, which is the
same wall the per-tool design hit for a different reason.

Four things make it hard, and only the first is about getting JSON out of a
model:

**Hierarchy.** Which fields belong together under a collection. Getting the
slots right and the nesting wrong scores full marks on any per-field metric and
produces a template nobody can use -- `MaxTorque` at the root instead of inside
`AxisData` is not a near miss, it is a different template.

**Cardinality.** Which slots are mandatory, which repeat. This is what makes a
template a template rather than a list of names, and it is the field
`smt.instantiate` reads to decide what an instance must carry.

**Value types.** `xs:double` against `xs:string`. The predecessor made every
new element a string, and a capacity written as a string is not comparable, not
summable, and not valid against a template that declares a decimal.

**Semantics.** A new template should *reuse* a ConceptDescription where one
exists rather than mint a synonym. Minting is what turns a repository into a
pile of near-duplicate concepts that no retrieval can tell apart -- and this
repository already shows the symptom: 677 distinct semanticIds over 3,789
elements, with `X`, `Y` and `Z` recurring 72 times each.

The output is deliberately **not** an AAS submodel. It is a description of one,
which a person can read and correct before anything is written. `aas smt author`
turns it into the real thing, and that separation is the same one `aas match`
and `aas populate` already have: propose, review, apply.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .aasctl.model import ELEMENT_TYPES, VALUE_TYPES

# Cardinalities `smt.cardinality` can read back. Written in the IDTA word form
# rather than `1..1`, because that is the spelling `TemplateQualifier`
# `SMT/Cardinality` uses and the one instantiation round-trips without
# translation.
CARDINALITIES = ("One", "ZeroToOne", "OneToMany", "ZeroToMany")

# Containers, i.e. the types whose `children` mean something.
CONTAINERS = ("SubmodelElementCollection", "SubmodelElementList")

_IDSHORT_SAFE = re.compile(r"[^A-Za-z0-9_]+")
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


class StructureError(ValueError):
    """The proposed structure cannot be read or is not buildable."""


@dataclass
class Slot:
    """One node of a proposed template."""

    id_short: str
    model_type: str = "Property"
    value_type: str = "xs:string"
    cardinality: str = "One"
    unit: str = ""
    description: str = ""
    semantic_id: Optional[str] = None
    # Where in the document this came from. Carried for the same reason
    # `extract.Extracted` carries it: a slot that traces to no source is a slot
    # somebody invented, and on a template that error propagates into every
    # instance built from it.
    source: str = ""
    type_value_list_element: Optional[str] = None
    children: List["Slot"] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "idShort": self.id_short,
            "modelType": self.model_type,
            "cardinality": self.cardinality,
        }
        if self.model_type == "Property":
            out["valueType"] = self.value_type
        if self.unit:
            out["unit"] = self.unit
        if self.description:
            out["description"] = self.description
        if self.semantic_id:
            out["semanticId"] = self.semantic_id
        if self.source:
            out["source"] = self.source
        if self.type_value_list_element:
            out["typeValueListElement"] = self.type_value_list_element
        if self.children:
            out["children"] = [c.as_dict() for c in self.children]
        return out

    def walk(self) -> Iterable["Slot"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Structure:
    """A proposed template: an idShort, a description, and a tree of slots."""

    id_short: str
    slots: List[Slot] = field(default_factory=list)
    description: str = ""
    source: str = ""
    semantic_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "idShort": self.id_short,
            "description": self.description,
            "source": self.source,
            "semanticId": self.semantic_id,
            "slots": [s.as_dict() for s in self.slots],
        }

    def walk(self) -> Iterable[Slot]:
        for slot in self.slots:
            yield from slot.walk()

    def counts(self) -> Dict[str, int]:
        slots = list(self.walk())
        return {
            "slots": len(slots),
            "mandatory": sum(1 for s in slots if s.cardinality in ("One", "OneToMany")),
            "containers": sum(1 for s in slots if s.model_type in CONTAINERS),
            "depth": max((_depth(s) for s in self.slots), default=0),
            "with_unit": sum(1 for s in slots if s.unit),
            "with_semantic_id": sum(1 for s in slots if s.semantic_id),
            "without_source": sum(1 for s in slots if not s.source),
        }


def _depth(slot: Slot, level: int = 1) -> int:
    return max((_depth(c, level + 1) for c in slot.children), default=level)


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

TEMPLATE_PROMPT = """You design Asset Administration Shell submodel templates \
(IEC 63278) from technical documents.

Read the document and propose the STRUCTURE of a submodel template that would \
hold the kind of information it describes. You are designing the form, not \
filling it in: do not include any values.

Return ONLY a JSON object:

{
  "idShort": "WeldingRobotSpec",
  "description": "one sentence saying what this template is for",
  "slots": [
    {
      "idShort": "ManufacturerName",
      "modelType": "Property",
      "valueType": "xs:string",
      "cardinality": "One",
      "description": "who made it",
      "source": "spec.pdf p.1"
    },
    {
      "idShort": "AxisData",
      "modelType": "SubmodelElementList",
      "typeValueListElement": "SubmodelElementCollection",
      "cardinality": "OneToMany",
      "description": "one entry per axis",
      "source": "spec.pdf p.2",
      "children": [
        {"idShort": "MaxTorque", "modelType": "Property", "valueType": "xs:double",
         "unit": "Nm", "cardinality": "One", "source": "spec.pdf p.2"}
      ]
    }
  ]
}

Rules that matter more than completeness:

1. GROUP related fields under a SubmodelElementCollection. A flat list of forty
   slots is a worse answer than twenty slots in five groups. If several fields
   describe the same subject, they belong together.

2. Use a SubmodelElementList with "typeValueListElement" when the document
   describes a repeating structure -- several axes, several markings, several
   test results. Give it ONE child describing the shape of an entry, not one
   child per occurrence.

3. cardinality is one of: One, ZeroToOne, OneToMany, ZeroToMany. Use One only
   when the document treats the field as always present. When in doubt use
   ZeroToOne -- a template that demands a field nobody can supply is worse than
   one that permits its absence.

4. valueType must be an xsd type: xs:string, xs:double, xs:integer, xs:boolean,
   xs:date, xs:dateTime, xs:anyURI. A number is never xs:string. A measured
   quantity is xs:double even when the example is a whole number.

5. idShort must match [A-Za-z][A-Za-z0-9_]* -- no spaces, no hyphens, no units
   in the name. Write MaxTorque, not "Max Torque [Nm]".

6. Put the unit in "unit", never in the idShort or the value type.

7. Every slot carries "source": where in the document it came from. A slot you
   cannot point at is a slot you invented; leave it out instead.

No preamble, no code fence, no commentary -- the JSON object alone."""


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def safe_id_short(raw: str, fallback: str = "Slot") -> str:
    cleaned = _IDSHORT_SAFE.sub("", (raw or "").strip().replace(" ", ""))
    if not cleaned:
        return fallback
    return cleaned if cleaned[0].isalpha() else f"X{cleaned}"


def parse(text: str, source: str = "") -> Structure:
    """Read the model's answer into a `Structure`, repairing what is repairable.

    Repaired rather than rejected, where the repair is unambiguous: an idShort
    with a space in it, an unknown modelType that is obviously a Property, a
    cardinality spelled `1..1`. Rejecting those would spend a whole extraction
    on a formatting slip. What is *not* repaired is a missing idShort, because
    inventing a name is inventing a slot.
    """
    body = text.strip()
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1)
    start = body.find("{")
    if start > 0:
        body = body[start:]
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise StructureError(f"the model did not return JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise StructureError("expected a JSON object describing one template")

    structure = Structure(
        id_short=safe_id_short(str(raw.get("idShort") or ""), "ProposedTemplate"),
        description=str(raw.get("description") or ""),
        source=source,
        semantic_id=raw.get("semanticId") or None,
    )
    structure.slots = [_slot(s, source) for s in raw.get("slots") or [] if isinstance(s, dict)]
    if not structure.slots:
        raise StructureError("the proposed template has no slots")
    return structure


_CARDINALITY_ALIASES = {
    "1..1": "One", "1": "One", "mandatory": "One", "required": "One",
    "0..1": "ZeroToOne", "optional": "ZeroToOne",
    "1..*": "OneToMany", "1..n": "OneToMany",
    "0..*": "ZeroToMany", "0..n": "ZeroToMany", "many": "ZeroToMany",
}


def _slot(raw: Dict[str, Any], source: str) -> Slot:
    id_short = str(raw.get("idShort") or "").strip()
    if not id_short:
        raise StructureError(f"a slot has no idShort: {json.dumps(raw)[:120]}")

    model_type = str(raw.get("modelType") or "Property")
    if model_type not in ELEMENT_TYPES:
        # A type outside the metamodel is almost always a Property described by
        # a word the model preferred; falling back keeps the slot rather than
        # losing the whole extraction to one bad guess.
        model_type = "Property"

    cardinality = str(raw.get("cardinality") or "One").strip()
    cardinality = _CARDINALITY_ALIASES.get(cardinality.lower(), cardinality)
    if cardinality not in CARDINALITIES:
        cardinality = "ZeroToOne"

    value_type = str(raw.get("valueType") or "xs:string")
    if value_type not in VALUE_TYPES:
        value_type = "xs:string"

    children = [
        _slot(c, source) for c in raw.get("children") or [] if isinstance(c, dict)
    ]
    list_element = raw.get("typeValueListElement")
    if model_type == "SubmodelElementList" and not list_element:
        # A list without a declared element type cannot be created at all --
        # `make_element` refuses it. Infer from the single child the prompt asks
        # for, and fall back to a collection, which is what a repeating group
        # nearly always is.
        list_element = children[0].model_type if children else "SubmodelElementCollection"

    return Slot(
        id_short=safe_id_short(id_short),
        model_type=model_type,
        value_type=value_type,
        cardinality=cardinality,
        unit=str(raw.get("unit") or ""),
        description=str(raw.get("description") or ""),
        semantic_id=raw.get("semanticId") or None,
        source=str(raw.get("source") or source),
        type_value_list_element=str(list_element) if list_element else None,
        children=children,
    )


# ---------------------------------------------------------------------------
# checking a proposal before it is built
# ---------------------------------------------------------------------------

def problems(structure: Structure) -> List[Dict[str, str]]:
    """What would go wrong if this were built, without building it.

    Runs the metamodel constraints `aas validate` applies, at proposal time.
    Finding them here costs nothing; finding them after a template has been
    written costs a repair and an instance built from a broken template.

    One constraint is deliberately *not* checked. AASd-120 forbids an idShort
    on a `SubmodelElementList` child, and a structure file names them anyway --
    "Axis" is how a person reads what an entry is, and dropping it here would
    make the proposal unreviewable to satisfy a rule that binds on the wire.
    `smt author` discards the name when it builds, which is the one place that
    knows the parent.
    """
    found: List[Dict[str, str]] = []
    id_short_re = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

    def check(slot: Slot, prefix: str) -> None:
        path = f"{prefix}.{slot.id_short}" if prefix else slot.id_short
        if not id_short_re.match(slot.id_short):
            found.append({"path": path, "rule": "AASd-002",
                          "detail": f"idShort {slot.id_short!r} is not [A-Za-z][A-Za-z0-9_]*"})
        if slot.model_type == "Property" and slot.value_type not in VALUE_TYPES:
            found.append({"path": path, "rule": "AASd-007",
                          "detail": f"valueType {slot.value_type!r} is not an xsd type"})
        if slot.model_type == "SubmodelElementList":
            declared = slot.type_value_list_element
            for child in slot.children:
                if child.model_type != declared:
                    found.append({"path": path, "rule": "AASd-108",
                                  "detail": f"child is {child.model_type}, "
                                            f"list declares {declared}"})
            if len(slot.children) > 1:
                # Not a metamodel rule. A list child describes the *shape* of an
                # entry, so several of them means the model listed occurrences
                # instead -- which builds a template that permits exactly the
                # entries the document happened to show.
                found.append({"path": path, "rule": "shape",
                              "detail": f"{len(slot.children)} children on a list; "
                                        "a list needs one, describing an entry"})
        if slot.children and slot.model_type not in CONTAINERS:
            found.append({"path": path, "rule": "shape",
                          "detail": f"{slot.model_type} cannot hold children"})

        names = [c.id_short for c in slot.children]
        for name in {n for n in names if names.count(n) > 1}:
            found.append({"path": f"{path}.{name}", "rule": "AASd-022",
                          "detail": "siblings share an idShort"})

        for child in slot.children:
            check(child, path)

    top = [s.id_short for s in structure.slots]
    for name in {n for n in top if top.count(n) > 1}:
        found.append({"path": name, "rule": "AASd-022",
                      "detail": "siblings share an idShort"})
    for slot in structure.slots:
        check(slot, "")
    return found


def render(structure: Structure) -> List[str]:
    """A readable outline, so a person can correct it before it is built."""
    counts = structure.counts()
    lines = [
        f"{structure.id_short}  ({counts['slots']} slots, "
        f"{counts['mandatory']} mandatory, {counts['containers']} containers, "
        f"depth {counts['depth']})",
    ]
    if structure.description:
        lines.append(f"  {structure.description}")
    lines.append("")

    def show(slot: Slot, indent: int) -> None:
        pad = "  " * indent
        detail = slot.model_type
        if slot.model_type == "Property":
            detail = f"Property({slot.value_type.removeprefix('xs:')})"
        elif slot.model_type == "SubmodelElementList":
            detail = f"SML<{slot.type_value_list_element}>"
        unit = f" [{slot.unit}]" if slot.unit else ""
        lines.append(
            f"  {pad}{slot.id_short:<34}{detail:<26}{slot.cardinality:<12}{unit}"
        )
        for child in slot.children:
            show(child, indent + 1)

    for slot in structure.slots:
        show(slot, 0)

    if counts["without_source"]:
        lines += ["", f"warning: {counts['without_source']} slot(s) cite no source; "
                      "a slot that traces to nothing is one somebody invented"]
    return lines

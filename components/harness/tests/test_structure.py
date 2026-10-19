"""Proposing a submodel template from a document, and checking it before it is built.

Everything here runs on hand-written model output, so no request is made and
the repair rules are checkable one at a time. That matters more than usual: the
repairs decide what a slightly-malformed proposal becomes, and a repair that is
too eager invents structure the document never described.

The line the parser holds is: **repair a formatting slip, refuse a missing
fact.** A cardinality written `1..1` is the same statement as `One`, so it is
translated. An absent idShort is not a formatting slip -- naming a slot is
inventing one -- so it raises.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from a4s.structure import (
    Slot,
    Structure,
    StructureError,
    parse,
    problems,
    render,
    safe_id_short,
)


def proposal(**overrides: Any) -> str:
    body: Dict[str, Any] = {
        "idShort": "RobotSpec",
        "description": "a spec",
        "slots": [
            {"idShort": "ManufacturerName", "modelType": "Property",
             "valueType": "xs:string", "cardinality": "One", "source": "p.1"},
        ],
    }
    body.update(overrides)
    return json.dumps(body)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_a_fenced_answer_is_unwrapped() -> None:
    """Models fence JSON despite being told not to, and that is not a defect
    worth failing a whole extraction over."""
    structure = parse("```json\n" + proposal() + "\n```", "doc.pdf")
    assert structure.id_short == "RobotSpec"
    assert structure.slots[0].id_short == "ManufacturerName"


def test_prose_before_the_object_is_skipped() -> None:
    structure = parse("Here is the template:\n" + proposal(), "doc.pdf")
    assert structure.slots


def test_a_slot_without_an_id_short_is_refused_not_named() -> None:
    """The one thing that is not repaired.

    Every other repair here restates something the model said. Inventing a name
    would add a slot to the template that no document described, and on a
    template that error propagates into every instance built from it.
    """
    with pytest.raises(StructureError, match="no idShort"):
        parse(proposal(slots=[{"modelType": "Property", "valueType": "xs:string"}]))


def test_a_proposal_with_no_slots_is_refused() -> None:
    with pytest.raises(StructureError, match="no slots"):
        parse(proposal(slots=[]))


def test_output_that_is_not_json_says_so() -> None:
    with pytest.raises(StructureError, match="did not return JSON"):
        parse("I think the template should have a manufacturer field.")


@pytest.mark.parametrize(
    "written,expected",
    [("1..1", "One"), ("1", "One"), ("mandatory", "One"), ("0..1", "ZeroToOne"),
     ("optional", "ZeroToOne"), ("1..*", "OneToMany"), ("0..*", "ZeroToMany"),
     ("One", "One")],
)
def test_cardinality_spellings_are_translated(written: str, expected: str) -> None:
    """Four dialects exist in the wild and `smt.cardinality` reads all of them.

    A proposal is written by a model that has seen all four, so it will produce
    any of them. Normalising here means the authored template carries one
    spelling -- the current IDTA one -- rather than propagating the dialect
    soup the reader already has to cope with.
    """
    structure = parse(proposal(slots=[
        {"idShort": "X", "cardinality": written},
    ]))
    assert structure.slots[0].cardinality == expected


def test_an_unreadable_cardinality_becomes_optional_not_mandatory() -> None:
    """The safe direction.

    A template demanding a field nobody can supply blocks every instance;
    one permitting an absence merely under-constrains. Guessing wrong towards
    mandatory is the expensive mistake.
    """
    assert parse(proposal(slots=[{"idShort": "X", "cardinality": "sometimes"}])
                 ).slots[0].cardinality == "ZeroToOne"


def test_a_value_type_outside_the_xsd_enumeration_falls_back_to_string() -> None:
    assert parse(proposal(slots=[
        {"idShort": "X", "valueType": "number"}
    ])).slots[0].value_type == "xs:string"


def test_an_unknown_model_type_becomes_a_property() -> None:
    """Almost always a word the model preferred for an ordinary Property.

    Losing the whole extraction to one bad type guess costs more than a slot
    that has to be corrected by hand.
    """
    assert parse(proposal(slots=[
        {"idShort": "X", "modelType": "Number"}
    ])).slots[0].model_type == "Property"


def test_an_id_short_with_spaces_is_cleaned() -> None:
    assert safe_id_short("Max Torque [Nm]") == "MaxTorqueNm"
    assert safe_id_short("2ndAxis").startswith("X")     # must begin with a letter
    assert safe_id_short("") == "Slot"


def test_a_list_without_a_declared_element_type_infers_one() -> None:
    """`make_element` refuses a SubmodelElementList with no
    `typeValueListElement`, so a proposal missing it is unbuildable.

    Inferred from the single child the prompt asks for, and falling back to a
    collection -- which is what a repeating group nearly always is.
    """
    structure = parse(proposal(slots=[
        {"idShort": "Axes", "modelType": "SubmodelElementList",
         "children": [{"idShort": "Axis", "modelType": "SubmodelElementCollection"}]},
    ]))
    assert structure.slots[0].type_value_list_element == "SubmodelElementCollection"

    empty = parse(proposal(slots=[
        {"idShort": "Axes", "modelType": "SubmodelElementList"},
    ]))
    assert empty.slots[0].type_value_list_element == "SubmodelElementCollection"


def test_the_document_source_reaches_every_slot() -> None:
    """A slot that traces to nothing is one somebody invented.

    Inherited from the document when the model omits it per slot, so the
    provenance is at worst coarse rather than absent -- the same choice
    `extract.Extracted` makes.
    """
    structure = parse(proposal(slots=[{"idShort": "X"}]), "datasheet.pdf")
    assert structure.slots[0].source == "datasheet.pdf"


def test_nesting_round_trips_through_as_dict() -> None:
    structure = parse(proposal(slots=[
        {"idShort": "Power", "modelType": "SubmodelElementCollection",
         "children": [{"idShort": "Voltage", "valueType": "xs:double", "unit": "V"}]},
    ]))
    again = parse(json.dumps(structure.as_dict()))
    assert again.slots[0].children[0].unit == "V"
    assert again.slots[0].children[0].value_type == "xs:double"


# ---------------------------------------------------------------------------
# checking a proposal
# ---------------------------------------------------------------------------

def slot(**kw: Any) -> Slot:
    return Slot(**{"id_short": "X", **kw})


def test_a_clean_structure_has_no_problems() -> None:
    structure = Structure(id_short="T", slots=[
        slot(id_short="Name", value_type="xs:string"),
        slot(id_short="Group", model_type="SubmodelElementCollection",
             children=[slot(id_short="Inner", value_type="xs:double")]),
    ])
    assert problems(structure) == []


def test_siblings_sharing_an_id_short_are_caught_before_they_are_built() -> None:
    """AASd-022, at proposal time.

    The same violation caught after building costs a `sm repair` and leaves a
    submodel that no path-addressed read can reach in the meantime.
    """
    structure = Structure(id_short="T", slots=[
        slot(id_short="Twin"), slot(id_short="Twin"),
    ])
    found = problems(structure)
    assert [f["rule"] for f in found] == ["AASd-022"]


def test_a_nested_duplicate_is_caught_too() -> None:
    structure = Structure(id_short="T", slots=[
        slot(id_short="Group", model_type="SubmodelElementCollection",
             children=[slot(id_short="Same"), slot(id_short="Same")]),
    ])
    assert any(f["rule"] == "AASd-022" for f in problems(structure))


def test_a_list_child_of_the_wrong_type_is_caught() -> None:
    structure = Structure(id_short="T", slots=[
        slot(id_short="Axes", model_type="SubmodelElementList",
             type_value_list_element="SubmodelElementCollection",
             children=[slot(id_short="Axis", model_type="Property")]),
    ])
    assert any(f["rule"] == "AASd-108" for f in problems(structure))


def test_several_children_on_a_list_is_reported_as_a_shape_error() -> None:
    """Not a metamodel rule, and the more damaging mistake.

    A list child describes the *shape* of an entry. Several of them means the
    model listed the occurrences the document happened to show, which builds a
    template permitting exactly those and nothing else -- valid AAS, wrong
    template, and no constraint check will ever complain.
    """
    structure = Structure(id_short="T", slots=[
        slot(id_short="Axes", model_type="SubmodelElementList",
             type_value_list_element="SubmodelElementCollection",
             children=[slot(id_short="Axis1", model_type="SubmodelElementCollection"),
                       slot(id_short="Axis2", model_type="SubmodelElementCollection")]),
    ])
    found = [f for f in problems(structure) if f["rule"] == "shape"]
    assert found and "one, describing an entry" in found[0]["detail"]


def test_children_on_a_property_are_reported() -> None:
    structure = Structure(id_short="T", slots=[
        slot(id_short="Leaf", model_type="Property", children=[slot(id_short="Deeper")]),
    ])
    assert any("cannot hold children" in f["detail"] for f in problems(structure))


def test_a_list_child_may_keep_its_name_in_the_proposal() -> None:
    """AASd-120 binds on the wire and not on the document.

    "Axis" is how a person reads what an entry is; dropping it here would make
    the proposal unreviewable to satisfy a rule `smt author` enforces when it
    builds. So this must **not** be reported.
    """
    structure = Structure(id_short="T", slots=[
        slot(id_short="Axes", model_type="SubmodelElementList",
             type_value_list_element="SubmodelElementCollection",
             children=[slot(id_short="Axis", model_type="SubmodelElementCollection")]),
    ])
    assert not any(f["rule"] == "AASd-120" for f in problems(structure))


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def test_the_outline_counts_what_a_reviewer_checks() -> None:
    structure = parse(proposal(slots=[
        {"idShort": "A", "cardinality": "One", "source": "p.1"},
        {"idShort": "B", "cardinality": "ZeroToOne", "source": "p.1",
         "modelType": "SubmodelElementCollection",
         "children": [{"idShort": "C", "unit": "V", "source": "p.2"}]},
    ]))
    counts = structure.counts()
    assert counts == {"slots": 3, "mandatory": 2, "containers": 1, "depth": 2,
                      "with_unit": 1, "with_semantic_id": 0, "without_source": 0}


def test_slots_citing_no_source_are_warned_about() -> None:
    structure = Structure(id_short="T", slots=[slot(id_short="Invented")])
    text = "\n".join(render(structure))
    assert "cite no source" in text

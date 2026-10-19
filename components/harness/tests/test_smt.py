"""Submodel-template instantiation.

The fixtures mirror shapes taken from the reference repository rather than
invented ones: four cardinality dialects coexist there, and the naive
copy-and-flip-kind instantiation that this module replaces left AASd-129
violations in live data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from a4s.aasctl.smt import (
    SmtError,
    cardinality,
    cardinality_label,
    instantiate,
    is_optional,
    outline,
)


def prop(
    id_short: str,
    *,
    card: Optional[str] = None,
    card_type: str = "SMT/Cardinality",
    card_kind: str = "TemplateQualifier",
    example: Optional[str] = None,
    value: Any = "",
    model_type: str = "Property",
) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "modelType": model_type,
        "idShort": id_short,
        "value": value,
        "valueType": "xs:string",
    }
    quals: List[Dict[str, Any]] = []
    if card is not None:
        quals.append({"kind": card_kind, "type": card_type, "value": card, "valueType": "xs:string"})
    if example is not None:
        quals.append({"kind": "ConceptQualifier", "type": "ExampleValue", "value": example})
    if quals:
        node["qualifiers"] = quals
    return node


def template(*elements: Dict[str, Any], kind: str = "Template") -> Dict[str, Any]:
    return {
        "modelType": "Submodel",
        "kind": kind,
        "id": "urn:t:np",
        "idShort": "DigitalNameplate",
        "submodelElements": list(elements),
    }


# ---------------------------------------------------------------------------
# cardinality: four dialects, one meaning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("One", (1, 1)),
        ("ZeroToOne", (0, 1)),
        ("ZeroToMany", (0, None)),
        ("OneToMany", (1, None)),
        ("1", (1, 1)),
        ("0..1", (0, 1)),
        ("0..*", (0, None)),
        ("1..n", (1, None)),
        ("3", (3, 3)),
    ],
)
def test_cardinality_dialects(raw: str, expected: tuple) -> None:
    assert cardinality(prop("X", card=raw)) == expected


@pytest.mark.parametrize("qtype", ["SMT/Cardinality", "Cardinality", "Multiplicity"])
def test_cardinality_read_from_every_qualifier_type(qtype: str) -> None:
    """The same declaration appears under three type names across IDTA vintages."""
    assert is_optional(prop("X", card="ZeroToOne", card_type=qtype))


def test_absent_cardinality_is_mandatory() -> None:
    """An unannotated slot is structure; dropping it would shrink the instance."""
    assert cardinality(prop("X")) == (1, 1)
    assert not is_optional(prop("X"))


def test_unrecognised_cardinality_keeps_the_slot_and_is_reported() -> None:
    result = instantiate(template(prop("X", card="lots")), submodel_id="urn:i")
    assert [e["idShort"] for e in result.submodel["submodelElements"]] == ["X"]
    assert result.unknown_cardinality == ["X: 'lots'"]


# ---------------------------------------------------------------------------
# instantiation
# ---------------------------------------------------------------------------

def test_optional_slots_are_dropped_by_default() -> None:
    result = instantiate(
        template(prop("Serial", card="One"), prop("Operator", card="ZeroToOne")),
        submodel_id="urn:i:np",
        id_short="Nameplate",
    )
    sm = result.submodel
    assert sm["kind"] == "Instance"
    assert sm["id"] == "urn:i:np"
    assert sm["idShort"] == "Nameplate"
    assert [e["idShort"] for e in sm["submodelElements"]] == ["Serial"]
    assert result.dropped_optional == ["Operator"]


def test_include_optional_keeps_them() -> None:
    result = instantiate(
        template(prop("Serial", card="One"), prop("Operator", card="ZeroToOne")),
        submodel_id="urn:i:np",
        include_optional=True,
    )
    assert [e["idShort"] for e in result.submodel["submodelElements"]] == ["Serial", "Operator"]
    assert result.dropped_optional == []


def test_template_qualifiers_are_stripped_for_aasd_129() -> None:
    """AASd-129: a TemplateQualifier may only live inside a Template.

    The tool this replaces flipped ``kind`` without touching qualifiers, and
    the instances it produced are invalid to this day.
    """
    result = instantiate(template(prop("Serial", card="One")), submodel_id="urn:i")
    assert "qualifiers" not in result.submodel["submodelElements"][0]
    assert result.stripped_qualifiers == 1


def test_functional_qualifiers_survive() -> None:
    """``invocationDelegation`` is what makes an Operation callable, not authoring metadata."""
    op = {
        "modelType": "Operation",
        "idShort": "Classify",
        "qualifiers": [
            {"kind": "ConceptQualifier", "type": "invocationDelegation", "value": "http://svc/run"},
            {"kind": "TemplateQualifier", "type": "SMT/Cardinality", "value": "One"},
        ],
    }
    result = instantiate(template(op), submodel_id="urn:i")
    kept = result.submodel["submodelElements"][0]["qualifiers"]
    assert [q["type"] for q in kept] == ["invocationDelegation"]


def test_data_slots_are_emptied() -> None:
    result = instantiate(
        template(
            prop("Serial", card="One", value="TEMPLATE-TEXT"),
            prop("Name", card="One", model_type="MultiLanguageProperty", value=[{"language": "en", "text": "x"}]),
        ),
        submodel_id="urn:i",
    )
    serial, name = result.submodel["submodelElements"]
    assert serial["value"] is None
    assert name["value"] == []


def test_keep_values_opts_out() -> None:
    result = instantiate(
        template(prop("Serial", card="One", value="TEMPLATE-TEXT")),
        submodel_id="urn:i",
        keep_values=True,
    )
    assert result.submodel["submodelElements"][0]["value"] == "TEMPLATE-TEXT"


def test_example_values_are_not_seeded_by_default() -> None:
    """An ExampleValue traces to no source; writing it into a passport fabricates evidence."""
    tmpl = template(prop("Manufacturer", card="One", example="Example Company"))
    default = instantiate(tmpl, submodel_id="urn:i")
    assert default.submodel["submodelElements"][0]["value"] is None
    assert default.seeded == []

    seeded = instantiate(tmpl, submodel_id="urn:i", seed_examples=True)
    assert seeded.submodel["submodelElements"][0]["value"] == "Example Company"
    assert seeded.seeded == ["Manufacturer"]


def test_example_value_into_a_multilanguage_slot_splits_the_lang_tag() -> None:
    tmpl = template(
        prop("Name", card="One", example="Electrical energy accelerator@en",
             model_type="MultiLanguageProperty")
    )
    result = instantiate(tmpl, submodel_id="urn:i", seed_examples=True)
    assert result.submodel["submodelElements"][0]["value"] == [
        {"language": "en", "text": "Electrical energy accelerator"}
    ]


def test_nested_collections_are_transformed_recursively() -> None:
    smc = {
        "modelType": "SubmodelElementCollection",
        "idShort": "Address",
        "qualifiers": [{"kind": "TemplateQualifier", "type": "SMT/Cardinality", "value": "One"}],
        "value": [prop("Street", card="One", value="Musterstr."), prop("Fax", card="ZeroToOne")],
    }
    result = instantiate(template(smc), submodel_id="urn:i")
    address = result.submodel["submodelElements"][0]
    assert [e["idShort"] for e in address["value"]] == ["Street"]
    assert address["value"][0]["value"] is None
    assert "qualifiers" not in address
    assert result.dropped_optional == ["Address.Fax"]
    assert result.kept == ["Address", "Address.Street"]


def test_list_items_are_not_carried_into_the_instance() -> None:
    """A template list holds exemplars; its items are data the caller has yet to add."""
    sml = {
        "modelType": "SubmodelElementList",
        "idShort": "Markings",
        "typeValueListElement": "SubmodelElementCollection",
        "value": [{"modelType": "SubmodelElementCollection", "value": [prop("MarkingName", card="One")]}],
    }
    result = instantiate(template(sml), submodel_id="urn:i")
    markings = result.submodel["submodelElements"][0]
    assert markings["value"] == []
    assert markings["typeValueListElement"] == "SubmodelElementCollection"
    # Dropping an exemplar is not an "optional slot dropped" event.
    assert result.dropped_optional == []


def test_non_template_source_is_refused_unless_forced() -> None:
    instance = template(prop("Serial", card="One"), kind="Instance")
    with pytest.raises(SmtError, match="not 'Template'"):
        instantiate(instance, submodel_id="urn:i")
    forced = instantiate(instance, submodel_id="urn:i", require_template=False)
    assert forced.submodel["kind"] == "Instance"


def test_the_template_is_not_mutated() -> None:
    tmpl = template(prop("Serial", card="One", value="TEMPLATE-TEXT"))
    instantiate(tmpl, submodel_id="urn:i")
    assert tmpl["kind"] == "Template"
    assert tmpl["submodelElements"][0]["value"] == "TEMPLATE-TEXT"
    assert tmpl["submodelElements"][0]["qualifiers"]


# ---------------------------------------------------------------------------
# outline
# ---------------------------------------------------------------------------

def test_outline_labels_cardinality_per_path() -> None:
    smc = {
        "modelType": "SubmodelElementCollection",
        "idShort": "Address",
        "value": [prop("Street", card="One"), prop("Fax", card="ZeroToMany")],
    }
    rows = outline(template(prop("Serial", card="ZeroToOne"), smc))
    assert rows[0].startswith("0..1   Serial")
    assert "Address.Street" in rows[2]
    assert rows[3].startswith("0..*")


def test_cardinality_label() -> None:
    assert cardinality_label(prop("X", card="OneToMany")) == "1..*"


# ---------------------------------------------------------------------------
# MultiLanguageProperty ValueOnly
# ---------------------------------------------------------------------------

def test_lang_strings_use_the_value_only_shape() -> None:
    """`$value` takes `[{"en": "..."}]`; the element form draws a BaSyx 500."""
    from a4s.aasctl.__main__ import _lang_strings

    assert _lang_strings("Fraunhofer IPA@en") == [{"en": "Fraunhofer IPA"}]


def test_lang_strings_split_several_translations() -> None:
    from a4s.aasctl.__main__ import _lang_strings

    assert _lang_strings("Motor@en|Motor@de") == [{"en": "Motor"}, {"de": "Motor"}]


def test_untagged_text_defaults_to_english() -> None:
    from a4s.aasctl.__main__ import _lang_strings

    assert _lang_strings("no tag here") == [{"en": "no tag here"}]


def test_an_at_sign_inside_the_text_is_not_a_language_tag() -> None:
    """`support@example.com` is an address, not a `de`-tagged string."""
    from a4s.aasctl.__main__ import _lang_strings

    assert _lang_strings("mail support@example com") == [
        {"en": "mail support@example com"}
    ]


# ---------------------------------------------------------------------------
# telling a repository URL from an AAS identifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "http://localhost:8081",
        "https://basyx.example.com",
        "http://localhost:8081/",
        "https://basyx.example.com/aasenv",
    ],
)
def test_endpoints_are_accepted(raw: str) -> None:
    from a4s.aasctl.__main__ import _looks_like_endpoint

    assert _looks_like_endpoint(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "https://fraunhoferIPA/ids/aas/CollectionofSMTs",
        "https://admin-shell.io/idta/SubmodelTemplate/DigitalNameplate/2/0",
        "urn:samm:io.admin-shell.idta.batterypass.circularity:1.0.0#Circularity/submodel",
        "CollectionofSMTs",
    ],
)
def test_identifiers_are_rejected(raw: str) -> None:
    """Two live runs passed a shell id to --from-endpoint; both are https:// strings."""
    from a4s.aasctl.__main__ import _looks_like_endpoint

    assert not _looks_like_endpoint(raw)


# ---------------------------------------------------------------------------
# gaps: template vs instance
# ---------------------------------------------------------------------------

def _instance(*elements: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "modelType": "Submodel",
        "kind": "Instance",
        "id": "urn:i",
        "idShort": "Nameplate",
        "submodelElements": list(elements),
    }


def filled(id_short: str, value: Any = "x", model_type: str = "Property") -> Dict[str, Any]:
    return {"modelType": model_type, "idShort": id_short, "value": value}


def test_a_filled_mandatory_slot_is_not_a_gap() -> None:
    from a4s.aasctl.smt import gaps

    assert gaps(template(prop("Serial", card="One")), _instance(filled("Serial", "SN-1"))) == []


def test_an_absent_slot_is_missing_and_an_empty_one_is_empty() -> None:
    """The two need different commands -- `el add` versus `set` -- so they are distinct."""
    from a4s.aasctl.smt import gaps

    found = gaps(
        template(prop("Serial", card="One"), prop("Facility", card="One")),
        _instance(filled("Serial", None)),
    )
    assert [(g.path, g.status) for g in found] == [
        ("Serial", "empty"),
        ("Facility", "missing"),
    ]


def test_optional_slots_are_not_gaps_unless_asked_for() -> None:
    from a4s.aasctl.smt import gaps

    tmpl = template(prop("Serial", card="One"), prop("Operator", card="ZeroToOne"))
    inst = _instance(filled("Serial", "SN-1"))
    assert gaps(tmpl, inst) == []
    assert [g.path for g in gaps(tmpl, inst, include_optional=True)] == ["Operator"]


def test_an_empty_multilanguage_property_is_a_gap() -> None:
    from a4s.aasctl.smt import gaps

    tmpl = template(prop("Name", card="One", model_type="MultiLanguageProperty"))
    assert [g.status for g in gaps(tmpl, _instance(filled("Name", [], "MultiLanguageProperty")))] == [
        "empty"
    ]
    assert gaps(tmpl, _instance(filled("Name", [{"en": "IPA"}], "MultiLanguageProperty"))) == []


def test_false_and_zero_are_answers_not_gaps() -> None:
    """A boolean `false` and a numeric `0` are values; treating them as empty would lie."""
    from a4s.aasctl.smt import gaps

    tmpl = template(prop("Flag", card="One"), prop("Count", card="One"))
    assert gaps(tmpl, _instance(filled("Flag", False), filled("Count", 0))) == []


def test_gaps_recurse_into_collections() -> None:
    from a4s.aasctl.smt import gaps

    tmpl = template(
        {
            "modelType": "SubmodelElementCollection",
            "idShort": "Address",
            "value": [prop("Street", card="One"), prop("City", card="One")],
        }
    )
    inst = _instance(
        {
            "modelType": "SubmodelElementCollection",
            "idShort": "Address",
            "value": [filled("Street", "Musterstr.")],
        }
    )
    assert [(g.path, g.status) for g in gaps(tmpl, inst)] == [("Address.City", "missing")]


def test_an_empty_collection_is_reported_once_not_per_child() -> None:
    """`Address` empty means one instruction to the caller, not two."""
    from a4s.aasctl.smt import gaps

    tmpl = template(
        {
            "modelType": "SubmodelElementCollection",
            "idShort": "Address",
            "value": [prop("Street", card="One")],
        }
    )
    inst = _instance({"modelType": "SubmodelElementCollection", "idShort": "Address", "value": []})
    found = gaps(tmpl, inst)
    assert [(g.path, g.status) for g in found] == [
        ("Address", "empty"),
        ("Address.Street", "missing"),
    ]


def test_gaps_carry_the_semantic_id_for_matching() -> None:
    """`aas match` targets slots by semanticId, so the gap list has to carry it."""
    from a4s.aasctl.smt import gaps

    node = prop("Serial", card="One")
    node["semanticId"] = {"type": "ExternalReference", "keys": [{"value": "0173-1#02-AAM556#004"}]}
    found = gaps(template(node), _instance())
    assert found[0].semantic_id == "0173-1#02-AAM556#004"


# ---------------------------------------------------------------------------
# resolving an idShort to a submodel
# ---------------------------------------------------------------------------

class FakeClient:
    """Enough of BasyxClient to exercise the fallback, and it counts requests."""

    def __init__(self, submodels: List[Dict[str, Any]]) -> None:
        self._by_id = {s["id"]: s for s in submodels}
        self._all = submodels
        self.gets = 0
        self.lists = 0

    def get_submodel(self, submodel_id: str) -> Dict[str, Any]:
        from a4s.aasctl.client import AasHttpError

        self.gets += 1
        if submodel_id not in self._by_id:
            raise AasHttpError(404, "http://x/submodels/y", "does not exist")
        return self._by_id[submodel_id]

    def list_submodels(self, limit: int = 100) -> List[Dict[str, Any]]:
        self.lists += 1
        return self._all


def _sm(sm_id: str, id_short: str) -> Dict[str, Any]:
    return {"modelType": "Submodel", "id": sm_id, "idShort": id_short}


def test_a_full_id_costs_no_extra_request() -> None:
    from a4s.aasctl.__main__ import get_submodel

    client = FakeClient([_sm("urn:a", "Nameplate")])
    assert get_submodel(client, "urn:a")["idShort"] == "Nameplate"
    assert (client.gets, client.lists) == (1, 0)


def test_a_unique_id_short_resolves() -> None:
    from a4s.aasctl.__main__ import get_submodel

    client = FakeClient([_sm("urn:a", "Nameplate"), _sm("urn:b", "TechnicalData")])
    assert get_submodel(client, "Nameplate")["id"] == "urn:a"


def test_an_ambiguous_id_short_is_not_guessed() -> None:
    """15 submodels here are called TechnicalData; writing to the wrong asset is worse
    than an error, so the candidates are listed instead."""
    from a4s.aasctl.__main__ import get_submodel

    client = FakeClient([_sm("urn:a", "TechnicalData"), _sm("urn:b", "TechnicalData")])
    with pytest.raises(SystemExit):
        get_submodel(client, "TechnicalData")


def test_an_unknown_name_fails_with_a_next_command() -> None:
    from a4s.aasctl.__main__ import get_submodel

    with pytest.raises(SystemExit):
        get_submodel(FakeClient([_sm("urn:a", "Nameplate")]), "Absent")


def test_a_non_404_error_is_not_swallowed() -> None:
    """A 500 means the repository is unwell, not that the name was short."""
    from a4s.aasctl.client import AasHttpError
    from a4s.aasctl.__main__ import get_submodel

    class Broken(FakeClient):
        def get_submodel(self, submodel_id: str) -> Dict[str, Any]:
            raise AasHttpError(500, "http://x", "boom")

    with pytest.raises(AasHttpError):
        get_submodel(Broken([]), "anything")

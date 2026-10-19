"""Extraction: reply parsing, tabular mapping, deduplication, unit splitting.

No network and no model. What is asserted is the part that turns a model's
reply, or a spreadsheet, into records `aas match` can consume -- and the two
properties the downstream chain cannot reconstruct if extraction loses them:
the unit as its own field, and where the value came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from a4s.extract import (
    ExtractError,
    Extracted,
    merge,
    parse_entities,
    parse_json_object,
    parse_mapping,
    read_table,
    rows_to_entities,
)


# ---------------------------------------------------------------------------
# reading a model's reply
# ---------------------------------------------------------------------------

def test_a_plain_json_array_is_read() -> None:
    reply = '[{"entity":"Rated voltage","value":"24","unit":"VDC","description":"supply"}]'
    (item,) = parse_entities(reply, "datasheet.pdf p.1")
    assert item.entity == "Rated voltage"
    assert item.value == "24"
    assert item.unit == "VDC"
    assert item.source == "datasheet.pdf p.1"


def test_a_fenced_array_is_read() -> None:
    reply = 'Here you go:\n```json\n[{"entity":"Mass","value":"24.1","unit":"kg"}]\n```'
    (item,) = parse_entities(reply, "x")
    assert item.value == "24.1"


def test_an_array_with_a_preamble_is_read() -> None:
    """A page whose extraction is dropped looks exactly like a page with
    nothing on it, so formatting is not worth failing over."""
    reply = 'I found two entries.\n[{"entity":"A","value":"1"},{"entity":"B","value":"2"}]'
    assert [e.entity for e in parse_entities(reply, "x")] == ["A", "B"]


def test_a_lone_object_is_treated_as_a_one_item_array() -> None:
    assert len(parse_entities('{"entity":"A","value":"1"}', "x")) == 1


def test_an_empty_reply_is_a_failure_not_an_empty_page() -> None:
    """This test used to assert the opposite, and the opposite was the defect.

    Returning `[]` gives "the model answered with nothing" the same meaning as
    "this page is blank". A 6-page ARBURG datasheet then extracted 3 entities,
    all from page 1, and reported success: the five dense pages spent their
    whole budget on reasoning -- `gpt-5` is a reasoning model and
    `max_completion_tokens` covers both -- returned empty, and each empty was
    read as a page with no data on it.

    The old test had no docstring and no reason attached. It recorded what the
    code did rather than what it should do, which is how it came to pin a bug
    in place instead of catching one.
    """
    with pytest.raises(ExtractError, match="returned nothing"):
        parse_entities("", "page 3")

    # And the message has to say which knob to turn, because the reader is
    # looking at a datasheet that came back nearly empty.
    with pytest.raises(ExtractError, match="A4S_EXTRACT_MAX_TOKENS"):
        parse_entities("   ", "page 3")


def test_entries_without_a_name_are_dropped() -> None:
    reply = '[{"value":"1"},{"entity":"","value":"2"},{"entity":"C","value":"3"}]'
    assert [e.entity for e in parse_entities(reply, "x")] == ["C"]


def test_a_reply_with_no_json_is_an_error() -> None:
    with pytest.raises(ExtractError, match="no JSON"):
        parse_entities("I could not read this page.", "x")


def test_the_unit_is_its_own_field() -> None:
    """The source asked for value 'as presented, including units', so `24 VDC`
    arrived as one string and `aas match` had nothing to compare."""
    (item,) = parse_entities(
        '[{"entity":"V","value":"24","unit":"VDC","raw_value":"24 VDC"}]', "x"
    )
    assert (item.value, item.unit, item.raw_value) == ("24", "VDC", "24 VDC")


def test_a_json_object_is_read_for_the_column_mapping() -> None:
    assert parse_json_object('```json\n{"entity":"Name"}\n```') == {"entity": "Name"}


def test_an_array_where_an_object_was_expected_is_an_error() -> None:
    with pytest.raises(ExtractError, match="expected a JSON object"):
        parse_json_object("[1, 2]")


# ---------------------------------------------------------------------------
# the output record
# ---------------------------------------------------------------------------

def test_the_record_always_carries_entity_and_value() -> None:
    """`value: null` is information -- the entity was seen without one."""
    record = Extracted(entity="A").as_dict()
    assert record["entity"] == "A"
    assert record["value"] is None
    assert "unit" not in record        # empty optional fields are omitted


def test_provenance_survives_into_the_record() -> None:
    record = Extracted(entity="A", value="1", source="datasheet.pdf p.3").as_dict()
    assert record["source"] == "datasheet.pdf p.3"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def test_a_csv_is_read_with_its_headers(tmp_path: Path) -> None:
    path = tmp_path / "specs.csv"
    path.write_text("Property,Spec,Unit\nRated capacity,68,Ah\n", encoding="utf-8")
    assert read_table(path) == [{"Property": "Rated capacity", "Spec": "68", "Unit": "Ah"}]


def test_a_json_list_is_read(tmp_path: Path) -> None:
    path = tmp_path / "specs.json"
    path.write_text('[{"a": 1}]', encoding="utf-8")
    assert read_table(path) == [{"a": 1}]


def test_jsonl_is_read_line_by_line(tmp_path: Path) -> None:
    path = tmp_path / "specs.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_table(path) == [{"a": 1}, {"a": 2}]


def test_an_unreadable_extension_says_what_it_accepts(tmp_path: Path) -> None:
    path = tmp_path / "specs.docx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ExtractError, match=r"\.csv, \.json"):
        read_table(path)


def test_an_explicit_column_mapping_is_parsed() -> None:
    assert parse_mapping("entity=Property,value=Spec,unit=Unit") == {
        "entity": "Property",
        "value": "Spec",
        "unit": "Unit",
    }


def test_a_mapping_must_name_the_entity_column() -> None:
    with pytest.raises(ExtractError, match="'entity' column"):
        parse_mapping("value=Spec")


def test_a_malformed_mapping_is_refused() -> None:
    with pytest.raises(ExtractError, match="field=column"):
        parse_mapping("entity")


def test_rows_become_entities_with_their_row_number() -> None:
    """Row 1 is the header, so data starts at 2 -- which is what a person
    reading the spreadsheet sees."""
    rows = [{"P": "Rated capacity", "S": "68", "U": "Ah"}, {"P": "Mass", "S": "24", "U": "kg"}]
    out = rows_to_entities(rows, {"entity": "P", "value": "S", "unit": "U"}, "specs.csv")
    assert [(e.entity, e.value, e.unit) for e in out] == [
        ("Rated capacity", "68", "Ah"),
        ("Mass", "24", "kg"),
    ]
    assert out[0].source == "specs.csv row 2"
    assert out[1].source == "specs.csv row 3"


@pytest.mark.parametrize("blank", ["", "  ", "nan", "NaN", "None"])
def test_rows_with_no_entity_name_are_skipped(blank: str) -> None:
    """Spreadsheets carry blank separator rows and pandas-style NaN."""
    rows = [{"P": blank, "S": "1"}, {"P": "Real", "S": "2"}]
    out = rows_to_entities(rows, {"entity": "P", "value": "S"}, "x.csv")
    assert [e.entity for e in out] == ["Real"]


def test_a_mapping_without_a_unit_column_is_fine() -> None:
    out = rows_to_entities([{"P": "A", "S": "1"}], {"entity": "P", "value": "S"}, "x.csv")
    assert out[0].unit == ""


# ---------------------------------------------------------------------------
# merging pages
# ---------------------------------------------------------------------------

def test_the_same_fact_on_two_pages_is_one_entity() -> None:
    """A spec in a summary table and again in a detail table is one fact;
    sending it twice means writing the slot twice and arbitrating it twice."""
    out = merge(
        [
            [Extracted(entity="Rated capacity", value="68", source="d.pdf p.1")],
            [Extracted(entity="Rated capacity", value="68", source="d.pdf p.4")],
        ]
    )
    assert len(out) == 1
    assert out[0].source == "d.pdf p.1; d.pdf p.4"


def test_the_same_name_with_a_different_value_is_kept_apart() -> None:
    """That is a discrepancy in the document. Hiding it is worse than the
    duplicate."""
    out = merge(
        [
            [Extracted(entity="Mass", value="24.1", source="p.1")],
            [Extracted(entity="Mass", value="25.0", source="p.4")],
        ]
    )
    assert [e.value for e in out] == ["24.1", "25.0"]


def test_deduplication_ignores_case_and_surrounding_space() -> None:
    out = merge(
        [
            [Extracted(entity="Rated Capacity", value="68")],
            [Extracted(entity=" rated capacity ", value=" 68 ")],
        ]
    )
    assert len(out) == 1


def test_merge_preserves_the_order_pages_were_read_in() -> None:
    out = merge([[Extracted(entity="A"), Extracted(entity="B")], [Extracted(entity="C")]])
    assert [e.entity for e in out] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# the handover to `aas match`
# ---------------------------------------------------------------------------

def test_extracted_records_load_as_match_entities(tmp_path: Path) -> None:
    """The two halves are separate commands, so the file format between them
    is the contract."""
    from a4s.aasctl.match import load_entities

    records = [
        Extracted(
            entity="Rated capacity", value="68", unit="Ah",
            description="nominal capacity", source="datasheet.pdf p.3",
        ).as_dict()
    ]
    path = tmp_path / "entities.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    (loaded,) = load_entities(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.entity == "Rated capacity"
    assert loaded.value == "68"
    assert loaded.unit == "Ah"
    assert loaded.source == "datasheet.pdf p.3"


class _Reply:
    """A completion, with the field that distinguishes the two failures."""

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason


def test_a_truncated_answer_is_reported_not_parsed() -> None:
    """`finish_reason == "length"` is the only thing that separates "the page
    held nothing" from "the answer was cut off", and they need different fixes
    from whoever reads the message.

    Handing half an array to `parse_entities` instead would report a JSON
    syntax error, which tells the reader nothing about the budget.
    """
    from a4s.extract import _answered

    with pytest.raises(ExtractError, match="before writing anything"):
        _answered(_Reply("", "length"), "page 2")

    with pytest.raises(ExtractError, match="truncated"):
        _answered(_Reply('[{"entity": "Clamping force", "val', "length"), "page 2")

    assert _answered(_Reply("[]"), "page 2") == "[]"


def test_the_extraction_budget_does_not_leak_into_the_agent_loop() -> None:
    """The same `ModelConfig` object belongs to the loop.

    Raising its ceiling in place would silently change every later tool call in
    the process -- action at a distance that makes a run irreproducible from
    the ledger, which is the one artefact meant to explain it.
    """
    from a4s.extract import EXTRACTION_MAX_TOKENS, _budgeted

    class Config:
        max_tokens = 4096

    original = Config()
    budgeted = _budgeted(original)

    assert budgeted.max_tokens == EXTRACTION_MAX_TOKENS
    assert original.max_tokens == 4096, "the caller's config was mutated"
    assert EXTRACTION_MAX_TOKENS > 4096, "the budget must exceed the loop's"


def test_a_config_already_generous_enough_is_left_alone() -> None:
    from a4s.extract import EXTRACTION_MAX_TOKENS, _budgeted

    class Config:
        max_tokens = EXTRACTION_MAX_TOKENS * 2

    config = Config()
    assert _budgeted(config) is config


# ---------------------------------------------------------------------------
# configurations, and what must never be merged
# ---------------------------------------------------------------------------

def test_a_property_stated_twice_under_two_names_is_one_fact() -> None:
    """`Clamping force` on the summary page and `Clamping force max.` in the
    detail table, both 1500 kN. A live run wrote the slot twice."""
    from a4s.extract import merge_reporting

    kept, notes = merge_reporting([[
        Extracted(entity="Clamping force", value="1500", unit="kN", source="p.1"),
        Extracted(entity="Clamping force max.", value="1500", unit="kN", source="p.3"),
    ]])
    assert len(kept) == 1
    assert kept[0].source == "p.1; p.3"
    assert any("same fact" in n for n in notes)


def test_the_same_number_for_two_properties_is_never_merged() -> None:
    """Both 390 in one real document, and nothing to do with each other.
    Merging by value would delete a fact and report success."""
    from a4s.extract import merge_reporting

    kept, _ = merge_reporting([[
        Extracted(entity="Vertical dimension at base", value="390", source="p.2"),
        Extracted(entity="Robotic system mounting | C", value="390", source="p.4"),
    ]])
    assert len(kept) == 2


def test_two_configurations_are_two_facts_even_at_the_same_value() -> None:
    from a4s.extract import merge_reporting

    kept, _ = merge_reporting([[
        Extracted(entity="Screw diameter", value="35", unit="mm", variant="IU 290"),
        Extracted(entity="Screw diameter", value="35", unit="mm", variant="IU 400"),
    ]])
    assert len(kept) == 2


def test_a_configuration_in_brackets_still_separates_the_facts() -> None:
    """Documents extracted before `variant` existed put the configuration in
    the name, and the bracket-stripping that collapses `(w x h)` would collapse
    these too."""
    from a4s.extract import merge_reporting

    kept, _ = merge_reporting([[
        Extracted(entity="Screw diameter (Injection unit 290)", value="35", unit="mm"),
        Extracted(entity="Screw diameter (Injection unit 400)", value="35", unit="mm"),
    ]])
    assert len(kept) == 2


def test_a_bracketed_clarification_still_collapses() -> None:
    from a4s.extract import merge_reporting

    kept, _ = merge_reporting([[
        Extracted(entity="Distance between tie bars", value="520 x 520", unit="mm", source="p.1"),
        Extracted(entity="Distance between tie bars (w x h)", value="520 x 520",
                  unit="mm", source="p.3"),
    ]])
    assert len(kept) == 1


def test_two_values_for_one_property_are_reported_not_resolved() -> None:
    """A document contradicting itself is something a person needs to see."""
    from a4s.extract import merge_reporting

    kept, notes = merge_reporting([[
        Extracted(entity="Overall height", value="2225", unit="mm", source="p.2"),
        Extracted(entity="Overall height max.", value="2400", unit="mm", source="p.5"),
    ]])
    assert len(kept) == 2
    assert any("more than one value" in n for n in notes)


def test_choosing_a_configuration_keeps_the_values_that_hold_for_all() -> None:
    from a4s.extract import for_configuration

    entities = [
        Extracted(entity="Manufacturer", value="ARBURG"),
        Extracted(entity="Screw diameter", value="35", variant="IU 290"),
        Extracted(entity="Screw diameter", value="40", variant="IU 400"),
    ]
    kept, notes = for_configuration(entities, "IU 290")
    assert [e.value for e in kept] == ["ARBURG", "35"]
    assert any("IU 400" in n for n in notes)


def test_what_a_configuration_choice_discards_is_reported() -> None:
    from a4s.extract import for_configuration

    _, notes = for_configuration(
        [Extracted(entity="A", value="1", variant="X"),
         Extracted(entity="A", value="2", variant="Y")],
        "X",
    )
    assert notes and "1 values describe other configurations" in notes[0]


def test_configurations_are_read_off_the_entities_in_order() -> None:
    from a4s.extract import configurations_of

    assert configurations_of([
        Extracted(entity="A", value="1", variant="IU 400"),
        Extracted(entity="B", value="2"),
        Extracted(entity="C", value="3", variant="IU 290"),
        Extracted(entity="D", value="4", variant="IU 400"),
    ]) == ["IU 400", "IU 290"]


def test_an_extraction_carries_what_the_page_said_about_itself() -> None:
    from a4s.extract import parse_extraction

    found = parse_extraction(
        '{"entities": [{"entity": "Screw diameter", "value": "35", '
        '"variant": "IU 290"}], "configurations": ["IU 290", "IU 400"], '
        '"warnings": ["column 3 could not be attributed"]}',
        "p.3",
    )
    assert found.configurations == ["IU 290", "IU 400"]
    assert found.warnings == ["column 3 could not be attributed"]
    assert found.entities[0].variant == "IU 290"


def test_a_bare_array_is_still_a_valid_answer() -> None:
    """Older prompts asked for one, and models still return one."""
    from a4s.extract import parse_extraction

    found = parse_extraction('[{"entity": "Weight", "value": "11"}]', "p.1")
    assert len(found.entities) == 1
    assert found.configurations == [] and found.warnings == []


def test_a_variant_on_a_row_counts_as_a_configuration_of_the_page() -> None:
    """The list is the model's summary; the rows are the data."""
    from a4s.extract import parse_extraction

    found = parse_extraction(
        '{"entities": [{"entity": "A", "value": "1", "variant": "IU 290"}], '
        '"configurations": []}',
        "p.1",
    )
    assert found.configurations == ["IU 290"]

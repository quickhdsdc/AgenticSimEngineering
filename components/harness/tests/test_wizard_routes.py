"""The wizard's file store, without a repository.

Most of this layer is filesystem and translation, so it is tested at unit
level: the round trip that has to hold is plan -> table row -> edited row ->
plan, and none of it needs BaSyx. What does need one -- resolving a row the
user retargeted into a different submodel -- is in
`tests/integration/test_server_wizard.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="the `ui` extra is not installed"
)

from a4s.server import workspace  # noqa: E402
from a4s.server.wizard_routes import _decision_of, _row_of, _target_of  # noqa: E402

# Nothing here talks to a repository. The address is one that refuses
# immediately, so a route that reached for BaSyx would fail loudly rather than
# hang -- which is the point of pointing it somewhere dead rather than nowhere.
DEAD_ENDPOINT = "http://127.0.0.1:9"


@pytest.fixture
def api(tmp_path: Path, monkeypatch) -> Any:
    monkeypatch.setenv("A4S_WEB_WORKSPACE", str(tmp_path))
    from a4s.server import build_app

    with fastapi_testclient.TestClient(
        build_app(DEAD_ENDPOINT), raise_server_exceptions=False
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# stage 1 -- upload and preview
# ---------------------------------------------------------------------------

def test_an_upload_returns_the_path_the_agent_will_open(api: Any) -> None:
    """`path` goes straight into the prompt the frontend sends the agent, and
    the agent runs on this host. A URL would only have to be fetched back
    through this same server."""
    body = api.post(
        "/api/upload",
        files={"file": ("datasheet.pdf", b"%PDF-1.4 fake", "application/pdf")},
    ).json()

    assert body["success"] is True
    assert body["filename"] == "datasheet.pdf"
    assert body["size"] == len(b"%PDF-1.4 fake")
    assert Path(body["path"]).read_bytes() == b"%PDF-1.4 fake"


def test_an_upload_named_to_escape_the_workspace_is_refused(api: Any) -> None:
    """The filename comes from the client and is the only thing deciding where
    the bytes land."""
    response = api.post(
        "/api/upload",
        files={"file": ("../../evil.pdf", b"x", "application/pdf")},
    )

    assert response.status_code == 400
    assert "not a plain filename" in response.json()["detail"]


def test_previewing_a_table_uses_the_same_reader_the_extractor_will(
    api: Any, tmp_path: Path
) -> None:
    """A preview that disagreed with what the extraction later reads would put
    the wrong column names in the prompt, and the model would map against
    headers that are not there."""
    api.post("/api/upload", files={"file": (
        "parts.csv", b"Name,Spec,Unit\nWeight,12,kg\nLength,3,m\n", "text/csv"
    )})

    body = api.get("/api/preview-structured-file/parts.csv").json()

    assert body["headers"] == ["Name", "Spec", "Unit"]
    assert body["rowCount"] == 2
    assert body["rows"][0]["Name"] == "Weight"


def test_previewing_something_that_is_not_a_table_says_so(api: Any) -> None:
    api.post("/api/upload", files={"file": ("notes.pdf", b"%PDF", "application/pdf")})

    response = api.get("/api/preview-structured-file/notes.pdf")

    assert response.status_code == 422
    assert "cannot read" in response.json()["detail"]


# ---------------------------------------------------------------------------
# stage 2 -- extraction
# ---------------------------------------------------------------------------

def test_a_missing_extraction_is_a_200_saying_no(api: Any) -> None:
    """The wizard branches on this. A 404 would show an error for the ordinary
    case of a document nobody has extracted yet."""
    body = api.get("/api/check-extraction/datasheet.pdf").json()

    assert body["exists"] is False
    assert body["filename"] is None


def test_the_miss_still_says_where_the_extraction_should_go(api: Any) -> None:
    """The wizard puts this path into the prompt it sends the agent.

    Without it the agent picks its own filename, `check-extraction` keeps
    answering no, and the wizard waits at stage 1 for something that has
    already been written. The naming convention stays in one place instead of
    being reimplemented in TypeScript.
    """
    body = api.get("/api/check-extraction/datasheet.pdf").json()

    assert body["expectedPath"].endswith("datasheet.json")
    assert "extractions" in body["expectedPath"]


def test_the_corrected_entities_win_over_the_extractors_own(
    api: Any, tmp_path: Path
) -> None:
    """Handing back the extractor's originals after somebody fixed them would
    discard the correction and read as a failed save."""
    workspace.extraction_file("datasheet.pdf").write_text(
        json.dumps([{"entity": "Weight", "value": "11"}]), encoding="utf-8"
    )
    workspace.extraction_file("datasheet.pdf", edited=True).write_text(
        json.dumps([{"entity": "Weight", "value": "12"}]), encoding="utf-8"
    )

    check = api.get("/api/check-extraction/datasheet.pdf").json()
    entities = api.get("/api/extracted-entities/datasheet.pdf").json()

    assert check["edited"] is True
    assert entities["entities"][0]["value"] == "12"


@pytest.mark.parametrize("payload", [
    [{"entity": "Weight", "value": "12"}],
    {"entities": [{"entity": "Weight", "value": "12"}]},
    {"result": [{"entity": "Weight", "value": "12"}]},
])
def test_every_shape_the_extractor_emits_is_accepted(
    api: Any, payload: Any
) -> None:
    """`aas match` accepts all three (`match.load_entities`). A route stricter
    than the tool it feeds would reject files the pipeline can use."""
    workspace.extraction_file("d.pdf").write_text(json.dumps(payload), encoding="utf-8")

    body = api.get("/api/extracted-entities/d.pdf").json()

    assert body["count"] == 1
    assert body["entities"][0]["entity"] == "Weight"


def test_saving_corrections_writes_beside_the_original_not_over_it(
    api: Any
) -> None:
    """The extractor's file is evidence of what the model read; the edit is
    evidence of what a person disagreed with. Overwriting makes the second
    unreviewable."""
    original = workspace.extraction_file("datasheet.pdf")
    original.write_text(json.dumps([{"entity": "Weight", "value": "11"}]), encoding="utf-8")

    body = api.post("/api/save-edited-entities", json={
        "original_filename": "datasheet.pdf",
        "entities": [
            {"entity": "Weight", "value": "12"},
            {"entity": "Colour", "value": "red", "selected": False},
        ],
    }).json()

    assert body["count"] == 1 and body["dropped"] == 1
    assert json.loads(original.read_text(encoding="utf-8"))[0]["value"] == "11"
    # The edit is written in the object shape, so what the extraction said
    # about the document -- its configurations, its warnings -- survives being
    # corrected. Both shapes are read back.
    edited = json.loads(Path(body["path"]).read_text(encoding="utf-8"))
    assert edited["entities"][0]["value"] == "12"


def test_a_chosen_configuration_is_recorded_with_what_it_cost(api: Any) -> None:
    """Dropping values is the point of choosing a configuration, and a drop
    nobody is told about is the failure the whole feature exists to prevent."""
    workspace.extraction_file("datasheet.pdf").write_text(
        json.dumps({"entities": [{"entity": "Weight", "value": "11"}],
                    "configurations": ["IU 290", "IU 400"]}),
        encoding="utf-8",
    )

    body = api.post("/api/save-edited-entities", json={
        "original_filename": "datasheet.pdf",
        "configuration": "IU 290",
        "configurations": ["IU 290", "IU 400"],
        "entities": [
            {"entity": "Screw diameter", "value": "35", "variant": "IU 290"},
            {"entity": "Screw diameter", "value": "40", "variant": "IU 400",
             "selected": False},
        ],
    }).json()

    assert body["configuration"] == "IU 290"
    saved = json.loads(Path(body["path"]).read_text(encoding="utf-8"))
    assert saved["configuration"] == "IU 290"
    assert saved["configurations"] == ["IU 290", "IU 400"]
    assert any("IU 290" in w and "not kept" in w for w in saved["warnings"])


def test_an_extraction_reports_what_it_could_not_do_cleanly(api: Any) -> None:
    workspace.extraction_file("datasheet.pdf").write_text(
        json.dumps({
            "entities": [{"entity": "Weight", "value": "11", "variant": "IU 290"}],
            "warnings": ["column 3 could not be attributed to a property"],
        }),
        encoding="utf-8",
    )

    body = api.get("/api/extracted-entities/datasheet.pdf").json()
    assert body["warnings"] == ["column 3 could not be attributed to a property"]
    # Derived from the rows: a file written before `configurations` existed
    # still distinguishes them if its entities carry a variant.
    assert body["configurations"] == ["IU 290"]


# ---------------------------------------------------------------------------
# stage 3 -- the plan round trip
# ---------------------------------------------------------------------------

def _plan_file(submodel_id: str = "urn:sm:np") -> dict:
    return {
        "submodel": submodel_id,
        "thresholds": {"high": 0.8, "low": 0.3},
        "arbitration_model": "gpt-x",
        "counts": {},
        "decisions": [{
            "entity": "Rated voltage", "value": "24", "unit": "V", "source": "p.2",
            "action": "write", "band": "auto", "reason": "score above theta_high",
            "path": "TechnicalData.RatedVoltage", "model_type": "Property",
            "value_type": "xs:int", "id_short": None, "score": 0.83,
            "margin": 0.12, "warnings": [], "candidates": [],
        }],
    }


def test_a_plan_renders_as_rows_that_carry_what_the_plan_needs_back(
    api: Any
) -> None:
    """The fields the plan needs travel through the browser as extra keys.

    Vue's object spread preserves them, which is what lets the round trip work
    without touching the components. Dropping one means a value written to the
    wrong element, or not written at all.
    """
    workspace.match_file("datasheet.pdf", "AGV").write_text(
        json.dumps(_plan_file()), encoding="utf-8"
    )

    body = api.get("/api/matching-results/datasheet.pdf",
                   params={"aas_id_short": "AGV"}).json()

    (row,) = body["results"]
    assert row["extractedEntity"] == "Rated voltage"
    assert row["aasProperty"] == "RatedVoltage"
    assert row["apiPath"] == "TechnicalData.RatedVoltage"
    assert row["selected"] is True
    # carried, not drawn
    assert row["submodelId"] == "urn:sm:np"
    assert row["action"] == "write"
    assert row["valueType"] == "xs:int"
    assert row["unit"] == "V"


def test_a_skipped_decision_arrives_unticked(api: Any) -> None:
    plan = _plan_file()
    plan["decisions"][0]["action"] = "skip"
    workspace.match_file("d.pdf", "AGV").write_text(json.dumps(plan), encoding="utf-8")

    body = api.get("/api/matching-results/d.pdf", params={"aas_id_short": "AGV"}).json()

    assert body["results"][0]["selected"] is False


def test_a_missing_plan_names_the_file_it_looked_for(api: Any) -> None:
    response = api.get("/api/matching-results/nothing.pdf",
                       params={"aas_id_short": "AGV"})

    assert response.status_code == 404
    assert "nothing_AGV_matching_result.json" in response.json()["detail"]


def test_saved_rows_come_back_as_a_plan_populate_can_read(api: Any) -> None:
    """The output of this route is the input to `aas populate`. If it is not a
    plan, the last stage of the wizard has nothing to apply."""
    from a4s.aasctl.match import Plan

    body = api.post("/api/save-matching-results", json={
        "filename": "datasheet_AGV",
        "results": [{
            "extractedEntity": "Rated voltage", "extractedValue": "24",
            "aasProperty": "RatedVoltage", "aasType": "Property",
            "apiPath": "TechnicalData.RatedVoltage", "submodelId": "urn:sm:np",
            "unit": "V", "valueType": "xs:int", "confidence": 0.83,
            "selected": True,
        }],
    }).json()

    assert body["success"] is True and body["unresolved"] == []
    (entry,) = body["plans"]

    plan = Plan.from_json(json.loads(Path(entry["path"]).read_text(encoding="utf-8")))
    assert plan.submodel_id == "urn:sm:np"
    (decision,) = plan.decisions
    assert decision.action == "write"
    assert decision.path == "TechnicalData.RatedVoltage"
    assert decision.value == "24"


def test_the_manifest_is_what_the_path_points_at(api: Any) -> None:
    """A table can span submodels and a `Plan` names exactly one, so several
    plans come out. The caller gets one path and should not have to guess the
    rest of the names."""
    rows = [
        {"extractedEntity": "Voltage", "extractedValue": "24", "aasType": "Property",
         "apiPath": "Data.Voltage", "submodelId": "urn:sm:a"},
        {"extractedEntity": "Weight", "extractedValue": "12", "aasType": "Property",
         "apiPath": "Data.Weight", "submodelId": "urn:sm:b"},
    ]

    body = api.post("/api/save-matching-results",
                    json={"filename": "d_AGV", "results": rows}).json()

    manifest = json.loads(Path(body["path"]).read_text(encoding="utf-8"))
    assert manifest["kind"] == "match-plan-manifest"
    assert {p["submodel"] for p in manifest["plans"]} == {"urn:sm:a", "urn:sm:b"}


def test_a_row_with_no_target_is_reported_not_dropped(api: Any) -> None:
    """Silently discarding it makes the generation stage write fewer values
    than the table showed, with nothing on screen saying which."""
    body = api.post("/api/save-matching-results", json={
        "filename": "d_AGV",
        "results": [{"extractedEntity": "Colour", "extractedValue": "red"}],
    }).json()

    assert body["plans"] == []
    assert body["unresolved"][0]["entity"] == "Colour"


# ---------------------------------------------------------------------------
# the translation itself
# ---------------------------------------------------------------------------

class _FixedLookup:
    def resolve(self, row, target):  # noqa: ANN001 - test double
        return row.get("submodelId") or "urn:sm:np"


def test_choosing_a_collection_means_create_a_new_element_in_it() -> None:
    """The matching table lets a collection be picked on purpose. It does not
    mean "write into the collection", which is not a thing a collection can
    hold -- it means the value needs an element that does not exist yet, which
    is `aas populate`'s create path."""
    submodel_id, decision = _decision_of({
        "extractedEntity": "Cycle count",
        "extractedValue": "12000",
        "aasType": "SubmodelElementCollection",
        "apiPath": "TechnicalData",
        "submodelId": "urn:sm:np",
    }, _FixedLookup())

    assert submodel_id == "urn:sm:np"
    assert decision.action == "create"
    assert decision.path == "TechnicalData"      # the parent it goes into
    assert decision.id_short == "Cycle_count"    # the element that gets made
    assert decision.value_type == "xs:integer"  # inferred, not carried


def test_the_property_picker_beats_the_matchers_suggestion() -> None:
    """`aasProperties` is written when somebody clicks a property. The flat
    fields are the matcher's original guess, and after an edit they are
    stale -- preferring them writes to the element the user rejected."""
    target = _target_of({
        "apiPath": "TechnicalData.Guessed",
        "aasType": "Property",
        "aasProperties": [{"idShort": "Chosen", "type": "Property",
                           "apiPath": "TechnicalData.Chosen",
                           "semanticPath": "Nameplate/TechnicalData.Chosen"}],
    })

    assert target["apiPath"] == "TechnicalData.Chosen"


def test_a_create_row_prints_the_element_being_made_not_its_parent() -> None:
    """Printing the collection would tell the user their value is going
    somewhere it is not."""
    from a4s.aasctl.match import Decision

    row = _row_of(
        Decision(entity="Cycle count", value="12000", unit="", source="",
                 action="create", band="arbitrated", reason="chosen",
                 path="TechnicalData", id_short="Cycle_count",
                 model_type="Property"),
        "urn:sm:np", "Nameplate",
    )

    assert row["aasProperty"] == "Cycle_count"
    assert row["apiPath"] == "TechnicalData"
    assert row["semanticPath"] == "Nameplate/TechnicalData"


def test_a_hosted_create_survives_the_round_trip_through_the_table() -> None:
    """The defect that lost three values in a live run.

    `_row_of` put the *new element's* type in `aasType` -- `Property`, because
    that is what gets created -- and `_decision_of` read `aasType` as the type
    of the thing at `apiPath`. A Property is not a container, so the create
    became a write onto its own host collection: `band: "hosted"`,
    `action: "write"`, and an HTTP 500 that named no reason. The value was not
    misplaced, it was gone.
    """
    from a4s.aasctl.match import Decision

    original = Decision(
        entity="Nozzle retraction stroke", value="300", unit="mm",
        source="datasheet p.3", action="create", band="hosted",
        reason="belongs under the nozzle group",
        path="TechnicalProperties.CoreMachine.InjectionUnitTechnicalProperties.Nozzle",
        model_type="Property", container_type="SubmodelElementCollection",
        id_short="Nozzle_retraction_stroke", value_type="xs:integer",
    )

    row = _row_of(original, "urn:sm:td", "TechnicalData")
    assert row["aasType"] == "Property"                       # what gets made
    assert row["targetType"] == "SubmodelElementCollection"   # what it goes in

    _, back = _decision_of(row, _FixedLookup())
    assert back.action == "create"
    assert back.path == original.path
    assert back.id_short == "Nozzle_retraction_stroke"
    assert back.model_type == "Property"


def test_a_write_row_still_round_trips_as_a_write() -> None:
    from a4s.aasctl.match import Decision

    row = _row_of(
        Decision(entity="Clamping force", value="1500", unit="kN", source="p.1",
                 action="write", band="auto", reason="similarity 0.81 >= 0.8",
                 path="TechnicalProperties.MaxClampingForce",
                 model_type="Property", value_type="xs:integer"),
        "urn:sm:td", "TechnicalData",
    )
    assert row["targetType"] == "Property"

    _, back = _decision_of(row, _FixedLookup())
    assert back.action == "write"
    assert back.path == "TechnicalProperties.MaxClampingForce"
    assert back.id_short is None


def test_repointing_a_row_at_a_collection_turns_it_into_a_create() -> None:
    """The user's click outranks the matcher's verdict, in both directions."""
    _, decision = _decision_of({
        "extractedEntity": "Cycle count",
        "extractedValue": "12000",
        "action": "write",
        "apiPath": "TechnicalData.Guessed",
        "targetType": "Property",
        "aasProperties": [{"idShort": "Group", "type": "SubmodelElementCollection",
                           "apiPath": "TechnicalData.Group",
                           "semanticPath": "TechnicalData/TechnicalData.Group"}],
        "submodelId": "urn:sm:np",
    }, _FixedLookup())

    assert decision.action == "create"
    assert decision.path == "TechnicalData.Group"
    assert decision.id_short == "Cycle_count"


def test_repointing_a_row_at_a_property_turns_a_create_into_a_write() -> None:
    _, decision = _decision_of({
        "extractedEntity": "Cycle count",
        "extractedValue": "12000",
        "action": "create",
        "apiPath": "TechnicalData",
        "targetType": "SubmodelElementCollection",
        "aasProperties": [{"idShort": "Cycles", "type": "Property",
                           "apiPath": "TechnicalData.Cycles",
                           "semanticPath": "TechnicalData/TechnicalData.Cycles"}],
        "submodelId": "urn:sm:np",
    }, _FixedLookup())

    assert decision.action == "write"
    assert decision.path == "TechnicalData.Cycles"


def test_a_create_through_the_table_is_typed_from_the_value() -> None:
    """Not left as whatever `aasType` held: the row may have been re-pointed
    since the matcher typed it."""
    _, decision = _decision_of({
        "extractedEntity": "Operating temperature",
        "extractedValue": "-20...60",
        "unit": "°C",
        "action": "create",
        "apiPath": "TechnicalData.Group",
        "targetType": "SubmodelElementCollection",
        "submodelId": "urn:sm:np",
    }, _FixedLookup())

    assert decision.action == "create"
    assert decision.model_type == "Range"
    assert (decision.min_value, decision.max_value) == ("-20", "60")


def test_a_create_into_a_list_takes_the_lists_child_type_and_no_id_short() -> None:
    _, decision = _decision_of({
        "extractedEntity": "Marking",
        "extractedValue": "CE",
        "action": "create",
        "apiPath": "Markings",
        "targetType": "SubmodelElementList",
        "typeValueListElement": "MultiLanguageProperty",
        "submodelId": "urn:sm:np",
    }, _FixedLookup())

    assert decision.model_type == "MultiLanguageProperty"
    assert decision.id_short is None

"""The HTTP surface, against the live repository.

Driven through FastAPI's `TestClient` rather than a running uvicorn: the
routing, validation and serialisation are the same code path, and a test that
has to start and stop a server is a test people skip.

What these assert is the **contract the Vue frontend already depends on** --
field names, response shape, status codes. That contract was fixed by the
predecessor's API, and the point of this server is to satisfy it from the
harness's own library rather than to redesign it. So an assertion here failing
means the UI breaks, which is the only thing worth testing at this layer.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from a4s.aasctl.client import BasyxClient

from .conftest import run_aas

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api(endpoint: str):
    """A client bound to the live repository.

    Module-scoped: building the app imports FastAPI and assembles the router,
    which is the same for every test here.
    """
    fastapi_testclient = pytest.importorskip(
        "fastapi.testclient", reason="the `ui` extra is not installed"
    )
    from a4s.server import build_app

    os.environ["A4S_AAS_ENDPOINT"] = endpoint
    with fastapi_testclient.TestClient(build_app(endpoint)) as client:
        yield client


def test_health_reports_the_repository_not_just_the_process(api: Any) -> None:
    """A health check proving only that uvicorn started is the kind that stays
    green while every request behind it fails."""
    body = api.get("/health").json()

    assert body["ok"] is True
    assert body["repository"]


def test_aas_list_returns_the_fields_the_asset_picker_reads(api: Any) -> None:
    response = api.get("/api/aas-list")
    assert response.status_code == 200

    shells = response.json()
    assert shells, "the repository served no shells"
    for shell in shells:
        assert set(shell) >= {"id", "idShort", "assetKind", "submodelCount", "endpoint"}
    assert all(isinstance(s["submodelCount"], int) for s in shells)


def test_templates_returns_only_templates(api: Any, client: BasyxClient) -> None:
    """`kind` is the filter, and getting it wrong would offer the UI instances
    to instantiate from -- which `smt instantiate` then refuses."""
    templates = api.get("/api/templates").json()
    assert templates

    by_id = {t["id"] for t in templates}
    for submodel in client.list_submodels(limit=400):
        if submodel["id"] in by_id:
            assert submodel.get("kind") == "Template", submodel["id"]


def test_properties_are_flat_and_grouped_and_the_two_agree(api: Any) -> None:
    """The UI renders `properties` and navigates `grouped`; a disagreement
    between them shows as a tree missing rows the list has."""
    shells = api.get("/api/aas-list").json()
    target = max(shells, key=lambda s: s["submodelCount"])

    body = api.get(f"/api/aas-properties/{target['idShort']}").json()

    assert body["success"] is True
    assert body["count"] == len(body["properties"]) > 0
    assert sum(len(rows) for rows in body["grouped"].values()) == body["count"]
    for row in body["properties"]:
        assert set(row) >= {"idShort", "type", "semanticPath", "apiPath", "submodelId"}


def test_a_property_row_can_be_read_back_by_its_own_api_path(
    api: Any, scratch: str
) -> None:
    """`apiPath` is what a write addresses, so it has to be an address.

    Round-tripped through the element route rather than inspected, because a
    path that merely *looks* right is what the percent-encoding defect
    (PROGRESS 3.3) produced for a year.
    """
    run_aas("el", "add", scratch, "", "--type", "SubmodelElementCollection",
            "--id-short", "Box")
    run_aas("el", "add", scratch, "Box", "--type", "Property", "--id-short", "Inner",
            "--value", "probe", "--value-type", "xs:string")

    # The scratch submodel is attached to no shell, so it is addressed directly
    # through the element route -- the same call the tree view makes when it
    # expands a node.
    value = api.get("/api/element",
                    params={"submodel_id": scratch, "path": "Box.Inner"}).json()

    assert value["value"] == "probe"
    assert value["path"] == "Box.Inner"


def test_an_unknown_shell_is_a_404_that_names_what_it_looked_for(api: Any) -> None:
    """An empty list would read to the user as "this asset has no properties"."""
    response = api.get("/api/aas-properties/NoSuchShellAnywhere")

    assert response.status_code == 404
    assert "NoSuchShellAnywhere" in response.json()["detail"]


def test_parse_aas_agrees_with_the_route_it_used_to_prime(api: Any) -> None:
    """These two disagreed on the first run: 200 with a count of 0 against 99
    for the same shell.

    The cause was calling a FastAPI route from another route -- the default of
    a `Query(...)` parameter is the marker object, not the value, so the
    submodel filter matched nothing. Silent, and only visible because two
    numbers for one shell differed. Pinned so it cannot come back.
    """
    shells = api.get("/api/aas-list").json()
    target = max(shells, key=lambda s: s["submodelCount"])["idShort"]

    parsed = api.post(f"/api/parse-aas/{target}").json()
    listed = api.get(f"/api/aas-properties/{target}").json()

    assert parsed["success"] is True
    assert parsed["count"] == listed["count"] > 0
    assert parsed["collections"] == len(listed["grouped"])


def test_shell_full_resolves_the_submodels_and_reports_dangling_refs(
    api: Any
) -> None:
    """A dangling submodel reference is normal in a plant model.

    Failing the whole shell on one would make most of this repository
    unviewable; reporting it lets the UI show what exists and say what does
    not.
    """
    shells = api.get("/api/aas-list").json()
    target = max(shells, key=lambda s: s["submodelCount"])

    body = api.get(f"/api/aas/shells/{target['id']}/full").json()

    assert body["shell"]["id"] == target["id"]
    assert isinstance(body["unresolved"], list)
    assert len(body["submodels"]) + len(body["unresolved"]) == target["submodelCount"]


def test_cors_lets_the_vite_dev_server_through(api: Any) -> None:
    """The frontend runs on :5173 in development and the API elsewhere, so
    every request is cross-origin. Without this the UI shows an empty page and
    an error only visible in the browser console."""
    response = api.get("/api/aas-list", headers={"Origin": "http://localhost:5173"})

    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_a_bad_endpoint_is_reported_rather_than_mocked(endpoint: str) -> None:
    """The predecessor answered with invented shells when the repository
    misbehaved, which turns "BaSyx is down" into "here is a motor that does not
    exist". A failure has to look like one."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from a4s.server import build_app

    with fastapi_testclient.TestClient(
        build_app("http://127.0.0.1:9"), raise_server_exceptions=False
    ) as client:
        response = client.get("/api/aas-list")
        assert response.status_code == 502
        assert "cannot list shells" in response.json()["detail"]
        # And it names the address it tried. The browser sends its own
        # `?endpoint=` from a field the user edits, so the one that failed is
        # not necessarily the one this process was started with -- without the
        # address in the message, someone reads "Bad Gateway" and goes to check
        # the wrong server.
        assert "http://127.0.0.1:9" in response.json()["detail"]

    os.environ["A4S_AAS_ENDPOINT"] = endpoint     # the module fixture shares it


def test_check_aas_answers_false_instead_of_404_for_a_shell_that_is_absent(
    api: Any
) -> None:
    """The wizard calls this to choose a branch, not to read something.

    A 404 is the right answer for a route that was asked for a shell's
    contents; here it would put an error on screen for the ordinary case of an
    asset the user has not created yet.
    """
    response = api.get("/api/check-aas/NoSuchShellAnywhere")

    assert response.status_code == 200
    assert response.json()["exists"] is False
    assert response.json()["id"] is None


def test_check_aas_resolves_an_idShort_and_reports_both_id_spellings(
    api: Any
) -> None:
    """`aas_id` and `id` carry the same value because the UI reads
    `aas_id ?? id` -- sending one of the two would work until whichever caller
    compiled against the other one asked."""
    target = api.get("/api/aas-list").json()[0]

    body = api.get(f"/api/check-aas/{target['idShort']}").json()

    assert body["exists"] is True
    assert body["id"] == body["aas_id"] == target["id"]


def test_download_json_is_an_environment_and_arrives_as_a_file(api: Any) -> None:
    """Content type is load-bearing here, not cosmetic.

    The frontend's fetch wrapper parses `application/json` into an object, and
    the download helper expects a `Blob` -- typed correctly, the browser saves
    a file containing `[object Object]`.
    """
    import json

    target = max(api.get("/api/aas-list").json(), key=lambda s: s["submodelCount"])

    response = api.get(f"/api/download/json/{target['idShort']}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert target["idShort"] in response.headers["content-disposition"]

    environment = json.loads(response.content)
    assert environment["assetAdministrationShells"][0]["id"] == target["id"]
    assert len(environment["submodels"]) <= target["submodelCount"]
    assert isinstance(environment["conceptDescriptions"], list)


def test_the_aasx_route_says_it_is_not_built_rather_than_404ing(api: Any) -> None:
    """A 404 reads as a wrong URL. The button exists in the UI, so the answer
    has to name what is missing."""
    response = api.get("/api/download/aasx/AnyShell")

    assert response.status_code == 501
    assert "download/json" in response.json()["detail"]


def test_match_requires_both_the_document_and_the_shell(api: Any) -> None:
    """422 rather than a guess: matching the wrong document into a shell writes
    a plan that looks legitimate and is about another machine."""
    for payload in ({}, {"filename": "x.pdf"}, {"aas_id_short": "Something"}):
        assert api.post("/api/match", json=payload).status_code == 422


def test_match_on_a_document_with_no_extraction_says_which_step_is_missing(
    api: Any
) -> None:
    """The wizard can reach this by resuming an old session whose files were
    cleaned up, and "run extraction first" is the only useful thing to say."""
    response = api.post("/api/match", json={
        "filename": "never-uploaded-anything.pdf",
        "aas_id_short": "InjectionMoldingMachine_653",
    })

    assert response.status_code == 404
    assert "extraction" in response.json()["detail"]


def test_match_names_the_shell_it_could_not_find(api: Any, tmp_path: Any) -> None:
    """Resolution runs through the same `_resolve_shell` the read routes use,
    so an idShort works and a wrong name is a 404 that says what it looked
    for."""
    import json as _json

    from a4s.server import workspace

    name = "route-probe.pdf"
    target = workspace.extraction_file(name)
    target.write_text(
        _json.dumps([{"entity": "Clamping force", "value": "1500", "unit": "kN"}]),
        encoding="utf-8",
    )
    try:
        response = api.post("/api/match", json={
            "filename": name, "aas_id_short": "NoSuchShellAtAll",
        })
        assert response.status_code == 404
        assert "NoSuchShellAtAll" in response.json()["detail"]
    finally:
        target.unlink(missing_ok=True)


def test_a_plan_is_found_by_the_document_name_or_by_its_own(api: Any) -> None:
    """Both are handed to this route, and only one used to work.

    `check-matching` answers with the *plan's* filename and the browser passes
    that straight back; the wizard's own bookkeeping holds the *document's*.
    Deriving only from the document appended the suffix a second time, so a
    plan sitting on disk answered 404 -- which is what the user saw as "no
    matches found yet" on a run where every plan already existed.
    """
    import json as _json

    from a4s.server import workspace

    document = "resolver-probe.pdf"
    plan = workspace.match_file(document, "SomeShell")
    plan.write_text(_json.dumps({
        "submodel": "urn:probe:sm", "thresholds": {"high": 0.8, "low": 0.5},
        "decisions": [{"entity": "X", "value": "1", "unit": "", "source": "probe.pdf",
                       "action": "skip", "band": "below", "reason": "probe",
                       "score": 0.1}],
    }), encoding="utf-8")
    try:
        by_document = api.get(f"/api/matching-results/{document}",
                              params={"aas_id_short": "SomeShell"})
        by_plan = api.get(f"/api/matching-results/{plan.name}",
                          params={"aas_id_short": "SomeShell"})

        assert by_document.status_code == 200, by_document.text
        assert by_plan.status_code == 200, by_plan.text
        assert by_document.json()["count"] == by_plan.json()["count"] == 1
    finally:
        plan.unlink(missing_ok=True)


def test_what_check_matching_returns_is_what_matching_results_accepts(
    api: Any
) -> None:
    """The two routes are used as a pair by the browser, so their contract is
    one contract. It was broken between them, in the direction that only shows
    up once a plan exists."""
    import json as _json

    from a4s.server import workspace

    document = "pair-probe.pdf"
    plan = workspace.match_file(document, "PairShell")
    plan.write_text(_json.dumps({
        "submodel": "urn:probe:sm", "thresholds": {"high": 0.8, "low": 0.5},
        "decisions": [],
    }), encoding="utf-8")
    try:
        checked = api.get(f"/api/check-matching/{document}",
                          params={"aas_id_short": "PairShell"}).json()
        assert checked["exists"] is True

        handed_back = api.get(f"/api/matching-results/{checked['filename']}",
                              params={"aas_id_short": "PairShell"})
        assert handed_back.status_code == 200, handed_back.text
    finally:
        plan.unlink(missing_ok=True)


def test_populate_refuses_two_writes_to_one_element(api: Any) -> None:
    """Applied in order they overwrite each other and the element holds
    whichever arrived last, with the rest charged for and discarded.

    A silent overwrite is indistinguishable from a successful write in the
    response, so it is a refusal rather than a warning.
    """
    import json as _json

    from a4s.server import workspace

    document = "collide-probe.pdf"
    plan = workspace.match_file(document, "CollideShell")
    plan.write_text(_json.dumps({
        "submodel": "urn:probe:sm", "thresholds": {"high": 0.8, "low": 0.5},
        "decisions": [
            {"entity": "A", "value": "1", "unit": "", "source": "p.1",
             "action": "write", "band": "auto", "reason": "", "score": 0.9,
             "path": "Group.Same", "model_type": "Property"},
            {"entity": "B", "value": "2", "unit": "", "source": "p.2",
             "action": "write", "band": "auto", "reason": "", "score": 0.9,
             "path": "Group.Same", "model_type": "Property"},
        ],
    }), encoding="utf-8")
    try:
        response = api.post("/api/populate", json={
            "filename": document, "aas_id_short": "CollideShell", "dry_run": True,
        })
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["collisions"][0]["count"] == 2
        assert detail["collisions"][0]["action"] == "write"
    finally:
        plan.unlink(missing_ok=True)


def test_many_creates_in_one_parent_are_not_a_collision(api: Any) -> None:
    """The over-refusal this guard shipped with.

    For a write, `path` is the element; for a create it is the *parent*. Keying
    both the same way refused a plan where 57 different properties were being
    created in one collection, which is an ordinary submodel being filled in.
    """
    import json as _json

    from a4s.server import workspace

    document = "siblings-probe.pdf"
    plan = workspace.match_file(document, "SiblingShell")
    plan.write_text(_json.dumps({
        "submodel": "urn:probe:sm", "thresholds": {"high": 0.8, "low": 0.5},
        "decisions": [
            {"entity": f"E{n}", "value": str(n), "unit": "", "source": "p.1",
             "action": "create", "band": "hosted", "reason": "", "score": 0.6,
             "path": "Configuration", "id_short": f"Distinct{n}",
             "model_type": "Property", "value_type": "xs:int"}
            for n in range(5)
        ],
    }), encoding="utf-8")
    try:
        response = api.post("/api/populate", json={
            "filename": document, "aas_id_short": "SiblingShell", "dry_run": True,
        })
        assert response.status_code == 200, response.text
        assert response.json()["counts"]["applied"] == 5
    finally:
        plan.unlink(missing_ok=True)


def test_two_creates_with_one_id_short_under_one_parent_are_refused(
    api: Any
) -> None:
    """AASd-022, before it is committed rather than after.

    A duplicate idShort makes *both* copies unreachable by path -- the defect
    §3.3 found in live data. `el add` refuses one at a time; this refuses the
    plan that would attempt it.
    """
    import json as _json

    from a4s.server import workspace

    document = "dupe-probe.pdf"
    plan = workspace.match_file(document, "DupeShell")
    plan.write_text(_json.dumps({
        "submodel": "urn:probe:sm", "thresholds": {"high": 0.8, "low": 0.5},
        "decisions": [
            {"entity": "Screw diameter (290)", "value": "30", "unit": "mm",
             "source": "p.3", "action": "create", "band": "hosted", "reason": "",
             "score": 0.6, "path": "Configuration", "id_short": "ScrewDiameter",
             "model_type": "Property", "value_type": "xs:double"},
            {"entity": "Screw diameter (400)", "value": "35", "unit": "mm",
             "source": "p.4", "action": "create", "band": "hosted", "reason": "",
             "score": 0.6, "path": "Configuration", "id_short": "ScrewDiameter",
             "model_type": "Property", "value_type": "xs:double"},
        ],
    }), encoding="utf-8")
    try:
        response = api.post("/api/populate", json={
            "filename": document, "aas_id_short": "DupeShell", "dry_run": True,
        })
        assert response.status_code == 409
        collision = response.json()["detail"]["collisions"][0]
        assert collision["action"] == "create"
        assert "AASd-022" in collision["why"]
    finally:
        plan.unlink(missing_ok=True)


def test_a_write_onto_an_operation_is_refused_before_it_is_sent(api: Any) -> None:
    """BaSyx answers such a write with a bare HTTP 500 naming no reason, and
    295 of 396 decisions failed that way on the first real run.

    An Operation is invoked, not written to, and the message says so."""
    import json as _json

    from a4s.server import workspace

    document = "optype-probe.pdf"
    plan = workspace.match_file(document, "OpShell")
    plan.write_text(_json.dumps({
        "submodel": "urn:probe:sm", "thresholds": {"high": 0.8, "low": 0.5},
        "decisions": [
            {"entity": "Injection speed", "value": "140", "unit": "mm/s",
             "source": "p.3", "action": "write", "band": "auto", "reason": "",
             "score": 0.9, "path": "DelegatedServices.SetInjectionSpeed",
             "model_type": "Operation"},
        ],
    }), encoding="utf-8")
    try:
        response = api.post("/api/populate", json={
            "filename": document, "aas_id_short": "OpShell", "dry_run": True,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["counts"]["applied"] == 0
        assert "invoked, not written" in body["failed"][0]["error"]
    finally:
        plan.unlink(missing_ok=True)

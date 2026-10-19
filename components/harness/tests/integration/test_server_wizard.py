"""The wizard's match stage, against the live repository.

One thing here needs a repository and it is the one worth the layer: when a
person retargets an entity to a property in a *different* submodel, the row
still carries the submodel the matcher chose. Believing it writes the value
into the wrong submodel and reports success.

The last test runs the saved plan through `aas populate` for real, because the
whole point of this translation is that its output is that command's input.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator, Tuple

import pytest

from a4s.aasctl.client import AasHttpError, BasyxClient

from .conftest import TEST_PREFIX, run_aas, value_at

pytestmark = pytest.mark.integration


@pytest.fixture
def api(endpoint: str, tmp_path: Path, monkeypatch) -> Any:
    fastapi_testclient = pytest.importorskip(
        "fastapi.testclient", reason="the `ui` extra is not installed"
    )
    from a4s.server import build_app

    monkeypatch.setenv("A4S_WEB_WORKSPACE", str(tmp_path))
    os.environ["A4S_AAS_ENDPOINT"] = endpoint
    with fastapi_testclient.TestClient(build_app(endpoint)) as client:
        yield client


@pytest.fixture
def two_submodels(client: BasyxClient) -> Iterator[Tuple[str, str, str, str]]:
    """Two submodels holding an element of the same name, on one shell.

    The same idShort in both is the point: it is what makes "which submodel did
    they pick" a real question rather than a lookup. The names carry a random
    suffix because the repository these run against is a populated one, and a
    `Nameplate` already in it would make the ambiguity a different test's fault.
    """
    suffix = uuid.uuid4().hex[:8]
    first_id = f"{TEST_PREFIX}wiz:np:{suffix}"
    second_id = f"{TEST_PREFIX}wiz:ops:{suffix}"
    first_short = f"NameplateIT{suffix}"
    second_short = f"OpsIT{suffix}"
    shell_id = f"{TEST_PREFIX}wiz:aas:{suffix}"

    for identifier, id_short in ((first_id, first_short), (second_id, second_short)):
        run_aas("sm", "create", identifier, "--id-short", id_short)
        run_aas("el", "add", identifier, "", "--type", "SubmodelElementCollection",
                "--id-short", "TechnicalData")
        run_aas("el", "add", identifier, "TechnicalData", "--type", "Property",
                "--id-short", "SerialNumber", "--value-type", "xs:string")

    run_aas("shell", "create", shell_id, "--id-short", f"WizardShell{suffix}")
    run_aas("shell", "attach", shell_id, first_id)
    run_aas("shell", "attach", shell_id, second_id)

    try:
        yield first_id, first_short, second_id, second_short
    finally:
        with contextlib.suppress(AasHttpError):
            client.delete_shell(shell_id)
        for identifier in (first_id, second_id):
            with contextlib.suppress(AasHttpError):
                client.delete_submodel(identifier)


def _row(**overrides: Any) -> dict:
    row = {
        "extractedEntity": "Serial number",
        "extractedValue": "SN-9",
        "aasProperty": "SerialNumber",
        "aasType": "Property",
        "apiPath": "TechnicalData.SerialNumber",
        "confidence": 0.71,
        "selected": True,
    }
    row.update(overrides)
    return row


def test_retargeting_to_another_submodel_moves_the_write_with_it(
    api: Any, two_submodels: Tuple[str, str, str, str]
) -> None:
    """The row's `submodelId` is stale after the user picks elsewhere.

    Both submodels here hold `TechnicalData.SerialNumber`, so trusting the
    stale field writes a real value to a real element and returns 200. Nothing
    on screen would say the wrong asset had just been updated.
    """
    first_id, _, second_id, second_short = two_submodels

    body = api.post("/api/save-matching-results", json={
        "filename": f"doc_{second_short}",
        "results": [_row(
            submodelId=first_id,                     # what the matcher chose
            aasProperties=[{                         # what the person clicked
                "idShort": "SerialNumber",
                "type": "Property",
                "apiPath": "TechnicalData.SerialNumber",
                "semanticPath": f"{second_short}/TechnicalData.SerialNumber",
            }],
        )],
    }).json()

    assert body["unresolved"] == []
    (entry,) = body["plans"]
    assert entry["submodel"] == second_id, "the write followed the stale submodel"


def test_an_untouched_row_keeps_the_submodel_the_matcher_chose(
    api: Any, two_submodels: Tuple[str, str, str, str]
) -> None:
    """The other half of the same rule. A row nobody edited carries no
    `aasProperties`, and its `submodelId` is the only evidence there is."""
    first_id, first_short, _, _ = two_submodels

    body = api.post("/api/save-matching-results", json={
        "filename": f"doc_{first_short}",
        "results": [_row(submodelId=first_id,
                         semanticPath=f"{first_short}/TechnicalData.SerialNumber")],
    }).json()

    (entry,) = body["plans"]
    assert entry["submodel"] == first_id


def test_the_saved_plan_is_something_aas_populate_actually_applies(
    api: Any, client: BasyxClient, two_submodels: Tuple[str, str, str, str]
) -> None:
    """The end of the translation, checked end to end.

    Everything upstream of here is a shape argument. This is the only assertion
    that says the shape was the right one: the file this route wrote is handed
    to the command the wizard's last stage runs, and the value has to arrive in
    the element the person picked.
    """
    first_id, _, second_id, second_short = two_submodels

    body = api.post("/api/save-matching-results", json={
        "filename": f"doc_{second_short}",
        "results": [_row(
            extractedValue="SN-ROUNDTRIP",
            submodelId=first_id,
            aasProperties=[{
                "idShort": "SerialNumber",
                "type": "Property",
                "apiPath": "TechnicalData.SerialNumber",
                "semanticPath": f"{second_short}/TechnicalData.SerialNumber",
            }],
        )],
    }).json()

    plan_path = body["plans"][0]["path"]
    run_aas("populate", "--plan", plan_path, "--skip-warnings")

    assert value_at(client, second_id, "TechnicalData.SerialNumber") == "SN-ROUNDTRIP"
    assert value_at(client, first_id, "TechnicalData.SerialNumber") in (None, "")


def test_choosing_a_collection_creates_the_element_that_was_missing(
    api: Any, client: BasyxClient, two_submodels: Tuple[str, str, str, str]
) -> None:
    """The capability the predecessor did not have.

    When the datasheet states something the submodel has no slot for, the
    person picks the collection it belongs in and the element is created there.
    `aas populate` has had the create path all along; what was missing was a
    way for the wizard to ask for it.
    """
    _, _, second_id, second_short = two_submodels

    body = api.post("/api/save-matching-results", json={
        "filename": f"doc_{second_short}",
        "results": [_row(
            extractedEntity="Cycle count",
            extractedValue="12000",
            aasProperties=[{
                "idShort": "TechnicalData",
                "type": "SubmodelElementCollection",
                "apiPath": "TechnicalData",
                "semanticPath": f"{second_short}/TechnicalData",
            }],
        )],
    }).json()

    plan_path = body["plans"][0]["path"]
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    assert plan["decisions"][0]["action"] == "create"

    run_aas("populate", "--plan", plan_path, "--skip-warnings")

    assert value_at(client, second_id, "TechnicalData.Cycle_count") == "12000"

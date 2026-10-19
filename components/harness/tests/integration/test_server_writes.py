"""The mutating HTTP routes, against the live repository.

These are the routes the ShopFloor's buttons call. They run no approval gate --
that control belongs to the agent, decided 2026-08-27 -- so what has to be
tested here is everything the gate would otherwise have caught: that a delete
reaches only what it was asked for, that a half-finished create does not leave
orphans, and that the ledger sees every one of them.

Everything created lives under `TEST_PREFIX` and is purged by prefix, so these
can run against the same populated repository the agent uses.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import Any, Iterator, Tuple

import pytest

from a4s.aasctl.client import AasHttpError, BasyxClient

from .conftest import TEST_PREFIX, run_aas

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def api(endpoint: str):
    fastapi_testclient = pytest.importorskip(
        "fastapi.testclient", reason="the `ui` extra is not installed"
    )
    from a4s.server import build_app

    os.environ["A4S_AAS_ENDPOINT"] = endpoint
    with fastapi_testclient.TestClient(build_app(endpoint)) as client:
        yield client


@pytest.fixture
def blueprint(client: BasyxClient) -> Iterator[Tuple[str, str]]:
    """A shell with one submodel carrying a value, to be copied from."""
    suffix = uuid.uuid4().hex[:8]
    shell_id = f"{TEST_PREFIX}bp:{suffix}"
    submodel_id = f"{TEST_PREFIX}bp:sm:{suffix}"

    run_aas("sm", "create", submodel_id, "--id-short", "Nameplate")
    run_aas("el", "add", submodel_id, "", "--type", "Property",
            "--id-short", "SerialNumber", "--value", "SN-0001",
            "--value-type", "xs:string")
    run_aas("shell", "create", shell_id, "--id-short", "Blueprint",
            "--asset-kind", "Type")
    run_aas("shell", "attach", shell_id, submodel_id)

    try:
        yield shell_id, submodel_id
    finally:
        for identifier, delete in (
            (submodel_id, client.delete_submodel),
            (shell_id, client.delete_shell),
        ):
            with contextlib.suppress(AasHttpError):
                delete(identifier)


def _cleanup_shell(client: BasyxClient, api: Any, shell_id: str) -> None:
    with contextlib.suppress(Exception):
        api.delete(f"/api/delete-aas/{shell_id}")
    with contextlib.suppress(AasHttpError):
        client.delete_shell(shell_id)


# ---------------------------------------------------------------------------
# instantiate
# ---------------------------------------------------------------------------

def test_instantiating_a_blueprint_copies_its_submodels_and_says_what_it_copied(
    api: Any, client: BasyxClient, blueprint: Tuple[str, str]
) -> None:
    """The ShopFloor drops an asset and expects a working shell behind it.

    `valuesCopied` is asserted because it is the sharp edge: a copied nameplate
    carries the blueprint's serial number, which is an assertion about a
    different asset. Reporting it is what makes that visible to whoever is
    looking at the screen.
    """
    blueprint_id, _ = blueprint
    new_id = f"{TEST_PREFIX}inst:{uuid.uuid4().hex[:8]}"

    body = api.post("/api/instantiate-aas", json={
        "blueprint_id": blueprint_id,
        "new_aas_id": new_id,
        "new_aas_id_short": "CopiedAsset",
    }).json()

    try:
        assert body["success"] is True
        assert body["aas_id"] == new_id           # already namespaced, not minted
        assert len(body["submodels"]) == 1
        assert body["submodels"][0]["valuesCopied"] is True

        copied = client.get_submodel(body["submodels"][0]["id"])
        assert copied["kind"] == "Instance"
        serial = [e for e in copied["submodelElements"] if e["idShort"] == "SerialNumber"]
        assert serial[0]["value"] == "SN-0001"
    finally:
        _cleanup_shell(client, api, new_id)


def test_a_bare_name_is_minted_into_the_blueprints_namespace(
    api: Any, client: BasyxClient, blueprint: Tuple[str, str]
) -> None:
    """`AGV_042` is what the ShopFloor sends, and it is not an identifier.

    The response has to report the id that was actually used, because the UI
    stores it and every later call -- delete, connect -- addresses the shell by
    it.
    """
    blueprint_id, _ = blueprint
    bare = f"Asset{uuid.uuid4().hex[:6]}"

    body = api.post("/api/instantiate-aas", json={
        "blueprint_id": blueprint_id,
        "new_aas_id": bare,
        "new_aas_id_short": bare,
    }).json()

    try:
        assert body["requested_id"] == bare
        assert body["aas_id"] != bare
        assert body["aas_id"].endswith(f"/{bare}")
        assert client.get_shell(body["aas_id"])["id"] == body["aas_id"]
    finally:
        _cleanup_shell(client, api, body["aas_id"])


def test_instantiating_onto_an_existing_id_is_refused_not_merged(
    api: Any, client: BasyxClient, blueprint: Tuple[str, str]
) -> None:
    """A silent overwrite here would replace a live asset's shell with a copy
    of a blueprint."""
    blueprint_id, _ = blueprint

    response = api.post("/api/instantiate-aas", json={
        "blueprint_id": blueprint_id,
        "new_aas_id": blueprint_id,
        "new_aas_id_short": "Duplicate",
    })

    assert response.status_code == 409
    assert blueprint_id in response.json()["detail"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_deleting_a_shell_keeps_a_submodel_another_shell_still_points_at(
    api: Any, client: BasyxClient
) -> None:
    """The defect this route was written around.

    Several shells pointing at one nameplate is ordinary in a plant model.
    Deleting a shell's referenced submodels wholesale takes them out from under
    the others -- the same shape as the submodel-corrupting defect 3.3 found,
    and invisible until someone opens the shell that was not deleted.
    """
    suffix = uuid.uuid4().hex[:8]
    shared_sm = f"{TEST_PREFIX}shared:{suffix}"
    own_sm = f"{TEST_PREFIX}own:{suffix}"
    doomed = f"{TEST_PREFIX}doomed:{suffix}"
    survivor = f"{TEST_PREFIX}survivor:{suffix}"

    run_aas("sm", "create", shared_sm, "--id-short", "SharedNameplate")
    run_aas("sm", "create", own_sm, "--id-short", "PrivateSubmodel")
    run_aas("shell", "create", doomed, "--id-short", "Doomed")
    run_aas("shell", "create", survivor, "--id-short", "Survivor")
    run_aas("shell", "attach", doomed, shared_sm)
    run_aas("shell", "attach", doomed, own_sm)
    run_aas("shell", "attach", survivor, shared_sm)

    try:
        body = api.delete(f"/api/delete-aas/{doomed}").json()

        assert body["success"] is True
        assert body["submodelsDeleted"] == [own_sm]
        assert body["submodelsKept"] == [shared_sm]

        assert client.get_submodel(shared_sm)["id"] == shared_sm
        with pytest.raises(AasHttpError):
            client.get_submodel(own_sm)
        with pytest.raises(AasHttpError):
            client.get_shell(doomed)
    finally:
        for identifier, delete in (
            (shared_sm, client.delete_submodel),
            (own_sm, client.delete_submodel),
            (doomed, client.delete_shell),
            (survivor, client.delete_shell),
        ):
            with contextlib.suppress(AasHttpError):
                delete(identifier)


# ---------------------------------------------------------------------------
# connections
# ---------------------------------------------------------------------------

@pytest.fixture
def two_shells(client: BasyxClient) -> Iterator[Tuple[str, str]]:
    suffix = uuid.uuid4().hex[:8]
    source = f"{TEST_PREFIX}src:{suffix}"
    target = f"{TEST_PREFIX}tgt:{suffix}"
    run_aas("shell", "create", source, "--id-short", "Source")
    run_aas("shell", "create", target, "--id-short", "Target")
    try:
        yield source, target
    finally:
        for identifier in (source, target):
            with contextlib.suppress(AasHttpError):
                client.delete_shell(identifier)


def test_a_connection_is_a_relationship_element_hanging_off_the_source(
    api: Any, client: BasyxClient, two_shells: Tuple[str, str]
) -> None:
    """The edge has a direction and the metamodel stores it nowhere else:
    `first`/`second` are ordered, and so is which shell carries the submodel."""
    source, target = two_shells

    body = api.post("/api/create-connection", json={
        "source_aas_id": source,
        "target_aas_id": target,
    }).json()

    submodel_id = body["relationship_sm_id"]
    try:
        assert body["success"] is True

        submodel = client.get_submodel(submodel_id)
        element = submodel["submodelElements"][0]
        assert element["modelType"] == "RelationshipElement"
        assert element["first"]["keys"][0]["value"] == source
        assert element["second"]["keys"][0]["value"] == target

        attached = [
            key["value"]
            for reference in client.list_submodel_refs(source)
            for key in reference["keys"]
        ]
        assert submodel_id in attached
        assert not client.list_submodel_refs(target)
    finally:
        with contextlib.suppress(AasHttpError):
            client.delete_submodel(submodel_id)


def test_deleting_a_connection_removes_the_reference_and_the_submodel(
    api: Any, client: BasyxClient, two_shells: Tuple[str, str]
) -> None:
    """A reference left behind is the dangling one `/full` has to report
    around, and it would show in the UI as an edge that came back."""
    source, target = two_shells
    created = api.post("/api/create-connection", json={
        "source_aas_id": source,
        "target_aas_id": target,
    }).json()
    submodel_id = created["relationship_sm_id"]

    body = api.post("/api/delete-connection", json={
        "source_aas_id": source,
        "relationship_sm_id": submodel_id,
    }).json()

    assert body["success"] is True
    assert body["detached"] is True
    assert not client.list_submodel_refs(source)
    with pytest.raises(AasHttpError):
        client.get_submodel(submodel_id)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_every_web_write_lands_in_the_ledger(
    api: Any, client: BasyxClient, blueprint: Tuple[str, str], tmp_path: Any,
    monkeypatch: Any
) -> None:
    """Trusting the UI was a decision about *asking*, not about *recording*.

    The ledger's founding claim is one log for every actor. A shell created
    through the browser and absent from it would make that claim false, and the
    provenance query an auditor runs -- who changed this shell -- would answer
    with a gap rather than with `web`.
    """
    from a4s.ledger import Kind, Ledger
    from a4s.server import write_routes

    monkeypatch.setenv("A4S_HOME", str(tmp_path))
    blueprint_id, _ = blueprint
    new_id = f"{TEST_PREFIX}logged:{uuid.uuid4().hex[:8]}"

    body = api.post("/api/instantiate-aas", json={
        "blueprint_id": blueprint_id,
        "new_aas_id": new_id,
        "new_aas_id_short": "Logged",
    }).json()

    try:
        assert body["unrecorded"] is None, body["unrecorded"]

        with Ledger(tmp_path / "ledger.db", write_routes.SESSION) as ledger:
            events = ledger.provenance(new_id)
            assert [e.kind for e in events] == [Kind.AAS_CREATE]
            assert events[0].actor == "web"
            assert events[0].payload["blueprint"] == blueprint_id
            assert ledger.verify() == (True, None)
    finally:
        _cleanup_shell(client, api, new_id)


def test_manifest_refresh_reports_the_repository_instead_of_claiming_a_cache(
    api: Any
) -> None:
    """The button exists in the UI and there is nothing behind it any more.

    Answering "refreshed" would be a lie; 404ing would read as a broken build.
    It reports what is actually in the repository, which is the same choice
    `/api/parse-aas` makes.
    """
    body = api.post("/api/manifests/refresh").json()

    assert body["status"] == "ok"
    assert "nothing is cached" in body["message"]
    assert body["shells"] > 0


def test_a_minted_id_containing_a_slash_survives_the_round_trip(
    api: Any, client: BasyxClient, blueprint: Tuple[str, str]
) -> None:
    """Minting puts a `/` in the identifier, and the UI sends it back through
    `encodeURIComponent`, so the delete route is asked for `%2F`.

    Two things have to hold at once: the `:path` converter must accept the
    segment, and the id must not be silently truncated at the slash -- which
    would delete a different shell, or none, while answering 200. Verified
    against uvicorn as well as this client, because %2F handling belongs to
    the server and not to the routing table.
    """
    blueprint_id, _ = blueprint
    bare = f"Asset{uuid.uuid4().hex[:6]}"

    created = api.post("/api/instantiate-aas", json={
        "blueprint_id": blueprint_id,
        "new_aas_id": bare,
        "new_aas_id_short": bare,
    }).json()
    minted = created["aas_id"]
    assert "/" in minted

    body = api.delete(f"/api/delete-aas/{minted}").json()

    assert body["aas_id"] == minted
    with pytest.raises(AasHttpError):
        client.get_shell(minted)

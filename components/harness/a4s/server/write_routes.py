"""The routes that change the repository.

**The approval gate does not run here, and that is a decision** (2026-08-27,
open question 8). The gate recovers an action from a command line, reads its
danger from the plant model, and asks a human. An HTTP request has neither a
command line nor a human to ask. Rather than invent a server-side approver with
nobody at the other end, the gate stays what it was built to be: a control on
the *agent*, which acts without being watched. The UI is trusted the way a
BaSyx admin panel is trusted -- somebody is sitting in front of it, and the
button they pressed is the confirmation.

**What that decision does not cover is the record**, so every mutation here is
appended to the ledger. `a4s/ledger/log.py` opens by claiming one log for every
actor -- "operator actions, agent tool calls, AAS mutations and service
invocations are the same kind of record" -- and a web write that left no trace
would make that claim false the first time someone deleted a shell through the
browser. The cost of trusting the UI is visible in those entries: the actor is
`web`, not a person, because an unauthenticated request cannot say who sent it
and a self-declared name in a header is worse than an honest label.

The ledger connection is opened and closed per request. Not laziness: sync
routes run in uvicorn's threadpool, and a sqlite connection shared across
threads is a runtime error waiting for the second concurrent write.
"""

from __future__ import annotations

import copy as copylib
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, Body, HTTPException

from ..aasctl import model, smt
from ..aasctl.client import AasHttpError, BasyxClient
from ..ledger import Kind, Ledger
from .aas_routes import _client, _fail, _resolve_shell

router = APIRouter()

# One session per server process. A restart starts a new one, which is the
# truth: a ledger session is a continuous stretch of activity, and pinning a
# fixed id across restarts would merge unrelated ones.
SESSION = f"web-{uuid.uuid4().hex[:12]}"

WEB_ACTOR = "web"


def _home() -> Path:
    """Read `$A4S_HOME` at call time, not at import.

    `.env` is loaded by `main()`, which runs after this module is imported, so
    a module-level constant would capture the value from before the file was
    read -- and the ledger would end up somewhere other than the one the CLI
    writes to, which defeats the point of there being one log.
    """
    return Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser()


def _recorded(kind: str, subject: str, payload: Dict[str, Any]) -> Optional[str]:
    """Append one event, and never fail the request because it could not.

    A write that succeeded against BaSyx and then 500s because sqlite was busy
    would leave the caller believing nothing happened while the repository
    disagrees. The unrecorded write is the lesser wrong, and it is named in the
    response rather than swallowed.
    """
    try:
        with Ledger(_home() / "ledger.db", SESSION, actor=WEB_ACTOR) as ledger:
            ledger.append(kind, payload, subject=subject)
    except Exception as exc:  # noqa: BLE001 - reported in the response body
        return f"{type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# shells
# ---------------------------------------------------------------------------

@router.post("/api/instantiate-aas")
def instantiate_aas(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Copy a blueprint shell into a new instance, submodels and all.

    The ShopFloor calls this when an asset is dropped on the canvas, passing a
    blueprint shell id and a bare name like `AGV_042` for both the new id and
    its idShort. A bare name is a poor identifier -- nothing namespaces it, and
    two plants would collide -- so one that carries no scheme is minted into
    the blueprint's own namespace. The response returns `requested_id` beside
    `aas_id` so the caller can see that happened, rather than discovering it
    when a later call 404s.

    A submodel whose `kind` is `Template` goes through `smt.instantiate`, which
    is what that function is for: the slots survive and the example values do
    not, unless A4S/Instantiation/KeepConfiguredValues explicitly retains site
    configuration. Anything else is copied as it stands, because a blueprint's values are
    usually the reason someone chose that blueprint. That is also the sharp
    edge -- copying a nameplate copies its serial number, an assertion about a
    different asset -- so every copied submodel is named in the response with
    `valuesCopied` against it.
    """
    blueprint_id = str(body.get("blueprint_id") or "").strip()
    requested_id = str(body.get("new_aas_id") or "").strip()
    id_short = str(body.get("new_aas_id_short") or requested_id).strip()
    endpoint = body.get("endpoint")

    if not blueprint_id or not requested_id:
        raise HTTPException(
            status_code=422,
            detail="blueprint_id and new_aas_id are both required",
        )

    with _client(endpoint) as client:
        try:
            blueprint = _resolve_shell(client, blueprint_id)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc, f"cannot read blueprint {blueprint_id}") from exc

        shell_id = _mint(requested_id, blueprint.get("id", ""))
        if _exists(client, shell_id):
            raise HTTPException(
                status_code=409,
                detail=f"a shell with id {shell_id!r} already exists",
            )

        created: List[Dict[str, Any]] = []
        try:
            for source in _submodels_of(client, blueprint):
                clone = _copy_submodel(source, shell_id)
                client.create_submodel(clone)
                created.append({
                    "id": clone["id"],
                    "idShort": clone.get("idShort"),
                    "from": source.get("id"),
                    "kind": source.get("kind"),
                    "valuesCopied": source.get("kind") != "Template" or _configured_defaults(source),
                })

            shell = model.make_shell(
                shell_id,
                _safe_id_short(id_short or requested_id),
                global_asset_id=f"{shell_id}/asset",
                asset_kind="Instance",
            )
            shell["submodels"] = [model.submodel_reference(s["id"]) for s in created]
            client.create_shell(shell)
        except Exception as exc:  # noqa: BLE001
            # Half a shell is worse than none: the submodels already written
            # would be orphans that nothing references and nothing lists.
            _rollback(client, [s["id"] for s in created])
            raise _fail(exc, f"cannot instantiate {shell_id}") from exc

    unrecorded = _recorded(Kind.AAS_CREATE, shell_id, {
        "route": "instantiate-aas",
        "blueprint": blueprint.get("id"),
        "submodels": [s["id"] for s in created],
    })
    return {
        "success": True,
        "aas_id": shell_id,
        "requested_id": requested_id,
        "idShort": _safe_id_short(id_short or requested_id),
        "blueprint_id": blueprint.get("id"),
        "submodels": created,
        "unrecorded": unrecorded,
    }


@router.delete("/api/delete-aas/{aas_id:path}")
def delete_aas(aas_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Delete a shell and the submodels only it references.

    The "only it" is the whole of the care here. Submodels are shared -- a
    plant model routinely points several shells at one nameplate -- and
    deleting a shell's references wholesale takes those out from under the
    other shells, which is the class of defect the integration layer caught in
    3.3. Every other shell is checked first, and what is kept is named in the
    response rather than silently skipped.
    """
    with _client(endpoint) as client:
        try:
            shell = _resolve_shell(client, aas_id)
            shell_id = shell["id"]
            mine = _referenced_ids(client, shell)
            shared = _shared_with_other_shells(client, shell_id, mine)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc, f"cannot read {aas_id}") from exc

        deleted: List[str] = []
        failed: List[Dict[str, str]] = []
        for submodel_id in mine:
            if submodel_id in shared:
                continue
            try:
                client.delete_submodel(submodel_id)
                deleted.append(submodel_id)
            except AasHttpError as exc:
                failed.append({"id": submodel_id, "detail": str(exc)})

        try:
            client.delete_shell(shell_id)
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc, f"cannot delete shell {shell_id}") from exc

    unrecorded = _recorded(Kind.AAS_DELETE, shell_id, {
        "route": "delete-aas",
        "submodelsDeleted": deleted,
        "submodelsKept": sorted(shared),
    })
    return {
        "success": True,
        "aas_id": shell_id,
        "submodelsDeleted": deleted,
        "submodelsKept": sorted(shared),
        "submodelsFailed": failed,
        "unrecorded": unrecorded,
    }


# ---------------------------------------------------------------------------
# relationships
# ---------------------------------------------------------------------------

@router.post("/api/create-connection")
def create_connection(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Wire two shells together as a `RelationshipElement`, in its own submodel.

    One submodel per connection, not one `Connections` submodel holding many.
    The frontend deletes by `relationship_sm_id`, which fixes the granularity:
    a shared submodel would make deleting one edge a read-modify-write against
    a document two browsers can be editing at once. One submodel per edge makes
    the delete a delete.

    The relationship hangs off the *source* shell. An edge has a direction and
    the metamodel stores it nowhere else -- `first` and `second` are ordered,
    and so is which shell carries the submodel.
    """
    source_id = str(body.get("source_aas_id") or "").strip()
    target_id = str(body.get("target_aas_id") or "").strip()
    rule_id = body.get("rule_id")
    endpoint = body.get("endpoint")

    if not source_id or not target_id:
        raise HTTPException(
            status_code=422,
            detail="source_aas_id and target_aas_id are both required",
        )

    with _client(endpoint) as client:
        try:
            source = _resolve_shell(client, source_id)
            target = _resolve_shell(client, target_id)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc, "cannot read the shells to connect") from exc

        submodel_id = f"{source['id']}/rel/{uuid.uuid4().hex[:12]}"
        submodel = model.make_submodel(
            submodel_id,
            _safe_id_short(f"Rel_{source.get('idShort')}_{target.get('idShort')}"),
            kind="Instance",
        )
        element = model.make_element(
            "RelationshipElement",
            "Connection",
            first=model.model_reference(
                [{"type": "AssetAdministrationShell", "value": source["id"]}]
            ),
            second=model.model_reference(
                [{"type": "AssetAdministrationShell", "value": target["id"]}]
            ),
        )
        submodel["submodelElements"] = [element]
        if rule_id:
            submodel["submodelElements"].append(
                model.make_element("Property", "RuleId", value=str(rule_id))
            )

        try:
            client.create_submodel(submodel)
            client.add_submodel_ref(source["id"], model.submodel_reference(submodel_id))
        except Exception as exc:  # noqa: BLE001
            _rollback(client, [submodel_id])
            raise _fail(exc, "cannot create the relationship") from exc

    unrecorded = _recorded(Kind.AAS_CREATE, source["id"], {
        "route": "create-connection",
        "target": target["id"],
        "relationship_sm_id": submodel_id,
        "rule_id": rule_id,
    })
    return {
        "success": True,
        "relationship_sm_id": submodel_id,
        "source_aas_id": source["id"],
        "target_aas_id": target["id"],
        "unrecorded": unrecorded,
    }


@router.post("/api/delete-connection")
def delete_connection(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Detach the relationship submodel from the source shell, then delete it.

    In that order. The reverse leaves the shell pointing at a submodel that no
    longer exists if the second call fails, which is the dangling reference
    `/api/aas/shells/{id}/full` then has to report around.
    """
    source_id = str(body.get("source_aas_id") or "").strip()
    submodel_id = str(body.get("relationship_sm_id") or "").strip()
    endpoint = body.get("endpoint")

    if not source_id or not submodel_id:
        raise HTTPException(
            status_code=422,
            detail="source_aas_id and relationship_sm_id are both required",
        )

    with _client(endpoint) as client:
        try:
            source = _resolve_shell(client, source_id)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc, f"cannot read {source_id}") from exc

        detached = True
        try:
            client.remove_submodel_ref(source["id"], submodel_id)
        except AasHttpError:
            # Already gone, or never attached. The submodel is still deleted:
            # what the caller asked for is that the connection stop existing.
            detached = False
        try:
            client.delete_submodel(submodel_id)
        except AasHttpError as exc:
            if exc.status != 404:
                raise _fail(exc, f"cannot delete {submodel_id}") from exc

    unrecorded = _recorded(Kind.AAS_DELETE, source["id"], {
        "route": "delete-connection",
        "relationship_sm_id": submodel_id,
        "detached": detached,
    })
    return {
        "success": True,
        "relationship_sm_id": submodel_id,
        "detached": detached,
        "unrecorded": unrecorded,
    }


# ---------------------------------------------------------------------------
# the predecessor's cache, which is not one here
# ---------------------------------------------------------------------------

@router.post("/api/manifests/refresh")
def refresh_manifests(endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Kept because the UI calls it; there is nothing to refresh.

    The predecessor cached a manifest of services and resources and this button
    rebuilt it. Every route here reads the repository on request, for the same
    reason `/api/parse-aas` no longer builds a graph: a cache with no
    invalidation shows a plant that has since changed. The answer reports what
    is actually there, so the button still tells the operator something true.
    """
    try:
        with _client(endpoint) as client:
            shells = client.list_shells(limit=500)
            submodels = client.list_submodels(limit=500)
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, "cannot read the repository") from exc

    return {
        "status": "ok",
        "message": (
            f"nothing is cached; the repository holds {len(shells)} shells "
            f"and {len(submodels)} submodels right now"
        ),
        "shells": len(shells),
        "submodels": len(submodels),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mint(requested: str, blueprint_id: str) -> str:
    """A namespaced identifier from a bare name, or the caller's own if it
    already carries a scheme. `AGV_042` becomes `<blueprint namespace>/AGV_042`.
    """
    if ":" in requested:
        return requested
    base = blueprint_id.rsplit("/", 1)[0] if "/" in blueprint_id else blueprint_id
    return f"{base}/{requested}" if base else requested


def _safe_id_short(raw: str) -> str:
    """idShort is constrained to `[a-zA-Z][a-zA-Z0-9_]*`, and BaSyx enforces it."""
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    return cleaned if cleaned[:1].isalpha() else f"X{cleaned}"


def _exists(client: BasyxClient, shell_id: str) -> bool:
    try:
        client.get_shell(shell_id)
        return True
    except AasHttpError:
        return False


def _referenced_ids(client: BasyxClient, shell: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for reference in client.list_submodel_refs(shell["id"]):
        for key in reference.get("keys") or []:
            if key.get("value"):
                out.append(key["value"])
    return out


def _shared_with_other_shells(
    client: BasyxClient, shell_id: str, submodel_ids: List[str]
) -> Set[str]:
    """Which of these submodels another shell also points at."""
    wanted = set(submodel_ids)
    if not wanted:
        return set()
    shared: Set[str] = set()
    for other in client.list_shells(limit=500):
        if other.get("id") == shell_id:
            continue
        for reference in other.get("submodels") or []:
            for key in reference.get("keys") or []:
                if key.get("value") in wanted:
                    shared.add(key["value"])
    return shared


def _submodels_of(client: BasyxClient, shell: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for submodel_id in _referenced_ids(client, shell):
        try:
            out.append(client.get_submodel(submodel_id))
        except AasHttpError:
            # A blueprint with a dangling reference still instantiates; the
            # copy simply has one fewer submodel than the original claimed.
            continue
    return out


def _configured_defaults(source: Dict[str, Any]) -> bool:
    """Explicit blueprint configuration, never implicit SMT example values."""
    return any(q.get("type") == "A4S/Instantiation/KeepConfiguredValues" and
               q.get("value") == "true" for q in source.get("qualifiers", []))


def _copy_submodel(source: Dict[str, Any], shell_id: str) -> Dict[str, Any]:
    id_short = source.get("idShort") or "Submodel"
    new_id = f"{shell_id}/sm/{id_short}"
    if source.get("kind") == "Template":
        clone = smt.instantiate(source, submodel_id=new_id, id_short=id_short,
                                keep_values=_configured_defaults(source)).submodel
    else:
        clone = copylib.deepcopy(source)
    clone["id"] = new_id
    clone["kind"] = "Instance"
    if id_short == "ProductionContext":
        for element in clone.get("submodelElements", []):
            if element.get("idShort") == "MaintenanceSubmodelId":
                element["value"] = f"{shell_id}/sm/MaintenanceInstructions"
    return clone


def _rollback(client: BasyxClient, submodel_ids: List[str]) -> None:
    for submodel_id in submodel_ids:
        try:
            client.delete_submodel(submodel_id)
        except Exception:  # noqa: BLE001 - the original failure is what matters
            continue


__all__ = ["router"]

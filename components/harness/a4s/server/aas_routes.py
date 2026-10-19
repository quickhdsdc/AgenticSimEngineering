"""Read-only AAS routes: list shells, walk a submodel, list templates, export.

These are what the UI needs before it can show anything, and each is one call
into `aasctl` plus a reshape. The reshape is the whole content of this module
-- the frontend's field names were fixed by the predecessor's API and changing
them would mean changing the UI, which is not the job.

Read-only throughout, and that is a boundary rather than an accident. The
agent's writes pass the approval gate in `a4s/kernel/approval.py`; a route that
wrote would not, so the UI's mutating calls wait on that question being
answered rather than on the code being typed.

Two departures from the predecessor's implementation, both deliberate:

**No graph cache.** `/api/aas-properties` used to read a NetworkX graph written
to a temp file by a separate `parse-aas` call, and returned 404 when nobody had
run it -- a cache with no invalidation, whose miss is an error message telling
the user to go and run something else. Here the submodel is walked on request.
It costs one GET against a repository that is on the same network, and the
route cannot be stale or absent.

**No mock fallback.** The predecessor answered with invented shells when the
repository returned something unexpected, which turns "BaSyx is down" into
"here is a motor that does not exist". A failure is reported as one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from ..aasctl.client import AasHttpError, BasyxClient, Endpoint
from ..aasctl.index import walk
from ..aasctl.paths import ElementPath
from ..aasctl.catalog import is_blueprint, shells_in_view

router = APIRouter()


def _client(endpoint: Optional[str] = None) -> BasyxClient:
    try:
        return BasyxClient(Endpoint.from_env(endpoint))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _fail(exc: Exception, what: str, endpoint: Optional[str] = None) -> HTTPException:
    """Turn a transport or HTTP failure into one the UI can display.

    The status is preserved where the repository gave one, because a 404 on a
    shell and a 500 from a broken BaSyx mean different things to whoever is
    looking at the screen.

    **The endpoint is named on a transport failure**, and the reason is a real
    screenful of confusion: the browser sends its own `?endpoint=` from a field
    the user can edit, so the address that failed is not necessarily the one
    `a4s-serve` was started with. "cannot list shells: ConnectTimeout" leaves
    someone checking the wrong server. Naming it points at the box on screen
    that is wrong.
    """
    if isinstance(exc, AasHttpError):
        return HTTPException(status_code=exc.status, detail=f"{what}: {exc.body[:400]}")
    where = ""
    try:
        where = f" at {Endpoint.from_env(endpoint).base_url}"
    except RuntimeError:
        pass
    return HTTPException(
        status_code=502,
        detail=f"{what}{where}: {type(exc).__name__}: {exc}",
    )


# ---------------------------------------------------------------------------
# shells
# ---------------------------------------------------------------------------

@router.get("/api/aas-list")
def aas_list(
    endpoint: Optional[str] = None, limit: int = 200,
    asset_kind: Literal["Instance", "Type", "NotApplicable", "all"] = "Instance",
) -> List[Dict[str, Any]]:
    """Operational instances by default; explicit opt-in for blueprint/admin views."""
    try:
        with _client(endpoint) as client:
            shells = shells_in_view(client.list_shells(limit=limit), asset_kind)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        raise _fail(exc, "cannot list shells", endpoint) from exc

    resolved = Endpoint.from_env(endpoint).base_url
    return [
        {
            "id": shell.get("id", ""),
            "idShort": shell.get("idShort") or "(unnamed)",
            "assetKind": (shell.get("assetInformation") or {}).get("assetKind", "Unknown"),
            "usageRole": "blueprint" if is_blueprint(shell) else "asset",
            "globalAssetId": (shell.get("assetInformation") or {}).get("globalAssetId"),
            "submodelCount": len(shell.get("submodels") or []),
            "endpoint": resolved,
        }
        for shell in shells
    ]


@router.get("/api/aas/shells/{shell_id:path}/full")
def shell_full(shell_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """One shell with each referenced submodel resolved.

    The UI's tree view needs the submodels themselves, not the references. Done
    here rather than as N calls from the browser: the references are ids the
    browser would have to base64url-encode, which is the encoding burden the
    CLI exists to keep away from *its* caller and there is no reason to hand it
    to this one either.
    """
    try:
        with _client(endpoint) as client:
            shell = client.get_shell(shell_id)
            submodels: List[Dict[str, Any]] = []
            missing: List[str] = []
            for reference in client.list_submodel_refs(shell_id):
                for key in reference.get("keys") or []:
                    identifier = key.get("value")
                    if not identifier:
                        continue
                    try:
                        submodels.append(client.get_submodel(identifier))
                    except AasHttpError:
                        # A dangling reference is normal in a plant model and
                        # is reported rather than failing the whole shell.
                        missing.append(identifier)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, f"cannot read shell {shell_id}", endpoint) from exc

    return {"shell": shell, "submodels": submodels, "unresolved": missing}


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------

def _properties_of(submodel: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one submodel into the row shape the property tree renders.

    `semanticPath` and `apiPath` are the predecessor's names and are kept.
    `apiPath` is what a write actually addresses, so it carries the idShortPath
    and the submodel id -- the UI hands it straight back on save.
    """
    identifier = submodel.get("id", "")
    rows: List[Dict[str, Any]] = []
    for node in walk(submodel):
        parent = node.parent or ""
        rows.append({
            "idShort": node.id_short or node.path.rsplit(".", 1)[-1],
            "type": node.model_type,
            "description": node.description,
            "semanticPath": f"{submodel.get('idShort') or identifier}/{node.path}",
            "apiPath": node.path,
            "submodelId": identifier,
            "semanticId": node.semantic_id or "",
            "valueType": node.value_type or "",
            "unit": node.unit or "",
            "value": node.value,
            "parentCollection": parent,
        })
    return rows


def collect_properties(
    aas_id_short: str,
    endpoint: Optional[str] = None,
    submodel: Optional[str] = None,
) -> Dict[str, Any]:
    """Every addressable element under a shell, flat and grouped by parent.

    A plain function, and the route below is a two-line wrapper around it.
    Calling a FastAPI route from another route looks like it works and does
    not: the default of a `Query(...)` parameter is the marker object, not the
    value it stands for, so `submodel` arrives truthy and every submodel is
    filtered out. It cost a route that answered 200 with a count of zero while
    the same request over HTTP returned 99 -- silent, and only visible because
    two numbers for one shell disagreed.
    """
    try:
        with _client(endpoint) as client:
            shell = _resolve_shell(client, aas_id_short)
            references = client.list_submodel_refs(shell["id"])
            properties: List[Dict[str, Any]] = []
            for reference in references:
                for key in reference.get("keys") or []:
                    identifier = key.get("value")
                    if not identifier or (submodel and identifier != submodel):
                        continue
                    try:
                        properties.extend(_properties_of(client.get_submodel(identifier)))
                    except AasHttpError:
                        continue
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, f"cannot read properties of {aas_id_short}", endpoint) from exc

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in properties:
        grouped.setdefault(row["parentCollection"] or "Root", []).append(row)

    return {
        "success": True,
        "properties": properties,
        "grouped": grouped,
        "count": len(properties),
    }


@router.get("/api/aas-properties/{aas_id_short}")
def aas_properties(
    aas_id_short: str,
    endpoint: Optional[str] = None,
    submodel: Optional[str] = Query(None, description="restrict to one submodel id"),
) -> Dict[str, Any]:
    """The route. The path parameter is an *idShort* because that is what the
    predecessor's UI passes; a shell id is accepted too, since one is
    unambiguous and the other is a label."""
    return collect_properties(aas_id_short, endpoint, submodel)


def _resolve_shell(client: BasyxClient, name: str) -> Dict[str, Any]:
    """A shell by id if that works, else by idShort. 404 names what it tried."""
    try:
        return client.get_shell(name)
    except AasHttpError:
        pass
    for shell in client.list_shells(limit=500):
        if shell.get("idShort") == name:
            return shell
    raise HTTPException(
        status_code=404,
        detail=f"no shell with id or idShort {name!r} in this repository",
    )


@router.post("/api/parse-aas/{aas_id_short}")
def parse_aas(aas_id_short: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Kept because the UI calls it, and now it is a no-op that reports.

    The predecessor wrote a NetworkX graph to a temp file here and
    `/api/aas-properties` read it back. That cache had no invalidation, so the
    UI could show a tree the repository no longer had, and the route existed
    mainly so the other one would stop returning 404. Properties are walked on
    request now, so nothing needs building -- the route answers with the counts
    it would have cached, which is what the UI displays anyway.
    """
    result = collect_properties(aas_id_short, endpoint)
    return {
        "success": True,
        "aasIdShort": aas_id_short,
        "count": result["count"],
        "collections": len(result["grouped"]),
        "note": "properties are read on demand; nothing is cached",
    }


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

@router.get("/api/templates")
def templates(endpoint: Optional[str] = None, limit: int = 400) -> List[Dict[str, Any]]:
    """Submodels whose kind is Template, with their top-level slot count."""
    try:
        with _client(endpoint) as client:
            submodels = client.list_submodels(limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, "cannot list templates", endpoint) from exc

    out: List[Dict[str, Any]] = []
    for submodel in submodels:
        if submodel.get("kind") != "Template":
            continue
        elements = submodel.get("submodelElements") or []
        semantic = (submodel.get("semanticId") or {}).get("keys") or []
        out.append({
            "id": submodel.get("id", ""),
            "idShort": submodel.get("idShort") or "(unnamed)",
            "name": submodel.get("idShort") or submodel.get("id", ""),
            "semanticId": semantic[0].get("value") if semantic else "",
            "slotCount": len(elements),
            "description": _first_text(submodel.get("description")),
        })
    return out


def _first_text(entries: Any) -> str:
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("text"):
                return str(entry["text"])
    return ""


@router.get("/api/aas/properties/{shell_id:path}")
def aas_properties_by_id(shell_id: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """The id-addressed spelling of the same thing, which the newer UI uses."""
    return collect_properties(shell_id, endpoint)


@router.get("/api/element")
def element_value(
    submodel_id: str,
    path: str,
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """One element's value, for the tree view's lazy expansion."""
    try:
        with _client(endpoint) as client:
            return {
                "submodelId": submodel_id,
                "path": path,
                "value": client.get_value(submodel_id, ElementPath.parse(path)),
            }
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, f"cannot read {path}", endpoint) from exc


# ---------------------------------------------------------------------------
# existence and export
# ---------------------------------------------------------------------------

@router.get("/api/check-aas/{aas_id_short}")
def check_aas(aas_id_short: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """Whether a shell exists, as data rather than as a status code.

    The one route here that must *not* 404 on a miss: the wizard calls it to
    decide which branch to take, and `_resolve_shell`'s 404 -- right for a
    route that was asked to read something -- would show the user an error for
    the ordinary case of a shell they have not created yet.

    `aas_id` and `id` carry the same value. The UI reads `aas_id ?? id`, so
    both are sent rather than betting on which spelling the caller compiled
    against.
    """
    try:
        with _client(endpoint) as client:
            try:
                shell = _resolve_shell(client, aas_id_short)
            except HTTPException:
                return {"exists": False, "id": None, "aas_id": None,
                        "idShort": aas_id_short}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, f"cannot look up {aas_id_short}", endpoint) from exc

    identifier = shell.get("id", "")
    return {
        "exists": True,
        "id": identifier,
        "aas_id": identifier,
        "idShort": shell.get("idShort") or aas_id_short,
        "submodelCount": len(shell.get("submodels") or []),
    }


@router.get("/api/download/json/{aas_id_short}")
def download_json(aas_id_short: str, endpoint: Optional[str] = None) -> Any:
    """One shell and its submodels as a Part 5 Environment, as a file.

    Sent as `application/octet-stream`, not `application/json`, and that is not
    cosmetic: the frontend's fetch wrapper switches on the content type and
    parses JSON into an object, so a correctly typed response would arrive
    where a `Blob` is expected and `triggerDownload` would save `[object
    Object]`.

    Concept descriptions are included when the repository will list them, so
    the file opens elsewhere with its semantics intact -- but a failure to
    fetch them costs the reader nothing that the shell does not already carry,
    so it is not allowed to fail the download.
    """
    from fastapi import Response

    try:
        with _client(endpoint) as client:
            shell = _resolve_shell(client, aas_id_short)
            submodels: List[Dict[str, Any]] = []
            for reference in client.list_submodel_refs(shell["id"]):
                for key in reference.get("keys") or []:
                    identifier = key.get("value")
                    if not identifier:
                        continue
                    try:
                        submodels.append(client.get_submodel(identifier))
                    except AasHttpError:
                        continue
            concepts = _referenced_concepts(client, submodels)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _fail(exc, f"cannot export {aas_id_short}", endpoint) from exc

    import json

    environment = {
        "assetAdministrationShells": [shell],
        "submodels": submodels,
        "conceptDescriptions": concepts,
    }
    name = shell.get("idShort") or aas_id_short
    return Response(
        content=json.dumps(environment, indent=2, ensure_ascii=False),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
    )


def _referenced_concepts(
    client: BasyxClient, submodels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The concept descriptions these submodels actually point at.

    Shipping the repository's whole dictionary would make the export larger
    than the thing being exported; shipping none makes every semanticId in the
    file a dangling pointer once it leaves this repository.
    """
    wanted = set()
    for submodel in submodels:
        for node in walk(submodel):
            if node.semantic_id:
                wanted.add(node.semantic_id)
        for key in (submodel.get("semanticId") or {}).get("keys") or []:
            if key.get("value"):
                wanted.add(key["value"])
    if not wanted:
        return []
    try:
        return [c for c in client.list_concept_descriptions() if c.get("id") in wanted]
    except Exception:  # noqa: BLE001 - an export without semantics beats no export
        return []


@router.get("/api/download/aasx/{aas_id_short}")
def download_aasx(aas_id_short: str, endpoint: Optional[str] = None) -> Any:
    """Not built, and answering so rather than 404ing.

    AASX is an OPC package -- a zip with a relationship graph and the
    supplementary files the shell points at -- and writing one correctly is a
    serialiser, not a route. A 404 here would read as a wrong URL; this says
    what is missing and what to use instead.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "AASX packaging is not implemented in this harness. "
            f"Use /api/download/json/{aas_id_short} for the Environment JSON."
        ),
    )

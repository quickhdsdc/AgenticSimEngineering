"""Live OPC UA values for the browser, addressed through the AAS.

**The browser never names an endpoint.** It asks for a submodel, and the
server reads the `valueSource` qualifiers off it to learn which node on which
server each element comes from. That is the same discovery `aas opc ls` does
for the agent, and having the two paths share it is the point: a console that
carried its own hard-coded address list would drift from the model the moment
anything moved, and would be showing the operator something the AAS does not
say.

**Writing is deliberately narrow.** A write goes through an *Operation* that
the model declares, not through a node id the browser chose. So the request
names a submodel and an operation path; the server resolves the delegation and
the safety declaration itself. A console that could post an arbitrary node and
value would be a remote PLC writer with a web front end, and nothing in the
AAS would constrain it.

The approval gate does not run here, consistent with the rest of the web half
(decision log, 2026-08-27): the gate recovers an action from a command line and
asks a human, and here the human is the one clicking. What does happen is the
ledger entry, so a start that came from a browser is as recorded as one that
came from an agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from ..aasctl import opc
from ..aasctl.client import AasHttpError
from ..aasctl.paths import ElementPath
from ..kernel.approval import SAFETY_APPROVAL
from .aas_routes import _client
from .write_routes import _recorded

router = APIRouter()

# A console polls. Bounded so a browser left open on a dashboard cannot hold a
# session open against a PLC for longer than the operator is looking at it.
READ_TIMEOUT = 8.0


def _qualifier(element: Dict[str, Any], kind: str) -> Optional[str]:
    for qual in element.get("qualifiers") or []:
        if qual.get("type") == kind:
            return str(qual.get("value"))
    return None


def _walk(node: Dict[str, Any], prefix: str, found: List[Dict[str, Any]]) -> None:
    children = node.get("submodelElements")
    if children is None:
        children = node.get("value") if isinstance(node.get("value"), list) else []
    for child in children or []:
        if not isinstance(child, dict):
            continue
        id_short = child.get("idShort") or ""
        path = f"{prefix}.{id_short}" if prefix else id_short
        source = _qualifier(child, opc.VALUE_SOURCE)
        if source:
            endpoint, node_id = opc.split_target(source)
            found.append({
                "path": path,
                "idShort": id_short,
                "endpoint": endpoint,
                "node": node_id,
                # The PLC's own name for the signal. An operator reading
                # `ConveyorForward` here and `xQA1_A1` in TIA Portal needs to
                # see that they are the same thing.
                "tag": opc.tag_name(node_id),
                "valueType": child.get("valueType"),
                "description": _description(child),
                "group": prefix or "",
            })
        _walk(child, path, found)


def _description(element: Dict[str, Any]) -> str:
    for entry in element.get("description") or []:
        if isinstance(entry, dict) and entry.get("text"):
            return str(entry["text"])
    return ""


def _sources_of(submodel_id: str, endpoint: Optional[str]) -> List[Dict[str, Any]]:
    with _client(endpoint) as client:
        try:
            submodel = client.get_submodel(submodel_id)
        except AasHttpError as exc:
            raise HTTPException(status_code=404,
                                detail=f"no submodel {submodel_id!r}: {exc}") from exc
    found: List[Dict[str, Any]] = []
    _walk(submodel, "", found)
    return found


@router.get("/api/opc/sources/{submodel_id:path}")
def sources(submodel_id: str, endpoint: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Every element in a submodel that names an OPC UA node."""
    found = _sources_of(submodel_id, endpoint)
    return {
        "success": True,
        "submodel": submodel_id,
        "sources": found,
        "endpoints": sorted({f["endpoint"] for f in found}),
    }


@router.post("/api/opc/read")
def read(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Read every value a submodel names, grouped by server.

    One session per server rather than one per node: a console showing seven
    values would otherwise open seven sessions every poll, and a PLC has a
    session limit an operator will meet before they understand why.
    """
    submodel_id = str(body.get("submodel") or "").strip()
    if not submodel_id:
        raise HTTPException(status_code=422, detail="submodel is required")

    found = _sources_of(submodel_id, body.get("endpoint"))
    by_server: Dict[str, List[Dict[str, Any]]] = {}
    for item in found:
        by_server.setdefault(item["endpoint"], []).append(item)

    values: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for server, items in by_server.items():
        try:
            read_back = opc.read(server, [i["node"] for i in items],
                                 timeout=READ_TIMEOUT)
        except opc.OpcError as exc:
            # One unreachable server is not the whole console. The rows it
            # would have filled are reported as unreachable and the rest of the
            # cell still shows, which is what an operator needs when one
            # subnet is down.
            failures.append({"endpoint": server, "error": str(exc),
                             "paths": [i["path"] for i in items]})
            continue
        for item, value in zip(items, read_back):
            values.append({**item, "value": value.value,
                           "dataType": value.data_type,
                           "writable": value.writable})

    return {"success": not failures, "submodel": submodel_id,
            "values": values, "failed": failures}


@router.get("/api/opc/operations/{submodel_id:path}")
def operations(submodel_id: str,
               endpoint: Optional[str] = Query(None)) -> Dict[str, Any]:
    """The Operations a submodel delegates to OPC UA, and whether they need a nod."""
    with _client(endpoint) as client:
        try:
            submodel = client.get_submodel(submodel_id)
        except AasHttpError as exc:
            raise HTTPException(status_code=404,
                                detail=f"no submodel {submodel_id!r}: {exc}") from exc

    found: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any], prefix: str) -> None:
        children = node.get("submodelElements")
        if children is None:
            children = node.get("value") if isinstance(node.get("value"), list) else []
        for child in children or []:
            if not isinstance(child, dict):
                continue
            id_short = child.get("idShort") or ""
            path = f"{prefix}.{id_short}" if prefix else id_short
            if child.get("modelType") == "Operation":
                delegation = _qualifier(child, "invocationDelegation")
                if delegation and opc.is_opc_url(delegation):
                    server, node_id = opc.split_target(delegation)
                    declared = _qualifier(child, SAFETY_APPROVAL)
                    found.append({
                        "path": path,
                        "idShort": id_short,
                        "endpoint": server,
                        "node": node_id,
                        "tag": opc.tag_name(node_id),
                        "description": _description(child),
                        # None means nobody classified it. The console shows
                        # that as its own state: an undeclared action is not a
                        # safe one, it is one no-one has decided about.
                        "requiresSafetyApproval":
                            None if declared is None else declared.lower() == "true",
                    })
            visit(child, path)

    visit(submodel, "")
    return {"success": True, "submodel": submodel_id, "operations": found}


@router.post("/api/opc/invoke")
def invoke(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Run one Operation the model declares. The browser names the path, not the node."""
    submodel_id = str(body.get("submodel") or "").strip()
    path = str(body.get("path") or "").strip()
    if not submodel_id or not path:
        raise HTTPException(status_code=422, detail="submodel and path are required")

    with _client(body.get("endpoint")) as client:
        try:
            element = client.get_element(submodel_id, ElementPath.parse(path))
        except AasHttpError as exc:
            raise HTTPException(status_code=404,
                                detail=f"no element at {path}: {exc}") from exc

    if element.get("modelType") != "Operation":
        raise HTTPException(
            status_code=422,
            detail=f"{path} is a {element.get('modelType')}, not an Operation",
        )
    delegation = _qualifier(element, "invocationDelegation")
    if not delegation or not opc.is_opc_url(delegation):
        raise HTTPException(
            status_code=422,
            detail=f"{path} does not delegate to an OPC UA node",
        )

    server, node = opc.split_target(delegation)
    if not node:
        raise HTTPException(status_code=422,
                            detail=f"{delegation} names a server but no node")

    value = str(body.get("value", "true"))
    try:
        after = opc.write(server, node, value, timeout=READ_TIMEOUT)
    except opc.OpcError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    unrecorded = _recorded(
        "aas.opc.invoke",
        submodel_id,
        {"path": path, "endpoint": server, "node": node, "wrote": value,
         "readBack": after.value},
    )
    return {
        "success": True,
        "path": path,
        "endpoint": server,
        "node": node,
        "wrote": value,
        "readBack": after.value,
        "dataType": after.data_type,
        # A PLC can accept a write and clear the bit in the same scan. Not a
        # failure, and the console has to be able to say which it was.
        "outcome": after.settled(),
        "explanation": after.explain(),
        "observed": [str(v) for v in after.observed],
        "standing": after.settled() == "stood",
        "unrecorded": unrecorded,
    }

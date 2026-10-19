"""Explicit Festo monitor identity; local demo provisioning stays separate."""

from ..aasctl.catalog import is_blueprint
from ..aasctl.client import Endpoint
from fastapi import HTTPException

BLUEPRINT = "https://fraunhoferIPA/ids/aas/FestoWorkCenterStamping"


def is_festo(shell):
    derived = (shell.get("derivedFrom") or {}).get("keys", [])
    return (
        shell.get("assetInformation", {}).get("assetKind") == "Instance"
        and not is_blueprint(shell)
        and (
            any(key.get("value") == BLUEPRINT for key in derived)
            or shell.get("idShort", "").startswith("FestoWorkCenterStamping_")
        )
    )


def identity(endpoint, asset_id):
    return {"endpoint": Endpoint.from_env(endpoint).base_url, "asset_id": asset_id}


def current_instance(aas):
    """Rediscover on each request; never silently choose between physical assets."""

    rows = []
    params = {"limit": 100}
    while True:
        page = aas.get("/shells", params=params)
        rows.extend(page.get("result", []))
        cursor = page.get("paging_metadata", {}).get("cursor")
        if not cursor:
            break
        params = {"limit": 100, "cursor": cursor}
    candidates = [row for row in rows if is_festo(row) and row["id"] != "urn:a4s:aas:FestoWorkCenterStamping_Demo"]
    if not candidates:
        raise HTTPException(404, "No Festo AAS instance available. Create an instance in Shop floor & AAS.")
    if len(candidates) != 1:
        raise HTTPException(409, "Multiple Festo AAS instances available; cannot choose an operations target unambiguously.")
    return candidates[0]


def instance_submodels(aas, shell):
    models = {}
    for ref in shell.get("submodels", []):
        for key in ref.get("keys", []):
            if key.get("type") != "Submodel":
                continue
            model = aas.get_submodel(key["value"])
            if model.get("kind") == "Template":
                continue
            name = model.get("idShort")
            if name in models and models[name]["id"] != model["id"]:
                raise HTTPException(409, f"Ambiguous {name} references on {shell.get('idShort')}")
            models[name] = model
    return models


def required_submodel(models, shell, name):
    if name not in models:
        raise HTTPException(409, f"Festo instance {shell.get('idShort')} is missing {name}; configure this instance's submodel before using this feature.")
    return models[name]

"""Operations discovery follows replacement shells and their actual references."""
from contextlib import nullcontext

import pytest
from fastapi import HTTPException

from a4s.server import festo_assets as assets, festo_demo_routes as demo
ASSET_ID = "urn:a4s:aas:FestoWorkCenterStamping_Demo"


def shell(number, kind="Instance"):
    return {"id": f"urn:festo:{number}", "idShort": f"FestoWorkCenterStamping_{number}",
            "assetInformation": {"assetKind": kind}, "submodels": []}


class Repository:
    def __init__(self, rows, models=None):
        self.rows, self.models = rows, models or {}

    def get(self, path, params):
        assert path == "/shells"
        # Exercise pagination, including a candidate outside the first page.
        offset = int(params.get("cursor", 0))
        return {"result": self.rows[offset:offset + 1], "paging_metadata":
                {"cursor": str(offset + 1)} if offset + 1 < len(self.rows) else {}}

    def get_submodel(self, identifier):
        return self.models[identifier]


def test_replacement_is_discovered_without_cached_id():
    simulator = {**shell("Demo"), "id": ASSET_ID}
    repo = Repository([shell("template", "Type"), simulator, shell("381")])
    assert assets.current_instance(repo)["id"] == "urn:festo:381"
    repo.rows[-1] = shell("967")
    assert assets.current_instance(repo)["id"] == "urn:festo:967"


@pytest.mark.parametrize("rows,status", [([], 404), ([shell("1"), shell("2")], 409)])
def test_absent_or_ambiguous_target_fails_explicitly(rows, status):
    with pytest.raises(HTTPException) as exc:
        assets.current_instance(Repository(rows))
    assert exc.value.status_code == status


def test_snapshot_reads_referenced_models_from_replacement(monkeypatch):
    current = shell("967")
    models = {f"urn:model:{name}": {"id": f"urn:model:{name}", "idShort": name,
              "submodelElements": []} for name in ("ProductionContext", "MaintenanceInstructions")}
    current["submodels"] = [{"keys": [{"type": "Submodel", "value": key}]} for key in models]
    repo = Repository([current], models)
    monkeypatch.setenv("A4S_AAS_ENDPOINT", "http://localhost:8081")
    monkeypatch.setattr(demo, "BasyxClient", lambda _: nullcontext(repo))
    # Stop after AAS resolution, before any MES network request.
    monkeypatch.setattr(demo.Config, "from_env", lambda: type("Config", (), {"endpoint": "trusted"})())
    with pytest.raises(HTTPException, match="trusted MES service"):
        demo.snapshot()
    repo.models.pop("urn:model:ProductionContext")
    current["submodels"] = current["submodels"][1:]
    with pytest.raises(HTTPException, match="967.*missing ProductionContext"):
        demo.snapshot()
    assert demo.read_asset()["asset"]["id"] == current["id"]

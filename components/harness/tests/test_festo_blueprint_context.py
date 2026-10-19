"""Blueprint recreation retains configured MES linkage without reusing asset IDs."""
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from a4s.server import build_app
from a4s.server.write_routes import _copy_submodel

ROOT = Path(__file__).resolve().parents[1]


def test_saved_blueprint_instantiates_business_templates_for_each_new_asset():
    document = json.loads((ROOT / "BaSyxMinimal/aas/festo-cp-lab.json").read_text(encoding="utf-8"))
    shell = document["assetAdministrationShells"][0]
    refs = {k["value"] for r in shell["submodels"] for k in r["keys"]}
    for name in ("ProductionContext", "MaintenanceInstructions"):
        source = next(m for m in document["submodels"] if m["idShort"] == name)
        original = copy.deepcopy(source)
        assert source["kind"] == "Template" and source["id"] in refs
        for number in ("new1", "new2"):
            identifier = f"urn:test:festo:{number}"
            clone = _copy_submodel(source, identifier)
            assert clone["kind"] == "Instance"
            assert clone["id"] == f"{identifier}/sm/{name}"
            if name == "ProductionContext":
                values = {e["idShort"]: e.get("value") for e in clone["submodelElements"]}
                assert values["MESBaseUrl"] == "http://localhost:8092"
                assert values["MESLineCode"] == "A4S-FESTO"
                assert "AffectedOrderId" not in values and "UpcomingOrderId" not in values
                assert values["MaintenanceSubmodelId"] == f"{identifier}/sm/MaintenanceInstructions"
            else:
                assert clone["submodelElements"] == source["submodelElements"]
        assert source == original


def test_ordinary_template_examples_are_still_cleared():
    source = {"modelType": "Submodel", "kind": "Template", "id": "urn:template",
              "idShort": "Example", "submodelElements": [{"modelType": "Property",
              "idShort": "ExampleValue", "valueType": "xs:string", "value": "not real data"}]}
    clone = _copy_submodel(source, "urn:new")
    assert not clone["submodelElements"][0].get("value")


@pytest.mark.parametrize("ui,endpoint,expected", [
    ("", "http://localhost:8092", "http://localhost:8092"),
    ("https://mes.example/ui", "http://internal:8000", "https://mes.example/ui"),
    ("", "", None),
    ("http://user:secret@localhost", "http://localhost:8092", None),
])
def test_service_link_requires_no_aas_or_mes_connection(monkeypatch, ui, endpoint, expected):
    monkeypatch.setenv("A4S_MES_UI_URL", ui)
    monkeypatch.setenv("A4S_MES_ENDPOINT", endpoint)
    client = TestClient(build_app("http://unreachable.invalid"))
    assert client.get("/api/service-links").json() == {"mes_url": expected}

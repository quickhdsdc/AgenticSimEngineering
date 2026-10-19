"""Exercise the MES-only reset with no AAS service and no real business writes."""
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from a4s.server.write_routes import _copy_submodel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reset(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("reset_plan", ROOT / "scripts/reset-production-plan.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    from a4s.kernel import config
    from a4s.mesctl import client, commands, executor
    from a4s.aasctl import client as aas

    def forbidden(*args, **kwargs):
        raise AssertionError("MES reset must not connect to AAS")

    monkeypatch.setattr(aas, "BasyxClient", forbidden)
    monkeypatch.setattr(config, "load_env", lambda _: None)
    monkeypatch.setattr(client.Config, "from_env", lambda: SimpleNamespace(endpoint="http://localhost:8092", server_timezone="Europe/Berlin"))
    from contextlib import nullcontext
    monkeypatch.setattr(client, "MesClient", lambda _: nullcontext(None))
    monkeypatch.setattr(commands, "build_parser", lambda: SimpleNamespace(parse_args=lambda words: words))
    state = {"commands": [], "orders": [], "processes": []}

    def execute(words, _):
        state["commands"].append(words)
        if words[:2] == ["order", "create"]:
            order = {"id": 101 + len(state["orders"]), "order_no": words[words.index("--order-no") + 1],
                     "status": "ACCEPTED", "planned_start_at": "future", "planned_end_at": "later"}
            state["orders"].append(order)
            return {"verified": True, "read_back": {"data": order}}
        if words[:2] == ["order", "show"]:
            return {"data": next(o for o in state["orders"] if str(o["id"]) == words[2])}
        return {"verified": True}

    monkeypatch.setattr(executor, "execute", execute)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/openmes-reset-plan.php").write_text("test reset transaction")

    def run(command, **kwargs):
        state["processes"].append(command)
        if "pg_dump" in command:
            kwargs["stdout"].write(b"test PostgreSQL backup")
            return SimpleNamespace(returncode=0)
        assert list(tmp_path.glob("temp/production-runs/*/openmes-before.dump"))
        return SimpleNamespace(stdout='A4S_RESET=' + json.dumps({"line_id": 1, "backup_line_id": 2,
            "product_id": 1, "archived_orders": 2, "archived_maintenance": 1}), stderr="")

    monkeypatch.setattr(script.subprocess, "run", run)
    return script, state, tmp_path


def test_reset_succeeds_without_aas_and_new_instance_keeps_static_binding(reset):
    script, state, root = reset
    script.main(["--apply"])
    manifest_path = root / "temp/production-runs/current.json"
    manifest = json.loads(manifest_path.read_text())
    assert [o["id"] for o in manifest["baseline"]] == [101, 102]
    assert len(state["processes"]) == 2
    assert len(state["commands"]) == 8
    for create, move in zip((command for command in state["commands"] if command[:2] == ["order", "create"]),
                            (command for command in state["commands"] if command[:2] == ["schedule", "move"])):
        end = datetime.fromisoformat(move[move.index("--end") + 1])
        assert create[create.index("--due-date") + 1] == end.astimezone(ZoneInfo("Europe/Berlin")).date().isoformat()
    assert Path(manifest["backup"]).read_bytes() == b"test PostgreSQL backup"
    document = json.loads((ROOT / "BaSyxMinimal/aas/festo-cp-lab.json").read_text(encoding="utf-8"))
    source = next(m for m in document["submodels"] if m["idShort"] == "ProductionContext")
    original = json.dumps(source)
    instance = _copy_submodel(source, "urn:new:festo")
    values = {e["idShort"]: e.get("value") for e in instance["submodelElements"]}
    assert values["MESLineCode"] == "A4S-FESTO"
    assert values["MESBackupLineCode"] == "A4S-BACKUP"
    assert "AffectedOrderId" not in values and "UpcomingOrderId" not in values
    assert "ProductionRunId" not in values
    assert values["MaintenanceSubmodelId"] == "urn:new:festo/sm/MaintenanceInstructions"
    assert json.dumps(source) == original


def test_preview_does_not_access_services_or_create_files(reset):
    script, state, root = reset
    script.main([])
    assert state["processes"] == state["commands"] == []
    assert not (root / "temp").exists()


def test_backup_failure_stops_before_any_mes_write(reset, monkeypatch):
    script, state, root = reset

    def fail(*args, **kwargs):
        raise script.subprocess.CalledProcessError(1, "pg_dump")

    monkeypatch.setattr(script.subprocess, "run", fail)
    with pytest.raises(script.subprocess.CalledProcessError):
        script.main(["--apply"])
    assert state["commands"] == []
    assert not (root / "temp/production-runs/current.json").exists()


def test_instance_binding_does_not_depend_on_manifest(reset):
    _, _, root = reset
    source = next(m for m in json.loads((ROOT / "BaSyxMinimal/aas/festo-cp-lab.json").read_text(encoding="utf-8"))["submodels"] if m["idShort"] == "ProductionContext")
    assert not (root / "temp/production-runs/current.json").exists()
    assert _copy_submodel(source, "urn:new:festo")["submodelElements"]

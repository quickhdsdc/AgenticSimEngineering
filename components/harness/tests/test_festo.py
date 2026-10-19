"""The Festo cell: what is modelled, and what the recorder watches.

No server and no PLC. What has to be right without one is the shape of the
model -- which node a delegation names, which values are readable, whether a
control operation carries a safety declaration -- and that the recorder writes
a timeline somebody can read back.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from a4s.aasctl import opc
from a4s.kernel.approval import SAFETY_APPROVAL
from a4s.machines.festo_cp_lab import model


def _elements(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    children = node.get("submodelElements")
    if children is None:
        children = node.get("value") if isinstance(node.get("value"), list) else []
    for child in children or []:
        if isinstance(child, dict):
            out.append(child)
            _elements(child, out)


def _qualifier(element: Dict[str, Any], kind: str):
    for qual in element.get("qualifiers") or []:
        if qual.get("type") == kind:
            return qual.get("value")
    return None


# ---------------------------------------------------------------------------
# one shell, because it is one asset
# ---------------------------------------------------------------------------

def test_the_work_center_is_one_shell() -> None:
    """Two work cells are not two assets: an operator buys, installs and
    decommissions the whole thing, and two shells would leave nothing holding
    the nameplate."""
    environment = model.build()
    assert len(environment["assetAdministrationShells"]) == 1
    shell = environment["assetAdministrationShells"][0]
    assert shell["idShort"] == "FestoWorkCenterStamping"
    assert len(shell["submodels"]) == len(environment["submodels"])


def test_each_opc_server_gets_its_own_data_sheet() -> None:
    """Three servers, three submodels, distinct idShorts. Identical ones
    collide when the shell is instantiated -- the new submodel id is built from
    the shell id and the idShort -- and are indistinguishable in a listing."""
    sheets = [s for s in model.build()["submodels"]
              if s["idShort"].startswith("UAServerDataSheet")]
    assert len(sheets) == len(model.SERVERS)
    assert len({s["idShort"] for s in sheets}) == len(sheets)
    assert {s["semanticId"]["keys"][0]["value"] for s in sheets} == {
        model.UA_DATASHEET_SMT
    }


# ---------------------------------------------------------------------------
# control goes to the remote-control surface, not the module's own inputs
# ---------------------------------------------------------------------------

def _operations() -> Dict[str, Dict[str, Any]]:
    services = next(s for s in model.build()["submodels"]
                    if s["idShort"] == "services_delegated")
    found: List[Dict[str, Any]] = []
    _elements(services, found)
    return {e["idShort"]: e for e in found if e.get("modelType") == "Operation"}


@pytest.mark.parametrize("cell", ["Press", "Turning"])
def test_start_and_stop_use_the_remote_control_block(cell: str) -> None:
    """The module's own start bit was written four times and moved nothing. The
    station's remote-control surface is `dbVar`, found in its own configuration
    rather than inferred from bit names."""
    operations = _operations()
    for verb, node in (("Start", "RemoteControlAutomatic_Start"),
                       ("Stop", "RemoteControlAutomatic_Stop")):
        delegation = _qualifier(operations[f"{verb}{cell}"], "invocationDelegation")
        assert "dbVar" in delegation
        assert node in delegation
        assert opc.is_opc_url(delegation)


def test_the_module_inputs_are_kept_but_named_apart() -> None:
    """They are real and their state explains a station that will not start.
    Prefixed so nobody reaches for ModuleStart when they want Start."""
    operations = _operations()
    assert "ModuleStartPress" in operations
    assert "ApplicationModule_DB" in _qualifier(
        operations["ModuleStartPress"], "invocationDelegation")


def test_every_control_operation_declares_that_it_needs_approval() -> None:
    """Selecting a mode is not a reading, and on a cell that can move a mode
    change is the precondition for movement rather than separate from it."""
    for name, element in _operations().items():
        assert _qualifier(element, SAFETY_APPROVAL) == "true", name


def test_a_control_operation_takes_a_value() -> None:
    """A mode bit is a level: deselecting auto means writing false, and an
    Operation that could only write true would model half of it."""
    start = _operations()["StartPress"]
    assert [v["value"]["idShort"] for v in start["inputVariables"]] == ["value"]


# ---------------------------------------------------------------------------
# readable values, and the units that are not claimed
# ---------------------------------------------------------------------------

def test_readings_name_their_source_and_need_no_approval() -> None:
    """Reading a pressure is not an action. Modelling it as an Operation would
    put every measurement through the approval gate."""
    operational = next(s for s in model.build()["submodels"]
                       if s["idShort"] == "OperationalData")
    found: List[Dict[str, Any]] = []
    _elements(operational, found)
    sources = [e for e in found if _qualifier(e, opc.VALUE_SOURCE)]
    assert len(sources) >= 20
    for element in sources:
        assert _qualifier(element, SAFETY_APPROVAL) == "false", element["idShort"]
        assert opc.is_opc_url(_qualifier(element, opc.VALUE_SOURCE))


def test_the_sensor_readings_claim_no_unit() -> None:
    """A scale was invented once and the agent caught it, reading the raw count
    back as an implausible pressure for a pneumatic cell. Nothing states what
    the counts mean, so nothing here says."""
    operational = next(s for s in model.build()["submodels"]
                       if s["idShort"] == "OperationalData")
    found: List[Dict[str, Any]] = []
    _elements(operational, found)
    pressure = next(e for e in found if e["idShort"] == "Pressure1")
    assert _qualifier(pressure, "unit") is None
    assert "not established" in pressure["description"][0]["text"]


# ---------------------------------------------------------------------------
# the recorder leaves a timeline
# ---------------------------------------------------------------------------

def test_the_recorder_watches_more_than_the_model_carries() -> None:
    """A diagnosis needs the safety inputs and the fault word; the model
    carries what an operator needs. The collective fault bit was true for a day
    while the application error bit stayed false, which is why the wider list
    exists."""
    from a4s.machines.festo_cp_lab import recorder

    names = {n for n, _ in recorder.PLC_NODES}
    assert "err.xSF" in names
    assert "in.xSF5_estop" in names
    assert "app.xError" in names
    assert {t.station for t in recorder.targets()} == set(model.SERVERS)


def test_a_timeline_records_changes_and_not_every_sample(tmp_path: Path) -> None:
    from a4s.machines.festo_cp_lab import recorder

    db = recorder.open_db(tmp_path / "festo.db")
    now = time.time()
    db.executemany(
        "INSERT INTO readings (ts, station, signal, node, value, kind)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [(now, "PressPLC", "app.xBusy", "n", "False", "change"),
         (now + 1, "PressPLC", "app.xBusy", "n", "True", "change"),
         (now + 2, "PressPLC", "app.xBusy", "n", "True", "heartbeat")],
    )
    db.commit()

    changes = recorder.timeline(db)
    assert [r["value"] for r in changes] == ["False", "True"]
    assert recorder.summary(db)["rows"] == 3


def test_a_note_marks_the_timeline(tmp_path: Path) -> None:
    """A note beats guessing which second an operator acted."""
    from a4s.machines.festo_cp_lab import recorder

    db = recorder.open_db(tmp_path / "festo.db")
    recorder.note(db, "about to press start")
    db.row_factory = sqlite3.Row
    assert db.execute("SELECT text FROM notes").fetchone()["text"] == \
        "about to press start"


def test_a_dropped_server_is_recorded_rather_than_leaving_a_blank(
    tmp_path: Path,
) -> None:
    """A gap in a timeline is unreadable unless it says why it is there."""
    from a4s.machines.festo_cp_lab import recorder

    db = recorder.open_db(tmp_path / "festo.db")
    db.execute(
        "INSERT INTO readings (ts, station, signal, node, value, kind)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), "PressPLC", "_connection", "opc.tcp://x",
         "TimeoutError", "error"))
    db.commit()
    assert recorder.summary(db)["errors"] == 1

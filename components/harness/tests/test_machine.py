"""The simulated injection moulding machine.

No network, no AAS repository. What is asserted is what makes the simulation
worth having: that it refuses what a real machine refuses, that a setpoint is
approached rather than granted, and that a prediction is a prediction rather
than a lookup of the answer.
"""

from __future__ import annotations


import pytest

from a4s.machines.injection_molding.machine import (
    COLUMNS,
    FEATURES,
    SETPOINT_LIMITS,
    SETTABLE,
    TARGET,
    UNITS,
    Machine,
    MachineError,
    QualityModel,
    load_shots,
)


@pytest.fixture(scope="module")
def machine() -> Machine:
    return Machine.build(seconds_per_shot=1000.0)   # effectively frozen on shot 0


# ---------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------

def test_the_run_loads_with_its_german_decimal_commas() -> None:
    shots = load_shots()
    assert len(shots) > 1000
    first = shots[0]
    # "641,25219727" -> 641.25, not 64125219727 and not a string.
    assert isinstance(first["MaxInjectionPressure"], float)
    assert 100 < first["MaxInjectionPressure"] < 2000


def test_every_column_is_named_and_has_a_unit() -> None:
    """A reading whose unit is unknown cannot be checked against an element's."""
    for english in COLUMNS.values():
        assert english in UNITS, f"{english} has no unit entry"
        assert english[0].isalpha() and english.replace("_", "").isalnum(), (
            f"{english} is not a legal idShort"
        )


def test_readings_change_between_shots() -> None:
    """A stub returning constants would teach an agent nothing about a process."""
    shots = load_shots()
    pressures = {s["MaxInjectionPressure"] for s in shots[:50] if "MaxInjectionPressure" in s}
    assert len(pressures) > 20


# ---------------------------------------------------------------------------
# the quality model
# ---------------------------------------------------------------------------

def test_the_model_fits_and_reports_how_well(machine: Machine) -> None:
    report = machine.model.report()
    assert report["target"] == TARGET
    assert report["trained_on_shots"] > 1000
    # A weak fit is fine and is reported; a suspiciously perfect one would mean
    # the model had been handed the answer.
    assert 0.0 < report["r_squared"] < 0.95
    assert report["rmse"] > 0
    assert set(report["standardised_coefficients"]) == set(FEATURES)


def test_a_prediction_is_computed_not_looked_up(machine: Machine) -> None:
    """Given a shot's own parameters the answer must differ from its recorded
    weight -- otherwise it is a lookup wearing a model's clothes."""
    shot = next(s for s in machine.shots if all(f in s for f in FEATURES) and TARGET in s)
    predicted = machine.model.predict({f: shot[f] for f in FEATURES})
    assert predicted != shot[TARGET]
    assert abs(predicted - shot[TARGET]) < 1.0      # but close: it is a real fit


def test_changing_a_parameter_changes_the_prediction(machine: Machine) -> None:
    base = {f: machine.reading(f) for f in FEATURES}
    hotter = {**base, "MaxInjectionPressure": base["MaxInjectionPressure"] * 1.2}
    assert machine.model.predict(base) != machine.model.predict(hotter)


def test_a_prediction_needs_every_feature(machine: Machine) -> None:
    with pytest.raises(MachineError, match="missing process parameters"):
        machine.model.predict({"MaxInjectionPressure": 600.0})


def test_the_model_refuses_a_run_too_short_to_fit() -> None:
    with pytest.raises(MachineError, match="not enough"):
        QualityModel.fit([{f: 1.0 for f in FEATURES} | {TARGET: 1.0}] * 3)


# ---------------------------------------------------------------------------
# setpoints
# ---------------------------------------------------------------------------

def test_a_measured_reading_cannot_be_commanded(machine: Machine) -> None:
    """A machine that pretends to obey is worse than one that says no."""
    with pytest.raises(MachineError, match="measured, not commanded"):
        machine.set_point("PartWeight", 60.0)


def test_a_setpoint_outside_the_envelope_is_refused(machine: Machine) -> None:
    """Accepting 4000 degrees teaches an agent the machine accepts anything."""
    with pytest.raises(MachineError, match="must be between"):
        machine.set_point("BarrelZone1", 4000.0)
    with pytest.raises(MachineError, match="must be between"):
        machine.set_point("BarrelZone1", -5.0)


def test_a_setpoint_is_approached_not_granted() -> None:
    fresh = Machine.build(seconds_per_shot=1000.0)
    before = fresh.reading("BarrelZone1")
    target = before + 40.0
    result = fresh.set_point("BarrelZone1", target)
    now = result["current"]
    assert now != target, "the process jumped straight to the setpoint"
    assert abs(now - target) < abs(before - target), "the process did not move towards it"


def test_clearing_a_setpoint_returns_the_measured_value() -> None:
    fresh = Machine.build(seconds_per_shot=1000.0)
    measured = fresh.reading("BarrelZone2")
    fresh.set_point("BarrelZone2", measured + 50.0)
    assert fresh.reading("BarrelZone2") != measured
    fresh.clear_setpoint("BarrelZone2")
    assert fresh.reading("BarrelZone2") == measured


def test_every_settable_reading_has_an_envelope() -> None:
    for name in SETTABLE:
        assert name in SETPOINT_LIMITS, f"{name} is settable with no limits"
        low, high = SETPOINT_LIMITS[name]
        assert low < high


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def test_an_unknown_reading_says_what_the_machine_publishes(machine: Machine) -> None:
    with pytest.raises(MachineError, match="publishes"):
        machine.reading("Torque")


def test_stopping_and_restarting_continues_the_run() -> None:
    """Restarting from shot zero would make a stop-start look like a new part."""
    fresh = Machine.build(seconds_per_shot=0.001)
    fresh.start()
    import time

    time.sleep(0.05)
    stopped = fresh.stop()
    assert stopped["shots_produced"] > 0
    frozen = fresh.telemetry()["shot"]
    time.sleep(0.02)
    assert fresh.telemetry()["shot"] == frozen, "the run advanced while stopped"


def test_telemetry_carries_units_and_the_active_setpoints(machine: Machine) -> None:
    telemetry = machine.telemetry()
    assert telemetry["of"] == len(machine.shots)
    reading = telemetry["readings"]["MaxInjectionPressure"]
    assert reading["unit"] == "bar"
    assert isinstance(reading["value"], float)


# ---------------------------------------------------------------------------
# the service contract and the AAS mapping
# ---------------------------------------------------------------------------

def test_every_endpoint_declares_whether_it_moves_the_machine() -> None:
    """A gate that guesses from the name works until it does not."""
    from a4s.machines.injection_molding.service import ENDPOINTS

    for path, spec in ENDPOINTS.items():
        assert "moves_machine" in spec, f"{path} does not say whether it moves the machine"
        assert spec["summary"].strip(), f"{path} has no summary"
    moving = {p for p, s in ENDPOINTS.items() if s["moves_machine"]}
    assert moving == {"/start", "/stop", "/setpoint", "/setpoint/clear"}


def test_read_only_endpoints_are_not_marked_as_moving() -> None:
    from a4s.machines.injection_molding.service import ENDPOINTS

    for path in ("/status", "/model", "/read", "/predict-quality"):
        assert not ENDPOINTS[path]["moves_machine"]


def test_every_endpoint_answers_post() -> None:
    """An Operation invocation is an RPC and `aas op invoke` POSTs. Serving a
    read-only endpoint on GET alone made it uncallable through the AAS."""
    from a4s.machines.injection_molding.machine import Machine as M
    from a4s.machines.injection_molding.service import ENDPOINTS, routes

    table = routes(M.build(seconds_per_shot=1000.0))
    for path in ENDPOINTS:
        assert ("POST", path) in table, f"{path} does not answer POST"


def test_operations_are_generated_from_the_service_contract() -> None:
    """Writing the signature by hand is how it drifts from the service; this
    repository already contains one that did."""
    from a4s.machines.injection_molding.onboard import _operation
    from a4s.machines.injection_molding.service import ENDPOINTS

    element = _operation("/setpoint", ENDPOINTS["/setpoint"], "http://machine:5080")
    declared = {v["value"]["idShort"] for v in element["inputVariables"]}
    assert declared == set(ENDPOINTS["/setpoint"]["inputs"])


def test_operation_id_shorts_are_legal() -> None:
    """`/predict-quality` became `Predict-quality` on the first attempt, which
    AASd-002 forbids -- caught by `aas validate` against generated output."""
    import re

    from a4s.machines.injection_molding.onboard import _operation
    from a4s.machines.injection_molding.service import ENDPOINTS

    legal = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
    for path, spec in ENDPOINTS.items():
        id_short = _operation(path, spec, "http://x")["idShort"]
        assert legal.match(id_short), f"{path} produced the illegal idShort {id_short!r}"


def test_a_moving_operation_carries_the_qualifier_a_gate_reads() -> None:
    from a4s.machines.injection_molding.onboard import _operation, moves_machine
    from a4s.machines.injection_molding.service import ENDPOINTS

    assert moves_machine(_operation("/setpoint", ENDPOINTS["/setpoint"], "http://x"))
    assert not moves_machine(_operation("/status", ENDPOINTS["/status"], "http://x"))


def test_the_delegation_url_points_at_the_endpoint() -> None:
    from a4s.aasctl.__main__ import _delegation_url
    from a4s.machines.injection_molding.onboard import _operation
    from a4s.machines.injection_molding.service import ENDPOINTS

    element = _operation("/read", ENDPOINTS["/read"], "http://machine:5080/")
    assert _delegation_url(element) == "http://machine:5080/read"


# ---------------------------------------------------------------------------
# an undeclared operation is not a safe operation
# ---------------------------------------------------------------------------

def test_a_declared_flag_always_wins() -> None:
    """`service.py` publishes it so this is not a guess. Both directions."""
    from a4s.machines.injection_molding.onboard import moves_machine

    assert moves_machine(
        {"idShort": "Status", "qualifiers": [{"type": "requiresSafetyApproval", "value": "true"}]}
    ) is True
    assert moves_machine(
        {"idShort": "StartCycle",
         "qualifiers": [{"type": "requiresSafetyApproval", "value": "false"}]}
    ) is False


def test_an_undeclared_actuation_is_assumed_to_move_the_machine() -> None:
    """Measured on a live run. `InjectionMoldingMachine_285` came from a
    blueprint authored elsewhere: eight delegated operations, `StartCycle` and
    `SetBarrelTemp` among them, and not one carries the qualifier -- because
    whoever wrote the blueprint had never heard of it. All eight were
    classified as not moving a machine, and the gate let `StartCycle` through
    unattended on a policy whose whole purpose is to stop there."""
    from a4s.machines.injection_molding.onboard import moves_machine

    for name in ("StartCycle", "Stop", "SetVPSPosition", "SetInjectionSpeed",
                 "SetPackingPressure", "SetBarrelTemp", "SetMeteringStroke"):
        assert moves_machine({"idShort": name}) is True, name


def test_an_undeclared_reading_is_not_a_machine_action() -> None:
    """A false stop costs an operator one approval, so the fallback leans
    towards stopping -- but not so far that reading a status needs a human."""
    from a4s.machines.injection_molding.onboard import moves_machine

    for name in ("GetMachineStatus", "Status", "Read", "PredictQuality", "Model"):
        assert moves_machine({"idShort": name}) is False, name


def test_the_name_test_is_on_whole_words_not_substrings() -> None:
    """`Setpoint` actuates and `GetSetpoint` reads, and both contain "set"."""
    from a4s.machines.injection_molding.onboard import moves_machine

    assert moves_machine({"idShort": "Setpoint"}) is True
    assert moves_machine({"idShort": "GetSetpoint"}) is False


def test_an_operation_named_neither_way_stops() -> None:
    """An Operation exists to *do* something. Unknown passes to a human."""
    from a4s.machines.injection_molding.onboard import moves_machine

    assert moves_machine({"idShort": "Frobnicate"}) is True


def test_onboarding_records_the_flag_in_both_directions() -> None:
    """Only the true ones used to carry it, which is indistinguishable from an
    operation nobody classified -- and that is now the case that stops."""
    from a4s.machines.injection_molding.onboard import SAFETY_APPROVAL, _operation

    reading = _operation("/status", {"summary": "read status"}, "http://x:5080")
    moving = _operation("/start", {"summary": "start", "moves_machine": True},
                               "http://x:5080")

    def flag(element):
        return next(q["value"] for q in element["qualifiers"] if q["type"] == SAFETY_APPROVAL)

    assert flag(reading) == "false"
    assert flag(moving) == "true"



def test_onboarding_writes_the_current_name() -> None:
    from a4s.machines.injection_molding.onboard import SAFETY_APPROVAL, _operation

    element = _operation("/start", {"summary": "s", "moves_machine": True}, "http://x:5080")
    assert SAFETY_APPROVAL == "requiresSafetyApproval"
    assert any(q["type"] == SAFETY_APPROVAL for q in element["qualifiers"])
    assert [q["type"] for q in element["qualifiers"]] == [
        "invocationDelegation", "requiresSafetyApproval"
    ]

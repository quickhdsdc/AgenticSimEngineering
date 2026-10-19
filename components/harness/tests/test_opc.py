"""OPC UA as a transport: target parsing, type coercion, and the gate.

No server. What has to be right here is what happens *before* a byte leaves
the machine -- which node a delegation names, what a string becomes when the
node is a Boolean, and whether a write can reach a PLC without an operator
seeing it. Those are the parts that are hard to reproduce against real
hardware on demand, and the parts where being wrong starts something.
"""

from __future__ import annotations

import pytest

from a4s.aasctl import opc
from a4s.kernel.approval import MOVING, READ, Gate, Policy, classify

pytest.importorskip("asyncua", reason="the opc extra is not installed")


# ---------------------------------------------------------------------------
# one delegation mechanism, two transports
# ---------------------------------------------------------------------------

def test_an_opc_delegation_is_recognised_by_its_scheme() -> None:
    assert opc.is_opc_url("opc.tcp://192.0.2.10:4042#ns=1;s=X")
    assert opc.is_opc_url("OPC.TCP://host:4840")
    assert not opc.is_opc_url("http://localhost:8001/api/start")
    assert not opc.is_opc_url(None)


def test_the_node_id_comes_out_of_the_fragment_intact() -> None:
    """An S7 node id is full of quotes and dots. Mangling it addresses a
    different tag, or none, and the server's error names neither."""
    endpoint, node = opc.split_target(
        'opc.tcp://192.0.2.10:4042#ns=3;s="ApplicationModule_DB"."xStart"'
    )
    assert endpoint == "opc.tcp://192.0.2.10:4042"
    assert node == 'ns=3;s="ApplicationModule_DB"."xStart"'


def test_a_delegation_with_no_node_is_reported_not_guessed() -> None:
    endpoint, node = opc.split_target("opc.tcp://192.0.2.10:4042")
    assert endpoint == "opc.tcp://192.0.2.10:4042"
    assert node is None


def test_a_numeric_node_id_survives_too() -> None:
    _, node = opc.split_target("opc.tcp://host:4840#ns=1;i=2258")
    assert node == "ns=1;i=2258"


# ---------------------------------------------------------------------------
# the type is read from the node, and a boolean is not guessed
# ---------------------------------------------------------------------------

def _variant(name: str):
    from asyncua import ua

    return getattr(ua.VariantType, name)


@pytest.mark.parametrize(
    "text,expected",
    [("true", True), ("TRUE", True), ("1", True), ("on", True), ("yes", True),
     ("false", False), ("False", False), ("0", False), ("off", False), ("no", False)],
)
def test_a_boolean_is_read_the_way_a_person_wrote_it(text: str, expected: bool) -> None:
    assert opc._coerce(text, _variant("Boolean")) is expected


@pytest.mark.parametrize("text", ["maybe", "2", "", "  ", "start"])
def test_anything_else_is_refused_rather_than_made_true(text: str) -> None:
    """`bool("false")` is `True` in Python. A start command that becomes True
    because the operator typed something else is the worst failure this
    function has available to it."""
    with pytest.raises(opc.OpcError, match="not a boolean"):
        opc._coerce(text, _variant("Boolean"))


def test_an_integer_node_refuses_a_word() -> None:
    assert opc._coerce("5170", _variant("Int16")) == 5170
    with pytest.raises(opc.OpcError, match="not an integer"):
        opc._coerce("true", _variant("Int16"))


def test_a_float_node_takes_a_number_and_not_a_word() -> None:
    assert opc._coerce("2.5", _variant("Double")) == 2.5
    with pytest.raises(opc.OpcError, match="not a number"):
        opc._coerce("fast", _variant("Double"))


def test_a_type_this_tool_has_not_met_is_passed_through() -> None:
    """Refusing here would block node types nobody has needed yet, for no
    gain: the server rejects what it cannot accept, and says so."""
    assert opc._coerce("anything", _variant("XmlElement")) == "anything"


# ---------------------------------------------------------------------------
# a write to a PLC does not get to be quieter than an invocation
# ---------------------------------------------------------------------------

def test_reading_a_node_is_not_a_machine_action() -> None:
    for command in ("aas opc read opc.tcp://h:4840 nodeX",
                    "aas opc browse opc.tcp://h:4840",
                    "aas opc info opc.tcp://h:4840"):
        assert classify("shell", {"command": command}).risk == READ, command


def test_writing_a_node_is_treated_as_moving_a_machine() -> None:
    """These verbs take a raw endpoint and node, so there is no AAS element to
    read a declaration off. What cannot be classified is treated as dangerous
    -- the same answer `_invoke_target` gives when it cannot parse a target."""
    for command in ("aas opc write opc.tcp://h:4840 nodeX true",
                    "aas opc call opc.tcp://h:4840 nodeM"):
        assert classify("shell", {"command": command}).risk == MOVING, command


def test_the_gate_refuses_an_opc_write_with_nobody_to_ask() -> None:
    gate = Gate(policy=Policy.from_env(), ledger=None, approver=None)
    assert gate.check("shell", {"command": "aas opc read opc.tcp://h:1 n"}).allowed
    denied = gate.check("shell", {"command": "aas opc write opc.tcp://h:1 n true"})
    assert not denied.allowed
    assert "approval" in denied.reason


def test_a_write_hidden_after_a_read_is_still_seen() -> None:
    """The classifier takes the worst segment, so a second command on the line
    cannot ride in under the first one's risk."""
    command = "aas opc read opc.tcp://h:1 n && aas opc write opc.tcp://h:1 n true"
    assert classify("shell", {"command": command}).risk == MOVING


# ---------------------------------------------------------------------------
# the extra is optional, and its absence is a message rather than a traceback
# ---------------------------------------------------------------------------

def test_a_missing_library_names_the_extra_that_provides_it(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "asyncua":
            raise ImportError("no module named asyncua")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(opc.OpcError, match=r"\[opc\]"):
        opc._run(None)


# ---------------------------------------------------------------------------
# a write is watched, not read once
# ---------------------------------------------------------------------------

def _written(before, wanted, observed) -> opc.NodeValue:
    return opc.NodeValue(node_id="n", value=observed[-1] if observed else None,
                         data_type="Boolean", writable=True,
                         before=before, wanted=wanted, observed=list(observed))


def test_a_level_that_is_still_set_stood() -> None:
    assert _written(False, True, [True, True, True]).settled() == "stood"


def test_a_trigger_seen_and_then_cleared_was_consumed() -> None:
    """The PLC took it. Evidence: it was observed set at least once."""
    assert _written(False, True, [True, False, False]).settled() == "consumed"


def test_a_value_never_observed_is_reported_as_not_seen() -> None:
    """Not "ignored". A PLC scan is milliseconds and the fastest reading here
    is a network round trip, so a trigger consumed within one scan looks
    exactly like a write nothing acted on. Claiming to tell them apart is the
    mistake the first version made -- it called every disappearance a healthy
    trigger, and said so reassuringly through four start attempts on a module
    in manual mode."""
    outcome = _written(False, True, [False, False, False])
    assert outcome.settled() == "not_seen"
    assert "or nothing acted on it" in outcome.explain()
    assert "ignored" not in outcome.explain()


def test_no_observations_is_unknown() -> None:
    assert _written(False, True, []).settled() == "unknown"


def test_a_plain_read_carries_no_write_history() -> None:
    value = opc.NodeValue(node_id="n", value=1, data_type="Int16", writable=False)
    assert value.observed == []
    assert value.settled() == "unknown"


# ---------------------------------------------------------------------------
# both names, so a screen and TIA Portal can be held against each other
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "node_id,expected",
    [
        ('ns=3;s="ApplicationModule_DB"."xReady"', "ApplicationModule_DB.xReady"),
        ('ns=3;s="dbVar"."Err"."xSF"', "dbVar.Err.xSF"),
        ('ns=3;s="xQA1_A1"', "xQA1_A1"),
        ("ns=1;s=Pressure_1", "Pressure_1"),
        ("ns=1;i=2258", "2258"),
        ("", ""),
        (None, ""),
    ],
)
def test_the_plc_tag_survives_beside_the_semantic_name(node_id, expected) -> None:
    """An operator reading `ConveyorForward` on a screen and `xQA1_A1` in TIA
    Portal has to be able to see they are the same signal."""
    assert opc.tag_name(node_id) == expected

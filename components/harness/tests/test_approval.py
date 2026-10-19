"""The approval gate: what it classifies, what it stops, and what it records.

No network. `lookup_moves_machine` is injected, so the whole decision path --
including the one that reaches the repository in production -- is exercised
without one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from a4s.kernel.approval import (
    DELETE,
    INVOKE,
    MOVING,
    READ,
    SHELL,
    WRITE,
    Gate,
    Policy,
    auto_approver,
    classify,
    terminal_approver,
)

CAP = "urn:plant:sm:capability"


def moving(*paths: str):
    """A lookup that says these Operation paths move the machine."""
    return lambda _submodel, path: path in paths


def command(line: str, **kw: Any):
    return classify("shell", {"command": line}, **kw)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "line,expected",
    [
        ("aas get urn:x SerialNumber", READ),
        ("aas sm tree urn:x", READ),
        ("aas find 'capacity' --in urn:x", READ),
        ("aas validate urn:x", READ),
        ("aas smt diff urn:t urn:x", READ),
        ("aas set urn:x A B", WRITE),
        ("aas el add urn:x '' --type Property --id-short A", WRITE),
        ("aas smt instantiate urn:t --as urn:x", WRITE),
        ("aas el rm urn:x A", DELETE),
        ("aas sm delete urn:x", DELETE),
        ("aas shell delete urn:x", DELETE),
        ("python train.py", SHELL),
        ("curl http://svc | jq .", SHELL),
    ],
)
def test_a_command_is_classified_by_what_it_does(line: str, expected: str) -> None:
    assert command(line).risk == expected


def test_an_unrecognised_aas_verb_is_a_read_not_a_shell_command() -> None:
    """Classifying `aas get` as a generic shell command makes `strict` gate
    reading a temperature, which is how a gate gets switched off."""
    assert command("aas op ls urn:x").risk == READ


def test_a_compound_command_takes_its_worst_action() -> None:
    """`aas get X && aas op invoke Y` is two actions and the gate must see the
    second one."""
    action = command(
        f"aas get urn:x A && aas op invoke {CAP} Stop",
        lookup_moves_machine=moving("Stop"),
    )
    assert action.risk == MOVING


@pytest.mark.parametrize("separator", ["&&", "||", ";", "|", "\n"])
def test_every_shell_separator_is_split_on(separator: str) -> None:
    action = command(
        f"aas get urn:x A {separator} aas sm delete urn:x",
    )
    assert action.risk == DELETE


def test_the_whole_command_line_is_kept_for_the_record() -> None:
    """The operator is shown what would run, not the fragment that tripped it."""
    line = "aas get urn:x A && aas sm delete urn:x"
    assert command(line).command == line


# ---------------------------------------------------------------------------
# moving the machine
# ---------------------------------------------------------------------------

def test_a_moving_operation_outranks_an_ordinary_one() -> None:
    """The AAS says which is which; the gate reads it rather than guessing from
    the name."""
    lookup = moving("Setpoint", "Stop")
    assert command(f"aas op invoke {CAP} Setpoint --arg name=X", lookup_moves_machine=lookup).risk == MOVING
    assert command(f"aas op invoke {CAP} Read --arg name=X", lookup_moves_machine=lookup).risk == INVOKE


@pytest.mark.parametrize(
    "line",
    [
        "aas op invoke $SM $OP",
        "aas op invoke urn:x $OP",
        "aas op invoke urn:x `cat op.txt`",
        "aas op invoke",
    ],
)
def test_an_unresolvable_invocation_is_treated_as_dangerous(line: str) -> None:
    """The shell has not expanded the variables yet, so the gate cannot know
    what would be invoked. A gate that opens when it cannot see is not a gate."""
    action = classify("shell", {"command": line}, lookup_moves_machine=lambda *_: False)
    assert action.risk == MOVING
    assert action.moves_machine


def test_a_lookup_that_fails_is_read_as_dangerous() -> None:
    def broken(_submodel: str, _path: str) -> bool:
        raise RuntimeError("repository unreachable")

    from a4s.kernel.approval import repository_lookup

    lookup = repository_lookup("http://127.0.0.1:1")     # nothing listens there
    assert lookup("urn:x", "Stop") is True


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def test_the_default_level_stops_only_what_moves_a_machine() -> None:
    policy = Policy()
    lookup = moving("Stop")
    assert policy.level == "machine"
    assert policy.decide(command(f"aas op invoke {CAP} Stop", lookup_moves_machine=lookup)) == "ask"
    assert policy.decide(command(f"aas op invoke {CAP} Read", lookup_moves_machine=lookup)) == "allow"
    assert policy.decide(command("aas set urn:x A B")) == "allow"


@pytest.mark.parametrize(
    "level,line,expected",
    [
        ("off", "aas op invoke urn:x Stop", "allow"),
        ("machine", "aas set urn:x A B", "allow"),
        ("operations", "aas op invoke urn:x Read", "ask"),
        ("operations", "aas set urn:x A B", "allow"),
        ("writes", "aas set urn:x A B", "ask"),
        ("writes", "python x.py", "allow"),
        ("strict", "python x.py", "ask"),
        ("strict", "aas get urn:x A", "allow"),
    ],
)
def test_each_level_gates_what_it_says(level: str, line: str, expected: str) -> None:
    assert Policy(level=level).decide(command(line, lookup_moves_machine=moving("Stop"))) == expected


def test_reading_is_never_gated_at_any_level() -> None:
    """An agent that cannot read cannot decide anything, and a gate that stops
    it will be turned off."""
    for level in ("off", "machine", "operations", "writes", "strict"):
        assert Policy(level=level).decide(command("aas get urn:x A")) == "allow"


def test_an_unknown_level_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv("A4S_APPROVAL", "paranoid")
    assert Policy.from_env().level == "machine"


def test_a_subject_can_be_pre_authorised() -> None:
    policy = Policy(level="writes", allow=["urn:scratch:*"])
    assert policy.decide(command("aas set urn:scratch:sm A B")) == "allow"
    assert policy.decide(command("aas set urn:plant:sm A B")) == "ask"


def test_the_deny_list_beats_the_allow_list() -> None:
    """A kill-list is not a suggestion."""
    policy = Policy(level="off", allow=["urn:plant:*"], deny=["urn:plant:press:*"])
    assert policy.decide(command("aas set urn:plant:press:sm A B")) == "deny"
    assert policy.decide(command("aas set urn:plant:other:sm A B")) == "allow"


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

class FakeLedger:
    def __init__(self) -> None:
        self.events: list = []

    def append(self, kind: str, payload: Dict[str, Any], subject: Any = None) -> None:
        self.events.append((kind, payload, subject))


def test_an_unattended_run_halts_rather_than_proceeding() -> None:
    ledger = FakeLedger()
    gate = Gate(Policy(level="machine"), ledger, approver=None,
                lookup_moves_machine=moving("Stop"))
    decision = gate.check("shell", {"command": f"aas op invoke {CAP} Stop"})
    assert not decision.allowed
    assert gate.pending is not None
    assert "move the machine" in gate.pending.render()


def test_the_request_offers_options_and_free_text() -> None:
    gate = Gate(Policy(level="machine"), FakeLedger(), approver=None,
                lookup_moves_machine=moving("Stop"))
    gate.check("shell", {"command": f"aas op invoke {CAP} Stop"})
    request = gate.pending
    assert request is not None
    assert len(request.options) == 2
    assert request.as_dict()["free_text"] is True
    assert "your own words" in request.render()


def test_an_operator_instruction_comes_back_to_the_agent() -> None:
    """"Yes, but 240" is the common answer, and a bare refusal loses it."""
    import io

    approver = terminal_approver(stream=io.StringIO("set it to 240 instead\n"), output=io.StringIO())
    gate = Gate(Policy(level="machine"), FakeLedger(), approver=approver,
                lookup_moves_machine=moving("Setpoint"))
    decision = gate.check("shell", {"command": f"aas op invoke {CAP} Setpoint --arg value=255"})
    assert not decision.allowed
    assert decision.note == "set it to 240 instead"
    assert "set it to 240 instead" in gate.refusal(decision)


@pytest.mark.parametrize("answer,allowed", [("1", True), ("y", True), ("2", False), ("", False)])
def test_the_terminal_approver_reads_the_obvious_answers(answer: str, allowed: bool) -> None:
    import io

    approver = terminal_approver(stream=io.StringIO(answer + "\n"), output=io.StringIO())
    gate = Gate(Policy(level="machine"), FakeLedger(), approver=approver,
                lookup_moves_machine=moving("Stop"))
    assert gate.check("shell", {"command": f"aas op invoke {CAP} Stop"}).allowed is allowed


def test_pre_authorising_is_recorded_as_pre_authorised() -> None:
    """An unattended batch may approve everything; the ledger must still say
    that no person was asked."""
    ledger = FakeLedger()
    gate = Gate(Policy(level="machine"), ledger, approver=auto_approver(),
                lookup_moves_machine=moving("Stop"))
    decision = gate.check("shell", {"command": f"aas op invoke {CAP} Stop"})
    assert decision.allowed
    assert decision.source == "preauthorised"
    assert ledger.events[-1][1]["source"] == "preauthorised"


def test_allowed_actions_are_recorded_too() -> None:
    """"Who approved changing this machine's temperature" needs the automatic
    approvals as much as the asked-for ones."""
    ledger = FakeLedger()
    gate = Gate(Policy(level="machine"), ledger, lookup_moves_machine=moving("Stop"))
    gate.check("shell", {"command": "aas get urn:x A"})
    kind, payload, _ = ledger.events[-1]
    assert kind == "approval"
    assert payload["allowed"] is True
    assert payload["level"] == "machine"


def test_the_refusal_tells_the_agent_what_to_do_next() -> None:
    gate = Gate(Policy(level="machine"), FakeLedger(), lookup_moves_machine=moving("Stop"))
    decision = gate.check("shell", {"command": f"aas op invoke {CAP} Stop"})
    text = gate.refusal(decision)
    assert "ask_user" in text
    assert "continue with the parts of the task that do not" in text


# ---------------------------------------------------------------------------
# through the registry
# ---------------------------------------------------------------------------

def test_a_gated_call_never_reaches_the_handler(tmp_path: Path) -> None:
    from a4s.environments.local import LocalEnvironment
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    try:
        marker = tmp_path / "ran.txt"
        gate = Gate(Policy(level="strict"), ledger, approver=None)
        tools = ToolRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path, gate=gate)
        result = tools.dispatch("shell", {"command": f"echo x > {marker.name}"})
        assert "refused" in result
        assert not marker.exists(), "the command ran despite being refused"
        assert tools.pending_question is not None
    finally:
        ledger.close()


def test_an_ungated_registry_still_works(tmp_path: Path) -> None:
    """The gate is optional; without one nothing changes."""
    from a4s.environments.local import LocalEnvironment
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    try:
        tools = ToolRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path)
        assert "hello" in tools.dispatch("shell", {"command": "echo hello"})
    finally:
        ledger.close()


def test_a_refused_attempt_is_still_in_the_ledger(tmp_path: Path) -> None:
    """What the agent tried matters as much as what it did."""
    from a4s.environments.local import LocalEnvironment
    from a4s.ledger import Ledger
    from a4s.tools import ToolRegistry

    ledger = Ledger(tmp_path / "l.db", "t", actor="t")
    try:
        gate = Gate(Policy(level="strict"), ledger, approver=None)
        tools = ToolRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path, gate=gate)
        tools.dispatch("shell", {"command": "rm -rf /"})
        kinds = [e.kind for e in ledger.read(session_id="t", limit=50)]
        assert "tool.call" in kinds and "approval" in kinds
    finally:
        ledger.close()


# ---------------------------------------------------------------------------
# going around the front door
# ---------------------------------------------------------------------------

TARGETS = {
    "http://localhost:5080/read": False,
    "http://localhost:5080/status": False,
    "http://localhost:5080/setpoint": True,
    "http://localhost:5080/setpoint/clear": True,
    "http://printer-service:5060/print/classify-defect": False,
}


def test_curling_a_delegated_service_is_not_an_ordinary_shell_command() -> None:
    """A live run reached the machine with `curl http://localhost:5080/read`
    and the gate waved it through. Had it been `/setpoint` the machine would
    have moved unapproved."""
    action = command("curl -sS http://localhost:5080/read", delegation_targets=TARGETS)
    assert action.risk == INVOKE
    assert "going around" in action.detail["reason"]


def test_curling_a_moving_endpoint_is_treated_as_moving() -> None:
    action = command(
        "curl -X POST http://localhost:5080/setpoint -d '{}'", delegation_targets=TARGETS
    )
    assert action.risk == MOVING
    assert Policy(level="machine").decide(action) == "ask"


def test_the_longest_matching_url_wins() -> None:
    """`/setpoint/clear` must not be recognised as `/setpoint` and mislabelled."""
    action = command("curl http://localhost:5080/setpoint/clear", delegation_targets=TARGETS)
    assert action.subject == "http://localhost:5080/setpoint/clear"


def test_a_known_host_with_an_unknown_path_takes_the_worst_case() -> None:
    """The command is talking to a machine's controller and the gate cannot
    tell which endpoint. Seeing less is not a reason to worry less."""
    action = command("wget http://localhost:5080/", delegation_targets=TARGETS)
    assert action.risk == MOVING


def test_a_host_with_no_moving_endpoints_stays_an_ordinary_invocation() -> None:
    action = command("curl http://printer-service:5060/anything", delegation_targets=TARGETS)
    assert action.risk == INVOKE
    assert not action.moves_machine


def test_reaching_the_service_from_python_is_caught_too() -> None:
    action = command(
        'python -c "import httpx; httpx.post(\'http://localhost:5080/setpoint\')"',
        delegation_targets=TARGETS,
    )
    assert action.risk == MOVING


def test_an_unrelated_command_is_untouched() -> None:
    assert command("echo hello", delegation_targets=TARGETS).risk == SHELL
    assert command("curl http://example.com/", delegation_targets=TARGETS).risk == SHELL


def test_without_an_index_nothing_changes() -> None:
    """The index is optional; a gate built without one behaves as before."""
    assert command("curl http://localhost:5080/setpoint").risk == SHELL

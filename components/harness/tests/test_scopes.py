"""Identity scoping: which key may write which shells.

The gate decided *whether* an action happens. This decides **under whose
authority**, which the ledger could answer after the fact and nothing could
answer before it. Every actor already signs its own rows, so "who changed this
property" was answerable; nothing stopped an asset agent for one machine
writing to another machine's shell.

The behaviour worth testing here is almost all at the edges: what happens with
no file, an empty file, a broken file, an unlisted actor, and an action that
names no subject at all. Each of those has a defensible wrong answer, and the
wrong answers are the ones that make a security feature either useless or
unshippable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from a4s.kernel.approval import (
    MOVING,
    READ,
    WRITE,
    Action,
    Gate,
    Policy,
    Scopes,
)

IMM = "https://fraunhoferIPA/ids/sm/imm_sim/OperationalData"
BATTERY = "https://admin-shell.io/ZVEI/TechnicalData/Submodel/1/2_DigitalBatteryPassport_527"


def scope_file(tmp_path: Path, entries: Dict[str, Any]) -> Path:
    path = tmp_path / "scopes.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def write_to(subject: str) -> Action:
    return Action(risk=WRITE, tool="shell", command=f"aas set {subject} X 1", subject=subject)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_no_file_means_scoping_is_off(tmp_path: Path) -> None:
    """An installation that has never heard of scopes keeps working.

    The alternative -- default-deny with no file -- locks every existing
    deployment out of its own repository on upgrade, which is how a safety
    feature gets reverted rather than adopted.
    """
    scopes = Scopes.load(tmp_path / "absent.json")

    assert not scopes.configured
    allowed, why = scopes.permits(write_to(IMM), "anyone")
    assert allowed and "not configured" in why


def test_an_empty_file_denies_everything(tmp_path: Path) -> None:
    """Empty is a deliberate statement and a missing file is not.

    Collapsing the two would make it impossible to express "nobody writes
    anything", which is the state a repository should be in while it is being
    audited.
    """
    scopes = Scopes.load(scope_file(tmp_path, {}))

    assert scopes.configured
    allowed, _ = scopes.permits(write_to(IMM), "orchestrator")
    assert not allowed


def test_a_broken_file_fails_closed(tmp_path: Path) -> None:
    """A file that exists says someone intended a restriction.

    Guessing what it was is worse than refusing to start, and this is the one
    place where refusing to start is right: the operator can see the error and
    fix the file, whereas an agent running with silently-dropped scopes cannot.
    """
    path = tmp_path / "scopes.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read scopes"):
        Scopes.load(path)

    path.write_text(json.dumps(["a", "list"]), encoding="utf-8")
    with pytest.raises(ValueError, match="object of actor"):
        Scopes.load(path)


def test_a_bare_string_is_accepted_as_one_pattern(tmp_path: Path) -> None:
    """The obvious typo, and it means exactly what it looks like."""
    scopes = Scopes.load(scope_file(tmp_path, {"imm": f"{IMM}*"}))

    assert scopes.permits(write_to(IMM), "imm")[0]


# ---------------------------------------------------------------------------
# deciding
# ---------------------------------------------------------------------------

def test_an_actor_may_write_inside_its_patterns_and_not_outside(tmp_path: Path) -> None:
    scopes = Scopes.load(scope_file(tmp_path, {
        "imm-agent": ["https://fraunhoferIPA/ids/sm/imm_sim/*"],
        "orchestrator": ["*"],
    }))

    assert scopes.permits(write_to(IMM), "imm-agent")[0]
    assert not scopes.permits(write_to(BATTERY), "imm-agent")[0]
    assert scopes.permits(write_to(BATTERY), "orchestrator")[0]


def test_an_unlisted_actor_is_refused_and_the_reason_names_the_file(
    tmp_path: Path,
) -> None:
    """The message has to be actionable by an operator, not just by an agent.

    "Not in scope" sends someone looking for a permission system; naming the
    file sends them to the line they need to add.
    """
    path = scope_file(tmp_path, {"orchestrator": ["*"]})
    scopes = Scopes.load(path)

    allowed, why = scopes.permits(write_to(IMM), "stranger")
    assert not allowed
    assert "stranger" in why and str(path) in why


def test_reads_are_never_scoped(tmp_path: Path) -> None:
    """Scoping a read is not a boundary, it is an obstacle.

    The repository serves `/shells` to anyone who can reach it, so hiding a
    shell from an agent that could list it anyway buys nothing and breaks
    discovery. What is worth restricting is what an identity can *change*.
    """
    scopes = Scopes.load(scope_file(tmp_path, {"orchestrator": ["urn:nothing:*"]}))

    read = Action(risk=READ, tool="shell", command=f"aas get {IMM} X", subject=IMM)
    assert scopes.permits(read, "orchestrator")[0]
    assert not scopes.permits(write_to(IMM), "orchestrator")[0]


def test_an_action_with_no_subject_cannot_be_authorised(tmp_path: Path) -> None:
    """A shell command touching no AAS path matches no shell pattern.

    Passing it would make the scope trivially escapable -- write through
    `curl`, or through a command the parser cannot classify, and the boundary
    is gone. This is the same reasoning `_invoke_target` uses for an
    unexpanded `$SM`: unclassifiable is refused, not waved through.
    """
    scopes = Scopes.load(scope_file(tmp_path, {"orchestrator": ["*"]}))

    opaque = Action(risk=WRITE, tool="shell", command="curl -X POST http://host/write")
    allowed, why = scopes.permits(opaque, "orchestrator")
    assert not allowed
    assert "no AAS subject" in why


def test_the_signing_key_is_preferred_over_the_label(tmp_path: Path) -> None:
    """`--actor` is a command-line label; the key is what actually signed.

    Both are accepted because an operator writes the label, but where a file
    names the key that entry wins -- otherwise the restriction is defeated by
    passing a different `--actor`.
    """
    key = "a" * 64
    scopes = Scopes.load(scope_file(tmp_path, {
        key: ["urn:only:key:*"],
        "imm-agent": ["*"],
    }))

    # The key is checked first, so its narrower grant is the one that applies.
    assert not scopes.permits(write_to(IMM), key, "imm-agent")[0]
    assert scopes.permits(write_to("urn:only:key:sm"), key, "imm-agent")[0]


# ---------------------------------------------------------------------------
# through the gate
# ---------------------------------------------------------------------------

class FakeLedger:
    def __init__(self) -> None:
        self.actor = "orchestrator"
        self.rows: List[Dict[str, Any]] = []

    def append(self, kind: str, payload: Dict[str, Any], *, subject: Any = None,
               actor: Any = None) -> None:
        self.rows.append({"kind": kind, "payload": payload, "subject": subject})


def test_the_gate_refuses_out_of_scope_writes_and_records_them(tmp_path: Path) -> None:
    ledger = FakeLedger()
    gate = Gate(
        policy=Policy(level="off"),          # nothing would have been asked about
        ledger=ledger,
        scopes=Scopes.load(scope_file(tmp_path, {"orchestrator": ["urn:mine:*"]})),
        actor="orchestrator",
    )

    decision = gate.check("shell", {"command": f"aas set {IMM} X 1"})

    assert not decision.allowed
    assert decision.source == "scope"
    recorded = [r for r in ledger.rows if r["kind"] == "approval"]
    assert recorded and recorded[-1]["payload"]["allowed"] is False


def test_scope_is_checked_before_the_policy_and_is_not_appealable(
    tmp_path: Path,
) -> None:
    """An operator can approve what the policy asks about. Not this.

    Nobody at the terminal can grant an identity authority over a shell it was
    not given, so the approver must never be consulted -- a prompt that cannot
    change the outcome is a prompt that teaches people to click through
    prompts.
    """
    asked: List[Any] = []

    def approver(request: Any) -> Any:
        asked.append(request)
        raise AssertionError("the operator was asked about a scope violation")

    gate = Gate(
        policy=Policy(level="strict"),       # would otherwise ask about everything
        ledger=FakeLedger(),
        approver=approver,
        scopes=Scopes.load(scope_file(tmp_path, {"orchestrator": ["urn:mine:*"]})),
        actor="orchestrator",
    )

    decision = gate.check("shell", {"command": f"aas set {IMM} X 1"})

    assert not decision.allowed
    assert asked == []


def test_the_refusal_tells_the_agent_not_to_ask(tmp_path: Path) -> None:
    """Its correct response to "needs a person" is to ask one. Not here.

    Offering `ask_user` would send it to collect an approval nobody present can
    give, which costs a step and ends in the same refusal.
    """
    gate = Gate(
        policy=Policy(level="off"),
        ledger=FakeLedger(),
        scopes=Scopes.load(scope_file(tmp_path, {"orchestrator": ["urn:mine:*"]})),
        actor="orchestrator",
    )
    decision = gate.check("shell", {"command": f"aas set {IMM} X 1"})

    message = gate.refusal(decision)
    assert "ask_user" not in message
    assert "authority boundary" in message


def test_an_unconfigured_gate_behaves_exactly_as_before(tmp_path: Path) -> None:
    """The upgrade path. Nothing changes until a scope file exists."""
    gate = Gate(policy=Policy(level="off"), ledger=FakeLedger(), actor="orchestrator")

    assert gate.check("shell", {"command": f"aas set {IMM} X 1"}).allowed
    assert gate.check("shell", {"command": "curl http://anywhere/"}).allowed


def test_scoping_does_not_replace_the_machine_gate(tmp_path: Path) -> None:
    """Two boundaries, and passing one does not pass the other.

    An identity permitted to write a machine's shells still has to have a
    person approve moving it. Collapsing the two would let a scope grant become
    an operating authorisation.
    """
    asked: List[Any] = []

    def approver(request: Any) -> Any:
        from a4s.kernel.approval import Decision

        asked.append(request)
        return Decision(False, "refused by the operator", "operator")

    gate = Gate(
        policy=Policy(level="machine"),
        ledger=FakeLedger(),
        approver=approver,
        scopes=Scopes.load(scope_file(tmp_path, {"orchestrator": ["*"]})),
        actor="orchestrator",
        lookup_moves_machine=lambda sm, path: True,
    )

    decision = gate.check(
        "shell", {"command": f"aas op invoke {IMM} Capability.Start"}
    )

    assert asked, "the machine gate did not fire for an in-scope identity"
    assert not decision.allowed
    assert decision.source == "operator"


def test_moving_is_still_classified_under_a_permissive_scope(tmp_path: Path) -> None:
    from a4s.kernel.approval import classify

    action = classify(
        "shell", {"command": f"aas op invoke {IMM} Capability.Start"},
        lookup_moves_machine=lambda sm, path: True,
    )
    assert action.risk == MOVING

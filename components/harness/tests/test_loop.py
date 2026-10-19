"""The loop's failure handling, exercised without a model or a BaSyx server.

A scripted provider makes the paths that matter testable: they are the ones
that only occur when something goes wrong, which is exactly when a real model
is least available to reproduce them on demand.
"""

from __future__ import annotations

import uuid
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from a4s.environments.local import LocalEnvironment
from a4s.kernel.loop import AgentLoop, LoopConfig
from a4s.kernel.providers import Completion, ModelConfig, ToolCall, TransientModelError
from a4s.ledger import Kind, Ledger
from a4s.tools import ToolRegistry


class ScriptedProvider:
    """Replays a list of completions; raises whatever is put in the list."""

    def __init__(self, script: List[Any]) -> None:
        self.script = list(script)
        self.calls: List[List[Dict[str, Any]]] = []

    def complete(self, messages, tools, config):  # noqa: ANN001 - Provider protocol
        self.calls.append(list(messages))
        if not self.script:
            return Completion(text="", tool_calls=[ToolCall("fin", "finish", {"summary": "done"})])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name: str, **arguments: Any) -> Completion:
    return Completion(text="", tool_calls=[ToolCall(f"c{uuid.uuid4().hex[:6]}", name, arguments)])


@pytest.fixture()
def harness(tmp_path: Path):
    def build(script: List[Any], observer: Any = None, **loop_kwargs: Any):
        ledger = Ledger(tmp_path / "ledger.db", "test-session", actor="tester")
        tools = ToolRegistry(LocalEnvironment(cwd=str(tmp_path)), ledger, tmp_path)
        provider = ScriptedProvider(script)
        loop = AgentLoop(
            provider=provider,
            model=ModelConfig(model="scripted", context_window=32_000),
            tools=tools,
            ledger=ledger,
            system_prompt="test",
            config=LoopConfig(**loop_kwargs),
            observer=observer,
        )
        return loop, provider, ledger

    return build


def test_finishes_and_records_the_trajectory(harness, tmp_path):
    loop, _, ledger = harness(
        [
            call("write_file", path="notes.txt", content="hello"),
            call("finish", summary="wrote notes"),
        ]
    )
    result = loop.run("write a note")

    assert result.status == "finished"
    assert result.summary == "wrote notes"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"

    # The ledger holds a paired call/result for every tool invocation, which
    # is what makes it exportable as a trajectory without reconstruction.
    records = list(ledger.export_trajectory("test-session"))
    tool_records = [r for r in records if "call" in r]
    assert len(tool_records) == 2
    assert tool_records[0]["call"]["tool"] == "write_file"
    assert ledger.verify() == (True, None)


def test_operator_stop_skips_remaining_tool_calls_and_next_turn_can_run(harness, tmp_path):
    stop = threading.Event()
    first = Completion(text="", tool_calls=[
        ToolCall("first", "write_file", {"path": "first.txt", "content": "done"}),
        ToolCall("second", "write_file", {"path": "second.txt", "content": "must not run"}),
    ])
    loop, _, ledger = harness([first, call("finish", summary="continued")],
                              observer=lambda kind, payload: stop.set() if kind == "tool_result" and payload["id"] == "first" else None)
    loop.stop_event = stop
    result = loop.run("write both")
    assert result.status == "stopped"
    assert (tmp_path / "first.txt").read_text() == "done"
    assert not (tmp_path / "second.txt").exists()
    assert any("skipped_due_to_operator_stop" in str(message) for message in result.messages)
    replayed = ledger.replay("test-session", "test")
    assert any(message.get("tool_call_id") == "second" and
               "skipped_due_to_operator_stop" in message.get("content", "") for message in replayed)
    stop.clear()
    assert loop.run("new instruction").status == "finished"
    assert ledger.verify() == (True, None)


def test_unknown_tool_is_answered_not_crashed(harness):
    """An invented tool name must produce a correctable error, not an exception.

    Small models invent tool names, and the recovery hint has to name the
    real route to the AAS or the model invents another one next turn.
    """
    loop, _, _ = harness([call("aas_read_property", path="X"), call("finish", summary="ok")])
    result = loop.run("read something")

    assert result.status == "finished"
    tool_replies = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert "no tool named" in tool_replies[0]
    assert "aas" in tool_replies[0]


def test_transient_model_errors_are_retried(harness, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    loop, provider, _ = harness(
        [TransientModelError("429"), TransientModelError("429"), call("finish", summary="recovered")]
    )
    result = loop.run("do a thing")

    assert result.status == "finished"
    assert result.summary == "recovered"
    assert len(provider.calls) == 3


def test_gives_up_after_the_retry_budget(harness, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    loop, _, _ = harness([TransientModelError("503")] * 6, max_transient_retries=3)
    result = loop.run("do a thing")

    assert result.status == "failed"
    assert "unavailable" in result.summary


def test_identical_repeated_calls_are_cut_off(harness):
    """Three identical calls in a row is a stuck model, and it is told so."""
    repeated = [call("read_file", path="missing.txt") for _ in range(5)]
    loop, _, _ = harness([*repeated, call("finish", summary="gave up")], repetition_limit=3)
    result = loop.run("read a file that is not there")

    tool_replies = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert any("will not change" in reply for reply in tool_replies)


def test_bare_text_is_nudged_once_then_accepted(harness):
    loop, _, _ = harness([Completion(text="I think we should..."), Completion(text="Still thinking.")])
    result = loop.run("do something")

    nudges = [
        m for m in result.messages
        if m.get("role") == "user" and str(m.get("content", "")).startswith("[harness]")
    ]
    assert len(nudges) == 1
    assert result.status == "finished"


def test_step_budget_is_enforced(harness):
    loop, _, _ = harness([call("read_file", path="a") for _ in range(20)], max_steps=4)
    result = loop.run("loop forever")

    assert result.status == "budget_exhausted"
    assert result.steps == 4


def test_aas_mutations_get_their_own_provenance_record(harness, tmp_path):
    """A shell command that mutates an AAS is indexed by AAS path, not by tool name."""
    loop, _, ledger = harness(
        [
            call("shell", command="echo aas set urn:x:sm:1 Nameplate.Manufacturer Fraunhofer"),
            call("finish", summary="done"),
        ]
    )
    loop.run("set a property")

    writes = ledger.read(kind=Kind.AAS_WRITE)
    assert len(writes) == 1
    assert writes[0].subject == "urn:x:sm:1/Nameplate.Manufacturer"


# ---------------------------------------------------------------------------
# skill selection
# ---------------------------------------------------------------------------

def test_a_matching_skill_is_injected_not_merely_indexed(tmp_path) -> None:
    """Across three live DPP runs the agent read the indexed skill zero times."""
    from a4s.kernel.prompt import build_system_prompt

    root = tmp_path / "skills"
    (root / "dpp-generation").mkdir(parents=True)
    (root / "dpp-generation" / "SKILL.md").write_text(
        "---\nname: dpp-generation\ndescription: build a passport\n"
        "triggers: DPP, product passport\n---\n\nSTEP ONE: instantiate the template.\n",
        encoding="utf-8",
    )
    (root / "machine-onboarding").mkdir(parents=True)
    (root / "machine-onboarding" / "SKILL.md").write_text(
        "---\nname: machine-onboarding\ndescription: onboard a machine\n"
        "triggers: onboard, commission\n---\n\nUNRELATED BODY\n",
        encoding="utf-8",
    )

    prompt = build_system_prompt(root, instruction="Build a Digital Product Passport for SN-4711")
    assert "STEP ONE: instantiate the template." in prompt
    assert "UNRELATED BODY" not in prompt
    # The one that did not match stays an index line, not a body.
    assert "machine-onboarding: onboard a machine" in prompt


def test_no_trigger_match_injects_nothing(tmp_path) -> None:
    from a4s.kernel.prompt import build_system_prompt

    root = tmp_path / "skills"
    (root / "dpp-generation").mkdir(parents=True)
    (root / "dpp-generation" / "SKILL.md").write_text(
        "---\nname: dpp-generation\ndescription: d\ntriggers: DPP\n---\n\nBODY\n",
        encoding="utf-8",
    )
    prompt = build_system_prompt(root, instruction="Read the spindle temperature")
    assert "BODY" not in prompt
    assert "dpp-generation: d" in prompt


def test_the_best_match_wins_when_several_trigger(tmp_path) -> None:
    from a4s.kernel.prompt import load_skills, select_skill

    root = tmp_path / "skills"
    for name, triggers, body in [
        ("a-skill", "passport", "WEAK"),
        ("b-skill", "DPP, product passport, datasheet", "STRONG"),
    ]:
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\ntriggers: {triggers}\n---\n\n{body}\n",
            encoding="utf-8",
        )
    chosen = select_skill(load_skills(root), "build a DPP product passport from a datasheet")
    assert chosen is not None and chosen.name == "b-skill"


def test_the_synopsis_can_be_withheld_for_the_ablation(tmp_path) -> None:
    from a4s.kernel.prompt import build_system_prompt

    root = tmp_path / "skills"
    root.mkdir()
    assert "aas smt instantiate" in build_system_prompt(root)
    assert "aas smt instantiate" not in build_system_prompt(root, include_synopsis=False)


# ---------------------------------------------------------------------------
# the shell commands actually run in
# ---------------------------------------------------------------------------

# Which of several candidates wins, and why "it exists on PATH" is the wrong
# question on Windows, is in `tests/test_shell_choice.py`. These pin the older
# and simpler rule: a POSIX shell is preferred to cmd.exe, and `sh` will do.

def test_a_posix_shell_is_chosen_when_one_exists(monkeypatch) -> None:
    """`shell=True` means cmd.exe on Windows, which does not merely refuse a
    POSIX script -- it exits 0 with empty stdout and empty stderr. A live run
    lost two steps concluding its files were missing."""
    from a4s.environments import local as env

    monkeypatch.delenv("A4S_SHELL", raising=False)
    monkeypatch.setattr(env, "_posix_candidates", lambda: ["/usr/bin/bash"])
    monkeypatch.setattr(env, "_shell_works", lambda argv: True)
    assert env.find_shell() == ["/usr/bin/bash", "-c"]


def test_the_shell_can_be_overridden(monkeypatch) -> None:
    from a4s.environments import local as env

    monkeypatch.setenv("A4S_SHELL", "/bin/zsh")
    assert env.find_shell() == ["/bin/zsh", "-c"]


def test_sh_is_the_fallback_before_cmd(monkeypatch) -> None:
    from a4s.environments import local as env

    monkeypatch.delenv("A4S_SHELL", raising=False)
    monkeypatch.setattr(env, "_posix_candidates", lambda: ["/bin/sh"])
    monkeypatch.setattr(env, "_shell_works", lambda argv: True)
    assert env.find_shell() == ["/bin/sh", "-c"]


def test_a_pipeline_runs_and_returns_its_output() -> None:
    """The composition claim rests on this working at all."""
    from a4s.environments.local import LocalEnvironment

    environment = LocalEnvironment()
    if not environment.shell or Path(environment.shell[0]).stem.lower() not in ("bash", "sh", "zsh", "dash", "ash", "ksh"):
        pytest.skip("POSIX pipeline requires a POSIX shell; Windows fallback has a separate prompt rule")
    # Windows PATH may resolve sort to System32's Unicode sort.exe inside Bash.
    result = environment.run("PATH=/usr/bin:/bin:$PATH; printf 'b\na\n' | sort | tr -d '\n'")
    assert result.exit_code == 0
    assert result.stdout.strip() == "ab"


def test_the_harness_puts_its_own_scripts_on_the_child_path() -> None:
    """`aas` has to be reachable without the operator having exported anything;
    the first live run of this harness failed exactly there."""
    import os

    from a4s.environments.local import LocalEnvironment, script_dir

    scripts = script_dir()
    if scripts is None:
        pytest.skip("not running from an installed environment")
    assert scripts in LocalEnvironment().env["PATH"].split(os.pathsep)


def test_a_silent_stop_is_not_reported_as_finished(harness) -> None:
    """A reasoning model can spend a whole turn on reasoning tokens and return
    neither a tool call nor a word. Calling that success puts an empty answer
    into a pass-rate."""
    silent = Completion(text="", tool_calls=[], finish_reason="stop")
    loop, _, ledger = harness([silent, silent])
    result = loop.run("do something")
    ledger.close()
    assert result.status == "failed"
    assert "without saying anything" in result.summary


def test_ending_on_a_real_answer_is_still_finished(harness) -> None:
    spoke = Completion(text="Here is the answer.", tool_calls=[], finish_reason="stop")
    loop, _, ledger = harness([spoke, spoke])
    result = loop.run("do something")
    ledger.close()
    assert result.status == "finished"
    assert result.summary == "Here is the answer."


# ---------------------------------------------------------------------------
# stopping for a person, and continuing afterwards
# ---------------------------------------------------------------------------

def test_a_session_replays_from_the_ledger_alone(tmp_path) -> None:
    """The process is gone when an operator answers. The ledger is the only
    copy of the conversation, and a second copy would be a second copy that can
    disagree with the audit record."""
    from a4s.ledger import Kind, Ledger

    ledger = Ledger(tmp_path / "l.db", "s", actor="t")
    try:
        ledger.append(Kind.TURN_USER, {"text": "set the temperature"})
        ledger.append(Kind.TURN_ASSISTANT, {
            "text": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "shell", "arguments": '{"command":"aas get x y"}'}}],
        })
        ledger.append(Kind.TOOL_RESULT, {"tool": "shell", "output": "250", "call_id": "c1"})
        ledger.append(Kind.TURN_ASSISTANT, {"text": "It reads 250.", "tool_calls": []})

        messages = ledger.replay("s", "SYSTEM")
    finally:
        ledger.close()

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[0]["content"] == "SYSTEM"
    assert messages[2]["tool_calls"][0]["id"] == "c1"
    assert messages[3]["tool_call_id"] == "c1"
    assert messages[4]["content"] == "It reads 250."


def test_a_session_without_call_ids_refuses_to_replay(tmp_path) -> None:
    """Attaching a result to the wrong call corrupts the history in a way the
    model cannot see. Better to say the session is not resumable."""
    from a4s.ledger import Kind, Ledger

    ledger = Ledger(tmp_path / "l.db", "s", actor="t")
    try:
        ledger.append(Kind.TURN_USER, {"text": "go"})
        ledger.append(Kind.TOOL_RESULT, {"tool": "shell", "output": "x"})
        with pytest.raises(ValueError, match="cannot be replayed"):
            ledger.replay("s", "SYSTEM")
    finally:
        ledger.close()


def test_the_refusal_resumed_from_is_the_refused_one(tmp_path) -> None:
    """By the time a run halts the agent has usually called `ask_user`, so the
    *last* approval is that harmless call, not the one being asked about."""
    from a4s.ledger import Kind, Ledger

    ledger = Ledger(tmp_path / "l.db", "s", actor="t")
    try:
        ledger.append(Kind.APPROVAL, {"allowed": True, "command": "aas get x y"})
        ledger.append(Kind.APPROVAL, {"allowed": False, "command": "aas op invoke x Stop"})
        ledger.append(Kind.APPROVAL, {"allowed": True, "command": "ask_user(...)"})
        refusal = ledger.last_refusal("s")
        assert refusal is not None
        assert refusal.payload["command"] == "aas op invoke x Stop"
    finally:
        ledger.close()


def test_no_refusal_means_nothing_to_resume(tmp_path) -> None:
    from a4s.ledger import Kind, Ledger

    ledger = Ledger(tmp_path / "l.db", "s", actor="t")
    try:
        ledger.append(Kind.APPROVAL, {"allowed": True, "command": "aas get x y"})
        assert ledger.last_refusal("s") is None
    finally:
        ledger.close()


def test_an_approval_on_resume_is_consumed_once(tmp_path) -> None:
    """The operator approved an action, not a habit."""
    from a4s.kernel.approval import Gate, Policy

    class Ledgerish:
        def append(self, *_a, **_k) -> None: ...

    gate = Gate(Policy(level="writes"), Ledgerish())
    gate.preapprove("aas set urn:x A B")

    first = gate.check("shell", {"command": "aas set urn:x A B"})
    second = gate.check("shell", {"command": "aas set urn:x A B"})
    assert first.allowed and first.source == "operator"
    assert not second.allowed, "the approval outlived the action it was given for"


def test_a_pre_approval_tolerates_reformatted_whitespace(tmp_path) -> None:
    from a4s.kernel.approval import Gate, Policy

    class Ledgerish:
        def append(self, *_a, **_k) -> None: ...

    gate = Gate(Policy(level="writes"), Ledgerish())
    gate.preapprove("aas set  urn:x   A B")
    assert gate.check("shell", {"command": "aas set urn:x A B"}).allowed


def test_resume_continues_the_restored_conversation(harness) -> None:
    """The operator's answer arrives as a new user turn, not by rewriting the
    refusal. History is what happened."""
    loop, provider, ledger = harness([call("finish", summary="done after approval")])
    restored = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "set the temperature"},
        {"role": "assistant", "content": "I was refused."},
    ]
    result = loop.resume(restored, "[operator] APPROVED. Run it now.")
    ledger.close()

    assert result.status == "finished"
    sent = provider.calls[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[-1]["content"].startswith("[operator] APPROVED")


def test_the_shipped_skills_match_the_tasks_they_are_for_and_no_others() -> None:
    """The trigger lists on the real skills, against the real task instructions.

    Written after a trigger list went out reading `every, all, each` and matched
    the `retrieve` and `validate` tasks, neither of which is bulk work. Ordinary
    English words make ordinary sentences match, and the failure is silent: the
    task still succeeds, having paid ~1,000 tokens for a procedure about
    something else.

    `select_skill` is a substring test on declared triggers rather than a model
    call, which the module docstring defends as "wrong in a way that is easy to
    see in the ledger". Easy to see is not the same as seen, so it is pinned
    here instead.
    """
    from pathlib import Path

    from a4s.experiment.tasks import TASKS
    from a4s.kernel.prompt import load_skills, select_skill

    skills = load_skills(Path(__file__).resolve().parents[1] / "a4s" / "skills")
    assert {s.name for s in skills} >= {"bulk-editing", "dpp-generation"}

    expected = {"batch": "bulk-editing", "prune": "bulk-editing", "chain": "dpp-generation"}
    for task in TASKS:
        chosen = select_skill(skills, task.instruction)
        got = chosen.name if chosen else None
        assert got == expected.get(task.name), (
            f"{task.name}: matched {got!r}, expected {expected.get(task.name)!r}"
        )


def test_every_shipped_skill_declares_triggers_and_a_description() -> None:
    """Both fields do work, and a skill missing either is inert or unfindable.

    Without triggers nothing selects it; without a description it appears in the
    index as a name with no reason to read it, which is the same as absent for
    ~1,000 tokens of body nobody fetches.
    """
    from pathlib import Path

    from a4s.kernel.prompt import load_skills

    for skill in load_skills(Path(__file__).resolve().parents[1] / "a4s" / "skills"):
        assert skill.triggers, f"{skill.name} declares no triggers and can never be selected"
        assert skill.description, f"{skill.name} has no description for the index"
        assert len(skill.body()) > 400, f"{skill.name} is too short to be a procedure"


def test_a_trigger_that_is_an_ordinary_english_word_is_the_failure_mode() -> None:
    """The regression, stated directly rather than only implied.

    A one-word trigger like `all` fires on any sentence containing it. Scoring
    counts matching triggers, so a skill declaring several such words outranks a
    specific one -- which is how the wrong procedure gets injected.
    """
    from a4s.kernel.prompt import Skill, select_skill

    generic = Skill(name="generic", description="", path=None, triggers=["all", "each"])
    specific = Skill(name="specific", description="", path=None, triggers=["delete every"])

    sentence = "Report all violations and the element each applies to"
    assert select_skill([generic, specific], sentence).name == "generic"
    assert select_skill([specific], sentence) is None


# ---------------------------------------------------------------------------
# the spectator
# ---------------------------------------------------------------------------

def test_an_observer_sees_the_run_as_it_happens(harness) -> None:
    """The chat socket cannot wait for `RunResult` in silence.

    A run is minutes of model calls and subprocesses, and a browser showing
    nothing for that long is indistinguishable from a broken one. The events
    are emitted where the ledger entries are already written, so what the
    watcher sees and what the log records cannot diverge.
    """
    seen: List[Any] = []
    loop, _, _ = harness(
        [
            Completion(text="I will write the file.",
                       tool_calls=[ToolCall("c1", "write_file",
                                            {"path": "n.txt", "content": "x"})]),
            call("finish", summary="done"),
        ],
        observer=lambda kind, payload: seen.append((kind, payload)),
    )

    loop.run("write a note")

    kinds = [kind for kind, _ in seen]
    assert kinds.count("thinking") == 2
    assert ("assistant", {"text": "I will write the file.", "step": 1}) in seen
    calls = [p for k, p in seen if k == "tool_call"]
    results = [p for k, p in seen if k == "tool_result"]
    assert [c["name"] for c in calls] == ["write_file", "finish"]
    assert [r["id"] for r in results] == [c["id"] for c in calls]


def test_a_failing_observer_does_not_fail_the_run(harness) -> None:
    """A watcher whose socket has closed must not end a run that is midway
    through writing to a repository. The ledger is the record; the observer is
    a window onto it."""
    def explode(kind: Any, payload: Any) -> None:
        raise RuntimeError("the browser went away")

    loop, _, _ = harness([call("finish", summary="done anyway")], observer=explode)

    result = loop.run("do the thing")

    assert result.status == "finished"
    assert result.summary == "done anyway"


def test_the_agents_shell_does_not_rewrite_its_arguments(tmp_path: Path) -> None:
    """Git Bash turns a bare `/` into the Git installation directory before the
    program sees it. `aas el ls <sm> /` reached BaSyx as a request for an
    element at `C:/Users/.../Git/` and came back a 400 -- one wasted step in a
    live run, and the agent recovered by trying a different verb, so the
    harness would never have heard about it."""
    import sys

    from a4s.environments.local import LocalEnvironment

    environment = LocalEnvironment(cwd=str(tmp_path))
    assert environment.env.get("MSYS_NO_PATHCONV") == "1"

    if sys.platform != "win32":
        return
    result = environment.run(
        f'"{sys.executable}" -c "import sys; print(sys.argv[1])" /', timeout=30
    )
    assert result.stdout.strip() == "/", result.stdout

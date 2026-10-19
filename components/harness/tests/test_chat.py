"""`/ws/chat`, with a scripted agent instead of a model.

What is worth testing here is not the agent -- `test_loop.py` owns that -- but
the state machine around it. One socket carries three different kinds of
inbound frame: a new instruction, an answer to a question the agent asked, and
an answer to a question the *gate* asked, with a worker thread blocked inside
the approver waiting for the third. Routing one of them to the wrong place puts
two agents on one ledger session, or hangs the run forever.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="the `ui` extra is not installed"
)

from a4s.kernel.loop import RunResult  # noqa: E402
from a4s.server import chat  # noqa: E402


class FakeLoop:
    """Stands in for `AgentLoop`. Blocking, like the real one."""

    def __init__(self, session: Any, script: List[Any]) -> None:
        self.session = session
        self.observer = session._observe
        self.script = list(script)
        self.instructions: List[str] = []

    def run(self, instruction: str) -> RunResult:
        self.instructions.append(instruction)
        step = self.script.pop(0) if self.script else RunResult("finished", "done", 1)
        if callable(step):
            return step(self)
        return step


@pytest.fixture
def chat_app(tmp_path: Path, monkeypatch):
    """Build the app with `ChatSession.build` replaced by a scripted loop.

    The seam is `build` rather than the provider, because everything it
    assembles -- the gate, the ledger, the tool registry, a model client --
    would otherwise have to exist for a test about frame routing.
    """
    monkeypatch.setenv("A4S_HOME", str(tmp_path))
    monkeypatch.setenv("A4S_AAS_ENDPOINT", "http://127.0.0.1:9")
    built: Dict[str, Any] = {}

    def make(script: List[Any]):
        def fake_build(self: chat.ChatSession) -> None:
            self.loop_ref = FakeLoop(self, script)
            built["session"] = self
            built["loop"] = self.loop_ref

        # `_build` rather than `build`: the public one dispatches to the
        # session's single worker thread, and that dispatch is the part these
        # tests need to keep exercising.
        monkeypatch.setattr(chat.ChatSession, "_build", fake_build)
        from a4s.server import build_app

        client = fastapi_testclient.TestClient(build_app("http://127.0.0.1:9"))
        return client, built

    return make


def _collect(socket: Any, until: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Read frames until one of `until` type arrives."""
    seen = []
    for _ in range(limit):
        message = socket.receive_json()
        seen.append(message)
        if message.get("type") == until:
            return seen
    raise AssertionError(f"never saw {until!r}; got {[m.get('type') for m in seen]}")


def test_approval_keeps_workflow_instruction_until_finished(chat_app):
    client, built = chat_app([RunResult('awaiting_operator', 'Approve maintenance?', 1),
                              RunResult('finished', 'Maintenance and replanning verified', 1)])
    prompts = []
    with client.websocket_connect('/ws/chat') as socket:
        socket.send_json({'message': 'Maintenance investigation: fault transition detected.'})
        _collect(socket, 'awaiting_input')
        session = built['session']
        session._prompt_for = lambda instruction: prompts.append(instruction) or instruction
        built['loop'].set_system_prompt = lambda prompt: None
        socket.send_json({'message': 'Approve'})
        _collect(socket, 'complete')
        assert prompts == ['Maintenance investigation: fault transition detected.']
        assert session._pending_instruction is None


# ---------------------------------------------------------------------------
# the ordinary turn
# ---------------------------------------------------------------------------

def test_a_run_streams_the_shapes_the_chat_pane_already_draws(chat_app) -> None:
    """The pane's switch statement is the contract, and it was written against
    the predecessor. Emitting a shape it does not handle shows as a pane that
    stays empty while the agent works."""
    def script(loop: FakeLoop) -> RunResult:
        loop.observer("assistant", {"text": "Reading the datasheet.", "step": 1})
        loop.observer("tool_call", {"id": "c1", "name": "shell",
                                    "arguments": {"command": "aas shell list"}})
        loop.observer("tool_result", {"id": "c1", "name": "shell", "output": "2 shells"})
        return RunResult("finished", "listed the shells", 2)

    client, _ = chat_app([script])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "list the shells"})
        seen = _collect(socket, "complete")

    kinds = [m["type"] for m in seen]
    assert "assistant_message" in kinds
    assert kinds.count("tool_call") == 2          # called, then completed
    called, completed = [m for m in seen if m["type"] == "tool_call"]
    assert called["tool_call_id"] == completed["tool_call_id"] == "c1"
    assert called["tool_status"] == "called"
    assert completed["tool_status"] == "completed"
    assert completed["tool_result"] == "2 shells"


def test_the_wizards_context_travels_in_the_instruction(chat_app) -> None:
    """Into the instruction rather than the system prompt, because the
    instruction is what the ledger records as the user turn. Context that
    reached the model another way would be invisible to a replay."""
    client, built = chat_app([RunResult("finished", "ok", 1)])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({
            "message": "extract the datasheet",
            "file_context": {"name": "d.pdf", "path": "/tmp/d.pdf"},
            "wizard_context": {"aasServerEndpoint": "http://basyx:8081",
                               "targetAAS": "AGV_042",
                               "selectedTemplates": ["urn:tpl:nameplate"]},
        })
        _collect(socket, "complete")

    (instruction,) = built["loop"].instructions
    assert instruction.startswith("extract the datasheet")
    assert "/tmp/d.pdf" in instruction
    assert "AGV_042" in instruction
    assert "urn:tpl:nameplate" in instruction


def test_a_state_update_is_not_an_instruction(chat_app) -> None:
    """The pane pushes wizard state as it changes. Treating one as a message
    would start a run every time somebody ticked a checkbox."""
    client, built = chat_app([RunResult("finished", "ok", 1)])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"type": "state_update", "key": "targetAAS", "value": "AGV"})
        socket.send_json({"message": "now do it"})
        _collect(socket, "complete")

    assert built["loop"].instructions == ["now do it"]


def test_stop_frame_ends_busy_turn_then_accepts_new_instruction(chat_app) -> None:
    started = threading.Event()

    def running(loop: FakeLoop) -> RunResult:
        started.set()
        assert loop.session.stop_event.wait(5)
        return RunResult("stopped", "Stopped by the operator", 1)

    client, built = chat_app([running, RunResult("finished", "new answer", 1)])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "long task"})
        assert started.wait(5)
        socket.send_json({"type": "stop"})
        _collect(socket, "stopped")
        socket.send_json({"message": "new task"})
        _collect(socket, "complete")
    assert built["loop"].instructions == ["long task", "new task"]


def test_asking_the_operator_releases_the_pane_without_ending_the_session(
    chat_app
) -> None:
    """`awaiting_input` and `complete` are different events in the pane: one
    lets the wizard opportunistically load results, the other says the run
    finished. Sending `complete` for a question would make the wizard give up
    waiting for an extraction that has not happened."""
    client, built = chat_app([
        RunResult("awaiting_operator", "Which template applies?", 1),
        RunResult("finished", "used the nameplate template", 2),
    ])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "build the passport"})
        first = _collect(socket, "awaiting_input")
        socket.send_json({"message": "the nameplate one"})
        _collect(socket, "complete")

    assert first[-1]["content"] == "Which template applies?"
    assert built["loop"].instructions[1].startswith("the nameplate one")


def test_a_failed_run_is_reported_rather_than_swallowed(chat_app) -> None:
    client, _ = chat_app([RunResult("budget_exhausted", "ran out of steps", 40)])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "do something enormous"})
        seen = _collect(socket, "complete")

    assert any(m["type"] == "error" and "ran out of steps" in m["content"] for m in seen)


# ---------------------------------------------------------------------------
# the gate, asking through the socket
# ---------------------------------------------------------------------------

class _Action:
    command = "aas set urn:sm BarrelZone3 255"
    subject = "urn:sm/BarrelZone3"
    moves_machine = True


class _Request:
    action = _Action()

    def as_dict(self) -> Dict[str, Any]:
        return {"question": "Approve this write?",
                "options": ["Approve, just this once", "Refuse, and tell the agent why"]}


@pytest.mark.parametrize("answer,allowed,note", [
    ("Approve, just this once", True, ""),
    ("Refuse, and tell the agent why", False, ""),
    ("yes, but 240", False, "yes, but 240"),
])
def test_the_gate_asks_through_the_socket_and_free_text_is_an_instruction(
    chat_app, monkeypatch, answer: str, allowed: bool, note: str
) -> None:
    """Free text is the point, not a fallback.

    An operator faced with "set BarrelZone3 to 255" often wants neither yes nor
    no but "yes, but 240" -- and the agent can act on that sentence, where a
    bare refusal only tells it to stop. Same contract as `terminal_approver`.
    """
    # A short deadline so a regression fails instead of hanging. The defect
    # this guards against is the receive loop awaiting the run: the answer is
    # then never read, the approver waits out its whole timeout, and a test
    # that blocks for fifteen minutes tells nobody anything.
    monkeypatch.setattr(chat, "APPROVAL_TIMEOUT_SECONDS", 5.0)
    decisions: List[Any] = []

    def script(loop: FakeLoop) -> RunResult:
        # The real gate calls the approver from the agent thread, which is
        # where this runs too: `run` is dispatched with `asyncio.to_thread`.
        # Reached through the loop's own session rather than through a variable
        # the test thread fills in later -- the run starts before that
        # assignment happens.
        decisions.append(loop.session.approver(_Request()))
        return RunResult("finished", "done", 1)

    client, _ = chat_app([script])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "set the barrel temperature"})
        question = _collect(socket, "tool_call")[-1]
        socket.send_json({"message": answer})
        _collect(socket, "complete")

    assert question["tool_name"] == "ask_user"
    assert question["tool_args"]["moves_machine"] is True
    (decision,) = decisions
    assert decision.allowed is allowed
    assert decision.note == note


def test_an_answer_to_the_gate_is_not_started_as_a_new_run(
    chat_app, monkeypatch
) -> None:
    """The run is blocked inside the approver when that frame arrives.

    Treating it as an instruction would start a second agent on the same ledger
    session while the first one waits for an answer that is never coming.
    """
    monkeypatch.setattr(chat, "APPROVAL_TIMEOUT_SECONDS", 5.0)

    def script(loop: FakeLoop) -> RunResult:
        loop.session.approver(_Request())
        return RunResult("finished", "done", 1)

    client, built = chat_app([script])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "set the barrel temperature"})
        _collect(socket, "tool_call")
        socket.send_json({"message": "Approve, just this once"})
        _collect(socket, "complete")

    assert built["loop"].instructions == ["set the barrel temperature"]


def test_nobody_at_the_browser_is_a_refusal_not_a_hang() -> None:
    """A run left blocked forever holds a thread and a ledger session open. The
    gate already treats "no answer" as a refusal; this only has to reach it."""
    from a4s.server.chat import BrowserApprover

    class _Sink:
        def emit_threadsafe(self, kind, payload):  # noqa: ANN001 - test double
            pass

    approver = BrowserApprover(_Sink())
    original = chat.APPROVAL_TIMEOUT_SECONDS
    chat.APPROVAL_TIMEOUT_SECONDS = 0.05
    try:
        decision = approver(_Request())
    finally:
        chat.APPROVAL_TIMEOUT_SECONDS = original

    assert decision.allowed is False
    assert "nobody answered" in decision.reason


def test_a_second_message_while_the_agent_works_is_refused(chat_app) -> None:
    """Two runs on one session interleave their turns in the ledger, and the
    replay that resumes a halted run cannot be untangled afterwards."""
    release = threading.Event()

    def script(loop: FakeLoop) -> RunResult:
        release.wait(5.0)
        return RunResult("finished", "done", 1)

    client, _ = chat_app([script])
    with client.websocket_connect("/ws/chat") as socket:
        socket.send_json({"message": "the first one"})
        socket.send_json({"message": "the second one"})
        busy = socket.receive_json()
        release.set()
        _collect(socket, "complete")

    assert busy["type"] == "error"
    assert "still working" in busy["content"]


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_clears_the_pane_and_says_it_deletes_nothing(chat_app) -> None:
    """A route that deleted from the ledger would break the hash chain, which
    is the property the log exists for."""
    client, _ = chat_app([])

    body = client.post("/api/chat/reset").json()

    assert body["status"] == "ok"
    assert "not deleted" in body["message"]


def test_long_tool_output_is_trimmed_for_the_socket_only() -> None:
    """A submodel dump is tens of thousands of characters into a collapsed card
    nobody expands. The agent still sees all of it."""
    trimmed = chat._trim("x" * 10_000, limit=100)

    assert trimmed.startswith("x" * 100)
    assert "9900 more characters" in trimmed

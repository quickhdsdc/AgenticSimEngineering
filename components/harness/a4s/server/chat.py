"""`/ws/chat` -- the agent, in the browser.

This is the exception `a4s/server/__init__.py` names. Every other route here
imports the library and calls it; inside this one a real `AgentLoop` runs, and
the agent reaches the repository the way it always does, through `shell` and
`aas`. Nothing about the agent is special-cased for the web: same five tools,
same system prompt, same ledger.

**The approval gate does run here, and this is the case it was built for.** The
decision of 2026-08-27 says the gate is an agent control, and an agent is
exactly what is on the other side of this socket. What was missing on the write
routes -- a human to ask -- is present: somebody is looking at the chat pane.
So the approver is the browser. A gated action pushes the question down the
socket, the worker thread blocks on the answer, and the operator's own words
come back as the decision's note, which is the same contract
`terminal_approver` has.

**One connection is one session.** The loop keeps its message list across
turns, so a follow-up continues the conversation rather than starting a new
one. Closing the tab ends it; the ledger holds what happened, and `a4s run
--resume` can pick it up from there, which is the point of there being one log.

**The run happens on one dedicated thread, and "one" is load-bearing.**
`AgentLoop.run` is blocking -- model calls, subprocesses, HTTP to BaSyx -- so
running it on the event loop would freeze every other request in the process,
including the ones the wizard makes while the agent works. But a thread *pool*
is not enough: the ledger is sqlite, its connection belongs to the thread that
opened it, and `asyncio.to_thread` hands the second turn to whichever worker is
free. That fails on turn two with "SQLite objects created in a thread can only
be used in that same thread" -- found by driving a real agent through the
socket, because the fakes in `test_chat.py` never touch a database. So the
session owns a single-worker executor, the loop and its ledger are built on
that thread, and everything that touches them runs there.

Events come back through a queue rather than by touching the socket from that
thread, because a websocket send from off the event loop is undefined
behaviour that usually looks like it works.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# How long the agent waits for an operator who has walked away. The run is not
# abandoned when it expires -- the gate is told nobody answered, which it
# already treats as a refusal, so the agent is told why and can report it.
APPROVAL_TIMEOUT_SECONDS = 900.0


def _home() -> Path:
    return Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser()


# ---------------------------------------------------------------------------
# the approver on the other end of a socket
# ---------------------------------------------------------------------------

class BrowserApprover:
    """Ask the person looking at the chat pane, from the worker thread.

    Two threads meet here and the handoff is the whole class: the agent thread
    blocks on an `Event`, the event loop fills in the answer when the browser
    replies. The question goes out as an `ask_user` tool call because the chat
    pane already draws those with option buttons -- reusing the shape it knows
    is what lets the gate appear in the UI without touching a component.
    """

    def __init__(self, session: "ChatSession") -> None:
        self._session = session
        self._answered = threading.Event()
        self._answer: Optional[str] = None
        self.pending: bool = False

    def __call__(self, request: Any) -> Any:
        from ..kernel.approval import Decision

        self._answered.clear()
        self._answer = None
        self.pending = True
        payload = request.as_dict()
        self._session.emit_threadsafe("tool_call", {
            "type": "tool_call",
            "tool_call_id": f"approval-{uuid.uuid4().hex[:8]}",
            "tool_name": "ask_user",
            "tool_args": {
                "question": payload["question"],
                "options": payload["options"],
                "command": request.action.command,
                "subject": request.action.subject,
                "moves_machine": request.action.moves_machine,
            },
            "tool_status": "called",
        })

        answered = self._answered.wait(APPROVAL_TIMEOUT_SECONDS)
        self.pending = False
        if not answered:
            return Decision(False, "nobody answered at the browser", "operator")

        answer = (self._answer or "").strip()
        if answer.lower() in ("1", "y", "yes", "approve") or answer.startswith("Approve"):
            return Decision(True, "approved in the browser", "operator")
        if answer.lower() in ("2", "n", "no", "") or answer.startswith("Refuse"):
            return Decision(False, "refused in the browser", "operator")
        # Free text is the point, not a fallback. "yes, but 240" is neither an
        # approval nor a refusal, and the agent can act on the sentence where a
        # bare no only tells it to stop.
        return Decision(False, "the operator sent an instruction instead", "operator",
                        note=answer)

    def answer(self, text: str) -> None:
        self._answer = text
        self._answered.set()


# ---------------------------------------------------------------------------
# one connection
# ---------------------------------------------------------------------------

class ChatSession:
    """The agent, its ledger, and the queue its events leave by."""

    def __init__(self, websocket: WebSocket, endpoint: str) -> None:
        self.websocket = websocket
        self.endpoint = endpoint
        self.session_id = f"web-chat-{uuid.uuid4().hex[:12]}"
        self.loop_ref: Any = None
        self._prompt_for: Any = None
        self._pending_instruction: str | None = None
        self.ledger: Any = None
        self.approver = BrowserApprover(self)
        self.stop_event = threading.Event()
        self.busy = False
        self._queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self._event_loop = asyncio.get_event_loop()
        # One worker, for the life of the connection. See the module docstring:
        # the ledger's sqlite connection belongs to whichever thread opened it.
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"a4s-chat-{self.session_id[-6:]}"
        )

    # -- events out ------------------------------------------------------

    def send(self, payload: Dict[str, Any]) -> None:
        """Queue one frame. From the event loop; `pump` does the sending.

        Everything outbound goes through the queue, including the replies the
        request handler writes. Two coroutines calling `send_json` on one
        websocket can interleave their frames, and a half of one JSON message
        inside another is a parse error in the browser with nothing on either
        side to explain it.
        """
        self._queue.put_nowait(payload)

    def emit_threadsafe(self, kind: str, payload: Dict[str, Any]) -> None:
        """The same, from the agent thread."""
        self._event_loop.call_soon_threadsafe(self._queue.put_nowait, payload)

    async def pump(self) -> None:
        """Forward events to the browser until the connection goes away."""
        try:
            while True:
                payload = await self._queue.get()
                await self.websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            return

    # -- the agent -------------------------------------------------------

    async def build(self) -> None:
        """Assemble the loop on the worker thread, and wait for it.

        On that thread because the ledger is opened here, and every later write
        to it happens there too.
        """
        await self._event_loop.run_in_executor(self._worker, self._build)

    def _build(self) -> None:
        """Assemble the loop, exactly as `a4s run` assembles it.

        Deliberately the same wiring rather than a second, simpler one: an
        agent that behaves differently in the browser than on the command line
        would make every observation made through this window untrue of the
        thing being observed.
        """
        from ..cli import SKILLS_ROOT
        from ..environments.local import LocalEnvironment
        from ..kernel.approval import (
            Gate,
            Policy,
            Scopes,
            delegation_index,
            repository_lookup,
        )
        from ..kernel.loop import AgentLoop, LoopConfig
        from ..kernel.prompt import build_system_prompt
        from ..kernel.providers import ModelConfig, build
        from ..ledger import Ledger
        from ..tools import ToolRegistry

        home = _home()
        self.ledger = Ledger(home / "ledger.db", self.session_id, actor="web-chat")

        policy = Policy.from_env()
        gate = Gate(
            policy=policy,
            ledger=self.ledger,
            approver=self.approver if policy.level != "off" else None,
            lookup_moves_machine=repository_lookup(self.endpoint),
            delegation_targets=(
                delegation_index(self.endpoint) if policy.level != "off" else {}
            ),
            # `Scopes.load`, the same call the CLI makes. This was `Scopes()`
            # -- an unconfigured instance, which permits every write -- so a
            # deployment that had written a scope file had it obeyed on the
            # command line and ignored in the browser. A missing file is
            # unconfigured either way, so the drift bit exactly the
            # installation that had bothered to restrict anything.
            scopes=Scopes.load(),
            actor="web-chat",
        )

        workdir = Path(os.getenv("A4S_WEB_WORKDIR") or Path.cwd())
        environment = LocalEnvironment(
            cwd=str(workdir), env={"A4S_AAS_ENDPOINT": self.endpoint}
        )
        tools = ToolRegistry(environment, self.ledger, workdir, gate=gate)

        model = ModelConfig.from_env(None)
        # Kept so each turn can rebuild the prompt against what was just
        # asked. See `run`.
        self._prompt_for = lambda instruction: build_system_prompt(
            SKILLS_ROOT,
            endpoint=self.endpoint,
            instruction=instruction,
            shell=environment.shell_name,
        )
        self.loop_ref = AgentLoop(
            provider=build(model),
            model=model,
            tools=tools,
            ledger=self.ledger,
            # No instruction yet -- nobody has typed anything. `run` rebuilds
            # this against each message, which is where a skill can match.
            system_prompt=self._prompt_for(""),
            config=LoopConfig(max_steps=int(os.getenv("A4S_WEB_MAX_STEPS", "40"))),
            observer=self._observe,
            stop_event=self.stop_event,
        )

    def _observe(self, kind: str, payload: Dict[str, Any]) -> None:
        """Translate loop events into the messages the chat pane already draws."""
        if kind == "thinking":
            self.emit_threadsafe(kind, {"type": "thinking", "content": "Working..."})
        elif kind == "assistant":
            if payload.get("text"):
                self.emit_threadsafe(kind, {"type": "assistant_message",
                                            "content": payload["text"]})
        elif kind == "tool_call":
            self.emit_threadsafe(kind, {
                "type": "tool_call",
                "tool_call_id": payload["id"],
                "tool_name": payload["name"],
                "tool_args": payload.get("arguments") or {},
                "tool_status": "called",
            })
        elif kind == "tool_result":
            self.emit_threadsafe(kind, {
                "type": "tool_call",
                "tool_call_id": payload["id"],
                "tool_name": payload["name"],
                "tool_result": _trim(payload.get("output") or ""),
                "tool_status": "completed",
            })

    async def run(self, instruction: str) -> Any:
        """Run one turn on the session's own thread and await the result.

        The system prompt is rebuilt against this instruction before the turn.
        The browser session is built once and every message runs through it, so
        a prompt fixed at construction could never carry a matched skill -- the
        first message had not been typed yet, and `instruction=""` selects
        nothing. That left the web arm doing what `select_skill`'s docstring
        measured and rejected: rediscovering a procedure by trial and error,
        three runs out of three.
        """
        self.stop_event.clear()
        if self._prompt_for is not None:
            # Approval replies must retain the original workflow procedure.
            # Otherwise "Approve" drops the skill before its replanning step.
            self.loop_ref.set_system_prompt(self._prompt_for(self._pending_instruction or instruction))
        result = await self._event_loop.run_in_executor(
            self._worker, self.loop_ref.run, instruction
        )
        self._pending_instruction = (self._pending_instruction or instruction) if result.status == 'awaiting_operator' else None
        return result

    def close(self) -> None:
        """Close the ledger where it was opened, then let the thread go."""
        if self.ledger is not None:
            try:
                self._worker.submit(self.ledger.close).result(timeout=5)
            except Exception:  # noqa: BLE001 - the connection is going away
                pass
        self._worker.shutdown(wait=False)


def _trim(text: str, limit: int = 4000) -> str:
    """Tool output as the browser gets it.

    A submodel dump is tens of thousands of characters and the pane renders it
    in a collapsed card nobody expands. The agent still sees all of it; this
    only bounds what crosses the socket.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} more characters]"


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------

@router.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    from ..aasctl.client import Endpoint

    await websocket.accept()
    try:
        endpoint = Endpoint.from_env(None).base_url
    except RuntimeError as exc:
        await websocket.send_json({"type": "error", "content": str(exc)})
        await websocket.close()
        return

    session = ChatSession(websocket, endpoint)
    try:
        await session.build()
    except Exception as exc:  # noqa: BLE001 - the browser needs the reason
        await websocket.send_json({
            "type": "error",
            "content": f"cannot start the agent: {type(exc).__name__}: {exc}",
        })
        await websocket.close()
        session.close()
        return

    pump = asyncio.create_task(session.pump())
    running: List[asyncio.Task] = []
    try:
        while True:
            raw = await websocket.receive_text()
            # As a task, not awaited. Awaiting the run here would stop this
            # loop from reading, and the frame it would then never read is the
            # operator's answer to the approval question the run is blocked on
            # -- a deadlock that only appears once the gate actually fires.
            running.append(asyncio.create_task(_handle(session, raw)))
            running = [task for task in running if not task.done()]
    except WebSocketDisconnect:
        pass
    finally:
        session.stop_event.set()
        if session.approver.pending:
            session.approver.answer("Refuse: browser disconnected")
        pump.cancel()
        for task in running:
            task.cancel()
        session.close()


async def _handle(session: ChatSession, raw: str) -> None:
    try:
        frame = json.loads(raw)
    except json.JSONDecodeError:
        session.send({"type": "error", "content": "the message was not JSON"})
        return

    if frame.get("type") == "state_update":
        # The pane pushes wizard state as it changes. Nothing here reads it:
        # the agent is told what it needs in the instruction itself, and a
        # second channel of context that the ledger does not record would make
        # a run unreproducible from the log.
        return

    if frame.get("type") == "stop":
        if session.busy:
            session.stop_event.set()
            if session.approver.pending:
                session.approver.answer("Refuse: operator stopped the run")
            session.send({"type": "stopping", "content": "Stopping after the current call returns..."})
        return

    text = str(frame.get("message") or "").strip()
    if not text:
        return

    # An answer to a gate question, not a new instruction. The run is blocked
    # inside the approver at this moment; starting a second one would put two
    # agents on one ledger session.
    if session.approver.pending:
        session.approver.answer(text)
        return

    if session.busy:
        session.send({
            "type": "error",
            "content": "the agent is still working on the previous message",
        })
        return

    instruction = _with_context(text, frame)
    session.busy = True
    try:
        result = await session.run(instruction)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        session.busy = False
        session.send({"type": "error", "content": f"{type(exc).__name__}: {exc}"})
        return
    session.busy = False

    if result.status == "awaiting_operator":
        # The pane renders the question as ordinary assistant feedback as well
        # as retaining the collapsed tool evidence, and releases the send button.
        session.send({"type": "awaiting_input", "content": result.summary})
        return

    if result.status == "stopped":
        session.send({"type": "stopped", "content": result.summary})
        return
    if result.status not in ("finished",):
        session.send({"type": "error", "content": result.summary})
    session.send({"type": "complete", "result": result.summary})


def _with_context(text: str, frame: Dict[str, Any]) -> str:
    """Fold the pane's context blocks into the instruction itself.

    Into the instruction rather than into the system prompt, because the
    instruction is what the ledger records as the user turn. Context that
    reached the model by another route would be invisible to `a4s ledger` and
    to a replay, and a run nobody can reconstruct is not evidence of anything.
    """
    lines: List[str] = [text]

    files = frame.get("file_context") or None
    if files and files.get("path"):
        lines.append(f"\nUploaded files: {files.get('name')}\nPaths: {files['path']}")

    wizard = frame.get("wizard_context") or None
    if wizard:
        details = [
            f"AAS repository: {wizard.get('aasServerEndpoint')}",
            f"Target AAS idShort: {wizard.get('targetAAS')}",
        ]
        if wizard.get("selectedTemplates"):
            details.append(
                "Selected submodel templates: "
                + ", ".join(str(t) for t in wizard["selectedTemplates"])
            )
        if wizard.get("editedEntitiesFile"):
            details.append(f"Edited entities file: {wizard['editedEntitiesFile']}")
        lines.append("\nWizard context:\n" + "\n".join(f"- {d}" for d in details))

    floor = frame.get("shop_floor_context") or None
    if floor and (floor.get("assets") or floor.get("connections")):
        assets = ", ".join(
            f"{a.get('label')} ({a.get('aasId') or 'no AAS yet'})"
            for a in floor.get("assets") or []
        )
        lines.append(f"\nShop floor: {assets or 'empty'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

@router.post("/api/chat/reset")
def reset_chat() -> Dict[str, Any]:
    """Clear the pane, and say what that does and does not do.

    It does not erase anything. The conversation lives in an append-only,
    hash-chained ledger, and a route that deleted from it would be a route that
    breaks the chain -- the property the log exists for. Reconnecting the
    socket is what starts a fresh session; the old one stays readable with
    `a4s ledger`.
    """
    return {
        "status": "ok",
        "message": (
            "the pane is cleared. Reconnect to start a new agent session; "
            "the previous one stays in the ledger and is not deleted"
        ),
    }


__all__ = ["router"]

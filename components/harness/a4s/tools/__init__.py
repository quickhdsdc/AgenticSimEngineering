"""The tool surface -- deliberately five entries.

The count is the design. The system this replaces exposed twelve to sixteen
LLM tools, most of them AAS verbs: read a property, write a property,
instantiate from a template, populate matched inputs. Every one of those
occupied a slot in the schema the model sees on every single request, and
every new submodel template argued for another.

Here the AAS is not a tool. It is a command-line program that ``shell``
runs, so the entire metamodel costs one slot, its documentation lives in
``aas --help`` where the agent can fetch it on demand instead of carrying
it, and a new template costs zero slots.

The second-order effect matters more than the token saving: the agent can
*compose*. A pipeline that reads a tree, filters it, and writes twenty
elements is one shell call and one turn. As sixteen tools it was twenty
round trips, twenty chances to lose the thread, and twenty entries in the
context.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..environments.base import Environment, ExecResult
from ..ledger import Kind, Ledger


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., str]
    # Mutating tools are the ones an approval policy and the ledger care
    # about. Marking them here keeps that knowledge out of the loop.
    mutates: bool = False

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """A tool answer that may carry pages the model has to *see*.

    Most tools answer with text and keep returning `str`; this exists for the
    ones whose answer is a document. `read_file` on a PDF is the case: the
    labels and the numbers of a specification table are separate runs in the
    text layer, so the association between them lives in the two-dimensional
    layout and a linear extraction loses it. Measured on the ARBURG datasheet:
    page 3's text layer holds 126 numbers and every label, and pairs almost
    none of them.

    **Attachments are delivered differently by each provider, and neither way
    is optional.** Anthropic accepts image blocks inside a `tool_result`.
    OpenAI and Azure do not -- a `role: "tool"` message's content must be a
    string -- so the images follow as a separate `user` message. Hiding that
    difference here is the point: a tool should not know which model is
    reading it.
    """

    text: str
    # PNG bytes, one per page or image. Not paths: the conversation has to
    # carry the pixels, and a path would make the record depend on a file that
    # can change or vanish after the fact.
    images: List[bytes] = field(default_factory=list)
    # What each attachment is, for the text placeholder. Same length as
    # `images` when present.
    labels: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


def default_roots(workdir: Path) -> List[Path]:
    """Where the file tools may read and write.

    The working directory, and `$A4S_HOME` -- which is not a convenience. The
    wizard hands the agent absolute paths into `$A4S_HOME/web/uploads`,
    `/extractions` and `/matches`, and those are the documents the whole task
    is about. A workdir-only rule would refuse the agent its own inputs.
    """
    home = Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser()
    return [workdir.resolve(), home.resolve()]


class ToolRegistry:
    def __init__(
        self,
        environment: Environment,
        ledger: Ledger,
        workdir: Path,
        gate: Optional[Any] = None,
        roots: Optional[Sequence[Path]] = None,
    ) -> None:
        self.environment = environment
        self.ledger = ledger
        self.workdir = workdir
        # Roots the file tools are confined to. `read_file` and `write_file`
        # took any absolute path, so the agent could read `~/.ssh/id_rsa` or
        # overwrite a file anywhere the process could write. That is worth
        # closing on its own, but the reason it is urgent is the other half of
        # this harness: the agent reads supplier PDFs and web pages, and text
        # in them reaches the model as context. An instruction hidden in a
        # datasheet is the thing that would use an unbounded path.
        #
        # Not a security boundary on its own -- the `shell` tool is still a
        # shell, and confining *that* needs a container, not a path check. This
        # closes the tools that have no reason to leave.
        self.roots = [Path(r).resolve() for r in (roots or default_roots(workdir))]
        self.gate = gate
        self._tools: Dict[str, ToolSpec] = {}
        # Turn-local, and cleared by `begin_turn`. They are attributes rather
        # than return values because the loop has to see them from inside a
        # dispatch, but they describe *this* turn and nothing longer: the web
        # chat builds one registry and runs many turns through it, so a
        # `finished` left standing from the last task ends the next one.
        self.finished: Optional[str] = None
        self.pending_question: Optional[str] = None
        # Set when the gate refuses a call. A refusal ends the *batch* -- the
        # model has to see it before it plans the rest -- but not the run.
        self.refused: Optional[str] = None
        self._register_builtins()

    def begin_turn(self) -> None:
        """Clear the state that describes one turn.

        Called by `AgentLoop.run`. Without it a session that finished a task
        carries `finished` into the next one, and the first tool call of the
        next task runs, has its result discarded, and the run reports the
        previous task's summary as though it were this task's answer. Measured
        in the web chat, which reuses one registry for every user message.
        """
        self.finished = None
        self.pending_question = None
        self.refused = None

    # -- registry --------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def schemas(self) -> List[Dict[str, Any]]:
        return [spec.schema() for spec in self._tools.values()]

    def names(self) -> List[str]:
        return list(self._tools)

    def dispatch(
        self, name: str, arguments: Dict[str, Any], call_id: Optional[str] = None
    ) -> Union[str, ToolResult]:
        spec = self._tools.get(name)
        if spec is None:
            # Naming a recovery path beats a bare rejection: the model that
            # invented a tool name will otherwise invent another one.
            return (
                f"error: no tool named {name!r}. Available: {', '.join(self._tools)}. "
                f"AAS operations are not tools -- run the 'aas' command through shell."
            )
        mes_context = None
        if name == "shell":
            from ..mesctl.commands import command_info
            from ..mesctl.execution import ExecutionContext
            info = command_info(str(arguments.get("command", "")))
            if info:
                mes_context = ExecutionContext.for_tool(self.ledger.session_id, info.get("trace_id"))
        correlation = {"execution": mes_context.as_dict()} if mes_context else {}
        self.ledger.append(
            Kind.TOOL_CALL,
            {"tool": name, "arguments": arguments, "call_id": call_id, **correlation},
            subject=_subject_of(name, arguments),
        )

        # The gate sits between the model's intent and the action. It runs
        # after the call is recorded, so a refused attempt is in the ledger
        # too -- what the agent tried matters as much as what it did.
        if self.gate is not None:
            decision = self.gate.check(name, arguments, **correlation)
            if not decision.allowed:
                refusal = self.gate.refusal(decision)
                # A refusal is a boundary for the rest of this batch. The model
                # asked for several things at once on a plan that has just been
                # rejected; running the remainder acts on a plan nobody
                # approved, and it denies the model the chance to replan.
                self.refused = refusal
                if getattr(self.gate, "pending", None) is not None:
                    self.pending_question = self.gate.pending.render()
                self.ledger.append(
                    Kind.TOOL_RESULT,
                    {"tool": name, "output": refusal, "call_id": call_id, **correlation},
                    subject=_subject_of(name, arguments),
                )
                return refusal

        try:
            internal = {"_mes_context": mes_context} if mes_context else {}
            result = spec.handler(**arguments, **internal)
        except TypeError as exc:
            result = f"error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            result = f"error: {type(exc).__name__}: {exc}"
        # The ledger records the text and *that* pages were attached, not the
        # pages. Base64 pixels would multiply the log by the size of every
        # document ever read, and the run is reproducible from the path and
        # the page numbers, which are in the arguments already.
        recorded = str(result)[:4000]
        entry: Dict[str, Any] = {"tool": name, "output": recorded, "call_id": call_id, **correlation}
        if isinstance(result, ToolResult) and result.images:
            entry["attachments"] = len(result.images)
            entry["attachment_labels"] = result.labels[:20]
        self.ledger.append(
            Kind.TOOL_RESULT, entry, subject=_subject_of(name, arguments)
        )
        return result

    # -- built-ins -------------------------------------------------------

    def _register_builtins(self) -> None:
        self.register(
            ToolSpec(
                name="shell",
                description=(
                    "Run a shell command and return its output. This is how you reach "
                    "the asset layer: `aas` manages Asset Administration Shells "
                    "(run `aas --help`, `aas el add --help`), and the usual Unix tools "
                    "are available for filtering and composing. Prefer one composed "
                    "pipeline over several round trips."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command line to run."},
                        "timeout": {
                            "type": "number",
                            "description": "Seconds before the process tree is killed. Default 120.",
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                handler=self._shell,
                mutates=True,
            )
        )
        self.register(
            ToolSpec(
                name="read_file",
                description=(
                    "Read a file. Text files come back as numbered lines; a PDF "
                    "comes back as rendered page images you can look at, plus its "
                    "text layer; an image comes back as itself. For a table or a "
                    "drawing, read the picture -- a PDF's text layer separates "
                    "labels from their values and loses the pairing."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file, absolute or relative to the working directory.",
                        },
                        "offset": {"type": "integer", "description": "1-based first line."},
                        "limit": {"type": "integer", "description": "Line count. Default 400."},
                        "pages": {
                            "type": "string",
                            "description": (
                                "PDFs only: which pages, as \"3\", \"1-4\" or "
                                "\"1,3,5-7\". At most 8 per call; omit for the "
                                "first 8. Pages stay in the conversation, so ask "
                                "for the ones you need."
                            ),
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self._read_file,
            )
        )
        self.register(
            ToolSpec(
                name="write_file",
                description=(
                    "Write a UTF-8 text file, creating parent directories. Use it for "
                    "scripts, extracted data, and reports -- not for AAS content, which "
                    "belongs in the repository via `aas`."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Where to write, relative to the working directory.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The full contents to write. Replaces the file.",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self._write_file,
                mutates=True,
            )
        )
        self.register(
            ToolSpec(
                name="ask_user",
                description=(
                    "Ask the operator a question and stop until they answer. Use it when "
                    "an action is irreversible on a physical asset and the instruction is "
                    "genuinely ambiguous -- not to confirm work you can verify yourself."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "What you need the operator to decide, in one sentence.",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Choices to offer, if the answer is a choice.",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
                handler=self._ask_user,
            )
        )
        self.register(
            ToolSpec(
                name="finish",
                description=(
                    "End the task and report the outcome. State what was changed, "
                    "which AAS paths it touched, and anything left undone."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "The report: what you did, what you found, and what you "
                                "left undone."
                            ),
                        }
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
                handler=self._finish,
            )
        )

    # -- handlers --------------------------------------------------------

    def _shell(self, command: str, timeout: float = 120.0, *, _mes_context=None) -> str:
        if _mes_context:
            self.ledger.append(Kind.MES_EXECUTION, {
                "command": command, "stage": "started", "execution": _mes_context.as_dict(),
            }, subject=_subject_of("shell", {"command": command}))
        try:
            result: ExecResult = self.environment.run(command, timeout=timeout,
                **({"env": _mes_context.environment()} if _mes_context else {}))
        except Exception:
            if _mes_context:
                self.ledger.append(Kind.MES_EXECUTION, {
                    "command": command, "stage": "execution_error",
                    "execution": {**_mes_context.as_dict(), "outcome": "unknown"},
                }, subject=_subject_of("shell", {"command": command}))
            raise
        execution = _mes_context.as_dict() if _mes_context else {}
        if _mes_context:
            # Keep correlation even when bounded process output is not JSON.
            try:
                payload = json.loads(result.stdout if result.ok or result.stdout.strip() else result.stderr)
                reported = payload.get("execution", {}) if isinstance(payload, dict) else {}
                if reported.get("execution_id") == execution["execution_id"]:
                    execution.update({k: reported[k] for k in ("outcome", "request_fingerprint", "action") if k in reported})
            except (ValueError, TypeError, AttributeError):
                pass
            execution.setdefault("outcome", "command_completed" if result.ok else "failed_or_unverified")
            self.ledger.append(Kind.MES_EXECUTION, {
                "stage": "finished", "exit_code": result.exit_code, "execution": execution,
            }, subject=_subject_of("shell", {"command": command}))
        # An `aas` mutation gets its own ledger entry beyond the generic
        # tool.call pair, because provenance queries are indexed by AAS
        # path, not by tool name.
        if result.ok and _is_aas_mutation(command):
            self.ledger.append(
                Kind.AAS_WRITE,
                {"command": command, "stdout": result.stdout[:2000]},
                subject=_aas_subject(command),
            )
        # Retain failed/uncertain writes too: an HTTP timeout after commit is
        # not evidence that nothing changed. Each tool call already has its result.
        from ..kernel.approval import _segments
        from ..mesctl.commands import command_info

        for segment in _segments(command):
            info = command_info(segment)
            if info and info["mutates"]:
                self.ledger.append(
                    Kind.MES_WRITE,
                    {"command": command, "exit_code": result.exit_code,
                     "stdout": result.stdout[:4000], "stderr": result.stderr[:2000],
                     "outcome": execution.get("outcome", "command_completed" if result.ok else "failed_or_unverified"),
                     "execution": execution},
                    subject=info["subject"],
                )
        return result.for_model()

    # Extensions whose bytes are not text. Read as text they produce mojibake,
    # which the model then tries to reason about -- the failure this replaces.
    PAGED = {".pdf"}
    PICTURES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    # How many pages one call may attach. A page is on the order of 1.5k tokens
    # and they stay in context for the rest of the run, so an unbounded read of
    # a 200-page manual would end the session inside one tool call.
    MAX_PAGES = 8

    def _resolve(self, path: str) -> Union[Path, str]:
        """The absolute path this argument names, or why it is refused.

        `resolve()` before the check, not after: it collapses `..` and follows
        symlinks, so `../../.ssh/id_rsa` and a symlink planted in the workdir
        are both answered by the same containment test rather than by two
        special cases.
        """
        target = Path(path) if Path(path).is_absolute() else (self.workdir / path)
        target = target.resolve()
        for root in self.roots:
            if target == root or root in target.parents:
                return target
        allowed = ", ".join(str(r) for r in self.roots)
        return (
            f"error: {target} is outside the directories this session may use. "
            f"Allowed: {allowed}. If the file you need is elsewhere, ask the "
            f"operator to copy it in rather than reaching for it."
        )

    def _read_file(
        self,
        path: str,
        offset: int = 1,
        limit: int = 400,
        pages: Optional[str] = None,
    ) -> Union[str, "ToolResult"]:
        target = self._resolve(path)
        if isinstance(target, str):
            return target
        if not target.exists():
            return f"error: no such file: {target}"

        suffix = target.suffix.lower()
        if suffix in self.PAGED:
            return self._read_pdf(target, pages)
        if suffix in self.PICTURES:
            return self._read_picture(target)

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        window = lines[max(offset - 1, 0) : max(offset - 1, 0) + limit]
        numbered = "\n".join(f"{offset + i:>6}  {line}" for i, line in enumerate(window))
        suffix = (
            f"\n[{len(lines)} lines total; showing {offset}-{offset + len(window) - 1}]"
            if len(lines) > len(window)
            else ""
        )
        return numbered + suffix

    def _read_pdf(self, target: Path, pages: Optional[str]) -> Union[str, "ToolResult"]:
        """Hand the model the rendered pages, with no model call in between.

        The alternative -- a tool that reads the document *for* the agent by
        calling a vision model itself -- is what `a4s extract` does, and it has
        to for bulk work that must run without an agent. But a nested call is
        invisible to the ledger, does not count against the step budget, cannot
        be approved, and cannot be interrupted; a six-page datasheet once
        extracted three entities and reported success because five nested calls
        returned nothing and nobody could see it. Where the agent can look at
        the page itself, it should.

        The text layer is offered alongside, not instead. It is free and
        sometimes sufficient. It is not sufficient here: on the ARBURG
        datasheet page 3 it yields every label and 126 numbers, and pairs
        almost none of them, because a table's meaning is in its layout.
        """
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return ("error: reading PDFs needs pypdfium2: "
                    "pip install 'a4s-harness[documents]'")

        try:
            document = pdfium.PdfDocument(target)
            total = len(document)
        except Exception as exc:  # noqa: BLE001 - a corrupt file is an answer
            return f"error: cannot open {target.name}: {type(exc).__name__}: {exc}"

        wanted = _page_range(pages, total)
        if not wanted:
            return (f"error: {target.name} has {total} page(s); "
                    f"{pages!r} selects none of them")
        if len(wanted) > self.MAX_PAGES:
            return (f"error: {len(wanted)} pages requested and the limit is "
                    f"{self.MAX_PAGES}. {target.name} has {total}; read it in "
                    f'parts with pages="1-{self.MAX_PAGES}", then the next.')

        images: List[bytes] = []
        labels: List[str] = []
        text_layer: List[str] = []
        for number in wanted:
            page = document[number - 1]
            images.append(_render(page))
            labels.append(f"{target.name} page {number}")
            extracted = page.get_textpage().get_text_range().strip()
            if extracted:
                text_layer.append(f"--- page {number} text layer ---\n{extracted}")

        header = (f"{target.name}: {total} page(s), showing "
                  f"{', '.join(str(n) for n in wanted)} as images below.")
        if len(wanted) < total:
            remaining = [n for n in range(1, total + 1) if n not in wanted]
            header += (f" Not shown: {', '.join(str(n) for n in remaining[:20])}"
                       f"{' ...' if len(remaining) > 20 else ''}.")
        body = "\n\n".join(text_layer)
        if body:
            header += ("\n\nThe extracted text layer follows. It is unreliable for "
                       "tables -- labels and values are separate runs and their "
                       "pairing is lost -- so read the images for anything laid "
                       "out in columns.\n\n" + body)
        return ToolResult(text=header, images=images, labels=labels)

    def _read_picture(self, target: Path) -> Union[str, "ToolResult"]:
        try:
            data = target.read_bytes()
        except OSError as exc:
            return f"error: cannot read {target.name}: {exc}"
        if len(data) > 20 * 1024 * 1024:
            return f"error: {target.name} is {len(data) // (1024 * 1024)} MB; the limit is 20 MB"
        return ToolResult(
            text=f"{target.name} ({len(data) // 1024} KB) is attached below.",
            images=[data],
            labels=[target.name],
        )

    def _write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        if isinstance(target, str):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {target}"

    def _ask_user(self, question: str, options: Optional[List[str]] = None) -> str:
        self.pending_question = question
        rendered = question
        if options:
            rendered += "\n" + "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(options))
        return f"[waiting for operator]\n{rendered}"

    def _finish(self, summary: str) -> str:
        self.finished = summary
        return summary


# ---------------------------------------------------------------------------
# ledger subject extraction
# ---------------------------------------------------------------------------

_AAS_MUTATING_VERBS = (
    "aas set",
    "aas el add",
    "aas el rm",
    "aas sm create",
    "aas sm delete",
    "aas shell create",
    "aas shell delete",
    "aas shell attach",
    "aas op invoke",
)


def _is_aas_mutation(command: str) -> bool:
    normalised = " ".join(command.split())
    return any(verb in normalised for verb in _AAS_MUTATING_VERBS)


def _aas_subject(command: str) -> Optional[str]:
    """Derive ``<container-id>/<path>`` from an aas command line.

    The command is re-parsed with the CLI's own parser rather than scanned
    for patterns. A second, approximate copy of the grammar would drift from
    the real one, and a provenance index that silently stops matching is
    worse than one that never existed.
    """
    import contextlib
    import io
    import shlex

    from ..aasctl.__main__ import build_parser

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if "aas" not in tokens:
        return None

    tail = tokens[tokens.index("aas") + 1 :]
    # argparse writes usage to stderr and raises SystemExit on a command it
    # cannot parse -- which happens routinely here, since the agent may pipe
    # or chain. Neither should reach the caller.
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            args = build_parser().parse_args(tail)
        except SystemExit:
            return None

    container = getattr(args, "submodel_id", None) or getattr(args, "shell_id", None)
    if not container:
        return None

    path = getattr(args, "path", None)
    if path is None and hasattr(args, "parent"):
        # `el add` names the element by its parent plus --id-short; the
        # subject is where the element ends up, not where it was attached.
        parent = getattr(args, "parent", "") or ""
        id_short = getattr(args, "id_short", None)
        path = f"{parent}.{id_short}" if parent and id_short else (id_short or parent)

    return f"{container}/{path}" if path else container


def _page_range(spec, total):
    """Which pages a `pages` argument selects, 1-based and in order.

    Accepts what a person would write -- "3", "1-4", "1,3,5-7" -- and silently
    clamps to what the document has, because a model that asks for pages 1-10
    of a six-page file wants the six, not an error. An empty spec means the
    first pages up to the cap rather than all of them: reading a whole manual
    into the conversation is the mistake this parameter exists to prevent.
    """
    if not spec or not str(spec).strip():
        return list(range(1, min(total, ToolRegistry.MAX_PAGES) + 1))
    wanted = []
    for piece in str(spec).replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece[1:]:
            head, _, tail = piece.partition("-")
            try:
                low, high = int(head), int(tail)
            except ValueError:
                continue
            wanted.extend(range(min(low, high), max(low, high) + 1))
        else:
            try:
                wanted.append(int(piece))
            except ValueError:
                continue
    seen, out = set(), []
    for number in wanted:
        if 1 <= number <= total and number not in seen:
            seen.add(number)
            out.append(number)
    return out


def _render(page, scale=2.0):
    """One page as PNG bytes.

    `scale=2` is about 150 dpi. Below that the small print in a specification
    table stops being legible to the model, which is the whole reason the page
    is being shown rather than its text layer.
    """
    import io

    bitmap = page.render(scale=scale)
    buffer = io.BytesIO()
    bitmap.to_pil().save(buffer, format="PNG")
    return buffer.getvalue()


def _subject_of(tool: str, arguments: Dict[str, Any]) -> Optional[str]:
    if tool == "shell":
        from ..kernel.approval import classify

        command = str(arguments.get("command", ""))
        action = classify("shell", {"command": command})
        if action.subject and action.subject.startswith("mes://"):
            return action.subject
        return _aas_subject(command)
    if tool in ("read_file", "write_file"):
        return str(arguments.get("path"))
    return None

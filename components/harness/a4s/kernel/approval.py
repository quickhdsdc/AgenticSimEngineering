"""Deciding whether an action needs a person, and asking one.

The harness can now start and stop a machine and change its setpoints. Nothing
about a language model makes it a safe holder of that authority by default, so
there is a gate between the model's intent and the action.

**The unit is the action, not the tool.** ``ToolSpec.mutates`` was the obvious
hook and it does not work: in the CLI arm every AAS operation goes through
``shell``, which is marked mutating wholesale, so a gate reading that flag
either passes everything or stops the agent from reading a temperature. What
the command *does* is what matters, and that is recoverable from the command
line -- with the CLI's own parser, so a second approximate copy of the grammar
cannot drift from the real one.

**Danger is read from the model, not guessed from the name.** An Operation that
moves equipment carries ``Qualifier/type = "requiresSafetyApproval"`` on its element in
the AAS. The gate looks it up rather than pattern-matching ``Start`` or
``Setpoint`` against a list, so onboarding a new machine teaches the gate
without a code change. If the repository cannot be reached to check, the action
is treated as dangerous: a safety gate that fails open is not a safety gate.

**Every decision is recorded**, including the ones that were allowed
automatically. "Who approved changing this machine's temperature, and when" is
a question the ledger has to be able to answer.

**What this is not.** Gating ``aas op invoke`` alone protects nothing. A live
run showed the agent reaching the machine with
``curl http://localhost:5080/read``, which the gate waved through as an
ordinary shell command; had it been ``/setpoint`` the machine would have moved
unapproved. The gate now also recognises commands addressing any service the
repository delegates an Operation to. That closes the obvious hole and not the
class of it: an agent with a shell can reach anything the host can, and
``python -c`` will always be able to open a socket. This is a guardrail against
an agent going somewhere it should not, not a sandbox against one determined
to. Real confinement is a network policy around the host -- the SSH
environment's job, not this module's.
"""

from __future__ import annotations

import fnmatch
import json
import os
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# What a command is doing, in rising order of what it can cost you.
READ = "read"
WRITE = "write"
DELETE = "delete"
INVOKE = "invoke"
SHELL = "shell"

# An Operation that only reads is not the same risk as one that moves a press,
# and the model already says which is which. MOVING is a fifth level of risk
# rather than a flag consulted afterwards, so `machine` can gate exactly the
# invocations that matter and let the harmless ones through.
MOVING = "moving"
BUSINESS = "business"

# The `aas` verbs that change something, by the risk they carry. Anything not
# listed is a read.
VERB_RISK: Dict[str, str] = {
    "set": WRITE,
    "el add": WRITE,
    "el rm": DELETE,
    "sm create": WRITE,
    "sm delete": DELETE,
    "shell create": WRITE,
    "shell delete": DELETE,
    "shell attach": WRITE,
    "smt instantiate": WRITE,
    "populate": WRITE,
    "op invoke": INVOKE,
    # A write to a PLC node is an invocation by another route. Leaving it
    # ungated would mean the gate guards `op invoke` while the same effect is
    # one verb away: `xStart := true` starts the module whichever command
    # sends it.
    #
    # MOVING rather than INVOKE, and unconditionally. These verbs take a raw
    # endpoint and node id, so there is no AAS element to read a declaration
    # off -- the same position `_invoke_target` is in when it cannot parse a
    # target, and it answers the same way: what cannot be classified is
    # treated as dangerous. An agent that wants the declaration honoured calls
    # `aas op invoke`, where there is an Operation to read it from.
    "opc write": MOVING,
    "opc call": MOVING,
}

RISK_ORDER = (READ, SHELL, WRITE, DELETE, INVOKE, BUSINESS, MOVING)

# How strict to be. `machine` is the shopfloor default: reading and writing the
# model is ordinary work, and so is asking a machine a question. Telling it to
# *do* something is not.
LEVELS: Dict[str, Dict[str, str]] = {
    "off":        {READ: "allow", SHELL: "allow", WRITE: "allow", DELETE: "allow",
                   INVOKE: "allow", MOVING: "allow"},
    "machine":    {READ: "allow", SHELL: "allow", WRITE: "allow", DELETE: "allow",
                   INVOKE: "allow", MOVING: "ask"},
    "operations": {READ: "allow", SHELL: "allow", WRITE: "allow", DELETE: "allow",
                   INVOKE: "ask",   MOVING: "ask"},
    "writes":     {READ: "allow", SHELL: "allow", WRITE: "ask",   DELETE: "ask",
                   INVOKE: "ask",   MOVING: "ask"},
    "strict":     {READ: "allow", SHELL: "ask",   WRITE: "ask",   DELETE: "ask",
                   INVOKE: "ask",   MOVING: "ask"},
}
DEFAULT_LEVEL = "machine"

# MES mutations have production consequences even when no PLC moves.
for _level, _rules in LEVELS.items():
    _rules[BUSINESS] = "allow" if _level == "off" else "ask"


@dataclass
class Action:
    """What a tool call would actually do."""

    risk: str
    tool: str
    command: str
    subject: Optional[str] = None
    moves_machine: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if self.risk == BUSINESS:
            what = "change production data in OpenMES"
        elif self.risk == MOVING:
            what = "invoke an Operation that is declared to move the machine"
        elif self.risk == INVOKE:
            what = "invoke an Operation"
        elif self.risk == DELETE:
            what = "delete"
        elif self.risk == WRITE:
            what = "write to the AAS repository"
        elif self.risk == SHELL:
            what = "run a shell command"
        else:
            what = "read"
        where = f" at {self.subject}" if self.subject else ""
        return f"{what}{where}"


@dataclass
class Request:
    """What the operator is shown. Options *and* free text, always."""

    action: Action
    question: str
    options: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "options": self.options,
            "free_text": True,
            "action": asdict(self.action),
        }

    def render(self) -> str:
        lines = [self.question, "", f"  command: {self.action.command}"]
        if self.action.subject:
            lines.append(f"  subject: {self.action.subject}")
        if self.action.moves_machine:
            lines.append("  WARNING: this Operation is declared to move the machine")
        for effect in self.action.detail.get("effects", []):
            lines.append(f"  effect: {effect}")
        lines.append("")
        lines += [f"  {i + 1}. {o}" for i, o in enumerate(self.options)]
        lines.append("  ... or type an instruction in your own words")
        return "\n".join(lines)


@dataclass
class Decision:
    allowed: bool
    reason: str
    source: str                 # policy | operator | preauthorised | denylist
    note: str = ""              # what the operator typed, if anything


# The operator's side: given a request, return a decision. `None` means nobody
# is available, and the run halts rather than proceeding.
Approver = Callable[[Request], Decision]

APPROVE = "Approve, just this once"
DENY = "Refuse, and tell the agent why"


# ---------------------------------------------------------------------------
# classifying
# ---------------------------------------------------------------------------

def _aas_verb(command: str) -> Optional[str]:
    """The `aas` sub-command a line runs, as `"op invoke"` or `"set"`."""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    if "aas" not in tokens:
        return None
    tail = [t for t in tokens[tokens.index("aas") + 1 :] if not t.startswith("-")]
    if not tail:
        return None
    two = " ".join(tail[:2])
    return two if two in VERB_RISK else (tail[0] if tail[0] in VERB_RISK else None)


def _is_aas_line(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    return "aas" in tokens


def _segments(command: str) -> List[str]:
    """Split a compound command line into the pieces that run separately.

    `aas get X && aas op invoke Y` is two actions, and the gate has to see the
    second one. Splitting on the shell's own separators is approximate -- a
    separator inside quotes is missed -- but it errs towards finding *more*
    commands, which is the safe direction for a gate.
    """
    parts = [command]
    for separator in ("&&", "||", ";", "|", "\n"):
        parts = [piece for part in parts for piece in part.split(separator)]
    return [p.strip() for p in parts if p.strip()]


def classify(
    tool: str,
    arguments: Dict[str, Any],
    *,
    lookup_moves_machine: Optional[Callable[[str, str], bool]] = None,
    delegation_targets: Optional[Dict[str, bool]] = None,
) -> Action:
    """Work out what a tool call would do.

    ``lookup_moves_machine(submodel_id, path)`` answers whether an Operation is
    declared to move equipment. It is injected rather than imported so the
    classification can be tested without a repository, and so the gate does not
    reach the network for commands that are obviously reads.

    ``delegation_targets`` maps every delegation URL the repository knows to
    whether it moves a machine. It exists because gating ``aas op invoke``
    alone protects nothing: a live run showed the agent reaching the machine
    with ``curl http://localhost:5080/read`` and the gate waving it through as
    an ordinary shell command. Had it curled ``/setpoint`` the machine would
    have moved unapproved.
    """
    if tool != "shell":
        # Non-shell tools are the file and conversation ones, plus the whole
        # baseline arm. Their names say what they do.
        risk = WRITE if tool in ("write_file",) else READ
        if tool.startswith("aas_") or tool == "input_pdf2entity":
            risk = _baseline_tool_risk(tool)
        return Action(
            risk=risk, tool=tool, command=f"{tool}({_short(arguments)})",
            subject=_first_str(arguments, "submodel_id", "shell_id", "path"),
        )

    command = str(arguments.get("command") or "")
    worst = Action(risk=READ, tool=tool, command=command)
    for segment in _segments(command):
        from ..mesctl.commands import command_info

        mes = command_info(segment)
        if mes is not None:
            candidate = Action(risk=BUSINESS if mes["mutates"] else READ,
                               tool=tool, command=segment, subject=mes["subject"],
                               detail={"mes_action": mes.get("action"),
                                       "effects": mes.get("effects", []),
                                       "policy_checks": mes.get("policy_checks", [])})
            if RISK_ORDER.index(candidate.risk) >= RISK_ORDER.index(worst.risk):
                worst = candidate
            continue
        verb = _aas_verb(segment)
        if verb is None:
            # An `aas` line whose verb is not in the risk table is a read --
            # `get`, `find`, `validate`, `sm tree`. Classifying it as a generic
            # shell command would make `strict` gate reading a temperature,
            # which is how a gate gets switched off.
            risk = READ if _is_aas_line(segment) else SHELL
            candidate = Action(risk=risk, tool=tool, command=segment,
                               subject=_subject_of_segment(segment) if risk == READ else None)
            if risk == SHELL and delegation_targets:
                reached = _reaches_delegated_service(segment, delegation_targets)
                if reached is not None:
                    url, moves = reached
                    candidate.risk = MOVING if moves else INVOKE
                    candidate.moves_machine = moves
                    candidate.subject = url
                    candidate.detail["reason"] = (
                        "this command addresses a service the AAS delegates an "
                        "Operation to, going around `aas op invoke`"
                    )
        else:
            candidate = Action(risk=VERB_RISK[verb], tool=tool, command=segment,
                               subject=_subject_of_segment(segment))
            if candidate.risk == INVOKE and lookup_moves_machine is not None:
                target = _invoke_target(segment)
                if target is None:
                    # Cannot tell what it would call: treat as dangerous.
                    candidate.moves_machine = True
                    candidate.risk = MOVING
                    candidate.detail["reason"] = "the invocation target could not be parsed"
                else:
                    candidate.moves_machine = lookup_moves_machine(*target)
            if candidate.moves_machine:
                candidate.risk = MOVING
        if RISK_ORDER.index(candidate.risk) >= RISK_ORDER.index(worst.risk):
            worst = candidate
    worst.command = command
    return worst


def _reaches_delegated_service(command: str, targets: Dict[str, bool]) -> Optional[tuple]:
    """Whether a shell command addresses a service an Operation delegates to.

    Matched longest-URL-first, so `/setpoint/clear` is recognised before
    `/setpoint`. If only the host and port are recognised and not the path, the
    answer is the most dangerous Operation on that host: the command is talking
    to a machine's controller and the gate cannot tell which endpoint.
    """
    from urllib.parse import urlsplit

    for url in sorted(targets, key=len, reverse=True):
        if url in command:
            return url, targets[url]

    hosts: Dict[str, bool] = {}
    for url, moves in targets.items():
        netloc = urlsplit(url).netloc
        if netloc:
            hosts[netloc] = hosts.get(netloc, False) or moves
    for netloc, moves in hosts.items():
        if netloc in command:
            return netloc, moves
    return None


# The qualifier that tells the approval gate an invocation needs a person.
#
# Named for what a reader has to decide, not for what the harness infers. The
# earlier name stated a *fact* about the operation and left the reader to work
# out the consequence, which requires knowing this harness's risk levels.
# `requiresSafetyApproval = true` is a sentence: someone opening the AAS in
# Package Explorer knows what it means without knowing anything about `a4s`,
# and that reader -- a process engineer, not this program -- is the one who has
# to get it right.
#
# The cost of the name is honest: it encodes a policy conclusion in a model
# that should hold facts, so a deployment whose policy differs cannot reuse the
# judgement. A real objection, and it lost to the reader.
#
# No alias for the old spelling. Nothing on disk carries it -- the only AAS that
# ever did was written into an ephemeral database by `onboard` and does not
# survive a restart -- and a compatibility branch for a case that cannot occur
# is a second thing to keep right.
SAFETY_APPROVAL = "requiresSafetyApproval"


# Verbs that make an operation an actuation rather than a reading. Used only
# when nothing declares one way or the other -- see `moves_machine`.
_ACTUATING_VERBS = (
    "start", "stop", "halt", "run", "execute", "move", "home", "jog",
    "open", "close", "eject", "inject", "clamp", "unclamp", "reset",
    "set", "write", "enable", "disable", "activate", "deactivate",
    "calibrate", "purge", "load", "unload", "abort", "trigger",
)

# Verbs that are unambiguously reads. Checked first, because `GetMachineStatus`
# contains no actuating verb but `SetpointClear` and `GetSetpoint` both contain
# "set" as a substring -- which is why the test is on whole words.
_READING_VERBS = (
    "get", "read", "status", "list", "show", "query", "predict", "report",
    "model", "info", "describe", "capabilities", "openapi",
)


def _words(text: str) -> set:
    import re

    # `SetBarrelTemp` -> {set, barrel, temp}. CamelCase and separators both.
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    return {w for w in re.split(r"[^A-Za-z0-9]+", parts.lower()) if w}


def moves_machine(element: Dict[str, Any]) -> bool:
    """Whether this Operation is one an approval gate should stop at.

    **A declared value always wins.** `service.py` publishes the flag on every
    endpoint precisely so this is not a guess, and an AAS this harness onboarded
    carries it as a qualifier.

    **An absent value used to mean "no", and that was the wrong direction.**
    Measured on a live run: `InjectionMoldingMachine_285`, instantiated from a
    blueprint authored elsewhere, carries eight delegated operations --
    `StartCycle`, `SetBarrelTemp`, `SetInjectionSpeed` among them -- and not one
    of them has the qualifier, because whoever wrote the blueprint had never
    heard of it. Every one was classified as not moving a machine, and the gate
    let `StartCycle` through unattended on a policy whose entire purpose is to
    stop at exactly that.

    The absent case is the *common* case: it is every AAS this harness did not
    author, which is every AAS in a real plant. Defaulting it to "safe" makes
    the gate a guard on the one deployment that needed it least.

    So an operation with no declaration is judged by its name, and the fallback
    leans towards stopping. Guessing from names is a poor way to decide this --
    the docstring in `service.py` says so and is right -- but the alternative on
    an undeclared operation is not "decide correctly", it is "decide no". A
    false stop costs an operator one approval; a false pass moves a machine.
    """
    for qual in element.get("qualifiers") or []:
        if qual.get("type") == SAFETY_APPROVAL:
            return str(qual.get("value", "")).strip().lower() in ("true", "1", "yes")

    words = _words(element.get("idShort"))
    if words & set(_READING_VERBS):
        return False
    if words & set(_ACTUATING_VERBS):
        return True
    # Named neither way. An Operation exists to *do* something, so the unknown
    # case stops rather than passes.
    return True


def delegation_index(endpoint: str) -> Dict[str, bool]:
    """Every delegation URL in the repository, and whether it moves a machine.

    Scanned once when the gate is built. A few hundred milliseconds against a
    repository of this size, and the alternative is a gate that only guards the
    front door.
    """
    from ..aasctl.client import BasyxClient, Endpoint
    targets: Dict[str, bool] = {}

    def walk(node: Dict[str, Any]) -> None:
        for qual in node.get("qualifiers") or []:
            if qual.get("type") == "invocationDelegation" and qual.get("value"):
                url = str(qual["value"])
                targets[url] = targets.get(url, False) or moves_machine(node)
        for key in ("submodelElements", "value", "statements"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        walk(child)

    try:
        with BasyxClient(Endpoint.from_env(endpoint)) as client:
            for submodel in client.list_submodels(limit=400):
                walk(submodel)
    except Exception:  # noqa: BLE001 - an unreachable repository yields no index
        return {}
    return targets


def _baseline_tool_risk(tool: str) -> str:
    if tool in ("aas_operation_delegation",):
        return INVOKE
    if tool in ("aas_write_property", "aas_create_shell", "aas_create_submodel",
                "aas_create_element", "aas_from_smt", "aas_populate_inputs"):
        return WRITE
    return READ


# A target still carrying shell expansion cannot be resolved before the shell
# runs, so the gate cannot know what would be invoked.
_UNRESOLVED = ("$", "`", "*", "?", "$(")


def _invoke_target(command: str) -> Optional[tuple]:
    """`(submodel_id, path)` for an `aas op invoke` line, or None if unknowable.

    `aas op invoke $SM $OP` parses perfectly well and names nothing: the shell
    has not expanded the variables yet. Returning the literal `$SM` would send
    the gate to look up an element that does not exist, and it would then be
    relying on the lookup failing closed to stay safe. That is safety by
    accident. Unresolvable is reported as unresolvable.
    """
    import contextlib
    import io

    from ..aasctl.__main__ import build_parser

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if "aas" not in tokens:
        return None
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            args = build_parser().parse_args(tokens[tokens.index("aas") + 1 :])
        except SystemExit:
            return None
    submodel = getattr(args, "submodel_id", None)
    path = getattr(args, "path", None)
    if not submodel or not path:
        return None
    if any(marker in submodel or marker in path for marker in _UNRESOLVED):
        return None
    return (submodel, path)


def _subject_of_segment(command: str) -> Optional[str]:
    from ..tools import _aas_subject

    return _aas_subject(command)


def _first_str(arguments: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _short(arguments: Dict[str, Any]) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in list(arguments.items())[:3])
    return rendered[:160]


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    level: str = DEFAULT_LEVEL
    allow: List[str] = field(default_factory=list)   # subject globs, pre-authorised
    deny: List[str] = field(default_factory=list)    # subject globs, always refused

    @classmethod
    def from_env(cls) -> "Policy":
        level = (os.getenv("A4S_APPROVAL") or DEFAULT_LEVEL).strip().lower()
        if level not in LEVELS:
            level = DEFAULT_LEVEL
        return cls(
            level=level,
            allow=_globs(os.getenv("A4S_APPROVAL_ALLOW")),
            deny=_globs(os.getenv("A4S_APPROVAL_DENY")),
        )

    def decide(self, action: Action) -> str:
        """`allow` | `ask` | `deny`, before any operator is consulted."""
        subject = action.subject or ""
        # Deny beats allow: a kill-list is not a suggestion.
        if subject and any(fnmatch.fnmatch(subject, pattern) for pattern in self.deny):
            return "deny"
        base = LEVELS[self.level][action.risk]
        if base == "allow":
            return "allow"
        if subject and any(fnmatch.fnmatch(subject, pattern) for pattern in self.allow):
            return "allow"
        return base


# ---------------------------------------------------------------------------
# identity scoping
# ---------------------------------------------------------------------------

# Where the scope file lives. Beside the keys rather than in the repository:
# it says what a *key* may do, so it belongs with the keys and travels with
# the host, not with the plant model.
DEFAULT_SCOPES = Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser() / "scopes.json"


@dataclass
class Scopes:
    """Which actor may write which shells.

    The gate decides whether an action happens. This decides **under whose
    authority** it happens, which is a different question and the one the
    ledger could already answer after the fact and not before. Every entry
    signs its own log rows, so "who changed this property" was answerable;
    nothing stopped an asset agent for one machine writing to another's shell.

    Written as actor -> subject globs, and matched against the same subject
    the ledger indexes on, so a scope and a provenance query cannot disagree
    about what an action touched::

        {
          "urn:a4s:actor:imm-agent": ["https://fraunhoferIPA/ids/sm/imm_sim/*"],
          "orchestrator": ["*"]
        }

    Actors may be named either by the `--actor` label or by the Ed25519 public
    key the ledger records. The label is what an operator writes and the key is
    what actually signed, so both are accepted and the key wins where both are
    present -- a label can be passed on a command line by anyone.

    **Reads are never scoped.** Scoping a read would stop an agent finding out
    that a shell exists, which is not a security boundary -- the repository
    serves `/shells` to anyone who can reach it -- and would break discovery
    for no gain.
    """

    entries: Dict[str, List[str]] = field(default_factory=dict)
    # False when no scope file was found. Distinguished from an empty file,
    # which is a deliberate "nobody may write anything" and must be obeyed.
    configured: bool = False
    source: Optional[str] = None

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Scopes":
        """Read the scope file, or return an unconfigured instance.

        **The presence of the file is what turns scoping on.** An installation
        that has never heard of it keeps working; one that has written a file
        gets an actor not named in it refused. The alternative -- default-deny
        with no file -- would lock every existing deployment out of its own
        repository on upgrade, which is how a safety feature gets reverted.
        """
        target = Path(path) if path else DEFAULT_SCOPES
        if not target.is_file():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Fails closed on a *broken* file, unlike a missing one. A file
            # that exists says someone intended a restriction, and guessing
            # what it was is worse than refusing.
            raise ValueError(f"cannot read scopes at {target}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{target} must be an object of actor -> patterns")
        entries = {
            str(actor): [str(p) for p in (patterns if isinstance(patterns, list) else [patterns])]
            for actor, patterns in raw.items()
        }
        return cls(entries=entries, configured=True, source=str(target))

    def patterns_for(self, *names: Optional[str]) -> Optional[List[str]]:
        """The patterns for the first name present, or None if none is listed."""
        for name in names:
            if name and name in self.entries:
                return self.entries[name]
        return None

    def permits(self, action: Action, *names: Optional[str]) -> tuple[bool, str]:
        """Whether this actor may perform `action`. Returns (allowed, reason)."""
        if not self.configured:
            return True, "no scope file; identity scoping is not configured"
        if action.risk == READ:
            return True, "reads are not scoped"

        patterns = self.patterns_for(*names)
        if patterns is None:
            who = next((n for n in names if n), "(unnamed)")
            return False, f"actor {who} is not named in {self.source}"

        subject = action.subject or ""
        if not subject:
            # A shell command that touches no AAS path cannot be checked
            # against a shell pattern. Refusing is the only honest answer: a
            # scope that silently passes whatever it cannot classify is not a
            # scope, and this is the same reasoning `_invoke_target` uses.
            return False, "the action names no AAS subject, so no scope can authorise it"
        for pattern in patterns:
            if fnmatch.fnmatch(subject, pattern):
                return True, f"{subject} matches {pattern}"
        return False, f"{subject} is outside this actor's scope ({', '.join(patterns)})"


def _globs(raw: Optional[str]) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

class Gate:
    """Consults the policy, asks the operator when it must, records everything."""

    def __init__(
        self,
        policy: Optional[Policy] = None,
        ledger: Any = None,
        approver: Optional[Approver] = None,
        lookup_moves_machine: Optional[Callable[[str, str], bool]] = None,
        delegation_targets: Optional[Dict[str, bool]] = None,
        scopes: Optional["Scopes"] = None,
        actor: Optional[str] = None,
    ) -> None:
        self.policy = policy or Policy.from_env()
        self.ledger = ledger
        self.approver = approver
        self.lookup_moves_machine = lookup_moves_machine
        self.delegation_targets = delegation_targets or {}
        self.scopes = scopes if scopes is not None else Scopes()
        # Both the label and the signing key, because a scope file may name
        # either and only one of them is hard to forge.
        self.actor = actor
        self.actor_key = getattr(ledger, "actor", None) if ledger is not None else None
        # Set when nobody could be asked. The loop reads it and halts.
        self.pending: Optional[Request] = None
        # Commands a person has already approved, consumed once each. A resume
        # carries the operator's decision about *one* action; leaving it
        # standing would silently widen the approval to every later attempt.
        self._one_shot: List[str] = []

    def preapprove(self, command: str) -> None:
        """Let exactly one occurrence of this command through."""
        self._one_shot.append(" ".join(command.split()))

    def check(self, tool: str, arguments: Dict[str, Any], *, execution=None) -> Decision:
        action = classify(
            tool,
            arguments,
            lookup_moves_machine=self.lookup_moves_machine,
            delegation_targets=self.delegation_targets,
        )
        if execution:
            action.detail["execution"] = execution
        # One MES operation per shell invocation: otherwise the highest-risk
        # action could hide another order outside this actor's scope.
        if tool == "shell":
            from ..mesctl.commands import command_info
            segments = _segments(action.command)
            if len(segments) > 1 and any(command_info(s) for s in segments):
                decision = Decision(False, "run each MES command in a separate shell call", "command-shape")
                self._record(action, decision)
                return decision
        # Scope first, and it is not appealable. An operator can approve an
        # action the policy would have asked about; nobody at this terminal can
        # grant an actor authority over a shell it was not given. Asking would
        # imply otherwise, and a prompt that cannot change the outcome is a
        # prompt that teaches people to click through prompts.
        permitted, why = self.scopes.permits(action, self.actor_key, self.actor)
        if not permitted:
            decision = Decision(False, f"outside this identity's scope: {why}", "scope")
            self._record(action, decision)
            return decision

        verdict = self.policy.decide(action)

        normalised = " ".join(action.command.split())
        if verdict == "ask" and normalised in self._one_shot:
            self._one_shot.remove(normalised)
            decision = Decision(True, "approved by the operator on resume", "operator")
            self._record(action, decision)
            return decision

        if verdict == "allow":
            decision = Decision(True, f"{action.risk} permitted at level {self.policy.level}",
                                "policy")
        elif verdict == "deny":
            decision = Decision(False, f"{action.subject} matches the deny list", "denylist")
        else:
            request = Request(
                action=action,
                question=f"The agent wants to {action.summary()}. Allow it?",
                options=[APPROVE, DENY],
            )
            if self.approver is None:
                self.pending = request
                decision = Decision(
                    False,
                    "this action needs an operator's approval and none is available",
                    "policy",
                )
            else:
                decision = self.approver(request)
        self._record(action, decision)
        return decision

    def _record(self, action: Action, decision: Decision) -> None:
        if self.ledger is None:
            return
        from ..ledger import Kind

        self.ledger.append(
            Kind.APPROVAL,
            {
                "risk": action.risk,
                "moves_machine": action.moves_machine,
                "command": action.command[:500],
                "allowed": decision.allowed,
                "source": decision.source,
                "reason": decision.reason,
                "note": decision.note,
                "level": self.policy.level,
                "detail": action.detail,
            },
            subject=action.subject,
        )

    def refusal(self, decision: Decision) -> str:
        """What the model is told when the gate stops it."""
        if decision.source == "command-shape":
            return f"refused: {decision.reason}. Each operation needs its own scope check and approval."
        if decision.source == "scope":
            # Said differently from an approval refusal on purpose. The agent's
            # correct response to "needs a person" is to ask one; its correct
            # response to this is to stop trying, because no answer at this
            # terminal changes it. Suggesting `ask_user` here would send it to
            # collect an approval nobody present can give.
            return (
                f"refused: {decision.reason}.\n"
                "This is an authority boundary, not an approval step. Asking the "
                "operator will not change it. Work within the shells this identity "
                "may write, and report what you could not do and why."
            )
        if decision.note:
            return (
                f"refused by the operator: {decision.note}\n"
                "Do not retry the same command. Either follow the instruction above "
                "or explain why you cannot."
            )
        return (
            f"refused: {decision.reason}.\n"
            "This action needs a person to approve it. Use `ask_user` to explain what "
            "you need and why, or continue with the parts of the task that do not "
            "require it."
        )


# ---------------------------------------------------------------------------
# asking a person
# ---------------------------------------------------------------------------

def terminal_approver(stream: Any = None, output: Any = None) -> Approver:
    """Ask on a terminal: numbered options, or type an instruction.

    Free text is not a third option, it is the point. An operator faced with
    "set BarrelZone3 to 255" often wants neither yes nor no but "yes, but 240" --
    and the agent can act on that sentence, where a bare refusal only tells it
    to stop.
    """
    import sys

    reader = stream or sys.stdin
    writer = output or sys.stderr

    def ask(request: Request) -> Decision:
        print("\n" + "=" * 72, file=writer)
        print(request.render(), file=writer)
        print("=" * 72, file=writer)
        writer.flush()
        try:
            answer = (reader.readline() or "").strip()
        except (EOFError, KeyboardInterrupt):
            return Decision(False, "no answer from the operator", "operator")

        if answer in ("1", "y", "yes"):
            return Decision(True, "approved by the operator", "operator")
        if answer in ("2", "n", "no", ""):
            return Decision(False, "refused by the operator", "operator")
        # Anything else is an instruction: refuse the action as asked, and pass
        # the operator's words back so the agent can act on them.
        return Decision(False, "the operator sent an instruction instead", "operator",
                        note=answer)

    return ask


def repository_lookup(endpoint: str) -> Callable[[str, str], bool]:
    """Ask the AAS whether an Operation is declared to move equipment.

    Fails closed. If the repository cannot be reached, or the element cannot be
    read, the answer is "yes, it moves the machine" -- a gate that opens when
    it cannot see is not a gate. The cost of being wrong that way is a needless
    question; the cost of being wrong the other way is a machine that moved.
    """
    from ..aasctl.client import BasyxClient, Endpoint
    from ..aasctl.paths import ElementPath

    cache: Dict[tuple, bool] = {}

    def lookup(submodel_id: str, path: str) -> bool:
        key = (submodel_id, path)
        if key in cache:
            return cache[key]
        try:
            with BasyxClient(Endpoint.from_env(endpoint)) as client:
                element = client.get_element(submodel_id, ElementPath.parse(path))
            answer = moves_machine(element)
        except Exception:  # noqa: BLE001 - unknown means dangerous
            answer = True
        cache[key] = answer
        return answer

    return lookup


def auto_approver(note: str = "pre-authorised for an unattended run") -> Approver:
    """Approve everything the policy would ask about.

    For a batch run whose whole point is to exercise the gated path. It is a
    separate function rather than a flag on Policy so it cannot be reached by
    setting an environment variable by accident, and every approval it grants
    still lands in the ledger with this note attached.
    """

    def ask(_: Request) -> Decision:
        return Decision(True, note, "preauthorised")

    return ask

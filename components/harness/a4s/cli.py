"""``a4s`` -- the harness entry point.

Three commands: run a task, inspect the ledger, export trajectories. The
session is the unit: one instruction, one ledger stream, one identity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .environments.local import LocalEnvironment
from .kernel.approval import (
    Gate,
    Policy,
    Scopes,
    auto_approver,
    delegation_index,
    repository_lookup,
    terminal_approver,
)
from .kernel.config import describe, load_env, redact
from .kernel.loop import AgentLoop, LoopConfig
from .kernel.prompt import build_system_prompt
from .kernel.providers import ENV_KEYS, ModelConfig, build, resolve_provider
from .ledger import Identity, Ledger
from .tools import ToolRegistry

DEFAULT_HOME = Path(os.getenv("A4S_HOME", Path.home() / ".a4s")).expanduser()
SKILLS_ROOT = Path(__file__).parent / "skills"



# How many extraction warnings to print. The rest stay in the output file:
# a terminal that scrolls a hundred lines has told the reader nothing.
MAX_LISTED_WARNINGS = 12

def cmd_run(args: argparse.Namespace) -> int:
    session_id = args.session or f"s-{uuid.uuid4().hex[:12]}"
    home = DEFAULT_HOME
    workdir = Path(args.workdir or Path.cwd())

    endpoint = args.endpoint or os.getenv("A4S_AAS_ENDPOINT")
    if not endpoint:
        print(
            "error: no AAS endpoint. Pass --endpoint or set A4S_AAS_ENDPOINT.",
            file=sys.stderr,
        )
        return 2

    ledger = Ledger(home / "ledger.db", session_id, actor=args.actor)
    try:
        ledger.use_identity(Identity.load_or_create(home / "keys" / f"{args.actor}.key"))
    except ImportError:
        # Without `cryptography` the chain still gives tamper evidence; only
        # per-actor attribution is lost. Not worth refusing to run.
        print("note: cryptography not installed -- ledger entries are hashed but unsigned.", file=sys.stderr)

    policy = Policy.from_env()
    if args.approval:
        policy.level = args.approval
    approver = None
    if policy.level != "off":
        # An unattended run has nobody to ask, so it halts instead -- unless it
        # was launched with --yes, which pre-authorises and says so in every
        # ledger entry it writes.
        approver = auto_approver() if args.yes else (
            terminal_approver() if sys.stdin.isatty() else None
        )
    try:
        scopes = Scopes.load(args.scopes)
    except ValueError as exc:
        # A broken scope file fails closed. A missing one does not: see
        # `Scopes.load`.
        print(f"error: {exc}", file=sys.stderr)
        ledger.close()
        return 2
    if scopes.configured:
        print(f"scopes:   {scopes.source} ({len(scopes.entries)} identities)",
              file=sys.stderr)

    gate = Gate(
        policy=policy,
        ledger=ledger,
        approver=approver,
        lookup_moves_machine=repository_lookup(endpoint),
        # Scanned once: without it the gate guards `aas op invoke` and lets a
        # `curl` at the same service straight through.
        delegation_targets=delegation_index(endpoint) if policy.level != "off" else {},
        scopes=scopes,
        actor=args.actor,
    )

    environment = LocalEnvironment(cwd=str(workdir), env={"A4S_AAS_ENDPOINT": endpoint})
    tools = ToolRegistry(environment, ledger, workdir, gate=gate)

    model = ModelConfig.from_env(args.provider)
    if args.model:
        model.model = args.model
    # Which model answered is part of the record, so it is stated up front
    # rather than inferred afterwards from the ledger.
    how = "pre-authorised" if args.yes else ("interactive" if approver else "halts to ask")
    print(
        f"provider: {model.provider}  model: {model.model}",
        file=sys.stderr,
    )
    # The full path, not just "bash". Two different `bash.exe` on a Windows
    # PATH is the ordinary case, and only one of them is a shell on this
    # machine -- printing the name alone is what made that invisible.
    print(f"shell:    {environment.shell[0]}", file=sys.stderr)
    if environment.shell_name.lower() in ("cmd", "powershell", "pwsh"):
        print(
            "warning:  no POSIX shell was found, so commands run in "
            f"{environment.shell_name}. The skills use pipelines, $VAR and "
            "heredocs; cmd.exe exits 0 with empty output rather than refusing "
            "them, so this run will fail silently. Install Git for Windows, or "
            "set $A4S_SHELL.",
            file=sys.stderr,
        )
    print(f"approval: {policy.level} ({how})", file=sys.stderr)
    system_prompt = build_system_prompt(
        SKILLS_ROOT,
        endpoint=endpoint,
        instruction=args.instruction,
        shell=environment.shell_name,
    )

    loop = AgentLoop(
        provider=build(model),
        model=model,
        tools=tools,
        ledger=ledger,
        system_prompt=system_prompt,
        config=LoopConfig(max_steps=args.max_steps),
    )

    result = loop.run(args.instruction)
    ledger.close()

    print(f"\n[{result.status}] after {result.steps} steps "
          f"({result.input_tokens} in / {result.output_tokens} out tokens)")
    print(result.summary)
    print(f"session: {session_id}")
    return 0 if result.status in ("finished", "awaiting_operator") else 1


def cmd_ledger(args: argparse.Namespace) -> int:
    with Ledger(DEFAULT_HOME / "ledger.db", args.session or "-") as ledger:
        if args.verify:
            ok, bad = ledger.verify()
            print("ledger intact" if ok else f"CHAIN BROKEN at seq {bad}")
            return 0 if ok else 1
        if args.subject:
            events = ledger.provenance(args.subject)
        else:
            events = ledger.read(session_id=args.session, kind=args.kind, limit=args.limit)
        for event in events:
            print(
                f"{event.seq:>5}  {event.kind:<16} {event.actor[:12]:<12} "
                f"{(event.subject or '-')[:48]:<48} "
                f"{json.dumps(event.payload, ensure_ascii=False)[:120]}"
            )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    with Ledger(DEFAULT_HOME / "ledger.db", args.session) as ledger:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with target.open("w", encoding="utf-8") as handle:
            for record in ledger.export_trajectory(args.session):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} records to {target}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Continue a session that stopped for an operator's decision.

    The conversation is rebuilt from the ledger. There is no checkpoint file:
    a second copy of the same conversation is a second copy that can disagree
    with the audit record.
    """
    endpoint = args.endpoint or os.getenv("A4S_AAS_ENDPOINT")
    if not endpoint:
        print("error: no AAS endpoint. Pass --endpoint or set A4S_AAS_ENDPOINT.", file=sys.stderr)
        return 2

    home = DEFAULT_HOME
    ledger = Ledger(home / "ledger.db", args.session, actor=args.actor)
    try:
        ledger.use_identity(Identity.load_or_create(home / "keys" / f"{args.actor}.key"))
    except ImportError:
        pass

    stopped = ledger.last_refusal(args.session)
    if stopped is None:
        print(f"error: session {args.session!r} has no refused action to resume from.",
              file=sys.stderr)
        ledger.close()
        return 2
    blocked_command = stopped.payload.get("command", "")
    print(f"blocked: {blocked_command[:120]}", file=sys.stderr)

    if args.approve:
        # Blunt on purpose. The agent asks for confirmation on its own, in
        # `ask_user`, which is a separate layer from the gate -- so a resume
        # that only says "approved" gets answered with a third request to
        # confirm. The note has to cancel the agent's own caution as well as
        # report the gate's decision.
        note = (
            "[operator] APPROVED. I have already reviewed and authorised this "
            f"exact action: {blocked_command} -- run that command now, "
            "unchanged. Do not ask me again and do not call ask_user about it: "
            "the approval you were waiting for is this message. After it runs, "
            "report the result and finish the task."
        )
    elif args.note:
        note = f"[operator] {args.note}"
    else:
        note = (
            "[operator] I am not approving that action. Continue with the parts of "
            "the task that do not need it, and report what you could not do."
        )

    policy = Policy.from_env()
    if args.approval:
        policy.level = args.approval
    gate = Gate(
        policy=policy,
        ledger=ledger,
        approver=terminal_approver() if sys.stdin.isatty() else None,
        lookup_moves_machine=repository_lookup(endpoint),
        delegation_targets=delegation_index(endpoint) if policy.level != "off" else {},
        scopes=Scopes.load(args.scopes),
        actor=args.actor,
    )
    if args.approve and blocked_command:
        # Exactly one occurrence. The operator approved an action, not a habit.
        gate.preapprove(blocked_command)

    workdir = Path(args.workdir or Path.cwd())
    environment = LocalEnvironment(cwd=str(workdir), env={"A4S_AAS_ENDPOINT": endpoint})
    tools = ToolRegistry(environment, ledger, workdir, gate=gate)

    model = ModelConfig.from_env(args.provider)
    if args.model:
        model.model = args.model
    # The original instruction, so resume gets the same skill the first run
    # had. Rebuilt rather than replayed, like the rest of the prompt.
    system_prompt = build_system_prompt(
        SKILLS_ROOT,
        endpoint=endpoint,
        instruction=ledger.first_instruction(args.session),
        shell=environment.shell_name,
    )

    try:
        history = ledger.replay(args.session, system_prompt)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        ledger.close()
        return 1

    turns = len([m for m in history if m.get("role") != "system"])
    print(f"resuming {args.session}: {turns} turns restored", file=sys.stderr)
    print(f"operator: {note.splitlines()[0][:100]}", file=sys.stderr)

    loop = AgentLoop(
        provider=build(model), model=model, tools=tools, ledger=ledger,
        system_prompt=system_prompt, config=LoopConfig(max_steps=args.max_steps),
    )
    result = loop.resume(history, note)
    ledger.close()

    print()
    print(f"[{result.status}] after {result.steps} further steps "
          f"({result.input_tokens} in / {result.output_tokens} out tokens)")
    print(result.summary)
    print(f"session: {args.session}")
    return 0 if result.status in ("finished", "awaiting_operator") else 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Turn documents into the entity records `aas match` consumes."""
    from . import extract as ex

    sources = [
        ("pdf", args.pdf), ("image", args.image), ("text", args.text),
        ("table", args.table), ("url", args.url),
    ]
    chosen = [(kind, value) for kind, value in sources if value]
    if len(chosen) != 1:
        print(
            "error: name exactly one source: --pdf, --image, --text, --table or --url",
            file=sys.stderr,
        )
        return 2
    kind, value = chosen[0]

    model = ModelConfig.from_env(args.provider)
    if args.model:
        model.model = args.model
    provider = build(model)

    if args.as_template:
        return _extract_template(args, kind, value, provider, model)

    batches: List[List[ex.Extracted]] = []
    # What the pages said about themselves, kept beside what they said about
    # the asset: which configurations the document distinguishes, and what a
    # person should check before any of it is written into a machine's model.
    found_configurations: List[str] = []
    found_warnings: List[str] = []

    def absorb(part: "ex.Extraction") -> List[ex.Extracted]:
        for name in part.configurations:
            if name not in found_configurations:
                found_configurations.append(name)
        found_warnings.extend(part.warnings)
        return part.entities
    try:
        if kind in ("pdf", "image"):
            for path in [Path(p) for p in value]:
                if not path.exists():
                    print(f"error: no file at {path}", file=sys.stderr)
                    return 2
                if kind == "pdf":
                    pages = ex.render_pdf(path, dpi=args.dpi)
                    # Pages do not depend on each other, so the wall clock was
                    # the sum of six model calls for no reason. Concurrent, it
                    # is the slowest page. Bounded rather than unbounded: the
                    # endpoint rate-limits, and a hundred-page manual would
                    # otherwise open a hundred connections at once.
                    #
                    # One page that cannot be answered must not cost the other
                    # five, and must not be silently indistinguishable from a
                    # page with nothing on it -- which is how a 6-page datasheet
                    # came back with 3 entities and no error.
                    from concurrent.futures import ThreadPoolExecutor

                    def one_page(item: Any) -> Any:
                        number, png = item
                        try:
                            reply = ex.ask_image(provider, model, ex.PDF_PROMPT, png)
                            return number, ex.parse_extraction(
                                reply, f"{path.name} p.{number}"
                            ), None
                        except ex.ExtractError as exc:
                            return number, None, str(exc)

                    numbered = list(enumerate(pages, start=1))
                    workers = min(len(numbered), int(os.getenv("A4S_EXTRACT_WORKERS", "6")))
                    print(f"  {path.name}: {len(numbered)} pages, {workers} at a time",
                          file=sys.stderr)
                    failed: List[str] = []
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        # Ordered, not as-completed: page order is the reading
                        # order of the document, and `merge` keeps the first
                        # sighting of a repeated fact -- so an arbitrary arrival
                        # order would make which page gets cited depend on
                        # timing, and two runs would disagree.
                        for number, found, error in pool.map(one_page, numbered):
                            if error:
                                failed.append(f"p.{number}: {error}")
                                print(f"    page {number} failed: {error}",
                                      file=sys.stderr)
                            else:
                                print(f"    page {number}: {len(found.entities)}",
                                      file=sys.stderr)
                                batches.append(absorb(found))
                    if failed:
                        print("", file=sys.stderr)
                        print(f"{len(failed)} of {len(pages)} pages produced "
                              f"nothing:", file=sys.stderr)
                        for line in failed:
                            print(f"  {line}", file=sys.stderr)
                else:
                    prompt = ex.IMAGE_PROMPT + (
                        f"\n\nPay particular attention to: {args.intent}" if args.intent else ""
                    )
                    print(f"  {path.name}", file=sys.stderr)
                    reply = ex.ask_image(provider, model, prompt, ex.read_image(path))
                    batches.append(absorb(ex.parse_extraction(reply, path.name)))
        elif kind in ("text", "url"):
            if kind == "url":
                body, source = ex.fetch_url(value), value
            elif value == "-":
                body, source = sys.stdin.read(), "stdin"
            else:
                path = Path(value)
                if not path.exists():
                    print(f"error: no file at {path}", file=sys.stderr)
                    return 2
                body, source = path.read_text(encoding="utf-8"), path.name
            reply = ex.ask_text(provider, model, ex.TEXT_PROMPT, body)
            batches.append(absorb(ex.parse_extraction(reply, source)))
        else:
            path = Path(value)
            if not path.exists():
                print(f"error: no file at {path}", file=sys.stderr)
                return 2
            rows = ex.read_table(path)
            if not rows:
                print(f"error: {path.name} has no rows", file=sys.stderr)
                return 2
            if args.map:
                mapping = ex.parse_mapping(args.map)
            else:
                # Let the model read the headers rather than guessing at names.
                reply = ex.ask_text(
                    provider,
                    model,
                    ex.MAP_PROMPT,
                    json.dumps({"HEADERS": list(rows[0].keys()), "SAMPLE_ROW": rows[0]},
                               ensure_ascii=False, default=str),
                )
                mapping = ex.parse_json_object(reply)
                print(f"  column mapping: {mapping}", file=sys.stderr)
            batches.append(ex.rows_to_entities(rows, mapping, path.name))
    except ex.ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entities, merge_notes = ex.merge_reporting(batches)
    configurations = found_configurations or ex.configurations_of(entities)
    warnings = found_warnings + merge_notes

    # An object rather than a bare array. `aas match` and the wizard already
    # read both shapes, and the array had nowhere to carry the two things a
    # reader of these values needs: which configurations the document covers,
    # and what the extraction could not do cleanly.
    payload: Any = {
        "entities": [e.as_dict() for e in entities],
        "configurations": configurations,
        "warnings": warnings,
    }
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    without_unit = sum(1 for e in entities if not e.unit)
    without_value = sum(1 for e in entities if e.value is None)
    print(f"{len(entities)} entities from {kind} ({without_unit} without a unit, "
          f"{without_value} without a value)")
    if configurations:
        # Said loudly, because it is the difference between a datasheet and an
        # asset: several configurations means these values do not all describe
        # one machine, and matching them all into one shell writes a machine
        # that does not exist.
        print(f"\n! this document describes {len(configurations)} configurations: "
              f"{', '.join(configurations)}")
        print("  pick one before matching, or the values of several machines "
              "are written into one shell")
    if warnings:
        print(f"\n{len(warnings)} things to check:")
        for line in warnings[:MAX_LISTED_WARNINGS]:
            print(f"  - {line}")
        if len(warnings) > MAX_LISTED_WARNINGS:
            print(f"  ... and {len(warnings) - MAX_LISTED_WARNINGS} more, "
                  f"all of them in the output")
    for item in entities[:20]:
        unit = f" {item.unit}" if item.unit else ""
        print(f"  {item.entity[:38]:<38} = {str(item.value)[:20]:<20}{unit:<8} [{item.source}]")
    if len(entities) > 20:
        print(f"  ... and {len(entities) - 20} more")
    if args.out:
        print(f"\nwritten to {args.out}")
        print(f"next: aas match --entities {args.out} --into <submodel-id> --out plan.json")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _extract_template(
    args: argparse.Namespace, kind: str, value: Any, provider: Any, model: Any
) -> int:
    """Propose a submodel *template* rather than extract values from a document.

    A separate path rather than a flag threaded through the existing one,
    because almost nothing is shared: a different prompt, a different output
    shape, a different consumer. What they do share is the source handling, so
    that stays above.

    Nothing is written to the repository here. The proposal is a file, reviewed
    before `aas smt author` builds it -- the same propose/review/apply shape
    `aas match` and `aas populate` have, and for a stronger reason: every
    instance built from a wrong template inherits the mistake.
    """
    from . import extract as ex
    from . import structure as st

    try:
        if kind == "pdf":
            pages: List[bytes] = []
            for raw in value:
                path = Path(raw)
                if not path.exists():
                    print(f"error: no file at {path}", file=sys.stderr)
                    return 2
                pages.extend(ex.render_pdf(path, dpi=args.dpi))
            if not pages:
                print("error: the PDF rendered no pages", file=sys.stderr)
                return 2
            # One call over the first pages rather than one per page and a
            # merge. A template is a single coherent shape, and merging five
            # independently-proposed trees is a harder problem than proposing
            # one -- the page boundaries are not where the structure divides.
            print(f"  proposing a structure from {len(pages)} page(s)", file=sys.stderr)
            reply = ex.ask_image(provider, model, st.TEMPLATE_PROMPT, pages[0])
            source = f"{Path(value[0]).name}"
        elif kind == "image":
            reply = ex.ask_image(provider, model, st.TEMPLATE_PROMPT,
                                 ex.read_image(Path(value[0])))
            source = Path(value[0]).name
        elif kind == "url":
            reply = ex.ask_text(provider, model, st.TEMPLATE_PROMPT, ex.fetch_url(value))
            source = value
        elif kind == "text":
            body = sys.stdin.read() if value == "-" else Path(value).read_text(encoding="utf-8")
            reply = ex.ask_text(provider, model, st.TEMPLATE_PROMPT, body)
            source = "stdin" if value == "-" else Path(value).name
        else:
            rows = ex.read_table(Path(value))
            if not rows:
                print(f"error: {value} has no rows", file=sys.stderr)
                return 2
            reply = ex.ask_text(
                provider, model, st.TEMPLATE_PROMPT,
                json.dumps({"HEADERS": list(rows[0].keys()), "ROWS": rows[:20]},
                           ensure_ascii=False, default=str),
            )
            source = Path(value).name
    except ex.ExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        proposed = st.parse(reply, source)
    except st.StructureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for line in st.render(proposed):
        print(line)

    found = st.problems(proposed)
    if found:
        print()
        print(f"{len(found)} problem(s) -- fix these before building:")
        for problem in found:
            print(f"  {problem['rule']}  {problem['path']}: {problem['detail']}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(proposed.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print()
        print(f"written to {args.out}")
        print(f"next: aas smt author <template-id> --structure {args.out}")
    else:
        print()
        print(json.dumps(proposed.as_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run the tool-surface comparison and print the table."""
    from .experiment import ARMS, TASKS, render, run_suite, to_json

    endpoint = args.endpoint or os.getenv("A4S_AAS_ENDPOINT")
    if not endpoint:
        print("error: no AAS endpoint. Pass --endpoint or set A4S_AAS_ENDPOINT.", file=sys.stderr)
        return 2

    tasks = TASKS if not args.tasks else [t for t in TASKS if t.name in args.tasks]
    if not tasks:
        print(f"error: no task named {args.tasks}. Have: "
              f"{', '.join(t.name for t in TASKS)}", file=sys.stderr)
        return 2
    arms = args.arms or list(ARMS)

    workdir = Path(args.workdir or Path.cwd())
    total = len(tasks) * len(arms) * args.repeats
    print(f"{total} runs: {len(tasks)} tasks x {len(arms)} arms x {args.repeats} "
          f"({'with' if args.skills else 'without'} skill injection)", file=sys.stderr)

    done = [0]

    def progress(outcome) -> None:
        done[0] += 1
        mark = "pass" if outcome.passed else "FAIL"
        print(
            f"[{done[0]:>3}/{total}] {outcome.arm:<9} {outcome.task:<10} {mark}  "
            f"{outcome.steps:>2} steps  {outcome.input_tokens:>7} in  "
            f"{outcome.seconds:>5}s  {outcome.detail[:60]}",
            file=sys.stderr,
        )

    outcomes = run_suite(
        endpoint=endpoint, workdir=workdir, home=DEFAULT_HOME, arms=arms, tasks=tasks,
        repeats=args.repeats, model_name=args.model, provider_name=args.provider,
        skills=args.skills, on_result=progress,
    )

    print()
    print(render(outcomes))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(to_json(outcomes), indent=2), encoding="utf-8")
        print()
        print(f"written to {args.out}")
    return 0


def cmd_retrieval(args: argparse.Namespace) -> int:
    """Score `aas find` against a labelled set, tier by tier."""
    from .aasctl import index as index_mod
    from .experiment import retrieval as rt

    path = Path(args.labels) if args.labels else None
    if path is None:
        available = sorted(rt.LABELS_DIR.glob("*.json"))
        if len(available) != 1:
            print(
                "error: name a label file with --labels. Available: "
                + (", ".join(p.name for p in available) or "(none)"),
                file=sys.stderr,
            )
            return 2
        path = available[0]

    try:
        labels = rt.LabelSet.load(path)
    except rt.LabelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    home = Path(args.index_home) if args.index_home else None
    try:
        index = index_mod.Index.load(labels.submodel, home or index_mod.DEFAULT_INDEX_HOME)
    except index_mod.IndexError_ as exc:
        # The index is the thing under test; a missing one is a setup problem
        # and saying which command builds it is cheaper than a stack trace.
        print(f"error: {exc}", file=sys.stderr)
        print(f"hint: aas index {labels.submodel}", file=sys.stderr)
        return 2

    print(f"{len(labels.cases)} cases from {path.name}", file=sys.stderr)

    if args.ablate:
        from .experiment import ablation as ab

        if args.ablate == "dedupe":
            results = ab.dedupe_ablation(index, labels, top_k=args.top_k)
            title = "deduplication key: what counts as the same element"
        else:
            print("re-embedding once per variant; this costs tokens", file=sys.stderr)
            results = ab.descriptor_ablation(
                index, labels, top_k=args.top_k,
                on_variant=lambda name: print(f"  {name}", file=sys.stderr),
            )
            title = "descriptor: how much ancestry to embed"
        print()
        print(ab.render_ablation(results, title))
        return 0

    if args.compare:
        arms = rt.compare(index, labels, top_k=args.top_k)
        print()
        for name, outcomes in arms.items():
            print(f"=== {name} ===")
            print(rt.render(index, labels, outcomes))
            print()
        print(rt.render_delta(arms["unfiltered"], arms["filtered"]))
        outcomes = arms["filtered"]
    else:
        outcomes = rt.evaluate(index, labels, top_k=args.top_k, use_filters=args.filters)
        print()
        print(rt.render(index, labels, outcomes))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(rt.to_json(index, labels, outcomes), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print()
        print(f"written to {args.out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report what is configured and what is reachable, without printing secrets.

    Two providers are normally configured at once here -- Azure for the main
    runs, Anthropic for comparison -- and the usual failure is a variable that
    is set in one shell and not another. Presence and length are enough to spot
    a missing or truncated key; the value itself never needs to be shown.
    """
    import httpx

    env_file = load_env(args.env_file)
    print(f"env file:  {env_file or '(none found)'}")
    print(f"resolved:  A4S_PROVIDER -> {resolve_provider()}\n")

    ok = True
    for name in ("azure", "anthropic", "openai"):
        config = ModelConfig.from_env(name)
        gaps = config.missing()
        mark = "ok " if not gaps else "-- "
        print(f"{mark}{name}")
        print(f"     model/deployment  {config.model}")
        print(f"     endpoint          {config.base_url or '(default)'}")
        print(f"     api key           {redact(config.api_key)}")
        if name == "azure":
            print(f"     api version       {config.api_version}")
        for row in describe(*ENV_KEYS[name]["api_key"], *ENV_KEYS[name]["base_url"]):
            state = f"set ({row['length']} chars)" if row["set"] else "unset"
            print(f"       {row['name']:<28} {state}")
        if gaps:
            print(f"     missing: {', '.join(gaps)}")
        print()

    # The shell, because the failure it causes is silent from outside. The run
    # that found this printed `shell: bash`, passed every check on this page,
    # and then failed every command it issued: `shutil.which("bash")` had
    # returned Windows' WSL relay, which is not a shell on this machine.
    from .environments.local import check_shell

    shell = check_shell()
    if shell["works"] and shell["posix"]:
        print(f"ok shell: {shell['path']}")
    elif shell["works"]:
        print(f"-- shell: {shell['path']} runs, but is not a POSIX shell.")
        print("     The skills use pipelines, $VAR and heredocs. cmd.exe does not")
        print("     refuse those -- it exits 0 with empty output, so a run fails")
        print("     silently. Install Git for Windows, or set $A4S_SHELL.")
        ok = False
    else:
        print(f"-- shell: {shell['path']} does not run commands.")
        print("     Set $A4S_SHELL to a working POSIX shell.")
        ok = False
    if shell["rejected"]:
        print(f"     ignored (WSL relay, not a shell here): {', '.join(shell['rejected'])}")
    print()

    endpoint = args.endpoint or os.getenv("A4S_AAS_ENDPOINT")
    if not endpoint:
        print("-- aas repository: no A4S_AAS_ENDPOINT and no --endpoint")
        ok = False
    else:
        try:
            body = httpx.get(f"{endpoint.rstrip('/')}/shells", params={"limit": 1}, timeout=5.0)
            body.raise_for_status()
            count = len(body.json().get("result") or [])
            print(f"ok aas repository: {endpoint} reachable ({count}+ shells)")
        except Exception as exc:  # noqa: BLE001 - any failure is the same answer
            print(f"-- aas repository: {endpoint} unreachable: {exc}")
            ok = False

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a4s",
        description="An industrial agent that manages assets through their AAS.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run one task to completion")
    p.add_argument("instruction")
    p.add_argument("--endpoint", help="AAS repository base URL")
    p.add_argument(
        "--provider",
        choices=("azure", "anthropic", "openai"),
        help="override $A4S_PROVIDER (default: inferred from what is configured)",
    )
    p.add_argument("--model", help="override $A4S_MODEL (a deployment name on Azure)")
    p.add_argument("--session", help="reuse a session id")
    p.add_argument("--actor", default="orchestrator", help="signing identity for the ledger")
    p.add_argument("--scopes", metavar="FILE",
                   help="which identity may write which shells "
                        "(default: $A4S_HOME/scopes.json if it exists)")
    p.add_argument("--workdir", help="working directory for shell commands")
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument(
        "--approval",
        choices=("off", "machine", "operations", "writes", "strict"),
        help="what needs a person: nothing; machine operations and MES business writes "
             "(default); any operation plus MES writes; any write; or any command. "
             "Else $A4S_APPROVAL.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="pre-authorise what the policy would ask about. Every approval it "
             "grants is still recorded as pre-authorised.",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("ledger", help="inspect the event log")
    p.add_argument("--session")
    p.add_argument("--kind", help="filter by event kind, e.g. aas.write")
    p.add_argument("--subject", help="full provenance for one AAS path")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--verify", action="store_true", help="recompute the hash chain")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("resume", help="continue a session that stopped for an operator")
    p.add_argument("session")
    p.add_argument("--approve", action="store_true",
                   help="allow the action that was refused, once")
    p.add_argument("--note", help="answer in your own words instead; the agent reads it")
    p.add_argument("--endpoint", help="AAS repository base URL")
    p.add_argument("--workdir")
    p.add_argument("--actor", default="orchestrator")
    p.add_argument("--scopes", metavar="FILE")
    p.add_argument("--provider", choices=("azure", "anthropic", "openai"))
    p.add_argument("--model")
    p.add_argument("--approval", choices=("off", "machine", "operations", "writes", "strict"))
    p.add_argument("--max-steps", type=int, default=20)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("extract", help="turn documents into entities for `aas match`")
    p.add_argument("--pdf", nargs="+", metavar="FILE", help="datasheet PDFs, page by page")
    p.add_argument("--image", nargs="+", metavar="FILE", help="photographs or scans")
    p.add_argument("--text", metavar="FILE", help="a text file, or - for stdin")
    p.add_argument("--table", metavar="FILE", help=".csv, .json, .jsonl or .xlsx")
    p.add_argument("--url", metavar="URL", help="a product page (no JavaScript rendering)")
    p.add_argument("--out", metavar="FILE", help="write the entities here")
    p.add_argument("--intent", help="with --image: what to look for")
    p.add_argument(
        "--map",
        metavar="SPEC",
        help="with --table: entity=Col,value=Col,unit=Col (else the model reads the headers)",
    )
    p.add_argument("--as-template", action="store_true",
                   help="propose a submodel template structure instead of extracting "
                        "values; feed the result to `aas smt author`")
    p.add_argument("--dpi", type=int, default=200, help="with --pdf: render resolution")
    p.add_argument("--provider", choices=("azure", "anthropic", "openai"))
    p.add_argument("--model", help="override $A4S_MODEL")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("experiment", help="run the tool-surface comparison")
    p.add_argument("--arms", nargs="+", choices=("baseline", "cli"),
                   help="which arms to run. Default both.")
    p.add_argument("--tasks", nargs="+", help="task names. Default all.")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--endpoint", help="AAS repository base URL")
    p.add_argument("--workdir", help="where fixtures live and files are written")
    p.add_argument("--provider", choices=("azure", "anthropic", "openai"))
    p.add_argument("--model", help="override $A4S_MODEL")
    p.add_argument("--skills", action="store_true",
                   help="inject the matching skill into both arms (off by default: "
                        "the skills are written in `aas` commands, which one arm can "
                        "copy and the other must translate)")
    p.add_argument("--out", metavar="FILE", help="write the full result set as JSON")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser(
        "retrieval",
        help="score `aas find` against a labelled set (recall@k, margin, rejection)",
    )
    p.add_argument("--labels", metavar="FILE",
                   help="label file (default: the only one in a4s/experiment/labels)")
    p.add_argument("--filters", action="store_true",
                   help="apply each case's declared structural filter before ranking")
    p.add_argument("--ablate", choices=("descriptor", "dedupe"),
                   help="compare configurations instead of scoring one "
                        "(descriptor re-embeds per variant and costs tokens)")
    p.add_argument("--compare", action="store_true",
                   help="score the set twice, filters off and on, and report the difference")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--index-home", help="cache directory (else $A4S_INDEX_HOME)")
    p.add_argument("--out", metavar="FILE", help="write the full result set as JSON")
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser("doctor", help="report configured providers and repository reachability")
    p.add_argument("--endpoint", help="AAS repository base URL to probe")
    p.add_argument("--env-file", help="path to a .env (default: ./.env, then the package root)")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("export", help="export a session as SFT/RL trajectories")
    p.add_argument("session")
    p.add_argument("--out", default="trajectories.jsonl")
    p.set_defaults(func=cmd_export)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # The same protection `aas` has carried since a `sm tree` over German
    # descriptions died halfway through its output on a cp1252 console. `a4s`
    # went without it until a retrieval report containing Chinese queries hit
    # the same wall: the data is fine, the terminal is not, and a traceback is
    # a failed command where a replacement character is only a bad rendering.
    from .aasctl.__main__ import force_utf8

    force_utf8()
    load_env()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

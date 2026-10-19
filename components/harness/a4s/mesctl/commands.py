"""CLI grammar, discovery and classification from shared action definitions."""

import argparse
import contextlib
import io
import shlex

from .actions import ActionSpec
from .execution import identifier
from .domains import catalog, orders, scheduling, issues, maintenance


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mes",
        allow_abbrev=False,
        description="OpenMES via A4S_MES_ENDPOINT and A4S_MES_TOKEN. JSON output; no inline credentials.",
    )
    roots = parser.add_subparsers(dest="group", required=True)
    groups = {}

    actions = {}

    def cmd(name, path, description, method="GET", resource=None, target=False, **metadata):
        if name in actions:
            raise ValueError(f"Duplicate MES action: {name}")
        action = ActionSpec(name, path, description, method, resource, target, **metadata)
        actions[name] = action
        words = name.split()
        if len(words) == 1:
            p = roots.add_parser(name, help=description, allow_abbrev=False)
        else:
            group, verb = words
            if group not in groups:
                groups[group] = roots.add_parser(group).add_subparsers(dest="verb", required=True)
            p = groups[group].add_parser(verb, help=description, allow_abbrev=False)
        p.set_defaults(
            action=action, api_path=path, method=method, resource=resource, command_name=name
        )
        p.add_argument(
            "--trace-id",
            type=identifier,
            help="Correlation only; not an approval or idempotency key",
        )
        p.add_argument("--json", action="store_true", help="JSON output (also the default)")
        if target:
            p.add_argument(
                "target",
                help="Resource ID; order and line commands also accept exact order number/code",
            )
        if method != "GET":
            p.add_argument(
                "--dry-run",
                action="store_true",
                help="Resolve and validate, then print request without writing",
            )
        return p

    def opt(p, name, dest=None, **kw):
        p.add_argument("--" + name, dest=dest or name.replace("-", "_"), **kw)

    for domain in (catalog, orders, scheduling, issues, maintenance):
        domain.register(cmd, opt)
    p = cmd("describe", None, "Describe available actions and their execution contracts")
    p.add_argument("action_words", nargs="*", help="Optional domain or action, e.g. schedule move")
    parser.mes_actions = actions
    return parser


def subject(args):
    return args.action.subject(args)


def command_info(command):
    """Parse the actual CLI grammar for the gate and Ledger; never touch HTTP."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    tail = None
    for i, token in enumerate(tokens):
        name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name in ("mes", "mes.exe"):
            tail = tokens[i + 1 :]
            break
        if token == "a4s.mesctl" and i and tokens[i - 1] == "-m":
            tail = tokens[i + 1 :]
            break
    if tail is None:
        return None
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            args = build_parser().parse_args(tail)
        except SystemExit as exc:
            return {"mutates": exc.code != 0, "subject": "mes://service"}
    return {
        "mutates": args.action.mutates and not getattr(args, "dry_run", False),
        "subject": subject(args),
        "action": args.action.name,
        "effects": list(args.action.effects),
        "trace_id": args.trace_id,
        "policy_checks": list(args.action.policy_checks),
    }


def describe_actions(words=()):
    parser = build_parser()
    name = " ".join(words)
    actions = [
        a
        for a in parser.mes_actions.values()
        if a.path is not None and (not name or a.name == name or a.name.startswith(name + " "))
    ]
    if not actions:
        from .client import MesError

        raise MesError("No matching MES action; use mes describe to list available actions")

    # Argument shapes come from the exact same parser, including required flags.
    def leaves(p):
        children = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        if children:
            return [
                leaf for a in children for child in a.choices.values() for leaf in leaves(child)
            ]
        return [p]

    by_name = {p.get_default("command_name"): p for p in leaves(parser)}
    result = []
    for action in actions:
        p = by_name[action.name]
        result.append(
            {
                **action.describe(),
                "usage": p.format_usage().strip(),
                "arguments": [
                    {
                        "name": a.dest,
                        "flags": a.option_strings,
                        "required": a.required,
                        "choices": list(a.choices) if a.choices is not None else None,
                        "type": "boolean"
                        if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction))
                        else getattr(a.type, "__name__", "string"),
                    }
                    for a in p._actions
                    if a.dest != "help"
                ],
            }
        )
    return {"schema_version": 1, "actions": result}


def command_synopsis():
    def walk(p):
        children = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        if not children:
            return [p.format_usage().strip().replace("usage: ", "").replace("\n", " ")]
        return [line for a in children for child in a.choices.values() for line in walk(child)]

    return walk(build_parser())

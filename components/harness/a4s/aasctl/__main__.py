"""``aas`` -- a composable command surface over the AAS metamodel.

Design premise
--------------
The system this replaces exposed one LLM tool per authoring intent
(``aas_from_smt``, ``aas_match_inputs``, ``aas_populate_inputs``, ...). That
bounds the agent to intents someone anticipated: a submodel template nobody
wrote a tool for is unreachable, and every new template costs a code change
and a redeploy.

Here the *metamodel* is the surface. ``aas el add --type <modelType>`` can
construct any element IEC 63278 defines, at any path, in any submodel. An
IDTA template becomes a sequence of commands -- data the agent can compose,
store as a skill, or generate -- rather than a Python module.

Three properties follow, and they are the reason this shape was chosen:

* **Composability.** Commands emit JSON on ``--json``, so they pipe. The
  agent writes a shell pipeline instead of a chain of tool calls, and a
  20-step authoring task costs one turn instead of twenty.
* **Auditability.** Every mutation is a line of text. The ledger records the
  command, not a summary of it, so a replay reproduces the AAS exactly.
* **Constrainability.** A command grammar is a cleaner target for
  constrained decoding than a union of per-tool JSON schemas: the verb set
  is closed, and ``--type`` / ``--value-type`` are closed enums drawn from
  the metamodel rather than from a hand-written schema.

Output contract
---------------
Human-readable by default, JSON under ``--json``, and bounded either way --
a submodel in this domain routinely holds >100 elements, which does not
belong in a model's context window verbatim. Use ``tree`` for shape and
``get`` for values.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import AasHttpError, BasyxClient, Endpoint
from .model import (
    ELEMENT_TYPES,
    KIND_VALUES,
    VALUE_TYPES,
    ModelError,
    delegation_qualifier,
    external_reference,
    lang_strings,
    make_element,
    make_shell,
    make_submodel,
    operation_variable,
    qualifier,
    submodel_reference,
)
from ..kernel.approval import SAFETY_APPROVAL, moves_machine
from . import opc
from .paths import ElementPath, PathError
from .smt import SmtError, gaps, instantiate, is_optional, outline

# A submodel here averages ~130 elements. Printing one whole costs more
# context than it informs, so listings truncate and say so; the agent can
# always ask for a subtree.
MAX_LISTED = 60


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

class Out:
    """Emit either JSON or a compact human form, decided once at startup."""

    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json
        force_utf8()

    def emit(self, payload: Any, text: Optional[str] = None) -> None:
        if self.as_json:
            json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
            sys.stdout.write("\n")
        elif text is not None:
            print(text)
        else:
            print(json.dumps(payload, ensure_ascii=False))

    def lines(self, payload: Any, lines: Sequence[str]) -> None:
        if self.as_json:
            self.emit(payload)
        else:
            for line in lines:
                print(line)


def force_utf8() -> None:
    """Stop a non-ASCII character from killing the command on a Windows console.

    The default Windows code page is cp1252, and this repository is full of
    German descriptions and degree signs. Without this a `sm tree` over a
    perfectly valid submodel dies with a UnicodeEncodeError halfway through its
    output -- the data is fine, the terminal is not. Replacement characters are
    a bad rendering; a traceback is a failed command.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def fail(message: str, hint: Optional[str] = None, code: int = 1) -> None:
    """Exit with a message the agent can act on.

    The hint matters more than it looks: a bare "404" makes an agent retry
    the same call, a hint naming the command that would list valid paths
    makes it recover in one step.
    """
    print(f"error: {message}", file=sys.stderr)
    if hint:
        print(f"hint: {hint}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _client(args: argparse.Namespace) -> BasyxClient:
    try:
        return BasyxClient(Endpoint.from_env(args.endpoint))
    except RuntimeError as exc:
        fail(str(exc))
        raise  # unreachable, keeps type checkers quiet


def _path(raw: Optional[str]) -> ElementPath:
    try:
        return ElementPath.parse(raw or "")
    except PathError as exc:
        fail(str(exc), "paths look like Parent.Child or List[0].Child")
        raise


def _parse_kv(pairs: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            fail(f"expected key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = value
    return out


def _detail(element: Dict[str, Any]) -> str:
    """The type-and-value half of an element line, without its idShort."""
    model_type = element.get("modelType", "?")
    if model_type == "Property":
        return f"Property({element.get('valueType', '?')}) = {element.get('value')!r}"
    if model_type == "SubmodelElementCollection":
        return f"SMC [{len(element.get('value') or [])} children]"
    if model_type == "SubmodelElementList":
        return (
            f"SML<{element.get('typeValueListElement', '?')}>"
            f" [{len(element.get('value') or [])} items]"
        )
    if model_type == "Operation":
        delegated = _delegation_url(element)
        return f"Operation -> {delegated}" if delegated else "Operation"
    return model_type


def _summarise(element: Dict[str, Any]) -> str:
    """One line per element: enough to decide the next command, no more."""
    id_short = element.get("idShort") or "<indexed>"
    return f"{id_short:<40} {_detail(element)}"


def _delegation_url(element: Dict[str, Any]) -> Optional[str]:
    for qual in element.get("qualifiers") or []:
        if qual.get("type") == "invocationDelegation":
            return remap_host(qual.get("value"))
    return None


def remap_host(url: Optional[str]) -> Optional[str]:
    """Rewrite a delegation host for the network this process is actually on.

    A delegation URL is written for the network the *service* lives on. In a
    compose deployment that is `http://printer-service:5060`, which resolves
    inside the docker network and nowhere else -- so a CLI running on the host
    gets a DNS failure for a service that is up and reachable on
    `localhost:5060`. Neither the model nor the service is wrong; they are on
    different networks.

    `$A4S_DELEGATION_HOSTS` maps between them, as
    `printer-service=localhost,mes-service=localhost:15030`. A port in the
    replacement overrides the URL's; without one the original port is kept.
    Unmapped hosts pass through untouched, so the default behaviour is exactly
    what the model says.
    """
    if not url:
        return url
    mapping = os.getenv("A4S_DELEGATION_HOSTS", "").strip()
    if not mapping:
        return url
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if not parts.hostname:
        return url
    for pair in mapping.split(","):
        source, _, target = pair.partition("=")
        if source.strip() != parts.hostname or not target.strip():
            continue
        host, _, port = target.strip().partition(":")
        netloc = f"{host}:{port or (parts.port or '')}".rstrip(":")
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


def _children(element: Dict[str, Any]) -> List[Dict[str, Any]]:
    if element.get("modelType") in ("SubmodelElementCollection", "SubmodelElementList"):
        return element.get("value") or []
    if element.get("modelType") == "Submodel":
        return element.get("submodelElements") or []
    if element.get("modelType") == "Entity":
        return element.get("statements") or []
    return []


def sibling_id_shorts(
    client: BasyxClient, submodel_id: str, parent: ElementPath
) -> Optional[List[str]]:
    """Every idShort already present directly under ``parent``.

    ``None`` means the question could not be answered -- the parent does not
    exist, or the repository did not respond. The caller must not read that as
    "no siblings": the whole point of asking is to refuse a write, and a
    failure to look has to be distinguishable from a clean answer.

    The two container shapes differ and both have to be handled. Asking for
    the submodel root returns AAS Part 2's paged envelope,
    ``{"paging_metadata": ..., "result": [...]}``; asking for a nested element
    returns the element itself with its children under ``value`` or
    ``statements``.
    """
    try:
        raw = client.get_element(submodel_id, parent)
    except AasHttpError:
        return None
    except Exception:  # noqa: BLE001 - a transport failure is also "cannot tell"
        return None

    if isinstance(raw, list):
        children: List[Dict[str, Any]] = raw
    elif isinstance(raw, dict) and "result" in raw:
        children = raw.get("result") or []
    elif isinstance(raw, dict):
        children = _children(raw)
    else:
        return None
    return [c.get("idShort") for c in children if isinstance(c, dict) and c.get("idShort")]


def _walk(
    element: Dict[str, Any], prefix: str, depth: int, out: List[str], level: int = 0
) -> None:
    """Append one indented ``path  detail`` line per descendant, breadth-first by level."""
    if depth < 0:
        return
    is_list = element.get("modelType") == "SubmodelElementList"
    for index, child in enumerate(_children(element)):
        # An SML child has no idShort of its own; it is addressed by index.
        step = f"[{index}]" if is_list else child.get("idShort", f"[{index}]")
        path = f"{prefix}{step}" if is_list else (f"{prefix}.{step}" if prefix else step)
        out.append(f"{'  ' * level}{path:<{max(46 - 2 * level, 20)}} {_detail(child)}")
        if len(out) >= MAX_LISTED:
            return
        _walk(child, path, depth - 1, out, level + 1)


# ---------------------------------------------------------------------------
# shell commands
# ---------------------------------------------------------------------------

def cmd_shell_list(args: argparse.Namespace, out: Out) -> None:
    from .catalog import is_blueprint, shells_in_view

    with _client(args) as client:
        shells = shells_in_view(client.list_shells(limit=args.limit), args.asset_kind)
    rows = [
        {
            "id": s.get("id"),
            "idShort": s.get("idShort"),
            "globalAssetId": (s.get("assetInformation") or {}).get("globalAssetId"),
            "assetKind": (s.get("assetInformation") or {}).get("assetKind", "Unknown"),
            "usageRole": "blueprint" if is_blueprint(s) else "asset",
            "submodels": len(s.get("submodels") or []),
        }
        for s in shells
    ]
    out.lines(
        rows,
        [f"{r['idShort'] or '?':<32} {r['submodels']:>3} sm  {r['id']}" for r in rows]
        or ["(no shells)"],
    )


def resolve_shell(client: Any, name: str) -> str:
    """A shell id, given either an id or an idShort.

    Every shell verb took an id, and the things that call them mostly have an
    idShort: the wizard passes `targetAAS`, which is a label, and an agent told
    "the AAS InjectionMoldingMachine_653" has no way to learn the id from that
    sentence. Encoding the label as an id produced
    `404 Element with Id InjectionMoldingMachine_653 does not exist` -- correct,
    unhelpful, and a dead end, because nothing in the message says the label is
    not an id or how to turn one into the other.

    The id is tried first and the scan happens only on a miss, so the common
    case costs nothing. **An ambiguous label is refused rather than guessed**:
    two shells may share an idShort, AASd-022 only forbids duplicates among
    siblings, and picking one of two machines is worse than stopping.
    """
    try:
        client.get_shell(name)
        return name
    except AasHttpError as exc:
        if exc.status != 404:
            raise

    matches = [s for s in client.list_shells(limit=500) if s.get("idShort") == name]
    if len(matches) == 1:
        resolved = matches[0]["id"]
        # On stderr: stdout may be JSON that a caller is parsing, and the
        # resolution is something the operator should see happening.
        print(f"  ({name} is an idShort; using {resolved})", file=sys.stderr)
        return resolved
    if len(matches) > 1:
        fail(
            f"{len(matches)} shells have the idShort {name!r}",
            "name one by its id: " + ", ".join(m["id"] for m in matches[:5]),
        )
    fail(
        f"no shell with the id or idShort {name!r}",
        "list what the repository holds with: aas shell list",
    )
    raise SystemExit(2)      # unreachable; `fail` exits


def cmd_shell_show(args: argparse.Namespace, out: Out) -> None:
    with _client(args) as client:
        shell_id = resolve_shell(client, args.shell_id)
        shell = client.get_shell(shell_id)
        refs = client.list_submodel_refs(shell_id)
    submodel_ids = [
        key.get("value")
        for ref in refs
        for key in (ref.get("keys") or [])
        if key.get("type") == "Submodel"
    ]
    payload = {**shell, "_submodelIds": submodel_ids}
    lines = [
        f"id:        {shell.get('id')}",
        f"idShort:   {shell.get('idShort')}",
        f"asset:     {(shell.get('assetInformation') or {}).get('globalAssetId')}",
        f"submodels: {len(submodel_ids)}",
        *[f"  - {sid}" for sid in submodel_ids],
    ]
    out.lines(payload, lines)


def cmd_shell_create(args: argparse.Namespace, out: Out) -> None:
    shell = make_shell(
        args.shell_id,
        args.id_short,
        global_asset_id=args.global_asset_id,
        asset_kind=args.asset_kind,
        description=args.description,
    )
    if args.dry_run:
        out.emit(shell)
        return
    with _client(args) as client:
        client.create_shell(shell)
    out.emit({"created": args.shell_id}, f"created shell {args.shell_id}")


def cmd_shell_delete(args: argparse.Namespace, out: Out) -> None:
    with _client(args) as client:
        shell_id = resolve_shell(client, args.shell_id)
        client.delete_shell(shell_id)
    out.emit({"deleted": shell_id}, f"deleted shell {shell_id}")


def cmd_shell_attach(args: argparse.Namespace, out: Out) -> None:
    reference = submodel_reference(args.submodel_id)
    if args.dry_run:
        out.emit(reference)
        return
    with _client(args) as client:
        shell_id = resolve_shell(client, args.shell_id)
        client.add_submodel_ref(shell_id, reference)
    out.emit(
        {"shell": shell_id, "submodel": args.submodel_id},
        f"attached {args.submodel_id} to {shell_id}",
    )


def cmd_shell_detach(args: argparse.Namespace, out: Out) -> None:
    """Unbind a submodel from a shell without deleting the submodel.

    The inverse of `attach`, which existed without one. An agent could bind a
    submodel to a shell and had no way to undo it -- and the recovery it would
    reach for instead, `sm delete`, destroys the submodel rather than the
    reference, which is a much larger action than the one it was trying to
    reverse.
    """
    if args.dry_run:
        out.emit({"shell": args.shell_id, "submodel": args.submodel_id, "action": "detach"})
        return
    with _client(args) as client:
        shell_id = resolve_shell(client, args.shell_id)
        try:
            client.remove_submodel_ref(shell_id, args.submodel_id)
        except AasHttpError as exc:
            if exc.status == 404:
                fail(
                    f"{args.shell_id} does not reference {args.submodel_id}",
                    f"list what it does reference with: aas shell show {args.shell_id}",
                )
            raise
    out.emit(
        {"shell": args.shell_id, "submodel": args.submodel_id},
        f"detached {args.submodel_id} from {args.shell_id} "
        f"(the submodel itself still exists)",
    )


# ---------------------------------------------------------------------------
# submodel commands
# ---------------------------------------------------------------------------

def _duplicate_id_shorts(node: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    """Every place where siblings share an idShort, deepest-first.

    Deepest-first so a repair can rewrite nested containers before the parents
    that hold them, without an index shifting underneath it.
    """
    found: List[Dict[str, Any]] = []
    children = _children(node)
    if node.get("modelType") != "SubmodelElementList":
        counts: Dict[str, int] = {}
        for child in children:
            name = child.get("idShort")
            if name:
                counts[name] = counts.get(name, 0) + 1
        for name, count in counts.items():
            if count > 1:
                found.append(
                    {"parent": prefix, "idShort": name, "count": count,
                     "path": f"{prefix}.{name}" if prefix else name}
                )
    is_list = node.get("modelType") == "SubmodelElementList"
    for index, child in enumerate(children):
        step = f"[{index}]" if is_list else child.get("idShort", f"[{index}]")
        path = f"{prefix}{step}" if is_list else (f"{prefix}.{step}" if prefix else step)
        found = _duplicate_id_shorts(child, path) + found
    return found


def _drop_duplicates(node: Dict[str, Any]) -> int:
    """Keep the first of each colliding idShort. Returns how many were removed.

    First rather than last, and not a merge. The first copy is the one that was
    readable before the duplicate arrived and broke the path, so keeping it
    restores the state the repository was in; picking the later one would
    silently substitute a value nobody asked for.
    """
    removed = 0
    for key in _CHILD_KEYS_SM:
        children = node.get(key)
        if not isinstance(children, list):
            continue
        if node.get("modelType") != "SubmodelElementList":
            seen: set = set()
            kept: List[Any] = []
            for child in children:
                name = child.get("idShort") if isinstance(child, dict) else None
                if name and name in seen:
                    removed += 1
                    continue
                if name:
                    seen.add(name)
                kept.append(child)
            node[key] = kept
            children = kept
        for child in children:
            if isinstance(child, dict):
                removed += _drop_duplicates(child)
    return removed


_CHILD_KEYS_SM = ("submodelElements", "value", "statements", "annotations")


def cmd_sm_repair(args: argparse.Namespace, out: Out) -> None:
    """Remove duplicate idShorts by rewriting the submodel whole.

    This is the only route that can. BaSyx does not enforce AASd-022: a second
    element with an existing idShort is appended, answered with 404, and from
    then on every path-addressed read *and delete* of that path answers 404
    because the lookup is ambiguous. `el rm` cannot reach either copy. A
    whole-submodel PUT can, because it addresses the submodel rather than the
    element.

    `el add` now refuses to create such a duplicate, so this is for damage that
    already exists -- written by an earlier version, by another client, or
    loaded from an AASX file at boot. A duplicate in a preloaded file is also
    what stops a BaSyx environment container from starting, in which case this
    cannot help and the file has to be fixed first.

    Defaults to reporting, not repairing. Overwriting a submodel is not a small
    action and the agent asking for it usually wants to know what it would lose.
    """
    with _client(args) as client:
        submodel = get_submodel(client, args.submodel_id)
        duplicates = _duplicate_id_shorts(submodel)

        if not duplicates:
            out.lines(
                {"submodel": args.submodel_id, "duplicates": [], "repaired": False},
                ["ok: no duplicate idShorts"],
            )
            return

        lines = [
            f"{len(duplicates)} colliding idShort(s) in {args.submodel_id}:",
            *[f"  {d['path']}  x{d['count']}  (all unreadable by path)" for d in duplicates],
        ]
        if not args.apply:
            lines += [
                "",
                "nothing was changed. Re-run with --apply to keep the first copy of "
                "each and remove the rest, which requires rewriting the whole submodel.",
            ]
            out.lines(
                {"submodel": args.submodel_id, "duplicates": duplicates, "repaired": False},
                lines,
            )
            return

        removed = _drop_duplicates(submodel)
        if args.dry_run:
            out.emit(submodel)
            return
        client.replace_submodel(args.submodel_id, submodel)

    lines += ["", f"removed {removed} duplicate element(s); the first copy of each was kept",
              f"confirm with: aas validate {args.submodel_id}"]
    out.lines(
        {"submodel": args.submodel_id, "duplicates": duplicates,
         "removed": removed, "repaired": True},
        lines,
    )


def cmd_sm_list(args: argparse.Namespace, out: Out) -> None:
    from .catalog import submodels_in_view

    with _client(args) as client:
        submodels = submodels_in_view(
            client.list_submodels(limit=args.limit), client.list_shells(limit=1000), args.kind
        )
    rows = [
        {
            "id": s.get("id"),
            "idShort": s.get("idShort"),
            "kind": s.get("kind"),
            "semanticId": _semantic_value(s),
            "elements": len(s.get("submodelElements") or []),
        }
        for s in submodels
    ]
    out.lines(
        rows,
        [
            f"{r['idShort'] or '?':<32} {r['kind'] or 'Instance':<9} "
            f"{r['elements']:>3} el  {r['id']}"
            for r in rows
        ]
        or ["(no submodels)"],
    )


def _semantic_value(node: Dict[str, Any]) -> Optional[str]:
    keys = (node.get("semanticId") or {}).get("keys") or []
    return keys[0].get("value") if keys else None


def cmd_sm_create(args: argparse.Namespace, out: Out) -> None:
    submodel = make_submodel(
        args.submodel_id,
        args.id_short,
        semantic_id=args.semantic_id,
        kind=args.kind,
        description=args.description,
    )
    if args.dry_run:
        out.emit(submodel)
        return
    with _client(args) as client:
        client.create_submodel(submodel)
        if args.attach_to:
            client.add_submodel_ref(args.attach_to, submodel_reference(args.submodel_id))
    out.emit({"created": args.submodel_id}, f"created submodel {args.submodel_id}")


def cmd_sm_delete(args: argparse.Namespace, out: Out) -> None:
    with _client(args) as client:
        client.delete_submodel(args.submodel_id)
    out.emit({"deleted": args.submodel_id}, f"deleted submodel {args.submodel_id}")


def cmd_sm_tree(args: argparse.Namespace, out: Out) -> None:
    with _client(args) as client:
        submodel = get_submodel(client, args.submodel_id)
    lines: List[str] = []
    _walk(submodel, "", args.depth, lines)
    truncated = len(lines) >= MAX_LISTED
    header = f"{submodel.get('idShort')}  ({submodel.get('id')})"
    if truncated:
        lines.append(f"... truncated at {MAX_LISTED} rows; narrow with 'aas el ls <sm> <path>'")
    out.lines(
        {"submodel": submodel.get("id"), "paths": lines, "truncated": truncated},
        [header, *lines],
    )


# ---------------------------------------------------------------------------
# element commands
# ---------------------------------------------------------------------------

def cmd_el_add(args: argparse.Namespace, out: Out) -> None:
    quals: List[Dict[str, Any]] = []
    if args.delegate_to:
        quals.append(delegation_qualifier(args.delegate_to))
    for raw in args.qualifier or []:
        key, _, value = raw.partition("=")
        quals.append(qualifier(key, value))

    inputs = [
        operation_variable(make_element("Property", name, value_type=vtype))
        for name, vtype in _parse_kv(args.input or []).items()
    ]
    outputs = [
        operation_variable(make_element("Property", name, value_type=vtype))
        for name, vtype in _parse_kv(args.output or []).items()
    ]

    try:
        element = make_element(
            args.type,
            args.id_short,
            value=args.value,
            value_type=args.value_type,
            semantic_id=args.semantic_id,
            description=args.description,
            qualifiers=quals or None,
            min_value=args.min,
            max_value=args.max,
            content_type=args.content_type,
            type_value_list_element=args.type_value_list_element,
            value_type_list_element=args.value_type_list_element,
            first=external_reference(args.first) if args.first else None,
            second=external_reference(args.second) if args.second else None,
            global_asset_id=args.global_asset_id,
            input_variables=inputs or None,
            output_variables=outputs or None,
        )
    except ModelError as exc:
        fail(str(exc))
        return

    if args.dry_run:
        out.emit(element)
        return

    parent = _path(args.parent)
    with _client(args) as client:
        # AASd-022 preflight. BaSyx does not enforce idShort uniqueness among
        # siblings and the two ways it fails are both silent: at a submodel
        # root it appends the duplicate and answers 404, and inside a
        # collection it appends it and answers 200. Either way both copies
        # become unaddressable -- every later read of that path answers 404,
        # because the lookup is ambiguous -- and the only repair is a
        # whole-submodel PUT, which no verb here offers.
        #
        # So the check happens before the write. It costs one GET per `el add`,
        # which is the price of not being able to undo the alternative. Skipped
        # for an SML child, which has no idShort by construction and where
        # appending is the correct behaviour.
        #
        # This is a guard against an agent repeating itself, not a lock: two
        # processes writing the same idShort at once can still race past it.
        # Single-writer is the deployment assumption everywhere else here too.
        if args.id_short:
            existing = sibling_id_shorts(client, args.submodel_id, parent)
            if existing is None:
                fail(
                    f"cannot read {parent.render() or '<root>'} to check for a "
                    f"duplicate idShort",
                    f"confirm the parent exists: aas sm tree {args.submodel_id}",
                )
            if args.id_short in existing:
                fail(
                    f"{args.id_short} already exists under "
                    f"{parent.render() or '<root>'}",
                    f"change its value instead: aas set {args.submodel_id} "
                    f"{parent.child(args.id_short).render()} <value>  "
                    f"(a second element with the same idShort would violate "
                    f"AASd-022 and make both unreadable)",
                )

        try:
            client.add_element(args.submodel_id, parent, element)
        except AasHttpError as exc:
            if exc.status == 404:
                fail(
                    f"parent path not found: {parent.render() or '<root>'}",
                    f"list what exists with: aas sm tree {args.submodel_id}",
                )
            if exc.status == 409:
                fail(
                    f"{args.id_short} already exists under {parent.render() or '<root>'}",
                    f"replace its value with: aas set {args.submodel_id} "
                    f"{parent.child(args.id_short).render()} <value>",
                )
            raise
    if args.id_short:
        created = parent.child(args.id_short).render()
        message = f"added {args.type} at {created}"
    else:
        # An SML child is appended, and its index is assigned by the server.
        # Reporting the parent path alone would read as if it replaced it.
        created = f"{parent.render()}[appended]"
        message = f"appended {args.type} to {parent.render()}; confirm its index with: aas el ls {args.submodel_id} {parent.render()}"
    out.emit(
        {"submodel": args.submodel_id, "path": created, "modelType": args.type}, message
    )


def cmd_el_ls(args: argparse.Namespace, out: Out) -> None:
    path = _path(args.path)
    with _client(args) as client:
        node = (
            client.get_element(args.submodel_id, path)
            if path
            else get_submodel(client, args.submodel_id)
        )
    lines: List[str] = []
    _walk(node, "", args.depth, lines)
    out.lines({"path": path.render(), "children": lines}, lines or ["(no children)"])


def cmd_el_show(args: argparse.Namespace, out: Out) -> None:
    path = _path(args.path)
    with _client(args) as client:
        element = client.get_element(args.submodel_id, path)
    lines = [
        f"path:       {path.render()}",
        f"modelType:  {element.get('modelType')}",
        f"semanticId: {_semantic_value(element)}",
        f"value:      {element.get('value')!r}",
    ]
    if _delegation_url(element):
        lines.append(f"delegation: {_delegation_url(element)}")
    # Qualifiers were not printed, and they are where the meaning is: an
    # invocation target, a safety declaration, an SMT cardinality, the OPC UA
    # node a value is read from. An agent that cannot see them has to reach
    # for raw HTTP to find out what an element is for, which is the failure
    # this CLI exists to prevent.
    for qual in element.get("qualifiers") or []:
        kind = qual.get("type", "?")
        lines.append(f"  @{kind:<22} {qual.get('value')}")
    out.lines(element, lines)


def cmd_el_rm(args: argparse.Namespace, out: Out) -> None:
    path = _path(args.path)
    with _client(args) as client:
        client.delete_element(args.submodel_id, path)
    out.emit({"deleted": path.render()}, f"deleted {path.render()}")


# ---------------------------------------------------------------------------
# value commands
# ---------------------------------------------------------------------------

def cmd_get(args: argparse.Namespace, out: Out) -> None:
    path = _path(args.path)
    with _client(args) as client:
        try:
            value = client.get_value(args.submodel_id, path)
        except AasHttpError as exc:
            if exc.status == 404:
                fail(
                    f"no element at {path.render()}",
                    f"see valid paths with: aas sm tree {args.submodel_id}",
                )
            raise
    out.emit(value, str(value))


# Lives in `model` now, because `populate` needs the same conversion and the
# two copies had drifted. Kept under this name because it is what `cmd_set`,
# the baseline and the tests already call.
_lang_strings = lang_strings


def cmd_set(args: argparse.Namespace, out: Out) -> None:
    path = _path(args.path)
    # A bare scalar is the common case; JSON is accepted for MLP/Range/SMC
    # values, which are structured under $value semantics.
    try:
        value: Any = json.loads(args.value) if args.json_value else args.value
    except json.JSONDecodeError as exc:
        fail(f"--json-value given but value is not JSON: {exc}")
        return
    if args.lang:
        value = _lang_strings(args.value)
    if args.dry_run:
        out.emit({"path": path.render(), "value": value})
        return
    with _client(args) as client:
        try:
            client.set_value(args.submodel_id, path, value)
        except AasHttpError as exc:
            if exc.status == 404:
                fail(
                    f"no element at {path.render()}",
                    f"create it with: aas el add {args.submodel_id} "
                    f"{path.parent.render()} --type Property --id-short {path.leaf_id_short}",
                )
            if exc.status == 500 and isinstance(value, str):
                # BaSyx answers 500, not 400, when a scalar is written to an
                # element whose $value is structured. The status does not say
                # so; the hint has to.
                fail(
                    f"HTTP 500 writing a plain string to {path.render()}",
                    "if this is a MultiLanguageProperty, use --lang: "
                    f"aas set {args.submodel_id} {path.render()} 'Some text@en'",
                )
            raise
    out.emit({"path": path.render(), "value": value}, f"set {path.render()} = {value!r}")


# ---------------------------------------------------------------------------
# operation commands
# ---------------------------------------------------------------------------

def cmd_op_invoke(args: argparse.Namespace, out: Out) -> None:
    """Invoke an Operation, through its delegation qualifier when it has one.

    Delegation is the orchestration half of A4S: the Operation is the
    semantic handle, and ``invocationDelegation`` names the service that
    actually does the work. Resolving it here -- rather than in a bespoke
    tool -- means any Operation an agent can author, it can also call.
    """
    path = _path(args.path)
    arguments = _parse_kv(args.arg or [])

    with _client(args) as client:
        element = client.get_element(args.submodel_id, path)
        if element.get("modelType") != "Operation":
            fail(
                f"{path.render()} is a {element.get('modelType')}, not an Operation",
                "list callable operations with: aas op ls <submodel-id>",
            )
        url = _delegation_url(element)

        if args.dry_run:
            out.emit({"path": path.render(), "delegation": url, "arguments": arguments})
            return

        if url and opc.is_opc_url(url) and not args.local:
            # The delegation names an OPC UA server and a node. Writing the
            # node is the invocation -- an S7-1500 offers no methods, and what
            # an AAS calls `StartApplication` is `xStart := true` in a data
            # block. The approval gate has already run by this point, on the
            # same qualifier it reads for an HTTP delegation.
            endpoint, node = opc.split_target(url)
            if not node:
                fail(
                    f"{url} names an OPC UA server but no node",
                    "write the node in the fragment: "
                    "opc.tcp://host:port#ns=3;s=\"DB\".\"Tag\"",
                )
            value = arguments.get("value", "true")
            try:
                after = opc.write(endpoint, node, str(value), timeout=args.timeout)
            except opc.OpcError as exc:
                fail(str(exc), "check the server is reachable: aas opc info " + endpoint)
                raise
            out.lines(
                {"via": url, "node": node, "wrote": value,
                 "readBack": after.value, "dataType": after.data_type,
                 "outcome": after.settled(), "observed": after.observed},
                [
                    f"wrote {value} to {node}",
                    f"observed over {opc.SETTLE_WINDOW}s: "
                    + ", ".join(repr(v) for v in after.observed),
                    f"{after.settled()}: {after.explain()}",
                ],
            )
            return

        if url and not args.local:
            import httpx  # local import: only the delegating path needs it

            try:
                response = httpx.post(url, json=arguments, timeout=args.timeout, verify=False)
            except httpx.TransportError as exc:
                # Say which host could not be reached. Reported as a repository
                # failure this sends the caller to check BaSyx, which is fine.
                fail(
                    f"cannot reach the delegated service at {url}: "
                    f"{type(exc).__name__}: {exc}",
                    "the delegation URL is written for the service's own network. "
                    "If it is a container name, map it with "
                    "A4S_DELEGATION_HOSTS=printer-service=localhost",
                )
                raise
            body = _maybe_json(response.text)
            if response.status_code >= 400:
                fail(f"delegated service returned {response.status_code}: {response.text[:300]}")
            out.emit({"via": url, "response": body}, json.dumps(body, ensure_ascii=False))
            return

        payload = {
            "inputArguments": [
                operation_variable(make_element("Property", key, value=value))
                for key, value in arguments.items()
            ],
            "clientTimeoutDuration": f"PT{int(args.timeout)}S",
        }
        result = client.invoke_operation(args.submodel_id, path, payload)
    out.emit(result, json.dumps(result, ensure_ascii=False))


def cmd_opc_ls(args: argparse.Namespace, out: Out) -> None:
    """Every element in a submodel that says where its value comes from.

    The counterpart of `aas op ls`. That one answers "what can this asset be
    asked to do"; this answers "what can be read off it, and from where".
    Both read a qualifier off the model rather than being told an address, so
    an agent discovers the cell from the AAS instead of from its prompt --
    which is the claim the whole harness is making.
    """
    with _client(args) as client:
        submodel = get_submodel(client, args.submodel_id)

    found: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any], prefix: str) -> None:
        is_list = node.get("modelType") == "SubmodelElementList"
        for index, child in enumerate(_children(node)):
            step = f"[{index}]" if is_list else child.get("idShort", f"[{index}]")
            path = f"{prefix}{step}" if is_list else (f"{prefix}.{step}" if prefix else step)
            source = next(
                (q.get("value") for q in child.get("qualifiers") or []
                 if q.get("type") == opc.VALUE_SOURCE),
                None,
            )
            if source:
                endpoint, node_id = opc.split_target(str(source))
                found.append({
                    "path": path,
                    "endpoint": endpoint,
                    "node": node_id,
                    "valueType": child.get("valueType"),
                    "lastValue": child.get("value"),
                })
            visit(child, path)

    visit(submodel, "")
    lines = [
        f"{f['path']:<24} {str(f['valueType'] or ''):<12} {f['endpoint']}  {f['node']}"
        for f in found
    ]
    if found:
        endpoints = sorted({f["endpoint"] for f in found})
        lines += ["", "read them with:",
                  f"  aas opc read {endpoints[0]} '<node>' ['<node>' ...]"]
    out.lines(found, lines or ["(no element in this submodel names a value source)"])


def cmd_opc_info(args: argparse.Namespace, out: Out) -> None:
    """What an OPC UA server says about itself."""
    try:
        info = opc.endpoints(args.endpoint, timeout=args.timeout)
    except opc.OpcError as exc:
        fail(str(exc))
        raise
    out.lines(info, [f"  {k:<18} {v}" for k, v in info.items()])


def cmd_opc_read(args: argparse.Namespace, out: Out) -> None:
    """Read one or more nodes. Never gated: reading is not an action."""
    try:
        values = opc.read(args.endpoint, args.node, timeout=args.timeout)
    except opc.OpcError as exc:
        fail(str(exc))
        raise
    out.lines([v.__dict__ for v in values], [v.line() for v in values])


def cmd_opc_write(args: argparse.Namespace, out: Out) -> None:
    """Write one node. Gated, because a write to a PLC moves something."""
    if args.dry_run:
        out.emit({"endpoint": args.endpoint, "node": args.node,
                  "value": args.value, "applied": False})
        return
    try:
        after = opc.write(args.endpoint, args.node, args.value,
                          timeout=args.timeout, expect_type=args.type)
    except opc.OpcError as exc:
        fail(str(exc))
        raise
    out.lines(
        after.__dict__,
        [f"wrote {args.value} to {args.node}",
         f"observed over {opc.SETTLE_WINDOW}s: "
         + ", ".join(repr(v) for v in after.observed),
         f"{after.settled()}: {after.explain()}"],
    )


def cmd_opc_browse(args: argparse.Namespace, out: Out) -> None:
    """Walk the address space, bounded. A PLC exposes every symbol it has."""
    try:
        found = opc.browse(args.endpoint, args.node, depth=args.depth,
                           timeout=args.timeout)
    except opc.OpcError as exc:
        fail(str(exc))
        raise

    lines: List[str] = []

    def render(nodes: List[Any]) -> None:
        for node in nodes:
            lines.append(node.line())
            render(node.children)

    render(found)
    out.lines([n.__dict__ for n in found],
              lines or ["(nothing under that node)"])


def cmd_opc_call(args: argparse.Namespace, out: Out) -> None:
    """Call an OPC UA method. Gated with the same risk as an invocation."""
    if args.dry_run:
        out.emit({"endpoint": args.endpoint, "node": args.node,
                  "arguments": args.arg or [], "applied": False})
        return
    try:
        result = opc.call(args.endpoint, args.node, args.arg or [],
                          timeout=args.timeout)
    except opc.OpcError as exc:
        fail(str(exc))
        raise
    out.emit({"result": result}, json.dumps(result, ensure_ascii=False, default=str))


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_op_ls(args: argparse.Namespace, out: Out) -> None:
    """List every Operation in a submodel with its delegation target.

    This is the agent's capability discovery step: what can this asset
    actually be asked to do, and where does each request land.
    """
    with _client(args) as client:
        submodel = get_submodel(client, args.submodel_id)

    found: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any], prefix: str) -> None:
        is_list = node.get("modelType") == "SubmodelElementList"
        for index, child in enumerate(_children(node)):
            step = f"[{index}]" if is_list else child.get("idShort", f"[{index}]")
            path = f"{prefix}{step}" if is_list else (f"{prefix}.{step}" if prefix else step)
            if child.get("modelType") == "Operation":
                found.append(
                    {
                        "path": path,
                        "delegation": _delegation_url(child),
                        "inputs": [
                            (v.get("value") or {}).get("idShort")
                            for v in child.get("inputVariables") or []
                        ],
                        # Whether a person decided this, or the gate is about to
                        # guess. An undeclared operation is not a safe one --
                        # it is one nobody has classified -- and a listing that
                        # does not say so hides the work still to do.
                        "declared": _safety_declaration(child),
                        "requiresSafetyApproval": moves_machine(child),
                    }
                )
            visit(child, path)

    visit(submodel, "")
    lines = [
        f"{_safety_mark(op):<10} {op['path']:<44} "
        f"in({', '.join(str(i) for i in op['inputs'])}) -> {op['delegation'] or 'local'}"
        for op in found
    ] or ["(no operations)"]
    undeclared = [op for op in found if op["declared"] is None]
    if undeclared:
        lines += [
            "",
            f"{len(undeclared)} of {len(found)} operations carry no "
            f"{SAFETY_APPROVAL} declaration, so the approval gate falls back to "
            f"reading their names. Settle one with:",
            f"  aas op declare {args.submodel_id} <path> --requires-approval true|false",
        ]
    out.lines(found, lines)


def _safety_declaration(element: Dict[str, Any]) -> Optional[bool]:
    """The declared value, or None when nobody has declared one."""
    for qual in element.get("qualifiers") or []:
        if qual.get("type") == SAFETY_APPROVAL:
            return str(qual.get("value", "")).strip().lower() in ("true", "1", "yes")
    return None


def _safety_mark(op: Dict[str, Any]) -> str:
    if op["declared"] is None:
        return "GUESSED" if op["requiresSafetyApproval"] else "guessed"
    return "APPROVAL" if op["declared"] else "free"


def cmd_op_declare(args: argparse.Namespace, out: Out) -> None:
    """Record whether invoking an Operation needs a person's approval.

    **The gate can only be as good as what the model tells it.** An Operation
    with no declaration is classified by reading its name, which is a poor way
    to decide whether something moves a press -- and the reason the fallback
    leans towards stopping is that being wrong the other way is unrecoverable.
    Measured on a live repository: 47 of 51 delegated operations carried no
    declaration, `StartCycle` among them, and the gate let it through.

    So this is the verb that turns a guess into a decision. It writes a
    `ConceptQualifier`, not a `TemplateQualifier`: AASd-129 confines the latter
    to templates and `smt instantiate` strips it, so a declaration made on a
    blueprint would vanish from every machine instantiated from it -- which is
    exactly the case this exists to fix.
    """
    _name = SAFETY_APPROVAL
    wanted = str(args.requires_approval).strip().lower() in ("true", "1", "yes")
    path = _path(args.path)

    with _client(args) as client:
        try:
            element = client.get_element(args.submodel_id, path)
        except AasHttpError as exc:
            if exc.status == 404:
                fail(
                    f"no element at {path.render()}",
                    f"list what is there with: aas op ls {args.submodel_id}",
                )
            raise
        if element.get("modelType") != "Operation":
            fail(
                f"{path.render()} is a {element.get('modelType')}, not an Operation",
                "only an Operation can be invoked, so only an Operation needs this",
            )

        quals = [q for q in element.get("qualifiers") or [] if q.get("type") != _name]
        previous = _safety_declaration(element)
        quals.append(qualifier(_name, "true" if wanted else "false", "xs:boolean"))
        element["qualifiers"] = quals

        if args.dry_run:
            out.emit({"path": path.render(), "qualifiers": quals, "applied": False})
            return
        client.replace_element(args.submodel_id, path, element)

    was = "undeclared" if previous is None else str(previous).lower()
    out.lines(
        {"path": path.render(), "requiresSafetyApproval": wanted, "previously": previous},
        [
            f"{path.render()}: {_name} = {str(wanted).lower()} (was {was})",
            "invoking it now "
            + ("asks an operator first" if wanted else "does not stop at the gate"),
        ],
    )


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# submodel-template commands
# ---------------------------------------------------------------------------

def get_submodel(client: BasyxClient, raw: str) -> Dict[str, Any]:
    """Fetch a submodel by id, falling back to a unique idShort.

    Identifiers here are long URIs and the listings print an idShort beside
    them, so an agent reaches for the short one -- twice in one live run, each
    costing a step to a 404 that only says the element does not exist. The
    fallback costs one extra request and only on the failing path.

    An ambiguous idShort is not resolved. Fifteen submodels in the reference
    repository are called `TechnicalData`; picking one would be a guess, and a
    guess that writes to the wrong asset is worse than an error.
    """
    try:
        return client.get_submodel(raw)
    except AasHttpError as exc:
        if exc.status != 404:
            raise
    matches = [s for s in client.list_submodels(limit=500) if s.get("idShort") == raw]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        fail(
            f"no submodel with id or idShort {raw!r}",
            "list what is there with: aas sm list",
        )
    fail(
        f"idShort {raw!r} is ambiguous -- {len(matches)} submodels share it",
        "use the full id: " + ", ".join(str(m.get("id")) for m in matches[:5]),
    )
    raise AssertionError("unreachable")


def _looks_like_endpoint(raw: str) -> bool:
    """Distinguish a repository URL from an AAS identifier.

    Both are commonly ``https://…``, so the discriminator is the path: an
    endpoint is a host, optionally with a short base path, while an AAS id is
    a long path-like identifier (``https://vendor/ids/aas/Something``). A
    non-http scheme -- ``urn:…`` -- is never an endpoint.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    return parts.path.strip("/").count("/") == 0


def _source_client(args: argparse.Namespace) -> BasyxClient:
    """The repository templates are read *from*, which is not always the target.

    A4S runs against two environments at once: the agent-internal one holding
    the SMT collection, and the customer environment named per task. Conflating
    them writes an instance into the template library, so the source endpoint is
    its own flag with its own variable and never defaults across.
    """
    override = getattr(args, "from_endpoint", None) or os.getenv("A4S_SMT_ENDPOINT")
    # Two live runs passed a shell id here when they meant --of-shell. Both
    # arguments are https:// strings, so nothing downstream catches it -- the
    # request just goes to a host that does not exist. Rejecting it here turns
    # a wasted step into a corrected one.
    if override and not _looks_like_endpoint(override):
        fail(
            f"--from-endpoint takes a repository URL, not {override!r}",
            "to filter templates by the shell that curates them, use "
            "--of-shell <aas-id>",
        )
    try:
        return BasyxClient(Endpoint.from_env(override or args.endpoint))
    except RuntimeError as exc:
        fail(str(exc))
        raise


def cmd_smt_list(args: argparse.Namespace, out: Out) -> None:
    with _source_client(args) as client:
        if args.of_shell:
            refs = client.list_submodel_refs(args.of_shell)
            ids = [k["value"] for r in refs for k in r.get("keys", []) if k.get("type") == "Submodel"]
            templates = [client.get_submodel(i) for i in ids]
        else:
            templates = client.list_submodels(limit=args.limit)

    rows = []
    for sm in templates:
        if sm.get("kind") != "Template":
            continue
        top = sm.get("submodelElements") or []
        rows.append(
            {
                "id": sm.get("id"),
                "idShort": sm.get("idShort"),
                "semanticId": _semantic_value(sm),
                "required": sum(1 for e in top if not is_optional(e)),
                "optional": sum(1 for e in top if is_optional(e)),
            }
        )
    out.lines(
        rows,
        [
            f"{r['idShort'] or '?':<32} {r['required']:>3} req {r['optional']:>3} opt  {r['id']}"
            for r in rows
        ]
        or ["(no submodel templates)"],
    )


def cmd_smt_show(args: argparse.Namespace, out: Out) -> None:
    with _source_client(args) as client:
        template = get_submodel(client, args.template_id)
    rows = outline(template, max_rows=MAX_LISTED)
    truncated = len(rows) >= MAX_LISTED
    if truncated:
        rows.append(f"... truncated at {MAX_LISTED} rows")
    out.lines(
        {
            "template": template.get("id"),
            "kind": template.get("kind"),
            "semanticId": _semantic_value(template),
            "slots": rows,
            "truncated": truncated,
        },
        [
            f"{template.get('idShort')}  ({template.get('id')})  kind={template.get('kind')}",
            f"{'card':<6} path                                         type",
            *rows,
        ],
    )


def cmd_smt_instantiate(args: argparse.Namespace, out: Out) -> None:
    with _source_client(args) as client:
        template = get_submodel(client, args.template_id)

    submodel_id = args.as_id or f"{args.template_id}_{args.id_short or 'Instance'}"
    try:
        result = instantiate(
            template,
            submodel_id=submodel_id,
            id_short=args.id_short,
            include_optional=args.include_optional,
            keep_values=args.keep_values,
            seed_examples=args.seed_examples,
            keep_template_qualifiers=args.keep_template_qualifiers,
            require_template=not args.force,
        )
    except SmtError as exc:
        fail(str(exc), "list what is available with 'aas smt list'")
        raise

    if args.dry_run:
        out.emit(result.submodel)
        return

    with _client(args) as client:
        existing = None
        try:
            existing = client.get_submodel(submodel_id)
        except AasHttpError as exc:
            if exc.status != 404:
                raise
        if existing is not None:
            if not args.replace:
                fail(
                    f"submodel {submodel_id!r} already exists",
                    "choose another --as, or pass --replace to overwrite it",
                )
            client.delete_submodel(submodel_id)
        client.create_submodel(result.submodel)
        if args.attach_to:
            try:
                client.add_submodel_ref(args.attach_to, submodel_reference(submodel_id))
            except AasHttpError as exc:
                # Re-instantiating with --replace deletes and recreates the
                # submodel, but the shell's reference to it never went away.
                # A 409 here means the link already exists, which is the state
                # the caller asked for.
                if exc.status != 409:
                    raise

    payload = result.summary()
    payload["from_template"] = args.template_id
    payload["attached_to"] = args.attach_to
    lines = [
        f"instantiated {args.template_id}",
        f"  -> {submodel_id}  ({payload['elements']} top-level elements)",
        f"  kept {payload['kept']} slots, dropped {payload['dropped_optional']} optional, "
        f"stripped {payload['stripped_qualifiers']} template qualifiers",
    ]
    if args.attach_to:
        lines.append(f"  attached to {args.attach_to}")
    if result.seeded:
        lines.append(f"  seeded {len(result.seeded)} example values -- NOT evidence, replace before use")
    for unknown in result.unknown_cardinality:
        lines.append(f"  warning: unrecognised cardinality at {unknown}; treated as mandatory")
    out.lines(payload, lines)


def cmd_smt_author(args: argparse.Namespace, out: Out) -> None:
    """Build a `kind: Template` submodel from a structure file.

    The other half of `a4s extract --as-template`, and deliberately a separate
    command: the structure is reviewed as a file before anything is written,
    which is the same propose/review/apply shape `aas match` and `aas populate`
    already have. A template is worse than a wrong instance to get wrong,
    because every instance built from it inherits the mistake.

    Built as one submodel and POSTed whole rather than as a sequence of
    `el add` calls. Nesting is the reason: `el add` addresses a parent by path,
    so a three-level template means creating the containers, reading back what
    they were called, and adding into them -- and a failure halfway leaves a
    partial template that validates as a complete one.
    """
    from ..structure import Structure, problems, render
    from .model import cardinality_qualifier

    path = Path(args.structure)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"no structure file at {path}",
             "produce one with: a4s extract --pdf <file> --as-template --out <file>")
        raise
    except json.JSONDecodeError as exc:
        fail(f"{path} is not JSON: {exc}")
        raise

    from ..structure import _slot  # noqa: PLC0415 - one call site, keeps the parser private

    try:
        structure = Structure(
            id_short=str(raw.get("idShort") or "ProposedTemplate"),
            description=str(raw.get("description") or ""),
            source=str(raw.get("source") or ""),
            semantic_id=raw.get("semanticId") or None,
            slots=[_slot(s, str(raw.get("source") or "")) for s in raw.get("slots") or []],
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a message, not a traceback
        fail(f"cannot read the structure: {exc}")
        raise

    found = problems(structure)
    if found and not args.force:
        out.lines(
            {"structure": str(path), "problems": found, "created": False},
            [f"{f['rule']}  {f['path']}: {f['detail']}" for f in found]
            + ["", "nothing was written. Fix the structure file, or pass --force to "
                  "build it anyway and repair afterwards."],
        )
        return

    def build(slot: Any, in_list: bool = False) -> Dict[str, Any]:
        # AASd-120: a SubmodelElementList child carries no idShort -- it is
        # addressed by position. The *structure* file keeps the name anyway,
        # because "Axis" is how a person reads what an entry is, and dropping
        # it there would make the proposal unreviewable to save a field the
        # metamodel forbids only on the wire. So the name lives in the document
        # and is discarded here, which is the one place that knows the parent.
        element = make_element(
            slot.model_type,
            None if in_list else slot.id_short,
            value=None,
            value_type=slot.value_type,
            semantic_id=slot.semantic_id,
            description=slot.description or None,
            qualifiers=[cardinality_qualifier(slot.cardinality)],
            type_value_list_element=slot.type_value_list_element,
        )
        children = [
            build(c, in_list=slot.model_type == "SubmodelElementList")
            for c in slot.children
        ]
        if children:
            element["value"] = children
        return element

    submodel = make_submodel(
        args.submodel_id,
        args.id_short or structure.id_short,
        semantic_id=args.semantic_id or structure.semantic_id,
        kind="Template",
        description=structure.description or None,
    )
    submodel["submodelElements"] = [build(s) for s in structure.slots]

    if args.dry_run:
        out.emit(submodel)
        return

    with _client(args) as client:
        if args.replace:
            with contextlib.suppress(AasHttpError):
                client.delete_submodel(args.submodel_id)
        try:
            client.create_submodel(submodel)
        except AasHttpError as exc:
            if exc.status == 409:
                fail(f"{args.submodel_id} already exists",
                     "pass --replace to overwrite it")
            raise
        if args.attach_to:
            client.add_submodel_ref(args.attach_to, submodel_reference(args.submodel_id))

    counts = structure.counts()
    out.lines(
        {"submodel": args.submodel_id, "kind": "Template",
         "slots": counts["slots"], "problems": found, "created": True},
        render(structure) + [
            "",
            f"created {args.submodel_id} as kind=Template with {counts['slots']} slots",
            f"check it with: aas validate {args.submodel_id}",
            f"use it with:   aas smt instantiate {args.submodel_id} --as <instance-id>",
        ],
    )


def cmd_smt_diff(args: argparse.Namespace, out: Out) -> None:
    """Report which of a template's slots the instance has not answered.

    This cannot be part of ``aas validate``. Which slots are mandatory is
    declared by cardinality qualifiers that AASd-129 forbids on an instance, so
    an instantiated submodel no longer carries the information -- completeness
    is a comparison between two submodels, not a property of one.
    """
    with _source_client(args) as source:
        template = get_submodel(source, args.template_id)
    with _client(args) as client:
        instance = get_submodel(client, args.submodel_id)

    found = gaps(template, instance, include_optional=args.include_optional)
    missing = [g for g in found if g.status == "missing"]
    empty = [g for g in found if g.status == "empty"]
    payload = {
        "template": args.template_id,
        "submodel": args.submodel_id,
        "complete": not found,
        "missing": [g.__dict__ for g in missing],
        "empty": [g.__dict__ for g in empty],
    }
    if not found:
        out.lines(payload, ["complete: every required slot is filled"])
        return
    out.lines(
        payload,
        [
            f"{len(missing)} missing, {len(empty)} empty "
            f"(missing needs 'aas el add', empty needs 'aas set')",
            f"{'status':<8} {'card':<6} path                                         type",
            *[g.line() for g in found[:MAX_LISTED]],
        ]
        + ([f"... truncated at {MAX_LISTED} rows"] if len(found) > MAX_LISTED else []),
    )


# ---------------------------------------------------------------------------
# retrieval commands
# ---------------------------------------------------------------------------

def _index_home(args: argparse.Namespace) -> "Path":
    from . import index as index_mod

    raw = getattr(args, "index_home", None) or os.getenv("A4S_INDEX_HOME")
    return Path(raw) if raw else index_mod.DEFAULT_INDEX_HOME


def cmd_index(args: argparse.Namespace, out: Out) -> None:
    """Build and cache the retrieval index for one submodel, or list what is cached."""
    from . import index as index_mod

    home = _index_home(args)
    if args.list:
        import datetime as _dt

        rows = index_mod.Index.cached(home)
        out.lines(
            rows,
            [
                f"{(r['idShort'] or '?'):<26} {r['nodes']:>4} nodes  "
                f"{r['enriched']:>4} enriched  "
                f"{'values' if r.get('with_values') else 'schema'}  "
                f"{_dt.datetime.fromtimestamp(r.get('built_at') or 0):%Y-%m-%d %H:%M}  "
                f"{r['submodel_id']}"
                for r in rows
            ]
            or [f"(no indexes cached in {home})"],
        )
        return

    if args.drop:
        if not args.submodel_id:
            fail("--drop needs a submodel id")
        removed = index_mod.Index.drop(args.submodel_id, home)
        out.emit(
            {"dropped": args.submodel_id, "removed": removed},
            f"dropped index for {args.submodel_id}" if removed else "no index to drop",
        )
        return

    if args.all:
        from .catalog import submodels_in_view

        with _client(args) as client:
            submodels = submodels_in_view(
                client.list_submodels(limit=args.limit), client.list_shells(limit=1000), args.kind
            )
            concepts = index_mod.concept_map(client.list_concept_descriptions())
        if not submodels:
            fail("the repository served no submodels")
        print(f"indexing {len(submodels)} submodels...", file=sys.stderr)
        try:
            built = index_mod.build_all(
                submodels, concepts,
                model=args.embedding_model, with_values=args.with_values,
            )
        except index_mod.IndexError_ as exc:
            fail(str(exc))
            raise
        path = built.save(home)
        out.lines(
            {
                "submodel": index_mod.ALL, "submodels": len(submodels),
                "nodes": len(built.nodes), "enriched": built.enriched,
                "model": built.model, "cache": str(path),
            },
            [
                f"indexed {len(built.nodes)} elements from {len(submodels)} submodels, "
                f"{built.enriched} enriched from {len(concepts)} concept descriptions",
                f"  model {built.model}, cached at {path}",
                "  search it with: aas find <query> --across",
            ],
        )
        return

    if not args.submodel_id:
        fail("nothing to index",
             "pass a submodel id, --all for the whole repository, "
             "or --list to see what is cached")

    with _client(args) as client:
        submodel = get_submodel(client, args.submodel_id)
        concepts = index_mod.concept_map(client.list_concept_descriptions())

    try:
        built = index_mod.build(
            submodel, concepts, model=args.embedding_model, with_values=args.with_values
        )
    except index_mod.IndexError_ as exc:
        fail(str(exc))
        raise

    path = built.save(home)
    payload = {
        "submodel": built.submodel_id,
        "idShort": built.id_short,
        "nodes": len(built.nodes),
        "enriched": built.enriched,
        "concept_descriptions": len(concepts),
        "model": built.model,
        "with_values": built.with_values,
        "cache": str(path),
    }
    lines = [
        f"indexed {built.id_short or built.submodel_id}: {len(built.nodes)} nodes, "
        f"{built.enriched} enriched from {len(concepts)} concept descriptions",
        f"  model {built.model}"
        f"{', values embedded' if built.with_values else ''}, cached at {path}",
    ]
    if built.enriched == 0 and concepts:
        lines.append(
            "  warning: no element semanticId matched a concept description -- "
            "descriptors fall back to idShort and element description"
        )
    if not concepts:
        lines.append(
            "  warning: the repository served no concept descriptions -- "
            "retrieval will be weaker than it should be"
        )
    out.lines(payload, lines)


def cmd_find(args: argparse.Namespace, out: Out) -> None:
    """Filter, rank, and hand the candidates back for the caller to judge.

    The output shape is the design decision here. An earlier arrangement sent
    the top hits to a *second* model to pick between them, which is the wrong
    place to put the judgement twice over: that model cannot see the task, and
    the thing calling this already is an agent. So the candidates come back
    into its context, with enough around each one to decide -- the meaning, the
    unit, the boundary it sits in, its siblings -- plus what the filter did and
    what would narrow it further. Then the caller either has its answer or runs
    a second, tighter search. One judgement, made by the party that knows why
    the question was asked.
    """
    from . import index as index_mod

    home = _index_home(args)
    target = index_mod.ALL if args.across else args.submodel_id
    if not target:
        fail("name a submodel with --in, or search the whole repository with --across")

    try:
        index = index_mod.Index.load(target, home)
    except index_mod.IndexError_ as exc:
        fail(
            str(exc),
            "aas index --all" if args.across else f"aas index {args.submodel_id}",
        )
        raise

    if args.across:
        # Filter even old mixed caches using current repository metadata, before
        # ranking/deduplication. A cached template must not reappear as a machine.
        from dataclasses import replace
        from .catalog import submodels_in_view

        with _client(args) as client:
            allowed = {sm["id"] for sm in submodels_in_view(
                client.list_submodels(limit=1000), client.list_shells(limit=1000), args.kind
            )}
        selected = [i for i, node in enumerate(index.nodes) if node.submodel in allowed]
        index = replace(index, nodes=[index.nodes[i] for i in selected],
                        vectors=index.vectors[selected])

    where = index_mod.Filter(
        semantic_id=args.semantic_id,
        model_type=args.type,
        value_type=args.value_type,
        unit=args.unit,
        submodel=args.submodel_id if (args.across and args.submodel_id) else None,
        in_template=args.in_template,
        id_short_like=args.id_short_like,
        has_value=True if args.has_value else None,
        under=args.under,
    )
    candidates = index_mod.rows_matching(index, where)
    if not candidates:
        out.lines(
            {"query": args.query, "filter": where.describe(), "candidates": 0, "hits": []},
            [f"no element matches the filter ({where.describe()}); nothing was ranked",
             "relax a constraint, or list what exists with: aas index --list"],
        )
        return

    try:
        query_vector = index_mod.embed([args.query], model=index.model)[0]
    except index_mod.IndexError_ as exc:
        fail(str(exc))
        raise

    hits = index_mod.search(
        index, query_vector, top_k=args.top_k, min_score=args.min_score, where=where
    )
    payload = {
        "query": args.query,
        "scope": index.submodel_id,
        "filter": where.describe(),
        "candidates": len(candidates),
        "corpus": len(index.nodes),
        "hits": [
            {
                "path": h.node.path,
                "submodel": h.node.submodel,
                "submodelIdShort": h.node.submodel_id_short,
                "score": round(h.score, 4),
                "modelType": h.node.model_type,
                "idShort": h.node.id_short,
                "preferredName": h.node.preferred_name or None,
                "semanticId": h.node.semantic_id,
                "unit": h.node.unit or None,
                "definition": h.node.definition or h.node.description or None,
                "value": h.node.value,
                "parent": h.parent.path if h.parent else None,
                "siblings": h.siblings,
            }
            for h in hits
        ],
    }
    payload["narrowing"] = _narrowing(index, candidates)

    scope = f"{len(candidates)} of {len(index.nodes)} elements"
    if not hits:
        out.lines(payload, [f"no match for {args.query!r} above {args.min_score} "
                            f"among {scope}"])
        return

    lines: List[str] = [
        f"{len(hits)} hits for {args.query!r}  [searched {scope}"
        + (f", filter: {where.describe()}" if not where.empty() else "")
        + "]"
    ]
    for hit in hits:
        block = hit.block()
        if args.across and hit.node.submodel:
            # Which shell it came from is the first thing that matters in a
            # repository-wide result and is absent from the path.
            block.insert(1, f"        in:       {hit.node.submodel_id_short or ''} "
                            f"({hit.node.submodel})")
        lines.extend(block)

    # The affordance the caller needs to run a *second*, tighter search rather
    # than accept a bad first answer. Printed always, because the decision to
    # stop is exactly the one being handed back.
    lines.append("")
    lines.append(_narrowing_advice(payload["narrowing"], len(candidates)))
    out.lines(payload, lines)


def _narrowing(index: "Any", rows: List[int]) -> Dict[str, Any]:
    """What the surviving candidates have in common, as filters worth trying."""
    from collections import Counter

    nodes = [index.nodes[i] for i in rows]
    def top(attr: str, limit: int = 4) -> List[List[Any]]:
        counts = Counter(getattr(n, attr) for n in nodes if getattr(n, attr))
        return [[value, count] for value, count in counts.most_common(limit)]

    return {
        "modelType": top("model_type"),
        "unit": top("unit"),
        "submodel": top("submodel_id_short"),
        "valueType": top("value_type"),
    }


def _narrowing_advice(narrowing: Dict[str, Any], candidates: int) -> str:
    if candidates <= 12:
        return f"[{candidates} candidates -- the whole set is small; ask for --top-k to see it]"
    parts: List[str] = []
    for flag, key in (("--type", "modelType"), ("--unit", "unit"),
                      ("--in <submodel-id>", "submodel")):
        values = narrowing.get(key) or []
        if len(values) > 1:
            shown = ", ".join(f"{v}({n})" for v, n in values[:3])
            parts.append(f"{flag}: {shown}")
    if not parts:
        return f"[{candidates} candidates searched]"
    return f"[{candidates} candidates searched. Narrow and search again with " + \
           "; ".join(parts) + "]"


# ---------------------------------------------------------------------------
# matching and population
# ---------------------------------------------------------------------------

def cmd_match(args: argparse.Namespace, out: Out) -> None:
    """Match extracted entities to elements and emit a plan. Writes nothing."""
    from . import index as index_mod
    from . import match as match_mod

    try:
        entities = match_mod.load_entities(
            json.loads(Path(args.entities).read_text(encoding="utf-8"))
        )
    except FileNotFoundError:
        fail(f"no entities file at {args.entities}")
        raise
    except (json.JSONDecodeError, match_mod.MatchError) as exc:
        fail(str(exc))
        raise
    if not entities:
        fail("the entities file is empty")

    try:
        index = index_mod.Index.load(args.submodel_id, _index_home(args))
        vectors = index_mod.embed([e.query() for e in entities], model=index.model)
    except index_mod.IndexError_ as exc:
        fail(str(exc))
        raise

    arbitrate = host = None
    model_name: Optional[str] = None
    if not args.no_arbitration:
        from ..kernel.providers import ModelConfig, build

        config = ModelConfig.from_env(args.provider)
        if args.model:
            config.model = args.model
        try:
            provider = build(config)
        except RuntimeError as exc:
            fail(
                str(exc),
                "or pass --no-arbitration to report the middle band unresolved "
                "instead of deciding it",
            )
            raise
        model_name = config.model
        arbitrate = match_mod.make_arbitrator(provider, config, match_mod.ARBITRATE_PROMPT)
        if not args.no_create:
            # Not `make_arbitrator`: hosting creates an element, so it has to
            # say what kind, and it does so in the round trip it was already
            # making. See `match.make_host_chooser`.
            host = match_mod.make_host_chooser(provider, config, match_mod.HOST_PROMPT)

    plan = match_mod.Plan(
        submodel_id=index.submodel_id,
        theta_high=args.theta_high,
        theta_low=args.theta_low,
        model=model_name,
    )
    # Each entity in the middle band costs one model round trip, and the bands
    # are decided per entity with nothing shared between them. On a 266-entity
    # datasheet 254 landed in that band: sequentially that is a quarter of an
    # hour, which is what made the wizard look hung. Concurrent, it is bounded
    # by the endpoint rather than by the count.
    #
    # `pool.map` keeps input order, which matters: `Plan.decisions` is read
    # back by the browser as a table and by `aas populate` as a work list, and
    # a plan whose row order depended on which call returned first would not be
    # comparable between two runs of the same input.
    from concurrent.futures import ThreadPoolExecutor

    def one(pair: Any) -> Any:
        entity, vector = pair
        return match_mod.decide(
            entity,
            match_mod.candidates_for(index, vector, args.top_k),
            theta_high=args.theta_high,
            theta_low=args.theta_low,
            arbitrate=arbitrate,
            host=host,
            create_in=args.create_in,
        )

    pairs = list(zip(entities, vectors))
    if arbitrate is None:
        # No round trips to overlap; a pool would only add threads.
        plan.decisions.extend(one(pair) for pair in pairs)
    else:
        workers = min(len(pairs), int(os.getenv("A4S_MATCH_WORKERS", "8")))
        print(f"  arbitrating {len(pairs)} entities, {workers} at a time",
              file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            plan.decisions.extend(pool.map(one, pairs))

    payload = plan.to_json()
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = plan.counts()
    payload["calibration_note"] = plan.calibration_note()
    lines = [
        f"{len(plan.decisions)} entities against {index.id_short or index.submodel_id} "
        f"(theta_high={args.theta_high}, theta_low={args.theta_low})",
        f"  {counts['write']} write, {counts['create']} create, {counts['skip']} skip; "
        f"{counts['auto']} auto, {counts['arbitrated']} arbitrated, "
        f"{counts['warnings']} warnings",
        "",
        *[d.line() for d in plan.decisions[:MAX_LISTED]],
    ]
    if payload["calibration_note"]:
        lines += ["", f"note: {payload['calibration_note']}"]
    if args.out:
        lines += ["", f"plan written to {args.out}", "review it, then: aas populate --plan " + args.out]
    out.lines(payload, lines)


def cmd_populate(args: argparse.Namespace, out: Out) -> None:
    """Apply a match plan. Every write is one the plan named."""
    from . import match as match_mod

    try:
        plan = match_mod.Plan.from_json(
            json.loads(Path(args.plan).read_text(encoding="utf-8"))
        )
    except FileNotFoundError:
        fail(f"no plan at {args.plan}", "produce one with: aas match --entities ... --out ...")
        raise
    except (json.JSONDecodeError, match_mod.MatchError) as exc:
        fail(str(exc))
        raise

    todo = [d for d in plan.decisions if d.action in ("write", "create")]
    if args.skip_warnings:
        todo = [d for d in todo if not d.warnings]
    if args.max_band == "auto":
        todo = [d for d in todo if d.band == "auto"]

    applied: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for decision in todo:
        if decision.value is None:
            failures.append({"entity": decision.entity, "error": "no value to write"})
            continue
        if args.dry_run:
            applied.append(
                {"entity": decision.entity, "action": decision.action, "path": decision.path}
            )
            continue
        try:
            with _client(args) as client:
                if decision.action == "write":
                    client.set_value(
                        plan.submodel_id,
                        _path(decision.path),
                        match_mod.value_for_write(decision),
                    )
                else:
                    client.add_element(
                        plan.submodel_id,
                        _path(decision.path),
                        match_mod.element_for_create(decision),
                    )
        except (AasHttpError, ModelError, PathError, match_mod.MatchError) as exc:
            failures.append({"entity": decision.entity, "path": decision.path, "error": str(exc)})
            continue
        applied.append(
            {
                "entity": decision.entity,
                "action": decision.action,
                "path": decision.path,
                "value": decision.value,
                "band": decision.band,
                "source": decision.source,
            }
        )

    payload = {
        "submodel": plan.submodel_id,
        "dry_run": args.dry_run,
        "applied": applied,
        "failed": failures,
        "not_attempted": len(plan.decisions) - len(todo),
    }
    verb = "would apply" if args.dry_run else "applied"
    lines = [
        f"{verb} {len(applied)} of {len(plan.decisions)} decisions "
        f"({len(failures)} failed, {payload['not_attempted']} not attempted)"
    ]
    lines += [f"  {a['action']:<6} {a['path']}" for a in applied[:MAX_LISTED]]
    lines += [f"  FAILED {f.get('path') or f['entity']}: {f['error'][:120]}" for f in failures]
    out.lines(payload, lines)


def cmd_validate(args: argparse.Namespace, out: Out) -> None:
    """Check a submodel against the constraints the strict readers enforce.

    Motivated by a concrete, recurring failure: BaSyx-Java accepts submodels
    that basyx-python-sdk then refuses to deserialise. Rather than
    sanitising silently on read, the violations are reported as findings the
    agent can fix with ordinary ``aas set`` / ``aas el`` commands.
    """
    import re

    with _client(args) as client:
        submodel = get_submodel(client, args.submodel_id)

    id_short_re = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
    findings: List[Dict[str, str]] = []
    is_template = submodel.get("kind") == "Template"

    def check(node: Dict[str, Any], prefix: str, in_list: bool) -> None:
        model_type = node.get("modelType")
        id_short = node.get("idShort")

        # AASd-129 is the one an instantiation gets wrong: copying a template
        # verbatim and flipping `kind` leaves TemplateQualifiers behind on an
        # Instance, where the spec does not allow them.
        if not is_template:
            for qual in node.get("qualifiers") or []:
                if qual.get("kind") == "TemplateQualifier":
                    findings.append(
                        {
                            "path": prefix,
                            "rule": "AASd-129",
                            "detail": (
                                f"qualifier {qual.get('type')!r} has kind=TemplateQualifier "
                                "but the submodel is an Instance"
                            ),
                        }
                    )

        if in_list and id_short:
            findings.append(
                {
                    "path": prefix,
                    "rule": "AASd-120",
                    "detail": "SubmodelElementList child must not carry an idShort",
                }
            )
        if not in_list and model_type != "Submodel":
            if not id_short:
                findings.append({"path": prefix, "rule": "AASd-117", "detail": "missing idShort"})
            elif not id_short_re.match(id_short):
                findings.append(
                    {
                        "path": prefix,
                        "rule": "AASd-002",
                        "detail": f"idShort {id_short!r} is not [A-Za-z][A-Za-z0-9_]*",
                    }
                )
        if model_type == "Property" and node.get("valueType") not in VALUE_TYPES:
            findings.append(
                {
                    "path": prefix,
                    "rule": "AASd-007",
                    "detail": f"valueType {node.get('valueType')!r} is not an xsd type",
                }
            )
        if model_type == "SubmodelElementList":
            declared = node.get("typeValueListElement")
            for index, child in enumerate(node.get("value") or []):
                if child.get("modelType") != declared:
                    findings.append(
                        {
                            "path": f"{prefix}[{index}]",
                            "rule": "AASd-108",
                            "detail": f"child is {child.get('modelType')}, list declares {declared}",
                        }
                    )

        is_list = model_type == "SubmodelElementList"

        # AASd-022: idShort unique among siblings. Checked here rather than
        # trusted to the repository because BaSyx does not enforce it, and the
        # damage it allows is invisible from every other angle -- the duplicate
        # is present in the submodel, absent from every path-addressed read,
        # and reported by nothing. An SML child is exempt: it has no idShort.
        if not is_list:
            counts: Dict[str, int] = {}
            for child in _children(node):
                name = child.get("idShort")
                if name:
                    counts[name] = counts.get(name, 0) + 1
            for name, count in counts.items():
                if count > 1:
                    findings.append(
                        {
                            "path": f"{prefix}.{name}" if prefix else name,
                            "rule": "AASd-022",
                            "detail": (
                                f"{count} siblings share idShort {name!r}; all of them "
                                "are unreadable by path and only a whole-submodel "
                                "rewrite can remove one"
                            ),
                        }
                    )

        for index, child in enumerate(_children(node)):
            step = f"[{index}]" if is_list else child.get("idShort", f"[{index}]")
            path = f"{prefix}{step}" if is_list else (f"{prefix}.{step}" if prefix else step)
            check(child, path, is_list)

    check(submodel, "", False)
    out.lines(
        {"submodel": args.submodel_id, "findings": findings, "ok": not findings},
        [f"{f['rule']}  {f['path'] or '<root>'}: {f['detail']}" for f in findings]
        or ["ok: no constraint violations"],
    )


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aas",
        description="Compose Asset Administration Shells from the metamodel up.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every mutating command accepts --dry-run to print the payload without\n"
            "sending it, and --json to emit machine-readable output for piping.\n\n"
            "Build a submodel from nothing:\n"
            "  aas sm create urn:x:sm:np --id-short Nameplate \\\n"
            "      --semantic-id https://admin-shell.io/zvei/nameplate/2/0/Nameplate\n"
            "  aas el add urn:x:sm:np '' --type Property --id-short ManufacturerName \\\n"
            "      --value Fraunhofer --value-type xs:string\n"
            "  aas el add urn:x:sm:np '' --type SubmodelElementList --id-short Markings \\\n"
            "      --type-value-list-element SubmodelElementCollection\n"
            "  aas el add urn:x:sm:np Markings --type SubmodelElementCollection\n"
            "  aas set urn:x:sm:np 'Markings[0].MarkingName' CE\n"
        ),
    )
    parser.add_argument("--endpoint", help="AAS repository base URL (else $A4S_AAS_ENDPOINT)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = parser.add_subparsers(dest="group", required=True)

    def mutating(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--dry-run", action="store_true", help="print the payload, send nothing")
        return p

    # -- shell ---------------------------------------------------------
    shell = sub.add_parser("shell", help="Asset Administration Shells").add_subparsers(
        dest="cmd", required=True
    )

    p = shell.add_parser("list", help="list operational asset instances by default")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--asset-kind", choices=("Instance", "Type", "NotApplicable", "all"),
                   default="Instance", help="Type includes configured application blueprints")
    p.set_defaults(func=cmd_shell_list)

    p = shell.add_parser("show", help="show one shell and its submodel refs")
    p.add_argument("shell_id")
    p.set_defaults(func=cmd_shell_show)

    p = mutating(shell.add_parser("create", help="create an empty shell"))
    p.add_argument("shell_id")
    p.add_argument("--id-short", required=True)
    p.add_argument("--global-asset-id")
    p.add_argument("--asset-kind", default="Instance", choices=("Instance", "Type", "NotApplicable"))
    p.add_argument("--description")
    p.set_defaults(func=cmd_shell_create)

    p = shell.add_parser("delete", help="delete a shell")
    p.add_argument("shell_id")
    p.set_defaults(func=cmd_shell_delete)

    p = mutating(shell.add_parser("attach", help="reference a submodel from a shell"))
    p.add_argument("shell_id")
    p.add_argument("submodel_id")
    p.set_defaults(func=cmd_shell_attach)

    p = mutating(shell.add_parser(
        "detach", help="remove a submodel reference, keeping the submodel"))
    p.add_argument("shell_id")
    p.add_argument("submodel_id")
    p.set_defaults(func=cmd_shell_detach)

    # -- sm ------------------------------------------------------------
    sm = sub.add_parser("sm", help="submodels").add_subparsers(dest="cmd", required=True)

    p = sm.add_parser("list", help="list instance submodels by default")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--kind", choices=("Instance", "Template", "all"), default="Instance")
    p.set_defaults(func=cmd_sm_list)

    p = mutating(sm.add_parser("create", help="create an empty submodel"))
    p.add_argument("submodel_id")
    p.add_argument("--id-short", required=True)
    p.add_argument("--semantic-id", help="the SMT this submodel instantiates")
    p.add_argument("--kind", default="Instance", choices=KIND_VALUES)
    p.add_argument("--attach-to", help="shell id to reference it from immediately")
    p.add_argument("--description")
    p.set_defaults(func=cmd_sm_create)

    p = sm.add_parser("delete", help="delete a submodel")
    p.add_argument("submodel_id")
    p.set_defaults(func=cmd_sm_delete)

    p = mutating(sm.add_parser(
        "repair", help="find and remove duplicate idShorts (AASd-022)"))
    p.add_argument("submodel_id")
    p.add_argument("--apply", action="store_true",
                   help="actually rewrite the submodel; without it, only report")
    p.set_defaults(func=cmd_sm_repair)

    p = sm.add_parser("tree", help="print the element structure (paths, not values)")
    p.add_argument("submodel_id")
    p.add_argument("--depth", type=int, default=3)
    p.set_defaults(func=cmd_sm_tree)

    # -- el ------------------------------------------------------------
    el = sub.add_parser("el", help="submodel elements").add_subparsers(dest="cmd", required=True)

    p = mutating(el.add_parser("add", help="add any metamodel element at any path"))
    p.add_argument("submodel_id")
    p.add_argument("parent", help="idShortPath of the parent; '' for the submodel root")
    p.add_argument("--type", required=True, choices=ELEMENT_TYPES, help="modelType to construct")
    p.add_argument("--id-short", help="omit only for a SubmodelElementList child")
    p.add_argument("--value")
    p.add_argument("--value-type", default="xs:string", choices=VALUE_TYPES)
    p.add_argument("--semantic-id")
    p.add_argument("--description")
    p.add_argument("--qualifier", action="append", metavar="TYPE=VALUE")
    p.add_argument("--delegate-to", metavar="URL", help="shorthand for an invocationDelegation qualifier")
    p.add_argument("--min", help="Range only")
    p.add_argument("--max", help="Range only")
    p.add_argument("--content-type", help="File/Blob only")
    p.add_argument("--type-value-list-element", choices=ELEMENT_TYPES, help="SubmodelElementList only")
    p.add_argument("--value-type-list-element", choices=VALUE_TYPES, help="SubmodelElementList only")
    p.add_argument("--first", help="RelationshipElement only")
    p.add_argument("--second", help="RelationshipElement only")
    p.add_argument("--global-asset-id", help="Entity only")
    p.add_argument("--input", action="append", metavar="NAME=xs:type", help="Operation input variable")
    p.add_argument("--output", action="append", metavar="NAME=xs:type", help="Operation output variable")
    p.set_defaults(func=cmd_el_add)

    p = el.add_parser("ls", help="list children at a path")
    p.add_argument("submodel_id")
    p.add_argument("path", nargs="?", default="")
    p.add_argument("--depth", type=int, default=1)
    p.set_defaults(func=cmd_el_ls)

    p = el.add_parser("show", help="show one element with its semantics")
    p.add_argument("submodel_id")
    p.add_argument("path")
    p.set_defaults(func=cmd_el_show)

    p = el.add_parser("rm", help="delete an element")
    p.add_argument("submodel_id")
    p.add_argument("path")
    p.set_defaults(func=cmd_el_rm)

    # -- values --------------------------------------------------------
    p = sub.add_parser("get", help="read an element value")
    p.add_argument("submodel_id")
    p.add_argument("path")
    p.set_defaults(func=cmd_get, group="get")

    p = mutating(sub.add_parser("set", help="write an element value"))
    p.add_argument("submodel_id")
    p.add_argument("path")
    p.add_argument("value")
    p.add_argument("--json-value", action="store_true", help="parse VALUE as JSON")
    p.add_argument(
        "--lang",
        action="store_true",
        help="value is 'text@lang' (or 'a@en|b@de') for a MultiLanguageProperty",
    )
    p.set_defaults(func=cmd_set, group="set")

    # -- op ------------------------------------------------------------
    op = sub.add_parser("op", help="operations").add_subparsers(dest="cmd", required=True)

    p = op.add_parser("ls", help="list operations and their delegation targets")
    p.add_argument("submodel_id")
    p.set_defaults(func=cmd_op_ls)

    p = mutating(op.add_parser(
        "declare", help="record whether an operation needs approval to invoke"))
    p.add_argument("submodel_id")
    p.add_argument("path")
    p.add_argument("--requires-approval", required=True,
                   choices=["true", "false"],
                   help="true when invoking it acts on real equipment")
    p.set_defaults(func=cmd_op_declare, group="declare")

    p = mutating(op.add_parser("invoke", help="invoke an operation"))
    p.add_argument("submodel_id")
    p.add_argument("path")
    p.add_argument("--arg", action="append", metavar="NAME=VALUE")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--local", action="store_true", help="use the BaSyx invoke endpoint, ignore delegation")
    p.set_defaults(func=cmd_op_invoke)

    # -- opc -----------------------------------------------------------
    # A transport, not a second modelling surface. `aas opc` is for reaching a
    # server the AAS has told you about -- the endpoint and node come out of
    # `UAServerDataSheet` and the `valueSource` qualifiers, not out of a prompt.
    opc_cmd = sub.add_parser(
        "opc", help="OPC UA servers named by the AAS"
    ).add_subparsers(dest="cmd", required=True)

    p = opc_cmd.add_parser(
        "ls", help="elements whose value comes from an OPC UA node")
    p.add_argument("submodel_id")
    p.set_defaults(func=cmd_opc_ls)

    p = opc_cmd.add_parser("info", help="what the server says about itself")
    p.add_argument("endpoint")
    p.add_argument("--timeout", type=float, default=opc.DEFAULT_TIMEOUT)
    p.set_defaults(func=cmd_opc_info)

    p = opc_cmd.add_parser("read", help="read one or more nodes")
    p.add_argument("endpoint")
    p.add_argument("node", nargs="+")
    p.add_argument("--timeout", type=float, default=opc.DEFAULT_TIMEOUT)
    p.set_defaults(func=cmd_opc_read)

    p = opc_cmd.add_parser("browse", help="walk the address space")
    p.add_argument("endpoint")
    p.add_argument("--node", help="start here instead of the Objects folder")
    p.add_argument("--depth", type=int, default=opc.DEFAULT_DEPTH)
    p.add_argument("--timeout", type=float, default=opc.DEFAULT_TIMEOUT)
    p.set_defaults(func=cmd_opc_browse)

    p = mutating(opc_cmd.add_parser("write", help="write one node"))
    p.add_argument("endpoint")
    p.add_argument("node")
    p.add_argument("value")
    p.add_argument("--type", help="assert the node's type; refuses if it differs")
    p.add_argument("--timeout", type=float, default=opc.DEFAULT_TIMEOUT)
    p.set_defaults(func=cmd_opc_write, group="opc write")

    p = mutating(opc_cmd.add_parser("call", help="call a method"))
    p.add_argument("endpoint")
    p.add_argument("node")
    p.add_argument("--arg", action="append")
    p.add_argument("--timeout", type=float, default=opc.DEFAULT_TIMEOUT)
    p.set_defaults(func=cmd_opc_call, group="opc call")

    # -- validate ------------------------------------------------------
    # -- smt -----------------------------------------------------------
    smt = sub.add_parser("smt", help="submodel templates").add_subparsers(
        dest="cmd", required=True
    )

    def sourced(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument(
            "--from-endpoint",
            help="repository to read templates from (else $A4S_SMT_ENDPOINT, "
            "else --endpoint). Keep separate from the write target.",
        )
        return p

    p = sourced(smt.add_parser("list", help="list submodel templates"))
    p.add_argument("--of-shell", help="only templates referenced by this shell")
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_smt_list)

    p = sourced(smt.add_parser("show", help="slot outline with cardinalities"))
    p.add_argument("template_id")
    p.set_defaults(func=cmd_smt_show)

    p = sourced(
        mutating(smt.add_parser("instantiate", help="create an Instance submodel from a template"))
    )
    p.add_argument("template_id")
    p.add_argument("--as", dest="as_id", help="id for the new submodel (default: <template>_<idShort>)")
    p.add_argument("--id-short", help="idShort for the new submodel (default: the template's)")
    p.add_argument("--attach-to", metavar="AAS_ID", help="also add a submodel-ref to this shell")
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="also create slots with minimum cardinality 0",
    )
    p.add_argument("--keep-values", action="store_true", help="do not empty data-bearing slots")
    p.add_argument(
        "--seed-examples",
        action="store_true",
        help="write ExampleValue qualifiers into the slots -- demo fixtures only, not evidence",
    )
    p.add_argument(
        "--keep-template-qualifiers",
        action="store_true",
        help="keep authoring qualifiers (violates AASd-129 on an Instance)",
    )
    p.add_argument("--replace", action="store_true", help="overwrite an existing submodel")
    p.add_argument("--force", action="store_true", help="allow a source whose kind is not Template")
    p.set_defaults(func=cmd_smt_instantiate)

    p = mutating(smt.add_parser(
        "author", help="build a kind=Template submodel from a structure file"))
    p.add_argument("submodel_id")
    p.add_argument("--structure", required=True, metavar="FILE",
                   help="a proposal from `a4s extract --as-template`")
    p.add_argument("--id-short", help="override the idShort the structure proposes")
    p.add_argument("--semantic-id", help="the concept this template stands for")
    p.add_argument("--attach-to", metavar="AAS_ID", help="also reference it from this shell")
    p.add_argument("--replace", action="store_true", help="overwrite an existing submodel")
    p.add_argument("--force", action="store_true",
                   help="build it even though the structure has known problems")
    p.set_defaults(func=cmd_smt_author)

    p = sourced(smt.add_parser("diff", help="which of a template's slots an instance has not filled"))
    p.add_argument("template_id")
    p.add_argument("submodel_id")
    p.add_argument(
        "--include-optional", action="store_true", help="report unfilled optional slots too"
    )
    p.set_defaults(func=cmd_smt_diff)

    # -- retrieval -----------------------------------------------------
    p = sub.add_parser("index", help="build the retrieval index for a submodel (slow, cached)")
    p.add_argument("submodel_id", nargs="?")
    p.add_argument("--all", action="store_true",
                   help="index repository instances; opt into templates with --kind")
    p.add_argument("--kind", choices=("Instance", "Template", "all"), default="Instance")
    p.add_argument("--limit", type=int, default=500,
                   help="with --all: how many submodels to read")
    p.add_argument("--list", action="store_true", help="show cached indexes instead of building")
    p.add_argument("--drop", action="store_true", help="delete the cached index for this submodel")
    p.add_argument("--embedding-model", help="embedding deployment (else $A4S_EMBEDDING_MODEL)")
    p.add_argument(
        "--with-values",
        action="store_true",
        help="embed stored values too, to search content rather than slots "
        "(list items always include theirs)",
    )
    p.add_argument("--index-home", help="cache directory (else $A4S_INDEX_HOME)")
    p.set_defaults(func=cmd_index, group="index")

    p = sub.add_parser(
        "find",
        help="filter structurally, then rank by meaning; returns candidates to judge",
    )
    p.add_argument("query")
    p.add_argument("--in", dest="submodel_id", metavar="SUBMODEL_ID",
                   help="one submodel. With --across, restricts the corpus to it instead")
    p.add_argument("--across", action="store_true",
                   help="search the whole repository (needs `aas index --all`)")
    p.add_argument("--kind", choices=("Instance", "Template", "all"), default="Instance",
                   help="with --across: filter current repository kind before ranking")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--min-score", type=float, default=0.0, help="drop hits below this cosine similarity"
    )
    # Structural filters. Applied before the similarity is computed, not as a
    # discard afterwards: an exact semanticId narrows 3,969 elements to a
    # median of 3, and ranking is far better at ordering 3 candidates than at
    # winning a contest against 3,966 irrelevant ones.
    p.add_argument("--semantic-id", help="exact concept id; the strongest filter there is")
    p.add_argument("--type", choices=ELEMENT_TYPES, help="modelType")
    p.add_argument("--value-type", choices=VALUE_TYPES)
    p.add_argument("--unit", help="substring of the declared unit, e.g. kWh")
    p.add_argument("--in-template", metavar="SEMANTIC_ID",
                   help="only elements in submodels instantiating this template")
    p.add_argument("--id-short-like", metavar="TEXT",
                   help="substring of the idShort, case-insensitive -- the grep-shaped filter")
    p.add_argument("--under", metavar="PATH", help="only below this idShortPath")
    p.add_argument("--has-value", action="store_true", help="only elements that hold a value")
    p.add_argument("--index-home", help="cache directory (else $A4S_INDEX_HOME)")
    p.set_defaults(func=cmd_find, group="find")

    # -- matching ------------------------------------------------------
    p = sub.add_parser("match", help="match extracted entities to elements, emit a plan")
    p.add_argument("--entities", required=True, metavar="FILE", help="extracted entities, JSON")
    p.add_argument("--into", dest="submodel_id", required=True, metavar="SUBMODEL_ID")
    p.add_argument("--out", metavar="FILE", help="write the plan here")
    p.add_argument("--theta-high", type=float, default=0.8, help="auto-accept at or above (0.8)")
    p.add_argument("--theta-low", type=float, default=0.5, help="auto-reject below (0.5)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--create-in",
        metavar="PATH",
        help="collection to create unmatched entities in when no host is chosen",
    )
    p.add_argument(
        "--no-create",
        action="store_true",
        help="never propose creating an element; unmatched entities are reported as skipped",
    )
    p.add_argument(
        "--no-arbitration",
        action="store_true",
        help="do not call a model for the middle band; report it unresolved instead",
    )
    p.add_argument("--provider", choices=("azure", "anthropic", "openai"))
    p.add_argument("--model", help="arbitration model (else $A4S_MODEL)")
    p.add_argument("--index-home")
    p.set_defaults(func=cmd_match, group="match")

    p = mutating(sub.add_parser("populate", help="apply a match plan"))
    p.add_argument("--plan", required=True, metavar="FILE")
    p.add_argument(
        "--max-band",
        choices=("auto", "any"),
        default="any",
        help="'auto' applies only decisions retrieval made on its own",
    )
    p.add_argument(
        "--skip-warnings", action="store_true", help="do not apply decisions carrying a warning"
    )
    p.set_defaults(func=cmd_populate, group="populate")

    p = sub.add_parser("validate", help="report metamodel constraint violations")
    p.add_argument("submodel_id")
    p.set_defaults(func=cmd_validate, group="validate")

    return parser


def command_synopsis() -> List[str]:
    """One line per verb, derived from the parser rather than written beside it.

    The agent otherwise spends its first several steps running ``--help`` to
    discover a surface that is already known here -- and in a comparison
    against per-tool JSON schemas that discovery cost is an artefact of the
    harness, not of the approach. Generating it from the parser is what keeps
    the prompt from drifting away from the program.
    """
    lines: List[str] = []

    def subparsers(parser: argparse.ArgumentParser):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                helps = {a.dest: a.help for a in action._choices_actions}
                for name, sub in action.choices.items():
                    yield name, sub, helps.get(name, "")

    def emit(parser: argparse.ArgumentParser, prefix: str, help_text: str) -> None:
        children = list(subparsers(parser))
        if children:
            for name, sub, sub_help in children:
                emit(sub, f"{prefix} {name}", sub_help)
            return
        positionals = " ".join(
            f"<{a.metavar or a.dest}>"
            for a in parser._actions
            if not a.option_strings and a.dest != "help"
        )
        options = " ".join(
            a.option_strings[-1] + ("" if a.nargs == 0 else f" <{a.metavar or a.dest}>")
            for a in parser._actions
            if a.option_strings and a.dest not in ("help", "dry_run")
        )
        lines.append(f"  {f'{prefix} {positionals}'.strip():<52} {help_text}")
        if options:
            lines.append(f"      [{options}]")

    for name, sub, help_text in subparsers(build_parser()):
        emit(sub, f"aas {name}", help_text)
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Imported here so `aas` stays usable with only httpx installed.
    from ..kernel.config import load_env

    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Out(args.json)
    try:
        args.func(args, out)
    except AasHttpError as exc:
        fail(str(exc))
    except (ModelError, PathError) as exc:
        fail(str(exc))
    except httpx_connect_errors() as exc:  # type: ignore[misc]
        fail(
            f"cannot reach the AAS repository: {type(exc).__name__}: {exc}",
            "check --endpoint / $A4S_AAS_ENDPOINT and that BaSyx is up "
            "(docker ps; curl $A4S_AAS_ENDPOINT/shells?limit=1)",
        )
    return 0


def httpx_connect_errors() -> tuple:
    """Every transport-level failure, not the three that were thought of first.

    The original list named ConnectError, ConnectTimeout and ReadTimeout, and a
    BaSyx that accepted a connection then dropped it raised RemoteProtocolError
    -- which fell straight through to a Python traceback. An agent reads these
    strings: a traceback tells it nothing it can act on and costs it a step,
    where "cannot reach the repository" tells it exactly what to check.
    `TransportError` is the base of the whole family, so nothing new in httpx
    can slip past it.
    """
    import httpx

    return (httpx.TransportError,)


if __name__ == "__main__":
    sys.exit(main())

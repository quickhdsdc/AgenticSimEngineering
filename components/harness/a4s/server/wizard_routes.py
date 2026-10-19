"""The DPP wizard's four stages: upload, extraction, matching, generation.

**The wizard is agent-driven, and these routes are not the pipeline.** Reading
`App.vue` makes that plain: every stage ends by sending a prompt to the chat
socket, and the agent does the work -- extracts the datasheet, matches the
entities, writes the values. What the browser calls over HTTP is only the file
store between stages: has the extraction been done, give me its entities, save
what the user corrected, give me the match plan. The stages do not share a
process, so they share files.

**Two vocabularies meet here, and translating between them is the whole job.**
The frontend's match row is `{extractedEntity, aasProperty, confidence,
matchType}`. The harness's `match.Plan` is `{entity, path, score, band,
action}` plus the value type and the parent for elements that do not exist yet.
The round trip has to survive the browser: a plan is read, rendered, edited by
a person, sent back, and handed to `aas populate`. Anything dropped on the way
out is a value written to the wrong element or not written at all, so the
fields the plan needs travel through the UI as extra keys on each row -- Vue's
object spread preserves them, which is what makes this work without touching
the components.

**The one place a user edit changes the target submodel.** When someone
retargets an entity to a property in a different submodel, the row still
carries the *old* `submodelId`. Trusting it writes into the wrong submodel and
reports success, so the target is re-resolved from the `semanticPath` the
property picker supplied, and a row that cannot be resolved is reported as
unresolved rather than guessed at.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from ..aasctl import match as match_mod
from ..aasctl.client import AasHttpError, BasyxClient
from . import workspace
from .aas_routes import _client, _fail, _resolve_shell
# The ledger writer lives with the other mutations. Imported rather than
# duplicated so a populate and a shell delete land in one session with one
# actor, which is the claim `ledger/log.py` opens with.
from .write_routes import _recorded

router = APIRouter()

# Containers. Selecting one in the property picker does not mean "write here",
# it means "make a new element in here" -- the create path in `aas populate`,
# and the reason the matching stage lets a collection be chosen at all.
_CONTAINERS = ("SubmodelElementCollection", "SubmodelElementList", "Entity")

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def _bad_name(exc: workspace.WorkspaceError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# stage 1 -- input
# ---------------------------------------------------------------------------

@router.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Store what the browser sent and return the path the agent will read.

    `path` is an absolute filesystem path, and that is deliberate rather than
    sloppy: the frontend puts it straight into the prompt it sends the agent,
    and the agent runs on this host. A URL would have to be fetched back
    through this same server for no gain.

    The size limit is enforced while streaming, not from the declared
    `Content-Length`, because that header is whatever the client wrote in it.
    """
    try:
        name = workspace.safe_name(file.filename or "")
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    target = workspace.under(workspace.uploads(), name)
    size = 0
    try:
        with target.open("wb") as sink:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    sink.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"{name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                sink.write(chunk)
    except HTTPException:
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"cannot store {name}: {exc}") from exc

    return {
        "success": True,
        "filename": name,
        "path": str(target),
        "size": size,
    }


@router.get("/api/preview-structured-file/{filename}")
def preview_structured_file(filename: str, rows: int = 5) -> Dict[str, Any]:
    """The first few rows of an uploaded table, for the prompt the agent gets.

    The wizard shows the model a sample so it can decide which column is the
    entity and which is the value. Read through `extract.read_table` -- the
    same reader the extractor itself uses -- so the preview cannot disagree
    with what the extraction later sees.
    """
    try:
        path = workspace.under(workspace.uploads(), filename)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no uploaded file named {filename!r}")

    from ..extract import read_table

    try:
        table = read_table(path)
    except Exception as exc:  # noqa: BLE001 - the reason belongs on screen
        raise HTTPException(
            status_code=422, detail=f"cannot read {filename} as a table: {exc}"
        ) from exc

    headers = list(table[0].keys()) if table else []
    return {
        "success": True,
        "filename": filename,
        "headers": headers,
        "rows": table[:rows],
        "rowCount": len(table),
    }


@router.get("/api/check-extraction/{filename}")
def check_extraction(filename: str) -> Dict[str, Any]:
    """Whether this document has already been extracted.

    The wizard branches on the answer, so a missing extraction is a 200 with
    `exists: false`, never a 404 -- the same reasoning as `/api/check-aas`.

    The edited file wins when both exist. A person corrected those entities;
    handing back the extractor's originals would discard the correction and
    look like the save had failed.
    """
    try:
        edited = workspace.extraction_file(filename, edited=True)
        original = workspace.extraction_file(filename)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    for candidate in (edited, original):
        if candidate.exists():
            return {"exists": True, "filename": candidate.name,
                    "edited": candidate == edited,
                    "expectedPath": str(original)}
    # `expectedPath` is here for the miss, not the hit. The wizard puts it into
    # the prompt it sends the agent -- "write the entities to this file" -- and
    # without it the agent would choose a name and `check-extraction` would go
    # on answering no forever. The convention lives in one place; the frontend
    # is told the answer rather than reimplementing it.
    return {"exists": False, "filename": None, "edited": False,
            "expectedPath": str(original)}


# ---------------------------------------------------------------------------
# stage 2 -- extraction
# ---------------------------------------------------------------------------

@router.get("/api/extracted-entities/{filename}")
def extracted_entities(filename: str) -> Dict[str, Any]:
    """The entities in one extraction file, for the review table.

    Accepts both shapes the extractor emits: a bare JSON array, and an object
    with an `entities` list. `aas match` accepts both too (`match.load_entities`),
    and a route that were stricter than the tool it feeds would reject files
    the pipeline can actually use.
    """
    entities, meta = _read_extraction(filename)
    return {
        "success": True,
        "entities": entities,
        "count": len(entities),
        # What the document said about itself. The table needs both: a
        # configuration picker cannot be offered without the first, and a
        # dropped value cannot be reported without the second.
        "configurations": meta.get("configurations") or [],
        "warnings": meta.get("warnings") or [],
    }


@router.post("/api/save-edited-entities")
def save_edited_entities(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Write the corrected entities beside the originals, never over them.

    `_edit.json` rather than in place: the extractor's output is evidence of
    what the model read out of the document, and the corrections are evidence
    of what a person disagreed with. Overwriting makes the second unreviewable.
    """
    filename = str(body.get("original_filename") or "").strip()
    entities = body.get("entities")
    if not filename or not isinstance(entities, list):
        raise HTTPException(
            status_code=422,
            detail="original_filename and an entities list are both required",
        )

    try:
        target = workspace.extraction_file(filename, edited=True)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    kept = [
        _canonical_entity(e)
        for e in entities
        if isinstance(e, dict) and e.get("selected") is not False
    ]

    # The corrections are written in the same shape the extraction was, so the
    # configurations and the warnings survive the edit. They were the
    # extractor's evidence about the document, not about the rows a person
    # happened to keep, and dropping them here would make the picker vanish the
    # moment anybody touched the table.
    dropped = len(entities) - len(kept)
    payload: Dict[str, Any] = {
        "entities": kept,
        "configurations": [str(c) for c in (body.get("configurations") or [])],
        "warnings": [str(w) for w in (body.get("warnings") or [])],
    }
    chosen = str(body.get("configuration") or "").strip()
    if chosen:
        payload["configuration"] = chosen
        payload["warnings"].append(
            f"{dropped} of {len(entities)} extracted values were not kept; "
            f"the configuration chosen for this asset is {chosen!r}."
        )
    _write_json(target, payload)

    return {
        "success": True,
        "filename": target.name,
        "path": str(target),
        "count": len(kept),
        "dropped": dropped,
        "configuration": chosen or None,
    }


# ---------------------------------------------------------------------------
# stage 3 -- matching
# ---------------------------------------------------------------------------

# Below this many addressable elements, a submodel cannot be a match target:
# every entity resolves to the same handful of paths and the plan is a list of
# overwrites. Measured rather than chosen -- `Capability` holds 5 and took 197
# decisions onto one element.
MIN_TARGET_ELEMENTS = 12

# Element types with no scalar `$value`. The list lives in `match` now, where
# `value_for_write` enforces it for both apply paths; this name is kept because
# the route refuses these earlier, with a message naming the row and the
# submodel, and a plan is a file -- one written before the matcher learned to
# exclude these is still a valid file and still reachable.
NOT_WRITABLE = match_mod.NOT_WRITABLE


def _walk_count(client: Any, submodel_id: str) -> List[Any]:
    from ..aasctl.index import walk

    return walk(client.get_submodel(submodel_id))


@router.post("/api/match")
def run_match(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Match one document's entities against one shell, and return the plan.

    **This is a button, so it is a function.** The wizard used to send the
    agent a prompt naming the exact commands to run and then poll for a file at
    an exact path -- which is a deterministic script executed by a
    non-deterministic runner, and it has both halves' drawbacks. It failed the
    way that design fails: the agent did the work correctly across all four
    submodels and never wrote the manifest the wizard was polling for, so a
    completed job showed as "no matches found yet" forever.

    A wizard stage is a structured workflow with defined inputs and outputs,
    which is the case for a deterministic chain. The agent is still there, on
    the socket, for what it is actually better at -- "why did this not match",
    "put the clamping force under BasicData instead" -- and it can answer those
    because the plan is a file it can read, carrying every score, band, reason
    and rejected candidate. That is a better basis for an answer than having
    performed the match and remembering it.

    Every submodel of the shell is matched, and the manifest is written here
    rather than asked for.
    """
    filename = str(body.get("filename") or "").strip()
    aas = str(body.get("aas_id_short") or body.get("targetAAS") or "").strip()
    if not filename or not aas:
        raise HTTPException(
            status_code=422,
            detail="filename and aas_id_short are both required",
        )

    try:
        entities_path = workspace.extraction_file(filename, edited=True)
        if not entities_path.exists():
            entities_path = workspace.extraction_file(filename)
        plan_path = workspace.match_file(filename, aas)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    if not entities_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no extracted entities for {filename}; run extraction first",
        )
    try:
        entities = match_mod.load_entities(_read_json(entities_path))
    except match_mod.MatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not entities:
        raise HTTPException(status_code=422,
                            detail=f"{entities_path.name} holds no entities")

    endpoint = body.get("endpoint")
    wanted = body.get("submodel_ids") or body.get("submodel_id")
    if isinstance(wanted, str):
        wanted = [wanted]
    skipped: List[Dict[str, str]] = []
    with _client(endpoint) as client:
        shell = _resolve_shell(client, aas)
        submodel_ids = [
            key.get("value")
            for ref in client.list_submodel_refs(shell["id"])
            for key in (ref.get("keys") or [])
            if key.get("value")
        ]
    if not submodel_ids:
        raise HTTPException(
            status_code=422,
            detail=f"{aas} has no submodels to match into",
        )

    if wanted:
        unknown = [w for w in wanted if w not in submodel_ids]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"{aas} does not reference {', '.join(unknown)}",
            )
        submodel_ids = [s for s in submodel_ids if s in wanted]
    else:
        # **Matching into every submodel manufactures nonsense**, and the first
        # real run showed how much: a datasheet's 317 entities went at the
        # Capability submodel, which holds five elements, and 197 of them
        # landed on `CapabilitySet.InjectionMolding` because it was the nearest
        # thing available. A retrieval index cannot decline; it returns the
        # best of what it has, and the best of five wrong answers is still
        # wrong.
        #
        # Nothing here can tell which submodel a document belongs in -- that is
        # the caller's knowledge, and the wizard has a field for it. Absent
        # that, the ones the datasheet plausibly describes are the ones with
        # enough elements to describe it, so a submodel too small to be a
        # target is skipped and said so rather than filled with near-misses.
        kept = []
        with _client(endpoint) as client:
            for submodel_id in submodel_ids:
                try:
                    count = len(_walk_count(client, submodel_id))
                except Exception:  # noqa: BLE001 - unreadable is not a target
                    skipped.append({"submodel": submodel_id, "reason": "unreadable"})
                    continue
                if count < MIN_TARGET_ELEMENTS:
                    skipped.append({"submodel": submodel_id,
                                    "reason": f"only {count} elements"})
                else:
                    kept.append(submodel_id)
        if kept:
            submodel_ids = kept

    written: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []
    for number, submodel_id in enumerate(submodel_ids):
        try:
            plan = _plan_for(submodel_id, entities, body)
        except Exception as exc:  # noqa: BLE001 - one submodel is not all of them
            failures.append({"submodel": submodel_id,
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        child = plan_path.with_name(f"{plan_path.stem}.{number}{plan_path.suffix}")
        child.write_text(
            json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append({"submodel": submodel_id, "path": str(child)})

    if not written:
        raise HTTPException(
            status_code=502,
            detail="no submodel could be matched: "
                   + "; ".join(f["error"] for f in failures[:3]),
        )

    # The manifest the reader already understands, written rather than
    # requested. `_load_plans` has read this shape since before there was a
    # route to produce it; what was missing was anything that reliably wrote it.
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps({"kind": "match-plan-manifest", "plans": written},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # The *document* name, not the plan's. `matching_results` derives the plan
    # path from the document the same way `match_file` did above, so handing it
    # the plan name appends the suffix twice -- which is how 621 s of completed
    # matching came back as a 404 naming a file with `_matching_result` in it
    # twice. Every plan had been written by then; only the read failed.
    results = matching_results(filename, aas)
    return {
        "success": True,
        "path": str(plan_path),
        "submodels": [w["submodel"] for w in written],
        "skipped": skipped,
        "failures": failures,
        "results": results["results"],
        "count": results["count"],
    }


def _plan_for(
    submodel_id: str, entities: List[Any], body: Dict[str, Any]
) -> match_mod.Plan:
    """One submodel's plan, indexing it first if it has not been indexed.

    The index is built on demand rather than asked of the caller. It is cached
    on disk, so the cost is paid once per submodel and the wizard has no reason
    to know the concept exists.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..aasctl import index as index_mod

    try:
        index = index_mod.Index.load(submodel_id)
    except index_mod.IndexError_:
        with _client(body.get("endpoint")) as client:
            submodel = client.get_submodel(submodel_id)
            concepts = index_mod.concept_map(client.list_concept_descriptions())
        index = index_mod.build(submodel, concepts)
        index.save()

    vectors = index_mod.embed([e.query() for e in entities], model=index.model)

    arbitrate = host = None
    model_name = None
    if not body.get("no_arbitration"):
        from ..kernel.providers import ModelConfig, build as build_provider

        config = ModelConfig.from_env(None)
        provider = build_provider(config)
        model_name = config.model
        arbitrate = match_mod.make_arbitrator(provider, config,
                                              match_mod.ARBITRATE_PROMPT)
        # Chooses the modelType alongside the collection, in the same call.
        host = match_mod.make_host_chooser(provider, config, match_mod.HOST_PROMPT)

    theta_high = float(body.get("theta_high") or 0.8)
    theta_low = float(body.get("theta_low") or 0.5)
    plan = match_mod.Plan(submodel_id=index.submodel_id, theta_high=theta_high,
                          theta_low=theta_low, model=model_name)

    def one(pair: Any) -> Any:
        entity, vector = pair
        return match_mod.decide(
            entity, match_mod.candidates_for(index, vector, 5),
            theta_high=theta_high, theta_low=theta_low,
            arbitrate=arbitrate, host=host, create_in=None,
        )

    pairs = list(zip(entities, vectors))
    if arbitrate is None:
        plan.decisions.extend(one(pair) for pair in pairs)
    else:
        # Same reason as the CLI: one round trip per middle-band entity, and
        # `map` keeps input order because the browser draws these as a table.
        # The same knob the CLI honours. Hardcoding it here meant the one
        # lever on how long this stage takes did not reach the stage that
        # takes longest.
        workers = max(1, int(os.getenv("A4S_MATCH_WORKERS", "8")))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            plan.decisions.extend(pool.map(one, pairs))
    return plan


@router.post("/api/populate")
def run_populate(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Apply a reviewed match plan to the repository.

    **The most deterministic of the four stages, and the one that writes.** By
    the time this runs there is no judgement left: the plan says which value
    goes to which path, and a person has just read those rows in a table and
    pressed a button. Handing that to an agent would be asking a
    non-deterministic runner to execute a script somebody already approved
    line by line.

    No approval gate, consistent with the rest of the write half (open question
    8): the gate recovers an action from a command line and asks a human, and
    here the human has already answered by reviewing the table. This is the
    most-reviewed write in the system rather than the least. Nothing here can
    move a machine either -- `populate` writes property values and never
    invokes an Operation.

    **Every decision is reported individually**, which is where a function has
    to beat the agent it replaces. An agent narrates "applied most of them";
    103 writes with one failure at number 47 needs the number 47.
    """
    filename = str(body.get("filename") or "").strip()
    aas = str(body.get("aas_id_short") or body.get("targetAAS") or "").strip()
    if not filename:
        raise HTTPException(status_code=422, detail="filename is required")

    dry_run = bool(body.get("dry_run"))
    path = _resolve_match_file(filename, aas or None)
    plans = _load_plans(path)

    applied: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    endpoint = body.get("endpoint")

    # Two values for one path is never what anyone asked for: applied in order
    # they overwrite each other and the element ends up holding whichever
    # arrived last, with the other writes charged for and discarded. Found on
    # the first real plan -- 197 of 197 decisions against the Capability
    # submodel pointed at `CapabilitySet.InjectionMolding`, because that
    # submodel holds five elements and 317 datasheet entities were matched
    # against it, so every one landed on the nearest thing available.
    #
    # Refused rather than warned: a silent overwrite is indistinguishable from
    # a successful write in the response, and the caller cannot tell that 196
    # of its values are gone.
    collisions = _collisions(plans)
    if collisions and not body.get("allow_overwrite"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "several decisions write to the same element",
                "collisions": collisions[:10],
                "hint": "narrow the match to one submodel, or drop the extra rows "
                        "in the table. Pass allow_overwrite to apply them in "
                        "order anyway.",
            },
        )

    for plan in plans:
        todo = [d for d in plan.decisions if d.action in ("write", "create")]
        for decision in todo:
            if decision.value is None:
                failures.append({"entity": decision.entity, "path": decision.path,
                                 "submodel": plan.submodel_id,
                                 "error": "no value to write"})
                continue
            # Refused here rather than sent, because BaSyx answers a write onto
            # a Capability or an Operation with a bare HTTP 500 that names no
            # reason -- 295 of 396 decisions failed that way on the first real
            # run, and the response told the user nothing about why. The
            # matcher no longer proposes these, but plans written before that
            # are still on disk and still reachable.
            if decision.action == "write" and decision.model_type in NOT_WRITABLE:
                failures.append({
                    "entity": decision.entity, "path": decision.path,
                    "submodel": plan.submodel_id,
                    "error": f"{decision.path} is a {decision.model_type}, "
                             f"which holds no value. An Operation is invoked, "
                             f"not written to.",
                })
                continue
            if dry_run:
                applied.append({"entity": decision.entity, "action": decision.action,
                                "path": decision.path, "submodel": plan.submodel_id})
                continue
            try:
                _apply_one(plan.submodel_id, decision, endpoint)
            except Exception as exc:  # noqa: BLE001 - one row is not the run
                failures.append({"entity": decision.entity, "path": decision.path,
                                 "submodel": plan.submodel_id,
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            applied.append({"entity": decision.entity, "action": decision.action,
                            "path": decision.path, "value": decision.value,
                            "band": decision.band, "submodel": plan.submodel_id})

    unrecorded = None
    if not dry_run and applied:
        # One entry for the run, not one per write. The per-element provenance
        # is already in `aas.write` events the client emits; this records that
        # a plan was applied, by whom, and how much of it landed.
        unrecorded = _recorded(
            "aas.populate",
            plans[0].submodel_id if plans else "",
            {"plan": path.name, "document": filename, "aas": aas,
             "applied": len(applied), "failed": len(failures),
             "paths": [a["path"] for a in applied[:50]]},
        )

    total = sum(len(plan.decisions) for plan in plans)
    return {
        "success": not failures,
        "dryRun": dry_run,
        "applied": applied,
        "failed": failures,
        "counts": {"applied": len(applied), "failed": len(failures),
                   "notAttempted": total - len(applied) - len(failures),
                   "total": total},
        "plan": str(path),
        "unrecorded": unrecorded,
    }


def _collisions(plans: List[match_mod.Plan]) -> List[Dict[str, Any]]:
    """Elements that more than one decision would land on.

    **A write and a create do not mean the same thing by `path`.** For a write
    it is the element itself; for a create it is the *parent* the new element
    goes into, and fifty-seven creates sharing one parent is an ordinary
    submodel being filled in, not a conflict. Keying both on `path` refused a
    plan where nothing was wrong -- 57 creates into `Configuration`, every one
    of them a different property.

    So a create is keyed on parent plus idShort, which is what it will actually
    occupy. That is also the check worth having: two creates with the same
    idShort under one parent violate AASd-022, and the damage that does is not
    theoretical -- it is the defect §3.3 found, where a duplicate makes *both*
    copies unreachable by path. `el add` refuses one at a time; this refuses
    the plan that would attempt it.

    Keyed within a submodel, because the same idShortPath under two submodels
    is two different elements.
    """
    from collections import defaultdict

    seen: Dict[tuple, List[str]] = defaultdict(list)
    for plan in plans:
        for decision in plan.decisions:
            if decision.action == "write" and decision.path:
                seen[(plan.submodel_id, decision.path, "write")].append(decision.entity)
            elif decision.action == "create" and decision.id_short:
                target = f"{decision.path}.{decision.id_short}" if decision.path                     else decision.id_short
                seen[(plan.submodel_id, target, "create")].append(decision.entity)
    return [
        {"submodel": submodel, "path": path, "action": action,
         "count": len(entities), "entities": entities[:5],
         "why": "two creates with one idShort under one parent break AASd-022, "
                "which makes both unreachable by path"
                if action == "create" else
                "applied in order these overwrite each other"}
        for (submodel, path, action), entities in seen.items()
        if len(entities) > 1
    ]


def _apply_one(submodel_id: str, decision: Any, endpoint: Optional[str]) -> None:
    """One decision against the repository, the same way `aas populate` does.

    Literally the same way, now. This used to carry its own copy of the value
    shaping and the element construction, and both copies were wrong in
    different places -- this one sent the *element* form of a language string
    into a `$value`, which is the HTTP 500 the shared helper exists to prevent,
    and both built a `Property` whatever the plan said.
    """
    from ..aasctl.paths import ElementPath

    with _client(endpoint) as client:
        target = ElementPath.parse(decision.path or "")
        if decision.action == "write":
            client.set_value(submodel_id, target, match_mod.value_for_write(decision))
        else:
            client.add_element(submodel_id, target, match_mod.element_for_create(decision))


@router.get("/api/check-matching/{filename}")
def check_matching(
    filename: str,
    aas_id_short: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Whether this document has already been matched against this AAS."""
    try:
        edited = workspace.edited_match_file(
            f"{workspace.stem(filename)}_{aas_id_short}" if aas_id_short else filename
        )
        plan = workspace.match_file(filename, aas_id_short)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    for candidate in (edited, plan):
        if candidate.exists():
            return {"exists": True, "filename": candidate.name,
                    "edited": candidate == edited,
                    "expectedPath": str(plan)}
    return {"exists": False, "filename": None, "edited": False,
            "expectedPath": str(plan)}


@router.get("/api/matching-results/{filename}")
def matching_results(
    filename: str,
    aas_id_short: Optional[str] = Query(None),
    endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """A match plan, rendered as the rows the matching table draws.

    Every field the plan will need on the way back is carried on the row --
    `submodelId`, `action`, `valueType`, `unit`, `source`. They are not
    displayed. They are how a plan survives being edited in a browser.
    """
    path = _resolve_match_file(filename, aas_id_short)
    plans = _load_plans(path)

    id_shorts = _submodel_id_shorts(
        endpoint, [plan.submodel_id for plan in plans]
    )
    results: List[Dict[str, Any]] = []
    for plan in plans:
        label = id_shorts.get(plan.submodel_id) or plan.submodel_id
        results.extend(_row_of(decision, plan.submodel_id, label)
                       for decision in plan.decisions)

    return {"success": True, "results": results, "count": len(results),
            "source": path.name}


@router.post("/api/save-matching-results")
def save_matching_results(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Turn the edited rows back into match plans `aas populate` can apply.

    One plan per submodel, because a `Plan` names one `submodel_id` and the
    table can span several. `path` points at a manifest listing them, so a
    caller that has to apply several does not have to guess the names.

    A row whose target submodel cannot be established is *not* written into
    some plausible plan. It comes back under `unresolved`, because the failure
    mode this avoids -- writing the value to the right-looking element of the
    wrong submodel -- succeeds silently and is found months later.
    """
    filename = str(body.get("filename") or "").strip()
    rows = body.get("results")
    if not filename or not isinstance(rows, list):
        raise HTTPException(
            status_code=422, detail="filename and a results list are both required"
        )

    endpoint = body.get("endpoint")
    aas_id_short = body.get("aas_idShort") or body.get("aas_id_short")

    grouped: Dict[str, List[match_mod.Decision]] = {}
    unresolved: List[Dict[str, Any]] = []
    with _client(endpoint) as client:
        lookup = _SubmodelLookup(client, aas_id_short)
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                submodel_id, decision = _decision_of(row, lookup)
            except _Unresolved as exc:
                unresolved.append({"entity": row.get("extractedEntity"),
                                   "reason": str(exc)})
                continue
            grouped.setdefault(submodel_id, []).append(decision)

    try:
        manifest_path = workspace.edited_match_file(filename)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    written: List[Dict[str, Any]] = []
    for index, (submodel_id, decisions) in enumerate(sorted(grouped.items())):
        plan = match_mod.Plan(
            submodel_id=submodel_id,
            theta_high=1.0,          # the person decided; no threshold applies
            theta_low=0.0,
            model=None,
            decisions=decisions,
        )
        plan_path = manifest_path.with_name(
            f"{manifest_path.stem}.{index}{manifest_path.suffix}"
        )
        _write_json(plan_path, plan.to_json())
        written.append({
            "submodel": submodel_id,
            "path": str(plan_path),
            "counts": plan.counts(),
        })

    _write_json(manifest_path, {
        "kind": "match-plan-manifest",
        "filename": filename,
        "plans": written,
        "unresolved": unresolved,
    })

    return {
        "success": True,
        "path": str(manifest_path),
        "plans": written,
        "unresolved": unresolved,
        "count": sum(len(d) for d in grouped.values()),
    }


# ---------------------------------------------------------------------------
# entities on disk
# ---------------------------------------------------------------------------

def _read_entities(filename: str) -> List[Dict[str, Any]]:
    return _read_extraction(filename)[0]


def _read_extraction(filename: str) -> "Tuple[List[Dict[str, Any]], Dict[str, Any]]":
    """The entities, and what the extraction said about them.

    Both shapes are on disk and both are read: a bare array from before
    extraction had anything to say beyond the facts, and the object that
    carries the configurations and the warnings alongside them.
    """
    try:
        edited = workspace.extraction_file(filename, edited=True)
        original = workspace.extraction_file(filename)
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    path = edited if edited.exists() else original
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"no extraction for {filename!r}; expected {original.name}",
        )

    raw = _read_json(path)
    meta: Dict[str, Any] = {}
    if isinstance(raw, dict):
        meta = {
            "configurations": raw.get("configurations") or [],
            "warnings": raw.get("warnings") or [],
        }
        raw = raw.get("entities") or raw.get("result") or []
    if not isinstance(raw, list):
        raise HTTPException(
            status_code=422,
            detail=f"{path.name} is neither a list of entities nor an object holding one",
        )
    rows = [_canonical_entity(item) for item in raw if isinstance(item, dict)]

    # A file written before `configurations` existed still distinguishes
    # configurations if its rows carry a variant; deriving it costs one pass
    # and means an old extraction does not silently lose the picker.
    if not meta.get("configurations"):
        seen: List[str] = []
        for row in rows:
            name = str(row.get("variant") or "").strip()
            if name and name not in seen:
                seen.append(name)
        meta["configurations"] = seen
    return rows, meta


def _canonical_entity(row: Dict[str, Any]) -> Dict[str, Any]:
    """One spelling of the name field, in both directions.

    The extractor writes `entity`; `match.load_entities` reads `entity` first
    and `name` second. The review table asked for `name`, which nothing set --
    so the column rendered blank, and a person who typed a name into it wrote
    `name` beside an `entity` the matcher went on preferring. The correction
    was accepted, saved, and then ignored.

    The table is fixed to use `entity`, and this closes the loop on both sides:
    a `name` that disagrees is a correction made through the old table and is
    the later of the two, so it wins. Rows are copied rather than mutated --
    the caller's list is the request body, and a read is not a place to edit
    it.
    """
    edited = str(row.get("name") or "").strip()
    if not edited or edited == str(row.get("entity") or "").strip():
        return row
    out = dict(row)
    out["entity"] = edited
    out.pop("name", None)
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"{path.name} is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"cannot read {path.name}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    """Write through a temporary file in the same directory, then replace.

    A half-written plan is worse than an absent one: the next stage reads it,
    fails to parse it, and the wizard reports a matching problem for what is
    actually a disk problem.
    """
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        shutil.move(str(temporary), str(path))
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"cannot write {path.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# plans on disk
# ---------------------------------------------------------------------------

def _resolve_match_file(filename: str, aas_id_short: Optional[str]) -> Path:
    """The plan for a document, given either the document's name or the plan's.

    **Both, because both are handed to it.** `check-matching` answers with the
    *plan's* name and the browser passes that straight to this route, while the
    wizard's own bookkeeping holds the *document's*. Deriving only from the
    document meant the plan name got the suffix appended a second time and a
    plan that was sitting on disk came back 404 -- twice, once inside
    `POST /api/match` after 621 s of completed work, and once from the browser
    on the next run.

    Taking a name that already resolves at face value is the fix that does not
    require the caller to know which of the two conventions it is holding.
    """
    try:
        direct = workspace.under(workspace.matches(), workspace.safe_name(filename))
        candidates = [
            # A name that is already a plan wins: it is unambiguous, where a
            # derivation is a guess about which convention the caller used.
            direct,
            workspace.edited_match_file(
                f"{workspace.stem(filename)}_{aas_id_short}" if aas_id_short else filename
            ),
            workspace.match_file(filename, aas_id_short),
            workspace.match_file(filename),
        ]
    except workspace.WorkspaceError as exc:
        raise _bad_name(exc) from exc

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise HTTPException(
        status_code=404,
        detail=f"no match plan for {filename!r}; expected {candidates[2].name}",
    )


def _load_plans(path: Path) -> List[match_mod.Plan]:
    """A plan file, or a manifest naming several."""
    raw = _read_json(path)
    if isinstance(raw, dict) and raw.get("kind") == "match-plan-manifest":
        plans = []
        for entry in raw.get("plans") or []:
            child = Path(entry.get("path", ""))
            if child.exists():
                plans.append(_as_plan(_read_json(child), child.name))
        return plans
    return [_as_plan(raw, path.name)]


def _as_plan(raw: Any, name: str) -> match_mod.Plan:
    try:
        return match_mod.Plan.from_json(raw)
    except (match_mod.MatchError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{name} is not a match plan: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# plan <-> table row
# ---------------------------------------------------------------------------

def _row_of(
    decision: match_mod.Decision, submodel_id: str, submodel_label: str
) -> Dict[str, Any]:
    """One plan decision as one table row.

    `aasProperty` is what the table prints, and for a `create` it is the
    element that does not exist yet, not the collection it will go in --
    printing the parent would tell the user their value is going somewhere it
    is not.
    """
    leaf = decision.id_short or (decision.path or "").rsplit(".", 1)[-1]
    return {
        # what the table draws
        "extractedEntity": decision.entity,
        "extractedValue": decision.value,
        "aasProperty": leaf,
        # The element being written or created. For a create this is NOT the
        # type of the thing at `apiPath` -- see `targetType`, and see
        # `_decision_of` for what happened when the two were the same field.
        "aasType": decision.model_type or "",
        # The modelType of the element at `apiPath`: the slot for a write, the
        # container for a create.
        "targetType": decision.container_type or decision.model_type or "",
        "confidence": decision.score,
        "matchType": decision.band,
        "selected": decision.action != "skip",
        "reason": decision.reason,
        "semanticPath": f"{submodel_label}/{decision.path}" if decision.path else "",
        "apiPath": decision.path or "",
        "semanticId": "",
        # carried, not drawn: what the plan needs back
        "submodelId": submodel_id,
        "action": decision.action,
        "valueType": decision.value_type or "",
        "unit": decision.unit,
        "source": decision.source,
        "margin": decision.margin,
        "warnings": decision.warnings,
    }


class _Unresolved(RuntimeError):
    """This row cannot be turned into a decision without guessing."""


def _decision_of(
    row: Dict[str, Any], lookup: "_SubmodelLookup"
) -> Tuple[str, match_mod.Decision]:
    """One table row as one plan decision, with its submodel."""
    entity = str(row.get("extractedEntity") or "").strip()
    if not entity:
        raise _Unresolved("row has no entity name")

    target = _target_of(row)
    if not target.get("apiPath"):
        raise _Unresolved(f"{entity} has no selected element")

    submodel_id = lookup.resolve(row, target)
    value = row.get("extractedValue")
    unit = str(row.get("unit") or "")
    path = str(target["apiPath"])

    # What kind of element is *at* that path. Never `aasType`, which is the
    # element being created and is a Property precisely when the target is a
    # collection. Reading one as the other is what silently turned every
    # hosted create into a write onto its own container: `band: "hosted"`,
    # `action: "write"`, path naming a SubmodelElementCollection, and an HTTP
    # 500 with no reason given. Three of them in one live run, and the value
    # was gone rather than misplaced.
    target_type = str(target.get("type") or row.get("targetType") or "")

    if _repicked(row):
        # The user clicked a different element, so its type decides: a
        # container means "make a new element in here".
        action = "create" if target_type in _CONTAINERS else "write"
    else:
        # Nobody touched the row, so the matcher's own verdict stands. It is
        # carried on the row for exactly this reason.
        action = str(row.get("action") or "").strip() or (
            "create" if target_type in _CONTAINERS else "write"
        )

    entity_obj = match_mod.Entity(
        entity=entity,
        value=None if value is None else str(value),
        description=str(row.get("description") or ""),
        unit=unit,
        source=str(row.get("source") or ""),
    )

    if action == "create":
        # What to create is decided here, not left as whatever `aasType`
        # happened to hold: the row may have been re-pointed at a different
        # container since the matcher typed it, and a list dictates its
        # children whatever the row says.
        container = match_mod.Candidate(
            path=path, id_short=path.rsplit(".", 1)[-1], model_type=target_type,
            score=0.0, parent=None,
            type_value_list_element=target.get("typeValueListElement"),
        )
        # The matcher already typed this element and a person reviewed that
        # type, so it carries -- unless the row was re-pointed, where the old
        # type belongs to a home the value no longer has. Either way the
        # container has the last word: a list dictates its children.
        carried = (
            None if _repicked(row) or not row.get("aasType")
            else match_mod.ElementSpec(model_type=str(row["aasType"]))
        )
        spec, spec_warnings = match_mod.resolve_spec(entity_obj, container, carried)
        in_list = target_type == "SubmodelElementList"
        return submodel_id, match_mod.Decision(
            entity=entity,
            value=entity_obj.value,
            unit=unit,
            source=entity_obj.source,
            action="create",
            band=str(row.get("matchType") or "manual"),
            reason=str(row.get("reason") or "confirmed in the wizard"),
            path=path,
            model_type=spec.model_type,
            container_type=target_type or "SubmodelElementCollection",
            value_type=spec.value_type,
            min_value=spec.min_value,
            max_value=spec.max_value,
            content_type=spec.content_type,
            id_short=None if in_list else match_mod.safe_id_short(entity),
            score=float(row.get("confidence") or 0.0),
            margin=float(row.get("margin") or 0.0),
            warnings=list(row.get("warnings") or []) + spec_warnings,
        )

    # A write. The structured `$value` types need their parts worked out the
    # same way `match.decide` does, or `value_for_write` has nothing to send.
    shape = match_mod.infer_element_type(entity_obj)
    warnings = list(row.get("warnings") or [])
    min_value = max_value = content_type = None
    if target_type == "Range":
        if shape.model_type == "Range":
            min_value, max_value = shape.min_value, shape.max_value
        else:
            warnings.append(
                f"{path} is a Range, but {value!r} gives no min and max"
            )
    elif target_type in ("File", "Blob"):
        content_type = shape.content_type or "application/octet-stream"

    return submodel_id, match_mod.Decision(
        entity=entity,
        value=entity_obj.value,
        unit=unit,
        source=entity_obj.source,
        action="write",
        band=str(row.get("matchType") or "manual"),
        reason=str(row.get("reason") or "confirmed in the wizard"),
        path=path,
        model_type=target_type or None,
        value_type=(
            str(row.get("valueType"))
            if row.get("valueType")
            else match_mod.infer_value_type(entity_obj.value, unit)
        ),
        min_value=min_value,
        max_value=max_value,
        content_type=content_type,
        id_short=None,
        score=float(row.get("confidence") or 0.0),
        margin=float(row.get("margin") or 0.0),
        warnings=warnings,
    )


def _repicked(row: Dict[str, Any]) -> bool:
    """Whether the user pointed this row at a different element by hand."""
    picked = row.get("aasProperties")
    return (
        isinstance(picked, list)
        and bool(picked)
        and isinstance(picked[0], dict)
        and bool(picked[0].get("apiPath"))
    )


def _target_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """The element this row points at, preferring what the user chose.

    `aasProperties` is written by the property picker and is authoritative when
    it exists: it carries the `apiPath` and `semanticPath` of the element
    actually clicked. The flat fields are the matcher's original suggestion,
    which is only still the target if nobody touched it.
    """
    picked = row.get("aasProperties")
    if isinstance(picked, list) and picked:
        first = picked[0]
        if isinstance(first, dict) and first.get("apiPath"):
            return first
    return {
        "apiPath": row.get("apiPath"),
        "semanticPath": row.get("semanticPath"),
        # `targetType`, not `aasType`: what is at the path, not what goes in it.
        "type": row.get("targetType") or row.get("aasType"),
        "typeValueListElement": row.get("typeValueListElement"),
    }


class _SubmodelLookup:
    """Which submodel a chosen element lives in.

    The picker hands back `semanticPath`, whose first segment is the submodel's
    idShort. That is the only evidence of a *re-targeted* row: `submodelId` on
    the row is whatever the matcher put there, and after the user picks an
    element in a different submodel it is stale. Preferring the stale field
    writes the value into the wrong submodel and reports success.
    """

    def __init__(self, client: BasyxClient, aas_id_short: Optional[str]) -> None:
        self._client = client
        self._aas_id_short = aas_id_short
        self._by_id_short: Optional[Dict[str, List[str]]] = None

    def resolve(self, row: Dict[str, Any], target: Dict[str, Any]) -> str:
        label = str(target.get("semanticPath") or "").split("/", 1)[0].strip()
        carried = str(row.get("submodelId") or "").strip()

        if label:
            matches = self._index().get(label, [])
            if len(matches) == 1:
                return matches[0]
            if carried in matches:
                return carried
            if len(matches) > 1:
                raise _Unresolved(
                    f"{len(matches)} submodels are called {label!r}; "
                    "cannot tell which one was selected"
                )
        if carried:
            return carried
        raise _Unresolved("no submodel could be established for the selected element")

    def _index(self) -> Dict[str, List[str]]:
        if self._by_id_short is None:
            self._by_id_short = {}
            for submodel in self._submodels():
                key = submodel.get("idShort") or submodel.get("id") or ""
                self._by_id_short.setdefault(key, []).append(submodel.get("id", ""))
        return self._by_id_short

    def _submodels(self) -> List[Dict[str, Any]]:
        """The submodels of the target AAS, or the repository's if none was named.

        Scoped to the shell when the caller says which one: two shells can hold
        submodels with the same idShort, and narrowing to the one being
        populated turns an ambiguity into an answer.
        """
        if self._aas_id_short:
            from .aas_routes import _resolve_shell

            try:
                shell = _resolve_shell(self._client, self._aas_id_short)
            except HTTPException:
                shell = None
            if shell:
                out = []
                for reference in self._client.list_submodel_refs(shell["id"]):
                    for key in reference.get("keys") or []:
                        if key.get("value"):
                            try:
                                out.append(self._client.get_submodel(key["value"]))
                            except AasHttpError:
                                continue
                return out
        try:
            return self._client.list_submodels(limit=500)
        except Exception as exc:  # noqa: BLE001
            raise _fail(exc, "cannot list submodels to resolve the match targets") from exc


def _submodel_id_shorts(
    endpoint: Optional[str], submodel_ids: List[str]
) -> Dict[str, str]:
    """idShorts for the plans' submodels, for the `semanticPath` the table shows.

    Best effort. A label that cannot be fetched falls back to the identifier,
    which is uglier on screen but never wrong.
    """
    out: Dict[str, str] = {}
    if not submodel_ids:
        return out
    try:
        with _client(endpoint) as client:
            for submodel_id in set(submodel_ids):
                try:
                    out[submodel_id] = client.get_submodel(submodel_id).get("idShort") or ""
                except AasHttpError:
                    continue
    except HTTPException:
        return out
    except Exception:  # noqa: BLE001 - a label is not worth failing the route
        return out
    return {k: v for k, v in out.items() if v}


__all__ = ["router"]

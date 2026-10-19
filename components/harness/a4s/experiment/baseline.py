"""The baseline arm: one LLM tool per authoring intent.

This is the surface the predecessor system exposed -- ``aas_explore``,
``aas_read_property``, ``aas_from_smt``, ``aas_match_inputs`` and the rest --
reconstructed here so the comparison isolates one variable.

**What is being compared is the shape of the tool surface, not the quality of
the implementation.** Every tool below calls the same library the ``aas``
verbs call, so the baseline arm inherits every fix made along the way: the
percent-encoded list indices, the unit alias table, the complete xsd type
enumeration, the cardinality dialects. Reimplementing the predecessor's bugs
would flatter the CLI for reasons that have nothing to do with tool surfaces.
The baseline here is *stronger* than the system it stands in for, which is the
conservative direction: if the CLI still wins, it wins against a debugged
opponent.

**Both arms are given the same capabilities.** Where the CLI grew a verb the
predecessor had no counterpart for -- ``aas smt diff``, ``aas validate`` --
the baseline gets a tool for it too. Withholding those would measure a
capability gap and call it a surface effect. What differs between the arms is
only *packaging*: 19 JSON schemas resident in every request against one
``shell`` schema and a command synopsis.

The parameter shapes deliberately follow the predecessor's granularity: one
call reads one property, one call writes one property, one call instantiates
one template. That granularity is the thing under test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..aasctl import index as index_mod
from ..aasctl import match as match_mod
from ..aasctl import smt as smt_mod
from ..aasctl.client import BasyxClient, Endpoint
from ..aasctl.model import make_element, make_shell, make_submodel, submodel_reference
from ..aasctl.paths import ElementPath
from ..environments.base import Environment
from ..ledger import Ledger
from ..tools import ToolRegistry, ToolSpec

MAX_OUTPUT_ROWS = 60


def _obj(**properties: Any) -> Dict[str, Any]:
    required = [name for name, spec in properties.items() if spec.pop("_required", False)]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _str(description: str, required: bool = True) -> Dict[str, Any]:
    return {"type": "string", "description": description, "_required": required}


def _int(description: str, required: bool = False) -> Dict[str, Any]:
    return {"type": "integer", "description": description, "_required": required}


def _bool(description: str, required: bool = False) -> Dict[str, Any]:
    return {"type": "boolean", "description": description, "_required": required}


class BaselineRegistry(ToolRegistry):
    """`read_file`, `write_file`, `ask_user`, `finish` -- and no shell.

    Dropping ``shell`` is the point. With it the baseline arm could simply run
    ``aas`` and the two arms would collapse into one.
    """

    def __init__(
        self,
        environment: Environment,
        ledger: Ledger,
        workdir: Path,
        endpoint: str,
        index_home: Optional[Path] = None,
    ) -> None:
        self.endpoint = endpoint
        self.index_home = index_home or index_mod.DEFAULT_INDEX_HOME
        super().__init__(environment, ledger, workdir)
        self._tools.pop("shell", None)
        for spec in _aas_tools(self):
            self.register(spec)

    def client(self) -> BasyxClient:
        return BasyxClient(Endpoint.from_env(self.endpoint))


# ---------------------------------------------------------------------------
# the tools
# ---------------------------------------------------------------------------

def _aas_tools(registry: "BaselineRegistry") -> List[ToolSpec]:
    def tool(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., str],
        mutates: bool = False,
    ) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            mutates=mutates,
        )

    # -- reading -----------------------------------------------------------

    def aas_explore(limit: int = 100) -> str:
        with registry.client() as client:
            shells = client.list_shells(limit=limit)
        return "\n".join(
            f"{s.get('idShort') or '?'}\t{len(s.get('submodels') or [])} sm\t{s.get('id')}"
            for s in shells
        ) or "(no shells)"

    def aas_list_submodels(shell_id: Optional[str] = None, limit: int = 200) -> str:
        with registry.client() as client:
            if shell_id:
                refs = client.list_submodel_refs(shell_id)
                ids = [
                    k["value"]
                    for r in refs
                    for k in r.get("keys", [])
                    if k.get("type") == "Submodel"
                ]
                submodels = [client.get_submodel(i) for i in ids]
            else:
                submodels = client.list_submodels(limit=limit)
        return "\n".join(
            f"{s.get('idShort') or '?'}\t{s.get('kind') or 'Instance'}\t"
            f"{len(s.get('submodelElements') or [])} el\t{s.get('id')}"
            for s in submodels
        ) or "(no submodels)"

    def aas_list_elements(submodel_id: str, path: str = "", depth: int = 2) -> str:
        from ..aasctl.__main__ import _walk, get_submodel

        with registry.client() as client:
            submodel = get_submodel(client, submodel_id)
            node = (
                submodel if not path
                else client.get_element(submodel.get("id", submodel_id), ElementPath.parse(path))
            )
        rows: List[str] = []
        _walk(node, path, depth, rows)
        if len(rows) >= MAX_OUTPUT_ROWS:
            rows.append(f"... truncated at {MAX_OUTPUT_ROWS} rows; narrow with `path`")
        return "\n".join(rows) or "(no children)"

    def aas_read_property(submodel_id: str, path: str) -> str:
        with registry.client() as client:
            value = client.get_value(submodel_id, ElementPath.parse(path))
        return json.dumps(value, ensure_ascii=False)

    def aas_describe_property(submodel_id: str, path: str) -> str:
        from ..aasctl.__main__ import _delegation_url, _semantic_value

        with registry.client() as client:
            element = client.get_element(submodel_id, ElementPath.parse(path))
        return json.dumps(
            {
                "path": path,
                "modelType": element.get("modelType"),
                "valueType": element.get("valueType"),
                "value": element.get("value"),
                "semanticId": _semantic_value(element),
                "delegation": _delegation_url(element),
            },
            ensure_ascii=False,
        )

    # -- writing -----------------------------------------------------------

    def aas_write_property(submodel_id: str, path: str, value: str, is_multilanguage: bool = False) -> str:
        from ..aasctl.__main__ import _lang_strings

        payload: Any = _lang_strings(value) if is_multilanguage else value
        with registry.client() as client:
            client.set_value(submodel_id, ElementPath.parse(path), payload)
        return f"set {path} = {payload!r}"

    def aas_create_shell(shell_id: str, id_short: str, global_asset_id: str = "") -> str:
        with registry.client() as client:
            client.create_shell(
                make_shell(shell_id, id_short, global_asset_id=global_asset_id or shell_id)
            )
        return f"created shell {shell_id}"

    def aas_create_submodel(
        submodel_id: str, id_short: str, semantic_id: str = "", attach_to: str = ""
    ) -> str:
        with registry.client() as client:
            client.create_submodel(
                make_submodel(submodel_id, id_short, semantic_id=semantic_id or None)
            )
            if attach_to:
                client.add_submodel_ref(attach_to, submodel_reference(submodel_id))
        return f"created submodel {submodel_id}"

    def aas_create_element(
        submodel_id: str,
        parent_path: str,
        model_type: str,
        id_short: str = "",
        value: str = "",
        value_type: str = "xs:string",
        semantic_id: str = "",
    ) -> str:
        element = make_element(
            model_type,
            id_short or None,
            value=value or None,
            value_type=value_type,
            semantic_id=semantic_id or None,
        )
        with registry.client() as client:
            client.add_element(submodel_id, ElementPath.parse(parent_path), element)
        return f"added {model_type} under {parent_path or '<root>'}"

    # -- templates ---------------------------------------------------------

    def aas_list_templates(of_shell: str = "", limit: int = 200) -> str:
        with registry.client() as client:
            if of_shell:
                refs = client.list_submodel_refs(of_shell)
                ids = [
                    k["value"] for r in refs for k in r.get("keys", [])
                    if k.get("type") == "Submodel"
                ]
                candidates = [client.get_submodel(i) for i in ids]
            else:
                candidates = client.list_submodels(limit=limit)
        rows = []
        for s in candidates:
            if s.get("kind") != "Template":
                continue
            top = s.get("submodelElements") or []
            rows.append(
                f"{s.get('idShort') or '?'}\t"
                f"{sum(1 for e in top if not smt_mod.is_optional(e))} required\t"
                f"{sum(1 for e in top if smt_mod.is_optional(e))} optional\t{s.get('id')}"
            )
        return "\n".join(rows) or "(no templates)"

    def aas_template_outline(template_id: str) -> str:
        with registry.client() as client:
            template = client.get_submodel(template_id)
        return "\n".join(smt_mod.outline(template, max_rows=MAX_OUTPUT_ROWS)) or "(empty)"

    def aas_from_smt(
        template_id: str,
        submodel_id: str,
        id_short: str = "",
        attach_to: str = "",
        include_optional: bool = False,
    ) -> str:
        with registry.client() as client:
            template = client.get_submodel(template_id)
            result = smt_mod.instantiate(
                template,
                submodel_id=submodel_id,
                id_short=id_short or None,
                include_optional=include_optional,
            )
            client.create_submodel(result.submodel)
            if attach_to:
                client.add_submodel_ref(attach_to, submodel_reference(submodel_id))
        return json.dumps(result.summary(), ensure_ascii=False)

    def aas_template_diff(template_id: str, submodel_id: str, include_optional: bool = False) -> str:
        with registry.client() as client:
            template = client.get_submodel(template_id)
            instance = client.get_submodel(submodel_id)
        found = smt_mod.gaps(template, instance, include_optional=include_optional)
        return "\n".join(g.line() for g in found[:MAX_OUTPUT_ROWS]) or "complete: every required slot is filled"

    def aas_validate(submodel_id: str) -> str:
        from ..aasctl.__main__ import build_parser, cmd_validate, Out
        import io
        import contextlib

        args = build_parser().parse_args(["--json", "validate", submodel_id])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cmd_validate(args, Out(as_json=True))
        return buffer.getvalue().strip()

    # -- retrieval ---------------------------------------------------------

    def aas_parse(submodel_id: str) -> str:
        with registry.client() as client:
            submodel = client.get_submodel(submodel_id)
            concepts = index_mod.concept_map(client.list_concept_descriptions())
        built = index_mod.build(submodel, concepts)
        built.save(registry.index_home)
        return json.dumps(
            {"submodel": built.submodel_id, "nodes": len(built.nodes), "enriched": built.enriched},
            ensure_ascii=False,
        )

    def aas_search_property(submodel_id: str, query: str, top_k: int = 5) -> str:
        loaded = index_mod.Index.load(submodel_id, registry.index_home)
        vector = index_mod.embed([query], model=loaded.model)[0]
        hits = index_mod.search(loaded, vector, top_k=top_k)
        lines: List[str] = []
        for hit in hits:
            lines.extend(hit.block())
        return "\n".join(lines) or "(no hits)"

    def aas_search_repository(query: str, top_k: int = 5) -> str:
        """Search every indexed submodel at once.

        Added when the suite grew a task that does not name its submodel. The
        CLI reaches this with `aas find --across`, and withholding it here
        would make the comparison measure a capability the baseline lacks
        rather than the shape of the surface -- which is the one thing this
        arm exists to hold constant.
        """
        loaded = index_mod.Index.load(index_mod.ALL, registry.index_home)
        vector = index_mod.embed([query], model=loaded.model)[0]
        hits = index_mod.search(loaded, vector, top_k=top_k)
        lines: List[str] = []
        for hit in hits:
            block = hit.block()
            if hit.node.submodel:
                block.insert(1, f"        in:       {hit.node.submodel}")
            lines.extend(block)
        return "\n".join(lines) or "(no hits)"

    def aas_delete_element(submodel_id: str, path: str) -> str:
        """Remove one element.

        The predecessor had no delete tool, and neither did this arm until the
        suite grew a task that deletes. Its absence would have made `prune`
        unattemptable rather than merely harder, and an arm that cannot attempt
        a task measures nothing about how it would have gone about it.
        """
        with registry.client() as client:
            client.delete_element(submodel_id, ElementPath.parse(path))
        return f"deleted {path} from {submodel_id}"

    # -- matching ----------------------------------------------------------

    def aas_match_inputs(
        entities_file: str,
        submodel_id: str,
        out_file: str,
        theta_high: float = match_mod.THETA_HIGH,
        theta_low: float = match_mod.THETA_LOW,
    ) -> str:
        from ..kernel.providers import ModelConfig, build as build_provider

        entities = match_mod.load_entities(
            json.loads((registry.workdir / entities_file).read_text(encoding="utf-8"))
        )
        loaded = index_mod.Index.load(submodel_id, registry.index_home)
        vectors = index_mod.embed([e.query() for e in entities], model=loaded.model)
        config = ModelConfig.from_env()
        provider = build_provider(config)
        arbitrate = match_mod.make_arbitrator(provider, config, match_mod.ARBITRATE_PROMPT)
        # The typed-create chooser, not the plain arbitrator: withholding it
        # here would leave the baseline unable to author an element type the
        # CLI arm can, which is a capability gap and not the surface effect
        # under test. See the module docstring.
        host = match_mod.make_host_chooser(provider, config, match_mod.HOST_PROMPT)

        plan = match_mod.Plan(
            submodel_id=loaded.submodel_id,
            theta_high=theta_high,
            theta_low=theta_low,
            model=config.model,
        )
        for entity, vector in zip(entities, vectors):
            plan.decisions.append(
                match_mod.decide(
                    entity,
                    match_mod.candidates_for(loaded, vector, 5),
                    theta_high=theta_high,
                    theta_low=theta_low,
                    arbitrate=arbitrate,
                    host=host,
                )
            )
        target = registry.workdir / out_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(plan.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return "\n".join([json.dumps(plan.counts()), *[d.line() for d in plan.decisions]])

    def aas_populate_inputs(plan_file: str, skip_warnings: bool = False) -> str:
        plan = match_mod.Plan.from_json(
            json.loads((registry.workdir / plan_file).read_text(encoding="utf-8"))
        )
        todo = [d for d in plan.decisions if d.action in ("write", "create")]
        if skip_warnings:
            todo = [d for d in todo if not d.warnings]
        applied, failed = [], []
        with registry.client() as client:
            for decision in todo:
                if decision.value is None:
                    failed.append(f"{decision.entity}: no value")
                    continue
                try:
                    if decision.action == "write":
                        client.set_value(
                            plan.submodel_id,
                            ElementPath.parse(decision.path or ""),
                            match_mod.value_for_write(decision),
                        )
                    else:
                        client.add_element(
                            plan.submodel_id,
                            ElementPath.parse(decision.path or ""),
                            match_mod.element_for_create(decision),
                        )
                except Exception as exc:  # noqa: BLE001 - reported to the model
                    failed.append(f"{decision.entity}: {exc}")
                    continue
                applied.append(f"{decision.action} {decision.path}")
        return json.dumps(
            {"applied": applied, "failed": failed, "not_attempted": len(plan.decisions) - len(todo)},
            ensure_ascii=False,
        )

    # -- extraction --------------------------------------------------------

    def input_pdf2entity(pdf_path: str, out_file: str, dpi: int = 200) -> str:
        from .. import extract as ex
        from ..kernel.providers import ModelConfig, build as build_provider

        config = ModelConfig.from_env()
        provider = build_provider(config)
        source = registry.workdir / pdf_path
        batches = []
        for number, png in enumerate(ex.render_pdf(source, dpi=dpi), start=1):
            reply = ex.ask_image(provider, config, ex.PDF_PROMPT, png)
            batches.append(ex.parse_entities(reply, f"{source.name} p.{number}"))
        entities = ex.merge(batches)
        target = registry.workdir / out_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([e.as_dict() for e in entities], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return json.dumps(
            {"entities": len(entities), "file": out_file,
             "preview": [e.entity for e in entities[:15]]},
            ensure_ascii=False,
        )

    # -- operations --------------------------------------------------------

    def aas_operation_delegation(submodel_id: str, path: str, arguments_json: str = "{}") -> str:
        import httpx

        from ..aasctl.__main__ import _delegation_url

        arguments = json.loads(arguments_json or "{}")
        with registry.client() as client:
            element = client.get_element(submodel_id, ElementPath.parse(path))
        url = _delegation_url(element)
        if not url:
            return f"error: {path} declares no invocationDelegation"
        response = httpx.post(url, json=arguments, timeout=120.0, verify=False)
        return f"{response.status_code}: {response.text[:1500]}"

    return [
        tool("aas_explore", "List the Asset Administration Shells in the repository.",
             _obj(limit=_int("Maximum shells to return.")), aas_explore),
        tool("aas_list_submodels",
             "List submodels, optionally only those referenced by one shell.",
             _obj(shell_id=_str("AAS id to restrict to. Omit for all.", False),
                  limit=_int("Maximum submodels.")), aas_list_submodels),
        tool("aas_list_elements",
             "List the elements of a submodel, or of one collection inside it, as "
             "idShortPaths with their types.",
             _obj(submodel_id=_str("Submodel id."),
                  path=_str("idShortPath of a collection. Omit for the root.", False),
                  depth=_int("How many levels to descend. Default 2.")), aas_list_elements),
        tool("aas_read_property", "Read the value of one element by its idShortPath.",
             _obj(submodel_id=_str("Submodel id."), path=_str("idShortPath, e.g. A.B[0].C")),
             aas_read_property),
        tool("aas_describe_property",
             "Show one element's type, value, semanticId and delegation target.",
             _obj(submodel_id=_str("Submodel id."), path=_str("idShortPath.")),
             aas_describe_property),
        tool("aas_write_property", "Write the value of one existing element.",
             _obj(submodel_id=_str("Submodel id."), path=_str("idShortPath."),
                  value=_str("The value to write."),
                  is_multilanguage=_bool(
                      "True for a MultiLanguageProperty; give value as 'text@en' "
                      "(or 'a@en|b@de').")),
             aas_write_property, mutates=True),
        tool("aas_create_shell", "Create an empty Asset Administration Shell.",
             _obj(shell_id=_str("Identifier for the new shell."),
                  id_short=_str("Short name."),
                  global_asset_id=_str("Asset identifier. Defaults to the shell id.", False)),
             aas_create_shell, mutates=True),
        tool("aas_create_submodel", "Create an empty submodel, optionally attaching it to a shell.",
             _obj(submodel_id=_str("Identifier for the new submodel."),
                  id_short=_str("Short name."),
                  semantic_id=_str("semanticId of the submodel.", False),
                  attach_to=_str("AAS id to reference it from.", False)),
             aas_create_submodel, mutates=True),
        tool("aas_create_element", "Add one element of any modelType at any path.",
             _obj(submodel_id=_str("Submodel id."),
                  parent_path=_str("idShortPath of the parent; empty for the submodel root."),
                  model_type=_str("Property, MultiLanguageProperty, SubmodelElementCollection, "
                                  "SubmodelElementList, File, Range, Operation, ..."),
                  id_short=_str("Short name. Omit for a list child.", False),
                  value=_str("Initial value.", False),
                  value_type=_str("xsd type, e.g. xs:string, xs:integer.", False),
                  semantic_id=_str("semanticId.", False)),
             aas_create_element, mutates=True),
        tool("aas_list_templates",
             "List the submodel templates available, with how many of their top-level "
             "slots are mandatory and how many optional.",
             _obj(of_shell=_str("Only templates referenced by this shell.", False),
                  limit=_int("Maximum to scan.")), aas_list_templates),
        tool("aas_template_outline",
             "Show a template's slots, one per line, with the cardinality of each.",
             _obj(template_id=_str("Template submodel id.")), aas_template_outline),
        tool("aas_from_smt",
             "Instantiate a submodel template as an Instance submodel and optionally "
             "attach it to a shell.",
             _obj(template_id=_str("Template submodel id."),
                  submodel_id=_str("Identifier for the new instance."),
                  id_short=_str("Short name for the instance.", False),
                  attach_to=_str("AAS id to reference it from.", False),
                  include_optional=_bool("Also create slots with minimum cardinality 0.")),
             aas_from_smt, mutates=True),
        tool("aas_template_diff",
             "Report which of a template's slots the instance has not filled: 'missing' "
             "means no element, 'empty' means an element with no value.",
             _obj(template_id=_str("Template submodel id."),
                  submodel_id=_str("Instance submodel id."),
                  include_optional=_bool("Also report unfilled optional slots.")),
             aas_template_diff),
        tool("aas_validate", "Report metamodel constraint violations in a submodel.",
             _obj(submodel_id=_str("Submodel id.")), aas_validate),
        tool("aas_parse",
             "Build the semantic index for a submodel. Must be run before "
             "aas_search_property.",
             _obj(submodel_id=_str("Submodel id.")), aas_parse),
        tool("aas_search_property",
             "Find where something belongs in an indexed submodel by meaning rather "
             "than by name. Each hit comes back with its parent collection and that "
             "collection's other children.",
             _obj(submodel_id=_str("Submodel id, already indexed."),
                  query=_str("What to look for, in plain words."),
                  top_k=_int("How many hits. Default 5.")), aas_search_property),
        tool("aas_match_inputs",
             "Match extracted entities to elements of an indexed submodel and write a "
             "plan file. Writes nothing to the repository.",
             _obj(entities_file=_str("JSON file of extracted entities."),
                  submodel_id=_str("Submodel id, already indexed."),
                  out_file=_str("Where to write the plan."),
                  theta_high={"type": "number", "description": "Auto-accept at or above.", "_required": False},
                  theta_low={"type": "number", "description": "Auto-reject below.", "_required": False}),
             aas_match_inputs),
        tool("aas_populate_inputs", "Apply a match plan to the repository.",
             _obj(plan_file=_str("Plan file from aas_match_inputs."),
                  skip_warnings=_bool("Do not apply decisions carrying a warning.")),
             aas_populate_inputs, mutates=True),
        tool("input_pdf2entity",
             "Extract entities from a PDF datasheet, page by page, into a JSON file "
             "of {entity, value, unit, description, source} records.",
             _obj(pdf_path=_str("Path to the PDF."), out_file=_str("Where to write entities."),
                  dpi=_int("Render resolution. Default 200.")), input_pdf2entity),
        tool("aas_operation_delegation",
             "Invoke an Operation through its invocationDelegation qualifier.",
             _obj(submodel_id=_str("Submodel id."), path=_str("idShortPath of the Operation."),
                  arguments_json=_str("JSON object of arguments.", False)),
             aas_operation_delegation, mutates=True),
        tool("aas_search_repository",
             "Search every indexed submodel at once, when you do not know which "
             "submodel holds what you are looking for.",
             _obj(query=_str("What to look for, in your own words."),
                  top_k=_int("How many hits to return. Default 5.")),
             aas_search_repository),
        tool("aas_delete_element", "Delete one element by its idShortPath.",
             _obj(submodel_id=_str("Submodel id."), path=_str("idShortPath to delete.")),
             aas_delete_element, mutates=True),
    ]

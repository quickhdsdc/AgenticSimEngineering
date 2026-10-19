# codegenerator.py
"""
Code generator adapted for your builder_final FlowUnitTemplate / GridUnitTemplate.

- Produces FlowUnit classes that mirror your runtime structure:
    - nodes (dict of node defs with scoped_id)
    - node_objs (actual node objects created with NodeClass(env, scoped_id, **params))
    - edges (dict of edge defs with scoped_id)
    - edge_objs (actual edge objects created with EdgeClass(env, scoped_id, src_obj, dest_obj, **params))
- Produces GridUnit classes that instantiate independent FU instances inside them.
- Produces run_sim.py which instantiates GU instances present in system.gu_instances or, if absent, instantiates one example per GU template.
- Writes package and zips it (optional).
"""

import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

# -------------------------
# Utility helpers
# -------------------------
def _safe_ident(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not s:
        s = "noname"
    if s[0].isdigit():
        s = "_" + s
    return s

def _render_params(params: dict) -> str:
    """Render params dict into keyword arguments, skipping env."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        if k == "env":
            continue
        # keep simple formatting
        if isinstance(v, str):
            parts.append(f"{k}={json.dumps(v)}")
        else:
            parts.append(f"{k}={repr(v)}")
    return (", " + ", ".join(parts)) if parts else ""

# -------------------------
# Extractors specific to your templates
# -------------------------
def ft_get_nodes(ft_template):
    """
    Extract node definitions from your FlowUnitTemplate instance.
    Your FlowUnitTemplate stores base_nodes as a dict {id: node_dict}.
    Return a list of node dicts.
    """
    if hasattr(ft_template, "base_nodes") and isinstance(ft_template.base_nodes, dict):
        # preserve original ordering if possible
        try:
            return list(ft_template.base_nodes.values())
        except Exception:
            return [v for k, v in ft_template.base_nodes.items()]
    # fallback attempts (rare)
    if hasattr(ft_template, "nodes") and isinstance(ft_template.nodes, dict):
        return list(ft_template.nodes.values())
    raise ValueError(f"FlowUnitTemplate '{getattr(ft_template, 'id', repr(ft_template))}' has no base_nodes dict")

def ft_get_edges(ft_template):
    """
    Extract edge definitions from your FlowUnitTemplate instance.
    Your FlowUnitTemplate stores base_edges as a dict {id: edge_dict}.
    """
    if hasattr(ft_template, "base_edges") and isinstance(ft_template.base_edges, dict):
        try:
            return list(ft_template.base_edges.values())
        except Exception:
            return [v for k, v in ft_template.base_edges.items()]
    if hasattr(ft_template, "edges") and isinstance(ft_template.edges, dict):
        return list(ft_template.edges.values())
    raise ValueError(f"FlowUnitTemplate '{getattr(ft_template, 'id', repr(ft_template))}' has no base_edges dict")

def gu_get_fu_map(gu_template, system_fu_templates):
    """
    gu_template.fu_map in builder_final is a dict mapping local_name -> template_id (string)
    but when builder.get_system() returns gu_templates, those will still be GridUnitTemplate objects.
    We expect gu_template.fu_map to still be local->template_id strings; map them to the actual names.
    """
    mapping = {}
    if hasattr(gu_template, "fu_map"):
        for local, tmpl in gu_template.fu_map.items():
            # tmpl may be a string (template id) or an object; prefer string
            if isinstance(tmpl, str):
                mapping[local] = tmpl
            else:
                # find template name by identity if possible
                found = None
                for name, obj in system_fu_templates.items():
                    if obj is tmpl or getattr(obj, "id", None) == getattr(tmpl, "id", None):
                        found = name
                        break
                mapping[local] = found or str(tmpl)
    return mapping

# -------------------------
# Renderers
# -------------------------
def _render_flowunit_py(template_id: str, ft_template) -> str:
    """
    Render a FlowUnit class file content mirroring your runtime FlowUnit structure.
    Assumes nodes from ft_get_nodes(ft_template) where each node dict has:
        {"id": "...", "type": "Source"/"Machine"/..., "params": {...}, ...}
    and edges from ft_get_edges(ft_template) where each edge dict has:
        {"id":"B01", "type":"Buffer", "src":"SRC1", "dest":"M01", "params":{...}, ...}
    """
    class_name = _safe_ident(template_id)
    nodes = ft_get_nodes(ft_template)
    edges = ft_get_edges(ft_template)

    lines = []
    lines.append("import importlib")
    lines.append("")
    lines.append(f"class {class_name}:")
    lines.append(f"    \"\"\"Auto-generated FlowUnit class from template: {template_id}\"\"\"")
    lines.append("    def __init__(self, id, env, parent_scope=None, alias=None):")
    lines.append("        self.id = id")
    lines.append("        self.alias = alias or id")
    lines.append("        self.env = env")
    lines.append("        self.parent_scope = parent_scope")
    lines.append("        self.scope = f\"{parent_scope}.{id}\" if parent_scope else id")
    lines.append("")
    lines.append("        # node/edge definitions (local ids -> defs with scoped_id)")
    lines.append("        self.nodes = {}       # local-id -> def dict (with scoped_id)")
    lines.append("        self.edges = {}       # local-id -> def dict (with scoped_id)")
    lines.append("        self.node_objs = {}   # local-id -> actual Node object")
    lines.append("        self.edge_objs = {}   # local-id -> actual Edge object")
    lines.append("")

    # populate nodes definitions
    lines.append("        # --- node definitions ---")
    for nd in nodes:
        nid = nd.get("id")
        ntype = nd.get("type")
        params = nd.get("params", {}) or {}
        # We'll include the params literal as JSON-like in the def dict; it's okay for readability
        param_text = json.dumps(params)
        lines.append(f"        self.nodes['{nid}'] = {{")
        lines.append(f"            'id': '{nid}',")
        lines.append(f"            'type': '{ntype}',")
        lines.append(f"            'params': {param_text},")
        lines.append(f"            'scoped_id': f\"{{self.scope}}.{nid}\"")
        lines.append("        }")
        lines.append("")

    # populate edges definitions
    lines.append("        # --- edge definitions ---")
    for ed in edges:
        eid = ed.get("id")
        etype = ed.get("type")
        src = ed.get("src")
        dest = ed.get("dest")
        params = ed.get("params", {}) or {}
        param_text = json.dumps(params)
        lines.append(f"        self.edges['{eid}'] = {{")
        lines.append(f"            'id': '{eid}',")
        lines.append(f"            'type': '{etype}',")
        lines.append(f"            'src': '{src}',")
        lines.append(f"            'dest': '{dest}',")
        lines.append(f"            'params': {param_text},")
        lines.append(f"            'scoped_id': f\"{{self.scope}}.{eid}\"")
        lines.append("        }")
        lines.append("")

    # instantiate node objects
    lines.append("        # --- instantiate node objects ---")
    for nd in nodes:
        nid = nd.get("id")
        ntype = nd.get("type")
        lines.append(f"        # node: {nid} ({ntype})")
        lines.append(f"        NodeModule = importlib.import_module('factorysimpy.nodes.{ntype.lower()}')")
        lines.append(f"        NodeClass = getattr(NodeModule, '{ntype}')")
        lines.append(f"        ndef = self.nodes['{nid}']")
        lines.append(f"        nparams = dict(ndef.get('params', {{}}))")
        lines.append(f"        # use the same constructor signature as runtime: NodeClass(env, scoped_id, **params)")
        lines.append(f"        self.node_objs['{nid}'] = NodeClass(self.env, ndef['scoped_id'], **nparams)")
        lines.append(f"        self.node_objs['{nid}'].scope = ndef['scoped_id']")
        lines.append(f"        self.node_objs['{nid}'].parent = self")
        lines.append(f"        setattr(self, '{nid}', self.node_objs['{nid}'])")
        lines.append("")

    # instantiate edges and wire them
    lines.append("        # --- instantiate edge objects and wire them ---")
    for ed in edges:
        eid = ed.get("id")
        etype = ed.get("type")
        src = ed.get("src")
        dest = ed.get("dest")
        lines.append(f"        # edge: {eid} ({etype}) {src} -> {dest}")
        lines.append(f"        EModule = importlib.import_module('factorysimpy.edges.{etype.lower()}')")
        lines.append(f"        EdgeClass = getattr(EModule, '{etype}')")
        lines.append(f"        edef = self.edges['{eid}']")
        lines.append(f"        eparams = dict(edef.get('params', {{}}))")
        # try standard constructor; fallback if TypeError in runtime
        lines.append("        try:")
        lines.append(f"            self.edge_objs['{eid}'] = EdgeClass(self.env, edef['scoped_id'], self.node_objs['{src}'], self.node_objs['{dest}'], **eparams)")
        lines.append("        except TypeError:")
        lines.append(f"            # fallback: constructor may accept (env, id, **params) and provide connect()")
        lines.append(f"            self.edge_objs['{eid}'] = EdgeClass(self.env, edef['scoped_id'], **eparams)")
        lines.append(f"            if hasattr(self.edge_objs['{eid}'], 'connect'):")
        lines.append(f"                try:")
        lines.append(f"                    self.edge_objs['{eid}'].connect(self.node_objs['{src}'], self.node_objs['{dest}'])")
        lines.append("                except Exception:")
        lines.append("                    pass")
        lines.append(f"            self.edge_objs['{eid}'].from_node = self.node_objs['{src}']")
        lines.append(f"            self.edge_objs['{eid}'].to_node = self.node_objs['{dest}']")
        lines.append(f"        self.edge_objs['{eid}'].scope = edef['scoped_id']")
        lines.append(f"        self.edge_objs['{eid}'].parent = self")
        lines.append(f"        setattr(self, '{eid}', self.edge_objs['{eid}'])")
        lines.append("")

    lines.append("    def __repr__(self):")
    lines.append(f"        return f\"{class_name}(id={{self.id}}, scope={{self.scope}})\"")
    return "\n".join(lines)

def _render_gridunit_py(gu_id: str, gu_template, fu_template_map: dict) -> str:
    """
    Render GridUnit class python that instantiates FU instances inside (Option A).
    fu_template_map: local_fu_name -> global_fu_template_name (string)
    """
    class_name = _safe_ident(gu_id)
    connections = getattr(gu_template, "interconnections", []) or []

    lines = []
    lines.append("import importlib")
    lines.append("")
    lines.append(f"class {class_name}:")
    lines.append(f"    \"\"\"Auto-generated GridUnit class from template: {gu_id}\"\"\"")
    lines.append("    def __init__(self, id, env, parent_scope=None, alias=None):")
    lines.append("        self.id = id")
    lines.append("        self.alias = alias or id")
    lines.append("        self.env = env")
    lines.append("        self.parent_scope = parent_scope")
    lines.append("        self.scope = f\"{parent_scope}.{id}\" if parent_scope else id")
    lines.append("")
    lines.append("        # instantiate FlowUnit instances for this GU")
    lines.append("        self.fus = {}")
    lines.append("")

    # create FU instances
    for local_fu, tmpl_name in fu_template_map.items():
        safe_tname = _safe_ident(tmpl_name)
        lines.append(f"        # instantiate FU local '{local_fu}' from template '{tmpl_name}'")
        lines.append(f"        from flowunits.{safe_tname} import {safe_tname} as FUClass_{local_fu}")
        lines.append(f"        self.{local_fu} = FUClass_{local_fu}('{local_fu}', self.env, parent_scope=self.scope, alias='{local_fu}')")
        lines.append(f"        self.{local_fu}.parent = self")
        lines.append(f"        self.fus['{local_fu}'] = self.{local_fu}")
        lines.append("")

    # create interconnections between FUs
    lines.append("        # --- interconnections between FUs ---")
    for conn in connections:
        iid = conn.get("id")
        etype = conn.get("type")
        frm = conn.get("from", {})
        to = conn.get("to", {})
        params = conn.get("params", {}) or {}
        param_text = _render_params(params)
        src_fu = frm.get("fu")
        src_node = frm.get("node")
        dst_fu = to.get("fu")
        dst_node = to.get("node")
        lines.append(f"        # interconnection: {iid} {src_fu}.{src_node} -> {dst_fu}.{dst_node}")
        lines.append(f"        EModule = importlib.import_module('factorysimpy.edges.{etype.lower()}')")
        lines.append(f"        EdgeClass = getattr(EModule, '{etype}')")
        lines.append(f"        try:")
        lines.append(f"            self.{iid} = EdgeClass(self.env, f\"{{self.scope}}.{iid}\", self.{src_fu}.node_objs['{src_node}'], self.{dst_fu}.node_objs['{dst_node}']{param_text})")
        lines.append("        except TypeError:")
        lines.append(f"            self.{iid} = EdgeClass(self.env, f\"{{self.scope}}.{iid}\"{param_text})")
        lines.append(f"            if hasattr(self.{iid}, 'connect'):")
        lines.append(f"                try:")
        lines.append(f"                    self.{iid}.connect(self.{src_fu}.node_objs['{src_node}'], self.{dst_fu}.node_objs['{dst_node}'])")
        lines.append("                except Exception:")
        lines.append("                    pass")
        lines.append(f"            self.{iid}.from_node = self.{src_fu}.node_objs['{src_node}']")
        lines.append(f"            self.{iid}.to_node = self.{dst_fu}.node_objs['{dst_node}']")
        lines.append(f"        self.{iid}.scope = f\"{{self.scope}}.{iid}\"")
        lines.append(f"        self.{iid}.parent = self")
        lines.append(f"        self.interedges['{iid}'] = self.{iid}")
        lines.append("")

    lines.append("    def __repr__(self):")
    lines.append(f"        return f\"{class_name}(id={{self.id}}, scope={{self.scope}})\"")
    return "\n".join(lines)

# -------------------------
# run_sim.py renderer
# -------------------------
def _render_run_script(system) -> str:
    lines = []
    lines.append("import simpy")
    lines.append("import sys")
    lines.append("")
    lines.append("def main():")
    lines.append("    env = simpy.Environment()")
    lines.append("")

    # Prefer GU instances already present in system (they are real objects)
    gu_instances = getattr(system, "gu_instances", {}) or {}
    if gu_instances:
        lines.append("    # instantiate GU instances as per the model")
        for gid, guobj in gu_instances.items():
            # guobj.__class__ likely is the runtime class created by builder; we want the generated class name derived from template id
            # We'll attempt to get the class name from the template id if available; fall back to class __name__
            gu_clsname = getattr(guobj, "__class__").__name__
            # The generated file name will be safe ident of template id; assume gu_clsname matches that or user can edit.
            lines.append(f"    try:")
            lines.append(f"        from gridunits.{gu_clsname} import {gu_clsname}")
            lines.append(f"        gu = {gu_clsname}('{gid}', env)")
            lines.append(f"        print('Created GU:', gu)")
            lines.append(f"    except Exception as e:")
            lines.append(f"        print('Could not auto-import GU class {gu_clsname}, please verify generated files.', e)")
            lines.append("")
    else:
        # instantiate one example for each GU template (use gu_classes names)
        lines.append("    # No GU instances recorded — instantiate one example GU per GU template")
        for guname in (getattr(system, "gu_classes", {}) or {}).keys():
            safe = _safe_ident(guname)
            lines.append(f"    try:")
            lines.append(f"        from gridunits.{safe} import {safe}")
            lines.append(f"        gu = {safe}('{safe}_inst', env)")
            lines.append(f"        print('Created GU:', gu)")
            lines.append(f"    except Exception as e:")
            lines.append(f"        print('Could not import/instantiate GU', '{safe}', e)")
            lines.append("")

    lines.append("    env.run()")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    return "\n".join(lines)

# -------------------------
# Main generator
# -------------------------
def generate_python_model(system: Any, output_dir: str = "generated_model", zip_output: bool = True, overwrite: bool = True) -> str:
    """
    Generate a Python package for the provided ModelSystem (builder.get_system()).

    Args:
        system: ModelSystem returned by ModelBuilder.get_system()
        output_dir: directory to write package
        zip_output: if True, create and return a .zip path
        overwrite: if True and output_dir exists, remove it first

    Returns:
        path to zip (if zip_output True) or directory path
    """
    out = Path(output_dir)
    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            raise FileExistsError(output_dir + " exists")
    out.mkdir(parents=True, exist_ok=True)

    (out / "flowunits").mkdir(parents=True, exist_ok=True)
    (out / "gridunits").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    # metadata
    metadata = {
        "fu_templates": list(getattr(system, "fu_templates", {}).keys()),
        "gu_templates": list(getattr(system, "gu_templates", {}).keys()),
        "fu_instances": list(getattr(system, "fu_instances", {}).keys()),
        "gu_instances": list(getattr(system, "gu_instances", {}).keys()),
    }

    # --- FlowUnit files ---
    fu_templates = getattr(system, "fu_templates", {}) or {}
    for tid, tmpl in fu_templates.items():
        fname = _safe_ident(tid) + ".py"
        code = _render_flowunit_py(tid, tmpl)
        with open(out / "flowunits" / fname, "w", encoding="utf-8") as f:
            f.write("# Auto-generated FlowUnit module\n")
            f.write(code)
            f.write("\n")

    # flowunits __init__
    with open(out / "flowunits" / "__init__.py", "w", encoding="utf-8") as f:
        for tid in fu_templates.keys():
            safe = _safe_ident(tid)
            f.write(f"from .{safe} import {safe}\n")

    # --- GridUnit files ---
    gu_templates = getattr(system, "gu_templates", {}) or {}
    for gid, gtmpl in gu_templates.items():
        fu_map = gu_get_fu_map(gtmpl, fu_templates)
        # fu_map: local -> template name
        fname = _safe_ident(gid) + ".py"
        code = _render_gridunit_py(gid, gtmpl, fu_map)
        with open(out / "gridunits" / fname, "w", encoding="utf-8") as f:
            f.write("# Auto-generated GridUnit module\n")
            f.write(code)
            f.write("\n")

    # gridunits __init__
    with open(out / "gridunits" / "__init__.py", "w", encoding="utf-8") as f:
        for gid in gu_templates.keys():
            safe = _safe_ident(gid)
            f.write(f"from .{safe} import {safe}\n")

    # run script
    with open(out / "run_sim.py", "w", encoding="utf-8") as f:
        f.write("# Auto-generated runner\n")
        f.write(_render_run_script(system))
        f.write("\n")

    # metadata file
    with open(out / "data" / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # package root __init__
    with open(out / "__init__.py", "w", encoding="utf-8") as f:
        f.write("# generated package\n")

    # zip if requested
    if zip_output:
        zip_path = str(out) + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(out):
                for file in files:
                    full = Path(root) / file
                    # preserve directory name relative to out.parent so generated_model.zip contains folder
                    arcname = full.relative_to(out.parent)
                    zf.write(full, arcname)
        return zip_path
    return str(out)

# End of codegenerator.py

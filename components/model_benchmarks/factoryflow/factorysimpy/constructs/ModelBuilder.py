# builder_final.py
import copy, os, sys
import importlib
from typing import Dict, Any, List, Optional

from codegenerator import generate_python_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))



def _load_node_class(type_name: str):
    try:
        module = importlib.import_module(f"factorysimpy.nodes.{type_name.lower()}")
        return getattr(module, type_name)
    except Exception:
        raise ImportError(f"Node class '{type_name}' could not be loaded.")

def _load_edge_class(type_name: str):
    try:
        module = importlib.import_module(f"factorysimpy.edges.{type_name.lower()}")
        return getattr(module, type_name)
    except Exception:
        raise ImportError(f"Edge class '{type_name}' could not be loaded.")

# -----------------------
# FlowUnitTemplate
# -----------------------
class FlowUnitTemplate:
    """
    Global blueprint for a FlowUnit.
    Supports template-level mods via 'mods' dict (override/add/delete).
    """
    def __init__(self, id: str, nodes: List[Dict[str,Any]], edges: List[Dict[str,Any]],
                 alias: Optional[str]=None, mods: Optional[Dict[str,Any]] = None):
        self.id = id
        self.alias = alias or id
        self.base_nodes = {n["id"]: copy.deepcopy(n) for n in (nodes or [])}
        self.base_edges = {e["id"]: copy.deepcopy(e) for e in (edges or [])}
        default_mods = {
            "override_nodes": {},
            "override_edges": {},
            "add_nodes": [],
            "add_edges": [],
            "delete_nodes": [],
            "delete_edges": []
        }
        self.mods = copy.deepcopy(mods or {})
        for k,v in default_mods.items():
            self.mods.setdefault(k, copy.deepcopy(v))

    # apply template-level mods to dicts of nodes/edges (local ids)
    def _apply_template_mods(self, nodes: Dict[str,Any], edges: Dict[str,Any]):
        # delete nodes and attached edges
        for nid in self.mods.get("delete_nodes", []):
            nodes.pop(nid, None)
            edges = {eid:e for eid,e in edges.items() if e.get("src") != nid and e.get("dest") != nid}
        # delete edges
        for eid in self.mods.get("delete_edges", []):
            edges.pop(eid, None)
        # add nodes
        for nd in self.mods.get("add_nodes", []):
            nodes[nd["id"]] = copy.deepcopy(nd)
        # add edges
        for ed in self.mods.get("add_edges", []):
            eid = ed["id"]
            if eid in edges:
                continue
            edges[ed["id"]] = copy.deepcopy(ed)
        # override nodes
        for nid, params in self.mods.get("override_nodes", {}).items():
            if nid in nodes:
                nodes[nid].setdefault("params", {})
                nodes[nid]["params"].update(params)
        # override edges
        for eid, params in self.mods.get("override_edges", {}).items():
            if eid in edges:
                edges[eid].setdefault("params", {})
                edges[eid]["params"].update(params)
        return nodes, edges

    def build_class(self):
        template = copy.deepcopy(self)

        class FlowUnit:
            def __init__(self, id: str, env: Optional[Any]=None, parent_scope: Optional[str]=None, alias:Optional[str]=None):
                self.id = id
                self.alias = alias or template.alias or id
                self.env = env
                self.parent_scope = parent_scope
                self.scope = f"{parent_scope}.{self.id}" if parent_scope else self.id

                # start from template base
                nodes = copy.deepcopy(template.base_nodes)
                edges = copy.deepcopy(template.base_edges)

                # apply template-level mods automatically
                nodes, edges = template._apply_template_mods(nodes, edges)

                # scope nodes and edges: keep keyed by local id but include scoped_id
                self.nodes = {}
                for nid, nd in nodes.items():
                    ndc = copy.deepcopy(nd)
                    ndc["scoped_id"] = f"{self.scope}.{nid}"
                    self.nodes[nid] = ndc

                self.edges = {}
                for eid, ed in edges.items():
                    edc = copy.deepcopy(ed)
                    edc["scoped_id"] = f"{self.scope}.{eid}"
                    self.edges[eid] = edc

                # instantiate node objects
                self.node_objs = {}
                for nid, nd in self.nodes.items():
                    NodeClass = _load_node_class(nd.get("type"))
                    params = dict(nd.get("params", {}))
                    print(params)
                    #params.setdefault("env", env)
                    print(params, nd["scoped_id"])
                    obj = NodeClass(env, nd["scoped_id"], **params)
                   
                    obj.in_edges = getattr(obj, "in_edges", [])
                    obj.out_edges = getattr(obj, "out_edges", [])
                    obj.scope = nd["scoped_id"]
                    obj.alias = nd.get("alias")
                    obj.parent = self
                    self.node_objs[nid] = obj
                    setattr(self, nid, obj)

                # instantiate edge objects and wire them (if src/dest exist)
                self.edge_objs = {}
                for eid, ed in self.edges.items():
                    EdgeClass = _load_edge_class(ed.get("type"))
                    params = dict(ed.get("params", {}))
                    #params.setdefault("env", env)
                    src_local = ed.get("src")
                    dest_local = ed.get("dest")
                    src_obj = self.node_objs.get(src_local)
                    dest_obj = self.node_objs.get(dest_local)
                    try:
                        #print("hello",env, ed["scoped_id"], src_obj, dest_obj, **params)
                        eobj = EdgeClass(env, ed["scoped_id"], src_obj, dest_obj, **params)
                    except TypeError:
                        eobj = EdgeClass(env, ed["scoped_id"], **params)
                        if src_obj and hasattr(eobj, "connect"):
                            try:
                                print("b4",eobj.src_node)
                                eobj.connect(src_obj, dest_obj)
                                print("after",eobj.src_node)
                            except Exception:
                                pass
                        eobj.from_node = src_obj
                        eobj.to_node = dest_obj
                    eobj.scope = ed["scoped_id"]
                    eobj.alias = ed.get("alias")
                    eobj.parent = self
                    # bookkeeping
                    # if src_obj:
                    #     src_obj.out_edges.append(eobj)
                    # if dest_obj:
                    #     dest_obj.in_edges.append(eobj)
                    self.edge_objs[eid] = eobj
                    setattr(self, eid, eobj)

                # prepare instance_mods (populated from JSON automatically)
                self.instance_mods = {
                    "override_nodes": {},
                    "override_edges": {},
                    "add_nodes": [],
                    "add_edges": [],
                    "delete_nodes": [],
                    "delete_edges": []
                }

            # instance-level apply method (will be invoked automatically if JSON supplies mods)
            def apply_instance_mods(self):
                # delete nodes safely (remove attached edges)
                for nid in self.instance_mods.get("delete_nodes", []):
                    if nid in self.node_objs:
                        node_obj = self.node_objs[nid]
                        # remove incoming edges
                        for e in list(node_obj.in_edges):
                            if e in getattr(e.from_node, "out_edges", []):
                                try:
                                    e.from_node.out_edges.remove(e)
                                except Exception:
                                    pass
                            # remove from edge_objs
                            for ek, eo in list(self.edge_objs.items()):
                                if eo is e:
                                    self.edge_objs.pop(ek, None)
                                    try:
                                        delattr(self, ek)
                                    except Exception:
                                        pass
                        # remove outgoing edges
                        for e in list(node_obj.out_edges):
                            if e in getattr(e.to_node, "in_edges", []):
                                try:
                                    e.to_node.in_edges.remove(e)
                                except Exception:
                                    pass
                            for ek, eo in list(self.edge_objs.items()):
                                if eo is e:
                                    self.edge_objs.pop(ek, None)
                                    try:
                                        delattr(self, ek)
                                    except Exception:
                                        pass
                        # finally remove node
                        self.node_objs.pop(nid, None)
                        try:
                            delattr(self, nid)
                        except Exception:
                            pass
                        self.nodes.pop(nid, None)

                # delete edges
                for eid in self.instance_mods.get("delete_edges", []):
                    eo = self.edge_objs.pop(eid, None)
                    if eo:
                        if getattr(eo, "from_node", None):
                            try:
                                eo.from_node.out_edges.remove(eo)
                            except Exception:
                                pass
                        if getattr(eo, "to_node", None):
                            try:
                                eo.to_node.in_edges.remove(eo)
                            except Exception:
                                pass
                        try:
                            delattr(self, eid)
                        except Exception:
                            pass
                        self.edges.pop(eid, None)

                # add nodes
                for nd in self.instance_mods.get("add_nodes", []):
                    nid = nd["id"]
                    ndc = copy.deepcopy(nd)
                    ndc["scoped_id"] = f"{self.scope}.{nid}"
                    self.nodes[nid] = ndc
                    NodeClass = _load_node_class(ndc.get("type"))
                    params = dict(ndc.get("params", {}))
                    #params.setdefault("env", self.env)
                    obj = NodeClass(self.env, ndc["scoped_id"], **params)
                    obj.in_edges = getattr(obj, "in_edges", [])
                    obj.out_edges = getattr(obj, "out_edges", [])
                    obj.scope = ndc["scoped_id"]
                    obj.alias = ndc.get("alias")
                    obj.parent = self
                    self.node_objs[nid] = obj
                    setattr(self, nid, obj)

                # add edges
                for ed in self.instance_mods.get("add_edges", []):
                    eid = ed["id"]
                    if eid  in self.edge_objs:
                        continue
                    edc = copy.deepcopy(ed)
                    edc["scoped_id"] = f"{self.scope}.{eid}"
                    self.edges[eid] = edc
                    EdgeClass = _load_edge_class(edc.get("type"))
                    params = dict(edc.get("params", {}))
                    #params.setdefault("env", self.env)
                    src_local = edc.get("src")
                    dest_local = edc.get("dest")
                    src_obj = self.node_objs.get(src_local)
                    dest_obj = self.node_objs.get(dest_local)
                    try:
                        eobj = EdgeClass(self.env, edc["scoped_id"], src_obj, dest_obj, **params)
                    except TypeError:
                        eobj = EdgeClass(self.env, edc["scoped_id"], **params)
                        if hasattr(eobj, "connect") and src_obj:
                            try:
                                print("b42",eobj.src_node)
                                eobj.connect(src_obj, dest_obj)
                                print("after2",eobj.src_node)
                            except Exception:
                                pass
                        eobj.from_node = src_obj
                        eobj.to_node = dest_obj
                    eobj.scope = edc["scoped_id"]
                    eobj.alias = edc.get("alias")
                    eobj.parent = self
                    # if src_obj:
                    #     src_obj.out_edges.append(eobj)
                    # if dest_obj:
                    #     dest_obj.in_edges.append(eobj)
                    self.edge_objs[eid] = eobj
                    setattr(self, eid, eobj)

                # override nodes
                for nid, params in self.instance_mods.get("override_nodes", {}).items():
                    if nid in self.node_objs:
                        obj = self.node_objs[nid]
                        obj.params = getattr(obj, "params", {})
                        obj.params.update(params)

                # override edges
                for eid, params in self.instance_mods.get("override_edges", {}).items():
                    if eid in self.edge_objs:
                        eo = self.edge_objs[eid]
                        eo.params.update(params)

            def __repr__(self):
                return f"FlowUnit(scope={self.scope}, alias={self.alias})"

        return FlowUnit

# -----------------------
# GridUnitTemplate
# -----------------------
class GridUnitTemplate:
    """
    GU references global FlowUnit template IDs via fu_map: {local_name: template_id}
    interconnections list uses local fu names and node names.
    GU-level 'mods' supported in template and instances.
    """
    def __init__(self, id: str, fu_map: Dict[str,str], interconnections: Optional[List[Dict[str,Any]]] = None,
                 alias: Optional[str]=None, mods: Optional[Dict[str,Any]] = None):
        self.id = id
        self.alias = alias or id
        self.fu_map = copy.deepcopy(fu_map or {})  # local->template id
        self.interconnections = copy.deepcopy(interconnections or [])
        default_mods = {
            "override_interconnections": {},
            "add_interconnections": [],
            "delete_interconnections": []
        }
        self.mods = copy.deepcopy(mods or {})
        for k,v in default_mods.items():
            self.mods.setdefault(k, copy.deepcopy(v))

    def _apply_template_mods(self, inters: List[Dict[str,Any]]):
        inters = [c for c in inters if c["id"] not in set(self.mods.get("delete_interconnections", []))]
        for c in self.mods.get("add_interconnections", []):
            inters.append(copy.deepcopy(c))
        for iid, params in self.mods.get("override_interconnections", {}).items():
            for c in inters:
                if c["id"] == iid:
                    c.setdefault("params", {}).update(params)
        return inters

    def build_class(self, fu_templates_lookup: Dict[str, FlowUnitTemplate]):
        template = copy.deepcopy(self)
        # resolve fu_map strings -> FlowUnitTemplate objects
        resolved = {}
        for local_name, fu_tid in template.fu_map.items():
            if fu_tid not in fu_templates_lookup:
                raise KeyError(f"FlowUnit template '{fu_tid}' not found for GU '{self.id}'")
            resolved[local_name] = fu_templates_lookup[fu_tid]
        template.fu_map = resolved

        class GridUnit:
            def __init__(self, id: str, env: Optional[Any]=None, parent_scope:Optional[str]=None, alias:Optional[str]=None):
                self.id = id
                self.alias = alias or template.alias or id
                self.env = env
                self.parent_scope = parent_scope
                self.scope = f"{parent_scope}.{self.id}" if parent_scope else self.id

                # instantiate fresh FU instances per mapping
                self.fus = {}
                for local_name, fu_template in template.fu_map.items():
                    FUClass = fu_template.build_class()
                    fu_inst = FUClass(local_name, env, parent_scope=self.scope, alias=local_name)
                    self.fus[local_name] = fu_inst
                    setattr(self, local_name, fu_inst)

                # apply GU template mods to interconnections and then create edges
                inters = copy.deepcopy(template.interconnections)
                inters = template._apply_template_mods(inters)
                self.interedges = {}
                for conn in inters:
                    iid = conn["id"]
                    typ = conn.get("type")
                    frm = conn["from"]
                    to = conn["to"]
                    params = dict(conn.get("params", {}))
                    #params.setdefault("env", env)

                    from_fu = self.fus[frm["fu"]]
                    to_fu = self.fus[to["fu"]]
                    from_node_obj = from_fu.node_objs.get(frm["node"])
                    to_node_obj = to_fu.node_objs.get(to["node"])

                    EdgeClass = _load_edge_class(typ)
                    try:
                        edge_obj = EdgeClass(env, f"{self.scope}.{iid}", from_node_obj, to_node_obj, **params)
                    except TypeError:
                        edge_obj = EdgeClass(env,f"{self.scope}.{iid}", **params)
                        if hasattr(edge_obj, "connect"):
                            try:
                                print("b4 gu interedge",edge_obj.src_node)
                                edge_obj.connect(from_node_obj, to_node_obj)
                                print("after gu interedge",edge_obj.src_node)
                            except Exception:
                                pass
                        edge_obj.from_node = from_node_obj
                        edge_obj.to_node = to_node_obj

                    # if from_node_obj:
                    #     from_node_obj.out_edges.append(edge_obj)
                    # if to_node_obj:
                    #     to_node_obj.in_edges.append(edge_obj)

                    edge_obj.scope = f"{self.scope}.{iid}"
                    edge_obj.parent = self
                    self.interedges[iid] = edge_obj
                    setattr(self, iid, edge_obj)

                # GU instance-level mods storage (applied automatically if provided in JSON)
                self.instance_mods = {
                    "override_interconnections": {},
                    "add_interconnections": [],
                    "delete_interconnections": []
                }

            def apply_instance_mods(self):
                # delete interconnections
                for iid in self.instance_mods.get("delete_interconnections", []):
                    eo = self.interedges.pop(iid, None)
                    if eo:
                        if getattr(eo, "from_node", None):
                            try:
                                eo.from_node.out_edges.remove(eo)
                            except Exception:
                                pass
                        if getattr(eo, "to_node", None):
                            try:
                                eo.to_node.in_edges.remove(eo)
                            except Exception:
                                pass
                        try:
                            delattr(self, iid)
                        except Exception:
                            pass
                # add interconnections
                for conn in self.instance_mods.get("add_interconnections", []):
                    iid = conn["id"]
                    if iid in self.interedges:
                        continue
                    typ = conn.get("type")
                    frm = conn["from"]
                    to = conn["to"]
                    params = dict(conn.get("params", {}))
                    # params.setdefault("env", self.env)
                    from_fu = self.fus[frm["fu"]]
                    to_fu = self.fus[to["fu"]]
                    from_node = from_fu.node_objs.get(frm["node"])
                    to_node = to_fu.node_objs.get(to["node"])
                    EdgeClass = _load_edge_class(typ)
                    try:
                        edge_obj = EdgeClass(self.env, f"{self.scope}.{iid}", from_node, to_node, **params)
                    except TypeError:
                        edge_obj = EdgeClass(self.env, f"{self.scope}.{iid}", **params)
                        if hasattr(edge_obj, "connect"):
                            try:
                                print("b4 gu interedge inst",edge_obj.src_node)
                                edge_obj.connect(from_node, to_node)
                                print("after gu interedge inst",edge_obj.src_node)
                            except Exception:
                                pass
                        edge_obj.from_node = from_node
                        edge_obj.to_node = to_node
                    # if from_node:
                    #     from_node.out_edges.append(edge_obj)
                    # if to_node:
                    #     to_node.in_edges.append(edge_obj)
                    edge_obj.scope = f"{self.scope}.{iid}"
                    edge_obj.parent = self
                    self.interedges[iid] = edge_obj
                    setattr(self, iid, edge_obj)
                # override interconnections
                for iid, params in self.instance_mods.get("override_interconnections", {}).items():
                    if iid in self.interedges:
                        print(self.interedges[iid], params)
                        self.interedges[iid].update(params)

            def __repr__(self):
                return f"GridUnit(scope={self.scope}, alias={self.alias})"

        return GridUnit

# -----------------------
# ModelBuilder
# -----------------------

class ModelSystem:
    """
    A simple container for the final built simulation system.
    Holds FU templates, FU classes, FU instances, GU templates,
    GU classes, and GU instances.

    Does NOT contain any logic. Only a structured, readable object.
    """

    def __init__(self):
        self.fu_templates = {}
        self.fu_classes = {}
        self.fu_instances = {}

        self.gu_templates = {}
        self.gu_classes = {}
        self.gu_instances = {}

    def __repr__(self):
        return (
            f"ModelSystem("
            f"FU_templates={len(self.fu_templates)}, "
            f"FU_classes={len(self.fu_classes)}, "
            f"FU_instances={len(self.fu_instances)}, "
            f"GU_templates={len(self.gu_templates)}, "
            f"GU_classes={len(self.gu_classes)}, "
            f"GU_instances={len(self.gu_instances)})"
        )


class ModelBuilder:
    def __init__(self, model_json: Dict[str,Any]):
        raw = copy.deepcopy(model_json.get("model") or model_json)
        self.raw = raw
        self.model_id = raw.get("id", "Model")
        self.fu_templates: Dict[str, FlowUnitTemplate] = {}
        self.gu_templates: Dict[str, GridUnitTemplate] = {}
        self.fu_classes: Dict[str,Any] = {}
        self.gu_classes: Dict[str,Any] = {}
        self.fu_instances: Dict[str,Any] = {}
        self.gu_instances: Dict[str,Any] = {}

    def get_system(self):
        """
        Returns a fully structured ModelSystem object.
        Contains:
        - FU templates
        - FU classes
        - FU instances
        - GU templates
        - GU classes
        - GU instances
        """
        system = ModelSystem()

        # Copy references
        system.fu_templates = dict(self.fu_templates)
        system.fu_classes = dict(self.fu_classes)
        system.fu_instances = dict(self.fu_instances)

        system.gu_templates = dict(self.gu_templates)
        system.gu_classes = dict(self.gu_classes)
        system.gu_instances = dict(self.gu_instances)

        return system


    
    def build(self, make_instances: bool = True, env: Optional[Any] = None):
        # 1) Load FU templates: support derived templates via instance_of
        fu_section = self.raw.get("FlowUnits", {})
        pending = []
        for ft in fu_section.get("fu_templates", []):
            if ft.get("instance_of"):
                pending.append(ft)
                continue
            tid = ft["id"]
            tmpl = FlowUnitTemplate(id=tid, nodes=ft.get("nodes", []), edges=ft.get("edges", []),
                                    alias=ft.get("alias"), mods=ft.get("mods"))
            self.fu_templates[tid] = tmpl
        # derived
        for ft in pending:
            base = ft["instance_of"]
            if base not in self.fu_templates:
                raise KeyError(f"Derived FU template base '{base}' not found for '{ft['id']}'")
            base_t = self.fu_templates[base]
            nodes = [copy.deepcopy(n) for n in base_t.base_nodes.values()]
            edges = [copy.deepcopy(e) for e in base_t.base_edges.values()]
            derived = FlowUnitTemplate(id=ft["id"], nodes=nodes, edges=edges, alias=ft.get("alias"), mods=ft.get("mods"))
            self.fu_templates[ft["id"]] = derived

        # 2) build FU classes
        for tid, tmpl in self.fu_templates.items():
            self.fu_classes[tid] = tmpl.build_class()

        # 3) Load GU templates
        gu_section = self.raw.get("GridUnits", {})
        for gt in gu_section.get("gu_templates", []):
            gid = gt["id"]
            gu_t = GridUnitTemplate(id=gid, fu_map=gt.get("fu_templates", {}), interconnections=gt.get("interconnections", []),
                                    alias=gt.get("alias"), mods=gt.get("mods"))
            self.gu_templates[gid] = gu_t

        # 4) build GU classes
        for gid, gu_t in self.gu_templates.items():
            self.gu_classes[gid] = gu_t.build_class(self.fu_templates)

        # 5) instantiate top-level FU instances (flowunit_instances) and apply instance-level mods automatically
        for inst in fu_section.get("flowunit_instances", []):
            if not make_instances: continue
            iid = inst["id"]
            base = inst["instance_of"]
            alias = inst.get("alias")
            if base not in self.fu_classes:
                raise KeyError(f"FU template '{base}' not found for instance '{iid}'")
            FUClass = self.fu_classes[base]
            fu_obj = FUClass(iid, env, parent_scope=self.model_id, alias=alias)
            # apply instance-level mods if provided (Option A: mods inside the instance block)
            mods = inst.get("mods", {})
            if mods:
                # merge into fu_obj.instance_mods and apply
                for k in fu_obj.instance_mods.keys():
                    if k in mods:
                        fu_obj.instance_mods[k] = copy.deepcopy(mods[k])
                fu_obj.apply_instance_mods()
            self.fu_instances[iid] = fu_obj

        # 6) instantiate GU instances and apply instance-level mods automatically (mods inside GU instance)
        for inst in gu_section.get("gridunit_instances", []):
            if not make_instances: continue
            iid = inst["id"]
            base = inst["instance_of"]
            alias = inst.get("alias")
            if base not in self.gu_classes:
                raise KeyError(f"GU template '{base}' not found for instance '{iid}'")
            GUClass = self.gu_classes[base]
            gu_obj = GUClass(iid, env, parent_scope=self.model_id, alias=alias)
            # GU instance-level mods may include FU instance-specific mods for FUs inside the GU,
            # and GU interconnection mods. We support both:
            mods = inst.get("mods", {})
            if mods:
                # 6.a: FU inside GU instance-level mods: expect mods.get("flowunits", {local_name: {mods}})
                for fu_local, fu_mods in mods.get("flowunits", {}).items():
                    if fu_local in gu_obj.fus:
                        fu_inst = gu_obj.fus[fu_local]
                        for k in fu_inst.instance_mods.keys():
                            if k in fu_mods:
                                fu_inst.instance_mods[k] = copy.deepcopy(fu_mods[k])
                        fu_inst.apply_instance_mods()
                # 6.b: GU interconnection mods
                # expected keys: override_interconnections, add_interconnections, delete_interconnections
                for k in ["override_interconnections", "add_interconnections", "delete_interconnections"]:
                    if k in mods:
                        gu_obj.instance_mods[k] = copy.deepcopy(mods[k])
                gu_obj.apply_instance_mods()
            self.gu_instances[iid] = gu_obj

        return self

# -----------------------
# Example usage and JSON
# -----------------------
if __name__ == "__main__":
    # Example JSON follows Option A: instance-level 'mods' inside each instance block.
    example = {
      "model": {
        "id": "Plant1",
        "description": "Example model for builder_final",

        "FlowUnits": {
          "fu_templates": [
            {
              "id": "flowunit_3_machine_template",
              "nodes": [
                {"id": "SRC1", "type": "Source", "params": {"inter_arrival_time": 2}},
                {"id": "M01", "type": "Machine", "params": {"work_capacity": 1}},
                {"id": "M02", "type": "Machine", "params": {"work_capacity": 1}},
                {"id": "M03", "type": "Machine", "params": {"work_capacity": 1}},
                {"id": "SINK1", "type": "Sink", "params": {}}
              ],
              "edges": [
                {"id": "B01", "type": "Buffer", "src": "SRC1", "dest": "M01", "params": {"capacity": 3}},
                {"id": "B02", "type": "Buffer", "src": "M01", "dest": "M02", "params": {"capacity": 4}},
                {"id": "B03", "type": "Buffer", "src": "M02", "dest": "M03", "params": {"capacity": 2}},
                {"id": "B04", "type": "Buffer", "src": "M03", "dest": "SINK1", "params": {"capacity": 5}}
              ]
            },
            {
              "id": "flowunit_3_machine_template_fastM02",
              "instance_of": "flowunit_3_machine_template",
              "mods": {
                "override_nodes": {"M02": {"work_capacity": 8}}
              }
            }
          ],
          "flowunit_instances": [
            {
              "id": "StandaloneFU",
              "instance_of": "flowunit_3_machine_template",
              "mods": {
                "override_nodes": {"M01": {"work_capacity": 5}},
                
              }
            }
          ]
        },

        "GridUnits": {
          "gu_templates": [
            {
              "id": "GridUnit_3x3_template",
              "fu_templates": {
                "FUA": "flowunit_3_machine_template",
                "FUB": "flowunit_3_machine_template",
                "FUC": "flowunit_3_machine_template_fastM02"
              },
              "interconnections": [
                {"id": "IA1", "type": "Buffer", "from": {"fu": "FUA", "node": "M02"}, "to": {"fu": "FUB", "node": "M02"}, "params": {"capacity": 4}},
                {"id": "IA2", "type": "Buffer", "from": {"fu": "FUB", "node": "M03"}, "to": {"fu": "FUC", "node": "M01"}, "params": {"capacity": 6}}
              ]
            }
          ],
          "gridunit_instances": [
            {
              "id": "GU1",
              "instance_of": "GridUnit_3x3_template",
              "mods": {
                # FU-level instance mods for fu local name FUB
                "flowunits": {
                  "FUB": {
                    "override_nodes": {"M02": {"work_capacity": 20}},
                    "add_nodes": [{"id": "M_extra", "type": "Machine", "params": {"work_capacity": 2}}],
                    "add_edges": [{"id": "B_extra", "type": "Buffer", "src": "M_extra", "dest": "M03", "params": {"capacity": 1}},
                                  {"id": "B_in_extra", "type": "Buffer", "src": "M02", "dest": "M_extra", "params": {"capacity": 1}}]
                  }
                },
                # GU-level interconnection mods
                "override_interconnections": {"IA1": {"params": {"capacity": 10}}},
                "add_interconnections": [
                  {"id": "IA3", "type": "Buffer", "from": {"fu":"FUC","node":"M03"}, "to": {"fu":"FUA","node":"M01"}, "params": {"capacity": 2}}
                ],
                "delete_interconnections": []
              }
            }
          ]
        }
      }
    }

    # usage
    import simpy
    env= simpy.Environment()
    builder = ModelBuilder(example).build(make_instances=True, env=env)
    print("FU templates:", list(builder.fu_templates.keys()))
    print("GU templates:", list(builder.gu_templates.keys()))
    print("FU instances:", list(builder.fu_instances.keys()))
    print("GU instances:", list(builder.gu_instances.keys()))

    gu = builder.gu_instances["GU1"]
    fu= builder.fu_instances["StandaloneFU"]
    print(gu)                        # GridUnit(scope=Plant1.GU1,...)
    print("FUB M02 params:", gu.FUB.M02.params)      # changed to 20 by instance mod
    print("standalone M01 work_capacity:", fu.M01.params)
    print("FUB has extra node M_extra?", hasattr(gu.FUB, "M_extra"))
    print("Interedges:", list(gu.interedges.keys()))

    system = builder.get_system()
    
    from codegenerator import generate_python_model
    # generate package
    # out_zip = generate_python_model(system, output_dir="generated_model", zip_output=True)
    # print("Generated:", out_zip)

    #env.run(until=25)

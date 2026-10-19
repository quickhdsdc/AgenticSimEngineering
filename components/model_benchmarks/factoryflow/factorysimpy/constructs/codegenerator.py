# ============================================================
# codegenerator.py
# Generates a standalone Python simulation package from a
# ModelSystem produced by builder_final.ModelBuilder
# ============================================================

import os, json, shutil, zipfile


# ============================================================
# Helper: Ensure directory exists
# ============================================================
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ============================================================
# Helper: Write text file
# ============================================================
def _write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ============================================================
# Serialize the system object into a minimal JSON IR
# ============================================================
def _serialize_system_to_json(system):
    """
    Convert ModelSystem (with FU & GU templates/instances)
    into a JSON IR that can be reloaded via build_model.py.
    """
    out = {
        "FlowUnits": {
            "fu_templates": [],
            "flowunit_instances": []
        },
        "GridUnits": {
            "gu_templates": [],
            "gridunit_instances": []
        }
    }

    # --------------------------
    # FU templates
    # --------------------------
    for tid, tmpl in system.fu_templates.items():
        data = {
            "id": tmpl.id,
            "nodes": list(tmpl.base_nodes.values()),
            "edges": list(tmpl.base_edges.values()),
            "mods": tmpl.mods
        }
        out["FlowUnits"]["fu_templates"].append(data)

    # --------------------------
    # FU instances
    # --------------------------
    for iid, inst in system.fu_instances.items():
        inst_data = {
            "id": inst.id,
            "instance_of": inst.alias if inst.alias else None,
            "mods": inst.instance_mods
        }
        out["FlowUnits"]["flowunit_instances"].append(inst_data)

    # --------------------------
    # GU templates
    # --------------------------
    for gid, tmpl in system.gu_templates.items():
        data = {
            "id": tmpl.id,
            "fu_templates": tmpl.fu_map,  # mapping local_name → template_id
            "interconnections": tmpl.interconnections,
            "mods": tmpl.mods
        }
        out["GridUnits"]["gu_templates"].append(data)

    # --------------------------
    # GU instances
    # --------------------------
    for iid, inst in system.gu_instances.items():
        inst_data = {
            "id": inst.id,
            "instance_of": inst.alias if inst.alias else None,
            "mods": inst.instance_mods
        }
        out["GridUnits"]["gridunit_instances"].append(inst_data)

    return out


# ============================================================
# Generate build_model.py
# ============================================================
def _generate_build_model_py():
    """
    Creates a generic builder that reconstructs FUs and GUs from JSON.
    This file does NOT contain template logic itself.
    It simply invokes the original FlowUnitTemplate and GridUnitTemplate.
    """
    return """
import json
import simpy
from builder_final import FlowUnitTemplate, GridUnitTemplate


def build_model(env, config):
    \"\"\"
    Reconstruct FlowUnit and GridUnit templates + instances
    from the supplied JSON config.
    Returns a dict:
        {
           "env": env,
           "flowunits": {...},
           "gridunits": {...}
        }
    \"\"\"

    # ----------------------------
    # 1) Build FU templates
    # ----------------------------
    fu_templates_lookup = {}
    for ft in config["FlowUnits"]["fu_templates"]:
        tmpl = FlowUnitTemplate(
            id=ft["id"],
            nodes=ft.get("nodes", []),
            edges=ft.get("edges", []),
            mods=ft.get("mods")
        )
        fu_templates_lookup[ft["id"]] = tmpl

    # ----------------------------
    # 2) Build GU templates
    # ----------------------------
    gu_templates_lookup = {}
    for gt in config["GridUnits"]["gu_templates"]:
        tmpl = GridUnitTemplate(
            id=gt["id"],
            fu_map=gt.get("fu_templates", {}),
            interconnections=gt.get("interconnections", []),
            mods=gt.get("mods")
        )
        gu_templates_lookup[gt["id"]] = tmpl

    # ----------------------------
    # 3) Build FU classes
    # ----------------------------
    fu_classes = {tid: t.build_class() for tid, t in fu_templates_lookup.items()}

    # ----------------------------
    # 4) Build GU classes
    # ----------------------------
    gu_classes = {
        gid: t.build_class(fu_templates_lookup)
        for gid, t in gu_templates_lookup.items()
    }

    # ----------------------------
    # 5) Instantiate FU instances
    # ----------------------------
    fu_instances = {}

    for inst in config["FlowUnits"].get("flowunit_instances", []):
        iid = inst["id"]
        base = inst["instance_of"]
        FUClass = fu_classes[base]
        fu_obj = FUClass(iid, env, parent_scope="Model")

        # apply instance mods
        mods = inst.get("mods", {})
        if mods:
            for k in fu_obj.instance_mods.keys():
                if k in mods:
                    fu_obj.instance_mods[k] = mods[k]
            fu_obj.apply_instance_mods()

        fu_instances[iid] = fu_obj

    # ----------------------------
    # 6) Instantiate GU instances
    # ----------------------------
    gu_instances = {}

    for inst in config["GridUnits"].get("gridunit_instances", []):
        iid = inst["id"]
        base = inst["instance_of"]
        GUClass = gu_classes[base]
        gu_obj = GUClass(iid, env, parent_scope="Model")

        # apply GU instance mods
        mods = inst.get("mods", {})
        if mods:
            # FU mods inside GU
            for fu_local, fu_mods in mods.get("flowunits", {}).items():
                fu_inst = gu_obj.fus[fu_local]
                for k in fu_inst.instance_mods.keys():
                    if k in fu_mods:
                        fu_inst.instance_mods[k] = fu_mods[k]
                fu_inst.apply_instance_mods()

            # GU interconnections
            for k in ["override_interconnections", "add_interconnections", "delete_interconnections"]:
                if k in mods:
                    gu_obj.instance_mods[k] = mods[k]
            gu_obj.apply_instance_mods()

        gu_instances[iid] = gu_obj

    return {
        "env": env,
        "flowunits": fu_instances,
        "gridunits": gu_instances
    }
"""


# ============================================================
# Generate run_sim.py
# ============================================================
def _generate_run_sim_py():
    return """
import json
import simpy
from build_model import build_model

if __name__ == "__main__":
    env = simpy.Environment()

    with open("config/model.json") as f:
        config = json.load(f)

    system = build_model(env, config)
    print("System built:", system)

    # Start simulation
    env.run(until=30)
    print("Simulation completed.")
"""


# ============================================================
# Main code generator
# ============================================================
def generate_python_model(system, output_dir="generated_model", zip_output=True):
    """
    Creates a folder:
        generated_model/
            build_model.py
            run_sim.py
            config/model.json

    If zip_output=True → returns path to ZIP file.
    """
    # cleanup
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # directories
    _ensure_dir(output_dir)
    _ensure_dir(os.path.join(output_dir, "config"))

    # 1) Serialize system to JSON
    json_ir = _serialize_system_to_json(system)
    _write(
        os.path.join(output_dir, "config", "model.json"),
        json.dumps(json_ir, indent=2)
    )

    # 2) Write build_model.py
    _write(
        os.path.join(output_dir, "build_model.py"),
        _generate_build_model_py()
    )

    # 3) Write run_sim.py
    _write(
        os.path.join(output_dir, "run_sim.py"),
        _generate_run_sim_py()
    )

    # ZIP if needed
    if zip_output:
        zip_path = output_dir + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, output_dir)
                    z.write(full, arcname=os.path.join("generated_model", rel))
        return zip_path

    return output_dir

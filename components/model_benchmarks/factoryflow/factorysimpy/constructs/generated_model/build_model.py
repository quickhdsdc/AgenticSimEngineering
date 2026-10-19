
import json
import simpy
from builder_final import FlowUnitTemplate, GridUnitTemplate


def build_model(env, config):
    """
    Reconstruct FlowUnit and GridUnit templates + instances
    from the supplied JSON config.
    Returns a dict:
        {
           "env": env,
           "flowunits": {...},
           "gridunits": {...}
        }
    """

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

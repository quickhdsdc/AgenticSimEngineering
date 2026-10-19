"""
Extract an MBSModel spec from an existing USD robot asset (round-trip validation).

Why: the strongest correctness check for the builder is to take a robot NVIDIA
already ships (official Franka USD), read its inertial + joint parameters
programmatically, rebuild it through mbs_isaac, and verify both models produce
identical dynamics under identical torques. Hand-transcribing URDF numbers would
introduce exactly the errors we are trying to detect, so we read them from USD.

Run (from ):
  projectsisaac_sim.venvScriptspython.exe -m mbs_isaac.extract_usd
"""
from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


def extract(stage, root_path: str, name: str = "extracted"):
    """Walk a USD subtree and build an MBSModel from its physics schema."""
    from pxr import Usd, UsdPhysics

    from .spec import JointSpec, LinkSpec, MBSModel, Shape

    root = stage.GetPrimAtPath(root_path)
    assert root and root.IsValid(), f"no prim at {root_path}"

    links, joints = [], []
    body_paths = set()

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_paths.add(str(prim.GetPath()))
            m = UsdPhysics.MassAPI(prim)
            mass = m.GetMassAttr().Get() if m.GetMassAttr() else None
            inertia = m.GetDiagonalInertiaAttr().Get() if m.GetDiagonalInertiaAttr() else None
            com = m.GetCenterOfMassAttr().Get() if m.GetCenterOfMassAttr() else None
            links.append(LinkSpec(
                name=prim.GetName(),
                mass=float(mass) if mass else 1.0,
                com=tuple(com) if com else (0.0, 0.0, 0.0),
                inertia_diag=tuple(inertia) if inertia and all(i > 0 for i in inertia) else None,
                shape=Shape(kind="box", size=(0.05, 0.05, 0.05)),
            ))

    def _leaf(rel):
        tg = rel.GetTargets() if rel else []
        return os.path.basename(str(tg[0])) if tg else ""

    for prim in Usd.PrimRange(root):
        jt = None
        if prim.IsA(UsdPhysics.RevoluteJoint):
            jt = "revolute"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            jt = "prismatic"
        elif prim.IsA(UsdPhysics.FixedJoint):
            jt = "fixed"
        if jt is None:
            continue
        j = UsdPhysics.Joint(prim)
        axis = "Z"
        if jt != "fixed":
            a = (UsdPhysics.RevoluteJoint(prim) if jt == "revolute"
                 else UsdPhysics.PrismaticJoint(prim)).GetAxisAttr().Get()
            axis = str(a) if a else "Z"
        lo = hi = None
        if jt == "revolute":
            rj = UsdPhysics.RevoluteJoint(prim)
            lo = rj.GetLowerLimitAttr().Get()
            hi = rj.GetUpperLimitAttr().Get()
        p0 = j.GetLocalPos0Attr().Get()
        p1 = j.GetLocalPos1Attr().Get()
        joints.append(JointSpec(
            name=prim.GetName(),
            joint_type=jt,
            parent=_leaf(j.GetBody0Rel()),
            child=_leaf(j.GetBody1Rel()),
            axis=axis,
            parent_anchor=tuple(p0) if p0 else (0.0, 0.0, 0.0),
            child_anchor=tuple(p1) if p1 else (0.0, 0.0, 0.0),
            lower=float(lo) if lo is not None else None,
            upper=float(hi) if hi is not None else None,
        ))

    return MBSModel(name=name, links=links, joints=joints)


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.storage.native")
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    usd = root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    print(f"asset: {usd}")
    create_new_stage()
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=usd, prim_path="/World/franka")

    model = extract(stage, "/World/franka", name="franka_rebuilt")
    print(f"\nextracted: {len(model.links)} links, {len(model.joints)} joints, "
          f"DoF={model.dof}, total mass={model.total_mass:.3f} kg\n")
    for l in model.links:
        print(f"  link {l.name:22s} m={l.mass:7.3f}  com={tuple(round(c,4) for c in l.com)}  "
              f"I={tuple(round(i,5) for i in l.resolved_inertia())}")
    for j in model.joints:
        lim = f"[{j.lower:.2f},{j.upper:.2f}]" if j.lower is not None else "free"
        print(f"  joint {j.name:20s} {j.joint_type:9s} {j.parent:20s} -> {j.child:20s} "
              f"axis={j.axis} lim={lim}")
    try:
        model.validate()
        print("\nSPEC VALIDATES OK")
    except ValueError as e:
        print(f"\nvalidation issue: {e}")
    app.close()


if __name__ == "__main__":
    main()

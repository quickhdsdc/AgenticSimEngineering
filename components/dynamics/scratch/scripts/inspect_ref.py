"""Dump every authored physics attribute on the reference Franka's joints/links.

Kinematics already matches to 1e-7 m while the dynamics diverge, so some
dynamic property (armature, joint friction, drive, velocity clamp, solver
settings) exists on the reference and is missing from the rebuild. This finds it.
"""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.storage.native")
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
from isaacsim.storage.native import get_assets_root_path

create_new_stage()
import omni.usd
from pxr import Usd

stage = omni.usd.get_context().get_stage()
add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    prim_path="/World/franka_ref",
)

for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/franka_ref")):
    name = prim.GetName()
    if not (name.startswith("panda_joint") or name.startswith("panda_finger")
            or name in ("panda_link1", "panda_link4")):
        continue
    authored = [(a.GetName(), a.Get()) for a in prim.GetAttributes() if a.HasAuthoredValue()]
    print(f"\n=== {prim.GetPath()}  [{prim.GetTypeName()}]")
    print(f"    APIs: {prim.GetAppliedSchemas()}")
    for k, v in authored:
        if any(s in k for s in ("physx", "drive", "limit", "armature", "friction",
                                "velocity", "Limit", "stiffness", "damping")):
            print(f"    {k:45s} = {v}")

app.close()

# Acceptance test: spawn the official Franka asset (downloads from NVIDIA cloud
# asset root on first run), read articulation properties, hold pose for 1s.
# Validates: asset pipeline + articulation API + stable physics stepping.
# Run:  env_isaacsim\Scripts\python.exe franka_acceptance.py
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot.manipulators.examples")  # deprecated ext, off by default in 6.0

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka

world = World(physics_dt=0.002, rendering_dt=0.002, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
robot = world.scene.add(Franka(prim_path="/World/franka", name="franka"))
world.reset()

print(f"ACCEPT: num_dof = {robot.num_dof}")
print(f"ACCEPT: dof_names = {robot.dof_names}")
props = robot.dof_properties
print(f"ACCEPT: lower  = {np.round(props['lower'], 3)}")
print(f"ACCEPT: upper  = {np.round(props['upper'], 3)}")
print(f"ACCEPT: maxEffort = {np.round(props['maxEffort'], 1)}")

q0 = robot.get_joint_positions().copy()
for _ in range(500):  # 1 s at 500 Hz, default position controller holds pose
    world.step(render=False)

q1 = robot.get_joint_positions()
drift = float(np.abs(q1 - q0).max())
print(f"ACCEPT: max joint drift over 1s hold = {drift:.5f} rad")
assert np.all(np.isfinite(q1)), "non-finite joint state"
assert drift < 0.1, f"arm drifted too much while holding: {drift}"
print("ACCEPTANCE PASSED")

sim_app.close()

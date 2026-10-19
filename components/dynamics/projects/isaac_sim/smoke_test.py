# Isaac Sim headless smoke test: rigid-body cube free fall under PhysX.
# Run:  env_isaacsim\Scripts\python.exe smoke_test.py
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

try:  # Isaac Sim >= 4.5 namespace
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
except ImportError:  # pre-4.5 fallback
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
cube = world.scene.add(
    DynamicCuboid(prim_path="/World/cube", name="cube", position=[0.0, 0.0, 1.0], size=0.1)
)
world.reset()

for _ in range(120):
    world.step(render=False)

pos, _ = cube.get_world_pose()
print(f"SMOKE TEST RESULT: cube z after 120 steps = {float(pos[2]):.4f} (started at 1.0)")
assert float(pos[2]) < 0.5, "cube did not fall - physics not stepping"
print("SMOKE TEST PASSED")

sim_app.close()

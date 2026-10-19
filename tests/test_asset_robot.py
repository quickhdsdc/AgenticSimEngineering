import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from agenticsim.adapters.asset_robot import AssetRobot
from agenticsim.adapters.execution import LocalWorker
from agenticsim.domain import EngineeringTask
from agenticsim.runtime.registry import evaluator_for

ROOT = Path(__file__).resolve().parents[1]


def example(name):
    return json.loads((ROOT / f"examples/{name}-study.json").read_text(encoding="utf-8"))


class AssetRobotContracts(unittest.TestCase):
    def test_real_robot_identity_mobility_and_local_integrity(self):
        for name, count, floating in [("kuka-iiwa14", 7, False), ("unitree-g1", 29, True)]:
            task = example(name)
            adapter = evaluator_for(task)
            self.assertEqual(len(adapter.joints), count)
            self.assertEqual(adapter.floating, floating)
            adapter.validate(EngineeringTask(task))
            manifest, files = adapter.verify_assets()
            self.assertEqual(len(manifest["commit"]), 40)
            self.assertTrue(any(f["path"].endswith('/LICENSE') for f in files))
            self.assertTrue(any('/assets/' in f["path"] for f in files))

    def test_reject_bad_robot_requests(self):
        cases = [lambda d: d["inputs"]["initial_q"].update(values=[0]*7),
                 lambda d: d["candidates"][0]["parameters"]["joint_index"].update(value=29),
                 lambda d: d["candidates"][0]["parameters"]["target_offset"].update(unit="deg"),
                 lambda d: d["execution"].update(algorithm="physx"),
                 lambda d: d["execution"].update(device="auto"),
                 lambda d: d["execution"].update(dt_s=0),
                 lambda d: d.update(required_capabilities=["gradient"])]
        adapter = AssetRobot("unitree-g1")
        for change in cases:
            task = example("unitree-g1")
            change(task)
            with self.assertRaises(ValueError):
                adapter.validate(EngineeringTask(task))

    def test_core_registry_does_not_import_physics_engines(self):
        subprocess.run([sys.executable, "-c", "from agenticsim.adapters.asset_robot import AssetRobot; "
                        "AssetRobot('unitree-g1'); import sys; "
                        "assert not {'newton','warp','mujoco','numpy','isaacsim'} & set(sys.modules)"], cwd=ROOT, check=True)


@unittest.skipUnless(os.getenv("ASE_NEWTON_PYTHON"), "set ASE_NEWTON_PYTHON for real robot solvers")
class AssetRobotIntegration(unittest.TestCase):
    def test_cpu_robot_rollouts_contact_and_mapping(self):
        executor = LocalWorker(os.environ["ASE_NEWTON_PYTHON"])
        for name in ("kuka-iiwa14", "unitree-g1"):
            d = example(name)
            d["execution"]["device"] = "cpu"
            result = executor.evaluate(EngineeringTask(d), d["candidates"][0])
            result.validate(AssetRobot(name).model)
            self.assertTrue(all(v < 1e-5 for v in result.provenance["import_audit"].values()))
            self.assertLess(result.observations[0].values[0], 0.1)
            if name == "unitree-g1":
                self.assertEqual(result.provenance["generalized_velocity_dofs"], 35)
                self.assertGreater(result.observations[1].values[0], 0.5)
                self.assertGreater(result.observations[2].values[0], 0)
            else:
                self.assertEqual(result.provenance["generalized_velocity_dofs"], 7)

    def test_imported_dynamics_against_original_mjcf_one_step(self):
        # This is an importer check against the original model, not an
        # independent physical-validation oracle (both use MuJoCo equations).
        script = r'''
import json
from pathlib import Path
import mujoco
for folder,key in [('kuka_iiwa_14','home'),('unitree_g1','stand')]:
    m=mujoco.MjModel.from_xml_path(str(Path('assets/robots/menagerie')/folder/'scene.xml'))
    m.opt.timestep=0.002; m.opt.iterations=50; m.opt.ls_iterations=20; m.opt.tolerance=1e-8
    m.opt.solver=mujoco.mjtSolver.mjSOL_NEWTON; m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,key))
    mujoco.mj_step(m,d)
    print(json.dumps({'folder':folder,'q':d.qpos[m.jnt_qposadr[m.actuator_trnid[:,0]]].tolist()}))
'''
        output = subprocess.check_output([os.environ["ASE_NEWTON_PYTHON"], "-c", script], cwd=ROOT, text=True)
        references = [json.loads(line)["q"] for line in output.splitlines()]
        for name, reference in zip(("kuka-iiwa14", "unitree-g1"), references):
            d = example(name)
            d["execution"].update(device="cpu", duration_s=0.002)
            result = LocalWorker(os.environ["ASE_NEWTON_PYTHON"]).evaluate(EngineeringTask(d), d["candidates"][0])
            actual = [o.values[-1] for o in result.observations[4:]]
            self.assertEqual(len(actual), len(reference))
            self.assertLess(max(abs(a-b) for a,b in zip(actual, reference)), 1e-4)


if __name__ == "__main__":
    unittest.main()

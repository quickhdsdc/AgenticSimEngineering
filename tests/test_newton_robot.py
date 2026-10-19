import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from agenticsim.adapters.newton_robot import NewtonRobot
from agenticsim.adapters.execution import LocalWorker
from agenticsim.domain import EngineeringTask
from agenticsim.runtime.registry import evaluator_for

ROOT = Path(__file__).resolve().parents[1]


def example():
    return json.loads((ROOT / "examples/robot-newton-study.json").read_text(encoding="utf-8"))


class RobotContracts(unittest.TestCase):
    def test_registry_and_lazy_imports(self):
        self.assertIsInstance(evaluator_for(example()), NewtonRobot)
        with self.assertRaises(ValueError):
            evaluator_for({"model": {"id": "arbitrary.module"}})
        subprocess.run([sys.executable, "-c",
                        "import agenticsim.adapters.newton_robot, sys; assert not {'newton','warp','numpy','isaacsim'} & set(sys.modules)"], cwd=ROOT, check=True)

    def test_rejects_unmapped_or_unsupported_robot_requests(self):
        cases = [lambda d: d["execution"].update(algorithm="mujoco"),
                 lambda d: d["execution"].update(device="automatic"),
                 lambda d: d["execution"].update(dt_s=0),
                 lambda d: d["execution"].update(duration_s=0.5001),
                 lambda d: d["inputs"]["initial_q"].update(values=[0, 0]),
                 lambda d: d["inputs"]["initial_q"].update(unit="deg"),
                 lambda d: d["candidates"][0]["parameters"]["kp"].update(unit="1"),
                 lambda d: d.update(required_capabilities=["gradient"]),
                 lambda d: d["objective"].update(semantic_id="urn:ase:robot:joint-0-q")]
        for change in cases:
            task = example()
            change(task)
            with self.subTest(task=task), self.assertRaises(ValueError):
                NewtonRobot().validate(EngineeringTask(task))


@unittest.skipUnless(os.getenv("ASE_NEWTON_PYTHON"), "set ASE_NEWTON_PYTHON to isolated Newton interpreter")
class NewtonNumerics(unittest.TestCase):
    def run_case(self, candidate, device="cpu", dt=0.001):
        d = example()
        d["candidates"] = [d["candidates"][candidate]]
        d["execution"].update(device=device, dt_s=dt)
        result = LocalWorker(os.environ["ASE_NEWTON_PYTHON"]).evaluate(EngineeringTask(d), d["candidates"][0])
        result.validate(NewtonRobot.model)
        return result

    def test_zero_torque_equilibrium(self):
        result = self.run_case(0)
        self.assertEqual(result.observations[1].values, (0.,))
        for observation in result.observations[2:]:
            self.assertLess(max(abs(q) for q in observation.values), 1e-6)
            self.assertEqual(observation.axis[0], 0.)
            self.assertAlmostEqual(observation.axis[-1], 0.5)

    def test_control_step_refinement_and_cpu_gpu_agreement(self):
        cpu = self.run_case(2)
        fine = self.run_case(2, dt=0.0005)
        gpu = self.run_case(2, device="cuda:0")
        for result in (cpu, fine, gpu):
            self.assertLess(result.observations[0].values[0], 0.05)
            self.assertLessEqual(result.observations[1].values[0], 20.)
        for coarse_q, fine_q, gpu_q in zip(cpu.observations[2:], fine.observations[2:], gpu.observations[2:]):
            self.assertEqual(coarse_q.axis, gpu_q.axis)
            self.assertLess(max(abs(a - b) for a, b in zip(coarse_q.values, fine_q.values[::2])), 0.005)
            self.assertLess(max(abs(a - b) for a, b in zip(coarse_q.values, gpu_q.values)), 0.0001)


if __name__ == "__main__":
    unittest.main()

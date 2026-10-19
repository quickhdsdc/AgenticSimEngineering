import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from agenticsim.adapters.fem import StampingFEM, ROOT, validate_final_frame
from agenticsim.adapters.process_tree import run_worker
from agenticsim.domain import EngineeringTask


class FEMTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ROOT / "examples/fem-sample7-study.json").read_text())

    def test_task_and_lazy_import(self):
        StampingFEM().validate(EngineeringTask(self.document))
        result = subprocess.run([sys.executable, "-c",
            "import sys; from agenticsim.runtime.registry import evaluator_for; "
            "evaluator_for({'model':{'id':'fem-sample7'}}); "
            "assert not any(n in sys.modules for n in ('numpy','scipy','torch'))"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_units_ranges_and_algorithm_rejected(self):
        changes = [(('inputs', 'density', 'unit'), 'g/mm3'),
                   (('inputs', 'density', 'values'), [0]),
                   (('execution', 'dt_min_s'), .1),
                   (('execution', 'threads'), True),
                   (('execution', 'algorithm'), 'implicit'),
                   (('candidates', 0, 'parameters', 'friction', 'value'), .6),
                   (('candidates', 0, 'parameters', 'clearance_ratio', 'unit'), 'mm')]
        for keys, value in changes:
            with self.subTest(keys=keys):
                document = copy.deepcopy(self.document)
                target = document
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    StampingFEM().validate(EngineeringTask(document))

    def test_incomplete_solver_frame_is_rejected(self):
        self.assertEqual(validate_final_frame('TIME 1 1 double\n12.5\n', 12.5), 12.5)
        for text in ('TIME 1 1 double\n11.0\n', 'no time', 'TIME 1 1 double\nnan\n'):
            with self.assertRaises(ValueError):
                validate_final_frame(text, 12.5)

    def test_timeout_terminates_solver_descendant(self):
        # Child would write after timeout if only the worker were killed.
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'orphan.txt'
            child = "import time; from pathlib import Path; time.sleep(2); Path(" + repr(str(marker)) + ").write_text('orphan')"
            worker = "import sys,subprocess,time; sys.stdin.read(); subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); print('spawned',flush=True); time.sleep(30)"
            with self.assertRaises(subprocess.TimeoutExpired) as captured:
                run_worker([sys.executable, '-c', worker], payload='go', cwd=ROOT,
                           env=os.environ.copy(), timeout=1)
            self.assertIn('spawned', captured.exception.output)
            time.sleep(2)
            self.assertFalse(marker.exists(), 'solver descendant survived worker timeout')


if __name__ == '__main__':
    unittest.main()

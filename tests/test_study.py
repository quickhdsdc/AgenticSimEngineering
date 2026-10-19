import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agenticsim.adapters.des import SingleServer, SOURCE
from agenticsim.adapters.execution import FileStudyStore, LocalWorker
from agenticsim.application import compare
from agenticsim.domain import EngineeringTask, Observable

ROOT = Path(__file__).resolve().parents[1]
HAS_DES = all(importlib.util.find_spec(name) for name in ("numpy", "simpy"))


def example():
    return json.loads((ROOT / "examples/queue-study.json").read_text(encoding="utf-8"))


class StudyContracts(unittest.TestCase):
    def test_bad_task_rejected_before_execution_or_storage(self):
        modifications = [
            lambda d: d.update(schema_version="99"),
            lambda d: d.update(required_capabilities=["gradient"]),
            lambda d: d["inputs"]["service"].update(unit="ms"),
            lambda d: d["inputs"]["service"].update(semantic_id="unknown"),
            lambda d: d["inputs"]["service"].update(values=[1]),
            lambda d: d["inputs"]["service"].update(values=[float("nan")] * 4),
            lambda d: d["candidates"][-1]["parameters"]["service_scale"].update(value=-1),
            lambda d: d["budget"].update(max_evaluations=2),
            lambda d: d["objective"].update(semantic_id="urn:ase:queue:departure"),
            lambda d: d.update(unknown_field=True),
        ]
        class NeverExecute:
            def evaluate(self, *args):
                raise AssertionError("must validate first")
        for change in modifications:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                task = example()
                change(task)
                with self.assertRaises(ValueError):
                    compare(task, SingleServer(), NeverExecute(), FileStudyStore(directory))
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_scalar_and_non_temporal_series(self):
        Observable("cost", "EUR", (2.,)).validate()
        Observable("departures", "s", (2., 4.), (0., 1.), "job_index").validate()
        for bad in [Observable("x", "s", (1., 2.)), Observable("x", "s", (1.,), (0., 1.), "s"),
                    Observable("x", "", (1.,)), Observable("x", "s", (1., 2.), (0., 0.), "s")]:
            with self.assertRaises(ValueError):
                bad.validate()

    def test_core_startup_is_lightweight(self):
        script = "import agenticsim.domain, agenticsim.application; import sys; assert not {'numpy','torch','simpy','a4s','openai'} & set(sys.modules)"
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)

    def test_failure_is_not_a_feasible_decision(self):
        class Broken:
            def evaluate(self, *args):
                raise RuntimeError("solver unavailable")
        with tempfile.TemporaryDirectory() as directory:
            study = compare(example(), SingleServer(), Broken(), FileStudyStore(directory))
            self.assertEqual(study["status"], "completed_with_errors")
            self.assertEqual(study["decision"]["status"], "inconclusive")
            self.assertTrue(all(r["status"] == "failed" for r in study["runs"]))

    def test_real_worker_timeout_is_recorded(self):
        task = example()
        task["candidates"] = task["candidates"][:1]
        task["budget"]["timeout_s"] = 0.000001
        with tempfile.TemporaryDirectory() as directory:
            study = compare(task, SingleServer(), LocalWorker(), FileStudyStore(directory))
            self.assertEqual(study["runs"][0]["status"], "timed_out")
            self.assertEqual(study["decision"]["status"], "inconclusive")


@unittest.skipUnless(HAS_DES, "install des-reference extra for real SimPy integration")
class DESIntegration(unittest.TestCase):
    def test_original_interface_and_independent_fifo_calculation(self):
        adapter = SingleServer()
        task = EngineeringTask(example())
        result = adapter.evaluate(task, task.document["candidates"][0])
        spec = importlib.util.spec_from_file_location("original_reference", SOURCE)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        expected = original.simpy_single_server([0, 1, 1, 1], [2, 2, 2, 2])
        self.assertEqual(result.observations[1].values, tuple(expected))
        self.assertEqual(result.observations[1].values, (2., 4., 6., 8.))
        self.assertEqual(result.source, "reference_simulation")
        self.assertEqual(result.provenance["input_source"], "synthetic_fixture")

    def test_isolated_study_evidence_and_constraint_rejection(self):
        task = example()
        before = copy.deepcopy(task)
        with tempfile.TemporaryDirectory() as directory:
            study = compare(task, SingleServer(), LocalWorker(), FileStudyStore(directory))
            self.assertEqual(task, before)
            self.assertEqual(study["decision"]["best_observed_candidate"], "shorter-service")
            self.assertEqual([r["assessment"]["objective"] for r in study["runs"]], [8., 4., 12.])
            self.assertEqual([r["assessment"]["status"] for r in study["runs"]], ["accepted", "accepted", "rejected"])
            self.assertTrue(all(r["status"] == "succeeded" for r in study["runs"]))
            path = Path(directory) / study["study_id"] / "study.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            canonical = json.dumps(record["payload"], sort_keys=True, allow_nan=False, ensure_ascii=False).encode("utf-8")
            self.assertEqual(record["sha256"], hashlib.sha256(canonical).hexdigest())
            self.assertEqual(record["payload"], study)
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_no_accepted_candidate_is_inconclusive(self):
        task = example()
        task["constraints"][0]["limit"] = 0
        with tempfile.TemporaryDirectory() as directory:
            study = compare(task, SingleServer(), SingleServer(), FileStudyStore(directory))
            self.assertEqual(study["status"], "completed")
            self.assertIsNone(study["decision"]["best_observed_candidate"])
            self.assertEqual(study["decision"]["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()

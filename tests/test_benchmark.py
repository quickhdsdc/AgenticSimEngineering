"""Leakage, tie handling, source mapping and immutable benchmark evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from agenticsim.adapters.screw_dataset import FEATURES, features
from agenticsim.domain.benchmark import classification_metrics, grouped_split
from agenticsim.runtime.benchmark import execute, validate_spec, verify
from agenticsim.runtime.paths import ROOT, RUNS


class BenchmarkTests(unittest.TestCase):
    def test_workpieces_never_cross_partitions_and_split_is_order_independent(self):
        records = [{"workpiece_id": str(i), "operation_id": f"{i}-{j}"} for i in range(20) for j in range(5)]
        split = grouped_split(records, 42)
        groups = {p: {r["workpiece_id"] for r in rows} for p, rows in split.items()}
        self.assertEqual([len(groups[p]) for p in ("train", "validation", "test")], [12, 4, 4])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertFalse(groups["train"] & groups["validation"])
        reverse = grouped_split(list(reversed(records)), 42)
        self.assertEqual(groups, {p: {r["workpiece_id"] for r in rows} for p, rows in reverse.items()})

    def test_station_decisions_do_not_enter_features(self):
        graph = {"time values": [0., .1, .1, .2], "angle values": [0., 1., 2., 3.],
                 "torque values": [0., 1.4, 1., .05]}
        document = {"torque unit": "Nm", "result": "OK", "quality code": 1,
                    "tightening steps": [{"result": "OK", "graph": graph}]}
        vector, flags = features(document)
        changed = copy.deepcopy(document)
        changed["result"], changed["quality code"] = "NOK", 999
        changed["tightening steps"][0]["result"] = "NOK"
        self.assertEqual(features(changed), (vector, flags))
        self.assertEqual(set(vector), set(FEATURES))
        self.assertEqual(vector["peak_torque_nm"], 1.4)
        self.assertEqual(vector["end_torque_nm"], .05)
        self.assertEqual(flags["duplicate_times"], 1)
        changed["torque unit"] = "Ncm"
        with self.assertRaises(ValueError):
            features(changed)
        changed["torque unit"] = "Nm"
        changed["tightening steps"][0]["graph"]["time values"][-1] = -.1
        with self.assertRaises(ValueError):
            features(changed)
        changed["tightening steps"][0]["graph"]["time values"][-1] = .2
        changed["tightening steps"][0]["graph"]["torque values"][0] = float("nan")
        with self.assertRaises(ValueError):
            features(changed)

    def test_average_precision_groups_ties_and_known_confusion(self):
        result = classification_metrics([1, 0, 1, 0], [1., 1., 1., 1.], 2.)
        self.assertEqual(result["average_precision"], .5)
        self.assertEqual(result["fn"], 2)
        result = classification_metrics([1, 0, 1, 0], [4., 1., 3., 2.], 2.5)
        self.assertEqual(result["average_precision"], 1.)
        self.assertEqual(result["f1"], 1.)

    def test_spec_rejects_wrong_source_and_nonfinite_floors(self):
        spec = json.loads((ROOT / "examples/screw-benchmark.json").read_text())
        validate_spec(spec)
        spec["source"] = "synthetic"
        with self.assertRaises(ValueError):
            validate_spec(spec)
        spec["source"] = "measurement"
        spec["scale_floors"]["duration_s"] = float("inf")
        with self.assertRaises(ValueError):
            validate_spec(spec)

    @unittest.skipUnless((ROOT / "work/cache/datasets/screw-s01-v1.1.3/s01_thread-degradation.zip").is_file(),
                         "download pinned real screw-driving dataset")
    def test_real_benchmark_freezes_fit_and_rejects_test_data_tampering(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as temp:
            case = Path(temp) / "case"
            result = execute("prepare", case)
            self.assertEqual(result["partitions"]["test"]["operations"], 1000)
            execute("fit", case)
            report = execute("score", case)
            self.assertEqual(report["test_metrics"]["positives"], 170)
            self.assertIn("reference_peak_torque_window", report["baselines"])
            self.assertEqual(report["physical_validation"], "not_established")
            verify(case, "score")
            with self.assertRaises(ValueError):
                execute("score", case)
            path = case / ".evaluation/test.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                verify(case, "prepare")


if __name__ == "__main__":
    unittest.main()

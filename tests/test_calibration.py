"""Calibration provenance, leakage boundaries and a real inverse-dynamics fit."""
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

from agenticsim.domain.calibration import JOINTS, UNITS, read_samples, validate_dataset, validate_spec
from agenticsim.runtime.calibration import artifact_path, execute
from agenticsim.runtime.paths import ROOT, RUNS


def manifest():
    return {"schema_version": "1.0", "source": "synthetic_fixture", "model_id": "kuka-iiwa14",
            "joint_order": list(JOINTS), "units": dict(UNITS), "frame": "source-MJCF-joint-coordinates",
            "torque_semantics": "required_generalized_joint_torque_no_external_load", "physical_validation": "not_established",
            "experiment": {"sample_dt_s": .01},
            "runs": [{"run_id": str(i), "role": "holdout" if i == 2 else "train", "sha256": str(i) * 64,
                      "file": str(i) + ".csv"} for i in range(3)]}


class CalibrationContracts(unittest.TestCase):
    def test_spec_rejects_measurement_relabelling_and_physical_acceptance(self):
        spec = json.loads((ROOT / "examples/kuka-calibration.json").read_text())
        validate_spec(spec)
        for change in (lambda s: s["context"].update(source="measurement"),
                       lambda s: s["acceptance"].update(scope="physical_validation"),
                       lambda s: s["parameters"]["armature"].update(unit="kg"),
                       lambda s: s["parameters"]["damping"].update(initial=-1)):
            bad = copy.deepcopy(spec)
            change(bad)
            with self.assertRaises(ValueError):
                validate_spec(bad)

    def test_train_holdout_runs_files_and_bytes_must_be_disjoint(self):
        valid = manifest()
        validate_dataset(valid)
        for key in ("run_id", "file", "sha256"):
            bad = copy.deepcopy(valid)
            bad["runs"][2][key] = bad["runs"][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dataset(bad)

    def test_wrong_units_joint_order_and_torque_meaning_rejected(self):
        valid = manifest()
        for change in (lambda s: s["units"].update(time_s="ms"), lambda s: s["joint_order"].reverse(),
                       lambda s: s.update(torque_semantics="commanded_motor_current")):
            bad = copy.deepcopy(valid)
            change(bad)
            with self.assertRaises(ValueError):
                validate_dataset(bad)

    def test_missing_nan_samples_and_nonmonotonic_clocks_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.csv"
            rows = [[i * .01, *([0.] * 28)] for i in range(100)]
            def write(values):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(UNITS)
                    writer.writerows(values)
            write(rows)
            read_samples(path, manifest())
            for column, value in ((0, 0.), (1, float("nan")), (2, "")):
                bad = copy.deepcopy(rows)
                bad[2][column] = value
                write(bad)
                with self.subTest(column=column), self.assertRaises(ValueError):
                    read_samples(path, manifest())

    def test_artifact_paths_cannot_escape_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                artifact_path(Path(temporary), "../private.json")
        with self.assertRaises(ValueError):
            execute("generate", ROOT / "examples/do-not-write")


@unittest.skipUnless(os.getenv("ASE_NEWTON_PYTHON"), "set ASE_NEWTON_PYTHON for real calibration checks")
class KukaCalibrationIntegration(unittest.TestCase):
    def test_generated_model_fit_holdout_and_frozen_evidence(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as temporary:
            directory = Path(temporary) / "case"
            worker = os.environ["ASE_NEWTON_PYTHON"]
            for phase in ("generate", "model", "fit"):
                execute(phase, directory, worker_python=worker)
            fitted = json.loads((directory / "fit.json").read_text())
            self.assertFalse(fitted["holdout_used_for_fitting"])
            self.assertFalse(fitted["fixture_truth_used_for_fitting"])
            self.assertAlmostEqual(fitted["parameters"]["damping"], .8, delta=.01)
            self.assertAlmostEqual(fitted["parameters"]["armature"], .12, delta=.002)
            # The held-out file was never opened for fitting. Changing it must
            # fail validation rather than fitting again or silently accepting it.
            holdout = directory / "excitation-c.csv"
            original = holdout.read_bytes()
            holdout.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                execute("validate", directory, worker_python=worker)
            self.assertIn("checksum mismatch", (directory / "logs/validate-stderr.log").read_text())
            holdout.write_bytes(original)
            result = execute("validate", directory, worker_python=worker)
            self.assertEqual(result["status"], "accepted")
            self.assertLess(result["identified_rmse_nm"], .01)
            self.assertGreater(result["baseline_rmse_nm"], .2)
            self.assertEqual(result["physical_validation"], "not_established")
            with self.assertRaisesRegex(ValueError, "already recorded"):
                execute("validate", directory, worker_python=worker)

"""Public measurement mapping, experiment leakage and frozen-fit validation."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from agenticsim.domain.calibration import JOINTS, UNITS
from agenticsim.domain.measurements import TRAIN, HOLDOUT, read_samples, validate_dataset, validate_spec
from agenticsim.runtime.identification import execute
from agenticsim.runtime.paths import ROOT, RUNS


def manifest():
    return {"schema_version": "1.0", "model_id": "franka-panda", "source": "measurement",
            "mapping_status": "provisional", "physical_validation": "not_established",
            "torque_semantics": "publisher_tau_interp_inverse_dynamics_target",
            "frame": "provisional-source-joint-coordinates", "joint_order": list(JOINTS), "units": dict(UNITS),
            "runs": [{"seed": s, "run_id": f"panda-seed-{s}", "file": f"panda-seed-{s}.csv",
                      "role": "train" if s in TRAIN else "holdout", "sha256": f"{s:064x}", "source_sha256": f"{s+100:064x}"}
                     for s in TRAIN + HOLDOUT]}


class MeasurementContracts(unittest.TestCase):
    def test_provisional_mapping_cannot_be_relabelled_as_physical_validation(self):
        valid = json.loads((ROOT / "examples/panda-identification.json").read_text())
        validate_spec(valid)
        for change in (lambda s: s.update(mapping_status="verified"),
                       lambda s: s.update(source="synthetic_fixture"),
                       lambda s: s["units"].update(position="deg"),
                       lambda s: s["mapping"].update(torque="tau"),
                       lambda s: s["acceptance"].update(scope="physical_validation")):
            bad = copy.deepcopy(valid)
            change(bad)
            with self.assertRaises(ValueError):
                validate_spec(bad)

    def test_experiment_split_and_variant_fingerprints_cannot_overlap(self):
        data = manifest()
        validate_dataset(data)
        for key in ("seed", "run_id", "file", "sha256", "source_sha256"):
            bad = copy.deepcopy(data)
            bad["runs"][-1][key] = bad["runs"][0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dataset(bad)

    def test_wrong_time_joint_units_and_torque_semantics_rejected(self):
        data = manifest()
        for change in (lambda d: d["units"].update(time_s="ms"),
                       lambda d: d["joint_order"].reverse(),
                       lambda d: d.update(torque_semantics="unmapped_motor_current"),
                       lambda d: d.update(physical_validation="established")):
            bad = copy.deepcopy(data)
            change(bad)
            with self.assertRaises(ValueError):
                validate_dataset(bad)

    def test_irregular_recorded_clock_accepted_nonfinite_or_reversed_samples_rejected(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as temporary:
            path = Path(temporary) / "samples.csv"
            def write(change=None):
                rows = [[.07 + i * .017 + (i % 2) * .001] + [0.] * 28 for i in range(20)]
                if change:
                    change(rows)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(UNITS)
                    writer.writerows(rows)
            write()
            self.assertEqual(len(read_samples(path, manifest())["time_s"]), 20)
            for change in (lambda r: r[2].__setitem__(0, r[1][0]),
                           lambda r: r[1].__setitem__(4, float("nan")),
                           lambda r: r[1].__setitem__(2, "")):
                write(change)
                with self.assertRaises(ValueError):
                    read_samples(path, manifest())


@unittest.skipUnless(os.getenv("ASE_IDENTIFICATION_PYTHON"), "set ASE_IDENTIFICATION_PYTHON for public measurement checks")
class PandaIdentificationIntegration(unittest.TestCase):
    def test_real_import_fit_blind_holdout_and_evidence_tamper(self):
        worker = os.environ["ASE_IDENTIFICATION_PYTHON"]
        RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as temporary:
            directory = Path(temporary) / "case"
            for phase in ("ingest", "model"):
                execute(phase, directory, worker_python=worker)
            data = json.loads((directory / "dataset.json").read_text())
            self.assertEqual(len(data["runs"]), 29)
            self.assertTrue(any(not r["clock"]["uniform"] for r in data["runs"]))
            # Fit must succeed while a holdout payload is unavailable.
            holdout = directory / "panda-seed-21.csv"
            parked = directory / "withheld.csv"
            holdout.rename(parked)
            execute("fit", directory, worker_python=worker)
            parked.rename(holdout)
            fit_path = directory / "fit.json"
            original_fit = fit_path.read_bytes()
            fitted = json.loads(original_fit)
            self.assertFalse(fitted["holdout_used_for_fitting"])
            self.assertEqual({r["run_id"] for r in fitted["training_runs"]}, {f"panda-seed-{s}" for s in TRAIN})
            fitted["parameters"]["joint1"]["offset"] += 1
            fit_path.write_text(json.dumps(fitted))
            with self.assertRaises(ValueError):
                execute("validate", directory, worker_python=worker)
            self.assertIn("phase evidence changed", (directory / "logs/validate-stderr.log").read_text())
            fit_path.write_bytes(original_fit)
            original = holdout.read_bytes()
            holdout.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                execute("validate", directory, worker_python=worker)
            self.assertIn("checksum mismatch", (directory / "logs/validate-stderr.log").read_text())
            holdout.write_bytes(original)
            result = execute("validate", directory, worker_python=worker)
            self.assertEqual(result["status"], "accepted")
            self.assertLess(result["identified_rmse_nm"], .3)
            self.assertGreater(result["baseline_rmse_nm"], .35)
            self.assertEqual(result["physical_validation"], "not_established")
            with self.assertRaisesRegex(ValueError, "already recorded"):
                execute("validate", directory, worker_python=worker)
            # Published data decoding never enables arbitrary pickle globals.
            code = (
                "import io,pickle; from agenticsim.adapters.merl_panda import ExperimentUnpickler; "
                "ExperimentUnpickler(io.BytesIO(pickle.dumps(eval))).load()")
            blocked = subprocess.run([worker, "-c", code], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("outside the pinned experiment allowlist", blocked.stderr)

    def test_predeclared_strict_criterion_produces_rejected_report(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNS) as temporary:
            root = Path(temporary)
            spec = json.loads((ROOT / "examples/panda-identification.json").read_text())
            spec["acceptance"]["pooled_rmse_limit_nm"] = .01
            spec_path = root / "strict.json"
            spec_path.write_text(json.dumps(spec))
            for phase in ("ingest", "model", "fit", "validate"):
                result = execute(phase, root / "case", spec_path=spec_path, worker_python=os.environ["ASE_IDENTIFICATION_PYTHON"])
            self.assertEqual(result["status"], "rejected")
            self.assertFalse(result["checks"]["absolute_rmse"])
            self.assertEqual(result["physical_validation"], "not_established")

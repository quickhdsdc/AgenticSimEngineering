"""Dependency-free contracts for the limited synthetic KUKA calibration case."""
import csv
from pathlib import Path

from . import Observable, number, require

JOINTS = tuple(f"joint{i}" for i in range(1, 8))
CHANNELS = (("q_rad", "rad"), ("qd_rad_s", "rad/s"), ("qdd_rad_s2", "rad/s^2"), ("tau_nm", "N*m"))
UNITS = {"time_s": "s", **{f"{joint}_{channel}": unit for joint in JOINTS for channel, unit in CHANNELS}}


def validate_spec(spec):
    require(isinstance(spec, dict) and spec.get("schema_version") == "1.0", "calibration schema 1.0 required")
    require(spec.get("model_id") == "kuka-iiwa14" and spec.get("joint") == "joint2", "only registered joint2 identification supported")
    require(isinstance(spec.get("case_id"), str) and bool(spec["case_id"]), "case identity required")
    context = spec.get("context", {})
    require(context.get("source") == "synthetic_fixture" and context.get("operator_source") == "delegated_operator",
            "this workflow accepts explicitly delegated synthetic fixtures only")
    require(bool(context.get("description")), "source description required")
    parameters = spec.get("parameters", {})
    require(set(parameters) == {"damping", "armature"}, "two explicit identification parameters required")
    for name, unit, maximum in (("damping", "N*m*s/rad", 10.), ("armature", "kg*m^2", 2.)):
        p = parameters[name]
        require(p.get("semantic_id") == f"urn:ase:kuka-iiwa14:joint2:{'viscous-damping' if name == 'damping' else name}"
                and p.get("unit") == unit, "parameter semantic/unit mismatch")
        bounds = p.get("bounds")
        require(isinstance(bounds, list) and len(bounds) == 2 and all(number(v) for v in bounds)
                and 0 <= bounds[0] < bounds[1] <= maximum and number(p.get("initial"))
                and bounds[0] <= p["initial"] <= bounds[1], "invalid initial parameter or bounds")
    experiment = spec.get("experiment", {})
    require(number(experiment.get("sample_dt_s")) and .005 <= experiment["sample_dt_s"] <= .02, "invalid sample period")
    require(type(experiment.get("samples_per_run")) is int and 100 <= experiment["samples_per_run"] <= 2000, "invalid sample count")
    require(number(experiment.get("torque_noise_std_nm")) and 0 <= experiment["torque_noise_std_nm"] <= .05, "invalid noise assumption")
    require(type(experiment.get("seed")) is int and 0 <= experiment["seed"] < 2**32 - 3, "explicit random seed required")
    acceptance = spec.get("acceptance", {})
    require(acceptance.get("scope") == "synthetic_holdout_only", "physical validation cannot be inferred from a fixture")
    require(number(acceptance.get("torque_rmse_limit_nm")) and 0 < acceptance["torque_rmse_limit_nm"] <= .1, "explicit synthetic RMSE criterion required")
    require(number(acceptance.get("minimum_relative_improvement")) and 0 < acceptance["minimum_relative_improvement"] < 1, "invalid relative improvement")
    require(isinstance(spec.get("assumptions"), list) and bool(spec["assumptions"]), "model assumptions required")


def validate_dataset(manifest):
    require(manifest.get("schema_version") == "1.0" and manifest.get("source") == "synthetic_fixture", "synthetic dataset schema required")
    require(manifest.get("model_id") == "kuka-iiwa14" and manifest.get("joint_order") == list(JOINTS), "dataset identity/joint order mismatch")
    require(manifest.get("units") == UNITS and manifest.get("frame") == "source-MJCF-joint-coordinates", "dataset units/frame mismatch")
    require(manifest.get("torque_semantics") == "required_generalized_joint_torque_no_external_load", "torque semantics must be explicit")
    require(manifest.get("physical_validation") == "not_established", "fixture cannot assert physical validation")
    runs = manifest.get("runs", [])
    require(isinstance(runs, list) and len(runs) == 3 and all(isinstance(r, dict) for r in runs), "two training runs and one holdout required")
    require([r.get("role") for r in runs].count("train") == 2 and [r.get("role") for r in runs].count("holdout") == 1, "invalid run split")
    require(len({r.get("run_id") for r in runs}) == 3 and all(isinstance(r.get("run_id"), str) and r["run_id"] for r in runs), "run split must be disjoint")
    require(len({r.get("sha256") for r in runs}) == 3 and all(isinstance(r.get("sha256"), str) and len(r["sha256"]) == 64 for r in runs), "duplicate or missing data fingerprints")
    require(len({r.get("file") for r in runs}) == 3, "run files must be distinct")


def read_samples(path, manifest):
    """Explicit columns/units and Observable axes; no automatic unit inference."""
    validate_dataset(manifest)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(UNITS), "CSV column mapping mismatch")
        rows = list(reader)
    require(100 <= len(rows) <= 2000, "unexpected run length")
    try:
        columns = {name: tuple(float(row[name]) for row in rows) for name in UNITS}
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV contains non-numeric or missing samples") from exc
    times = columns["time_s"]
    for name, unit in UNITS.items():
        if name != "time_s":
            Observable("urn:ase:calibration:" + name, unit, columns[name], times, "s").validate()
    require(abs(times[0]) < 1e-12, "each experiment must start at zero")
    dt = manifest["experiment"]["sample_dt_s"]
    require(all(abs(t - i * dt) < 1e-8 for i, t in enumerate(times)), "sample clock/gap mismatch")
    require(all(abs(v) <= 3.2 for joint in JOINTS for v in columns[joint + "_q_rad"]), "radian position range mismatch")
    return columns

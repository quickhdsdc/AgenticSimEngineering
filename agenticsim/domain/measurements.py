"""Standard-library contracts for one explicitly mapped public Panda dataset."""
import csv
from pathlib import Path

from . import Observable, number, require
from .calibration import JOINTS, UNITS

TRAIN = (2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 18)
HOLDOUT = (21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38)
PARAMETER_UNITS = {"damping": "N*m*s/rad", "armature": "kg*m^2", "coulomb": "N*m", "offset": "N*m"}


def validate_spec(spec):
    require(spec.get("schema_version") == "1.0" and spec.get("model_id") == "franka-panda", "registered Panda spec required")
    require(spec.get("source") == "measurement" and spec.get("operator_source") == "delegated_operator", "explicit measurement and delegated context required")
    require(spec.get("mapping_status") == "provisional", "current mapping cannot assert device calibration")
    require(spec.get("joint_order") == list(JOINTS), "explicit seven-joint order required")
    require(spec.get("mapping") == {"position": "q", "velocity": "dq", "acceleration": "ddq", "torque": "tau_interp", "time": "t"}, "unsupported source column mapping")
    require(spec.get("units") == {"position": "rad", "velocity": "rad/s", "acceleration": "rad/s^2", "torque": "N*m", "time": "s"}, "explicit SI mapping required")
    require(type(spec.get("sample_stride")) is int and 1 <= spec["sample_stride"] <= 10, "invalid deterministic stride")
    require(spec.get("case_id") and spec.get("sources") and spec.get("assumptions"), "context and mapping provenance required")
    bounds = spec.get("bounds", {})
    require(set(bounds) == set(PARAMETER_UNITS), "four named correction parameters required")
    for name, maximum in (("damping", 20), ("armature", 2), ("coulomb", 10), ("offset", 5)):
        values = bounds[name]
        require(isinstance(values, list) and len(values) == 2 and all(number(v) for v in values)
                and (-5 if name == "offset" else 0) <= values[0] < values[1] <= maximum, "invalid correction bounds")
    require(number(spec.get("smooth_velocity_rad_s")) and .005 <= spec["smooth_velocity_rad_s"] <= .1, "invalid Coulomb smoothing")
    criteria = spec.get("acceptance", {})
    require(criteria.get("scope") == "public_dataset_holdout_only", "physical acceptance is unavailable")
    require(number(criteria.get("pooled_rmse_limit_nm")) and 0 < criteria["pooled_rmse_limit_nm"] <= 5, "explicit dataset RMSE limit required")
    require(number(criteria.get("minimum_relative_improvement")) and 0 < criteria["minimum_relative_improvement"] < 1, "explicit improvement required")
    require(number(criteria.get("maximum_joint_rmse_regression_nm")) and 0 <= criteria["maximum_joint_rmse_regression_nm"] <= .5, "explicit joint regression criterion required")


def validate_dataset(data):
    require(data.get("schema_version") == "1.0" and data.get("model_id") == "franka-panda" and data.get("source") == "measurement", "Panda measurement dataset required")
    require(data.get("joint_order") == list(JOINTS) and data.get("units") == UNITS, "measurement joint/unit mapping mismatch")
    require(data.get("mapping_status") == "provisional" and data.get("physical_validation") == "not_established", "provisional mapping cannot assert physical acceptance")
    require(data.get("torque_semantics") == "publisher_tau_interp_inverse_dynamics_target", "explicit publisher target required")
    require(data.get("frame") == "provisional-source-joint-coordinates", "coordinate mapping must be explicit")
    runs = data.get("runs", [])
    require(len(runs) == len(TRAIN) + len(HOLDOUT), "complete registered train/holdout split required")
    require({r["seed"] for r in runs if r["role"] == "train"} == set(TRAIN)
            and {r["seed"] for r in runs if r["role"] == "holdout"} == set(HOLDOUT), "publisher excitation split required")
    for key in ("run_id", "file", "sha256", "source_sha256"):
        require(len({r[key] for r in runs}) == len(runs), "train/holdout files and content must be disjoint")
    for run in runs:
        require(run["file"] == f"panda-seed-{run['seed']}.csv" and run["run_id"] == f"panda-seed-{run['seed']}", "run identity mismatch")
        require(all(isinstance(run[k], str) and len(run[k]) == 64 for k in ("sha256", "source_sha256")), "data fingerprints required")


def read_samples(path, data):
    validate_dataset(data)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(UNITS), "canonical CSV mapping mismatch")
        rows = list(reader)
    require(10 < len(rows) < 5000, "unexpected canonical experiment size")
    try:
        columns = {name: tuple(float(row[name]) for row in rows) for name in UNITS}
    except (TypeError, ValueError) as exc:
        raise ValueError("missing or nonnumeric measurement") from exc
    for name, unit in UNITS.items():
        if name != "time_s":
            Observable("urn:ase:panda:measurement:" + name, unit, columns[name], columns["time_s"], "s").validate()
    require(columns["time_s"][0] >= 0, "negative acquisition time")
    return columns

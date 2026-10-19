"""Pinned KUKA synthetic inverse-dynamics identification, isolated CPU worker.

Only joint2 viscous damping and armature are identified. Prescribed kinematics
and MuJoCo-generated torques are fixtures, never physical measurements.
"""
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from agenticsim.adapters.asset_robot import AssetRobot
from agenticsim.domain import require
from agenticsim.domain.calibration import JOINTS, UNITS, read_samples, validate_dataset, validate_spec
from agenticsim.runtime.calibration import artifact_path, directory_path, load_json, save_json, sha256


def sdk():
    import numpy as np
    import mujoco
    require(mujoco.__version__ == "3.8.1", "pinned MuJoCo 3.8.1 required")
    return np, mujoco


def assets():
    adapter = AssetRobot("kuka-iiwa14")
    manifest, records = adapter.verify_assets()
    require(adapter.joints == JOINTS, "source joint order mismatch")
    return adapter, {"commit": manifest["commit"], "files": records, "template_sha256": sha256(adapter.asset)}


def make_model(directory, name, parameters):
    adapter, source = assets()
    xml = ET.parse(adapter.asset)
    xml.getroot().find("compiler").set("meshdir", str(adapter.asset.parent / "assets"))
    joint = xml.getroot().find(".//joint[@name='joint2']")
    for key, value in parameters.items():
        joint.set(key, str(value))
    option = xml.getroot().find("option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("contact", "disable")
    path = artifact_path(directory, name)
    xml.write(path, encoding="utf-8", xml_declaration=True)
    _, mj = sdk()
    model = mj.MjModel.from_xml_path(str(path))
    require(model.nq == model.nv == 7 and model.nu == 7, "generated model identity/dimension mismatch")
    require(tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)) == JOINTS, "generated joint mapping mismatch")
    j = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "joint2")
    dof = int(model.jnt_dofadr[j])
    require(abs(model.dof_damping[dof] - parameters["damping"]) < 1e-12
            and abs(model.dof_armature[dof] - parameters["armature"]) < 1e-12, "generated parameters were not imported")
    return model, source


def inverse(model, q, qd, qdd):
    np, mj = sdk()
    require(q.shape == qd.shape == qdd.shape and q.shape[1] == 7, "kinematic array shape mismatch")
    require(np.all(q >= model.jnt_range[:, 0]) and np.all(q <= model.jnt_range[:, 1]), "excitation exceeds source joint limits")
    data = mj.MjData(model)
    torques = []
    for position, velocity, acceleration in zip(q, qd, qdd):
        data.qpos[:] = position
        data.qvel[:] = velocity
        data.qacc[:] = acceleration
        mj.mj_inverse(model, data)
        require(data.nefc == 0, "identification fixture must remain unconstrained")
        torques.append(data.qfrc_inverse.copy())
    result = np.asarray(torques)
    require(np.isfinite(result).all(), "inverse dynamics produced nonfinite torque")
    return result


def generate(directory, spec):
    np, mj = sdk()
    private = directory / ".fixture"
    private.mkdir()
    # Truth belongs solely to fixture generation, not the fitting/Agent surface.
    truth = {"damping": .8, "armature": .12}
    model, source = make_model(directory, ".fixture/truth-model.xml", truth)
    save_json(private / "truth.json", {"source": "synthetic_fixture", "parameters": truth,
              "note": "Generator-only assumed truth; not read by fit or validation."})
    exp = spec["experiment"]
    times = np.arange(exp["samples_per_run"]) * exp["sample_dt_s"]
    runs = []
    for index, (run_id, role, frequency) in enumerate((("excitation-a", "train", .4), ("excitation-b", "train", .65),
                                                      ("excitation-c", "holdout", .95))):
        rng = np.random.default_rng(exp["seed"] + index)
        omega = 2 * np.pi * (frequency + np.arange(7) * .035)
        phase = np.arange(7) * .31 + index * .45
        amplitude = .055 + np.arange(7) * .004
        angle = times[:, None] * omega + phase
        q = amplitude * np.sin(angle)
        qd = amplitude * omega * np.cos(angle)
        qdd = -amplitude * omega**2 * np.sin(angle)
        tau = inverse(model, q, qd, qdd) + rng.normal(0., exp["torque_noise_std_nm"], q.shape)
        filename = run_id + ".csv"
        with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(UNITS)
            for i, t in enumerate(times):
                writer.writerow([float(t), *[float(array[i, j]) for j in range(7) for array in (q, qd, qdd, tau)]])
        runs.append({"run_id": run_id, "role": role, "file": filename, "sha256": sha256(directory / filename),
                     "samples": len(times), "frequency_base_hz": frequency, "random_seed": exp["seed"] + index})
    manifest = {"schema_version": "1.0", "case_id": spec["case_id"], "model_id": spec["model_id"], "source": "synthetic_fixture",
                "physical_validation": "not_established", "joint_order": list(JOINTS), "units": UNITS,
                "frame": "source-MJCF-joint-coordinates", "torque_semantics": "required_generalized_joint_torque_no_external_load",
                "experiment": exp, "runs": runs, "source_assets": source, "spec_sha256": sha256(directory / "intake.json"),
                "generator": {"engine": "MuJoCo", "version": mj.__version__, "method": "analytic prescribed q/qd/qdd + mj_inverse + seeded Gaussian torque noise",
                              "implementation_sha256": sha256(Path(__file__)), "contact": "disabled", "accelerations": "exact analytic fixture values"}}
    validate_dataset(manifest)
    for run in runs:
        read_samples(directory / run["file"], manifest)
    save_json(directory / "dataset.json", manifest)
    return {"phase": "generate", "status": "completed", "artifact": str(directory / "dataset.json"),
            "source": "synthetic_fixture", "training_runs": 2, "held_out_runs": 1, "physical_validation": "not_established"}


def dataset(directory, spec):
    manifest = load_json(directory / "dataset.json")
    validate_dataset(manifest)
    require(manifest["case_id"] == spec["case_id"] and manifest["spec_sha256"] == sha256(directory / "intake.json"), "intake changed after data generation")
    _, current = assets()
    require(current == manifest["source_assets"], "source assets changed since generation")
    return manifest


def samples(directory, manifest, run):
    path = artifact_path(directory, run["file"])
    require(sha256(path) == run["sha256"], "run data checksum mismatch")
    values = read_samples(path, manifest)
    np, _ = sdk()
    arrays = [np.column_stack([values[j + "_" + channel] for j in JOINTS])
              for channel in ("q_rad", "qd_rad_s", "qdd_rad_s2", "tau_nm")]
    require(len(values["time_s"]) == run["samples"], "manifest sample count mismatch")
    return arrays


def seed_model(directory, spec):
    manifest = dataset(directory, spec)
    parameters = {name: value["initial"] for name, value in spec["parameters"].items()}
    _, source = make_model(directory, "seed-model.xml", parameters)
    record = {"schema_version": "1.0", "model_id": spec["model_id"], "parameters": parameters,
              "source": "reference_model_with_assumed_initial_parameters", "file": "seed-model.xml",
              "sha256": sha256(directory / "seed-model.xml"), "source_assets": source,
              "dataset_sha256": sha256(directory / "dataset.json"), "assumptions": spec["assumptions"],
              "physical_validation": "not_established", "dataset_source": manifest["source"]}
    save_json(directory / "model.json", record)
    return {"phase": "model", "status": "completed", "artifact": str(directory / "model.json"),
            "model": str(directory / "seed-model.xml"), "physical_validation": "not_established"}


def checked_model(directory, filename, checksum):
    _, mj = sdk()
    path = artifact_path(directory, filename)
    require(sha256(path) == checksum, "model checksum mismatch")
    return mj.MjModel.from_xml_path(str(path))


def model_record(directory):
    record = load_json(directory / "model.json")
    require(record["dataset_sha256"] == sha256(directory / "dataset.json"), "dataset manifest changed after model creation")
    return record


def fit(directory, spec):
    np, _ = sdk()
    manifest = dataset(directory, spec)
    seed = model_record(directory)
    model = checked_model(directory, seed["file"], seed["sha256"])
    train = [r for r in manifest["runs"] if r["role"] == "train"]
    features, residuals = [], []
    j = JOINTS.index(spec["joint"])
    for run in train:
        q, qd, qdd, tau = samples(directory, manifest, run)
        predicted = inverse(model, q, qd, qdd)
        features.append(np.column_stack((qd[:, j], qdd[:, j])))
        residuals.append(tau[:, j] - predicted[:, j])
    design, residual = np.concatenate(features), np.concatenate(residuals)
    scale = np.linalg.norm(design, axis=0)
    require(np.all(scale > 1e-10), "insufficient velocity/acceleration excitation")
    condition = float(np.linalg.cond(design / scale))
    require(condition < 100., "parameters not separately identifiable from excitation")
    delta, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
    require(rank == 2, "rank-deficient identification data")
    parameters = dict(zip(("damping", "armature"), (float(seed["parameters"][name] + delta[i])
                                                   for i, name in enumerate(("damping", "armature")))))
    for name, value in parameters.items():
        lo, hi = spec["parameters"][name]["bounds"]
        require(lo <= value <= hi, "identified parameter outside declared bounds; no silent clipping")
    _, _ = make_model(directory, "identified-model.xml", parameters)
    record = {"schema_version": "1.0", "source": "synthetic_fixture", "parameters": parameters,
              "parameter_units": {name: p["unit"] for name, p in spec["parameters"].items()},
              "file": "identified-model.xml", "sha256": sha256(directory / "identified-model.xml"),
              "seed_record_sha256": sha256(directory / "model.json"), "dataset_sha256": sha256(directory / "dataset.json"),
              "training_runs": [{"run_id": r["run_id"], "sha256": r["sha256"]} for r in train],
              "holdout_used_for_fitting": False, "fixture_truth_used_for_fitting": False,
              "algorithm": "linear least squares: delta_tau = delta_damping*qd + delta_armature*qdd",
              "rank": int(rank), "column_normalised_condition": condition,
              "training_torque_rmse_nm": float(np.sqrt(np.mean((residual - design @ delta)**2))),
              "physical_validation": "not_established"}
    save_json(directory / "fit.json", record)
    return {"phase": "fit", "status": "completed", "artifact": str(directory / "fit.json"),
            "parameters": parameters, "holdout_used_for_fitting": False, "physical_validation": "not_established"}


def metrics(residual):
    np, _ = sdk()
    return {"rmse_nm": float(np.sqrt(np.mean(residual**2))), "mae_nm": float(np.mean(np.abs(residual))),
            "p95_absolute_error_nm": float(np.percentile(np.abs(residual), 95)), "bias_nm": float(np.mean(residual))}


def validate(directory, spec):
    np, _ = sdk()
    manifest = dataset(directory, spec)
    seed = model_record(directory)
    fitted = load_json(directory / "fit.json")
    require(fitted["seed_record_sha256"] == sha256(directory / "model.json")
            and fitted["dataset_sha256"] == sha256(directory / "dataset.json"), "fit inputs changed")
    require(fitted["holdout_used_for_fitting"] is False and fitted["fixture_truth_used_for_fitting"] is False, "fitting must exclude holdout/truth")
    train = [{"run_id": r["run_id"], "sha256": r["sha256"]} for r in manifest["runs"] if r["role"] == "train"]
    require(fitted["training_runs"] == train, "fit/run split mismatch")
    # Verify training bytes also remain those fitted; holdout is scored only now.
    for run in manifest["runs"]:
        require(sha256(artifact_path(directory, run["file"])) == run["sha256"], "run data checksum mismatch")
    baseline = checked_model(directory, seed["file"], seed["sha256"])
    identified = checked_model(directory, fitted["file"], fitted["sha256"])
    before, after = [], []
    holdout = [r for r in manifest["runs"] if r["role"] == "holdout"]
    j = JOINTS.index(spec["joint"])
    for run in holdout:
        q, qd, qdd, tau = samples(directory, manifest, run)
        before.extend(tau[:, j] - inverse(baseline, q, qd, qdd)[:, j])
        after.extend(tau[:, j] - inverse(identified, q, qd, qdd)[:, j])
    initial, final = metrics(np.asarray(before)), metrics(np.asarray(after))
    improvement = 1 - final["rmse_nm"] / initial["rmse_nm"] if initial["rmse_nm"] > 0 else 0.
    accepted = (final["rmse_nm"] <= spec["acceptance"]["torque_rmse_limit_nm"]
                and improvement >= spec["acceptance"]["minimum_relative_improvement"])
    report = {"schema_version": "1.0", "status": "accepted" if accepted else "rejected", "scope": "synthetic_holdout_only",
              "source": "synthetic_fixture", "physical_validation": "not_established", "baseline": initial, "identified": final,
              "relative_rmse_improvement": improvement, "criteria": spec["acceptance"],
              "held_out_runs": [{"run_id": r["run_id"], "sha256": r["sha256"]} for r in holdout],
              "fit_sha256": sha256(directory / "fit.json"), "identified_model_sha256": fitted["sha256"],
              "independence": "disjoint experiments/seeds/frequencies; same source model and MuJoCo engine; no independent physical evidence",
              "limitations": ["Only two joint2 parameters, not full inertial identification", "Exact kinematic fixture derivatives; no sensor/time-alignment uncertainty",
                              "No external loads, contacts, hardware commands or deployed twin", "Source geometry/inertias and model family assumed correct"]}
    save_json(directory / "validation.json", report)
    return {"phase": "validate", "status": report["status"], "artifact": str(directory / "validation.json"),
            "scope": report["scope"], "baseline_rmse_nm": initial["rmse_nm"], "identified_rmse_nm": final["rmse_nm"],
            "physical_validation": "not_established"}


def main():
    payload = json.loads(sys.stdin.read())
    directory = directory_path(payload["directory"])
    spec = load_json(directory / "intake.json")
    validate_spec(spec)
    phases = {"generate": generate, "model": seed_model, "fit": fit, "validate": validate}
    require(payload["phase"] in phases, "unknown worker phase")
    result = phases[payload["phase"]](directory, spec)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

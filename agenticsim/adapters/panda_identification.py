"""Public Panda measurements -> nominal MJCF -> bounded correction -> holdout.

Only the isolated identification worker imports NumPy, pandas, SciPy or MuJoCo.
The nominal reference is not asserted identical to the acquisition robot.
"""
import csv
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
import zipfile

from agenticsim.domain import require
from agenticsim.domain.calibration import JOINTS, UNITS
from agenticsim.domain.measurements import TRAIN, HOLDOUT, PARAMETER_UNITS, read_samples, validate_dataset, validate_spec
from agenticsim.runtime.calibration import artifact_path, directory_path, load_json, save_json, sha256
from agenticsim.runtime.paths import ROOT

REFERENCE = ROOT / "assets/robots/panda-identification"


def sdk():
    import numpy as np
    import mujoco as mj
    require(mj.__version__ == "3.8.1", "pinned MuJoCo 3.8.1 required")
    return np, mj


def reference():
    record = load_json(REFERENCE / "manifest.json")
    require(record["commit"] == "8161bba264d7fa7c99ca301e91e7fb44737676ad", "pinned Panda reference required")
    for item in record["files"]:
        require(sha256(artifact_path(REFERENCE, item["path"])) == item["sha256"], "Panda reference checksum mismatch")
    require({r["path"] for r in record["files"]} == {"panda.xml", "LICENSE", "README.md"}, "reference notices/XML missing")
    return record


def make_model(directory, filename, parameters):
    reference()
    tree = ET.parse(REFERENCE / "panda.xml")
    root = tree.getroot()
    # All bodies retain their explicitly declared inertials. Contacts/meshes
    # are irrelevant to this free-space inverse-dynamics benchmark.
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in {"geom", "material"}:
                parent.remove(child)
    for name in ("asset", "actuator", "tendon", "equality", "keyframe", "contact"):
        node = root.find(name)
        if node is not None:
            root.remove(node)
    root.find("compiler").attrib.pop("meshdir", None)
    root.find("option").set("gravity", "0 0 -9.81")
    ET.SubElement(root.find("option"), "flag", contact="disable")
    # Weld stationary fingers at their source zero position, preserving mass.
    for body in root.findall(".//worldbody//body"):
        for joint in list(body.findall("joint")):
            if joint.get("name", "").startswith("finger_joint"):
                body.remove(joint)
    for name in JOINTS:
        joint = root.find(f".//worldbody//joint[@name='{name}']")
        require(joint is not None, "source arm joint missing")
        for key in ("damping", "armature"):
            joint.set(key, str(parameters[name][key]))
        joint.set("frictionloss", "0")
    path = artifact_path(directory, filename)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    _, mj = sdk()
    model = mj.MjModel.from_xml_path(str(path))
    require(model.nq == model.nv == 7, "generated Panda must have seven named scalar joints")
    require(tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(7)) == JOINTS, "compiled Panda joint order mismatch")
    for j, name in enumerate(JOINTS):
        require(abs(model.dof_damping[j] - parameters[name]["damping"]) < 1e-12
                and abs(model.dof_armature[j] - parameters[name]["armature"]) < 1e-12, "compiled correction mismatch")
    return model


def ingest(directory, spec):
    from agenticsim.adapters.merl_panda import ARCHIVE, SHA256, member, read_experiment, verify_archive
    np, _ = sdk()
    verify_archive()
    source = reference()
    runs = []
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for seed in TRAIN + HOLDOUT:
            frame, checksum = read_experiment(bundle, seed)
            times = frame.t.to_numpy()
            require(np.all(np.diff(times) > 0) and times[0] >= 0, "acquisition clock must increase strictly")
            indices = np.arange(0, len(frame), spec["sample_stride"])
            arrays = [np.column_stack([frame[f"{prefix}_{j}"].to_numpy() for j in range(1, 8)])
                      for prefix in ("q", "dq", "ddq", "tau_interp")]
            run_id = f"panda-seed-{seed}"
            path = directory / (run_id + ".csv")
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(UNITS)
                for i in indices:
                    writer.writerow([float(times[i]), *[float(a[i, j]) for j in range(7) for a in arrays]])
            clock = np.diff(times)
            # Diagnostics only: published acausal/filtering processing is not
            # reconstructed or retuned using the held-out experiments.
            derivative = np.gradient(arrays[0], times, axis=0)
            runs.append({"seed": seed, "run_id": run_id, "role": "train" if seed in TRAIN else "holdout",
                         "file": path.name, "sha256": sha256(path), "samples": len(indices),
                         "source_file": member(seed), "source_sha256": checksum, "original_samples": len(frame),
                         "clock": {"start_s": float(times[0]), "end_s": float(times[-1]),
                                   "median_dt_s": float(np.median(clock)), "max_dt_s": float(clock.max()),
                                   "uniform": bool(np.allclose(clock, clock[0], atol=1e-8))},
                         "diagnostics": {"position_velocity_derivative_rmse_rad_s": np.sqrt(np.mean((derivative - arrays[1])**2, axis=0)).tolist(),
                                         "tau_vs_tau_interp_rmse_nm": [float(np.sqrt(np.mean((frame[f'tau_{j}'] - frame[f'tau_interp_{j}'])**2))) for j in range(1, 8)]}})
    record = {"schema_version": "1.0", "model_id": "franka-panda", "source": "measurement",
              "case_id": spec["case_id"], "mapping_status": "provisional", "physical_validation": "not_established",
              "joint_order": list(JOINTS), "units": UNITS, "frame": "provisional-source-joint-coordinates",
              "torque_semantics": "publisher_tau_interp_inverse_dynamics_target",
              "archive_sha256": SHA256, "license": "CC-BY-SA-4.0", "attribution": "MERL, 2024",
              "spec_sha256": sha256(directory / "intake.json"), "source_assets": source, "runs": runs,
              "mapping": spec["mapping"], "sources": spec["sources"], "assumptions": spec["assumptions"],
              "processing": {"filter": "publisher 4 Hz filtered files", "accelerations": "publisher acausal derivative of velocity",
                             "stride": spec["sample_stride"], "clock": "original recorded irregular t; no uniform resampling",
                             "importer": "hash-verified pinned experiment; allowlisted legacy DataFrame constructors",
                             "no_simulated_or_learned_payloads": True,
                             "reader_sha256": sha256(Path(__file__).with_name("merl_panda.py")),
                             "implementation_sha256": sha256(Path(__file__))}}
    validate_dataset(record)
    for run in runs:
        read_samples(directory / run["file"], record)
    save_json(directory / "dataset.json", record)
    return {"phase": "ingest", "status": "completed", "source": "measurement", "training_runs": len(TRAIN),
            "held_out_runs": len(HOLDOUT), "artifact": str(directory / "dataset.json"), "physical_validation": "not_established"}


def dataset(directory, spec):
    check_receipt(directory, "ingest", "dataset.json")
    data = load_json(directory / "dataset.json")
    validate_dataset(data)
    require(data["spec_sha256"] == sha256(directory / "intake.json") and data["case_id"] == spec["case_id"], "identification intake changed")
    require(data["source_assets"] == reference(), "reference changed after ingest")
    return data


def samples(directory, data, run):
    np, _ = sdk()
    path = artifact_path(directory, run["file"])
    require(sha256(path) == run["sha256"], "measurement file checksum mismatch")
    columns = read_samples(path, data)
    require(len(columns["time_s"]) == run["samples"], "measurement count mismatch")
    return [np.column_stack([columns[j + "_" + channel] for j in JOINTS])
            for channel in ("q_rad", "qd_rad_s", "qdd_rad_s2", "tau_nm")]


def inverse(model, q, qd, qdd):
    np, mj = sdk()
    require(q.shape == qd.shape == qdd.shape and q.shape[1] == 7, "measurement shape mismatch")
    require(np.all(q >= model.jnt_range[:, 0]) and np.all(q <= model.jnt_range[:, 1]), "measured pose outside nominal joint limits")
    state = mj.MjData(model)
    result = []
    for position, velocity, acceleration in zip(q, qd, qdd):
        state.qpos[:] = position
        state.qvel[:] = velocity
        state.qacc[:] = acceleration
        mj.mj_inverse(model, state)
        require(state.nefc == 0, "free-space inverse model unexpectedly constrained")
        result.append(state.qfrc_inverse.copy())
    torque = np.asarray(result)
    require(np.isfinite(torque).all(), "nonfinite inverse prediction")
    return torque


def seed_model(directory, spec):
    dataset(directory, spec)
    parameters = {j: dict.fromkeys(PARAMETER_UNITS, 0.) for j in JOINTS}
    make_model(directory, "seed-model.xml", parameters)
    record = {"schema_version": "1.0", "model_id": "franka-panda", "file": "seed-model.xml",
              "sha256": sha256(directory / "seed-model.xml"), "dataset_sha256": sha256(directory / "dataset.json"),
              "parameters": parameters, "source_assets": reference(), "physical_validation": "not_established",
              "assumptions": spec["assumptions"] + ["Nominal seed sets arm damping/armature to zero before fitting"]}
    save_json(directory / "model.json", record)
    return {"phase": "model", "status": "completed", "artifact": str(directory / "model.json"), "model": str(directory / "seed-model.xml")}


def checked_model(directory, record):
    _, mj = sdk()
    require(sha256(directory / record["file"]) == record["sha256"], "compiled model changed")
    return mj.MjModel.from_xml_path(str(directory / record["file"]))


def model_record(directory):
    check_receipt(directory, "model", "model.json")
    record = load_json(directory / "model.json")
    require(record["dataset_sha256"] == sha256(directory / "dataset.json"), "dataset changed after model creation")
    return record


def check_receipt(directory, phase, filename):
    receipt = load_json(directory / "receipts" / (phase + ".json"))
    require(receipt["phase"] == phase and receipt["file"] == filename
            and receipt["sha256"] == sha256(directory / filename), "recorded phase evidence changed")


def features(qd, qdd, joint, spec):
    np, _ = sdk()
    velocity = qd[:, joint]
    return np.column_stack((velocity, qdd[:, joint], np.tanh(velocity / spec["smooth_velocity_rad_s"]), np.ones(len(velocity))))


def fit(directory, spec):
    np, _ = sdk()
    from scipy.optimize import lsq_linear
    data, seed = dataset(directory, spec), model_record(directory)
    model = checked_model(directory, seed)
    train = [r for r in data["runs"] if r["role"] == "train"]
    velocities, accelerations, residuals = [], [], []
    for run in train:
        q, qd, qdd, tau = samples(directory, data, run)
        residuals.append(tau - inverse(model, q, qd, qdd))
        velocities.append(qd)
        accelerations.append(qdd)
    qd, qdd, residual = np.concatenate(velocities), np.concatenate(accelerations), np.concatenate(residuals)
    parameters, diagnostics = {}, {}
    lower = np.array([spec["bounds"][name][0] for name in PARAMETER_UNITS])
    upper = np.array([spec["bounds"][name][1] for name in PARAMETER_UNITS])
    for j, name in enumerate(JOINTS):
        design = features(qd, qdd, j, spec)
        scale = np.linalg.norm(design, axis=0)
        require(np.all(scale > 1e-8) and np.linalg.matrix_rank(design) == 4, "insufficient per-joint excitation")
        condition = float(np.linalg.cond(design / scale))
        require(condition < 1000, "ill-conditioned per-joint identification")
        result = lsq_linear(design, residual[:, j], bounds=(lower, upper), tol=1e-10)
        require(result.success and np.isfinite(result.x).all(), "bounded least-squares failed")
        parameters[name] = dict(zip(PARAMETER_UNITS, map(float, result.x)))
        diagnostics[name] = {"column_normalised_condition": condition, "active_bounds": result.active_mask.tolist(),
                             "training_rmse_nm": float(np.sqrt(np.mean((residual[:, j] - design @ result.x)**2)))}
    make_model(directory, "identified-model.xml", parameters)
    record = {"schema_version": "1.0", "source": "measurement", "prediction_source": "learned_prediction",
              "file": "identified-model.xml", "sha256": sha256(directory / "identified-model.xml"),
              "parameters": parameters, "parameter_units": PARAMETER_UNITS,
              "prediction_equation": "mj_inverse(identified MJCF) + coulomb*tanh(qd/smooth_velocity) + offset",
              "smooth_velocity_rad_s": spec["smooth_velocity_rad_s"], "diagnostics": diagnostics,
              "seed_record_sha256": sha256(directory / "model.json"), "dataset_sha256": sha256(directory / "dataset.json"),
              "training_runs": [{"run_id": r["run_id"], "sha256": r["sha256"]} for r in train],
              "holdout_used_for_fitting": False, "algorithm": "per-joint bounded linear least squares; four correction terms",
              "physical_validation": "not_established", "parameter_interpretation": "effective corrections; no uniqueness/manufacturer parameter claim"}
    save_json(directory / "fit.json", record)
    return {"phase": "fit", "status": "completed", "artifact": str(directory / "fit.json"), "parameters": parameters,
            "holdout_used_for_fitting": False, "physical_validation": "not_established"}


def metrics(residual):
    np, _ = sdk()
    return {"pooled_rmse_nm": float(np.sqrt(np.mean(residual**2))),
            "joint_rmse_nm": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
            "joint_mae_nm": np.mean(np.abs(residual), axis=0).tolist(),
            "joint_bias_nm": np.mean(residual, axis=0).tolist(),
            "joint_p95_absolute_error_nm": np.percentile(np.abs(residual), 95, axis=0).tolist()}


def validate(directory, spec):
    np, _ = sdk()
    check_receipt(directory, "fit", "fit.json")
    data, seed, fitted = dataset(directory, spec), model_record(directory), load_json(directory / "fit.json")
    require(fitted["seed_record_sha256"] == sha256(directory / "model.json")
            and fitted["dataset_sha256"] == sha256(directory / "dataset.json"), "fit inputs changed")
    require(fitted["holdout_used_for_fitting"] is False and fitted["training_runs"] == [
        {"run_id": r["run_id"], "sha256": r["sha256"]} for r in data["runs"] if r["role"] == "train"], "fit split mismatch")
    for run in data["runs"]:
        require(sha256(artifact_path(directory, run["file"])) == run["sha256"], "measurement file checksum mismatch")
    baseline, identified = checked_model(directory, seed), checked_model(directory, fitted)
    before, after, per_run = [], [], []
    for run in (r for r in data["runs"] if r["role"] == "holdout"):
        q, qd, qdd, tau = samples(directory, data, run)
        initial = tau - inverse(baseline, q, qd, qdd)
        prediction = inverse(identified, q, qd, qdd)
        for j, name in enumerate(JOINTS):
            p = fitted["parameters"][name]
            prediction[:, j] += p["coulomb"] * np.tanh(qd[:, j] / fitted["smooth_velocity_rad_s"]) + p["offset"]
        final = tau - prediction
        before.append(initial)
        after.append(final)
        per_run.append({"run_id": run["run_id"], "sha256": run["sha256"], "samples": len(tau),
                        "baseline": metrics(initial), "identified": metrics(final)})
    initial, final = metrics(np.concatenate(before)), metrics(np.concatenate(after))
    improvement = 1 - final["pooled_rmse_nm"] / initial["pooled_rmse_nm"] if initial["pooled_rmse_nm"] else 0.
    criteria = spec["acceptance"]
    checks = {"absolute_rmse": final["pooled_rmse_nm"] <= criteria["pooled_rmse_limit_nm"],
              "relative_improvement": improvement >= criteria["minimum_relative_improvement"],
              "no_joint_regression": all(a <= b + criteria["maximum_joint_rmse_regression_nm"] for a, b in zip(final["joint_rmse_nm"], initial["joint_rmse_nm"]))}
    report = {"schema_version": "1.0", "status": "accepted" if all(checks.values()) else "rejected",
              "scope": "public_dataset_holdout_only", "source": "measurement", "prediction_source": "learned_prediction",
              "physical_validation": "not_established", "mapping_status": "provisional", "baseline": initial, "identified": final,
              "relative_rmse_improvement": improvement, "criteria": criteria, "checks": checks, "held_out_runs": per_run,
              "fit_sha256": sha256(directory / "fit.json"), "identified_model_sha256": fitted["sha256"],
              "independence": "13 publisher training excitations versus 16 wider-frequency real-robot test excitations; variants never split across roles",
              "limitations": spec["assumptions"] + ["Offline conditional dataset evidence, not target equipment calibration or safety acceptance",
                  "MJCF contains damping/armature; fitted Coulomb/offset are explicit predictor terms in fit.json, not embedded MJCF dynamics",
                  "No full mass/inertia identification, forward trajectory/control validation or uncertainty intervals"]}
    save_json(directory / "validation.json", report)
    return {"phase": "validate", "status": report["status"], "artifact": str(directory / "validation.json"),
            "baseline_rmse_nm": initial["pooled_rmse_nm"], "identified_rmse_nm": final["pooled_rmse_nm"],
            "checks": checks, "scope": report["scope"], "physical_validation": "not_established"}


def main():
    payload = json.loads(sys.stdin.read())
    directory = directory_path(payload["directory"])
    spec = load_json(directory / "intake.json")
    validate_spec(spec)
    phases = {"ingest": ingest, "model": seed_model, "fit": fit, "validate": validate}
    require(payload["phase"] in phases, "unknown identification phase")
    print(json.dumps(phases[payload["phase"]](directory, spec), allow_nan=False))


if __name__ == "__main__":
    main()

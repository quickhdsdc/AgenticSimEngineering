"""Validate the registered robot scene and full solver pose artifact."""
import hashlib
import json
import math
from pathlib import Path
import struct

from agenticsim.adapters.asset_robot import AssetRobot, CATALOG, ASSETS
from agenticsim.domain import EngineeringTask, Evaluation, assess, require, number
from .queue import digest
from agenticsim.runtime.paths import RUNS, relocated_artifact


def checked_json(path, checksum):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == checksum, "robot artifact checksum mismatch")
    return json.loads(raw)


def robot_replay(path, bundle=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    study = record["payload"]
    require(record.get("sha256") == digest(study), "Study payload checksum mismatch")
    require(study.get("schema_version") == "1.0" and study.get("status") in ("completed", "completed_with_errors"), "completed Study required")
    task = EngineeringTask(study["task"])
    adapter = AssetRobot(task.document["model"]["id"])
    adapter.validate(task)
    manifest, files = adapter.verify_assets()
    runs = study["runs"]
    candidates = {c["id"]: c for c in task.document["candidates"]}
    require(isinstance(runs, list) and all(isinstance(r, dict) for r in runs) and len(runs) == len(candidates), "one run per candidate required")
    require({r.get("candidate_id") for r in runs} == set(candidates), "run/candidate mapping mismatch")
    require(all(isinstance(r.get("run_id"), str) and r["run_id"] for r in runs) and len({r["run_id"] for r in runs}) == len(runs), "unique run IDs required")
    replay = {"schema_version": "1.0", "mapping": "newton-robot-playback/1", "study_id": study["study_id"],
              "task_id": task.document["task_id"], "study_sha256": record["sha256"], "time_unit": "s",
              "input_source": task.document["context"]["source"], "description": task.document["context"]["description"],
              "decision": study["decision"], "model": task.document["model"], "runs": []}
    copies = {}
    for index, run in enumerate(runs):
        mapped = {"run_id": run["run_id"], "candidate_id": run["candidate_id"], "status": run["status"]}
        replay["runs"].append(mapped)
        if run["status"] != "succeeded":
            require(run["status"] in ("failed", "timed_out", "cancelled") and "result" not in run, "invalid failed run")
            mapped.update(duration=None, error=run.get("error", {}))
            continue
        evaluation = Evaluation.from_dict(run["result"]); evaluation.validate(adapter.model)
        require(run.get("assessment") == assess(task, evaluation), "assessment does not match observations")
        prov = evaluation.provenance
        require(evaluation.source == "reference_simulation" and prov.get("input_source") == task.document["context"]["source"], "robot provenance mismatch")
        require(prov.get("engine") == "Newton" and prov.get("execution") == task.document["execution"], "robot engine/config mismatch")
        require(prov.get("asset_commit") == manifest["commit"] and prov.get("asset_files") == files, "robot asset provenance mismatch")
        descriptors = prov.get("robot_playback")
        require(isinstance(descriptors, dict), "Study has no full robot pose artifacts; rerun with current adapter")
        prefix = f"robot-source/run-{index}"
        paths = {}
        for kind in ("scene", "trajectory"):
            original = relocated_artifact(descriptors[kind]["path"]) if not bundle else None
            filename = f"{prefix}/robot-{kind}.json"
            source = Path(bundle) / filename if bundle else original
            if not bundle:
                require(original.is_relative_to(RUNS), "robot source must be in project runs")
            paths[kind] = source
            copies[filename] = source
        scene = checked_json(paths["scene"], descriptors["scene"]["sha256"])
        trace = checked_json(paths["trajectory"], descriptors["trajectory"]["sha256"])
        require(scene["schema_version"] == trace["schema_version"] == "1.0" and scene["model_id"] == adapter.model_id, "robot artifact schema/model mismatch")
        require(scene["length_unit"] == trace["length_unit"] == "m" and trace["time_unit"] == "s" and trace["angle_unit"] == "rad", "robot artifact units mismatch")
        require(scene["frame"] == trace["frame"] == "right-handed-Z-up" and scene["quaternion"] == trace["quaternion"] == "xyzw", "robot frame/quaternion mismatch")
        bodies = scene["bodies"]
        require(len(bodies) > 0 and len(set(bodies)) == len(bodies) and trace["body_order"] == bodies, "body order mismatch")
        require(trace["joint_order"] == list(adapter.joints) == list(prov["joint_order"]), "joint order mismatch")
        require(len(scene["joint_bodies"]) == len(adapter.joints) and all(type(i) is int and 0 <= i < len(bodies) for i in scene["joint_bodies"]), "joint/body mapping mismatch")
        dt, duration = task.document["execution"]["dt_s"], task.document["execution"]["duration_s"]
        times = trace["times"]; n = round(duration / dt) + 1
        require(len(times) == n and all(number(t) and abs(t - i * dt) < 1e-9 for i, t in enumerate(times)), "robot time axis mismatch")
        require(len(trace["body_poses"]) == len(trace["joint_values"]) == len(trace["contact_counts"]) == n, "robot frame count mismatch")
        for poses in trace["body_poses"]:
            require(len(poses) == len(bodies), "body pose count mismatch")
            for pose in poses:
                require(len(pose) == 7 and all(number(v) for v in pose) and abs(sum(v*v for v in pose[3:]) - 1) < 1e-4, "invalid body pose/quaternion")
        require(all(len(q) == len(adapter.joints) and all(number(v) for v in q) for q in trace["joint_values"]), "invalid joint frame")
        for j, name in enumerate(adapter.joints):
            observation = next(o for o in evaluation.observations if o.semantic_id == adapter.prefix + name)
            require(observation.axis == tuple(times) and observation.axis_unit == "s" and all(abs(v - trace["joint_values"][i][j]) < 1e-9 for i, v in enumerate(observation.values)), "joint frame/observation mismatch")
        root = bodies.index("pelvis" if adapter.floating else "base")
        height = next(o for o in evaluation.observations if o.semantic_id == adapter.prefix + "base-height")
        require(height.axis == tuple(times) and all(abs(v - trace["body_poses"][i][root][2]) < 1e-9 for i, v in enumerate(height.values)), "base pose/observation mismatch")
        require(trace["source"] == "reference_simulation" and trace["pose_source"] == "Newton state.body_q at every integration step", "pose provenance mismatch")
        require(trace["contact_counts"][0] is None and all(type(v) is int and v >= 0 for v in trace["contact_counts"][1:]), "invalid contact samples")
        maximum = next(o for o in evaluation.observations if o.semantic_id == adapter.prefix + "max-contact-count").values[0]
        require(maximum == max(trace["contact_counts"][1:]), "contact count/observation mismatch")
        mesh_ids = set()
        for mesh in scene["meshes"]:
            filename = mesh["file"]; source = (paths["scene"].parent / filename).resolve()
            require(source.is_relative_to(paths["scene"].parent.resolve()) and source.is_file(), "invalid robot mesh path")
            raw = source.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == mesh["sha256"] and len(raw) == mesh["bytes"] == mesh["vertex_count"] * 24, "robot mesh checksum/size mismatch")
            require(mesh["encoding"] == "float32-le-positions-then-normals" and mesh["vertex_count"] > 0 and mesh["vertex_count"] % 3 == 0, "invalid robot mesh encoding")
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)), "non-finite robot mesh")
            require(mesh["id"] not in mesh_ids, "duplicate mesh id"); mesh_ids.add(mesh["id"])
            copies[f"{prefix}/{filename}"] = source
            mesh["file"] = f"{prefix}/{filename}"
        for geom in scene["geometries"]:
            require(type(geom["body"]) is int and 0 <= geom["body"] < len(bodies) and geom["mesh"] in mesh_ids, "invalid geometry mapping")
            require(len(geom["position"]) == 3 and len(geom["quaternion_xyzw"]) == 4 and all(number(v) for v in geom["position"] + geom["quaternion_xyzw"]), "invalid geometry transform")
        copies[f"{prefix}/ROBOT-LICENSE.txt"] = (Path(bundle) / prefix / "ROBOT-LICENSE.txt") if bundle else ASSETS / adapter.folder / "LICENSE"
        target = list(task.document["inputs"]["target_q"]["values"])
        candidate = candidates[run["candidate_id"]]["parameters"]
        target[candidate["joint_index"]["value"]] += candidate["target_offset"]["value"]
        mapped.update(source=evaluation.source, assessment=run["assessment"], provenance=prov, scene=scene,
                      times=times, body_poses=trace["body_poses"], joint_values=trace["joint_values"],
                      contact_counts=trace["contact_counts"], target_q=target, duration=duration,
                      license=f"{prefix}/ROBOT-LICENSE.txt")
    accepted = [r for r in replay["runs"] if r.get("assessment", {}).get("status") == "accepted"]
    failed = any(r["status"] != "succeeded" for r in runs)
    best = (min if task.document["objective"]["direction"] == "minimize" else max)(accepted, key=lambda r: r["assessment"]["objective"]) if accepted else None
    require(study["status"] == ("completed_with_errors" if failed else "completed"), "Study/run status mismatch")
    require(study["decision"] == {"status": "inconclusive" if failed or not best else "candidate_selected", "best_observed_candidate": best["candidate_id"] if best else None, "scope": "supplied_candidates_and_model_constraints_only", "physical_validation": "not_established"}, "decision/assessment mismatch")
    return replay, copies


def playback(path, bundle=None):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record["payload"]["task"]["model"]["id"] in CATALOG:
        return robot_replay(path, bundle)
    from .queue import queue_replay
    return queue_replay(Path(path)), {}

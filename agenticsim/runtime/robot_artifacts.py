"""Export compiled visual meshes and solver body poses in the isolated worker."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from agenticsim.domain import require

ROOT = Path(__file__).resolve().parents[2]


def save_robot_playback(native, native_state, body_names, poses, times, qs, contacts, adapter):
    import numpy as np
    import mujoco

    directory = Path(os.environ.get("ASE_RUN_DIRECTORY", ROOT / "work/runs/robot-playback" / uuid4().hex)).resolve()
    require(directory.is_relative_to(ROOT / "work/runs"), "robot artifact directory must be project-local")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meshes").mkdir(exist_ok=True)
    native_ids = {mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, name): i for i, name in enumerate(body_names)}
    meshes, geoms = {}, []
    for g in range(native.ngeom):
        # Pinned assets explicitly separate detailed visuals (group 2) and collisions (group 3).
        if native.geom_group[g] != 2:
            continue
        require(native.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH, "unsupported robot visual geometry")
        mid = int(native.geom_dataid[g])
        if mid not in meshes:
            v0, vn = int(native.mesh_vertadr[mid]), int(native.mesh_vertnum[mid])
            f0, fn = int(native.mesh_faceadr[mid]), int(native.mesh_facenum[mid])
            n0, nn = int(native.mesh_normaladr[mid]), int(native.mesh_normalnum[mid])
            faces = native.mesh_face[f0:f0 + fn]
            vertices = native.mesh_vert[v0:v0 + vn][faces].reshape(-1, 3)
            normal_ids = native.mesh_facenormal[f0:f0 + fn]
            normals = native.mesh_normal[n0:n0 + nn][normal_ids].reshape(-1, 3)
            packed = np.concatenate([vertices.reshape(-1), normals.reshape(-1)]).astype('<f4')
            require(np.isfinite(packed).all(), "non-finite robot visual mesh")
            filename = f"meshes/mesh-{mid}.bin"
            raw = packed.tobytes(); (directory / filename).write_bytes(raw)
            meshes[mid] = {"id": mid, "file": filename, "vertex_count": len(vertices),
                           "encoding": "float32-le-positions-then-normals", "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest()}
        # Resolve welded visual-only descendants to an imported solver body.
        ancestor = int(native.geom_bodyid[g])
        while ancestor and ancestor not in native_ids:
            ancestor = int(native.body_parentid[ancestor])
        require(ancestor in native_ids, "robot visual has no solver body mapping")
        rotation = native_state.xmat[ancestor].reshape(3, 3)
        local_pos = rotation.T @ (native_state.geom_xpos[g] - native_state.xpos[ancestor])
        local_rotation = rotation.T @ native_state.geom_xmat[g].reshape(3, 3)
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, local_rotation.reshape(-1))
        material = int(native.geom_matid[g])
        rgba = native.mat_rgba[material] if material >= 0 else native.geom_rgba[g]
        geoms.append({"id": f"geom-{g}", "body": native_ids[ancestor], "mesh": mid,
                      "position": local_pos.tolist(), "quaternion_xyzw": [*quat[1:].tolist(), float(quat[0])],
                      "rgba": rgba.tolist(), "specular": float(native.mat_specular[material]) if material >= 0 else .3,
                      "shininess": float(native.mat_shininess[material]) if material >= 0 else .2})
    require(len(geoms) > 0, "robot visual meshes missing")
    scene = {"schema_version": "1.0", "model_id": adapter.model_id, "length_unit": "m",
             "frame": "right-handed-Z-up", "quaternion": "xyzw", "bodies": body_names,
             "meshes": list(meshes.values()), "geometries": geoms,
             "geometry_source": "pinned_menagerie_visual_meshes", "asset_folder": adapter.folder,
             "floor_z_m": 0, "fixed_body_mapping": "nearest imported ancestor; compiled initial local transform"}
    scene["joint_bodies"] = []
    for name in adapter.joints:
        jid = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, name)
        bid = int(native.jnt_bodyid[jid])
        while bid not in native_ids and bid:
            bid = int(native.body_parentid[bid])
        require(bid in native_ids, "joint visual body mapping missing")
        scene["joint_bodies"].append(native_ids[bid])
    trajectory = {"schema_version": "1.0", "time_unit": "s", "length_unit": "m", "angle_unit": "rad",
                  "frame": scene["frame"], "quaternion": "xyzw", "body_order": body_names,
                  "joint_order": list(adapter.joints), "times": list(times),
                  "body_poses": np.asarray(poses).tolist(), "joint_values": qs.tolist(),
                  "contact_counts": [None, *contacts], "source": "reference_simulation",
                  "pose_source": "Newton state.body_q at every integration step"}
    descriptors = {}
    for name, payload in [("scene", scene), ("trajectory", trajectory)]:
        path = directory / f"robot-{name}.json"
        path.write_text(json.dumps(payload, allow_nan=False, separators=(',', ':')), encoding="utf-8")
        descriptors[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return descriptors

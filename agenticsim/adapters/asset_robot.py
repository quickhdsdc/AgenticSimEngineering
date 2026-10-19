"""Versioned industrial/humanoid assets through Newton's MJCF importer and SolverMuJoCo.

MJCF is a model format. All stepping here goes through Newton; the selected
numerical backend (MuJoCo CPU or MuJoCo-Warp CUDA) is recorded explicitly.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from agenticsim.domain import BackendDescriptor, EngineeringTask, Evaluation, ModelDefinition, Observable, number, require

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/robots/menagerie"
CATALOG = {
    "kuka-iiwa14": ("kuka_iiwa_14", "iiwa14.xml", "home", False),
    "unitree-g1": ("unitree_g1", "g1.xml", "stand", True),
}


class AssetRobot:
    backend = BackendDescriptor("newton-mujoco", "1", ("evaluate",))

    def __init__(self, model_id):
        require(model_id in CATALOG, "unknown robot asset")
        self.model_id = model_id
        self.folder, filename, self.key, self.floating = CATALOG[model_id]
        self.asset = ASSETS / self.folder / filename
        self.scene = ASSETS / self.folder / "scene.xml"
        require(self.asset.is_file(), "robot assets missing; run scripts/fetch_robot_assets.py")
        xml = ET.parse(self.asset).getroot()
        self.joints = tuple(a.attrib["joint"] for a in xml.find("actuator"))
        self.home = tuple(float(v) for v in xml.find(f"keyframe/key[@name='{self.key}']").attrib["ctrl"].split())
        self.prefix = f"urn:ase:{model_id}:"
        self.scalars = ((self.prefix + "final-joint-error", "rad"), (self.prefix + "min-base-height", "m"),
                        (self.prefix + "max-contact-count", "1"))
        self.model = ModelDefinition(model_id, "1", (
            ("initial_q", self.prefix + "initial-q", "rad"), ("target_q", self.prefix + "target-q", "rad")),
            self.scalars + ((self.prefix + "base-height", "m"),) +
            tuple((self.prefix + name, "rad") for name in self.joints),
            ("Pinned Menagerie robot description and scene; no physical validation claim",
             "Named scalar joints ordered by source actuator list: " + ", ".join(self.joints),
             "Source actuators via ctrl_direct; native limits, armature, collision geometry and self-collisions imported",
             "Floating base with floor contacts" if self.floating else "Fixed-base industrial arm with floor",
             "Short-horizon posture control; not a locomotion policy or balance guarantee"))

    def verify_assets(self):
        manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
        records = [r for r in manifest["files"] if r["path"].startswith(self.folder + "/")]
        require(bool(records), "asset manifest entry missing")
        for record in records:
            path = (ASSETS / record["path"]).resolve()
            require(path.is_relative_to(ASSETS.resolve()), "asset path escape")
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                    f"asset integrity mismatch: {record['path']}")
        return manifest, records

    def validate(self, task):
        task.validate(self.model, self.backend)
        for value in task.document["inputs"].values():
            require(len(value["values"]) == len(self.joints) and all(abs(v) <= 3.2 for v in value["values"]),
                    "joint vector size or radian range mismatch")
        cfg = task.document.get("execution")
        require(isinstance(cfg, dict) and set(cfg) == {"device", "dt_s", "duration_s", "algorithm"}, "execution config required")
        require(cfg["device"] in ("cpu", "cuda:0") and cfg["algorithm"] == "mujoco-implicitfast", "unsupported execution mode")
        require(number(cfg["dt_s"]) and 0.0005 <= cfg["dt_s"] <= 0.002, "dt outside [0.0005, 0.002]")
        require(number(cfg["duration_s"]) and 0 < cfg["duration_s"] <= 2., "duration outside (0, 2]")
        steps = cfg["duration_s"] / cfg["dt_s"]
        require(1 <= steps <= 4000 and abs(steps - round(steps)) < 1e-8, "duration must be integral steps")
        for c in task.document["candidates"]:
            require(set(c) == {"id", "parameters"} and isinstance(c["parameters"], dict)
                    and set(c["parameters"]) == {"joint_index", "target_offset"}, "invalid candidate parameters")
            index, offset = c["parameters"]["joint_index"], c["parameters"]["target_offset"]
            require(isinstance(index, dict) and set(index) == {"value", "unit"} and index["unit"] == "1"
                    and type(index["value"]) is int and 0 <= index["value"] < len(self.joints), "invalid joint index")
            require(isinstance(offset, dict) and set(offset) == {"value", "unit"} and offset["unit"] == "rad"
                    and number(offset["value"]) and abs(offset["value"]) <= 0.2, "offset must be radians within +/-0.2")
        for criterion in [task.document["objective"], *task.document["constraints"]]:
            require((criterion["semantic_id"], criterion["unit"]) in self.scalars, "criterion must be scalar")
        self.verify_assets()

    def evaluate(self, task: EngineeringTask, candidate):
        self.validate(task)
        require(candidate in task.document["candidates"], "unregistered candidate")
        import numpy as np
        import warp as wp
        import newton
        import mujoco
        import mujoco_warp  # noqa: F401 - required optional backend registration
        require(newton.__version__ == "1.2.1" and wp.__version__ == "1.13.0", "pinned Newton environment required")
        wp.config.kernel_cache_dir = str(ROOT / "work/cache/warp")
        cfg = task.document["execution"]
        native = mujoco.MjModel.from_xml_path(str(self.scene))
        native.opt.timestep = cfg["dt_s"]
        q0 = np.asarray(task.document["inputs"]["initial_q"]["values"], dtype=np.float64)
        target = np.asarray(task.document["inputs"]["target_q"]["values"], dtype=np.float64)
        p = candidate["parameters"]
        target[p["joint_index"]["value"]] += p["target_offset"]["value"]
        native_jids = [mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joints]
        require(all(j >= 0 for j in native_jids), "joint name missing from source compiler")
        require(native.nu == len(self.joints) and list(native.actuator_trnid[:, 0]) == native_jids, "actuator ordering mismatch")
        for k, j in enumerate(native_jids):
            if native.jnt_limited[j]:
                lo, hi = native.jnt_range[j]
                require(lo <= q0[k] <= hi and lo <= target[k] <= hi, f"joint limits violated: {self.joints[k]}")
            if native.actuator_ctrllimited[k]:
                lo, hi = native.actuator_ctrlrange[k]
                require(lo <= target[k] <= hi, f"actuator target limit violated: {self.joints[k]}")
        with wp.ScopedDevice(cfg["device"]):
            builder = newton.ModelBuilder()
            builder.add_mjcf(str(self.scene), ctrl_direct=True, parse_visuals=False, enable_self_collisions=True)
            model = builder.finalize(device=cfg["device"])
            labels = [name.rsplit("/", 1)[-1] for name in model.joint_label]
            ids = [labels.index(name) for name in self.joints]
            q_indices = model.joint_q_start.numpy()[ids]
            d_indices = model.joint_qd_start.numpy()[ids]
            require(model.joint_dof_count == len(self.joints) + (6 if self.floating else 0), "base mobility mismatch")
            coordinates = model.joint_q.numpy()
            coordinates[q_indices] = q0
            key_id = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_KEY, self.key)
            require(key_id >= 0, "source keyframe missing")
            native_state = mujoco.MjData(native)
            mujoco.mj_resetDataKeyframe(native, native_state, key_id)
            for k, j in enumerate(native_jids):
                native_state.qpos[native.jnt_qposadr[j]] = q0[k]
            if self.floating:
                free_index = labels.index("floating_base_joint")
                start = int(model.joint_q_start.numpy()[free_index])
                # MJCF free quaternion wxyz -> Newton xyzw, explicit mapping.
                free = native_state.qpos[:7]
                coordinates[start:start + 7] = [*free[:3], *free[4:7], free[3]]
            model.joint_q.assign(coordinates)
            model.joint_qd.zero_()
            state, next_state = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            mujoco.mj_forward(native, native_state)
            body_names = [name.rsplit("/", 1)[-1] for name in model.body_label]
            native_body_ids = [mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
            require(all(i > 0 for i in native_body_ids), "body name mapping failed")
            audit = {
                "initial_fk_position_max_m": float(np.max(np.abs(state.body_q.numpy()[:, :3] - native_state.xpos[native_body_ids]))),
                "body_mass_max_kg": float(np.max(np.abs(model.body_mass.numpy() - native.body_mass[native_body_ids]))),
                "armature_max": float(np.max(np.abs(model.joint_armature.numpy()[d_indices] - native.dof_armature[native.jnt_dofadr[native_jids]]))),
                "joint_lower_max_rad": float(np.max(np.abs(model.joint_limit_lower.numpy()[d_indices] - native.jnt_range[native_jids, 0]))),
                "joint_upper_max_rad": float(np.max(np.abs(model.joint_limit_upper.numpy()[d_indices] - native.jnt_range[native_jids, 1]))),
            }
            require(all(v < 1e-5 for v in audit.values()), f"import audit failed: {audit}")
            cpu = cfg["device"] == "cpu"
            solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=cpu, integrator="implicitfast", solver="newton",
                                                iterations=50, ls_iterations=20, tolerance=1e-8, use_mujoco_contacts=True,
                                                update_data_interval=1, njmax=500, nconmax=200)
            control = model.control()
            require(control.mujoco.ctrl.shape == (len(self.joints),), "control dimension mismatch")
            control.mujoco.ctrl.assign(target.astype(np.float32))
            root_id = body_names.index("pelvis" if self.floating else "base")
            qs = [state.joint_q.numpy()[q_indices].copy()]
            poses = [state.body_q.numpy().copy()]
            zs = [float(state.body_q.numpy()[root_id, 2])]
            contacts = []
            for _ in range(round(cfg["duration_s"] / cfg["dt_s"])):
                state.clear_forces()
                solver.step(state, next_state, control, None, cfg["dt_s"])
                state, next_state = next_state, state
                qs.append(state.joint_q.numpy()[q_indices].copy())
                poses.append(state.body_q.numpy().copy())
                zs.append(float(state.body_q.numpy()[root_id, 2]))
                contacts.append(int(solver.mj_data.ncon) if cpu else int(solver.mjw_data.nacon.numpy()[0]))
            qs = np.asarray(qs)
            times = tuple(i * cfg["dt_s"] for i in range(len(qs)))
            observations = [Observable(self.scalars[0][0], "rad", (float(np.max(np.abs(qs[-1] - target))),)),
                            Observable(self.scalars[1][0], "m", (min(zs),)),
                            Observable(self.scalars[2][0], "1", (float(max(contacts)),)),
                            Observable(self.prefix + "base-height", "m", tuple(zs), times, "s")]
            observations += [Observable(self.prefix + name, "rad", tuple(float(v) for v in qs[:, i]), times, "s") for i, name in enumerate(self.joints)]
            manifest, records = self.verify_assets()
            from agenticsim.runtime.robot_artifacts import save_robot_playback
            playback = save_robot_playback(native, native_state, body_names, poses, times, qs, contacts, self)
            result = Evaluation(tuple(observations), "reference_simulation", {
                "engine": "Newton", "numerical_backend": "MuJoCo CPU" if cpu else "MuJoCo Warp CUDA",
                "newton": newton.__version__, "warp": wp.__version__, "mujoco": mujoco.__version__,
                "device": str(wp.get_device()), "device_name": wp.get_device().name,
                "execution": cfg, "floating_base": self.floating, "joint_dofs": len(self.joints),
                "generalized_velocity_dofs": model.joint_dof_count, "joint_order": self.joints,
                "asset_commit": manifest["commit"], "asset_files": records,
                "import_audit": audit, "collision_shapes": model.shape_count,
                "actuation": "source actuators through ctrl_direct, source gains and force limits",
                "input_source": task.document["context"]["source"], "physical_validation": "not_established",
                "robot_playback": playback,
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            result.validate(self.model)
            return result

"""Newton Featherstone robot rollout using the vendored DynamicsWM model builder."""
import hashlib
import importlib.util
from pathlib import Path
import platform
import sys

from agenticsim.domain import BackendDescriptor, EngineeringTask, Evaluation, ModelDefinition, Observable, number, require

ROOT = Path(__file__).resolve().parents[2]
MBS = ROOT / "components/dynamics/projects/mbs_isaac"
PREFIX = "urn:ase:robot:"


def load_mbs():
    """Load this repository's package under a private namespace, never a sibling."""
    name = "_ase_vendored_mbs"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MBS / "__init__.py", submodule_search_locations=[str(MBS)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name + ".models"), import_module(name + ".newton_backend")


class NewtonRobot:
    model = ModelDefinition("planar-arm3", "1", (
        ("initial_q", PREFIX + "initial-q", "rad"),
        ("target_q", PREFIX + "target-q", "rad")),
        ((PREFIX + "final-error", "rad"), (PREFIX + "peak-torque", "N*m")) +
        tuple((PREFIX + f"joint-{i}-q", "rad") for i in range(3)),
        ("Three uniform rods, length 0.5 m, mass 1 kg, radius 0.03 m each",
         "Fixed base; revolute Z axes in horizontal XY plane; gravity -9.81 Z",
         "No contacts, joint limits, friction, armature or built-in drives",
         "External saturated PD torque; target is a fixed joint configuration",
         "Parameterized robot fixture, not a Franka model or measured robot"))
    backend = BackendDescriptor("newton-featherstone", "1", ("evaluate",))

    def validate(self, task: EngineeringTask):
        task.validate(self.model, self.backend)
        for value in task.document["inputs"].values():
            require(len(value["values"]) == 3 and all(abs(v) <= 3.141592653589793 for v in value["values"]),
                    "three joint angles in [-pi, pi] required")
        config = task.document.get("execution")
        require(isinstance(config, dict) and set(config) == {"device", "dt_s", "duration_s", "algorithm"}, "explicit execution configuration required")
        require(config["device"] in ("cpu", "cuda:0"), "device must be cpu or cuda:0; no silent fallback")
        require(config["algorithm"] == "featherstone-semi-implicit", "unsupported numerical algorithm")
        require(number(config["dt_s"]) and 0.0001 <= config["dt_s"] <= 0.002, "dt outside supported interval")
        require(number(config["duration_s"]) and 0 < config["duration_s"] <= 2, "duration outside supported interval")
        steps = config["duration_s"] / config["dt_s"]
        require(1 <= steps <= 10000 and abs(steps - round(steps)) < 1e-8, "duration must be integral steps, maximum 10000")
        for c in task.document["candidates"]:
            require(set(c) == {"id", "parameters"} and isinstance(c["parameters"], dict)
                    and set(c["parameters"]) == {"kp", "kd", "torque_limit"}, "PD parameters required")
            for name, unit, lo, hi in (("kp", "N*m/rad", 0, 100), ("kd", "N*m*s/rad", 0, 10), ("torque_limit", "N*m", 0.01, 100)):
                value = c["parameters"][name]
                require(isinstance(value, dict) and set(value) == {"value", "unit"}
                        and value["unit"] == unit and number(value["value"]) and lo <= value["value"] <= hi,
                        f"invalid {name} value/unit")
        for criterion in [task.document["objective"], *task.document["constraints"]]:
            require(criterion["semantic_id"] in (PREFIX + "final-error", PREFIX + "peak-torque"), "criterion must be scalar")

    def evaluate(self, task: EngineeringTask, candidate: dict) -> Evaluation:
        self.validate(task)
        require(candidate in task.document["candidates"], "candidate not in task")
        import numpy as np
        import warp as wp
        import newton
        require(newton.__version__ == "1.2.1" and wp.__version__ == "1.13.0", "use pinned Newton environment")
        wp.config.kernel_cache_dir = str(ROOT / "work/cache/warp")
        models, translation = load_mbs()
        config = task.document["execution"]
        with wp.ScopedDevice(config["device"]):
            model = models.n_link_arm(n=3, damping=0., planar_axis="Z", name="planar-arm3")
            nm, info = translation.to_newton(model, device=config["device"])
            require(nm.joint_dof_count == 3 and info["eq_count"] == 0, "unexpected robot topology")
            solver = newton.solvers.SolverFeatherstone(nm, angular_damping=0.)
            state, next_state = nm.state(), nm.state()
            control = nm.control()
            q0 = np.asarray(task.document["inputs"]["initial_q"]["values"], dtype=np.float32)
            target = np.asarray(task.document["inputs"]["target_q"]["values"], dtype=np.float32)
            nm.joint_q.assign(q0)
            nm.joint_qd.zero_()
            newton.eval_fk(nm, nm.joint_q, nm.joint_qd, state)
            p = {k: v["value"] for k, v in candidate["parameters"].items()}
            steps = round(config["duration_s"] / config["dt_s"])
            positions = [state.joint_q.numpy().copy()]
            peak = 0.
            for _ in range(steps):
                q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                torque = np.clip(p["kp"] * (target - q) - p["kd"] * qd, -p["torque_limit"], p["torque_limit"])
                require(bool(np.isfinite(torque).all()), "nonfinite control")
                peak = max(peak, float(np.abs(torque).max()))
                state.clear_forces()
                control.joint_f.assign(torque.astype(np.float32))
                solver.step(state, next_state, control, None, config["dt_s"])
                state, next_state = next_state, state
                positions.append(state.joint_q.numpy().copy())
            qs = np.asarray(positions)
            times = tuple(i * config["dt_s"] for i in range(steps + 1))
            observations = [Observable(PREFIX + "final-error", "rad", (float(np.max(np.abs(qs[-1] - target))),)),
                            Observable(PREFIX + "peak-torque", "N*m", (peak,))]
            observations.extend(Observable(PREFIX + f"joint-{i}-q", "rad", tuple(float(v) for v in qs[:, i]), times, "s") for i in range(3))
            device = wp.get_device(config["device"])
            result = Evaluation(tuple(observations), "reference_simulation", {
                "newton": newton.__version__, "warp": wp.__version__, "numpy": np.__version__,
                "python": platform.python_version(), "device": str(device), "device_name": device.name,
                "execution": config, "mapping_version": "1", "frame": "right-handed, Z up; radians",
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [MBS / "spec.py", MBS / "models.py", MBS / "newton_backend.py", Path(__file__)]},
                "input_source": task.document["context"]["source"], "randomness": "none",
                "gradient": "not_exposed_or_verified"})
            result.validate(self.model)
            return result

# Franka torque-level trajectory collector -> MoSim-format .npz dataset.
#
# Physics: Isaac Sim (PhysX) headless @ 500 Hz (dt=0.002, matching MoSim's panda config).
# Control: 7 arm joints in pure effort mode (PD gains zeroed); the excitation policy is
#   a PD law tracking smooth random sum-of-sines joint references, computed in numpy so
#   the recorded action IS exactly the torque applied. Fingers held by position control
#   and excluded from state/action.
# Output (MoSim MoSim/src/tools.py:load_data convention):
#   data:     float32 (episodes, steps, nq+nv | nu | nq+nv)  = [state, action, next_state]
#   metadata: dict(nq=7, nv=7, dt=0.002, ...)
#
# Run:  env_isaacsim\Scripts\python.exe collect_trajectories.py --episodes 5 --steps 1000
import argparse
import os
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--steps", type=int, default=1000, help="steps per episode (500 Hz)")
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--out", type=str, default="data/franka_torque_dataset.npz")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot.manipulators.examples")  # deprecated ext, off by default in 6.0

from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka

rng = np.random.default_rng(args.seed)
NARM = 7

# No ground plane: fixed-base arm in free space. A ground collider caused PhysX
# blowups (1e3+ rad/s velocity spikes) when random reset poses penetrated it.
world = World(physics_dt=args.dt, rendering_dt=args.dt, stage_units_in_meters=1.0)
robot = world.scene.add(Franka(prim_path="/World/franka", name="franka"))
world.reset()

num_dof = robot.num_dof
props = robot.dof_properties
lower = props["lower"].astype(np.float64)
upper = props["upper"].astype(np.float64)
tau_max = props["maxEffort"].astype(np.float64)

controller = robot.get_articulation_controller()
# Arm joints: effort mode (zero internal PD). Fingers: keep stiff position hold.
kps = np.zeros(num_dof)
kds = np.zeros(num_dof)
kps[NARM:] = 1.0e5
kds[NARM:] = 1.0e3
controller.set_gains(kps=kps, kds=kds)
finger_targets = robot.get_joint_positions()[NARM:].copy()

# Excitation PD gains (numpy-side policy, not PhysX internal)
KP = np.array([50.0, 50.0, 50.0, 50.0, 20.0, 12.0, 8.0])
KD = 2.0 * np.sqrt(KP) * 0.5
TAU_FRAC = 0.6  # clip applied torque to this fraction of the joint effort limit

mid = 0.5 * (lower[:NARM] + upper[:NARM])
halfspan = 0.5 * (upper[:NARM] - lower[:NARM])


def sample_reference():
    """Per-episode smooth random reference: q_ref(t) = center + sum_k A_k sin(w_k t + phi_k)."""
    n_waves = 3
    center = mid + rng.uniform(-0.3, 0.3, NARM) * halfspan
    amp = rng.uniform(0.05, 0.25, (n_waves, NARM)) * halfspan / n_waves
    freq = rng.uniform(0.1, 0.6, (n_waves, NARM)) * 2.0 * np.pi
    phase = rng.uniform(0.0, 2.0 * np.pi, (n_waves, NARM))
    def q_ref(t):
        q = center + (amp * np.sin(freq * t + phase)).sum(axis=0)
        dq = (amp * freq * np.cos(freq * t + phase)).sum(axis=0)
        return np.clip(q, lower[:NARM] + 0.05, upper[:NARM] - 0.05), dq
    return q_ref


episodes = []
t_start = time.time()
for ep in range(args.episodes):
    # Random interior initial pose, zero velocity
    q0 = np.zeros(num_dof)
    q0[:NARM] = mid + rng.uniform(-0.4, 0.4, NARM) * halfspan
    q0[NARM:] = finger_targets
    robot.set_joint_positions(q0)
    robot.set_joint_velocities(np.zeros(num_dof))
    controller.apply_action(
        ArticulationAction(joint_positions=[None] * NARM + list(finger_targets))
    )

    q_ref = sample_reference()
    states = np.zeros((args.steps + 1, 2 * NARM), dtype=np.float32)
    actions = np.zeros((args.steps, NARM), dtype=np.float32)

    q = robot.get_joint_positions()[:NARM]
    dq = robot.get_joint_velocities()[:NARM]
    states[0] = np.concatenate([q, dq])

    for i in range(args.steps):
        t = i * args.dt
        qr, dqr = q_ref(t)
        tau = KP * (qr - q) + KD * (dqr - dq)
        tau = np.clip(tau, -TAU_FRAC * tau_max[:NARM], TAU_FRAC * tau_max[:NARM])

        efforts = np.zeros(num_dof)
        efforts[:NARM] = tau
        controller.apply_action(ArticulationAction(joint_efforts=efforts))
        world.step(render=False)

        q = robot.get_joint_positions()[:NARM]
        dq = robot.get_joint_velocities()[:NARM]
        actions[i] = tau
        states[i + 1] = np.concatenate([q, dq])

    assert np.all(np.isfinite(states)), f"episode {ep}: non-finite state"
    spike_steps = np.flatnonzero(np.abs(states[:, NARM:]).max(axis=1) > 50.0)
    if spike_steps.size:
        print(f"episode {ep}: WARNING velocity spike >50 rad/s at steps {spike_steps[:5]}...")
    ep_data = np.concatenate([states[:-1], actions, states[1:]], axis=-1)
    episodes.append(ep_data)
    print(
        f"episode {ep}: |q| range [{states[:, :NARM].min():.2f}, {states[:, :NARM].max():.2f}] rad, "
        f"|dq| max {np.abs(states[:, NARM:]).max():.2f} rad/s, "
        f"|tau| max {np.abs(actions).max():.1f} Nm"
    )

data = np.stack(episodes).astype(np.float32)  # (episodes, steps, 14+7+14)
metadata = {
    "nq": NARM,
    "nv": NARM,
    "nu": NARM,
    "dt": args.dt,
    "engine": "isaacsim-6.0.1.0/physx",
    "robot": "FrankaPanda (official cloud asset), fingers position-held, arm effort mode",
    "action": "applied joint torque [Nm], PD-excitation toward random sum-of-sines refs",
    "joint_names": list(robot.dof_names[:NARM]),
    "effort_limits": tau_max[:NARM].tolist(),
    "tau_clip_fraction": TAU_FRAC,
    "seed": args.seed,
}

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.savez_compressed(args.out, data=data, metadata=metadata)
wall = time.time() - t_start
sim_time = args.episodes * args.steps * args.dt
print(
    f"COLLECT DONE: {data.shape} -> {args.out} "
    f"({os.path.getsize(args.out) / 1e6:.1f} MB) | sim {sim_time:.0f}s in wall {wall:.0f}s "
    f"({sim_time / wall:.1f}x realtime)"
)

sim_app.close()

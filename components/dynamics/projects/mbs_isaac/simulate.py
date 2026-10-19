"""
Headless simulation + trajectory export for a built MBS model.

Drives the PhysX articulation at a fixed dt with zero-order-hold torques and
records (state, action, next_state) into the SAME .npz layout the Isaac/PhysX
Franka collector and MoSim use (see projects/mosim/MoSim/src/tools.py:load_data).
So any MBSModel becomes a drop-in world-model training source.

This module DOES start a SimulationApp (PhysX). It is headless (no renderer),
so it is light on GPU, but it is the one step to run when resources are free.
"""
from __future__ import annotations

import os

import numpy as np


def simulate(model, *, episodes=5, steps=1000, dt=0.002, out="data/mbs_dataset.npz",
             torque_scale=0.0, seed=0, q_init=None, headless=True):
    """Build `model` in a fresh PhysX stage and roll out `episodes` trajectories.

    torque_scale=0.0 -> passive (gravity only); >0 -> random smooth torque
    excitation scaled by each joint's max_effort (for world-model datasets).
    Returns the saved npz path.
    """
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.core.api")
    enable_extension("isaacsim.core.prims")

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction

    from .builder import build

    rng = np.random.default_rng(seed)
    ndof = model.dof

    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    root = build(model, world.stage, "/World")
    art = world.scene.add(SingleArticulation(prim_path=root, name=model.name))
    world.reset()

    # sanity: does PhysX agree with the spec on DoF and mass?
    print(f"[build check] spec DoF={ndof}, PhysX DoF={art.num_dof}; "
          f"spec mass={model.total_mass:.3f} kg")
    assert art.num_dof == ndof, "PhysX DoF != spec DoF (check joints/articulation root)"

    tau_max = np.array([j.max_effort for j in model.joints if j.joint_type != "fixed"])
    tau_max = np.clip(tau_max, 0, 500.0)  # cap the 1e6 'free' default for excitation

    episodes_data, energies = [], []
    for ep in range(episodes):
        q0 = np.zeros(ndof) if q_init is None else np.asarray(q_init, float).copy()
        if q_init is None and torque_scale == 0.0:
            q0[:] = rng.uniform(-0.3, 0.3, ndof)  # small release offset
        art.set_joint_positions(q0)
        art.set_joint_velocities(np.zeros(ndof))
        world.step(render=False)

        # per-episode smooth random torque reference (sum of sines)
        n_waves = 3
        amp = rng.uniform(0.1, 0.5, (n_waves, ndof)) * torque_scale
        freq = rng.uniform(0.2, 1.0, (n_waves, ndof)) * 2 * np.pi
        phase = rng.uniform(0, 2 * np.pi, (n_waves, ndof))

        states = np.zeros((steps + 1, 2 * ndof), np.float32)
        actions = np.zeros((steps, ndof), np.float32)
        q = art.get_joint_positions()[:ndof]
        dq = art.get_joint_velocities()[:ndof]
        states[0] = np.concatenate([q, dq])
        for i in range(steps):
            t = i * dt
            tau = (amp * np.sin(freq * t + phase)).sum(0) * tau_max if torque_scale else np.zeros(ndof)
            art.set_joint_efforts(tau.astype(np.float32))
            world.step(render=False)
            q = art.get_joint_positions()[:ndof]
            dq = art.get_joint_velocities()[:ndof]
            actions[i] = tau
            states[i + 1] = np.concatenate([q, dq])
        assert np.all(np.isfinite(states)), f"episode {ep}: non-finite state"
        episodes_data.append(np.concatenate([states[:-1], actions, states[1:]], -1))
        print(f"episode {ep}: |q|<={np.abs(states[:, :ndof]).max():.2f} rad, "
              f"|dq|<={np.abs(states[:, ndof:]).max():.2f} rad/s")

    data = np.stack(episodes_data).astype(np.float32)
    metadata = {
        "nq": ndof, "nv": ndof, "nu": ndof, "dt": dt,
        "engine": "isaacsim/physx (coded MBS)", "model": model.name,
        "total_mass": model.total_mass, "gravity": list(model.gravity),
        "joint_names": [j.name for j in model.joints if j.joint_type != "fixed"],
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, data=data, metadata=metadata)
    print(f"SAVED {data.shape} -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    app.close()
    return out

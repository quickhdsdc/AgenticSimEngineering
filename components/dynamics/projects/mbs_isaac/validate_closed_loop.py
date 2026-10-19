"""
BENCHMARK LAYER 3 — closed-loop mechanism (IFToMM-style four-bar).

A PhysX articulation is a reduced-coordinate TREE, so the loop must be closed by
an extra constraint solved in maximal coordinates. That constraint is enforced
iteratively and therefore DRIFTS. This script quantifies the two things that
matter for constrained (DAE-like) multibody systems:

  1. loop-closure residual — the rocker's ground pivot must stay pinned at D.
     Reference answer: exactly 0 for all time.
  2. mechanical energy — released from rest, undamped, undriven under gravity.
     Reference answer: constant (the IFToMM A03/A04 criterion).

Run (from ):
  projectsisaac_sim.venvScriptspython.exe -m mbs_isaac.validate_closed_loop
"""
from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

DT = 0.002  # overridable via argv[3]
STEPS = 2500  # 5 s (scaled with dt below)


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.core.api")
    enable_extension("isaacsim.core.prims")

    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation

    from .builder import build
    from .closed_loop import four_bar

    import sys

    global DT, STEPS
    pos_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    vel_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if len(sys.argv) > 3:  # keep the simulated span at 5 s when dt changes
        DT = float(sys.argv[3])
        STEPS = int(round(5.0 / DT))
    model = four_bar(pos_iters=pos_iters, vel_iters=vel_iters)
    print(f"solver: {pos_iters} position / {vel_iters} velocity iterations, "
          f"dt={DT}, steps={STEPS}")
    A, B, C, D = (model.assembly[k] for k in "ABCD")
    print(f"four-bar assembly: A={np.round(A,4)} B={np.round(B,4)} "
          f"C={np.round(C,4)} D={np.round(D,4)}")
    print(f"tree DoF={model.dof} (mechanism mobility is 1), links={len(model.links)}, "
          f"loop joints={sum(1 for j in model.joints if j.loop_closure)}")

    world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    root = build(model, world.stage, "/World")
    world.reset()

    art = SingleArticulation(prim_path=root, name="fourbar")
    art.initialize()
    print(f"[DoF check] PhysX={art.num_dof}, spec tree DoF={model.dof}")

    moving = ["crank", "coupler", "rocker"]
    views = {}
    for nm in moving:
        v = RigidPrim(prim_paths_expr=f"{root}/{nm}", name=f"v_{nm}")
        v.initialize()
        views[nm] = v

    spec_by_name = {l.name: l for l in model.links}
    g = abs(model.gravity[2])

    def quat_to_R(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def energy_and_drift():
        E = 0.0
        for nm in moving:
            v = views[nm]
            p, q = v.get_world_poses()
            p, q = np.asarray(p)[0], np.asarray(q)[0]
            lin, ang = v.get_velocities()[0][:3], v.get_velocities()[0][3:]
            spec = spec_by_name[nm]
            R = quat_to_R(q)
            com_w = p + R @ np.asarray(spec.com)
            I_body = np.diag(spec.resolved_inertia())
            I_w = R @ I_body @ R.T
            E += 0.5 * spec.mass * float(np.dot(lin, lin))
            E += 0.5 * float(ang @ I_w @ ang)
            E += spec.mass * g * float(com_w[2])
        # loop residual: the rocker's own origin is its ground pivot; it must stay at D
        pr = np.asarray(views["rocker"].get_world_poses()[0])[0]
        drift = float(np.linalg.norm(pr - np.asarray(D)))
        return E, drift

    # release from rest, no drive, no damping
    art.set_joint_positions(np.zeros(art.num_dof))
    art.set_joint_velocities(np.zeros(art.num_dof))
    c = art.get_articulation_controller()
    c.set_gains(kps=np.zeros(art.num_dof), kds=np.zeros(art.num_dof))
    world.step(render=False)

    E0, d0 = energy_and_drift()
    print(f"\ninitial: E={E0:.6f} J, loop residual={d0:.3e} m")

    Es, Ds = np.zeros(STEPS), np.zeros(STEPS)
    # P1b: also record the tree joint angles. Energy drift is only a proxy; what a
    # world model actually needs from its ground truth is the TRAJECTORY, so the
    # Newton comparison needs theta(t) from both engines on the same axis.
    Qs = np.zeros((STEPS, art.num_dof))
    for i in range(STEPS):
        art.set_joint_efforts(np.zeros(art.num_dof))
        world.step(render=False)
        Es[i], Ds[i] = energy_and_drift()
        Qs[i] = np.asarray(art.get_joint_positions()).reshape(-1)

    t = (np.arange(STEPS) + 1) * DT
    print("\n--- LAYER 3 RESULT: four-bar under gravity, undamped ---")
    for h in (0.5, 1.0, 2.0, 5.0):
        k = min(int(h / DT) - 1, STEPS - 1)
        print(f"  t={h:>4.1f}s: energy {Es[k]:10.5f} J (drift {Es[k]-E0:+.3e} J, "
              f"{abs(Es[k]-E0)/max(abs(E0),1e-9)*100:6.3f}%) | loop residual {Ds[k]:.3e} m")
    e_rel = np.abs(Es - E0).max() / max(abs(E0), 1e-9)
    print(f"\n  max |energy drift| = {np.abs(Es-E0).max():.3e} J ({e_rel*100:.3f}%)")
    print(f"  max loop residual  = {Ds.max():.3e} m")
    ok = e_rel < 0.02 and Ds.max() < 1e-3
    print(f"\n{'PASS' if ok else 'FAIL'}: PhysX holds the loop and conserves energy")

    from .paths import result_path
    # dt 扫描时不要覆盖 2 ms 的基线结果（P1b 的对照基准就是那一份）
    out_npz = result_path("four_bar_validation.npz" if abs(DT - 0.002) < 1e-12
                          else f"four_bar_validation_dt{round(DT*1e6)}us.npz")
    print(f"  crank sweep = {Qs[:,0].max()-Qs[:,0].min():.3f} rad "
          f"(mechanism is actually moving, not frozen)")
    np.savez(out_npz, t=t, energy=Es, drift=Ds, E0=E0, joint_q=Qs)
    print(f"saved -> {out_npz}")
    app.close()


if __name__ == "__main__":
    main()

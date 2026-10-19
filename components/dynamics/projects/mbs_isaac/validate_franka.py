"""
Benchmark 第 1 层 —— Franka 往返交叉验证（round-trip）。

要回答的问题：`mbs_isaac` 搭出来的模型，和 NVIDIA 官方发布的模型在物理上是不是同一个？

方法（同一个场景里，两条独立的构建路径）：
  A) 参考模型：官方 Franka USD 资产，原样加载
  B) 重建模型：从 A 读出运行时惯性参数（PhysX 实际使用的质量/质心/惯量）
               + 从 USD 读关节拓扑、锚点、关节帧朝向，
               然后走 mbs_isaac 的 spec → builder 路径重新搭一遍

这是最强的正确性检验：质量、惯量、质心、关节轴、锚点、关节帧任何一处填错，
都会立刻在对比中暴露。而且参数是程序化提取的，避免了手抄数字引入的
恰好是我们想检测的那类错误。

验证分层进行（失败时能立刻定位是哪一层错）：
  [0] 运动方程 M/C/G —— 判定模型是否等价的**核心判据**
  [1] 运动学 —— 锚点/轴/关节帧对不对
  [1b] 极短 rollout —— 整体一致性
  [2][3] 长一点的轨迹 —— 仅作诊断，因混沌放大不能作判据

运行方式（在  目录下）：
  projectsisaac_sim.venvScriptspython.exe -m mbs_isaac.validate_franka

详细教程见 `docs/guides/MBS建模教程_Isaac实现.md`
"""
from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

NARM = 7  # compare the 7 arm joints (fingers exist on both but stay locked)
STEPS = 500
DT = 0.002


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import numpy as np
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.core.api")
    enable_extension("isaacsim.core.prims")
    enable_extension("isaacsim.storage.native")

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd, UsdPhysics

    from .builder import build
    from .spec import JointSpec, LinkSpec, MBSModel, Shape

    world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    stage = world.stage

    # ---------------- A) reference: official asset ----------------
    usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=usd, prim_path="/World/franka_ref")
    world.reset()  # physics init -> PhysX computes inertials from collision meshes

    # runtime inertials (what PhysX really uses; USD attrs are unauthored/zero)
    view = Articulation(prim_paths_expr="/World/franka_ref", name="refview")
    view.initialize()
    names = list(view.body_names)
    # shapes: masses (M,K) | coms -> (pos (M,K,3), quat (M,K,4)) | inertias (M,K,9)
    masses = np.asarray(view.get_body_masses())[0]
    # get_body_coms() -> (position, ORIENTATION). That orientation is the body's
    # principal-inertia frame, and get_body_inertias() is expressed in it (hence
    # its zero off-diagonals). Dropping the orientation keeps the right principal
    # moments but mis-aims the inertia ellipsoid -> correct kinematics, wrong
    # rotational dynamics.
    com_pos = np.asarray(view.get_body_coms()[0])[0]
    com_quat = np.asarray(view.get_body_coms()[1])[0]
    inertias = np.asarray(view.get_body_inertias())[0]
    print(f"reference: {len(names)} bodies, total mass={masses.sum():.3f} kg")

    # joint topology / anchors / frame rotations straight from USD
    def leaf(rel):
        t = rel.GetTargets() if rel else []
        return os.path.basename(str(t[0])) if t else ""

    jspecs = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/franka_ref")):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            jt, axis = "revolute", str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            jt, axis = "prismatic", str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get())
        elif prim.IsA(UsdPhysics.FixedJoint):
            jt, axis = "fixed", "Z"
        else:
            continue
        j = UsdPhysics.Joint(prim)
        p0, p1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
        r0, r1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()

        def q(r):
            if r is None:
                return (1.0, 0.0, 0.0, 0.0)
            im = r.GetImaginary()
            return (float(r.GetReal()), float(im[0]), float(im[1]), float(im[2]))

        lo = hi = None
        if jt == "revolute":
            rj = UsdPhysics.RevoluteJoint(prim)
            lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
        jspecs.append(JointSpec(
            name=prim.GetName(), joint_type=jt,
            parent=leaf(j.GetBody0Rel()), child=leaf(j.GetBody1Rel()), axis=axis,
            parent_anchor=tuple(p0) if p0 else (0, 0, 0),
            child_anchor=tuple(p1) if p1 else (0, 0, 0),
            parent_rot=q(r0), child_rot=q(r1), max_effort=1000.0,
            lower=float(lo) if lo is not None else None,   # USD: degrees
            upper=float(hi) if hi is not None else None,
            # PhysX dynamic properties that are invisible to kinematics
            max_joint_velocity=mjv.Get() if (mjv := prim.GetAttribute(
                "physxJoint:maxJointVelocity")) and mjv.HasAuthoredValue() else None,
            armature=arm.Get() if (arm := prim.GetAttribute(
                "physxJoint:armature")) and arm.HasAuthoredValue() else 0.0,
        ))
    # Rebuild the WHOLE robot, hand and fingers included: the 0.586 kg gripper
    # hangs off link7 and materially changes the arm's dynamics under gravity.
    keep = set(names)
    jspecs.insert(0, JointSpec(name="ground", joint_type="fixed",
                              parent="", child="panda_link0"))

    # Reference link world poses -> use as the rebuilt model's initial assembly.
    # Deriving poses by accumulating joint anchors (the builder's default) ignores
    # the links' relative rotations, which for a real robot is simply wrong.
    from isaacsim.core.prims import RigidPrim

    pose_by_name = {}
    for nm in names:
        v = RigidPrim(prim_paths_expr=f"/World/franka_ref/{nm}", name=f"rp_{nm}")
        v.initialize()
        pp, qq = v.get_world_poses()
        pose_by_name[nm] = (np.asarray(pp)[0], np.asarray(qq)[0])

    def diagonalize(I9):
        """3x3 inertia tensor -> (principal moments, principal-axes quaternion).

        PhysX reports the full tensor in the body frame; USD wants it as diagonal
        moments plus the rotation to the principal frame. eigh gives exactly that.
        """
        I = np.asarray(I9, float).reshape(3, 3)
        I = 0.5 * (I + I.T)  # symmetrize against round-off
        vals, vecs = np.linalg.eigh(I)
        if np.linalg.det(vecs) < 0:  # keep it a proper rotation
            vecs[:, 0] *= -1.0
        m = vecs
        tr = m.trace()
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            if i == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        return tuple(float(v) for v in vals), (float(w), float(x), float(y), float(z))

    lspecs = []
    max_offdiag = 0.0
    for i, nm in enumerate(names):
        if nm not in keep:
            continue
        p, quat = pose_by_name[nm]
        Imat = np.asarray(inertias[i], float).reshape(3, 3)
        max_offdiag = max(max_offdiag, float(np.abs(Imat - np.diag(np.diag(Imat))).max()))
        diag, paxes_from_eig = diagonalize(inertias[i])
        # inertia already diagonal in the principal frame -> its orientation is the
        # COM orientation, not something to recover by eigendecomposition
        paxes = tuple(float(x) for x in com_quat[i]) if max_offdiag < 1e-12 else paxes_from_eig
        lspecs.append(LinkSpec(
            name=nm, mass=float(masses[i]),
            com=tuple(float(x) for x in com_pos[i][:3]),
            inertia_diag=diag, principal_axes=paxes,
            shape=Shape(kind="box", size=(0.04, 0.04, 0.04)), collision=False,
            init_pos=tuple(float(x) for x in p),
            init_rot=tuple(float(x) for x in quat),  # (w,x,y,z)
        ))
    print(f"max off-diagonal inertia term in reference: {max_offdiag:.3e} "
          f"(diagonal-only copy would lose this)")
    print(f"rebuilt spec: {len(lspecs)} links, {len(jspecs)} joints, "
          f"mass={sum(l.mass for l in lspecs):.3f} kg")

    # ---------------- B) rebuilt through mbs_isaac ----------------
    model = MBSModel(name="franka_mbs", links=lspecs, joints=jspecs).validate()
    root = build(model, stage, "/World")
    # move the rebuilt robot aside so the two never touch
    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(stage.GetPrimAtPath(root)).AddTranslateOp().Set(Gf.Vec3d(3.0, 0, 0))

    # both robots now exist on the stage -> re-initialize physics, THEN create the
    # handles (adding an articulation after a reset invalidates the physics view)
    world.reset()
    ref = SingleArticulation(prim_path="/World/franka_ref", name="ref")
    mbs = SingleArticulation(prim_path=root, name="mbs")
    ref.initialize()
    mbs.initialize()
    print(f"[DoF check] reference={ref.num_dof}  rebuilt={mbs.num_dof}  spec={model.dof}")

    # ---------------- identical excitation, compare ----------------
    rng = np.random.default_rng(0)
    amp = rng.uniform(2.0, 8.0, NARM)
    freq = rng.uniform(0.3, 1.2, NARM) * 2 * np.pi
    phase = rng.uniform(0, 2 * np.pi, NARM)

    q0 = np.zeros(max(ref.num_dof, mbs.num_dof))
    q0[:NARM] = [0.0, -0.4, 0.0, -1.8, 0.0, 1.6, 0.8]

    def reset_both():
        for art in (ref, mbs):
            art.set_joint_positions(q0[: art.num_dof])
            art.set_joint_velocities(np.zeros(art.num_dof))
            c = art.get_articulation_controller()
            c.set_gains(kps=np.zeros(art.num_dof), kds=np.zeros(art.num_dof))
        world.step(render=False)

    def rollout(torque_fn, n):
        qr, qm = np.zeros((n, NARM)), np.zeros((n, NARM))
        for i in range(n):
            tau = torque_fn(i * DT)
            for art in (ref, mbs):
                e = np.zeros(art.num_dof)
                e[:NARM] = tau
                art.set_joint_efforts(e)
            world.step(render=False)
            qr[i] = ref.get_joint_positions()[:NARM]
            qm[i] = mbs.get_joint_positions()[:NARM]
        return qr, qm

    print("\n--- LAYER 1: rebuilt vs official Franka ---")

    # ========================================================================
    # 【核心方法论】直接比较运动方程，而不是比轨迹。
    #
    # 多体动力学方程：  M(q)·q̈ + C(q,q̇) + G(q) = τ
    # "模型"就是这三个量。在同一个 q、同一个 q̇ 下比这三项，
    # 相对误差到 1e-4（float32 噪声底）就说明两个模型是同一个刚体系统 ——
    # **完全不需要跑仿真**。
    #
    # 为什么不能比轨迹：多体系统在重力下是混沌的。本例中两个数学上等价的模型，
    # 10 步（0.02s）差 3.1e-08 rad，但 0.2s 就差到 0.23 rad —— 放大了 7 个数量级。
    # 所以"轨迹不重合"根本不能说明模型错了。
    #
    # 这个方法对后续 world model 工作同样适用：评估学出来的神经网络模型时，
    # 比它隐含的 M/C/G 比比长时程 rollout 更能分辨"物理学对了没有"。
    # ========================================================================
    vr = Articulation(prim_paths_expr="/World/franka_ref", name="vr")
    vm = Articulation(prim_paths_expr=root, name="vm")
    vr.initialize()
    vm.initialize()
    for v in (vr, vm):
        v.set_joint_positions(np.array([q0[: ref.num_dof]]))
        v.set_joint_velocities(np.array([np.full(ref.num_dof, 0.3)]))
    world.step(render=False)
    Mr = np.asarray(vr.get_mass_matrices())[0]
    Mm = np.asarray(vm.get_mass_matrices())[0]
    Gr = np.asarray(vr.get_generalized_gravity_forces())[0]
    Gm = np.asarray(vm.get_generalized_gravity_forces())[0]
    Cr = np.asarray(vr.get_coriolis_and_centrifugal_forces())[0]
    Cm = np.asarray(vm.get_coriolis_and_centrifugal_forces())[0]
    def relerr(a, b):
        return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-12))

    eom_err = max(relerr(Mr, Mm), relerr(Gr, Gm), relerr(Cr, Cm))
    for lbl, a, b in (("M(q)  mass matrix", Mr, Mm),
                      ("G(q)  gravity    ", Gr, Gm),
                      ("C(q,qd) coriolis ", Cr, Cm)):
        print(f"  [0 EOM      ] {lbl}: abs {np.abs(a-b).max():.3e}, "
              f"rel {relerr(a,b):.2e}  (scale {np.abs(a).max():.3f})")
    print(f"                -> worst relative EOM error {eom_err:.2e} "
          f"(PhysX computes these in float32, so ~1e-4 is the noise floor)")

    # stage 1: kinematics — same joint angles must give the same end-effector pose
    reset_both()
    ee_r = np.asarray(RigidPrim(prim_paths_expr="/World/franka_ref/panda_link7",
                                name="eer").get_world_poses()[0])[0]
    ee_m = np.asarray(RigidPrim(prim_paths_expr=f"{root}/panda_link7",
                                name="eem").get_world_poses()[0])[0]
    ee_m_local = ee_m - np.array([3.0, 0.0, 0.0])  # undo the display offset
    k_err = float(np.abs(ee_r - ee_m_local).max())
    print(f"  [1 kinematics] link7 world pos err = {k_err:.3e} m "
          f"(ref {np.round(ee_r,4)}, mbs {np.round(ee_m_local,4)})")

    # stage 1b: very short rollout. If the EOM match, 10 steps must still track;
    # anything longer is dominated by how fast small differences amplify.
    reset_both()
    qr, qm = rollout(lambda t: np.zeros(NARM), 10)  # 0.02 s
    s_err = np.abs(qr - qm)[-1].max()
    print(f"  [1b 10 steps] max |dq| after 0.02 s = {s_err:.3e} rad")

    # stage 2: passive gravity, SHORT horizon. The arm must stay in free space:
    # over longer falls it slams into joint limits and (on the reference, which
    # keeps its collision meshes) into itself, so the two models legitimately
    # differ there. Free-space rigid-body dynamics is what the builder owns.
    reset_both()
    qr, qm = rollout(lambda t: np.zeros(NARM), 100)  # 0.2 s
    g_err = np.abs(qr - qm)[-1].max()
    print(f"  [2 gravity  ] max |dq| after 0.2 s (free space) = {g_err:.3e} rad")

    # stage 3: gentle torque excitation, still free space
    reset_both()
    qr, qm = rollout(lambda t: 1.0 * np.sin(freq * t + phase), 100)
    t_err = np.abs(qr - qm)[-1].max()
    print(f"  [3 torque   ] max |dq| after 0.2 s at +-1 Nm  = {t_err:.3e} rad")

    # diagnostic only: long, harsh rollout that does hit limits / self-collision
    reset_both()
    qr, qm = rollout(lambda t: amp * np.sin(freq * t + phase), STEPS)
    err = np.abs(qr - qm)
    print(f"  [diagnostic ] {STEPS*DT:.1f}s at +-{amp.max():.0f} Nm (hits limits): "
          f"max |dq| = {err[-1].max():.3e} rad")
    print(f"                per-joint: {np.round(err[-1], 4)}")

    # The model is judged by its EQUATIONS OF MOTION and kinematics, not by long
    # trajectories: the arm under gravity is chaotic, so float32-level differences
    # in M/G amplify far beyond their own magnitude within a few tenths of a second.
    ok = k_err < 1e-4 and eom_err < 5e-4
    print(f"\n{'PASS' if ok else 'FAIL'}: builder reproduces the official model's "
          f"free-space rigid-body dynamics")

    from .paths import result_path
    out_npz = result_path("franka_crossvalidation.npz")
    np.savez(out_npz, q_ref=qr, q_mbs=qm, dt=DT)
    print(f"saved -> {out_npz}")
    app.close()


if __name__ == "__main__":
    main()

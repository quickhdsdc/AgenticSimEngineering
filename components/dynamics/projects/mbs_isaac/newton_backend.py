"""
Newton 后端：把 spec.py 的 MBSModel 翻译成 newton.ModelBuilder。

【为什么要有这一层】
`builder.py` 把同一个 MBSModel 落到 USD/PhysX 上；这里把它落到 Newton 上。
两条路径读的是**同一份声明**（同样的质量、惯量、质心、关节帧、装配构型），
所以两个引擎的差异只可能来自求解器本身，不可能来自建模差异。
这正是 P1b 求解器对比实验成立的前提。

【与 PhysX 路径的语义对应】（逐项等价，不是近似）
    USD LocalPos0 / LocalRot0   ←→   Newton parent_xform
    USD LocalPos1 / LocalRot1   ←→   Newton child_xform
    USD physics:axis (关节帧内)  ←→   Newton axis (parent anchor 帧内)
    MassAPI diagonalInertia      ←→   add_link(inertia=R diag R^T)（均为绕质心）

【两个必须注意的坑】
1. **四元数分量顺序**：spec / USD 用 (w,x,y,z)，Warp 用 (x,y,z,w)。
   写反了不会报错，只会得到一个被扭转过的机构。
2. **闭环关节**：Newton 的减缩坐标求解器（Featherstone）和 PhysX 一样只吃树，
   闭环必须走 equality constraint（MuJoCo 后端才支持）。这里把 loop_closure
   关节翻译成 `EqType.CONNECT`（点约束）。对平面四杆而言，点约束与转动铰
   在平面内是等价的：都只留 1 个自由度。

【关于"焊到世界上的连杆"】
spec 里常有一个 mass 不为零但用 fixed 关节焊死在 world 上的 `ground` 件。
它没有自由度，物理上就是惯性系的一部分，所以这里把它**折叠进 world**
（world 索引 = -1），并把它的世界位姿复合进以它为父件的关节的 parent_xform。
这是恒等变换，不改变任何动力学。
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .spec import MBSModel

_AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _wq(wxyz) -> wp.quat:
    """(w,x,y,z) -> wp.quat(x,y,z,w)。全项目最容易写反的一行。"""
    w, x, y, z = wxyz
    return wp.quat(float(x), float(y), float(z), float(w))


def _xf(pos, rot_wxyz=(1.0, 0.0, 0.0, 0.0)) -> wp.transform:
    return wp.transform(wp.vec3(*[float(v) for v in pos]), _wq(rot_wxyz))


def _quat_to_R(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def to_newton(model: MBSModel, close_loops: bool = True, device=None,
              eq_solref=None, eq_solimp=None):
    """MBSModel -> (newton.Model, info)。

    Args:
        model: 声明层模型（与喂给 PhysX builder 的是同一个对象）。
        close_loops: False 时丢掉闭环约束，只留树。用于把"闭环约束的耗散"
            和"积分器本身的耗散"分开测——这是本实验的对照组。
        device: warp device；None 用当前默认。
        eq_solref: MuJoCo equality 约束的 (timeconst, dampratio)。默认 (0.02, 1)
            意味着约束是**软**的：时间常数 0.02 s 比 dt=2 ms 大一个数量级，
            所以残差不会像 PhysX 的硬约束那样落到微米。要与 PhysX 同台比较
            必须把它收紧到 ~2·dt。
        eq_solimp: MuJoCo 约束阻抗曲线 (d0, dwidth, width, midpoint, power)。

    Returns:
        (newton_model, info)，info 含 body 名→索引映射、树自由度、约束数。
    """
    import newton

    model.validate()

    links = {l.name: l for l in model.links}
    tree_joints = [j for j in model.joints if not getattr(j, "loop_closure", False)]
    loop_joints = [j for j in model.joints if getattr(j, "loop_closure", False)]

    # ---- 1) 找出"焊死在 world 上"的连杆，算出它们的世界位姿 ----
    # 判据：只被 fixed 关节连接，且顺着 fixed 链一直能追到 world。
    static_xf: dict[str, wp.transform] = {}
    changed = True
    while changed:
        changed = False
        for j in tree_joints:
            if j.joint_type != "fixed" or j.child in static_xf:
                continue
            if j.parent == "":
                base = wp.transform_identity()
            elif j.parent in static_xf:
                base = static_xf[j.parent]
            else:
                continue
            # 关节要求 父帧 == 子帧，于是 世界_子体 = 世界_父体 ∘ X_p ∘ inv(X_c)
            Xp = _xf(j.parent_anchor, j.parent_rot)
            Xc = _xf(j.child_anchor, j.child_rot)
            static_xf[j.child] = wp.transform_multiply(
                wp.transform_multiply(base, Xp), wp.transform_inverse(Xc))
            changed = True

    gravity_z = float(model.gravity[2]) if hasattr(model, "gravity") else -9.81
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity_z)
    if eq_solref is not None or eq_solimp is not None:
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # ---- 2) 建刚体 ----
    body_idx: dict[str, int] = {}
    for l in model.links:
        if l.name in static_xf:
            continue  # 折叠进 world
        I = np.diag(np.asarray(l.resolved_inertia(), dtype=float))
        R = _quat_to_R(l.principal_axes)
        I_body = R @ I @ R.T  # 绕质心、在连杆局部系下表达
        xform = _xf(l.init_pos or (0.0, 0.0, 0.0),
                    l.init_rot or (1.0, 0.0, 0.0, 0.0))
        body_idx[l.name] = builder.add_link(
            xform=xform, mass=float(l.mass), com=wp.vec3(*[float(c) for c in l.com]),
            inertia=wp.mat33(*I_body.flatten().tolist()), label=l.name,
        )

    def _parent_of(j):
        """返回 (newton 父体索引, 复合后的 parent_xform)。"""
        Xp = _xf(j.parent_anchor, j.parent_rot)
        if j.parent == "":
            return -1, Xp
        if j.parent in static_xf:
            return -1, wp.transform_multiply(static_xf[j.parent], Xp)
        return body_idx[j.parent], Xp

    # ---- 3) 建关节（树）----
    joint_ids, joint_name_to_id = [], {}
    for j in tree_joints:
        if j.child in static_xf:
            continue  # 这条 fixed 关节已经被折叠掉了
        p, Xp = _parent_of(j)
        Xc = _xf(j.child_anchor, j.child_rot)
        kw = dict(parent=p, child=body_idx[j.child], parent_xform=Xp,
                  child_xform=Xc, label=j.name)
        if j.joint_type == "fixed":
            jid = builder.add_joint_fixed(**kw)
        elif j.joint_type == "prismatic":
            jid = builder.add_joint_prismatic(axis=wp.vec3(*_AXIS_VEC[j.axis]), **kw)
        else:
            jid = builder.add_joint_revolute(axis=wp.vec3(*_AXIS_VEC[j.axis]), **kw)
        joint_ids.append(jid)
        joint_name_to_id[j.name] = jid
    builder.add_articulation(joint_ids, label=model.name)

    # ---- 4) 闭环 -> equality constraint ----
    n_eq = 0
    if close_loops:
        for j in loop_joints:
            b1 = body_idx[j.child]
            b2 = -1 if (j.parent == "" or j.parent in static_xf) else body_idx[j.parent]
            if b2 != -1:
                raise NotImplementedError(
                    "loop closure to a moving body needs the anchor expressed on body1; "
                    "only ground-anchored loops are handled so far")
            # anchor 在 body1(=子件) 的局部系下；MuJoCo 编译器再由初始装配构型
            # 反算出它在 body2 上的对应点。这要求 joint_q=0 就是装配构型——
            # 我们的关节帧正是这样定的（见 closed_loop.py 的长注释）。
            ca = {}
            if eq_solref is not None:
                ca["mujoco:eq_solref"] = wp.vec2(*[float(v) for v in eq_solref])
            if eq_solimp is not None:
                ca["mujoco:eq_solimp"] = tuple(float(v) for v in eq_solimp)
            builder.add_equality_constraint_connect(
                body1=b1, body2=b2,
                anchor=wp.vec3(*[float(v) for v in j.child_anchor]),
                label=j.name, custom_attributes=ca or None)
            n_eq += 1

    nm = builder.finalize(device=device)
    info = {
        "body_idx": body_idx,
        "static_links": sorted(static_xf.keys()),
        "tree_dof": nm.joint_dof_count,
        "eq_count": n_eq,
        "joint_ids": joint_name_to_id,
    }
    return nm, info


# ---------------------------------------------------------------- energy ----
def energy_and_residual(nm, state, spec_model: MBSModel, body_idx: dict,
                        anchor_world=None, anchor_body: str | None = None):
    """从 body_q / body_qd 直接算机械能，与引擎无关。

    Newton 约定（state.py）：body_qd 前三位是**质心**线速度（世界系），
    后三位是角速度（世界系）。这与 PhysX 侧脚本用的量一致，两边可比。
    """
    q = state.body_q.numpy()      # (nb, 7) = (px,py,pz, qx,qy,qz,qw)
    qd = state.body_qd.numpy()    # (nb, 6) = (vx,vy,vz, wx,wy,wz)
    g = abs(float(spec_model.gravity[2]))
    specs = {l.name: l for l in spec_model.links}

    E = 0.0
    for name, bi in body_idx.items():
        s = specs[name]
        p = q[bi, :3]
        x, y, z, w = q[bi, 3:7]
        R = _quat_to_R((w, x, y, z))
        v, om = qd[bi, :3], qd[bi, 3:6]
        I = np.diag(np.asarray(s.resolved_inertia(), dtype=float))
        Rp = _quat_to_R(s.principal_axes)
        I_w = R @ (Rp @ I @ Rp.T) @ R.T
        com_w = p + R @ np.asarray(s.com, dtype=float)
        E += 0.5 * s.mass * float(v @ v)
        E += 0.5 * float(om @ I_w @ om)
        E += s.mass * g * float(com_w[2])

    res = float("nan")
    if anchor_world is not None and anchor_body is not None:
        res = float(np.linalg.norm(q[body_idx[anchor_body], :3]
                                   - np.asarray(anchor_world, dtype=float)))
    return E, res

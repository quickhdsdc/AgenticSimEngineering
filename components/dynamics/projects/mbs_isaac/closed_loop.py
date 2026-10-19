"""
Closed-loop mechanism specs (IFToMM-style benchmarks) — BENCHMARK LAYER 3.

Why these are the interesting case: a PhysX articulation is a reduced-coordinate
TREE. A four-bar (or Bricard) linkage is a LOOP, so it cannot be represented by
the tree alone — the loop must be closed by an extra constraint layered on top.
That constraint is enforced numerically, so it drifts. Measuring that drift, and
the energy behaviour alongside it, is exactly how one judges whether an engine
handles DAE-type constrained multibody systems well.

Both mechanisms below are released under gravity with no damping and no drive, so
the reference answers are known analytically:
  * total mechanical energy must stay constant
  * the loop-closure residual must stay zero
"""
from __future__ import annotations

import math

from .spec import JointSpec, LinkSpec, MBSModel, Shape


def _rod(name, length, mass, radius=0.02, pos=None, ang=None):
    """Uniform rod whose local frame sits at its proximal joint, along +X.

    `ang` is the rod's direction in the XZ plane measured from +X; the stored
    quaternion is a rotation about +Y by -ang, because rotating about +Y carries
    +X toward -Z.
    """
    q = None
    if ang is not None:
        h = -0.5 * ang
        q = (math.cos(h), 0.0, math.sin(h), 0.0)
    return LinkSpec(
        name=name, mass=mass,
        com=(length / 2, 0.0, 0.0),
        # thin uniform rod about its COM: axial ~0, transverse m L^2/12
        inertia_diag=(0.5 * mass * radius**2,
                      mass * length**2 / 12.0,
                      mass * length**2 / 12.0),
        shape=Shape(kind="cylinder", radius=radius, length=length, axis="X"),
        shape_offset=(length / 2, 0.0, 0.0),
        # No colliders: a linkage's bars overlap by construction at the joints and
        # sweep through each other. Contact here is a modelling artefact — it shows
        # up as a one-off energy loss (an inelastic impact) in an otherwise
        # perfectly conservative run.
        collision=False,
        init_pos=pos, init_rot=q,
    )


def four_bar(a=0.3, b=1.0, c=0.8, d=1.0, theta0=math.pi / 2, mass_per_m=1.0,
             pos_iters=32, vel_iters=4) -> MBSModel:
    """Planar four-bar linkage (crank a, coupler b, rocker c, ground d).

    Grashof crank-rocker by default (a+b <= c+d), so the crank fully rotates.
    Topology: ground -> crank -> coupler -> rocker as the tree, plus a
    loop-closure revolute pinning the rocker back to the ground at D.
    Mechanism mobility is 1, while the underlying tree has 3 DoF.
    """
    # ---- 求解装配构型（闭环位置分析）----
    # 闭环机构不能像开链那样"顺着链条摆下去"，必须解闭环方程。
    # 已知：A、D 是两个机架铰点；给定曲柄角 theta0 得到 B；
    # 求 C —— 它必须同时满足 |C-B| = 连杆长 b 且 |C-D| = 摇杆长 c，
    # 即两个圆的交点。
    A = (0.0, 0.0, 0.0)                                    # 机架铰点 1（曲柄轴）
    D = (d, 0.0, 0.0)                                      # 机架铰点 2（摇杆轴）
    B = (a * math.cos(theta0), 0.0, a * math.sin(theta0))  # 曲柄末端

    # 圆-圆求交（在 XZ 平面内；本机构绕 Y 轴运动）
    dx, dz = D[0] - B[0], D[2] - B[2]
    dist = math.hypot(dx, dz)
    # 可装配性检查：两圆必须相交（这也是 Grashof 条件在具体构型下的体现）
    if dist > b + c or dist < abs(b - c):
        raise ValueError(f"four-bar cannot be assembled at theta0={theta0}: dist={dist:.3f}")
    t = (b * b - c * c + dist * dist) / (2 * dist)   # B 到"根轴线"的投影距离
    h = math.sqrt(max(b * b - t * t, 0.0))          # 交点到根轴线的垂距
    mx, mz = B[0] + t * dx / dist, B[2] + t * dz / dist   # 垂足
    # 两个交点对应机构的两种装配分支（open/crossed），这里取其中一支
    C = (mx - h * dz / dist, 0.0, mz + h * dx / dist)

    ang_crank = theta0
    ang_coupler = math.atan2(C[2] - B[2], C[0] - B[0])
    ang_rocker = math.atan2(C[2] - D[2], C[0] - D[0])

    links = [
        LinkSpec(name="ground", mass=1.0, shape=Shape(kind="box", size=(0.04, 0.04, 0.04)),
                 collision=False, init_pos=A),
        _rod("crank", a, mass_per_m * a, pos=A, ang=ang_crank),
        _rod("coupler", b, mass_per_m * b, pos=B, ang=ang_coupler),
        _rod("rocker", c, mass_per_m * c, pos=D, ang=ang_rocker),
    ]
    # ============================================================================
    # 【全项目最关键的一点】装配信息存在**关节帧**里，不在连杆的 transform 里。
    #
    # PhysX 沿关节树推算每个连杆的位姿，并**覆盖**你写在连杆 prim 上的 transform。
    # 所以关节帧的旋转必须编码各杆件在装配构型下的相对朝向。
    #
    # 留成单位四元数会怎样（我实测踩过）：整个机构被摊成一条直线，
    # 曲柄 0.3 + 连杆 1.0 = 1.3，减去摇杆锚点 0.8 → 摇杆原点落在 x=0.5，
    # 而机架点 D 在 x=1.0 —— 仿真还没开始，闭环残差就是 0.5 m，且**不报任何错**。
    # 修正后初始残差降到 4.5e-05 m。
    #
    # 自查技巧：在 step 之前就测一次几何量（闭环残差 / 末端位姿）。
    # t=0 时刻就不对，说明模型压根没按你以为的方式装配。
    #
    # 数学：指向角为 t 的杆件，其世界旋转是 Roty(-t)（绕 +Y 转会把 +X 带向 -Z），
    # 所以把它的局部系转回"世界对齐"的旋转是 Roty(+t)。
    # ============================================================================
    def _qy(t):
        """绕 Y 轴旋转 t 的四元数 (w, x, y, z)"""
        return (math.cos(t / 2), 0.0, math.sin(t / 2), 0.0)

    ident = (1.0, 0.0, 0.0, 0.0)
    joints = [
        JointSpec(name="j_ground", joint_type="fixed", parent="", child="ground"),
        # tree: ground -A- crank -B- coupler -C- rocker
        JointSpec(name="jA", joint_type="revolute", parent="ground", child="crank",
                  axis="Y", parent_anchor=(0, 0, 0), child_anchor=(0, 0, 0),
                  parent_rot=ident, child_rot=_qy(ang_crank)),
        JointSpec(name="jB", joint_type="revolute", parent="crank", child="coupler",
                  axis="Y", parent_anchor=(a, 0, 0), child_anchor=(0, 0, 0),
                  parent_rot=_qy(ang_crank), child_rot=_qy(ang_coupler)),
        JointSpec(name="jC", joint_type="revolute", parent="coupler", child="rocker",
                  axis="Y", parent_anchor=(b, 0, 0), child_anchor=(c, 0, 0),
                  parent_rot=_qy(ang_coupler), child_rot=_qy(ang_rocker)),
        # 闭环关节：把摇杆的近端钉回机架上的 D 点。
        # loop_closure=True 会让 builder 给它设 excludeFromArticulation，
        # 否则 PhysX 静默忽略这个约束、机构散架（实测残差涨到 2.9 m）。
        JointSpec(name="jD_loop", joint_type="revolute", parent="ground", child="rocker",
                  axis="Y", parent_anchor=(d, 0, 0), child_anchor=(0, 0, 0),
                  parent_rot=ident, child_rot=_qy(ang_rocker), loop_closure=True),
    ]
    m = MBSModel(name="four_bar", links=links, joints=joints,
                 solver_position_iterations=pos_iters,
                 solver_velocity_iterations=vel_iters).validate()
    m.assembly = {"A": A, "B": B, "C": C, "D": D}  # for the drift metric
    return m


CLOSED_LOOP_CATALOG = {"four_bar": four_bar}

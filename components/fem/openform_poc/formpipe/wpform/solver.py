# -*- coding: utf-8 -*-
"""wpform — 用 NVIDIA Warp 写的可微板料成形显式求解器。

不是调用现成求解器,是把 FEM 本身写成 Warp kernel:
  * 单元    CST 三角膜元(更新拉格朗日) + 无旋转自由度的铰链弯曲
  * 本构    Hencky(对数)应变空间的 J2 平面应力弹塑性,硬化查流动曲线
  * 料厚    塑性不可压缩 t = t0/det(F) —— 与 AutoForm GEOM_THINNING 同定义
  * 接触    wp.Mesh 符号距离查询 + 罚函数法向力 + 库仑摩擦
  * 时间    显式辛欧拉 + 质量缩放 + 数值阻尼

为什么用 Hencky 空间:主应变下 J2 返回映射退化成对主对数应变的简单缩放,
数值稳定、无需迭代求塑性乘子,且全程只用可微算子(便于后续 wp.Tape 求梯度)。

单位: mm / ms / g  (应力 MPa, 密度 g/mm^3) —— 与 OpenRadioss 后端一致。
"""
from __future__ import annotations

import numpy as np
import warp as wp


# ---------------------------------------------------------------- 工具函数
@wp.func
def sym2x2_eig(a: float, b: float, c: float):
    """对称 2x2 [[a,b],[b,c]] 的特征分解,返回 (l1, l2, cos, sin)。

    闭式解,处处可微(退化时 b=0 分支避免 0/0)。
    """
    tr = a + c
    d = 0.5 * (a - c)
    r = wp.sqrt(d * d + b * b)
    l1 = 0.5 * tr + r
    l2 = 0.5 * tr - r
    # 特征向量:(b, l1-a) 方向;b→0 时退化为坐标轴
    vx = b
    vy = l1 - a
    n = wp.sqrt(vx * vx + vy * vy)
    if n < 1.0e-12:
        return l1, l2, 1.0, 0.0
    return l1, l2, vx / n, vy / n


@wp.func
def flow_stress(eps: float, curve_e: wp.array(dtype=float), curve_s: wp.array(dtype=float),
                n: int) -> float:
    """分段线性流动曲线查表;超出末端按最后一段斜率线性外插。"""
    if eps <= curve_e[0]:
        return curve_s[0]
    for i in range(1, n):
        if eps <= curve_e[i]:
            t = (eps - curve_e[i - 1]) / (curve_e[i] - curve_e[i - 1])
            return curve_s[i - 1] + t * (curve_s[i] - curve_s[i - 1])
    slope = (curve_s[n - 1] - curve_s[n - 2]) / (curve_e[n - 1] - curve_e[n - 2])
    return curve_s[n - 1] + slope * (eps - curve_e[n - 1])


# ---------------------------------------------------------------- 膜元
@wp.kernel
def membrane_forces(
    x: wp.array(dtype=wp.vec3),
    tris: wp.array2d(dtype=int),
    dm_inv: wp.array(dtype=wp.mat22),
    area0: wp.array(dtype=float),
    fp_inv: wp.array(dtype=wp.mat22),      # 塑性变形梯度的逆(状态量,原地更新)
    eqps: wp.array(dtype=float),           # 累积等效塑性应变(状态量)
    thickness: wp.array(dtype=float),      # 当前料厚(输出)
    curve_e: wp.array(dtype=float),
    curve_s: wp.array(dtype=float),
    n_curve: int,
    mu: float, lam_ps: float,              # 平面应力 Lamé 常数
    t0: float,
    f: wp.array(dtype=wp.vec3),
):
    e = wp.tid()
    i0 = tris[e, 0]
    i1 = tris[e, 1]
    i2 = tris[e, 2]
    p0 = x[i0]
    p1 = x[i1]
    p2 = x[i2]

    # 当前构型的局部标架(随单元转动 -> 自动共旋)
    u = p1 - p0
    v = p2 - p0
    nrm = wp.cross(u, v)
    ln = wp.length(nrm)
    if ln < 1.0e-16:
        return
    ex = wp.normalize(u)
    ez = nrm / ln
    ey = wp.cross(ez, ex)

    ds = wp.mat22(wp.dot(u, ex), wp.dot(v, ex),
                  wp.dot(u, ey), wp.dot(v, ey))
    F = ds * dm_inv[e]

    # 弹性试探: Fe = F * Fp^-1
    Fe = F * fp_inv[e]
    # C = Fe^T Fe 的特征分解给出右拉伸主值与主方向
    c00 = Fe[0, 0] * Fe[0, 0] + Fe[1, 0] * Fe[1, 0]
    c01 = Fe[0, 0] * Fe[0, 1] + Fe[1, 0] * Fe[1, 1]
    c11 = Fe[0, 1] * Fe[0, 1] + Fe[1, 1] * Fe[1, 1]
    l1, l2, cs, sn = sym2x2_eig(c00, c01, c11)
    l1 = wp.max(l1, 1.0e-12)
    l2 = wp.max(l2, 1.0e-12)
    e1 = 0.5 * wp.log(l1)          # Hencky 主应变
    e2 = 0.5 * wp.log(l2)

    # 平面应力弹性试探应力(主轴上,Kirchhoff)
    a11 = 2.0 * mu + lam_ps          # 弹性矩阵 [[a11, lam_ps], [lam_ps, a11]]
    s1 = a11 * e1 + lam_ps * e2
    s2 = lam_ps * e1 + a11 * e2

    # 平面应力 von Mises: seq^2 = s1^2 - s1 s2 + s2^2
    seq = wp.sqrt(wp.max(s1 * s1 - s1 * s2 + s2 * s2, 1.0e-20))
    ep0 = eqps[e]
    sy = flow_stress(ep0, curve_e, curve_s, n_curve)

    if seq > sy:
        # 平面应力径向返回。流动方向冻结在试探态(各向同性 von Mises 的标准做法):
        #   n_i = dSeq/dS_i,  a_i(dg) = a_i^tr - dg * n_i  (线性)
        # 一致性条件 seq(dg) = sy(ep0 + dg) 是标量方程,Newton 迭代求解。
        # 注意:平面应力下 seq 含静水分量,不能像 3D 那样把偏应变按 sy/seq 直接缩放
        # (第一版就是这么写的,单向拉伸下塑性应变差了 5 倍)。
        n1 = (2.0 * s1 - s2) / (2.0 * seq)
        n2 = (2.0 * s2 - s1) / (2.0 * seq)
        g1 = a11 * n1 + lam_ps * n2       # d(s1)/d(dg) = -g1
        g2 = lam_ps * n1 + a11 * n2
        dg = float(0.0)
        for _ in range(12):
            t1 = s1 - dg * g1
            t2 = s2 - dg * g2
            sq = wp.sqrt(wp.max(t1 * t1 - t1 * t2 + t2 * t2, 1.0e-20))
            syd = flow_stress(ep0 + dg, curve_e, curve_s, n_curve)
            r = sq - syd
            # dsq/ddg
            dsq = (-(2.0 * t1 - t2) * g1 - (2.0 * t2 - t1) * g2) / (2.0 * sq)
            # 硬化模量(差分,曲线是分段线性)
            hh = (flow_stress(ep0 + dg + 1.0e-4, curve_e, curve_s, n_curve) - syd) / 1.0e-4
            den = dsq - hh
            if wp.abs(den) < 1.0e-12:
                break
            dg = dg - r / den
            dg = wp.max(dg, 0.0)
        eqps[e] = ep0 + dg

        # 弹性主应变回退 + 应力更新
        e1n = e1 - dg * n1
        e2n = e2 - dg * n2
        s1 = a11 * e1n + lam_ps * e2n
        s2 = lam_ps * e1n + a11 * e2n

        # 塑性状态更新: Fe_new = Fe * M,  M = V diag(exp(e_new - e_tr)) V^T
        # 因 Fp^-1 = F^-1 Fe,故 Fp^-1_new = Fp^-1 * M(无需求 F 的逆)
        k1 = wp.exp(e1n - e1)
        k2 = wp.exp(e2n - e2)
        m00 = k1 * cs * cs + k2 * sn * sn
        m01 = (k1 - k2) * cs * sn
        m11 = k1 * sn * sn + k2 * cs * cs
        M = wp.mat22(m00, m01, m01, m11)
        fp_inv[e] = fp_inv[e] * M
        Fe = Fe * M
        l1 = wp.exp(2.0 * e1n)
        l2 = wp.exp(2.0 * e2n)

    # 料厚:塑性不可压缩 -> t = t0 / det(F)(与 AutoForm GEOM_THINNING 同定义)
    det_F = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    if det_F > 1.0e-6:
        thickness[e] = t0 / det_F

    # P = dpsi/dF = [Fe * (V diag(tau_i / l_i) V^T)] * Fp^-T
    b1 = s1 / l1
    b2 = s2 / l2
    q00 = b1 * cs * cs + b2 * sn * sn
    q01 = (b1 - b2) * cs * sn
    q11 = b1 * sn * sn + b2 * cs * cs
    B = wp.mat22(q00, q01, q01, q11)
    fpi = fp_inv[e]
    fpiT = wp.mat22(fpi[0, 0], fpi[1, 0], fpi[0, 1], fpi[1, 1])
    P = (Fe * B) * fpiT

    # f = -A0 * t0 * P * Dm^-T
    dmi = dm_inv[e]
    h = -area0[e] * t0
    g00 = h * (P[0, 0] * dmi[0, 0] + P[0, 1] * dmi[0, 1])
    g10 = h * (P[1, 0] * dmi[0, 0] + P[1, 1] * dmi[0, 1])
    g01 = h * (P[0, 0] * dmi[1, 0] + P[0, 1] * dmi[1, 1])
    g11 = h * (P[1, 0] * dmi[1, 0] + P[1, 1] * dmi[1, 1])

    f1 = ex * g00 + ey * g10
    f2 = ex * g01 + ey * g11
    wp.atomic_add(f, i1, f1)
    wp.atomic_add(f, i2, f2)
    wp.atomic_add(f, i0, -(f1 + f2))


# ---------------------------------------------------------------- 弯曲(铰链)
@wp.kernel
def bending_forces(
    x: wp.array(dtype=wp.vec3),
    hinges: wp.array2d(dtype=int),     # [n_hinge, 4] = (共享边 a,b, 对角点 c,d)
    rest_angle: wp.array(dtype=float),
    k_bend: float,
    f: wp.array(dtype=wp.vec3),
):
    h = wp.tid()
    ia = hinges[h, 0]
    ib = hinges[h, 1]
    ic = hinges[h, 2]
    idd = hinges[h, 3]
    xa = x[ia]
    xb = x[ib]
    xc = x[ic]
    xd = x[idd]
    e = xb - xa
    le = wp.length(e)
    if le < 1.0e-12:
        return
    n1 = wp.cross(xc - xa, xb - xa)
    n2 = wp.cross(xb - xa, xd - xa)
    l1 = wp.length(n1)
    l2 = wp.length(n2)
    if l1 < 1.0e-16 or l2 < 1.0e-16:
        return
    # 简化铰链模型(Bridson):力沿两侧法向,大小正比于角度偏差
    ang = wp.atan2(wp.dot(wp.cross(n1, n2), e / le), l1 * l2 + wp.dot(n1, n2))
    d = ang - rest_angle[h]
    k = k_bend * le * le / (l1 + l2)
    wp.atomic_add(f, ic, -(k * d) * (n1 / l1))
    wp.atomic_add(f, idd, -(k * d) * (n2 / l2))
    wp.atomic_add(f, ia, (k * d) * 0.5 * (n1 / l1 + n2 / l2))
    wp.atomic_add(f, ib, (k * d) * 0.5 * (n1 / l1 + n2 / l2))


# ---------------------------------------------------------------- 接触
@wp.kernel
def contact_forces(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    tool: wp.uint64,                 # wp.Mesh id
    tool_vel: wp.vec3,               # 刚性工具平动速度
    up: float,                       # 合模方向:凹模 +1(板料在上), 凸模 -1
    half_gap: float,                 # 板料半厚 + 间隙
    k_contact: float,
    friction: float,
    damp_ratio: float,               # 接触法向阻尼比
    inv_mass: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    p = x[i]
    # 查询半径只比接触间隙略大(相当于 OpenRadioss TYPE7 的 GAP_MAX)。
    # 开太大会出事:深腔上方的平板节点,其最近面是近乎垂直的侧壁,它相对该面
    # 位于法向负侧 -> 被判成穿透几毫米 -> 被侧壁横向猛推,一百步内发散。
    # 真实接触的最近距离在 half_gap 量级,伪接触在数毫米量级,靠半径就能分开。
    q = wp.mesh_query_point(tool, p, half_gap * 2.5)
    if not q.result:
        return
    # 最近点必须落在三角形内部。悬在深腔上方的平板节点,其最近点落在侧壁的
    # 棱边/顶点上(重心坐标贴边),这类投影不是真接触,是非凸开放曲面上最近点
    # 判据的固有假阳性 —— 必须在这里剔除。
    eps = 1.0e-4
    if q.u < eps or q.v < eps or (q.u + q.v) > 1.0 - eps:
        return
    cp = wp.mesh_eval_position(tool, q.face, q.u, q.v)
    # 工具是开放曲面,sign_normal 的内外号不可靠;改用面法向。
    # 朝向在建工具时按整体绕序统一好了(见 add_tool),这里不能再逐面按 z 分量翻转——
    # 侧壁接近垂直(本算例最大 87°)时 n_z≈0,逐面翻转会被数值噪声决定方向,
    # 产生方向随机的巨大接触力(第一版就是这么炸的)。
    # 绕序已在 add_tool 里按 up 统一好,面法向直接就指向板料一侧。
    # 这里不能再乘一次 up —— 那会把凸模(up=-1)的法向翻反(凹模乘 +1 无害,
    # 所以"只有凸模接触发散"正是这个双重翻转的症状)。
    nrm = wp.mesh_eval_face_normal(tool, q.face)
    dn = wp.dot(p - cp, nrm)
    pen = half_gap - dn
    # 限制最大穿透(相当于 TYPE7 的 Fpenmax)。真实接触的穿透从 0 逐渐长起来,
    # 不可能一步就深过 half_gap;而深腔上方的节点相对陡壁面天然位于法向负侧
    # (dn 可达 -1 mm 以上),只能靠这条判据剔除,查询半径分不开(真实间隙 0.6mm
    # 与伪接触 1.39mm 太接近)。
    # 单侧面接触:只在节点位于面的正侧、且已进入间隙时施力。
    # dn<=0 说明节点在面背后 —— 对这种一侧铺料的成形问题只可能是几何假阳性,
    # 真实接触会在 dn 从大到 half_gap 时就被推回,dn 始终保持为正。
    if dn <= 0.0 or dn >= half_gap:
        return
    rel = v[i] - tool_vel
    # 法向接触阻尼(相当于 TYPE7 的 VisS)。纯弹性罚函数在节点跨面滑动、法向
    # 跳变时会持续泵入能量 —— 实测无阻尼时动能单调增长,一万多步后个别节点被
    # 踢到 30 mm/ms。阻尼系数按临界阻尼 2*zeta*sqrt(k*m) 取。
    vn = wp.dot(rel, nrm)
    im = inv_mass[i]
    c = 0.0
    if im > 0.0:
        c = 2.0 * damp_ratio * wp.sqrt(k_contact / im)
    fn = k_contact * pen - c * vn
    fn = wp.max(fn, 0.0)              # 接触只能推,不能拉
    fvec = fn * nrm
    # 库仑摩擦:沿相对切向速度反向
    vt = rel - vn * nrm
    lvt = wp.length(vt)
    if lvt > 1.0e-9:
        fvec = fvec - (friction * fn) * (vt / lvt)
    wp.atomic_add(f, i, fvec)


# ---------------------------------------------------------------- 积分
@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    damping: float,
    dt: float,
):
    i = wp.tid()
    im = inv_mass[i]
    if im == 0.0:
        v[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    vel = (v[i] + dt * im * f[i]) * damping
    v[i] = vel
    x[i] = x[i] + dt * vel


@wp.kernel
def zero_forces(f: wp.array(dtype=wp.vec3)):
    f[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def prescribe_x(
    x: wp.array(dtype=wp.vec3),
    idx: wp.array(dtype=int),
    x0: wp.array(dtype=wp.vec3),
    disp: wp.vec3,
):
    """把指定节点摆到 x0+disp(位移控制边界,避免每步 GPU<->CPU 往返)。"""
    i = wp.tid()
    x[idx[i]] = x0[i] + disp


@wp.kernel
def compute_rest(
    x: wp.array(dtype=wp.vec3),
    tris: wp.array2d(dtype=int),
    dm_inv: wp.array(dtype=wp.mat22),
    area0: wp.array(dtype=float),
    mass: wp.array(dtype=float),
    rho_t: float,
):
    e = wp.tid()
    i0 = tris[e, 0]
    i1 = tris[e, 1]
    i2 = tris[e, 2]
    p0 = x[i0]
    u = x[i1] - p0
    v = x[i2] - p0
    nrm = wp.cross(u, v)
    a = 0.5 * wp.length(nrm)
    area0[e] = a
    ex = wp.normalize(u)
    ez = wp.normalize(nrm)
    ey = wp.cross(ez, ex)
    dm = wp.mat22(wp.dot(u, ex), wp.dot(v, ex),
                  wp.dot(u, ey), wp.dot(v, ey))
    det = dm[0, 0] * dm[1, 1] - dm[0, 1] * dm[1, 0]
    if wp.abs(det) > 1.0e-14:
        dm_inv[e] = wp.mat22(dm[1, 1] / det, -dm[0, 1] / det,
                             -dm[1, 0] / det, dm[0, 0] / det)
    m = rho_t * a / 3.0
    wp.atomic_add(mass, i0, m)
    wp.atomic_add(mass, i1, m)
    wp.atomic_add(mass, i2, m)


@wp.kernel
def init_state(fp_inv: wp.array(dtype=wp.mat22), thickness: wp.array(dtype=float), t0: float):
    e = wp.tid()
    fp_inv[e] = wp.mat22(1.0, 0.0, 0.0, 1.0)
    thickness[e] = t0


@wp.kernel
def finalize_mass(mass: wp.array(dtype=float), inv_mass: wp.array(dtype=float),
                  fixed: wp.array(dtype=int), scale: float):
    i = wp.tid()
    if fixed[i] != 0 or mass[i] <= 0.0:
        inv_mass[i] = 0.0
    else:
        inv_mass[i] = 1.0 / (mass[i] * scale)


# ---------------------------------------------------------------- 驱动类
def build_hinges(tris: np.ndarray) -> np.ndarray:
    """内部边 -> (a, b, c, d):共享边两端 + 两侧对角点。"""
    edge: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for t in tris:
        for k in range(3):
            a, b, c = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            key = (min(a, b), max(a, b))
            edge.setdefault(key, []).append((a, b, c))
    out = []
    for (a, b), lst in edge.items():
        if len(lst) == 2:
            out.append([a, b, lst[0][2], lst[1][2]])
    return np.array(out, dtype=np.int32) if out else np.zeros((0, 4), np.int32)


class FormingSolver:
    """板料 + 若干刚性工具的显式成形求解器。"""

    def __init__(self, points: np.ndarray, tris: np.ndarray, material,
                 device: str = "cuda:0", mass_scale: float = 1.0,
                 damping: float = 0.999, k_bend: float = 0.0):
        wp.init()
        self.device = device
        self.n = len(points)
        self.ne = len(tris)
        self.material = material
        self.t0 = material.thickness
        self.mass_scale = mass_scale
        self.damping = damping
        self.k_bend = k_bend

        E, nu = material.youngs, material.poisson
        self.mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        # 平面应力等效: lam* = 2*lam*mu/(lam+2mu)
        self.lam_ps = 2.0 * lam * self.mu / (lam + 2.0 * self.mu)

        with wp.ScopedDevice(device):
            self.x = wp.array(points.astype(np.float32), dtype=wp.vec3)
            self.v = wp.zeros(self.n, dtype=wp.vec3)
            self.f = wp.zeros(self.n, dtype=wp.vec3)
            self.tris = wp.array(tris.astype(np.int32), dtype=int)
            self.dm_inv = wp.zeros(self.ne, dtype=wp.mat22)
            self.area0 = wp.zeros(self.ne, dtype=float)
            self.fp_inv = wp.zeros(self.ne, dtype=wp.mat22)
            self.eqps = wp.zeros(self.ne, dtype=float)
            self.thickness = wp.zeros(self.ne, dtype=float)
            self.mass = wp.zeros(self.n, dtype=float)
            self.inv_mass = wp.zeros(self.n, dtype=float)
            self.fixed = wp.zeros(self.n, dtype=int)

            ce = np.array([p[0] for p in material.flow_curve], np.float32)
            cs = np.array([p[1] for p in material.flow_curve], np.float32)
            self.curve_e = wp.array(ce, dtype=float)
            self.curve_s = wp.array(cs, dtype=float)
            self.n_curve = len(ce)

            wp.launch(compute_rest, dim=self.ne,
                      inputs=[self.x, self.tris, self.dm_inv, self.area0, self.mass,
                              material.density * self.t0])
            wp.launch(init_state, dim=self.ne, inputs=[self.fp_inv, self.thickness, self.t0])

            hin = build_hinges(tris)
            self.n_hinge = len(hin)
            self.hinges = wp.array(hin, dtype=int) if self.n_hinge else None
            self.rest_angle = wp.zeros(max(self.n_hinge, 1), dtype=float)

        self.tools: list[dict] = []
        self._mass_done = False
        self.contact_damping = 0.5

    def add_tool(self, points: np.ndarray, tris: np.ndarray, up: float = 1.0,
                 velocity=(0.0, 0.0, 0.0)):
        """up = +1 表示板料在工具上方(凹模), -1 表示板料在工具下方(凸模)。

        网格绕序在这里统一一次:让面法向整体指向板料所在一侧。之后 kernel 直接
        用面法向 * up,不再逐面判断(垂直侧壁上逐面判断会被噪声主导)。
        """
        p = points[tris]
        fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])     # 未归一,模长=2A(面积加权)
        if fn[:, 2].sum() * up < 0.0:
            tris = tris[:, [0, 2, 1]]                            # 整体翻转绕序
            fn = -fn
        # 清理接触网格:偏置面在小圆角处会自交,留下法向翻转/退化的三角形。
        # 这类面在接触查询里会注入方向错误的大力,必须剔除(工具接触网格清理是
        # 常规做法,不动板料网格)。
        area2 = np.linalg.norm(fn, axis=1)
        nz = fn / np.maximum(area2[:, None], 1e-30)
        mean_n = (fn.sum(axis=0)) / max(np.linalg.norm(fn.sum(axis=0)), 1e-30)
        keep = (area2 > 1e-9) & ((nz * mean_n).sum(axis=1) > -0.5)
        n_drop = int((~keep).sum())
        if n_drop:
            tris = tris[keep]
            print(f"  接触网格清理: 剔除 {n_drop} 个翻转/退化三角形 (剩 {len(tris)})")
        with wp.ScopedDevice(self.device):
            mp = wp.array(points.astype(np.float32), dtype=wp.vec3)
            mi = wp.array(tris.astype(np.int32).flatten(), dtype=int)
            mesh = wp.Mesh(points=mp, indices=mi)
        self.tools.append({"mesh": mesh, "points": mp, "base": points.copy(), "up": float(up),
                           "vel": np.array(velocity, np.float32), "offset": np.zeros(3, np.float32)})
        return len(self.tools) - 1

    def apply_nodal_mass_scaling(self, dt_target: float, k_contact: float = 0.0,
                                 verbose: bool = True) -> dict:
        """逐节点质量缩放(等价于 OpenRadioss 的 /DT/NODA/CST)。

        显式稳定性要求每个节点 dt <= 2*sqrt(m/k)。板料网格里最小单元
        (Sample 7 最短边 0.13 mm)决定了全局步长,只给这些节点加质量,
        比全局统一放大惯性失真小得多。接触刚度也计入(否则小质量节点在
        接触上先炸——这正是第一次跑飞的原因)。
        """
        pts = self.x.numpy().astype(np.float64)
        tris = self.tris.numpy()
        c = np.sqrt(self.material.youngs / self.material.density)   # 弹性波速 mm/ms

        p = pts[tris]
        e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
        e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
        area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        # 特征长度 = 最小高 = 2A / 最长边
        h = 2.0 * area / np.maximum(np.maximum(e0, np.maximum(e1, e2)), 1e-12)
        dt_e = h / c

        dt_node = np.full(len(pts), np.inf)
        for k in range(3):
            np.minimum.at(dt_node, tris[:, k], dt_e)
        dt_node[~np.isfinite(dt_node)] = dt_target

        m = self.mass.numpy().astype(np.float64)
        safety = float(getattr(self, "stab_safety", 0.6))                      # 稳定裕度:取临界步长的 0.6 倍
        need = np.maximum(1.0, (dt_target / np.maximum(safety * dt_node, 1e-12)) ** 2)
        m_scaled = m * need
        if k_contact > 0.0:
            # 接触弹簧的稳定条件 dt <= 2*sqrt(m/k),同样留裕度
            m_scaled = np.maximum(m_scaled, k_contact * (0.5 * dt_target / safety) ** 2)

        fixed = self.fixed.numpy()
        inv = np.where((fixed != 0) | (m_scaled <= 0), 0.0, 1.0 / np.maximum(m_scaled, 1e-30))
        self.inv_mass = wp.array(inv.astype(np.float32), dtype=float, device=self.device)
        self._mass_done = True

        info = {
            "dt_target": dt_target,
            "dt_min_natural": float(dt_node.min()),
            "added_mass_pct": float((m_scaled.sum() / m.sum() - 1.0) * 100.0),
            "n_scaled": int((need > 1.0).sum()),
        }
        if verbose:
            print(f"  质量缩放: 自然步长 {info['dt_min_natural']:.2e} ms -> 目标 {dt_target:.2e} ms, "
                  f"{info['n_scaled']}/{len(pts)} 个节点加质量, 总质量 +{info['added_mass_pct']:.1f}%")
        return info

    def set_fixed(self, idx: np.ndarray):
        fx = self.fixed.numpy()
        fx[idx] = 1
        self.fixed = wp.array(fx, dtype=int, device=self.device)
        self._mass_done = False

    def _ensure_mass(self):
        if not self._mass_done:
            wp.launch(finalize_mass, dim=self.n,
                      inputs=[self.mass, self.inv_mass, self.fixed, self.mass_scale],
                      device=self.device)
            self._mass_done = True

    def move_tool(self, i: int, offset: np.ndarray):
        """把工具整体平移到 base+offset(刚体运动,直接改网格顶点)。"""
        t = self.tools[i]
        t["offset"] = np.asarray(offset, np.float32)
        t["points"].assign((t["base"] + t["offset"]).astype(np.float32))
        t["mesh"].refit()

    def step(self, dt: float, k_contact: float, friction: float, gap: float | None = None,
             contact_damping: float | None = None):
        if contact_damping is not None:
            self.contact_damping = contact_damping
        self._ensure_mass()
        half_gap = (0.5 * self.t0) if gap is None else gap
        with wp.ScopedDevice(self.device):
            wp.launch(zero_forces, dim=self.n, inputs=[self.f])
            wp.launch(membrane_forces, dim=self.ne,
                      inputs=[self.x, self.tris, self.dm_inv, self.area0, self.fp_inv,
                              self.eqps, self.thickness, self.curve_e, self.curve_s,
                              self.n_curve, self.mu, self.lam_ps, self.t0, self.f])
            if self.n_hinge and self.k_bend > 0.0:
                wp.launch(bending_forces, dim=self.n_hinge,
                          inputs=[self.x, self.hinges, self.rest_angle, self.k_bend, self.f])
            for t in self.tools:
                wp.launch(contact_forces, dim=self.n,
                          inputs=[self.x, self.v, t["mesh"].id, wp.vec3(*t["vel"]),
                                  t["up"], half_gap, k_contact, friction,
                                  self.contact_damping, self.inv_mass, self.f])
            wp.launch(integrate, dim=self.n,
                      inputs=[self.x, self.v, self.f, self.inv_mass, self.damping, dt])

    # ---------- 结果 ----------
    def thinning_pct(self) -> np.ndarray:
        return (self.t0 - self.thickness.numpy()) / self.t0 * 100.0

    def state(self) -> dict:
        th = self.thickness.numpy()
        return {
            "positions": self.x.numpy(),
            "thickness": th,
            "thinning_pct": (self.t0 - th) / self.t0 * 100.0,
            "eqps": self.eqps.numpy(),
        }

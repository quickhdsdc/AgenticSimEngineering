# -*- coding: utf-8 -*-
"""解析验证:单向拉伸。这是判断本构写对没有的唯一硬标准。

一条矩形板条沿 x 拉到伸长比 λ,理论解(金属塑性、塑性不可压):
    真应变      eps = ln λ
    流动应力    sigma = 流动曲线(eps)          <- 硬化律是否接对
    横向/厚度   λ2 = λ3 = λ^(-1/2)             <- 不可压缩是否满足
    料厚        t/t0 = λ^(-1/2)
    减薄率      1 - λ^(-1/2)

跑法:
    PYTHONUTF8=1 uv run --with warp-lang,numpy python -m formpipe.wpform.validate_uniaxial
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..spec import MaterialSpec
from .solver import FormingSolver


def make_strip(nx: int = 12, ny: int = 4, lx: float = 60.0, ly: float = 20.0):
    xs = np.linspace(0, lx, nx + 1)
    ys = np.linspace(0, ly, ny + 1)
    pts = np.array([[x, y, 0.0] for y in ys for x in xs], np.float64)
    tris = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + nx + 1
            d = c + 1
            tris += [[a, b, d], [a, d, c]]
    return pts, np.array(tris, np.int32), xs, ys


def run(stretch: float = 1.25, steps: int = 40000, verbose: bool = True) -> dict:
    mat = MaterialSpec(
        name="test_1050O", thickness=1.2, youngs=70000.0, poisson=0.33, density=2.7e-3,
        flow_curve=[(0.0, 48.4), (0.05, 66.27), (0.10, 78.96), (0.20, 96.62),
                    (0.30, 109.98), (0.50, 130.90), (0.80, 155.15), (1.00, 168.69),
                    (2.00, 234.4)],
        r00=1.0, r45=1.0, r90=1.0)

    pts, tris, xs, ys = make_strip()
    lx = xs[-1]
    solver = FormingSolver(pts, tris, mat, mass_scale=1.0, damping=0.995)

    left = np.where(np.abs(pts[:, 0] - 0.0) < 1e-9)[0]
    right = np.where(np.abs(pts[:, 0] - lx) < 1e-9)[0]
    solver.set_fixed(np.concatenate([left, right]))

    # 准静态:缓慢把右端拉到目标伸长(位移控制,全程留在 GPU 上)
    import warp as _wp
    from .solver import prescribe_x
    bc_idx = _wp.array(np.concatenate([left, right]).astype(np.int32), dtype=int,
                       device=solver.device)
    bc_x0 = _wp.array(pts[np.concatenate([left, right])].astype(np.float32),
                      dtype=_wp.vec3, device=solver.device)
    n_left = len(left)
    li = _wp.array(left.astype(np.int32), dtype=int, device=solver.device)
    ri = _wp.array(right.astype(np.int32), dtype=int, device=solver.device)
    lx0 = _wp.array(pts[left].astype(np.float32), dtype=_wp.vec3, device=solver.device)
    rx0 = _wp.array(pts[right].astype(np.float32), dtype=_wp.vec3, device=solver.device)

    target = lx * (stretch - 1.0)
    dt = 2.0e-4
    ramp = int(steps * 0.85)
    for s in range(steps):
        a = min(1.0, s / ramp)
        disp = target * (1.0 - np.cos(np.pi * a)) * 0.5
        _wp.launch(prescribe_x, dim=n_left, inputs=[solver.x, li, lx0, _wp.vec3(0.0, 0.0, 0.0)],
                   device=solver.device)
        _wp.launch(prescribe_x, dim=len(right), inputs=[solver.x, ri, rx0,
                                                        _wp.vec3(float(disp), 0.0, 0.0)],
                   device=solver.device)
        solver.step(dt, k_contact=0.0, friction=0.0)

    st = solver.state()
    # 只看中间区域单元(避开夹持端约束影响)
    cent = st["positions"][tris].mean(axis=1)
    span = cent[:, 0].max() - cent[:, 0].min()
    lo = cent[:, 0].min() + 0.3 * span
    hi = cent[:, 0].min() + 0.7 * span
    mid = (cent[:, 0] > lo) & (cent[:, 0] < hi)

    lam = stretch
    exp_eps = np.log(lam)
    exp_thick = mat.thickness * lam ** -0.5
    exp_thinning = (1.0 - lam ** -0.5) * 100.0
    exp_sigma = np.interp(exp_eps, [p[0] for p in mat.flow_curve], [p[1] for p in mat.flow_curve])

    got_thick = float(st["thickness"][mid].mean())
    got_eqps = float(st["eqps"][mid].mean())
    got_thinning = float(st["thinning_pct"][mid].mean())

    res = {
        "stretch": lam,
        "eps_expected": exp_eps, "eqps_got": got_eqps,
        "thickness_expected": exp_thick, "thickness_got": got_thick,
        "thinning_expected_pct": exp_thinning, "thinning_got_pct": got_thinning,
        "sigma_flow_at_eps": exp_sigma,
    }
    if verbose:
        print(f"\n=== 单向拉伸 λ={lam} ===")
        print(f"  等效塑性应变   理论 {exp_eps:.4f}   实测 {got_eqps:.4f}   "
              f"误差 {abs(got_eqps - exp_eps) / exp_eps * 100:.2f}%")
        print(f"  料厚 (mm)      理论 {exp_thick:.4f}   实测 {got_thick:.4f}   "
              f"误差 {abs(got_thick - exp_thick) / exp_thick * 100:.2f}%")
        print(f"  减薄率 (%)     理论 {exp_thinning:.2f}     实测 {got_thinning:.2f}")
    return res


if __name__ == "__main__":
    wp.config.quiet = True
    for lam in (1.10, 1.25, 1.50):
        run(lam)

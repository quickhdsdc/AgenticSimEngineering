# -*- coding: utf-8 -*-
"""分段诊断:把发散来源拆开定位,而不是猜。

  A. 静止平板(无工具、无外力)   -> 应完全不动。动了 = 膜元内力有问题
  B. 平板 + 凹模(凸模不动)      -> 应几乎不动。动了 = 接触方向/刚度有问题
  C. 缓慢压下                    -> 观察能量与料厚演化

    PYTHONUTF8=1 uv run --with warp-lang,numpy,scipy python -m formpipe.wpform.diagnose
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..quicklink import load
from ..spec import ProcessSpec
from .solver import FormingSolver
from .run_sample7 import ROOT


def kinetic_energy(solver) -> float:
    v = solver.v.numpy()
    im = solver.inv_mass.numpy()
    m = np.where(im > 0, 1.0 / np.maximum(im, 1e-30), 0.0)
    return float(0.5 * (m * (v ** 2).sum(axis=1)).sum())


def main():
    spec, ref = load(ROOT / "sample7_run" / "quicklink",
                     process=ProcessSpec(clearance_ratio=1.15))
    blank, die, punch = spec.blank, spec.die.mesh, spec.punch.mesh
    dt = 1.0e-4
    k_contact = 1.5e5

    print("=" * 70)
    print("A. 静止平板(无工具):检验膜元内力")
    s = FormingSolver(blank.points3d, blank.tris, spec.material, damping=0.999)
    s.apply_nodal_mass_scaling(dt, k_contact=0.0)
    for i in range(2000):
        s.step(dt, k_contact=0.0, friction=0.0)
    st = s.state()
    dx = np.abs(st["positions"] - blank.points3d).max()
    print(f"   最大位移 {dx:.3e} mm   最大 eqps {st['eqps'].max():.3e}   "
          f"料厚范围 {st['thickness'].min():.4f}~{st['thickness'].max():.4f}")
    print(f"   -> {'OK' if dx < 1e-3 else '异常:膜元内力有问题'}")

    print("=" * 70)
    print("B. 平板 + 静止凹模:检验接触方向")
    s = FormingSolver(blank.points3d, blank.tris, spec.material, damping=0.999)
    s.add_tool(die.points3d, die.tris, up=+1.0)
    s.apply_nodal_mass_scaling(dt, k_contact=k_contact)
    for i in range(2000):
        s.step(dt, k_contact=k_contact, friction=0.0, gap=0.5 * spec.material.thickness)
        if i in (99, 499, 1999):
            st = s.state()
            print(f"   step {i+1:5d}  最大位移 {np.abs(st['positions'] - blank.points3d).max():9.4f} mm"
                  f"  动能 {kinetic_energy(s):10.3e}  最小料厚 {st['thickness'].min():.4f}")

    print("=" * 70)
    print("C. 缓慢压下(2 万步到全行程):看能量与料厚")
    s = FormingSolver(blank.points3d, blank.tris, spec.material, damping=0.9995)
    s.add_tool(die.points3d, die.tris, up=+1.0)
    ip = s.add_tool(punch.points3d, punch.tris, up=-1.0)
    s.apply_nodal_mass_scaling(dt, k_contact=k_contact)
    stroke = spec.punch.stroke
    N = 20000
    for i in range(N):
        a = min(1.0, i / (N * 0.9))
        d = -stroke * 0.5 * (1.0 - np.cos(np.pi * a))
        s.move_tool(ip, np.array([0.0, 0.0, d], np.float32))
        s.step(dt, k_contact=k_contact, friction=spec.process.friction,
               gap=0.5 * spec.material.thickness)
        if (i + 1) % 4000 == 0:
            st = s.state()
            th = st["thickness"]
            print(f"   step {i+1:6d}  行程 {-d:6.3f}  动能 {kinetic_energy(s):10.3e}  "
                  f"料厚 {np.nanmin(th):.4f}~{np.nanmax(th):.4f}  "
                  f"eqps_max {st['eqps'].max():.3f}")


if __name__ == "__main__":
    main()

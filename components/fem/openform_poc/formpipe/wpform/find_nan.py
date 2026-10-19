# -*- coding: utf-8 -*-
"""逐步推进,定位第一个 NaN 出现的步数与节点,并打印该节点的接触状态。"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..quicklink import load
from ..spec import ProcessSpec
from .solver import FormingSolver
from .run_sample7 import ROOT


def main():
    import sys
    with_punch = "--punch" in sys.argv
    n_steps = 20000 if with_punch else 300

    spec, _ = load(ROOT / "sample7_run" / "quicklink", process=ProcessSpec(clearance_ratio=1.15))
    blank, die, punch = spec.blank, spec.die.mesh, spec.punch.mesh
    half_gap = 0.5 * spec.material.thickness
    dt, k_contact = 1.0e-4, 1.5e5

    s = FormingSolver(blank.points3d, blank.tris, spec.material, damping=0.9995)
    s.add_tool(die.points3d, die.tris, up=+1.0)
    ip = s.add_tool(punch.points3d, punch.tris, up=-1.0) if with_punch else None
    s.apply_nodal_mass_scaling(dt, k_contact=k_contact, verbose=False)
    stroke = spec.punch.stroke

    x0 = blank.points3d.copy()
    prev_x = x0.copy()
    for i in range(n_steps):
        if with_punch:
            a = min(1.0, i / (n_steps * 0.9))
            s.move_tool(ip, np.array([0.0, 0.0, -stroke * 0.5 * (1.0 - np.cos(np.pi * a))],
                                     np.float32))
        s.step(dt, k_contact=k_contact, friction=0.0, gap=half_gap)
        x = s.x.numpy()
        v = s.v.numpy()
        bad = ~np.isfinite(x).all(axis=1)
        big = np.linalg.norm(x - x0, axis=1)
        if bad.any() or big.max() > (12.0 if with_punch else 1.0):
            j = int(np.argmax(np.where(bad, np.inf, big)))
            print(f"步 {i+1}: {'出现 NaN' if bad.any() else '位移异常'}, "
                  f"NaN 节点数 {bad.sum()}, 最大位移 {np.nanmax(big):.4f} mm")
            print(f"  节点 {j}: 初始 {x0[j]}  当前 {x[j]}  速度 {v[j]}")
            print(f"  上一步位置 {prev_x[j]}")
            im = s.inv_mass.numpy()[j]
            print(f"  inv_mass {im:.6e}  -> 质量 {1.0/im if im>0 else 0:.6e} g")
            # 该节点到凹模的几何关系
            from scipy.spatial import cKDTree
            d, k = cKDTree(die.points3d).query(prev_x[j])
            print(f"  上一步到凹模最近节点距离 {d:.4f} mm (凹模节点 {k} @ {die.points3d[k]})")
            # 邻域单元
            tri_ids = np.where((blank.tris == j).any(axis=1))[0]
            print(f"  所属单元 {len(tri_ids)} 个, eqps {s.eqps.numpy()[tri_ids]}")
            print(f"  料厚 {s.thickness.numpy()[tri_ids]}")
            return
        prev_x = x.copy()
    print("300 步内未发散")


if __name__ == "__main__":
    main()

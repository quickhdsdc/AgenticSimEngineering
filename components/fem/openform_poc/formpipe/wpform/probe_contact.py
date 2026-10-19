# -*- coding: utf-8 -*-
"""把第 0 步的接触查询结果导出来看:哪些节点判定接触、穿透量多大、法向朝哪。"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..quicklink import load
from ..spec import ProcessSpec
from .run_sample7 import ROOT


@wp.kernel
def probe(
    x: wp.array(dtype=wp.vec3),
    tool: wp.uint64,
    up: float,
    half_gap: float,
    max_dist: float,
    hit: wp.array(dtype=int),
    pen: wp.array(dtype=float),
    nz: wp.array(dtype=float),
    dist: wp.array(dtype=float),
):
    i = wp.tid()
    p = x[i]
    q = wp.mesh_query_point(tool, p, max_dist)
    if not q.result:
        hit[i] = 0
        return
    hit[i] = 1
    cp = wp.mesh_eval_position(tool, q.face, q.u, q.v)
    nrm = wp.mesh_eval_face_normal(tool, q.face) * up
    d = wp.dot(p - cp, nrm)
    dist[i] = wp.length(p - cp)
    pen[i] = half_gap - d
    nz[i] = nrm[2]


def main():
    wp.init()
    spec, _ = load(ROOT / "sample7_run" / "quicklink", process=ProcessSpec(clearance_ratio=1.15))
    blank, die = spec.blank, spec.die.mesh
    half_gap = 0.5 * spec.material.thickness

    pts = die.points3d
    tris = die.tris
    fn = np.cross(pts[tris][:, 1] - pts[tris][:, 0], pts[tris][:, 2] - pts[tris][:, 0])
    print(f"凹模面法向 z 分量: 和 {fn[:,2].sum():+.3f}  "
          f"正 {(fn[:,2]>0).sum()} / 负 {(fn[:,2]<0).sum()} (共 {len(tris)})")
    print(f"  -> 绕序{'一致' if abs(fn[:,2].sum())/np.abs(fn[:,2]).sum() > 0.5 else '不一致!'}")
    if fn[:, 2].sum() * 1.0 < 0.0:
        tris = tris[:, [0, 2, 1]]
        print("  已整体翻转绕序")

    dev = "cuda:0"
    mp = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=dev)
    mi = wp.array(tris.astype(np.int32).flatten(), dtype=int, device=dev)
    mesh = wp.Mesh(points=mp, indices=mi)
    bx = wp.array(blank.points3d.astype(np.float32), dtype=wp.vec3, device=dev)
    n = blank.n_nodes
    hit = wp.zeros(n, dtype=int, device=dev)
    pen = wp.zeros(n, dtype=float, device=dev)
    nz = wp.zeros(n, dtype=float, device=dev)
    dist = wp.zeros(n, dtype=float, device=dev)

    for md in (half_gap * 6.0, 50.0):
        wp.launch(probe, dim=n, inputs=[bx, mesh.id, 1.0, half_gap, md, hit, pen, nz, dist],
                  device=dev)
        h, pe, z, dd = hit.numpy(), pen.numpy(), nz.numpy(), dist.numpy()
        m = h == 1
        print(f"\nmax_dist={md:.2f}: 命中 {m.sum()}/{n}")
        if m.sum():
            print(f"  最近距离 min {dd[m].min():.3f} / 中位 {np.median(dd[m]):.3f} / max {dd[m].max():.3f}")
            print(f"  穿透量 pen: min {pe[m].min():.3f} / 中位 {np.median(pe[m]):.3f} / max {pe[m].max():.3f}")
            print(f"  pen>0(判定接触)的节点数: {(pe[m] > 0).sum()}")
            print(f"  法向 n_z: min {z[m].min():+.3f} / 中位 {np.median(z[m]):+.3f} / max {z[m].max():+.3f}")
            bad = m & (pe > 0)
            if bad.sum():
                print(f"  ** pen>0 的节点里, n_z<0 的有 {(z[bad] < 0).sum()} 个, "
                      f"最大穿透 {pe[bad].max():.3f} mm (板料应在凹模上方 0.8mm 处,不该有穿透)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""几何构建:法向偏置生成凸模、退化修复、平面坯料刚性配准。

crash-form 的核心假设:零件成形面 = 凹模型面,凸模 = 该面沿法向偏置一个间隙。
偏置在曲率半径小于间隙的凹圆角处会自交/退化,必须修复,否则那里会出现
虚假的极端减薄(本项目实测:未修复时 max 减薄 82%,真值 38%)。
"""
from __future__ import annotations

import numpy as np


def nodal_normals(points: np.ndarray, tris: np.ndarray, n_smooth: int = 4) -> np.ndarray:
    """面积加权节点法向 + 邻域平滑(平滑让偏置面在圆角处更平顺)。"""
    p = points[tris]
    face_n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])   # 模长 = 2×面积
    nn = np.zeros_like(points, dtype=float)
    for k in range(3):
        np.add.at(nn, tris[:, k], face_n)
    for _ in range(n_smooth):
        en = nn[tris[:, 0]] + nn[tris[:, 1]] + nn[tris[:, 2]]
        acc = np.zeros_like(points, dtype=float)
        for k in range(3):
            np.add.at(acc, tris[:, k], en)
        nn = acc
    return nn / (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-30)


def neighbor_lists(n_nodes: int, tris: np.ndarray) -> list[np.ndarray]:
    nbr: list[set] = [set() for _ in range(n_nodes)]
    for a, b, c in tris:
        nbr[a].update((b, c))
        nbr[b].update((a, c))
        nbr[c].update((a, b))
    return [np.fromiter(s, dtype=int) for s in nbr]


def offset_surface(points: np.ndarray, tris: np.ndarray, distance: float,
                   direction: int = -1, repair_iters: int = 60,
                   repair_tol: float = 0.95) -> tuple[np.ndarray, dict]:
    """沿节点法向偏置 distance;对退化处(到原面距离过近)做邻域平滑修复。

    direction: -1 表示朝 -z 一侧偏置(法向先统一到 -z)。
    返回 (偏置后节点, 诊断信息)。
    """
    from scipy.spatial import cKDTree

    # 先把法向统一到 direction 指定的 z 朝向,再反向偏置:
    # direction=-1 时法向朝 -z,偏置结果落在型面上方(凸模位置)
    nrm = nodal_normals(points, tris)
    if np.sign(nrm[:, 2].mean()) != np.sign(direction):
        nrm = -nrm
    out = points - abs(distance) * nrm

    nbr = neighbor_lists(len(points), tris)
    tree = cKDTree(points)
    it = 0
    for it in range(repair_iters):
        d, _ = tree.query(out)
        bad = d < repair_tol * abs(distance)
        if not bad.any():
            break
        idx = np.where(bad)[0]
        avg = np.array([out[nbr[i]].mean(axis=0) for i in idx])
        out[idx] = 0.5 * out[idx] + 0.5 * avg
    d, _ = tree.query(out)
    info = {
        "repair_iters": it + 1,
        "min_clearance": float(d.min()),
        "target_clearance": float(abs(distance)),
        "n_below_tol": int((d < repair_tol * abs(distance)).sum()),
    }
    return out, info


def register_2d(source: np.ndarray, target: np.ndarray,
                allow_mirror: bool = True) -> tuple[np.ndarray, dict]:
    """把展开平面坐标刚性配准到成形件的 (x,y) 投影上(Kabsch)。

    AutoForm 的展开坐标系与冲压坐标系可能互为镜像(实测残差 58mm vs 1.2mm),
    所以默认两种手性都试,取残差小的。配准后的残差即面内拉延收缩量。
    """
    best = None
    for mirror in ([False, True] if allow_mirror else [False]):
        s = source * (np.array([-1.0, 1.0]) if mirror else 1.0)
        sc, tc = s - s.mean(0), target - target.mean(0)
        U, _, Vt = np.linalg.svd(sc.T @ tc)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U = U.copy()
            U[:, -1] *= -1
            R = U @ Vt
        fitted = sc @ R + target.mean(0)
        res = np.linalg.norm(fitted - target, axis=1)
        cand = (res.mean(), fitted, {"mirrored": mirror,
                                     "residual_mean": float(res.mean()),
                                     "residual_max": float(res.max())})
        if best is None or cand[0] < best[0]:
            best = cand
    return best[1], best[2]

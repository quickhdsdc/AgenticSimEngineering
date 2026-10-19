# -*- coding: utf-8 -*-
"""参数扫描 -> 代理模型数据集。

目的:前期评估要在报价时间尺度(秒级)给成形性结论,而一次求解 ~100 s。
所以用求解器离线批量生成 (工艺参数, 几何特征) -> (减薄场) 的样本,
在线阶段由代理模型直接预测。这与 P15 的 world model 架构同构:
仿真产轨迹 -> 数据集 -> 学代理。

每个样本存:
    输入  工艺参数(摩擦/间隙比/料厚/材料强度系数) + 每单元几何特征
    输出  每单元减薄率 + 全局判据
几何特征逐单元算(展开态边长/曲率/到边界距离等),使代理模型可以是
逐单元回归或 GNN,而不必绑定在某一个零件上。

用法:
    uv run --with numpy,scipy python -m formpipe.sweep \
        --quicklink sample7_run/quicklink --out datasets/s7 --n 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import openradioss
from .quicklink import load
from .run import solve
from .spec import ProcessSpec, SpecError


def element_features(spec) -> np.ndarray:
    """逐单元几何特征(与工艺参数无关的部分),(NE, F)。

    这些量在展开态/成形态都能算,是代理模型区分"哪个单元会减薄"的依据。
    """
    from scipy.spatial import cKDTree

    blank, die = spec.blank, spec.die.mesh
    tris = blank.tris
    flat = blank.points3d
    formed = die.points3d

    # 1. 展开态单元尺寸
    a_flat = blank.areas()
    # 2. 成形态面积 -> 面内伸长(面积比是减薄的一阶指示)
    p = formed[tris]
    a_form = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    area_ratio = a_form / np.maximum(a_flat, 1e-9)

    # 3. 单元法向与冲压方向夹角(侧壁 vs 平坦区)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    wall_angle = np.degrees(np.arccos(np.clip(np.abs(n[:, 2]), 0, 1)))

    # 4. 局部曲率代理:单元法向与其邻域平均法向的夹角
    from .geometry import nodal_normals
    nn = nodal_normals(formed, tris, n_smooth=2)
    node_n = nn[tris].mean(axis=1)
    node_n /= np.linalg.norm(node_n, axis=1, keepdims=True) + 1e-30
    curvature = np.degrees(np.arccos(np.clip(np.abs((n * node_n).sum(axis=1)), 0, 1)))

    # 5. 到自由边(孔边/轮廓)的距离 —— 边界附近材料流动自由,行为不同
    bn = blank.boundary_nodes()
    cent_form = p.mean(axis=1)
    dist_edge = cKDTree(formed[bn]).query(cent_form)[0] if len(bn) else np.zeros(len(tris))

    # 6. 成形深度(相对型面最高点)
    depth = formed[:, 2].max() - cent_form[:, 2]

    return np.column_stack([a_flat, area_ratio, wall_angle, curvature, dist_edge, depth])


FEATURE_NAMES = ["flat_area", "area_ratio", "wall_angle_deg", "curvature_deg",
                 "dist_to_edge", "draw_depth"]


def sample_grid(n: int, seed: int = 0) -> list[dict]:
    """工艺参数样本。少量样本用拉丁超立方,保证边界也被覆盖。"""
    rng = np.random.default_rng(seed)
    ranges = {
        "friction": (0.02, 0.18),          # 润滑良好 ~ 干摩擦
        "clearance_ratio": (1.05, 1.35),   # 壁部间隙
        "strength_factor": (0.85, 1.15),   # 材料批次强度波动
    }
    keys = list(ranges)
    strata = np.arange(n)[:, None].repeat(len(keys), axis=1)   # (n, k)

    def draw() -> np.ndarray:
        # 每一维独立打乱才是拉丁超立方。注意 rng.permutation(axis=1) 会对所有行
        # 套用同一个排列 -> 各维完全同序,样本退化成一条对角线(实测三个参数与
        # 减薄的相关系数都是 -0.9x,无法区分谁的贡献);必须用 rng.permuted。
        return (rng.permuted(strata, axis=0) + rng.random((n, len(keys)))) / n

    # 随机 LHS 各维仍可能残留 ~1/sqrt(n) 量级的相关。一次求解要 ~95 s,
    # 一个坏样本设计浪费的是分钟级机时,所以多抽几组取相关性最小的那组。
    iu = np.triu_indices(len(keys), 1)
    best, best_score = None, np.inf
    for _ in range(200):
        c = draw()
        score = np.abs(np.corrcoef(c.T)[iu]).max() if n > 2 else 0.0
        if score < best_score:
            best, best_score = c, score
    cube = best
    out = []
    for row in cube:
        out.append({k: float(ranges[k][0] + v * (ranges[k][1] - ranges[k][0]))
                    for k, v in zip(keys, row)})
    return out


def run_sweep(quicklink: str | Path, out: str | Path, n: int, seed: int = 0,
              form_time: float = 12.0) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = sample_grid(n, seed)

    feats = None
    tris = None
    records = []
    for i, s in enumerate(samples):
        tag = f"case{i:03d}"
        wd = out / tag
        proc = ProcessSpec(friction=s["friction"], clearance_ratio=s["clearance_ratio"],
                           form_time_ms=form_time)
        spec, reference = load(quicklink, process=proc, name=tag)
        spec.material = spec.material.scaled(s["strength_factor"])
        if feats is None:
            feats = element_features(spec)
            tris = spec.blank.tris.astype(np.int32)   # 全批次连接关系相同(同一零件)

        rec = {"case": tag, **s}
        t0 = time.time()
        try:
            res = solve(spec, wd, verbose=False)
            rec.update({
                "status": "ok",
                "min_thickness_mm": res["min_thickness_mm"],
                "max_thinning_pct": res["max_thinning_pct"],
                "p99_thinning_pct": res["p99_thinning_pct"],
                "interior_max_thinning_pct": res["interior_max_thinning_pct"],
                "max_plastic_strain": res["max_plastic_strain"],
                "closure_mean_mm": res["closure_mean_mm"],
                "thinning_ok": res["thinning_ok"],
                "wall_seconds": round(time.time() - t0, 1),
            })
            np.save(wd / "thinning.npy", res["element_thinning_pct"].astype(np.float32))
        except SpecError as e:
            rec.update({"status": "spec_rejected", "detail": str(e).splitlines()[0]})
        except Exception as e:  # 求解失败不应中断整批
            rec.update({"status": "failed", "detail": f"{type(e).__name__}: {e}"})
        records.append(rec)
        print(f"[{i + 1}/{n}] {tag}  " + ("  ".join(f"{k}={v:.3f}" for k, v in s.items()))
              + f"  -> {rec.get('status')}"
              + (f"  减薄 p99={rec['p99_thinning_pct']:.1f}%  "
                 f"内部max={rec['interior_max_thinning_pct']:.1f}%  "
                 f"{rec['wall_seconds']}s" if rec.get("status") == "ok" else ""), flush=True)

    # 打包数据集
    ok = [r for r in records if r.get("status") == "ok"]
    (out / "sweep.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    if not ok:
        print(f"\n完成 0/{n}:没有成功样本,未生成数据集(细节见 {out / 'sweep.json'})")
        return {"records": records, "n_ok": 0}

    y = np.stack([np.load(out / r["case"] / "thinning.npy") for r in ok])
    X_proc = np.array([[r["friction"], r["clearance_ratio"], r["strength_factor"]] for r in ok],
                      dtype=np.float32)
    np.savez_compressed(
        out / "dataset.npz",
        process_params=X_proc,
        process_names=np.array(["friction", "clearance_ratio", "strength_factor"]),
        element_features=feats.astype(np.float32),
        feature_names=np.array(FEATURE_NAMES),
        thinning_pct=y.astype(np.float32),
        tris=tris,
    )
    print(f"\n完成 {len(ok)}/{n};数据集 -> {out / 'dataset.npz'}"
          f"  (工艺参数 {X_proc.shape}, 单元特征 {feats.shape}, 减薄标签 {y.shape})")
    return {"records": records, "n_ok": len(ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description="formpipe 参数扫描 -> 代理模型数据集")
    ap.add_argument("--quicklink", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--form-time", type=float, default=12.0)
    a = ap.parse_args()
    run_sweep(a.quicklink, a.out, a.n, a.seed, a.form_time)


if __name__ == "__main__":
    main()

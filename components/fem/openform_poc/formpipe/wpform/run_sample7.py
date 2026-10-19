# -*- coding: utf-8 -*-
"""用 Warp 求解器跑 Sample 7,与 AutoForm / OpenRadioss 三方对标。

几何、材料、摩擦全部来自同一个 StampingSpec(formpipe.quicklink.load),
所以三者比的是求解器本身,不是建模差异。

    PYTHONUTF8=1 uv run --with warp-lang,numpy,scipy python -m formpipe.wpform.run_sample7
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from ..quicklink import load
from ..spec import ProcessSpec
from .solver import FormingSolver

ROOT = Path(__file__).resolve().parent.parent.parent


def run(quicklink: str | Path = ROOT / "sample7_run" / "quicklink",
        clearance: float = 1.15, steps: int = 60000, dt: float = 1.0e-4,
        mass_scale: float = 15.0, k_contact: float = 1.5e5, damping: float = 0.9995,
        k_bend: float = 0.0, out: str | Path | None = None, verbose: bool = True) -> dict:
    spec, reference = load(quicklink, process=ProcessSpec(clearance_ratio=clearance))
    warns = spec.validate(strict=True)
    if verbose:
        print(spec.summary())
        for w in warns:
            print(f"  ! {w[:110]}")

    blank, die, punch = spec.blank, spec.die.mesh, spec.punch.mesh
    stroke = spec.punch.stroke

    solver = FormingSolver(blank.points3d, blank.tris, spec.material,
                           mass_scale=1.0, damping=damping, k_bend=k_bend)
    solver.add_tool(die.points3d, die.tris, up=+1.0)          # 凹模在下,板料在上
    i_punch = solver.add_tool(punch.points3d, punch.tris, up=-1.0)   # 凸模在上
    ms_info = solver.apply_nodal_mass_scaling(dt, k_contact=k_contact, verbose=verbose)

    half_gap = 0.5 * spec.material.thickness
    t0 = time.time()
    for s in range(steps):
        a = min(1.0, s / (steps * 0.9))
        d = -stroke * 0.5 * (1.0 - np.cos(np.pi * a))
        v = np.array([0.0, 0.0, 0.0], np.float32)
        solver.tools[i_punch]["vel"] = v
        solver.move_tool(i_punch, np.array([0.0, 0.0, d], np.float32))
        solver.step(dt, k_contact=k_contact, friction=spec.process.friction, gap=half_gap)
        if verbose and (s + 1) % max(1, steps // 10) == 0:
            th = solver.thickness.numpy()
            print(f"  step {s + 1:6d}/{steps}  行程 {-d:6.3f}/{stroke:.3f} mm  "
                  f"最小料厚 {np.nanmin(th):.4f}  用时 {time.time() - t0:5.1f}s", flush=True)
    wp.synchronize()
    wall = time.time() - t0

    st = solver.state()
    wpf = st["thinning_pct"]
    af = reference["element_thinning_pct"]

    bn = set(blank.boundary_nodes().tolist())
    edge = np.array([any(int(n) in bn for n in t) for t in blank.tris])
    interior = ~edge & np.isfinite(wpf)
    ok = np.isfinite(wpf)

    res = {
        "solver": "warp (wpform)",
        "wall_seconds": round(wall, 1),
        "steps": steps,
        "min_thickness_mm": float(np.nanmin(st["thickness"])),
        "max_thinning_pct": float(np.nanmax(wpf)),
        "p99_thinning_pct": float(np.nanpercentile(wpf[ok], 99)),
        "p995_thinning_pct": float(np.nanpercentile(wpf[ok], 99.5)),
        "interior_max_thinning_pct": float(np.nanmax(wpf[interior])),
        "max_eqps": float(np.nanmax(st["eqps"])),
        "vs_autoform": {
            "autoform_p99": round(float(np.percentile(af, 99)), 2),
            "autoform_p995": round(float(np.percentile(af, 99.5)), 2),
            "autoform_max": round(float(af.max()), 2),
            "pearson_r": round(float(np.corrcoef(af[ok], wpf[ok])[0, 1]), 3),
            "pearson_r_interior": round(float(np.corrcoef(af[interior], wpf[interior])[0, 1]), 3),
            "mean_abs_diff_pct": round(float(np.abs(af[ok] - wpf[ok]).mean()), 3),
        },
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.save(Path(out).with_suffix(".thinning.npy"), wpf.astype(np.float32))
        Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    if verbose:
        print("\n" + json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--dt", type=float, default=1.0e-4)
    ap.add_argument("--clearance", type=float, default=1.15)
    ap.add_argument("--k-contact", type=float, default=1.5e5)
    ap.add_argument("--k-bend", type=float, default=0.0)
    ap.add_argument("--mass-scale", type=float, default=15.0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "wpform_s7.json"))
    a = ap.parse_args()
    run(clearance=a.clearance, steps=a.steps, dt=a.dt, mass_scale=a.mass_scale,
        k_contact=a.k_contact, k_bend=a.k_bend, out=a.out)

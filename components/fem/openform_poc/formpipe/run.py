# -*- coding: utf-8 -*-
"""端到端跑一个 spec:校验 -> 生成 deck -> 求解 -> 结果 JSON。

用法:
    uv run --with numpy,scipy python -m formpipe.run \
        --quicklink sample7_run/quicklink --out runs/s7 [--clearance 1.15] [--friction 0.0375]
    uv run ... python -m formpipe.run --quicklink ... --dry-run   # 只校验,不求解
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import openradioss
from .quicklink import load
from .spec import ProcessSpec, SpecError


def _jsonable(d: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) and v.size < 50 else
                None if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.floating,)) else
                int(v) if isinstance(v, (np.integer,)) else v)
            for k, v in d.items()}


def solve(spec, workdir: str | Path, job: str = "FORM", verbose: bool = True) -> dict:
    """校验 -> deck -> 求解 -> 解析。抛 SpecError 表示模型本身有问题。"""
    workdir = Path(workdir)
    warns = spec.validate(strict=True)
    if verbose:
        print(spec.summary())
        for w in warns:
            print(f"  ! 警告: {w}")
    info = openradioss.write_deck(spec, workdir, job=job)
    t0 = time.time()
    logs = openradioss.run(workdir, job=job)
    wall = time.time() - t0
    res = openradioss.parse(workdir, spec, job=job)
    res["solver"] = {"backend": "openradioss", "wall_seconds": round(wall, 1), **logs}
    res["deck"] = {k: str(v) for k, v in info.items()}
    return res


def compare_reference(res: dict, reference: dict) -> dict:
    """与 AutoForm 逐单元对比(连接关系相同 -> 直接按单元号对齐)。"""
    af = reference["element_thinning_pct"]
    orr = res["element_thinning_pct"]
    ok = np.isfinite(orr)
    interior = ~res["edge_element_mask"] & ok
    return {
        "autoform_min_thickness_mm": reference["min_thickness_mm"],
        "autoform_max_thinning_pct": round(reference["max_thinning_pct"], 2),
        "autoform_p99_thinning_pct": round(float(np.percentile(af[ok], 99)), 2),
        "autoform_p995_thinning_pct": round(float(np.percentile(af[ok], 99.5)), 2),
        "pearson_r": round(float(np.corrcoef(af[ok], orr[ok])[0, 1]), 3),
        "pearson_r_interior": round(float(np.corrcoef(af[interior], orr[interior])[0, 1]), 3),
        "mean_abs_diff_pct": round(float(np.abs(af[ok] - orr[ok]).mean()), 3),
        "verdict_match": bool((reference["max_thinning_pct"] > 30) == (not res["thinning_ok"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="formpipe: 冲压成形仿真端到端")
    ap.add_argument("--quicklink", required=True, help="QuickLink 解包目录")
    ap.add_argument("--out", required=True, help="工作目录")
    ap.add_argument("--clearance", type=float, default=1.15, help="凸凹模间隙比 (×料厚)")
    ap.add_argument("--friction", type=float, default=None, help="覆盖摩擦系数")
    ap.add_argument("--form-time", type=float, default=12.0, help="成形时间 ms")
    ap.add_argument("--dry-run", action="store_true", help="只校验,不求解")
    a = ap.parse_args()

    proc = ProcessSpec(clearance_ratio=a.clearance, form_time_ms=a.form_time)
    if a.friction is not None:
        proc.friction = a.friction
    spec, reference = load(a.quicklink, process=proc)

    if a.dry_run:
        errs, warns = spec.check_all()
        print(spec.summary())
        print(f"\n校验:{len(errs)} 项错误,{len(warns)} 项警告")
        for e in errs:
            print("  ✗", e)
        for w in warns:
            print("  !", w)
        return

    try:
        res = solve(spec, a.out)
    except SpecError as e:
        print(f"\n模型校验未通过,已在求解前拦下:\n{e}")
        raise SystemExit(1)

    cmp = compare_reference(res, reference)
    out = {"spec": spec.name, "result": _jsonable(res), "vs_autoform": cmp}
    Path(a.out, "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + json.dumps({"result": {k: v for k, v in out["result"].items()
                                        if k not in ("deck",)},
                             "vs_autoform": cmp}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

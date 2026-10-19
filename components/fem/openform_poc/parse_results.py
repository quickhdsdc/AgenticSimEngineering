# -*- coding: utf-8 -*-
"""解析 OpenRadioss 冲压仿真结果 → 料厚/减薄/塑性应变/回弹判定。

流程:
  1. 用 anim_to_vtk_win64.exe 把 ANIM 文件(DBEND_44A0xx)转成 legacy VTK
  2. meshio 读取 VTK, 取壳单元厚度(THICK)/等效塑性应变(EPSP)/位移(DISP)
  3. 按判据输出: 最小料厚 >= 初始料厚 x 70% (即减薄 <= 30%)
输出 JSON, 供 agent_system 的 rule_engine 消费。

用法: uv run --with meshio,numpy python parse_results.py <run_dir> <t0_mm>
"""
import json
import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np

OR_EXEC = Path(__file__).parent / "tools" / "OpenRadioss" / "exec"
ANIM_TO_VTK = OR_EXEC / "anim_to_vtk_win64.exe"


def anim_files(run_dir: Path):
    """按序号排好的 ANIM 文件(名字形如 <JOB>A001, A002...)。"""
    files = [p for p in run_dir.iterdir() if p.suffix == "" and "A" in p.name[-4:]
             and p.name[-3:].isdigit()]
    return sorted(files, key=lambda p: p.name[-3:])


def to_vtk(anim: Path) -> Path:
    vtk = anim.with_suffix(".vtk")
    if not vtk.exists():
        out = subprocess.run([str(ANIM_TO_VTK), str(anim)],
                             capture_output=True, text=True, check=True)
        vtk.write_text(out.stdout)
    return vtk


def find_field(mesh, *keywords):
    """在 cell_data 里按关键词模糊找字段名。"""
    for name in mesh.cell_data:
        low = name.lower()
        if all(k in low for k in keywords):
            return name
    return None


def blank_selection(mesh, t0: float):
    """自动识别坯料 PART: 壳(quad)单元里中位厚度最接近初始料厚 t0 的 PART。

    返回 (quad_block_idx, 坯料单元掩码, 坯料节点索引)。
    """
    qi = next(i for i, b in enumerate(mesh.cells) if b.type == "quad")
    pid = np.ravel(mesh.cell_data["PART_ID"][qi])
    thick = np.ravel(mesh.cell_data[find_field(mesh, "2delem", "thic")][qi])
    parts = np.unique(pid)
    blank_pid = min(parts, key=lambda p: abs(np.median(thick[pid == p]) - t0))
    cell_mask = pid == blank_pid
    node_idx = np.unique(mesh.cells[qi].data[cell_mask])
    return qi, cell_mask, node_idx, int(blank_pid)


def analyze(run_dir: Path, t0: float, stamp_end_state: int | None = None) -> dict:
    """stamp_end_state: 冲压结束的 ANIM 序号(1-based), 用于回弹量计算;
    None 则用倒数第二帧近似。"""
    anims = anim_files(run_dir)
    if not anims:
        raise SystemExit(f"no ANIM files in {run_dir}")
    mesh = meshio.read(to_vtk(anims[-1]))
    qi, cell_mask, node_idx, blank_pid = blank_selection(mesh, t0)

    thick_name = find_field(mesh, "2delem", "thic")
    epsp_name = find_field(mesh, "2delem", "pla") or find_field(mesh, "epsp")
    result = {
        "run_dir": str(run_dir),
        "anim_last": anims[-1].name,
        "n_states": len(anims),
        "initial_thickness_mm": t0,
        "blank_part_id": blank_pid,
        "n_blank_shells": int(cell_mask.sum()),
    }

    thick = np.ravel(mesh.cell_data[thick_name][qi])[cell_mask]
    thinning = (t0 - thick) / t0 * 100.0
    result.update({
        "min_thickness_mm": float(thick.min()),
        "max_thinning_pct": float(thinning.max()),
        "thinning_ok": bool(thick.min() >= 0.7 * t0),  # 判据: 减薄<=30%
    })
    if epsp_name:
        epsp = np.ravel(mesh.cell_data[epsp_name][qi])[cell_mask]
        result["max_plastic_strain"] = float(epsp[np.isfinite(epsp)].max())

    # 回弹: 冲压结束帧 → 终态帧, 只看坯料节点
    ref_anim = anims[stamp_end_state - 1] if stamp_end_state else anims[-2]
    ref = meshio.read(to_vtk(ref_anim))
    if ref.points.shape == mesh.points.shape:
        d = np.linalg.norm(mesh.points[node_idx] - ref.points[node_idx], axis=1)
        result.update({
            "springback_ref_state": ref_anim.name,
            "springback_max_disp_mm": float(d.max()),
            "springback_mean_disp_mm": float(d.mean()),
        })
    return result


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.74
    stamp_end = int(sys.argv[3]) if len(sys.argv) > 3 else None
    res = analyze(run_dir, t0, stamp_end)
    print(json.dumps(res, indent=2, ensure_ascii=False))

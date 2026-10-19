# -*- coding: utf-8 -*-
"""OpenRadioss 后端:StampingSpec -> deck -> 求解 -> 结果。

deck 是固定列宽的纯文本卡片,所有数值字段右对齐 20 列、整数字段右对齐 10 列。
把这层隔离在后端里,将来换 Newton/warp.fem 只需实现同样的三个函数:
    write_deck(spec, workdir) / run(workdir) / parse(workdir, spec)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from .spec import StampingSpec

OR_ROOT = Path(__file__).resolve().parent.parent / "tools" / "OpenRadioss"

# deck 里的 ID 分段:坯料 1..N,凹模 10001..,凸模 20001..
BLANK_BASE, DIE_BASE, PUNCH_BASE = 0, 10000, 20000
DIE_MASTER, PUNCH_MASTER = 90001, 90002


def _i(v, w: int = 10) -> str:
    return str(int(v)).rjust(w)


def _f(v, w: int = 20) -> str:
    return f"{float(v):.9g}".rjust(w)


def write_deck(spec: StampingSpec, workdir: str | Path, job: str = "FORM") -> dict:
    """生成 starter(_0000.rad)与 engine(_0001.rad)。返回文件路径。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    m, p = spec.material, spec.process
    blank, die, punch = spec.blank, spec.die.mesh, spec.punch.mesh
    N, NE = blank.n_nodes, blank.n_elems
    L: list[str] = []

    def card(*lines):
        L.extend(lines)

    card("#RADIOSS STARTER",
         "/BEGIN",
         job,
         "       140         0",
         "                   g                  mm                  ms",
         "                   g                  mm                  ms",
         "/TITLE",
         f"{spec.name} crash-form (formpipe)",
         "/ANALY",
         "         0         0         0",
         "/DEF_SHELL",
         "         0         0         0         0         0                             0         0",
         "/IOFLAG",
         "         0                   0         0         0         0         0",
         "/SPMD",
         "         0         0                   0                   4")

    # 材料:板料 Hill 各向异性 + 表格硬化;工具弹性(实际由 RBODY 变刚体)
    card("/MAT/HILL_TAB/1",
         m.name[:80] or "BLANK_MAT",
         _f(m.density) + _f(m.density),
         _f(m.youngs) + _f(m.poisson),
         _i(0) + " " * 10 + _f(0) + _f(0),
         _f(m.r00) + _f(m.r45) + _f(m.r90) + _f(0),
         _f(0) + _f(0) + _f(0),
         _i(1) + " " * 10 + _f(1.0) + _f(0))
    card("/MAT/ELAST/2",
         "TOOL_RIGID",
         _f(7.8e-3) + _f(7.8e-3),
         _f(206000) + _f(0.3))

    # 节点:坯料 / 凹模 / 凸模 + 两个刚体主节点
    card("/NODE")
    for base, pts in ((BLANK_BASE, blank.points3d), (DIE_BASE, die.points3d), (PUNCH_BASE, punch.points3d)):
        for i, q in enumerate(pts):
            L.append(_i(base + i + 1) + _f(q[0]) + _f(q[1]) + _f(q[2]))
    for nid, pts in ((DIE_MASTER, die.points3d), (PUNCH_MASTER, punch.points3d)):
        c = pts.mean(axis=0)
        L.append(_i(nid) + _f(c[0]) + _f(c[1]) + _f(c[2]))

    card("/PART/1", "BLANK", _i(1) + _i(1) + _i(0))
    card("/PART/2", "DIE", _i(2) + _i(2) + _i(0))
    card("/PART/3", "PUNCH", _i(3) + _i(2) + _i(0))

    # 板料属性:QEPH 壳(Ishell=24),5 个厚度积分点,厚度更新开启(Ithick=1)
    card("/PROP/SH_ORTH/1",
         "BLANK_PROP",
         _i(24) + _i(0) + _i(2) + _i(0),
         _f(0) + _f(0) + _f(0) + _f(0) + _f(0),
         _i(5) + _i(1) + _f(m.thickness) + _f(0) + " " * 19 + "1" + _i(1),
         _f(1) + _f(0) + _f(0) + _f(0))
    for pid, nm in ((2, "DIE_PROP"), (3, "PUNCH_PROP")):
        card(f"/PROP/SHELL/{pid}",
             nm,
             _i(0) + _i(0) + _i(2) + _i(0),
             _f(0) + _f(0) + _f(0) + _f(0) + _f(0),
             _i(1) + _i(0) + _f(1.0) + _f(0) + " " * 19 + "0" + _i(0))

    # 单元(三角壳)
    for pid, base, tris in ((1, BLANK_BASE, blank.tris), (2, DIE_BASE, die.tris), (3, PUNCH_BASE, punch.tris)):
        card(f"/SH3N/{pid}")
        for e, t in enumerate(tris):
            L.append(_i(base + e + 1) + _i(base + t[0] + 1) + _i(base + t[1] + 1) + _i(base + t[2] + 1))

    card("/GRNOD/PART/6", "BLANK_NODES", _i(1))
    card("/GRNOD/PART/9", "DIE_NODES", _i(2))
    card("/GRNOD/PART/10", "PUNCH_NODES", _i(3))
    card("/GRNOD/NODE/11", "DIE_MASTER", _i(DIE_MASTER))
    card("/GRNOD/NODE/12", "PUNCH_MASTER", _i(PUNCH_MASTER))
    card("/SURF/PART/1", "DIE_SURF", _i(2))
    card("/SURF/PART/2", "PUNCH_SURF", _i(3))

    for rb, (mast, grn) in enumerate(((DIE_MASTER, 9), (PUNCH_MASTER, 10)), start=1):
        card(f"/RBODY/{rb}",
             f"RB_{rb}",
             _i(mast) + _i(0) + _i(0) + _i(0) + _f(0) + _i(grn) + _i(0) + _i(0) + _i(0),
             _f(0) + _f(0) + _f(0),
             _f(0) + _f(0) + _f(0),
             _i(0))

    card("/BCS/1", "DIE_FIX", "   111 111" + _i(0) + _i(11))
    card("/BCS/2", "PUNCH_GUIDE", "   110 111" + _i(0) + _i(12))

    gapmin = 0.5 * m.thickness
    for cid, (surf, nm) in enumerate(((1, "DIE"), (2, "PUNCH")), start=1):
        card(f"/INTER/TYPE7/{cid}",
             f"BLANK_{nm}",
             _i(6) + _i(surf) + _i(0) + _i(0) + _i(0) + " " * 10 + _i(0) + _i(0) + _i(0) + _i(0),
             _f(0) + _f(0) + _f(0),
             _f(0) + _f(0) + _f(0) + _f(0) + _i(0),
             _f(1) + _f(p.friction) + _f(gapmin) + _f(0) + _f(0),
             "       000" + " " * 10 + _i(0) + _f(1) + _f(1) + _f(0),
             _i(0) + _i(0) + _f(0) + _i(0) + _i(0))

    card("/FUNCT/1", "FLOW_CURVE")
    for eps, sig in m.flow_curve:
        L.append(_f(eps) + _f(sig))

    # 凸模位移:余弦 S 曲线(起停平滑,减少冲击),保压段保持到底
    stroke, tf = spec.punch.stroke, p.form_time_ms
    card("/FUNCT/2", "PUNCH_DISP")
    for t in np.linspace(0.0, tf, 33):
        L.append(_f(t) + _f(-stroke * 0.5 * (1 - np.cos(np.pi * t / tf))))
    L.append(_f(tf + p.dwell_ms) + _f(-stroke))
    L.append(_f(1000.0) + _f(-stroke))

    card("/IMPDISP/1",
         "PUNCH_MOVE",
         _i(2) + "         Z" + _i(0) + _i(0) + _i(12) + _i(0) + _i(0),
         _f(0) + _f(1) + _f(0) + _f(tf + p.dwell_ms))
    card("/END")

    starter = workdir / f"{job}_0000.rad"
    starter.write_text("\n".join(L) + "\n", newline="\n")

    t_end = tf + p.dwell_ms
    engine = workdir / f"{job}_0001.rad"
    engine.write_text(
        f"/RUN/{job}/1\n{t_end}\n/VERS/140\n/ANIM/DT\n 0. {t_end / 20:.4g}\n"
        "/ANIM/ELEM/EPSP\n/ANIM/ELEM/THICK\n/ANIM/SHELL/TENS\n/ANIM/VECT/DISP\n"
        f"/TFILE\n.1\n/PRINT/-1000\n/RFILE\n 100000 0 0\n/DT\n0.9 0.0\n"
        f"/DT/NODA/CST\n0.9 {p.dt_min_ms:g}\n/MON/ON\n", newline="\n")
    return {"starter": starter, "engine": engine, "job": job,
            "n_nodes": N, "n_elems": NE}


def _env() -> dict:
    env = dict(os.environ)
    env.update(RAD_CFG_PATH=str(OR_ROOT / "hm_cfg_files"),
               RAD_H3D_PATH=str(OR_ROOT / "extlib" / "h3d" / "lib" / "win64"),
               KMP_STACKSIZE="400m", OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"))
    return env


def run(workdir: str | Path, job: str = "FORM", timeout: int = 3600) -> dict:
    """跑 starter + engine。返回终止状态与耗时。"""
    workdir = Path(workdir)
    exe = OR_ROOT / "exec"
    logs = {}
    for phase, args in (("starter", [str(exe / "starter_win64.exe"), "-i", f"{job}_0000.rad", "-np", "1"]),
                        ("engine", [str(exe / "engine_win64.exe"), "-i", f"{job}_0001.rad"])):
        r = subprocess.run(args, cwd=workdir, env=_env(), capture_output=True,
                           text=True, timeout=timeout)
        out = r.stdout + r.stderr
        (workdir / f"{phase}.log").write_text(out, encoding="utf-8", errors="ignore")
        # starter 正常结束时打印的是 "TERMINATION WITH WARNING"(接触 gap 提示),
        # 只有 "ERROR TERMINATION" 才是真失败
        failed = "ERROR TERMINATION" in out
        logs[phase] = {
            "returncode": r.returncode,
            "ok": not failed,
            "termination": ("ERROR" if failed else
                            "NORMAL" if "NORMAL TERMINATION" in out else "WARNING"),
            "n_warnings": int(next((ln.split()[0] for ln in out.splitlines()
                                    if "WARNING(S)" in ln and ln.split()[0].isdigit()), 0)),
        }
        if failed:
            raise RuntimeError(f"{phase} 求解失败(ERROR TERMINATION),见 {workdir / phase}.log")
    return logs


# ---------- 结果 ----------
def parse_vtk(path: Path) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """legacy ASCII VTK 最小解析器。

    meshio 5.x 读不了 anim_to_vtk 输出(标题行 "vtk output" 被当作数据类型),
    所以自己解析。只取需要的:节点坐标、单元连接、单元标量场。
    """
    lines = path.read_text().splitlines()
    pts = None
    cells: list[np.ndarray] = []
    cdata: dict[str, np.ndarray] = {}
    n_cells, mode, i = 0, None, 0
    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1
            continue
        key = tok[0].upper()
        if key == "POINTS":
            n = int(tok[1]); vals: list[str] = []; i += 1
            while len(vals) < 3 * n:
                vals += lines[i].split(); i += 1
            pts = np.array(vals, float).reshape(n, 3)
            continue
        if key == "CELLS":
            n_cells, total = int(tok[1]), int(tok[2]); vals = []; i += 1
            while len(vals) < total:
                vals += lines[i].split(); i += 1
            flat = np.array(vals, int); p = 0
            for _ in range(n_cells):
                k = flat[p]; cells.append(flat[p + 1:p + 1 + k]); p += k + 1
            continue
        if key in ("CELL_DATA", "POINT_DATA"):
            mode = key; i += 1
            continue
        if key == "SCALARS" and mode == "CELL_DATA":
            nm = tok[1]; i += 2; vals = []
            while len(vals) < n_cells:
                vals += lines[i].split(); i += 1
            cdata[nm] = np.array(vals, float)
            continue
        i += 1
    return pts, cells, cdata


def anim_to_vtk(anim: Path) -> Path:
    vtk = anim.with_suffix(".vtk")
    if not vtk.exists():
        r = subprocess.run([str(OR_ROOT / "exec" / "anim_to_vtk_win64.exe"), str(anim)],
                           capture_output=True, text=True, check=True)
        vtk.write_text(r.stdout)
    return vtk


def parse(workdir: str | Path, spec: StampingSpec, job: str = "FORM") -> dict:
    """最后一帧 -> 逐单元料厚/减薄/塑性应变 + 判据。

    单元号即 deck 里的 SH3N id,与 spec.blank.tris 行号一一对应,
    因此可以直接和 AutoForm 的逐单元结果对齐。
    """
    workdir = Path(workdir)
    NE, t0 = spec.blank.n_elems, spec.material.thickness
    anims = sorted([p for p in workdir.iterdir()
                    if p.name.startswith(f"{job}A") and p.name[-3:].isdigit()],
                   key=lambda p: p.name[-3:])
    if not anims:
        raise FileNotFoundError(f"{workdir} 下没有 ANIM 文件")

    pts, cells, cdata = parse_vtk(anim_to_vtk(anims[-1]))
    eid = cdata["ELEMENT_ID"].astype(int)
    # SH3N 在 VTK 里导出为退化四边形;2 节点单元是刚体连接
    mask = np.array([len(c) >= 3 for c in cells]) & (eid >= 1) & (eid <= NE)

    thickness = np.full(NE, np.nan)
    thickness[eid[mask] - 1] = cdata["2DELEM_Thickness"][mask]
    epsp = np.full(NE, np.nan)
    if "2DELEM_Plastic_Strain" in cdata:
        epsp[eid[mask] - 1] = cdata["2DELEM_Plastic_Strain"][mask]
    thinning = (t0 - thickness) / t0 * 100.0

    # 自由边(孔边/翻边边缘)单元:crash-form 下这些位置会被两模夹住而虚假过度减薄,
    # 判据应基于内部单元或高分位数,而非全局 max
    bnodes = set(spec.blank.boundary_nodes().tolist())
    edge_elem = np.array([any(int(n) in bnodes for n in t) for t in spec.blank.tris])
    interior = ~edge_elem & np.isfinite(thinning)

    # 合模程度:坯料终态节点到凹模型面的距离
    blank_nodes = np.unique(np.concatenate([cells[k] for k in np.where(mask)[0]]))
    try:
        from scipy.spatial import cKDTree
        closure, _ = cKDTree(spec.die.mesh.points3d).query(pts[blank_nodes])
    except ImportError:
        closure = np.array([np.nan])

    valid = np.isfinite(thinning)
    return {
        "n_frames": len(anims),
        "n_elements": int(valid.sum()),
        "element_thickness": thickness,
        "element_thinning_pct": thinning,
        "element_plastic_strain": epsp,
        "edge_element_mask": edge_elem,
        "min_thickness_mm": float(np.nanmin(thickness)),
        "max_thinning_pct": float(np.nanmax(thinning)),
        "p99_thinning_pct": float(np.nanpercentile(thinning[valid], 99)),
        "p995_thinning_pct": float(np.nanpercentile(thinning[valid], 99.5)),
        "interior_max_thinning_pct": float(np.nanmax(thinning[interior])),
        "max_plastic_strain": float(np.nanmax(epsp)) if np.isfinite(epsp).any() else None,
        "closure_mean_mm": float(np.mean(closure)),
        "closure_p95_mm": float(np.percentile(closure, 95)),
        # 判据:内部单元减薄 ≤30%(避开自由边伪影)
        "thinning_ok": bool(np.nanmax(thinning[interior]) <= 30.0),
    }

# -*- coding: utf-8 -*-
"""AutoForm QuickLink 导出包 -> StampingSpec。

QuickLink zip 里能拿到的东西(无需 AutoForm GUI/授权):
  *_formcheck_results_Sheet_NodeData.csv    成形态坐标 + 每节点展开平面坐标
  *_formcheck_results_Sheet_ElementData.csv 连接关系 + GEOM_THINNING/THICKNESS/主次应变
  *.mtb                                     材料卡(big-endian double 二进制)
  *_v100.xml                                料厚、摩擦系数、冲压方向

注意符号约定:AutoForm 的 GEOM_THINNING = (t - t0)/t0,负值代表减薄;
本管线内部统一用"减薄率百分比"(正值 = 变薄)。
"""
from __future__ import annotations

import csv
import re
import struct
import zipfile
from pathlib import Path

import numpy as np

from .geometry import offset_surface, register_2d
from .spec import MaterialSpec, MeshSpec, ProcessSpec, StampingSpec, ToolSpec


# ---------- 低层读取 ----------
def _read_csv(path: Path, ncol: int) -> np.ndarray:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row and row[0] != "":
                rows.append([float(x) for x in row[:ncol]])
    return np.array(rows)


def decode_mtb(path: Path) -> dict:
    """从 .mtb 里抽流动曲线与弹性/各向异性参数。

    格式未公开,但数值是 big-endian double:文件里存在一段
    (eps, sigma) 交替的长游程即流动曲线;标签(UTF-32)前 64 字节窗口内
    的合理范围 double 即对应的 r 值。
    """
    data = path.read_bytes()

    def runs(pred, min_len=8):
        out, run, start = [], [], 0
        for off in range(0, len(data) - 8, 8):
            (x,) = struct.unpack_from(">d", data, off)
            if x == x and pred(x):
                if not run:
                    start = off
                run.append(x)
            else:
                if len(run) >= min_len:
                    out.append((start, run))
                run = []
        if len(run) >= min_len:
            out.append((start, run))
        return out

    flow = []
    flow_end = 0
    for run_start, run in runs(lambda x: x == 0.0 or 1e-9 < abs(x) < 1e9, min_len=20):
        pairs = [(run[i], run[i + 1]) for i in range(0, len(run) - 1, 2)]
        pairs = [(e, s) for e, s in pairs if 0.0 <= e <= 5.0 and 1.0 < s < 5000.0]
        # 流动曲线特征:应变递增、应力递增、首点 eps=0
        if len(pairs) >= 10 and pairs[0][0] == 0.0:
            eps = [p[0] for p in pairs]
            sig = [p[1] for p in pairs]
            if all(np.diff(eps) > 0) and all(np.diff(sig) >= 0) and len(pairs) > len(flow):
                flow = pairs
                flow_end = run_start + 8 * len(run)
    if not flow:
        raise ValueError(f"{path.name}: 未能定位流动曲线")

    # r 值:紧跟流动曲线之后的主数据块里,3 个连续 double(前面紧邻一个 0)。
    # 注意 double 不一定 8 字节对齐(实测 1050_O 在 offset%8==4),必须逐字节扫描。
    # 不要用 "r0/r45/r90" 标签附近的数值——那里存的是公差带上下限,不是本征值。
    def r_triple(after: int) -> tuple[float, float, float] | None:
        for off in range(after, len(data) - 40):
            try:
                prev, = struct.unpack_from(">d", data, off - 8)
                a, b, c = (struct.unpack_from(">d", data, off + 8 * k)[0] for k in range(3))
                nxt, = struct.unpack_from(">d", data, off + 24)
            except struct.error:
                continue
            vals = (a, b, c)
            if prev != 0.0 or any(v != v for v in vals):
                continue
            if all(0.3 <= v <= 3.5 for v in vals) and nxt == nxt and 0.0 < abs(nxt) < 1e4:
                return float(a), float(b), float(c)
        return None

    # 弹性常数在文件头部的定长记录里:相邻的 (E, nu)
    elastic = None
    for off in range(0, len(data) - 16):
        try:
            E, = struct.unpack_from(">d", data, off)
            nu, = struct.unpack_from(">d", data, off + 8)
        except struct.error:
            continue
        if E == E and nu == nu and 1e4 < E < 5e5 and 0.2 < nu < 0.45:
            elastic = (float(E), float(nu))
            break

    rs = r_triple(flow_end) or (1.0, 1.0, 1.0)

    return {
        "flow_curve": flow,
        "youngs": elastic[0] if elastic else None,
        "poisson": elastic[1] if elastic else None,
        "r00": rs[0], "r45": rs[1], "r90": rs[2],
    }


def parse_quicklink_xml(path: Path) -> dict:
    """料厚、摩擦系数、材料名(简单正则,避免引入 xml 命名空间的麻烦)。"""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: dict = {}
    m = re.search(r"<Thickness>([\d.eE+-]+)</Thickness>", txt)
    if m:
        out["thickness"] = float(m.group(1))
    m = re.search(r"<Type>FrictionCoefficient</Type>\s*<Value>([\d.eE+-]+)</Value>", txt)
    if m:
        out["friction"] = float(m.group(1))
    m = re.search(r"<Material>\s*<MaterialId>\d+</MaterialId>\s*<Name>([^<]+)</Name>", txt)
    if m:
        out["material_name"] = m.group(1).strip()
    return out


# ---------- 高层:zip -> spec ----------
def extract(zip_path: str | Path, dest: str | Path) -> Path:
    """解开 QuickLink zip(跳过巨大的 .afd 工程文件)。"""
    zip_path, dest = Path(zip_path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.lower().endswith(".afd"):
                continue
            if not (dest / Path(n).name).exists():
                z.extract(n, dest)
    return dest


def load(quicklink_dir: str | Path, process: ProcessSpec | None = None,
         name: str | None = None, density: float = 2.7e-3) -> tuple[StampingSpec, dict]:
    """QuickLink 目录 -> (StampingSpec, AutoForm 参考结果)。

    坐标系:AutoForm 零件系 (X,Y,Z) -> 压机系 (X, -Z, Y),使冲压方向为 -z。
    """
    d = Path(quicklink_dir)
    node_csv = next(d.glob("*NodeData.csv"))
    elem_csv = next(d.glob("*ElementData.csv"))
    xml = next(d.glob("*_v100.xml"))
    mtb = next(d.glob("*.mtb"))

    meta = parse_quicklink_xml(xml)
    mat_raw = decode_mtb(mtb)
    t0 = meta.get("thickness", 1.0)
    process = process or ProcessSpec(friction=meta.get("friction", 0.1))
    if "friction" in meta and process.friction == 0.1:
        process.friction = meta["friction"]

    nodes = _read_csv(node_csv, 6)
    elems = _read_csv(elem_csv, 10)
    tris = elems[:, 1:4].astype(int)

    # 成形件(= 凹模型面),换到压机系
    part_xyz = nodes[:, 1:4]
    formed = np.stack([part_xyz[:, 0], -part_xyz[:, 2], part_xyz[:, 1]], axis=1)

    # 凸模 = 型面沿法向偏置 clearance
    clearance = process.clearance_ratio * t0
    punch_pts, offset_info = offset_surface(formed, tris, clearance, direction=-1)

    # 坯料 = 展开平面坐标配准到成形件 xy 投影
    blank2d, reg_info = register_2d(nodes[:, 4:6], formed[:, :2])
    z0 = formed[:, 2].max() + process.blank_gap
    blank = np.column_stack([blank2d, np.full(len(blank2d), z0)])

    # 凸模抬到坯料上方,行程 = 抬升量(保证合模)
    lift = (z0 + process.blank_gap) - punch_pts[:, 2].min()
    punch_pts = punch_pts + np.array([0.0, 0.0, lift])

    flow = list(mat_raw["flow_curve"])
    if flow[-1][0] < 2.0:      # 补线性外插点,避免求解器把大应变区按常数应力处理
        (e1, s1), (e2, s2) = flow[-2], flow[-1]
        slope = (s2 - s1) / (e2 - e1)
        flow.append((2.0, s2 + slope * (2.0 - e2)))

    material = MaterialSpec(
        name=meta.get("material_name", mtb.stem),
        thickness=t0,
        youngs=mat_raw["youngs"] or 70000.0,
        poisson=mat_raw["poisson"] or 0.33,
        density=density,
        flow_curve=flow,
        r00=mat_raw["r00"] or 1.0,
        r45=mat_raw["r45"] or 1.0,
        r90=mat_raw["r90"] or 1.0,
    )

    spec = StampingSpec(
        name=name or d.parent.name,
        material=material,
        blank=MeshSpec(blank, tris, "blank"),
        tools=[
            ToolSpec("die", MeshSpec(formed, tris, "die"), motion="fixed"),
            ToolSpec("punch", MeshSpec(punch_pts, tris, "punch"),
                     motion="translate_z", stroke=lift),
        ],
        process=process,
    )

    # AutoForm 参考结果(逐单元,与 blank/die 同一连接关系 -> 可直接对比)
    af_thickness = elems[:, 8]
    reference = {
        "source": "AutoForm formcheck (QuickLink)",
        "element_thickness": af_thickness,
        "element_thinning_pct": (t0 - af_thickness) / t0 * 100.0,
        "min_thickness_mm": float(af_thickness.min()),
        "max_thinning_pct": float(((t0 - af_thickness) / t0 * 100.0).max()),
        "major_strain": elems[:, 4],
        "minor_strain": elems[:, 5],
        "offset_info": offset_info,
        "registration_info": reg_info,
    }
    spec.reference = {k: v for k, v in reference.items() if not isinstance(v, np.ndarray)}
    return spec, reference

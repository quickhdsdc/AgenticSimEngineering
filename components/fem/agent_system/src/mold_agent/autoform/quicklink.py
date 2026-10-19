"""解析 AutoForm QuickLink 导出包 → SimResult。

QuickLink "Export to QuickLink" 生成一个 .zip,内含(实测 R10):
- `*_formcheck_results_Sheet_ElementData.csv`  —— 元素级结果(⭐ 主要数据源)
    列: Element Idx, Node Idx 1/2/3, GEOM_MAJOR, GEOM_MINOR, GEOM_STRAIN,
        GEOM_THINNING, GEOM_THICKNESS, GEOM_FORMABILITY
- `*_formcheck_results_Sheet_NodeData.csv`     —— 节点坐标(变形后 XYZ + 初始 XY)
- `*_formcheck_BlankOutline.igs`               —— 料片轮廓
- `*_part.igs` / `*_materialZone0.igs`         —— 几何
- `control.xml` / `*_v100.xml`                 —— 清单/元数据
- `1050_O.mtb` / `*.afd`                       —— 材料 / 工程文件

GEOM_THINNING 约定: 负=减薄, 正=增厚。 GEOM_THINNING = (t - t0)/t0,
故可反推原始料厚 t0 = GEOM_THICKNESS / (1 + GEOM_THINNING)(各元素应一致)。

GEOM_FORMABILITY 分区码(实测,按料厚相关性推断,对应 Crashform 图例四区):
    0 = Potential Splits 开裂(最薄) | 1 = Safe 安全
    2 = Insufficient Stretch 拉伸不足 | 3 = Potential Wrinkles 起皱
"""
from __future__ import annotations

import csv
import io
import re
import statistics
import zipfile
from pathlib import Path

from ..models import FldStatus, SimResult

# 分区码 → 语义
FORMABILITY_CODES = {
    0: "splits",               # 开裂 (红)
    1: "safe",                 # 安全 (绿)
    2: "insufficient_stretch", # 拉伸不足 (灰)
    3: "wrinkles",             # 起皱 (蓝)
}
SPLIT_CODE = 0


def parse_quicklink_meta(package: str | Path) -> dict:
    """从 QuickLink 包的文件名提取真实元信息(材料/零件号),不猜测、不硬编。

    实测包内含材料卡 `<牌号>.mtb`(如 1050_O.mtb)与文件名前缀(零件号 + 材料串)。
    取不到的字段返回空字符串(诚实留空,不编造)。
    """
    package = Path(package)
    names: list[str] = []
    if package.is_dir():
        names = [p.name for p in package.rglob("*")]
    else:
        with zipfile.ZipFile(package) as zf:
            names = zf.namelist()

    material_file = next((n for n in names if n.lower().endswith(".mtb")), "")
    material = Path(material_file).stem if material_file else ""   # e.g. "1050_O"

    # 零件号:包/文件名前缀第一段(字母数字),如 "0220M000ND_stamping-Al 1050 Stamping" → 0220M000ND
    stem = package.stem
    m = re.match(r"([A-Za-z0-9]+)", stem)
    part_number = m.group(1) if m else ""
    return {"material_file": material_file, "material": material, "part_number": part_number}


def _read_member(package: Path, needle: str) -> str | None:
    """从 QuickLink 包(.zip 或目录)读出文件名含 needle 的 CSV 文本。"""
    if package.is_dir():
        for p in package.rglob("*.csv"):
            if needle in p.name:
                return p.read_text(encoding="utf-8", errors="replace")
        return None
    with zipfile.ZipFile(package) as zf:
        for name in zf.namelist():
            if name.endswith(".csv") and needle in name:
                return zf.read(name).decode("utf-8", errors="replace")
    return None


def _blank_bbox(package: Path) -> dict | None:
    """从 NodeData.csv 的初始平面坐标算料片外接框(展开料片长×宽)。"""
    text = _read_member(package, "NodeData")
    if not text:
        return None
    xs, ys = [], []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            xs.append(float(r["Initial XCoord"]))
            ys.append(float(r["Initial YCoord"]))
        except (KeyError, ValueError):
            continue
    if not xs:
        return None
    length = round(max(max(xs) - min(xs), max(ys) - min(ys)), 1)
    width = round(min(max(xs) - min(xs), max(ys) - min(ys)), 1)
    return {"length_mm": length, "width_mm": width, "size": f"{length}×{width}"}


def parse_quicklink(package: str | Path) -> SimResult:
    """解析 QuickLink 包(.zip 或已解压目录)→ SimResult。"""
    package = Path(package)
    text = _read_member(package, "ElementData")
    if text is None:
        raise FileNotFoundError("QuickLink 包内未找到 *ElementData.csv")
    rows = list(csv.DictReader(io.StringIO(text)))
    thickness, thinning, major, minor, form = [], [], [], [], []
    t0_est = []
    for r in rows:
        try:
            t = float(r["GEOM_THICKNESS"])
            th = float(r["GEOM_THINNING"])
        except (KeyError, ValueError):
            continue
        thickness.append(t)
        thinning.append(th)
        major.append(float(r.get("GEOM_MAJOR") or "nan"))
        minor.append(float(r.get("GEOM_MINOR") or "nan"))
        if r.get("GEOM_FORMABILITY"):
            form.append(int(float(r["GEOM_FORMABILITY"])))
        if 1 + th != 0:
            t0_est.append(t / (1 + th))

    if not thickness:
        raise ValueError("ElementData.csv 中无有效结果行")

    t0 = round(statistics.median(t0_est), 4) if t0_est else round(max(thickness), 4)
    min_thk = min(thickness)
    max_thinning = max(0.0, (t0 - min_thk) / t0)          # 减薄比(正)
    n_splits = sum(1 for c in form if c == SPLIT_CODE)

    # FLD 状态: 有开裂分区 → red;否则按减薄给 yellow/green
    if n_splits > 0:
        fld = FldStatus.red
        formability = "crack"
    elif max_thinning > 0.28:
        fld = FldStatus.yellow
        formability = "ok"
    else:
        fld = FldStatus.green
        formability = "ok"

    zone_counts = {FORMABILITY_CODES.get(c, str(c)): form.count(c) for c in set(form)}
    blank = _blank_bbox(package)

    return SimResult(
        min_thickness_mm=round(min_thk, 4),
        thinning_ratio=round(max_thinning, 4),
        fld_status=fld,
        formability_status=formability,
        max_springback_mm=0.0,           # NodeData 里有坐标,回弹后续从多阶段结果算
        n_steps=1,
        raw={
            "source": str(package),
            "t0_mm": t0,
            "n_elements": len(thickness),
            "max_thickness_mm": round(max(thickness), 4),
            "n_split_elements": n_splits,
            "zone_counts": zone_counts,
            "fld_major_max": round(max(x for x in major if x == x), 4) if major else None,
            "blank": blank,
        },
    )

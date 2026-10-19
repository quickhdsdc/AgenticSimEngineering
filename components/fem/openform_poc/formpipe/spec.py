# -*- coding: utf-8 -*-
"""冲压成形问题的声明层 + 校验层(纯 Python/numpy,不碰求解器)。

分层动机与 mbs_isaac 相同:建错了却照样能跑出结果,是最贵的错误。
所有能在几毫秒内查出来的错误,都不应该等 100 秒的求解跑完再靠结果异常发现。

单位约定统一为 g / mm / ms:
  长度 mm, 时间 ms, 质量 g  ->  应力 MPa, 密度 g/mm^3, 速度 mm/ms = m/s
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


class SpecError(ValueError):
    """校验不通过。消息里列出全部问题,不是遇到第一个就抛。"""


@dataclass
class MaterialSpec:
    """板料材料。flow_curve 是 (等效塑性应变, 流动应力 MPa) 的单调曲线。"""

    name: str
    thickness: float                      # mm,初始料厚 t0
    youngs: float                         # MPa
    poisson: float
    density: float                        # g/mm^3 (钢 7.8e-3, 铝 2.7e-3)
    flow_curve: list[tuple[float, float]]
    r00: float = 1.0
    r45: float = 1.0
    r90: float = 1.0

    @property
    def yield_stress(self) -> float:
        return self.flow_curve[0][1]

    @property
    def r_mean(self) -> float:
        return (self.r00 + 2.0 * self.r45 + self.r90) / 4.0

    def scaled(self, factor: float) -> "MaterialSpec":
        """整条流动曲线按比例缩放——用于参数扫描里模拟材料强度批次波动。"""
        return MaterialSpec(
            name=f"{self.name}_x{factor:g}",
            thickness=self.thickness,
            youngs=self.youngs,
            poisson=self.poisson,
            density=self.density,
            flow_curve=[(e, s * factor) for e, s in self.flow_curve],
            r00=self.r00, r45=self.r45, r90=self.r90,
        )

    def check(self) -> list[str]:
        errs = []
        if self.thickness <= 0:
            errs.append(f"料厚必须为正: {self.thickness}")
        if self.youngs <= 0:
            errs.append(f"弹性模量必须为正: {self.youngs}")
        if not 0.0 < self.poisson < 0.5:
            errs.append(f"泊松比应在 (0, 0.5): {self.poisson}")
        if self.density <= 0:
            errs.append(f"密度必须为正: {self.density}")
        if self.density > 1.0:
            errs.append(f"密度 {self.density} 看起来是 kg/m^3 而非 g/mm^3(钢应为 7.8e-3)")
        if len(self.flow_curve) < 2:
            errs.append("流动曲线至少需要 2 个点")
        else:
            eps = np.array([p[0] for p in self.flow_curve])
            sig = np.array([p[1] for p in self.flow_curve])
            if eps[0] != 0.0:
                errs.append(f"流动曲线首点应为 eps=0(屈服点),实为 {eps[0]}")
            if np.any(np.diff(eps) <= 0):
                errs.append("流动曲线的应变必须严格递增")
            if np.any(np.diff(sig) < 0):
                errs.append("流动曲线出现软化段(应力递减),显式求解会不稳定")
            if sig[0] <= 0:
                errs.append(f"屈服应力必须为正: {sig[0]}")
            if eps[-1] < 0.5:
                errs.append(f"流动曲线只到 eps={eps[-1]:.2f};成形区常超此值,"
                            "外插会被求解器按常数处理 -> 需补外插点")
        for nm, r in (("r00", self.r00), ("r45", self.r45), ("r90", self.r90)):
            if r <= 0:
                errs.append(f"{nm} 必须为正: {r}")
        return errs


@dataclass
class MeshSpec:
    """三角网格。points 为 (N,3) 或 (N,2)(平面坯料);tris 为 (NE,3) 0-based。"""

    points: np.ndarray
    tris: np.ndarray
    name: str = "mesh"

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_elems(self) -> int:
        return len(self.tris)

    def areas(self) -> np.ndarray:
        p = self.points3d[self.tris]
        return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)

    @property
    def points3d(self) -> np.ndarray:
        if self.points.shape[1] == 3:
            return self.points
        return np.column_stack([self.points, np.zeros(len(self.points))])

    def edge_lengths(self) -> np.ndarray:
        p = self.points3d[self.tris]
        return np.concatenate([np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3)])

    def boundary_nodes(self) -> np.ndarray:
        """只被一个单元用到的边上的节点(自由边/孔边)。"""
        from collections import Counter
        cnt = Counter()
        for a, b, c in self.tris:
            for e in ((a, b), (b, c), (c, a)):
                cnt[tuple(sorted(e))] += 1
        bn = {n for e, k in cnt.items() if k == 1 for n in e}
        return np.array(sorted(bn), dtype=int)

    def check(self) -> list[str]:
        errs = []
        if self.points.ndim != 2 or self.points.shape[1] not in (2, 3):
            errs.append(f"{self.name}: points 形状应为 (N,2) 或 (N,3),实为 {self.points.shape}")
            return errs
        if self.tris.ndim != 2 or self.tris.shape[1] != 3:
            errs.append(f"{self.name}: tris 形状应为 (NE,3),实为 {self.tris.shape}")
            return errs
        if self.tris.min() < 0 or self.tris.max() >= self.n_nodes:
            errs.append(f"{self.name}: 单元引用了越界节点 "
                        f"[{self.tris.min()}, {self.tris.max()}] vs N={self.n_nodes}")
            return errs
        if not np.isfinite(self.points).all():
            errs.append(f"{self.name}: 节点坐标含 NaN/Inf")
        degen = self.areas() < 1e-9
        if degen.any():
            errs.append(f"{self.name}: {degen.sum()} 个退化单元(面积≈0)")
        unused = self.n_nodes - len(np.unique(self.tris))
        if unused > 0:
            errs.append(f"{self.name}: {unused} 个孤立节点(未被任何单元引用)")
        return errs


ToolMotion = Literal["fixed", "translate_z"]


@dataclass
class ToolSpec:
    """刚性工具(凸模/凹模/压边圈)。"""

    name: str
    mesh: MeshSpec
    motion: ToolMotion = "fixed"
    stroke: float = 0.0        # mm,translate_z 时沿 -z 的总行程

    def check(self) -> list[str]:
        errs = [f"{self.name}: {e}" for e in self.mesh.check()]
        if self.motion == "translate_z" and self.stroke <= 0:
            errs.append(f"{self.name}: 运动工具的行程必须为正,实为 {self.stroke}")
        if self.motion == "fixed" and self.stroke != 0:
            errs.append(f"{self.name}: 固定工具不应有行程 {self.stroke}")
        return errs


@dataclass
class ProcessSpec:
    """工艺参数。这些是参数扫描里真正会变的量。"""

    friction: float = 0.1
    clearance_ratio: float = 1.15   # 凸模偏置 = ratio × 料厚
    form_time_ms: float = 12.0
    dwell_ms: float = 0.5
    dt_min_ms: float = 1.0e-4       # 质量缩放的最小时间步
    blank_gap: float = 0.8          # 坯料初始离凹模最高点的高度 mm

    def check(self) -> list[str]:
        errs = []
        if not 0.0 <= self.friction <= 0.5:
            errs.append(f"摩擦系数 {self.friction} 超出合理范围 [0, 0.5]")
        if self.clearance_ratio < 1.0:
            errs.append(f"凸凹模间隙比 {self.clearance_ratio} < 1.0:凸模比板料还薄,"
                        "合模末段会全面熨薄板料,减薄率严重虚高")
        if self.clearance_ratio > 1.5:
            errs.append(f"凸凹模间隙比 {self.clearance_ratio} > 1.5:间隙过大,型面贴合不足")
        if self.form_time_ms <= 0:
            errs.append(f"成形时间必须为正: {self.form_time_ms}")
        if self.dwell_ms < 0:
            errs.append(f"保压时间不能为负: {self.dwell_ms}")
        if self.dt_min_ms <= 0:
            errs.append(f"最小时间步必须为正: {self.dt_min_ms}")
        return errs


@dataclass
class StampingSpec:
    """一次冲压成形仿真的完整声明。求解器无关。"""

    name: str
    material: MaterialSpec
    blank: MeshSpec                  # 平面坯料(展开态),points 为 (N,2) 或 z=常数的 (N,3)
    tools: list[ToolSpec]
    process: ProcessSpec = field(default_factory=ProcessSpec)
    reference: dict = field(default_factory=dict)   # 对标值(AutoForm 等),仅供报告

    # ---------- 派生量 ----------
    @property
    def die(self) -> ToolSpec:
        return next(t for t in self.tools if t.motion == "fixed")

    @property
    def punch(self) -> ToolSpec:
        return next(t for t in self.tools if t.motion == "translate_z")

    def quasi_static_ratio(self) -> float:
        """冲压速度 / 材料弹性波速。远小于 1 才算准静态。"""
        v = self.punch.stroke / self.process.form_time_ms          # mm/ms
        c = np.sqrt(self.material.youngs / self.material.density)  # mm/ms
        return float(v / c)

    def mass_scaling_factor(self) -> float:
        """质量缩放倍数估计:(dt_min / dt_courant)^2。>100 会明显污染惯性。"""
        c = np.sqrt(self.material.youngs / self.material.density)
        l_min = self.blank.edge_lengths().min()
        dt_courant = l_min / c
        return float(max(1.0, (self.process.dt_min_ms / dt_courant) ** 2))

    # ---------- 校验 ----------
    def check_all(self) -> tuple[list[str], list[str]]:
        """返回 (阻断性错误, 警告)。

        区分二者的标准:错误会让结果失去意义(如间隙比 <1 导致全面熨薄),
        警告只是提示精度风险(如偏置面在个别小圆角处未能完全修复)。
        """
        errs: list[str] = []
        warns: list[str] = []
        errs += self.material.check()
        errs += [f"blank: {e}" for e in self.blank.check()]
        for t in self.tools:
            errs += t.check()
        errs += self.process.check()
        if 1.0 <= self.process.clearance_ratio < 1.10:
            warns.append(
                f"间隙比 {self.process.clearance_ratio} 偏小:拉延侧壁会增厚(本项目 Al 1050-O "
                "实测可达 +15%),间隙不足会在合模末段熨薄板料。1.0 时实测最大减薄虚高到 "
                "78%(对标真值 38%),建议 ≥1.10")

        fixed = [t for t in self.tools if t.motion == "fixed"]
        moving = [t for t in self.tools if t.motion == "translate_z"]
        if len(fixed) != 1:
            errs.append(f"需要恰好 1 个固定工具(凹模),实有 {len(fixed)}")
        if len(moving) != 1:
            errs.append(f"需要恰好 1 个运动工具(凸模),实有 {len(moving)}")

        # 几何自洽性:只有在前面都过关时才有意义
        if not errs:
            t0 = self.material.thickness
            die_pts = self.die.mesh.points3d
            pun_pts = self.punch.mesh.points3d
            blank_z = self.blank.points3d[:, 2]

            # 1. 坯料应在凹模上方,不能初始穿透
            if blank_z.min() < die_pts[:, 2].max():
                errs.append(f"坯料起始高度 {blank_z.min():.2f} 低于凹模最高点 "
                            f"{die_pts[:, 2].max():.2f},初始即穿透")
            # 2. 凸模应在坯料上方
            if pun_pts[:, 2].min() < blank_z.max():
                errs.append(f"凸模最低点 {pun_pts[:, 2].min():.2f} 低于坯料 "
                            f"{blank_z.max():.2f},初始即穿透")
            # 3. 行程要够:凸模最低点压到凹模最低点 + 间隙
            need = pun_pts[:, 2].min() - (die_pts[:, 2].min() + self.process.clearance_ratio * t0)
            if self.punch.stroke < need - 1e-6:
                errs.append(f"行程 {self.punch.stroke:.2f} 不足以合模(至少需 {need:.2f})")
            # 4. 合模状态下的凸凹模实际间隙——本项目踩过的坑。
            #    必须把凸模落到行程终点再量,否则量到的只是抬升高度。
            try:
                from scipy.spatial import cKDTree
                closed = pun_pts - np.array([0.0, 0.0, self.punch.stroke])
                d, _ = cKDTree(die_pts).query(closed)
                want = self.process.clearance_ratio * t0
                if d.min() < 0.5 * want:
                    errs.append(f"合模时凸模最近处到凹模仅 {d.min():.3f} mm < 0.5×间隙 "
                                f"{want:.3f},该处板料会被挤压出虚假减薄")
                elif d.min() < 0.8 * want:
                    n_bad = int((d < 0.8 * want).sum())
                    warns.append(f"合模时有 {n_bad} 个凸模节点间隙偏紧(最小 {d.min():.3f} "
                                 f"vs 目标 {want:.3f}),法向偏置在小圆角处未完全修复,"
                                 "这些位置的减薄率偏高,判读时应排除")
            except ImportError:
                pass
            # 5. 坯料与工具网格量级是否匹配
            if self.blank.edge_lengths().min() < 0.05:
                warns.append(f"坯料最小边长 {self.blank.edge_lengths().min():.4f} mm 过小,"
                             "显式时间步靠质量缩放维持,惯性可能失真")
            # 6. 准静态性
            qs = self.quasi_static_ratio()
            if qs > 0.02:
                errs.append(f"冲压速度/波速 = {qs:.3f} > 0.02,惯性效应会污染结果,"
                            f"应加长 form_time_ms(当前 {self.process.form_time_ms})")
            # 7. 质量缩放程度
            ms = self.mass_scaling_factor()
            if ms > 100:
                warns.append(f"质量缩放约 {ms:.0f}×(dt_min={self.process.dt_min_ms:g} ms),"
                             "惯性效应被放大,建议减小 dt_min_ms 或加密时间")

        return errs, warns

    def validate(self, strict: bool = True) -> list[str]:
        """校验。strict 时遇到阻断性错误抛 SpecError;返回警告列表。"""
        errs, warns = self.check_all()
        if strict and errs:
            raise SpecError(f"[{self.name}] 校验未通过,{len(errs)} 项错误:\n  - "
                            + "\n  - ".join(errs))
        return warns

    def summary(self) -> str:
        m, p = self.material, self.process
        return (
            f"StampingSpec[{self.name}]\n"
            f"  材料   {m.name}  t0={m.thickness} mm  E={m.youngs} MPa  "
            f"sigma_y={m.yield_stress:.1f} MPa  r_mean={m.r_mean:.3f}\n"
            f"  坯料   {self.blank.n_nodes} 节点 / {self.blank.n_elems} 单元  "
            f"边长 {self.blank.edge_lengths().min():.2f}~{self.blank.edge_lengths().max():.2f} mm\n"
            f"  工具   " + ", ".join(f"{t.name}({t.motion}"
                                     + (f", 行程 {t.stroke:.2f}mm)" if t.stroke else ")")
                                     for t in self.tools) + "\n"
            f"  工艺   摩擦 {p.friction}  间隙比 {p.clearance_ratio}  "
            f"成形 {p.form_time_ms} ms + 保压 {p.dwell_ms} ms\n"
            f"  数值   准静态比 {self.quasi_static_ratio():.4f}  "
            f"质量缩放 ~{self.mass_scaling_factor():.0f}x"
        )

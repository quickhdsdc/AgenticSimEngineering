"""核心数据模型 (贯穿各 Agent 的数据契约)。"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FldStatus(str, Enum):
    green = "green"
    yellow = "yellow"
    red = "red"


class Verdict(str, Enum):
    ok = "OK"
    ng = "NG"


# ---------- 输入侧 ----------
class ProjectSpec(BaseModel):
    """信息提取 Agent 的产物: 一个 RFQ 的结构化输入。"""
    project_no: str = ""
    customer: str = ""
    part_number: str = ""
    part_name: str = ""
    material: str = ""
    thickness_mm: float = 0.0
    part_size: str = ""              # 如 "231*215.5*6.2"
    min_radius_mm: float = 2.0       # 零件最小R角(数模检查产出),优化闭环的起点
    annual_volume: int | None = None
    press_tonnage: float | None = None
    source_files: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------- 仿真侧 ----------
class ProcessStep(BaseModel):
    name: str
    kind: Literal["blank", "form", "pre_trim", "restrike", "trim", "deburr", "bend", "flatten"] = "form"


class SimResult(BaseModel):
    """一次成型仿真的结果 (AutoForm 适配器返回)。"""
    min_thickness_mm: float
    thinning_ratio: float                      # (t - min)/t
    fld_status: FldStatus
    formability_status: Literal["ok", "wrinkle", "crack"] = "ok"
    max_springback_mm: float = 0.0
    n_steps: int = 1
    raw: dict = Field(default_factory=dict)     # 原始结果引用/路径


class Evaluation(BaseModel):
    verdict: Verdict
    failed_criteria: list[str] = Field(default_factory=list)
    detail: list[str] = Field(default_factory=list)


# ---------- 优化侧 ----------
class OptimizationAction(BaseModel):
    strategy_id: str
    action: str
    params: dict = Field(default_factory=dict)
    reason: str = ""


# ---------- 报告侧 ----------
class ToleranceSuggestion(BaseModel):
    location: str
    original_tol: str
    sim_basis: str
    suggested_tol: str
    reason: str


class ProcessPlan(BaseModel):
    blank_size: str = ""
    utilization: float | None = None
    tool_size: str = ""
    press_tonnage: float | None = None
    feed_type: str = ""
    stations: list[str] = Field(default_factory=list)


class FeasibilityReport(BaseModel):
    spec: ProjectSpec
    before: SimResult | None = None
    after: SimResult | None = None
    optimizations: list[OptimizationAction] = Field(default_factory=list)
    tolerance: list[ToleranceSuggestion] = Field(default_factory=list)
    process: ProcessPlan | None = None
    iterations: int = 0
    converged: bool = False
    conclusion: str = ""

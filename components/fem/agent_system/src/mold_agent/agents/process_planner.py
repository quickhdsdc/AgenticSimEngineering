"""工艺排样: 按 process_rules 生成标准工站序骨架。

注: 这是知识库里的通用工站模板,不是某零件的真实料带排样(真实料带需 AutoForm Strip 阶段)。
料片尺寸/传送方式来自真实解析结果;利用率需 Strip 工序确定,此处留空。
"""
from __future__ import annotations

from ..knowledge_base import loader
from ..models import ProcessPlan, ProjectSpec, SimResult


def plan_process(spec: ProjectSpec, sim: SimResult, unfold: dict) -> ProcessPlan:
    rules = loader.process_rules()
    stations = [s["name"] for s in rules["station_sequence"]["stations"]]

    return ProcessPlan(
        blank_size=unfold.get("blank_size", ""),
        utilization=unfold.get("utilization"),   # 需 Strip 工序确定,通常为空
        tool_size="",
        press_tonnage=spec.press_tonnage,
        feed_type="",                             # 需产量/尺寸判定,信息不足则留空
        stations=stations,
    )

"""报告生成 Agent (MVP 版): 按 report_template 结构输出 Markdown 报告草稿。

TODO(M3): 用 python-pptx 套样件 PPTX 模板;此处先出 Markdown 便于审阅。
"""
from __future__ import annotations

from ..models import FeasibilityReport, SimResult


def _sim_line(sim: SimResult | None) -> str:
    if not sim:
        return "—"
    return (
        f"最小料厚 {sim.min_thickness_mm}mm,减薄 {sim.thinning_ratio:.1%},"
        f"FLD {sim.fld_status.value},判定 "
        f"{'OK' if sim.fld_status.value != 'red' and sim.formability_status == 'ok' else 'NG'}"
    )


def render_markdown(report: FeasibilityReport) -> str:
    s = report.spec
    lines: list[str] = []
    lines.append(f"# 前期评估仿真报告 — {s.part_name or s.part_number}")
    lines.append("")
    lines.append("## 封面")
    lines.append(f"- 客户: {s.customer}")
    lines.append(f"- 零件号: {s.part_number}")
    lines.append(f"- 材质: {s.material}    料厚: {s.thickness_mm}mm")
    lines.append(f"- 零件尺寸: {s.part_size}    年产量: {s.annual_volume}")
    lines.append("")

    lines.append("## 成型性(优化前)")
    lines.append(f"- {_sim_line(report.before)}")
    lines.append("")

    lines.append("## 数模优化说明")
    if report.optimizations:
        for i, o in enumerate(report.optimizations, 1):
            lines.append(f"{i}. [{o.strategy_id}] {o.reason}")
    else:
        lines.append("- 无需优化,一次成型即达标")
    lines.append("")

    lines.append("## 成型性(优化后)")
    lines.append(f"- {_sim_line(report.after)}")
    lines.append(f"- 迭代次数: {report.iterations}    收敛: {'是' if report.converged else '否(需人工介入)'}")
    lines.append("")

    if report.process:
        p = report.process
        lines.append("## 工艺方案")
        lines.append(f"- 料片尺寸: {p.blank_size}")
        lines.append(f"- 材料利用率: {f'{p.utilization:.1%}' if p.utilization else '—'}")
        lines.append(f"- 传送方式: {p.feed_type}")
        lines.append(f"- 机床吨位: {p.press_tonnage or '待定'}")
        lines.append(f"- 工站顺序: {' → '.join(p.stations)}")
        lines.append("")

    if report.tolerance:
        lines.append("## 公差建议")
        for t in report.tolerance:
            lines.append(f"- {t.location}: {t.original_tol} → 建议 {t.suggested_tol} ({t.reason})")
        lines.append("")

    lines.append("## 结论")
    lines.append(report.conclusion or "—")
    lines.append("")
    lines.append("> 本报告由 mold-agent 依据 AutoForm QuickLink 结果自动生成,需工程师审核。")
    return "\n".join(lines)

"""前期评估技能 —— BaseTool 子类,包装确定性领域模块。全部基于真实数据,无任何桩/mock。

- run_autoform: 解析真实 QuickLink 包 → SimResult;同时从包内文件名取真实材料/零件号、
  由结果反推真实料厚 t0,写入 session 的 ProjectSpec(不硬编、取不到就留空)。
- evaluate_forming: 按知识库判 OK/NG(确定性)。
- generate_report: 出 PPTX;对外交付物 → hitl_required=True(执行前挂人审)。
"""
from __future__ import annotations

from pathlib import Path

from ..harness.tool import BaseTool, ToolResult

OUT_DIR = Path(__file__).resolve().parents[3] / "outputs"


class RunAutoForm(BaseTool):
    name: str = "run_autoform"
    description: str = ("解析 AutoForm QuickLink 导出包(.zip),得成型结果(料厚/减薄/FLD/料片),"
                        "并从包内取真实材料/零件号/料厚。")
    parameters: dict = {"type": "object",
                        "properties": {"quicklink_zip": {"type": "string", "description": "QuickLink .zip 路径"}},
                        "required": ["quicklink_zip"]}

    async def execute(self, session: dict, quicklink_zip: str = "") -> ToolResult:
        from ..autoform import parse_quicklink, parse_quicklink_meta
        from ..models import ProjectSpec
        sim = parse_quicklink(quicklink_zip)
        meta = parse_quicklink_meta(quicklink_zip)
        # 真实 ProjectSpec:材料/零件号来自包内文件名,料厚 t0 由结果反推(THICKNESS/(1+THINNING))
        session["sim"] = sim
        session["spec"] = ProjectSpec(
            part_number=meta["part_number"],
            material=meta["material"],
            thickness_mm=sim.raw.get("t0_mm", 0.0),
        )
        b = sim.raw.get("blank", {})
        return ToolResult(output=(
            f"零件号 {meta['part_number'] or '未知'}, 材料 {meta['material'] or '未知'}, "
            f"料厚(反推) {sim.raw.get('t0_mm')}mm; 最小料厚 {sim.min_thickness_mm}mm, "
            f"减薄 {sim.thinning_ratio:.1%}, FLD {sim.fld_status.value}, "
            f"开裂单元 {sim.raw.get('n_split_elements')}, 料片 {b.get('size', '-')}mm"))


class EvaluateForming(BaseTool):
    name: str = "evaluate_forming"
    description: str = "按成型判据(料厚≥70%、FLD无红、减薄≤30%)判 OK/NG。"
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, session: dict) -> ToolResult:
        from ..models import ProjectSpec
        from ..rule_engine import evaluate_forming
        sim = session.get("sim")
        if not sim:
            return ToolResult(error="尚无 AutoForm 结果,请先 run_autoform。")
        spec = session.get("spec") or ProjectSpec(thickness_mm=sim.raw.get("t0_mm", 0.0))
        ev = evaluate_forming(spec, sim)
        session["evaluation"] = ev
        return ToolResult(output=f"判定 {ev.verdict.value}。" + " ".join(ev.detail))


class GenerateReport(BaseTool):
    name: str = "generate_report"
    description: str = "根据判定与结果生成 PPTX 可行性报告草稿。"
    parameters: dict = {"type": "object", "properties": {"out_name": {"type": "string"}}}
    hitl_required: bool = True                 # 对外交付物,执行前需人审
    writes: list[str] = ["outputs/*.pptx"]

    async def execute(self, session: dict, out_name: str = "agent_feasibility_report.pptx") -> ToolResult:
        from ..agents import report_pptx
        from ..agents.process_planner import plan_process
        from ..models import FeasibilityReport, Verdict
        spec, sim, ev = session.get("spec"), session.get("sim"), session.get("evaluation")
        if not (spec and sim):
            return ToolResult(error="缺 spec/sim,无法出报告。")
        rep = FeasibilityReport(spec=spec, before=sim, after=sim)
        rep.converged = bool(ev) and ev.verdict == Verdict.ok
        rep.process = plan_process(spec, sim, {"blank_size": sim.raw.get("blank", {}).get("size", "")})
        rep.conclusion = ("成型性达标,工艺可行。" if rep.converged else
                          f"最薄 {sim.min_thickness_mm}mm(减薄 {sim.thinning_ratio:.0%})判 NG,"
                          "建议放大危险区R角/加高翻边后重评。")
        OUT_DIR.mkdir(exist_ok=True)
        path = report_pptx.build_pptx(rep, OUT_DIR / out_name)
        session["report_path"] = str(path)
        return ToolResult(output=f"已生成报告:{path}")


def build_forming_tools():
    """返回前期评估的 ToolCollection(run_autoform → evaluate_forming → generate_report)。"""
    from ..harness.tool import ToolCollection
    return ToolCollection(RunAutoForm(), EvaluateForming(), GenerateReport())

"""端到端:QuickLink 导出包 → 解析 → 判定 → 生成 PPTX 可行性报告。

用样件7真实数据。运行:
  PYTHONUTF8=1 agent_system/.venv/Scripts/python.exe agent_system/examples/generate_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mold_agent.agents import report_pptx, report_writer   # noqa: E402
from mold_agent.agents.process_planner import plan_process  # noqa: E402
from mold_agent.autoform import parse_quicklink             # noqa: E402
from mold_agent.models import FeasibilityReport, ProjectSpec, Verdict  # noqa: E402
from mold_agent.rule_engine import evaluate_forming         # noqa: E402

ZIP = (Path(__file__).resolve().parents[2] / "Samples" / "7-Q2603019"
       / "0220M000ND_stamping-Al 1050 Stamping.zip")


def main() -> None:
    spec = ProjectSpec(customer="BOSCH", part_number="0220M000ND",
                       part_name="Cover", material="AL 1050",
                       thickness_mm=1.2, press_tonnage=300)

    sim = parse_quicklink(ZIP)
    ev = evaluate_forming(spec, sim)

    report = FeasibilityReport(spec=spec, before=sim, after=sim)
    report.converged = ev.verdict == Verdict.ok
    report.process = plan_process(spec, sim, {"blank_size": sim.raw.get("blank", {}).get("size", "")})
    report.conclusion = (
        "成型性达标,工艺可行。" if report.converged else
        f"最薄处 {sim.min_thickness_mm}mm(减薄 {sim.thinning_ratio:.0%})出现在凸包,判 NG。"
        "建议放大该处 R 角 / 加高翻边后重评;详细公差待收到 2D 图纸复核。"
    )

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    pptx_path = out_dir / "0220M000ND_feasibility_report.pptx"
    md_path = out_dir / "0220M000ND_feasibility_report.md"

    report_pptx.build_pptx(report, pptx_path)
    md_path.write_text(report_writer.render_markdown(report), encoding="utf-8")

    print("解析:", ZIP.name)
    print(f"  最小料厚 {sim.min_thickness_mm}mm / 减薄 {sim.thinning_ratio:.1%} / FLD {sim.fld_status.value} / 料片 {sim.raw.get('blank', {}).get('size')}")
    print(f"  判定 {ev.verdict.value}")
    print("已生成报告:")
    print("  PPTX:", pptx_path)
    print("  MD  :", md_path)


if __name__ == "__main__":
    main()

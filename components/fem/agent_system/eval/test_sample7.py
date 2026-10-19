"""回归测试 —— 仅样件7(唯一在 AutoForm GUI 真实跑过、有 QuickLink 导出的样件)。

全部基于真实数据:
  · 输入 = Samples/7-Q2603019/…Stamping.zip(真实 QuickLink 导出)
  · 期望 = 该样件公司报告 Simulation-Report.pptx 里工程师写的判定(NG,最小料厚~0.765mm)
断言我们的解析+判据引擎结论与工程师一致。

其他样件(1-6,8-10)尚未在 GUI 运行、无 QuickLink 数据,故不纳入(信息不完整)。

pytest 可收集;也可直接运行:
  PYTHONUTF8=1 agent_system/.venv/Scripts/python.exe agent_system/eval/test_sample7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # agent_system/
REPO = ROOT.parent                               # _MoldDesign/
sys.path.insert(0, str(ROOT / "src"))

from mold_agent.autoform import parse_quicklink, parse_quicklink_meta  # noqa: E402
from mold_agent.models import ProjectSpec        # noqa: E402
from mold_agent.rule_engine import evaluate_forming  # noqa: E402

ZIP = REPO / "Samples" / "7-Q2603019" / "0220M000ND_stamping-Al 1050 Stamping.zip"

# 工程师报告(Samples/7-Q2603019/输出_output/*Simulation-Report*.pptx)的优化前判定:
ENGINEER = {"verdict": "NG", "min_thickness_mm": 0.765, "thinning": 0.362}


def test_sample7_pipeline():
    """样件7 真实 QuickLink → 解析 → 判定,结论与工程师一致。"""
    assert ZIP.exists(), f"缺真实 QuickLink 包:{ZIP}"
    sim = parse_quicklink(ZIP)
    meta = parse_quicklink_meta(ZIP)
    spec = ProjectSpec(part_number=meta["part_number"], material=meta["material"],
                       thickness_mm=sim.raw.get("t0_mm", 0.0))
    ev = evaluate_forming(spec, sim)

    assert ev.verdict.value == ENGINEER["verdict"], f"判定应 {ENGINEER['verdict']},实得 {ev.verdict.value}"
    # 我们的一步法(Crashform)与工程师值应接近(不同方法允许偏差)
    assert abs(sim.min_thickness_mm - ENGINEER["min_thickness_mm"]) < 0.05, \
        f"最小料厚 {sim.min_thickness_mm} 偏离工程师 {ENGINEER['min_thickness_mm']} 过大"
    return sim, meta, ev


if __name__ == "__main__":
    try:
        sim, meta, ev = test_sample7_pipeline()
    except AssertionError as e:
        print("❌ FAIL:", e); raise SystemExit(1)
    print("样件7 真实 QuickLink 回归:")
    print(f"  材料(包内)   : {meta['material']}   零件号: {meta['part_number']}")
    print(f"  料厚(反推)   : {sim.raw.get('t0_mm')} mm")
    print(f"  最小料厚      : {sim.min_thickness_mm} mm   减薄 {sim.thinning_ratio:.1%}   FLD {sim.fld_status.value}")
    print(f"  判定          : {ev.verdict.value}")
    print(f"  工程师(报告) : {ENGINEER['verdict']}, 最小料厚 {ENGINEER['min_thickness_mm']}mm, 减薄 {ENGINEER['thinning']:.0%}")
    print("✅ PASS:解析+判据与工程师一致")

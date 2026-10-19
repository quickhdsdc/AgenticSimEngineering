"""用真实 QuickLink 导出包(样件7)验证解析器 + 规则引擎。

数据来源: Samples/7-Q2603019 手动 GUI dry-run 导出的 QuickLink .zip。
运行:
  PYTHONUTF8=1 agent_system/.venv/Scripts/python.exe agent_system/examples/parse_real_quicklink.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mold_agent.autoform import parse_quicklink          # noqa: E402
from mold_agent.models import ProjectSpec                 # noqa: E402
from mold_agent.rule_engine import evaluate_forming       # noqa: E402

ZIP = (Path(__file__).resolve().parents[2] / "Samples" / "7-Q2603019"
       / "0220M000ND_stamping-Al 1050 Stamping.zip")


def main() -> None:
    spec = ProjectSpec(customer="BOSCH", part_number="0220M000ND",
                       material="AL 1050", thickness_mm=1.2)

    print("=" * 64)
    print("解析真实 QuickLink 包:", ZIP.name)
    print("=" * 64)
    sim = parse_quicklink(ZIP)
    print(f"原始料厚 t0        : {sim.raw['t0_mm']} mm")
    print(f"最小料厚           : {sim.min_thickness_mm} mm")
    print(f"最大减薄           : {sim.thinning_ratio:.1%}")
    print(f"FLD 状态           : {sim.fld_status.value}")
    print(f"成型性             : {sim.formability_status}")
    print(f"单元数             : {sim.raw['n_elements']}  (开裂单元 {sim.raw['n_split_elements']})")
    print(f"分区统计           : {sim.raw['zone_counts']}")

    ev = evaluate_forming(spec, sim)
    print("-" * 64)
    print(f"规则引擎判定       : {ev.verdict.value}")
    for d in ev.detail:
        print("   ", d)

    print("-" * 64)
    print("对标准答案(公司报告): 最小料厚 0.765mm / 减薄 36.2% / 判 NG")
    print(f"本次(Crashform)    : 最小料厚 {sim.min_thickness_mm}mm / 减薄 {sim.thinning_ratio:.1%} / 判 {ev.verdict.value}")


if __name__ == "__main__":
    main()

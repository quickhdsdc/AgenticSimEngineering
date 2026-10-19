# -*- coding: utf-8 -*-
"""校验层回归测试:确认坏 spec 在碰求解器之前就被拦下(秒级,不跑求解器)。

    PYTHONUTF8=1 uv run --with numpy,scipy python -m formpipe.test_validation
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

QUICKLINK = Path(__file__).resolve().parent.parent / "sample7_run" / "quicklink"


def main() -> int:
    from .quicklink import load

    spec, _ = load(QUICKLINK)

    # 基准模型必须干净通过(否则说明校验过严,会误伤正常模型)
    errs, warns = spec.check_all()
    print(f"基准 spec: {len(errs)} 错误 / {len(warns)} 警告")
    for w in warns:
        print(f"  ! {w[:100]}")
    failures = []
    if errs:
        failures.append("基准 spec 不应有错误")

    bad_cases = {
        "间隙比 0.9(凸模比板料薄)": lambda s: setattr(s.process, "clearance_ratio", 0.9),
        "摩擦系数 0.9(超出物理范围)": lambda s: setattr(s.process, "friction", 0.9),
        "成形时间 0.05ms(冲击而非准静态)": lambda s: setattr(s.process, "form_time_ms", 0.05),
        "料厚为负": lambda s: setattr(s.material, "thickness", -1.0),
        "流动曲线出现软化段": lambda s: setattr(
            s.material, "flow_curve", [(0.0, 100.0), (0.1, 90.0), (0.5, 120.0)]),
        "坯料初始穿透凹模": lambda s: setattr(
            s.blank, "points", s.blank.points3d - [0, 0, 50]),
        "密度用错单位(kg/m^3)": lambda s: setattr(s.material, "density", 2700.0),
    }
    print()
    for name, mutate in bad_cases.items():
        s = copy.deepcopy(spec)
        mutate(s)
        e, _ = s.check_all()
        ok = bool(e)
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if ok:
            print(f"        {e[0][:100]}")
        else:
            failures.append(f"未拦下: {name}")

    print()
    if failures:
        print(f"{len(failures)} 项未通过:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"全部通过({len(bad_cases)} 个反例 + 1 个正例)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

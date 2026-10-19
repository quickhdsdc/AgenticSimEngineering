"""规则引擎: 读知识库,对仿真结果做判定,并决定下一步优化动作。"""
from __future__ import annotations

from .knowledge_base import loader
from .models import (
    Evaluation,
    OptimizationAction,
    ProjectSpec,
    SimResult,
    Verdict,
)


def evaluate_forming(spec: ProjectSpec, sim: SimResult) -> Evaluation:
    """按 forming_rules.yaml 判定成型性 OK/NG。"""
    rules = loader.forming_rules()
    t = spec.thickness_mm or 1.2
    failed: list[str] = []
    detail: list[str] = []

    crit = {c["id"]: c for c in rules["criteria"]}

    # 最小料厚
    retain = crit["min_thickness"]["retain_ratio"]
    threshold = round(t * retain, 3)
    if sim.min_thickness_mm < threshold:
        failed.append("min_thickness")
    detail.append(
        f"最小料厚 {sim.min_thickness_mm}mm vs 阈值 {threshold}mm "
        f"(料厚{t}×{retain}) → {'NG' if sim.min_thickness_mm < threshold else 'OK'}"
    )

    # 减薄率
    max_thin = crit["thinning_ratio"]["max_thinning"]
    if sim.thinning_ratio > max_thin:
        failed.append("thinning_ratio")
    detail.append(
        f"减薄率 {sim.thinning_ratio:.1%} vs 上限 {max_thin:.0%} "
        f"→ {'NG' if sim.thinning_ratio > max_thin else 'OK'}"
    )

    # FLD 无红
    if sim.fld_status.value == "red":
        failed.append("fld_no_red")
    detail.append(f"FLD 状态 {sim.fld_status.value} → {'NG' if sim.fld_status.value == 'red' else 'OK'}")

    # 成型性
    if sim.formability_status != "ok":
        failed.append("formability")
    detail.append(f"成型性 {sim.formability_status}")

    verdict = Verdict.ng if failed else Verdict.ok
    return Evaluation(verdict=verdict, failed_criteria=failed, detail=detail)


def decide_optimization(
    spec: ProjectSpec,
    sim: SimResult,
    evaluation: Evaluation,
    state: dict,
) -> OptimizationAction | None:
    """按 optimization_rules.yaml 的 priority 选下一个优化动作。

    state 记录闭环内已尝试的策略/已达到的几何值 (放大后的R、是否分步等),
    以便按"先放大R → 再加凸台 → 再分步 → 凸包多步"的阶梯推进。
    """
    if evaluation.verdict == Verdict.ok:
        return None

    rules = loader.optimization_rules()
    t = spec.thickness_mm or 1.2
    opt = optimization_state(state)

    # 阶梯 1: R角以 0.2mm 递增放大 (设上限,避免无限放大)
    enlarge = _strategy(rules, "enlarge_r_iterative")
    r_step = enlarge["params"]["step_mm"]
    r_cap = 8.0  # 经验上界 (样件: R3.8→R8)
    if opt["effective_radius_mm"] + r_step <= r_cap and not opt["r_capped"]:
        opt["effective_radius_mm"] = round(opt["effective_radius_mm"] + r_step, 3)
        return OptimizationAction(
            strategy_id="enlarge_r_iterative",
            action="enlarge_radius",
            params={"effective_radius_mm": opt["effective_radius_mm"], "step_mm": r_step},
            reason=f"FLD/减薄 NG → R角放大 {r_step}mm 至 {opt['effective_radius_mm']}mm",
        )
    opt["r_capped"] = True

    # 阶梯 2: 凸包下加台阶 (R到上限仍NG)
    step_rule = _strategy(rules, "add_step_below_boss")
    if not opt["boss_step_added"]:
        opt["boss_step_added"] = True
        opt["boss_step_mm"] = step_rule["params"]["height_mm"]
        return OptimizationAction(
            strategy_id="add_step_below_boss",
            action="add_step",
            params={"height_mm": opt["boss_step_mm"]},
            reason=f"R角放大已达上限仍NG → 凸包下方增加{opt['boss_step_mm']}mm台阶",
        )

    # 阶梯 3: 分步 (由里向外)
    if not opt["split_done"]:
        opt["split_done"] = True
        opt["n_form_steps"] = 3
        return OptimizationAction(
            strategy_id="split_into_steps",
            action="split_forming_steps",
            params={"order": "inside_out", "n_form_steps": 3},
            reason="一步成型无法实现 → 由里向外拆为3步成型",
        )

    # 阶梯 4: 凸包多步
    if opt["n_form_steps"] < 5:
        opt["n_form_steps"] += 1
        return OptimizationAction(
            strategy_id="multi_step_boss",
            action="multi_step_boss_forming",
            params={"n_form_steps": opt["n_form_steps"]},
            reason=f"分步后凸包仍不足 → 凸包多步成型,增至{opt['n_form_steps']}步",
        )

    return None  # 策略用尽


def optimization_state(state: dict) -> dict:
    """初始化/取出闭环内的优化状态。"""
    return state.setdefault(
        "opt",
        {
            "effective_radius_mm": 2.0,
            "r_capped": False,
            "boss_step_added": False,
            "boss_step_mm": 0.0,
            "split_done": False,
            "n_form_steps": 1,
        },
    )


def _strategy(rules: dict, sid: str) -> dict:
    for s in rules["strategies"]:
        if s["id"] == sid:
            return s
    raise KeyError(sid)

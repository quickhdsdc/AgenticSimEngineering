"""GradientShop, registered for the HTTP surface.

One scenario and one world model, but they carry the framework's sharpest
demonstration: the *same* generic solvers that drive the intralogistics world
models drive this one, with no domain code on either side. That is Axiom 1
made operational, and the registry shows it as a fact rather than a claim —
``/api/schema`` lists the same solver ids under both domains' worlds.

Dispatch answers in about a millisecond, so everything here is interactive:
the temperature sweep and the pre-flight verdict included, which is what makes
the RQ1 result something a reader can *drag* rather than read about.
"""

from __future__ import annotations

import torch

from opsworld.service.registry import ParamSpec, Registry, ScenarioSpec, WorldSpec

from .ddes.dispatch import dispatch_descent_check, fidelity_curve, unresolved_fraction
from .world import DispatchWorld

__all__ = ["register", "build_dispatch_scenario", "tau_sweep"]


_PARAMS = (
    ParamSpec("n_jobs", "int", 12, "工件数", "单机排序问题的规模",
              minimum=3, maximum=40, step=1),
    ParamSpec("tau", "float", 0.1, "松弛温度 τ",
              "SoftSort 的温度。趋零时调度保真度改善,但梯度会崩溃并翻号",
              minimum=0.005, maximum=5.0, step=0.005),
    ParamSpec("objective", "enum", "wct", "目标函数",
              "加权完工时间可靠可优化;加权拖期是一个有记录的失效模式",
              choices=("wct", "tardiness")),
    ParamSpec("seed", "int", 7, "随机种子", "决定加工时间、权重、初始分数",
              minimum=0, maximum=9999, step=1),
)


def build_dispatch_scenario(n_jobs: int = 12, tau: float = 0.1,
                            objective: str = "wct", seed: int = 7) -> dict:
    """The raw ingredients, kept separate from the world model.

    Returned as a plain dict rather than a built world so the tensors can be
    reused by the temperature sweep without rebuilding — the sweep varies τ,
    which is a property of the *world model*, not of the instance.
    """
    generator = torch.Generator().manual_seed(int(seed))
    n = int(n_jobs)
    processing = torch.rand(n, generator=generator, dtype=torch.float64) + 0.2
    weight = torch.rand(n, generator=generator, dtype=torch.float64) + 0.2
    scores = torch.randn(n, generator=generator, dtype=torch.float64)
    due = processing.cumsum(0) * 0.6 + 1.0
    return {"processing": processing, "weight": weight, "due": due,
            "scores": scores, "tau": float(tau), "objective": objective,
            "n_jobs": n, "seed": int(seed)}


def _summarise(scenario: dict) -> dict:
    return {
        "id": f"dispatch(n={scenario['n_jobs']}, tau={scenario['tau']}, "
              f"{scenario['objective']})",
        "n_jobs": scenario["n_jobs"],
        "tau": scenario["tau"],
        "objective": scenario["objective"],
        "jobs": [
            {"job": f"J{i + 1}",
             "processing": round(float(scenario["processing"][i]), 4),
             "weight": round(float(scenario["weight"][i]), 4),
             "due": round(float(scenario["due"][i]), 4)}
            for i in range(scenario["n_jobs"])
        ],
    }


def _build_world(scenario: dict) -> DispatchWorld:
    return DispatchWorld(
        scenario["processing"], scenario["weight"], scenario["due"],
        tau=scenario["tau"],
        objective=scenario["objective"],
    )


def _encode_vector(world, decision) -> list[float]:
    return [float(v) for v in decision]


def _decode_vector(world, payload) -> list[float]:
    if not isinstance(payload, (list, tuple)):
        raise ValueError("expected a list of weights, one per dispatching rule")
    space = world.decision_space()
    if len(payload) != space.dimension:
        raise ValueError(
            f"expected {space.dimension} weights ({', '.join(space.names)}), "
            f"got {len(payload)}"
        )
    return [float(v) for v in payload]


def _describe(world, scenario: dict) -> dict:
    return {"rules": list(world.decision_space().names),
            "tau": scenario["tau"], "objective": scenario["objective"]}


def tau_sweep(scenario: dict, taus: list[float]) -> dict:
    """The RQ1 probe, with the verdict the cosine cannot give.

    Returns the fidelity curve *and*, per temperature, whether a step along the
    relaxed gradient actually lowers the exact discrete cost. Those two
    disagree — decision record D15 — and putting them in one payload is what
    lets a UI show the disagreement instead of describing it.

    Costs about 9 ms for eight temperatures, so it belongs in a request.
    """
    curve = fidelity_curve(
        scenario["processing"], scenario["scores"], scenario["weight"], list(taus),
        due=scenario["due"] if scenario["objective"] == "tardiness" else None,
    )
    rows = []
    for index, tau in enumerate(curve["tau"]):
        # positional order is (processing, scores, weight) -- naming them keeps
        # the next reader from repeating the swap this line started life with
        report = dispatch_descent_check(
            processing=scenario["processing"],
            scores=scenario["scores"],
            weight=scenario["weight"],
            tau=tau,
            due=scenario["due"] if scenario["objective"] == "tardiness" else None,
        )
        rows.append({
            "tau": tau,
            "soft_cost": curve["soft_cost"][index],
            "gap": curve["gap"][index],
            "gap_pct": 100 * curve["gap"][index] / curve["exact_cost"],
            "grad_cosine": curve["grad_cosine"][index],
            "unresolved_fraction": unresolved_fraction(scenario["scores"], tau),
            "descends": report.descends,
            "improvement": report.improvement,
        })
    return {"exact_cost": curve["exact_cost"], "rows": rows,
            "note": "grad_cosine 与 descends 会互相矛盾 —— 前者比较的两个梯度"
                    "共用同一个 Jacobian,恰在双双消失时完全一致(D15)。"}


def register(registry: Registry) -> Registry:
    registry.add_scenario(ScenarioSpec(
        id="gradientshop.dispatch",
        label="单机派工",
        description=(
            "把 argsort 这个离散排序决策用 SoftSort 松弛。RQ1 的原始场景 —— "
            "也是那个「保真度改善但梯度崩溃」的结果的来源。"
        ),
        params=_PARAMS,
        build=build_dispatch_scenario,
        summarise=_summarise,
    ))
    registry.add_world(WorldSpec(
        id="gradientshop.dispatch",
        label="派工策略(可微)",
        description="决策 = 命名派工规则上的权重向量。C0 + C2a。",
        scenario="gradientshop.dispatch",
        build=_build_world,
        encode_decision=_encode_vector,
        decode_decision=_decode_vector,
        interactive=True,
        # about a millisecond per replay; a 2000-evaluation run is ~2 s inline
        # against ~3.8 s of spawn overhead alone
        inline_max_budget=2000,
        describe=_describe,
    ))
    return registry

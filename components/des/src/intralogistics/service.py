"""Intralogistics, registered for the HTTP surface.

The domain declares what a client may build and how its decisions cross the
JSON boundary; the shared layer never learns what a ``SupplyPlan`` is
(architecture constraint A6). Registration is a function rather than an import
side effect so the composition root stays the one place that knows both sides.

The world models are split by **how fast they answer**, because that decides
whether the UI can put them behind a slider or has to put them behind a job:

``sizing``   0.2 ms per evaluation, and it hands back a gradient — the one
             world model here that can drive a control the user drags
``plan``     0.5 ms to construct, 8.8 ms to evaluate at 6x3 (87 ms at 40x4) —
             fast enough to answer, too slow to search inside a request
``static``   the same evaluator over raw variables, for inspecting a decision
             the client already has
``milp``     seconds to minutes, and it is the only one that can prove
             optimality — always a job
"""

from __future__ import annotations

import torch

from opsworld.service.codec import (
    decode_index_map,
    decode_table,
    encode_index_map,
    encode_table,
)
from opsworld.service.registry import (
    ParamSpec,
    Registry,
    ScenarioSpec,
    SolverSpec,
    WorldSpec,
)
from opsworld.solver import (
    BayesianOptimisationSolver,
    ExactSolver,
    GradientDescentSolver,
    HillClimbSolver,
    RandomSearchSolver,
)
from opsworld.world import Capability

from .data import build_minimal_instance
from .decision import Decision
from .heuristic import GreedySupplySolver, PlanWorld, SupplyPlan
from .instances import build_network_instance
from .milp_model import available_solvers
from .policy import Items, SizingPolicyWorld
from .world import MilpWorld, StaticWorld

__all__ = ["register", "encode_plan", "decode_plan", "encode_decision", "decode_decision"]


# ----------------------------------------------------------------------
# codecs — the domain's own, because the shared layer does not know these types
# ----------------------------------------------------------------------
def encode_decision(world, decision: Decision) -> dict:
    return {name: encode_table(table) for name, table in decision.as_dict().items()}


def decode_decision(world, payload: dict) -> Decision:
    unknown = set(payload) - set(Decision.VARIABLE_NAMES)
    if unknown:
        raise ValueError(f"unknown variable family: {sorted(unknown)}")
    return Decision(**{name: decode_table(payload.get(name, {}))
                       for name in Decision.VARIABLE_NAMES})


def encode_plan(world, plan: SupplyPlan) -> dict:
    """A plan is three tables; the derived variables are recomputed, not sent.

    That asymmetry is the point of the plan representation — sending the
    expansion back would invite a client to edit a variable whose value is a
    *consequence*, which is exactly the dead end Phase 2 measured. The values
    here are entity names, not numbers, so this uses the untyped index map.
    """
    return {
        "pick": encode_index_map(plan.pick),
        "carrier": encode_index_map(plan.carrier),
        "hub": {str(k): v for k, v in plan.hub.items()},
    }


def decode_plan(world, payload: dict) -> SupplyPlan:
    return SupplyPlan(
        pick=decode_index_map(payload.get("pick", {})),
        carrier=decode_index_map(payload.get("carrier", {})),
        hub=dict(payload.get("hub", {})),
    )


def _encode_vector(world, decision) -> list[float]:
    return [float(v) for v in decision]


def _decode_vector(world, payload) -> list[float]:
    if not isinstance(payload, (list, tuple)):
        raise ValueError("expected a list of weights, one per named rule")
    space = world.decision_space()
    if len(payload) != space.dimension:
        raise ValueError(
            f"expected {space.dimension} weights ({', '.join(space.names)}), "
            f"got {len(payload)}"
        )
    return [float(v) for v in payload]


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------
_NETWORK_PARAMS = (
    ParamSpec("n_items", "int", 4, "物料数", "越多组合越难,精确求解在 8 个左右撞墙",
              minimum=1, maximum=60, step=1),
    ParamSpec("n_resources", "int", 2, "资源(工位)数", "每个资源有自己的使用点仓",
              minimum=1, maximum=6, step=1),
    ParamSpec("n_forward", "int", 2, "前置仓数", "平行的前置仓给移库路线以备选",
              minimum=1, maximum=3, step=1),
    ParamSpec("seed", "int", 0, "随机种子", "决定需求、价值、体积的抽样;结果可复现",
              minimum=0, maximum=9999, step=1),
    ParamSpec("tighten", "float", 1.0, "容量松紧", "小于 1 收紧仓储与备料区容量,直到贪心构造不再可行",
              minimum=0.4, maximum=2.0, step=0.05),
    ParamSpec("storage_rate_point_of_use", "float", 4.0, "使用点仓储费率",
              "Phase 1 发现源模型这里填的是 1e5 —— 一个占位值,占目标值 99.994% 却不改变决策",
              minimum=0.1, maximum=100000.0, step=0.1),
)


def _summarise_instance(instance) -> dict:
    return {
        "id": instance.id(),
        "items": list(instance.I),
        "resources": list(instance.R),
        "warehouses": list(instance.L),
        "carriers": list(instance.B),
        "points_of_use": list(instance.L_prod),
        "intermediate": list(instance.L_log),
        "relocation_variables": (
            len(instance.I) * len(instance.R)
            * len(instance.U_edges) * len(instance.B) ** 2
        ),
    }


_SIZING_PARAMS = (
    ParamSpec("n_items", "int", 8, "物料数", "每个物料一条需求流",
              minimum=2, maximum=40, step=1),
    ParamSpec("demand_cv", "float", 0.4, "需求变异系数",
              "Phase 3 的核心变量:时间平均在 CV=0 时精确,随 CV 退化",
              minimum=0.0, maximum=1.2, step=0.05),
    ParamSpec("lead_time", "int", 3, "补货提前期", "期", minimum=1, maximum=10, step=1),
    ParamSpec("horizon", "int", 600, "仿真时域", "期数;越长仿真评分越稳但越慢",
              minimum=100, maximum=3000, step=100),
    ParamSpec("seed", "int", 0, "随机种子", "", minimum=0, maximum=9999, step=1),
)


def _build_sizing(n_items: int, demand_cv: float, lead_time: int,
                  horizon: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))
    items = Items(
        demand_rate=torch.rand(n_items, generator=generator, dtype=torch.float64) * 12 + 2,
        unit_value=torch.rand(n_items, generator=generator, dtype=torch.float64) * 10 + 0.5,
        unit_area=torch.rand(n_items, generator=generator, dtype=torch.float64) * 0.3 + 0.05,
        lead_time=torch.full((n_items,), float(lead_time), dtype=torch.float64),
        demand_cv=torch.full((n_items,), float(demand_cv), dtype=torch.float64),
    )
    shape = 1.0 / max(demand_cv, 1e-6) ** 2
    demand = (
        torch.distributions.Gamma(shape, shape / items.demand_rate.unsqueeze(1))
        .sample((int(horizon),)).squeeze(-1).T.to(torch.float64)
        if demand_cv > 0
        else items.demand_rate.unsqueeze(1).repeat(1, int(horizon))
    )
    return SizingPolicyWorld(items, demand, lead_time=int(lead_time))


def _summarise_sizing(world) -> dict:
    return {
        "id": world.provenance,
        "rules": list(world.rule_names),
        "items": [
            {
                "item": f"A{i + 1}",
                "demand_rate": round(float(world.items.demand_rate[i]), 3),
                "unit_value": round(float(world.items.unit_value[i]), 3),
                "unit_area": round(float(world.items.unit_area[i]), 4),
                "demand_cv": round(float(world.items.demand_cv[i]), 3),
            }
            for i in range(len(world.items))
        ],
        "capacity_range": [world.min_capacity, world.max_capacity],
    }


def _describe_sizing(world, instance) -> dict:
    """Extras a sizing UI needs: the rule names are the axis labels."""
    return {"rules": list(world.rule_names),
            "capacity_range": [world.min_capacity, world.max_capacity]}


# ----------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------
def register(registry: Registry) -> Registry:
    exact_backend_choices = ("auto", *available_solvers())
    registry.add_scenario(ScenarioSpec(
        id="intralogistics.network",
        label="物料供应网络",
        description=(
            "同一套源模型代数,但决策是打开的:每个资源有 ≥2 个可取料仓、三种载具尺寸、"
            "平行前置仓,容量确实会吃紧。"
        ),
        params=_NETWORK_PARAMS,
        build=build_network_instance,
        summarise=_summarise_instance,
    ))
    registry.add_scenario(ScenarioSpec(
        id="intralogistics.source",
        label="源算例(最小)",
        description=(
            "合作方 notebook 里的最小算例,Phase 1 的交叉验证锚点。"
            "结构是被约束强制的 —— 几乎没有可选的配置。"
        ),
        params=(),
        build=build_minimal_instance,
        summarise=_summarise_instance,
    ))
    registry.add_scenario(ScenarioSpec(
        id="intralogistics.sizing",
        label="载具装载策略",
        description=(
            "每个物料一条需求流,决策是 8 条命名规则上的权重。"
            "这是唯一能实时拖动的那个 —— 一次评估 0.2 毫秒,而且带梯度。"
        ),
        params=_SIZING_PARAMS,
        build=_build_sizing,
        summarise=_summarise_sizing,
    ))

    registry.add_world(WorldSpec(
        id="intralogistics.sizing",
        label="装载策略(可微)",
        description="决策 = 8 个命名规则的权重。C0 + C2a。",
        scenario="intralogistics.sizing",
        build=lambda world: world,          # the scenario already built it
        encode_decision=_encode_vector,
        decode_decision=_decode_vector,
        interactive=True,
        # 0.2 ms per evaluation, so a few thousand still beats one process spawn
        inline_max_budget=4000,
        describe=_describe_sizing,
    ))
    registry.add_world(WorldSpec(
        id="intralogistics.plan",
        label="配置规划(决策级)",
        description=(
            "决策 = 每条需求从哪取料、哪里放什么载具、主库存放哪。"
            "其余变量全部推导。Phase 5.1 表明这才是能被搜索的参数化。"
        ),
        scenario="intralogistics.network",
        build=PlanWorld,
        encode_decision=encode_plan,
        decode_decision=decode_plan,
        interactive=True,
    ))
    registry.add_world(WorldSpec(
        id="intralogistics.static",
        label="作用模型(原始变量)",
        description="同一个评估器,但决策是源模型的全部变量。用来检视一个已有配置。",
        scenario="intralogistics.network",
        build=StaticWorld,
        encode_decision=encode_decision,
        decode_decision=decode_decision,
        interactive=True,
    ))
    registry.add_world(WorldSpec(
        id="intralogistics.milp",
        label="决策模型(MILP)",
        description="唯一能证明最优的那个,也是唯一会撞墙的那个。C0 + C2b + C3。",
        scenario="intralogistics.network",
        build=MilpWorld,
        encode_decision=encode_decision,
        decode_decision=decode_decision,
        interactive=False,
    ))

    registry.add_solver(SolverSpec(
        id="random", label="随机搜索",
        description="诚实的基线 —— 不建任何模型。",
        requires=RandomSearchSolver.requires,
        build=lambda seed=0: RandomSearchSolver(seed=int(seed)),
        params=(ParamSpec("seed", "int", 0, "种子", "", minimum=0, maximum=9999),),
    ))
    registry.add_solver(SolverSpec(
        id="hillclimb", label="爬山",
        description="扰动现任,只接受严格改进。",
        requires=HillClimbSolver.requires,
        build=lambda seed=0: HillClimbSolver(seed=int(seed)),
        params=(ParamSpec("seed", "int", 0, "种子", "", minimum=0, maximum=9999),),
    ))
    registry.add_solver(SolverSpec(
        id="bayes", label="贝叶斯优化",
        description="高斯过程 + 期望改进。最强的无导数基线。",
        requires=BayesianOptimisationSolver.requires,
        requires_space=frozenset({"vectorize"}),
        build=lambda seed=0, n_initial=8: BayesianOptimisationSolver(
            seed=int(seed), n_initial=int(n_initial)),
        params=(
            ParamSpec("seed", "int", 0, "种子", "", minimum=0, maximum=9999),
            ParamSpec("n_initial", "int", 8, "初始设计点数", "先随机采多少个点再开始建模",
                      minimum=3, maximum=32),
        ),
    ))
    registry.add_solver(SolverSpec(
        id="gradient", label="梯度下降",
        description=(
            "需要 C2a。带回溯线搜索,且每次试探步都计入预算 —— "
            "否则线搜索等于白拿评估次数。"
        ),
        requires=GradientDescentSolver.requires,
        build=lambda lr=0.3, backtrack=True: GradientDescentSolver(
            lr=float(lr), backtrack=bool(backtrack)),
        params=(
            ParamSpec("lr", "float", 0.3, "初始步长", "回溯会自动收缩它",
                      minimum=0.001, maximum=5.0, step=0.001),
            ParamSpec("backtrack", "bool", True, "回溯线搜索",
                      "关掉它,结果就变成关于这个常数的结论"),
        ),
    ))
    registry.add_solver(SolverSpec(
        id="exact", label="MILP 全局优化",
        description=(
            "需要 C3。唯一能给出**最优性证明**的求解器 —— 也是唯一会撞墙的:"
            "模型随物料数指数增长,8 个物料左右就只能返回不完全解。"
            "结果的 provenance 里带终止状态,`optimal` 才是证明过的。"
        ),
        requires=ExactSolver.requires,
        build=lambda time_limit=60.0, backend="auto": ExactSolver(
            time_limit=float(time_limit),
            backend=None if backend == "auto" else str(backend),
        ),
        params=(
            ParamSpec("backend", "enum", "auto", "MILP 求解引擎",
                      "auto 按 appsi_highs → highs → SCIP → CBC → Gurobi 的顺序选择本机第一个可用引擎",
                      choices=exact_backend_choices),
            ParamSpec("time_limit", "float", 60.0, "时间上限(秒)",
                      "到点返回当前不完全解;没有它,大算例的请求可能永远不返回",
                      minimum=1.0, maximum=600.0, step=1.0),
        ),
        seconds_per_evaluation=None,
    ))
    registry.add_solver(SolverSpec(
        id="greedy-supply", label="构造式启发式",
        description=(
            "贪心构造 + 决策级最优改进局部搜索。跑得到精确求解器回不来的规模。"
        ),
        requires=GreedySupplySolver.requires,
        # C0 is all it asks of the world model, but it enumerates a
        # neighbourhood -- so it needs a decision space that has one. A weight
        # vector does not, and without this it would be offered for the
        # dispatch policy and then fail with an AttributeError.
        requires_space=frozenset({"moves"}),
        build=lambda max_passes=20, max_moves=0: GreedySupplySolver(
            max_passes=int(max_passes),
            max_moves=int(max_moves) or None),
        params=(
            ParamSpec("max_passes", "int", 20, "最多几轮", "", minimum=1, maximum=60),
            ParamSpec("max_moves", "int", 0, "每轮最多看几个邻居",
                      "0 = 全部。大算例上每个邻居要一次完整评估,这里是刹车",
                      minimum=0, maximum=400),
        ),
        seconds_per_evaluation=None,
    ))
    return registry

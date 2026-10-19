/* Offline stand-in for the opsworld HTTP surface.
   Same payload shapes as docs/api.md, computed from small honest models so the
   charts move for the right reasons when 127.0.0.1:8000 is not running.
   Also carries the English label overlay used for the live server too — the
   registry's own labels are German-project Chinese. */

const mul = (a) => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const sig = (x) => 1 / (1 + Math.exp(-x));
const sum = (a) => a.reduce((x, y) => x + y, 0);
const std = (col) => { const m = sum(col) / col.length; const v = sum(col.map((x) => (x - m) ** 2)) / Math.max(col.length - 1, 1); return [m, Math.sqrt(v)]; };
const standardize = (mat) => { const F = mat[0].length; const out = mat.map((r) => r.slice()); for (let f = 0; f < F; f++) { const col = mat.map((r) => r[f]); const [m, s] = std(col); if (s > 1e-8) for (let i = 0; i < mat.length; i++) out[i][f] = (col[i] - m) / s; } return out; };
const round = (x, n = 4) => Math.round(x * 10 ** n) / 10 ** n;

/* ── English overlay ─────────────────────────────────────────────── */
export const ZH = {
  scenario: {
    'gradientshop.dispatch': ['生产调度输入', '配置任务数量、优化目标和可微排序温度。'],
    'intralogistics.network': ['物料供给网络输入', '配置物料、工位、前置仓、容量和随机种子。'],
    'intralogistics.source': ['Lena 源模型最小实例', '用于与原始模型进行交叉验证的固定小实例。'],
    'intralogistics.sizing': ['载具尺寸策略输入', '配置物料需求波动、提前期和仿真长度。'],
  },
  world: {
    'gradientshop.dispatch': ['可微离散事件调度模型', '用真实离散顺序评分，用 SoftSort 代理提供梯度。'],
    'intralogistics.sizing': ['按物料属性学习载具容量（可微）', '学习 8 个“物料特征 → 载具大小”规则的权重，用流体成本代理训练。'],
    'intralogistics.plan': ['按业务规则构造供料方案（启发式）', '搜索“从哪里拣选、在哪里存放、选哪种载具”，再按固定业务规则补全底层变量。'],
    'intralogistics.static': ['直接填写并校验全部 Lena 变量', '不自动优化；用 NumPy 计算已给方案的成本和 24 类约束违反。'],
    'intralogistics.milp': ['用完整 MILP 优化供料网络（Pyomo）', '让 HiGHS、SCIP、CBC 或 Gurobi 求解 Lena 的完整目标与约束，并报告是否证明最优。'],
  },
  solver: {
    random: ['随机搜索', '从初始方案附近反复随机采样，始终保留目前最好的方案。'],
    hillclimb: ['爬山搜索', '从当前最好方案产生邻居，只接受严格改善；可能停在局部最优。'],
    bayes: ['贝叶斯优化', '用高斯过程学习“决策—成本”关系，再选择最有希望的向量决策。'],
    gradient: ['梯度下降', '沿模型给出的下降方向移动，并用回溯线搜索自动缩小步长。'],
    exact: ['MILP 全局优化', '用下方选定的求解引擎执行分支定界/割；只有 optimal 才构成最优性证明。'],
    'greedy-supply': ['物流构造式启发算法', '先按业务规则构造完整配置，再扫描邻域并采用改善最大的移动。'],
  },
  param: {
    n_items: '物料数量', n_resources: '工位数量', n_forward: '前置仓数量', seed: '随机种子',
    tighten: '容量宽松系数', storage_rate_point_of_use: '线边库存储费率',
    demand_cv: '需求波动系数', lead_time: '补货提前期', horizon: '仿真周期数', n_jobs: '任务数量',
    tau: '排序松弛温度 τ', objective: '优化目标', lr: '初始步长', backtrack: '回溯线搜索',
    n_initial: '初始采样点数', max_passes: '最大搜索轮数', max_moves: '每轮邻居上限', backend: 'MILP 求解引擎', time_limit: '时间上限（秒）',
  },
  help: {
    n_items: 'More items means a harder combinatorial problem; exact solution hits a wall around 8.',
    n_resources: 'Each resource carries its own point-of-use store.',
    n_forward: 'Parallel forward stores give relocation routes an alternative.',
    seed: 'Fixes the sampling of demand, value and volume. Reproducible.',
    tighten: 'Below 1 it tightens storage and preparation capacity, until greedy construction is no longer feasible.',
    storage_rate_point_of_use: 'Phase 1 found the source model carries 1e5 here — a placeholder worth 99.994% of the objective that changes no decision.',
    demand_cv: 'The Phase 3 variable: time averaging is exact at CV=0 and degrades from there.',
    lead_time: 'Replenishment lead time, in periods.',
    horizon: 'Periods simulated. Longer is steadier and slower.',
    n_jobs: 'Size of the single-machine sequencing problem.',
    tau: 'SoftSort temperature. As it approaches zero the schedule gets truer and the gradient collapses and flips sign.',
    objective: 'Weighted completion time is reliably optimisable; weighted tardiness is a documented failure mode.',
    lr: '回溯线搜索会在候选方案没有改善时自动缩小步长。', backtrack: '启用后可减少结果对手工步长的依赖。',
    n_initial: '开始拟合高斯过程前先随机评价多少个方案。', max_passes: '局部搜索最多扫描多少轮邻域。',
    max_moves: '每轮最多评价多少个邻居；0 表示全部。',
    backend: 'auto 会按 appsi_highs → highs → SCIP → CBC → Gurobi 的顺序选择本机第一个可用引擎。',
    time_limit: '分支定界到达时限后返回当前可行解；只有终止状态 optimal 才证明最优。',
  },
  capability: {
    C0: '可以计算一个方案的成本和约束', C1: '可以按时间推进动态环境',
    C2a: '可以给出相对于决策变量的梯度',
    C2b: '可以给出约束影子价格（不是搜索方向）',
    C3: '保留完整代数形式，可供精确求解器使用',
  },
};

export const EN = {
  scenario: {
    'gradientshop.dispatch': ['Production scheduling inputs', 'Configure the number of jobs, objective and differentiable sorting temperature.'],
    'intralogistics.network': ['Material-supply network inputs', 'Configure items, stations, forward stores, capacities and random seed.'],
    'intralogistics.source': ['Lena source-model instance', 'The fixed small instance used to cross-check the migrated model.'],
    'intralogistics.sizing': ['Carrier-sizing inputs', 'Configure demand variability, replenishment lead time and simulation horizon.'],
  },
  world: {
    'gradientshop.dispatch': ['Differentiable DES dispatch model', 'Score the exact discrete order and obtain gradients from a SoftSort surrogate.'],
    'intralogistics.sizing': ['Learn carrier sizes from item attributes (differentiable)', 'Learn weights over eight item-feature-to-carrier-size rules using a fluid cost surrogate.'],
    'intralogistics.plan': ['Construct a supply plan with business rules (heuristic)', 'Search pick, storage and carrier choices, then derive the low-level variables with fixed business rules.'],
    'intralogistics.static': ['Enter and validate every Lena variable', "Do not optimise automatically; use NumPy to score a supplied plan and check all 24 constraint families."],
    'intralogistics.milp': ['Optimise the full supply network as a MILP (Pyomo)', "Let HiGHS, SCIP, CBC or Gurobi solve Lena's complete objective and constraints and report whether optimality was proved."],
  },
  solver: {
    random: ['Random search', 'Sample repeatedly around the initial decision and retain the best result.'],
    hillclimb: ['Hill climb', 'Perturb the incumbent and accept strict improvements; may stop at a local optimum.'],
    bayes: ['Bayesian optimisation', 'Fit a Gaussian process to decision-cost observations and select promising vector decisions.'],
    gradient: ['Gradient descent', 'Follow the reported descent direction and shrink the step with backtracking line search.'],
    exact: ['MILP global optimisation', 'Use the selected backend for branch-and-bound/cut; only termination optimal constitutes a proof.'],
    'greedy-supply': ['Constructive supply heuristic', 'Build a complete business-feasible plan, then take the best move in each neighbourhood.'],
  },
  param: {
    n_items: 'Items', n_resources: 'Stations', n_forward: 'Forward stores', seed: 'Random seed',
    tighten: 'Capacity allowance', storage_rate_point_of_use: 'Point-of-use storage rate',
    demand_cv: 'Demand variability', lead_time: 'Replenishment lead time', horizon: 'Simulation periods', n_jobs: 'Jobs',
    tau: 'Relaxation temperature τ', objective: 'Objective', lr: 'Initial step size', backtrack: 'Backtracking line search',
    n_initial: 'Initial design points', max_passes: 'Maximum search passes', max_moves: 'Neighbours per pass', backend: 'MILP solver engine', time_limit: 'Time limit (seconds)',
  },
  help: {
    n_items: 'More items make the combinatorial problem harder.',
    n_resources: 'Each station has its own point-of-use store.',
    n_forward: 'Parallel forward stores create alternative replenishment routes.',
    seed: 'Controls generated demand, value and volume so the case is reproducible.',
    tighten: 'Values below 1 tighten storage and staging capacity.',
    storage_rate_point_of_use: 'Storage cost rate at the point of use.',
    demand_cv: 'Demand coefficient of variation.', lead_time: 'Periods between ordering and replenishment.',
    horizon: 'Number of periods represented by the generated demand sample.',
    n_jobs: 'Number of jobs competing for the machine.',
    tau: 'SoftSort temperature: smaller is more discrete but can weaken the gradient.',
    objective: 'Cost measure minimised by the solver.',
    lr: 'Backtracking shrinks this step whenever a candidate fails to improve.', backtrack: 'Reduces sensitivity to a manually chosen step size.',
    n_initial: 'Random decisions evaluated before fitting the Gaussian process.', max_passes: 'Maximum neighbourhood scans in local search.',
    max_moves: 'Maximum neighbours evaluated per pass; 0 means all.',
    backend: 'Auto selects the first available local engine in this order: appsi_highs, highs, SCIP, CBC, then Gurobi.',
    time_limit: 'Return the current feasible incumbent at the deadline; only termination optimal proves optimality.',
  },
  capability: {
    C0: 'Can evaluate the cost and constraints of one decision', C1: 'Can step through a sequential environment',
    C2a: 'Can return a gradient with respect to decision variables',
    C2b: 'Can return constraint shadow prices (not a search direction)',
    C3: 'Preserves full algebra for an exact solver',
  },
};

/* ── the schema, as GET /api/schema returns it ───────────────────── */
const P = (name, kind, def, extra = {}) => ({ name, kind, default: def, label: ZH.param[name] || name, help: ZH.help[name] || '', ...extra });

const NETWORK_PARAMS = [
  P('n_items', 'int', 4, { min: 1, max: 60, step: 1 }),
  P('n_resources', 'int', 2, { min: 1, max: 6, step: 1 }),
  P('n_forward', 'int', 2, { min: 1, max: 3, step: 1 }),
  P('seed', 'int', 0, { min: 0, max: 9999, step: 1 }),
  P('tighten', 'float', 1.0, { min: 0.4, max: 2.0, step: 0.05 }),
  P('storage_rate_point_of_use', 'float', 4.0, { min: 0.1, max: 100000, step: 0.1 }),
];
const SIZING_PARAMS = [
  P('n_items', 'int', 8, { min: 2, max: 40, step: 1 }),
  P('demand_cv', 'float', 0.4, { min: 0, max: 1.2, step: 0.05 }),
  P('lead_time', 'int', 3, { min: 1, max: 10, step: 1 }),
  P('horizon', 'int', 600, { min: 100, max: 3000, step: 100 }),
  P('seed', 'int', 0, { min: 0, max: 9999, step: 1 }),
];
const DISPATCH_PARAMS = [
  P('n_jobs', 'int', 12, { min: 3, max: 40, step: 1 }),
  P('tau', 'float', 0.1, { min: 0.005, max: 5, step: 0.005 }),
  P('objective', 'enum', 'wct', { choices: ['wct', 'tardiness'] }),
  P('seed', 'int', 7, { min: 0, max: 9999, step: 1 }),
];
const sc = (id, params) => ({ id, label: ZH.scenario[id][0], description: ZH.scenario[id][1], params });
const wd = (id, scenario, capabilities, solvers, interactive, inline_max_budget) =>
  ({ id, label: ZH.world[id][0], description: ZH.world[id][1], scenario, capabilities, solvers, interactive, inline_max_budget });
const sv = (id, requires, requires_space, params) =>
  ({ id, label: ZH.solver[id][0], description: ZH.solver[id][1], requires, requires_space, params, seconds_per_evaluation: null });

export const SCHEMA = {
  scenarios: [
    sc('gradientshop.dispatch', DISPATCH_PARAMS),
    sc('intralogistics.network', NETWORK_PARAMS),
    sc('intralogistics.source', []),
    sc('intralogistics.sizing', SIZING_PARAMS),
  ],
  worlds: [
    wd('gradientshop.dispatch', 'gradientshop.dispatch', ['C0', 'C2a'], ['random', 'hillclimb', 'bayes', 'gradient'], true, 2000),
    wd('intralogistics.sizing', 'intralogistics.sizing', ['C0', 'C2a'], ['random', 'hillclimb', 'bayes', 'gradient'], true, 4000),
    wd('intralogistics.plan', 'intralogistics.network', ['C0'], ['random', 'hillclimb', 'greedy-supply'], true, 0),
    wd('intralogistics.static', 'intralogistics.network', ['C0'], ['random', 'hillclimb'], true, 0),
    wd('intralogistics.milp', 'intralogistics.network', ['C0', 'C2b', 'C3'], ['random', 'hillclimb', 'exact'], false, 0),
  ],
  solvers: [
    sv('random', ['C0'], [], [P('seed', 'int', 0, { min: 0, max: 9999 })]),
    sv('hillclimb', ['C0'], [], [P('seed', 'int', 0, { min: 0, max: 9999 })]),
    sv('bayes', ['C0'], ['vectorize'], [P('seed', 'int', 0, { min: 0, max: 9999 }), { name: 'n_initial', kind: 'int', default: 8, label: '初始采样点数', help: '先随机评价多少个方案，再开始拟合高斯过程。', min: 3, max: 32 }]),
    sv('gradient', ['C2a'], [], [{ name: 'lr', kind: 'float', default: 0.3, label: 'Initial step', help: 'Backtracking shrinks it.', min: 0.001, max: 5, step: 0.001 }, { name: 'backtrack', kind: 'bool', default: true, label: 'Backtracking line search', help: 'Turn it off and the result becomes a conclusion about that constant.' }]),
    sv('exact', ['C0', 'C3'], [], [{ name: 'backend', kind: 'enum', default: 'auto', label: 'MILP solver engine', help: 'Auto selects the first available local engine.', choices: ['auto', 'appsi_highs', 'highs', 'scip', 'cbc', 'gurobi'] }, { name: 'time_limit', kind: 'float', default: 60, label: 'Time limit (seconds)', help: 'Only termination optimal proves optimality.', min: 1, max: 600, step: 1 }]),
    sv('greedy-supply', ['C0'], ['moves'], [{ name: 'max_passes', kind: 'int', default: 20, label: 'Max passes', help: '', min: 1, max: 60 }, { name: 'max_moves', kind: 'int', default: 0, label: 'Neighbours per pass', help: '0 = all. On large instances each neighbour costs a full evaluation.', min: 0, max: 400 }]),
  ],
  capabilities: ZH.capability,
  limits: { max_concurrent_jobs: 2, default_timeout_seconds: 300 },
};

/* ── sizing world: 8 named rules over a generated item set ───────── */
export const SIZING_RULES = ['HIGH_TURNOVER', 'LOW_VALUE', 'SMALL_PART', 'LONG_LEAD', 'VOLATILE', 'EOQ', 'VALUE_DENSITY', 'FLAT'];
const CAP = [5, 80];

function sizingItems(p) {
  const r = mul(p.seed | 0), n = p.n_items | 0, it = [];
  for (let i = 0; i < n; i++) it.push({ item: `A${i + 1}`, demand_rate: r() * 12 + 2, unit_value: r() * 10 + 0.5, unit_area: r() * 0.3 + 0.05, lead_time: p.lead_time, demand_cv: p.demand_cv });
  return it;
}
function sizingFeatures(items) {
  return standardize(items.map((it) => [it.demand_rate, -it.unit_value, -it.unit_area, it.lead_time, it.demand_cv,
    Math.sqrt(it.demand_rate / Math.max(it.unit_value, 1e-8)), -(it.unit_value / Math.max(it.unit_area, 1e-8)), 1]));
}
function sizingCost(p, w) {
  const items = sizingItems(p), X = sizingFeatures(items);
  const caps = X.map((row) => CAP[0] + (CAP[1] - CAP[0]) * sig(sum(row.map((x, f) => x * w[f]))));
  let handling = 0, holding = 0, space = 0, area = 0;
  items.forEach((it, i) => {
    const safety = 1.65 * it.demand_cv * it.demand_rate * Math.sqrt(it.lead_time);
    handling += 12 * (it.demand_rate / caps[i]);
    holding += 0.25 * it.unit_value * (caps[i] / 2 + safety);
    space += 4 * it.unit_area * caps[i];
    area += it.unit_area * caps[i];
  });
  const limit = 0.42 * sum(items.map((it) => it.unit_area * CAP[1]));
  return { items, caps, kpis: { cost_handling: handling, cost_holding: holding, cost_space: space }, objective: handling + holding + space, area, limit };
}
function sizingEvaluate(p, w) {
  const c = sizingCost(p, w);
  const excess = Math.max(0, c.area - c.limit);
  const violations = excess > 1e-9 ? { '03_storage_area': excess } : {};
  return {
    objective: round(c.objective, 4), kpis: Object.fromEntries(Object.entries(c.kpis).map(([k, v]) => [k, round(v, 3)])),
    violations: Object.fromEntries(Object.entries(violations).map(([k, v]) => [k, round(v, 3)])),
    feasible: excess <= 1e-9, total_violation: round(excess, 4),
    provenance: 'intralogistics/sizing-policy (8 named rules, time-averaged)',
    sensitivity: null,
    text: `objective ${round(c.objective, 2)} · mean carrier ${round(sum(c.caps) / c.caps.length, 1)} units · area ${round(c.area, 1)}/${round(c.limit, 1)}`,
  };
}
function sizingGradient(p, w) {
  const h = 1e-4, g = w.map((_, f) => {
    const a = w.slice(); a[f] += h; const b = w.slice(); b[f] -= h;
    return (sizingCost(p, a).objective - sizingCost(p, b).objective) / (2 * h);
  });
  const norm = Math.hypot(...g) || 1;
  return {
    sensitivity: { kind: 'pathwise', names: SIZING_RULES, values: g.map((v) => round(v, 5)), text: pathwiseText(SIZING_RULES, g) },
    descent_direction: g.map((v) => round(-v / norm, 6)),
  };
}
function pathwiseText(names, g) {
  const order = names.map((n, i) => [n, g[i]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const dir = ([n, v]) => `${v > 0 ? 'lower' : 'raise'} ${n} (∂=${round(v, 3)})`;
  return `pathwise gradient, ‖∇‖=${round(Math.hypot(...g), 4)}\nstrongest: ${dir(order[0])}, then ${dir(order[1])}\nweakest: ${order[order.length - 1][0]} (∂=${round(order[order.length - 1][1], 4)}) — a rule the cost barely reads`;
}

/* ── dispatch world: sequencing, relaxed pairwise ────────────────── */
export const DISPATCH_RULES = ['FIFO', 'SPT', 'LPT', 'WSPT', 'EDD', 'MINSLACK', 'WEIGHT', 'CR'];
function dispatchJobs(p) {
  const r = mul(p.seed | 0), n = p.n_jobs | 0, proc = [], wt = [], sco = [];
  for (let i = 0; i < n; i++) { proc.push(r() + 0.2); wt.push(r() + 0.2); sco.push((r() * 2 - 1) * 1.4); }
  const due = []; let acc = 0;
  proc.forEach((x) => { acc += x; due.push(acc * 0.6 + 1); });
  return { proc, wt, due, sco, n };
}
function dispatchFeatures(j) {
  return standardize(j.proc.map((p_, i) => [-i, -p_, p_, j.wt[i] / p_, -j.due[i], -(j.due[i] - p_), j.wt[i], -j.due[i] / p_]));
}
const scoresOf = (j, w) => dispatchFeatures(j).map((row) => sum(row.map((x, f) => x * w[f])));
function exactCost(j, s, objective) {
  const order = s.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
  let t = 0, cost = 0;
  order.forEach((i) => { t += j.proc[i]; cost += objective === 'tardiness' ? j.wt[i] * Math.max(0, t - j.due[i]) : j.wt[i] * t; });
  return cost;
}
function softCost(j, s, tau, objective) {
  let cost = 0;
  for (let i = 0; i < j.n; i++) {
    let c = j.proc[i];
    for (let k = 0; k < j.n; k++) if (k !== i) c += j.proc[k] * sig((s[k] - s[i]) / tau);
    cost += objective === 'tardiness' ? j.wt[i] * Math.max(0, c - j.due[i]) : j.wt[i] * c;
  }
  return cost;
}
function dispatchEvaluate(p, w) {
  const j = dispatchJobs(p), s = scoresOf(j, w);
  const exact = exactCost(j, s, p.objective), soft = softCost(j, s, p.tau, p.objective);
  return {
    objective: round(exact, 4),
    kpis: { exact_cost: round(exact, 3), soft_cost: round(soft, 3), gap: round(soft - exact, 3), makespan: round(sum(j.proc), 3) },
    violations: {}, feasible: true, total_violation: 0,
    provenance: `gradientshop/dispatch (SoftSort τ=${p.tau}, ${p.objective})`,
    sensitivity: null,
    text: `exact ${round(exact, 2)} · relaxed ${round(soft, 2)} · gap ${round(100 * (soft - exact) / exact, 2)}%`,
  };
}
function dispatchGradient(p, w) {
  const j = dispatchJobs(p), h = 1e-4;
  const f = (v) => softCost(j, scoresOf(j, v), p.tau, p.objective);
  const g = w.map((_, i) => { const a = w.slice(); a[i] += h; const b = w.slice(); b[i] -= h; return (f(a) - f(b)) / (2 * h); });
  const norm = Math.hypot(...g) || 1;
  return { sensitivity: { kind: 'pathwise', names: DISPATCH_RULES, values: g.map((v) => round(v, 5)), text: pathwiseText(DISPATCH_RULES, g) }, descent_direction: g.map((v) => round(-v / norm, 6)) };
}
function tauSweep(p, taus) {
  const j = dispatchJobs(p), w = new Array(8).fill(0).map((_, i) => (i === 3 ? 1 : 0.05));
  const s = scoresOf(j, w), exact = exactCost(j, s, p.objective);
  const gradAt = (tau) => { const h = 1e-4; const f = (v) => softCost(j, scoresOf(j, v), tau, p.objective); return w.map((_, i) => { const a = w.slice(); a[i] += h; const b = w.slice(); b[i] -= h; return (f(a) - f(b)) / (2 * h); }); };
  const ref = gradAt(Math.min(...taus));
  const rows = taus.map((tau) => {
    const soft = softCost(j, s, tau, p.objective), g = gradAt(tau);
    const dot = sum(g.map((v, i) => v * ref[i])), cos = dot / ((Math.hypot(...g) || 1e-12) * (Math.hypot(...ref) || 1e-12));
    let pairs = 0, un = 0;
    for (let a = 0; a < j.n; a++) for (let b = a + 1; b < j.n; b++) { pairs++; const v = sig((s[a] - s[b]) / tau); if (v > 0.1 && v < 0.9) un++; }
    const norm = Math.hypot(...g) || 1;
    const stepped = w.map((v, i) => v - 0.35 * g[i] / norm);
    const after = exactCost(j, scoresOf(j, stepped), p.objective);
    return { tau, soft_cost: round(soft, 4), gap: round(soft - exact, 4), gap_pct: round(100 * (soft - exact) / exact, 3), grad_cosine: round(cos, 4), unresolved_fraction: round(un / pairs, 4), descends: after < exact - 1e-9, improvement: round(exact - after, 4) };
  });
  return { exact_cost: round(exact, 4), rows, params: p, seconds: 0.011, note: 'grad_cosine and descends contradict each other on purpose — the two gradients being compared share a Jacobian, and agree exactly where both vanish (D15).' };
}

/* ── network / plan world ────────────────────────────────────────── */
function networkSummary(p) {
  const I = Array.from({ length: p.n_items }, (_, i) => `A${i + 1}`);
  const R = Array.from({ length: p.n_resources }, (_, i) => `R${i + 1}`);
  const F = Array.from({ length: p.n_forward }, (_, i) => `VZ${i + 1}`);
  const prod = R.map((r) => `PU-${r}`), L = [...F, ...prod, 'ZL'], B = ['PL1', 'PL2', 'KLT'];
  const edges = L.length * (L.length - 1);
  return { id: `network(n=${p.n_items}, r=${p.n_resources}, f=${p.n_forward}, seed=${p.seed}, tighten=${p.tighten})`, items: I, resources: R, warehouses: L, carriers: B, points_of_use: prod, intermediate: [...F, 'ZL'], relocation_variables: I.length * R.length * edges * B.length ** 2 };
}
function planEvaluate(p) {
  const s = networkSummary(p), r = mul((p.seed | 0) + 17);
  const scale = s.items.length * s.resources.length;
  const storage = 40 + 9.2 * scale * (0.8 + 0.4 * r());
  const reloc = 12 + 4.1 * scale * (0.6 + 0.6 * r());
  const handling = 8 + 2.4 * scale;
  const pou = p.storage_rate_point_of_use * 0.011 * scale;
  const slack = (p.tighten - 0.6) * 10;
  const v = {};
  if (slack < 0) { v['01_storage_capacity'] = round(-slack * 3.2 * (1 + 0.1 * scale), 3); v['07_preparation_area'] = round(-slack * 1.4, 3); }
  if (p.tighten < 0.52) v['11_kanban_two_carriers'] = round((0.52 - p.tighten) * 12.5, 3);
  const total = sum(Object.values(v));
  return {
    objective: round(storage + reloc + handling + pou, 3),
    kpis: { cost_storage: round(storage, 2), cost_reloc: round(reloc, 2), cost_handling: round(handling, 2), cost_point_of_use: round(pou, 2) },
    violations: v, feasible: total === 0, total_violation: round(total, 3),
    provenance: 'intralogistics/static (T0, time-averaged) via greedy construction',
    sensitivity: null,
    text: `${s.items.length}×${s.resources.length} configuration · ${s.relocation_variables.toLocaleString()} relocation variables · ${total === 0 ? 'feasible' : 'INFEASIBLE, ' + Object.keys(v).length + ' constraint families breached'}`,
  };
}

/* ── the endpoints ───────────────────────────────────────────────── */
const worldOf = (id) => SCHEMA.worlds.find((w) => w.id === id);
const scenarioOf = (id) => SCHEMA.scenarios.find((s) => s.id === id);
export function resolveParams(scenarioId, params) {
  const spec = scenarioOf(scenarioId); const out = {};
  for (const p of spec.params) {
    let v = params && params[p.name] !== undefined ? params[p.name] : p.default;
    if (p.kind === 'int') v = Math.min(Math.max(Math.round(+v), p.min), p.max);
    else if (p.kind === 'float') v = Math.min(Math.max(+v, p.min), p.max);
    else if (p.kind === 'bool') v = !!v;
    out[p.name] = v;
  }
  return out;
}
const initialWeights = (n) => Array.from({ length: n }, (_, i) => round(0.05 + 0.3 * i, 3));

export const sample = {
  schema: () => SCHEMA,
  health: () => ({ ok: true, worlds: SCHEMA.worlds.length, solvers: SCHEMA.solvers.length, jobs: 0 }),
  scenario: ({ scenario, params }) => {
    const rp = resolveParams(scenario, params);
    const summary = scenario === 'intralogistics.sizing'
      ? { id: 'intralogistics/sizing-policy (8 named rules)', rules: SIZING_RULES, items: sizingItems(rp).map((it) => ({ item: it.item, demand_rate: round(it.demand_rate, 3), unit_value: round(it.unit_value, 3), unit_area: round(it.unit_area, 4), demand_cv: round(it.demand_cv, 3) })), capacity_range: CAP }
      : scenario === 'gradientshop.dispatch'
        ? (() => { const j = dispatchJobs(rp); return { id: `dispatch(n=${rp.n_jobs}, tau=${rp.tau}, ${rp.objective})`, n_jobs: rp.n_jobs, tau: rp.tau, objective: rp.objective, jobs: j.proc.map((x, i) => ({ job: `J${i + 1}`, processing: round(x, 4), weight: round(j.wt[i], 4), due: round(j.due[i], 4) })) }; })()
        : networkSummary(rp);
    return { scenario, params: rp, summary };
  },
  initial: ({ world, scenario }) => {
    const w = worldOf(world), rp = resolveParams(w.scenario, scenario);
    if (world === 'intralogistics.sizing') { const d = initialWeights(8); return { world, scenario_params: rp, capabilities: w.capabilities, decision: d, evaluation: sizingEvaluate(rp, d), describe: { rules: SIZING_RULES, capacity_range: CAP } }; }
    if (world === 'gradientshop.dispatch') { const d = initialWeights(8); return { world, scenario_params: rp, capabilities: w.capabilities, decision: d, evaluation: dispatchEvaluate(rp, d), describe: { rules: DISPATCH_RULES, tau: rp.tau, objective: rp.objective } }; }
    return { world, scenario_params: rp, capabilities: w.capabilities, decision: {}, evaluation: planEvaluate(rp) };
  },
  evaluate: ({ world, scenario, decision }) => {
    const w = worldOf(world), rp = resolveParams(w.scenario, scenario);
    const t = performance.now();
    const evaluation = world === 'intralogistics.sizing' ? sizingEvaluate(rp, decision || initialWeights(8))
      : world === 'gradientshop.dispatch' ? dispatchEvaluate(rp, decision || initialWeights(8)) : planEvaluate(rp);
    return { evaluation, seconds: round((performance.now() - t) / 1000, 4) };
  },
  gradient: ({ world, scenario, decision }) => {
    const w = worldOf(world), rp = resolveParams(w.scenario, scenario);
    if (!w.capabilities.includes('C2a')) throw new Error(`400 · ${world} does not expose C2a; it offers ${w.capabilities.join(', ')}`);
    return world === 'intralogistics.sizing' ? sizingGradient(rp, decision || initialWeights(8)) : dispatchGradient(rp, decision || initialWeights(8));
  },
  tau: ({ scenario, taus }) => tauSweep(resolveParams('gradientshop.dispatch', scenario), taus),
  solve: ({ world, scenario, solver, budget }) => {
    if (solver === 'exact') throw new Error('Exact branch-and-bound requires the live Python backend; offline sample mode cannot issue an optimality proof.');
    const w = worldOf(world), rp = resolveParams(w.scenario, scenario);
    const grad = world === 'intralogistics.sizing' ? (d) => sizingGradient(rp, d) : (d) => dispatchGradient(rp, d);
    const score = world === 'intralogistics.sizing' ? (d) => sizingEvaluate(rp, d).objective
      : world === 'gradientshop.dispatch' ? (d) => dispatchEvaluate(rp, d).objective
        : (d) => planEvaluate(rp).objective * (1 - 0.0006 * (d.step || 0));
    const dim = 8, r = mul(((rp.seed | 0) + solver.length) | 0);
    let cur = initialWeights(dim), best = score(cur), initial = best;
    const history = [best], cap = Math.min(budget, 400);
    for (let k = 1; k < cap; k++) {
      let cand;
      if (solver === 'gradient' && w.capabilities.includes('C2a')) { const d = grad(cur).descent_direction; cand = cur.map((v, i) => v + (0.3 / (1 + k / 40)) * d[i]); }
      else if (solver === 'random') cand = Array.from({ length: dim }, () => (r() * 4 - 1.5));
      else if (solver === 'bayes') cand = k < 8 ? Array.from({ length: dim }, () => (r() * 4 - 1.5)) : cur.map((v) => v + (r() - 0.5) * 0.55 * Math.exp(-k / 90));
      else cand = cur.map((v) => v + (r() - 0.5) * 0.6);
      const s = score(cand);
      if (s < best) { best = s; cur = cand; }
      history.push(round(best, 4));
    }
    const evaluation = world === 'intralogistics.sizing' ? sizingEvaluate(rp, cur) : world === 'gradientshop.dispatch' ? dispatchEvaluate(rp, cur) : planEvaluate(rp);
    return { score: round(best, 4), initial_score: round(initial, 4), improvement: round(initial - best, 4), evaluations_used: history.length, history, evaluation, decision: cur.map((v) => round(v, 4)), summary: `${solver} on ${world}: ${round(initial, 2)} → ${round(best, 2)} in ${history.length} evaluations` };
  },
};

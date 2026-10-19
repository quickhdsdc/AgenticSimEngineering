"""
P1b — 用 Newton 跑与 PhysX 完全同一个四杆 benchmark，回答"用工具还是造工具"。

判据与 validate_closed_loop.py 逐字相同（IFToMM A03/A04 风格）：
  1. 无阻尼、无驱动、从静止释放，5 秒内**机械能应当守恒**
  2. 摇杆近端必须一直钉在机架点 D 上，**闭环残差应为 0**

PhysX 基线（本机实测，dt=2 ms，32/4 迭代）：能量漂移 7.08%，闭环残差 7.5 μm。

  漂移 < 1e-3     -> 直接用 Newton，不必自写求解器
  仍是百分之几    -> GPU 物理生态共性问题，自写辛积分器才成为真正的贡献

用法（不需要启动 Isaac Sim，Newton 是纯 Python + Warp 包）：
  projects/isaac_sim/.venv/Scripts/python.exe -m mbs_isaac.validate_closed_loop_newton
  ... --backend mujoco --integrator rk4 --dt 0.002
  ... --sweep            # 跑完整对比矩阵并写出汇总表
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def build(backend: str, dt: float, close_loops: bool, device: str, **kw):
    """返回 (newton_model, solver, spec_model, info)。"""
    import newton

    from .closed_loop import four_bar
    from .newton_backend import to_newton

    spec_model = four_bar()
    nm, info = to_newton(spec_model, close_loops=close_loops, device=device,
                         eq_solref=kw.pop("eq_solref", None),
                         eq_solimp=kw.pop("eq_solimp", None))

    if backend == "featherstone":
        # angular_damping 默认 0.05 —— 在守恒系统的能量测试里这是个**必须清零**
        # 的隐藏耗散项，忘了清就会把求解器的账算到别人头上。
        solver = newton.solvers.SolverFeatherstone(nm, angular_damping=0.0)
    elif backend in ("mujoco", "mujoco_cpu"):
        solver = newton.solvers.SolverMuJoCo(
            nm,
            integrator=kw.get("integrator", "implicitfast"),
            solver=kw.get("mj_solver", "newton"),
            iterations=kw.get("iterations", 100),
            ls_iterations=kw.get("ls_iterations", 50),
            tolerance=kw.get("tolerance", 1e-10),
            disable_contacts=True,
            use_mujoco_cpu=(backend == "mujoco_cpu"),
            # 【实测第一大误差源，且完全是"管道"问题不是物理问题】
            # 默认 update_data_interval=1：**每一步**都把 Newton 的状态推回 MuJoCo 的
            # Data。而 newton.State.body_q / joint_q 是 **float32** 数组，于是每步都
            # 把 float64 的积分状态截断一次。这个扰动的大小与 dt 无关，所以它
            # (a) 造出一个减小 dt 也下不去的"地板"，(b) dt 越小步数越多、反而更差。
            # 实测 dt=0.5 ms：改成 0 之后轨迹 RMS 从 1.07e-3 降到 1.44e-5 rad，
            # 二阶收敛也恢复了。
            # 设为 0 的前提：步与步之间没有任何东西往 Newton 状态里写。
            # 本 benchmark 无驱动无外力，成立；采集数据要施加力矩时必须重新确认。
            update_data_interval=kw.get("update_data_interval", 0),
        )
    elif backend == "semi_implicit":
        solver = newton.solvers.SolverSemiImplicit(nm, angular_damping=0.0)
    else:
        raise SystemExit(f"unknown backend {backend!r}")
    return nm, solver, spec_model, info


def run(backend="mujoco", dt=0.002, seconds=5.0, close_loops=True,
        device="cuda:0", verbose=True, **kw):
    import newton

    from .newton_backend import energy_and_residual

    steps = int(round(seconds / dt))
    nm, solver, spec_model, info = build(backend, dt, close_loops, device, **kw)
    A, B, C, D = (spec_model.assembly[k] for k in "ABCD")

    state_0, state_1 = nm.state(), nm.state()
    control = nm.control()
    newton.eval_fk(nm, nm.joint_q, nm.joint_qd, state_0)

    # ---- 装配自检：step 之前先量几何 ----
    # 方法论（见 docs/tutorial/05）：t=0 就不对，说明模型压根没按你以为的方式装配，
    # 这时候看 5 秒的能量曲线毫无意义。
    q0 = state_0.body_q.numpy()
    asm_err = 0.0
    for name, bi in info["body_idx"].items():
        s = {l.name: l for l in spec_model.links}[name]
        if s.init_pos is not None:
            asm_err = max(asm_err, float(np.linalg.norm(q0[bi, :3] - np.asarray(s.init_pos))))
    E0, d0 = energy_and_residual(nm, state_0, spec_model, info["body_idx"], D, "rocker")

    if verbose:
        print(f"\n=== backend={backend} dt={dt} steps={steps} "
              f"loop={'closed' if close_loops else 'OPEN (tree only)'} "
              f"device={device} {kw} ===")
        print(f"  bodies={len(info['body_idx'])} (folded into world: {info['static_links']}), "
              f"tree DoF={info['tree_dof']}, equality constraints={info['eq_count']}")
        print(f"  assembly check vs spec init poses: max |dp| = {asm_err:.3e} m")
        print(f"  initial: E={E0:.6f} J, loop residual={d0:.3e} m")

    Es, Ds = np.zeros(steps), np.zeros(steps)
    Qs = np.zeros((steps, int(nm.joint_coord_count)))
    t0 = time.time()
    for i in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        Es[i], Ds[i] = energy_and_residual(nm, state_0, spec_model,
                                           info["body_idx"], D, "rocker")
        Qs[i] = state_0.joint_q.numpy()
    wall = time.time() - t0

    t = (np.arange(steps) + 1) * dt
    finite = np.isfinite(Es)
    drift_rel = np.abs(Es[finite] - E0).max() / max(abs(E0), 1e-9) if finite.any() else float("inf")
    res_max = np.nanmax(Ds) if np.isfinite(Ds).any() else float("inf")

    if verbose:
        for h in (0.5, 1.0, 2.0, 5.0):
            if h > seconds:
                continue
            k = min(int(h / dt) - 1, steps - 1)
            print(f"  t={h:>4.1f}s: E={Es[k]:10.5f} J (drift {Es[k]-E0:+.3e} J, "
                  f"{abs(Es[k]-E0)/max(abs(E0),1e-9)*100:7.4f}%) | residual {Ds[k]:.3e} m")
        print(f"  max |energy drift| = {drift_rel*100:.4f}%   crank sweep = {Qs[:,0].max()-Qs[:,0].min():.3f} rad   "
              f"max loop residual = {res_max:.3e} m   [{wall:.1f}s wall]")

    return dict(backend=backend, dt=dt, seconds=seconds, closed=close_loops,
                drift_rel=float(drift_rel), residual_max=float(res_max),
                E0=float(E0), assembly_err=float(asm_err), wall=wall,
                t=t, energy=Es, residual=Ds, joint_q=Qs, opts=kw)


PHYSX_BASELINE = dict(drift_rel=0.0708, residual_max=7.5e-6)


def sweep(seconds, device, dt=0.002):
    """P1b 的完整对比矩阵。

    分两组：
      A. 开链三连杆（丢掉闭环约束）—— 只考验积分器，把"约束投影的耗散"隔离掉
      B. 闭环四杆 —— 与 PhysX 基线同题
    两组的初始能量都是同一个 8.885 J（同一份 spec），所以数字直接可比。
    """
    tight = (2 * dt, 1.0)  # 约束时间常数收紧到 2 步（MuJoCo 内部也钳在这里）
    # 再把约束阻抗下限 d0 从默认 0.9 提到 0.999 —— 这两个旋钮是 Newton 侧
    # 唯一需要调的东西，默认值是为接触碰撞调的，对闭环机构太软。
    hard = dict(eq_solref=tight, eq_solimp=(0.999, 0.99999, 1e-5, 0.5, 2.0))
    cases = [
        # ---- A 组：只看积分器 ----
        dict(backend="featherstone", close_loops=False, tag="A tree | Featherstone symplectic Euler"),
        dict(backend="mujoco", close_loops=False, integrator="euler", tag="A tree | mjWarp euler"),
        dict(backend="mujoco", close_loops=False, integrator="implicitfast", tag="A tree | mjWarp implicitfast"),
        dict(backend="mujoco", close_loops=False, integrator="rk4", tag="A tree | mjWarp RK4"),
        dict(backend="mujoco_cpu", close_loops=False, integrator="rk4", tag="A tree | MuJoCo-C RK4 (f64)"),
        # ---- B 组：闭环，与 PhysX 同题 ----
        dict(backend="mujoco", close_loops=True, integrator="euler", tag="B loop | mjWarp euler"),
        dict(backend="mujoco", close_loops=True, integrator="implicitfast", tag="B loop | mjWarp implicitfast"),
        dict(backend="mujoco", close_loops=True, integrator="rk4", tag="B loop | mjWarp RK4"),
        dict(backend="mujoco", close_loops=True, integrator="rk4", eq_solref=tight,
             tag="B loop | mjWarp RK4, solref=2dt"),
        dict(backend="mujoco", close_loops=True, integrator="rk4", **hard,
             tag="B loop | mjWarp RK4, hard eq"),
        dict(backend="mujoco_cpu", close_loops=True, integrator="rk4", tag="B loop | MuJoCo-C RK4 (f64)"),
        dict(backend="mujoco_cpu", close_loops=True, integrator="rk4", **hard,
             tag="B loop | MuJoCo-C RK4 (f64), hard eq"),
        # Featherstone 是减缩坐标法，文档明说不支持 equality constraint —— 跑一次证实
        dict(backend="featherstone", close_loops=True, tag="B loop | Featherstone (no eq support)"),
    ]
    rows = []
    for c in cases:
        tag = c.pop("tag")
        try:
            r = run(seconds=seconds, dt=dt, device=device, verbose=True, **c)
        except Exception as e:  # noqa: BLE001 - 结果的一部分：这个组合跑不通
            print(f"  !! {type(e).__name__}: {e}")
            r = dict(backend=c["backend"], closed=c.get("close_loops", True),
                     drift_rel=float("nan"), residual_max=float("nan"),
                     error=f"{type(e).__name__}: {e}")
        r["tag"] = tag
        rows.append(r)
    _print_table(rows, seconds, dt)
    return rows


def _print_table(rows, seconds, dt):
    print("\n" + "=" * 96)
    print(f"P1b RESULT - four-bar benchmark, {seconds} s, undamped/undriven, dt={dt*1000:g} ms")
    print("=" * 96)
    print(f"{'case':<46} {'energy drift':>14} {'loop residual':>18}")
    print("-" * 96)
    print(f"{'PhysX (Isaac 6.0, baseline)  loop':<46} {7.08:>13.4f}% {7.66e-6:>16.2e} m")
    for r in rows:
        if "error" in r:
            print(f"{r['tag']:<46} {'--':>14}   {r['error'][:44]}")
        else:
            res = "n/a (tree, open)" if not r["closed"] else f"{r['residual_max']:.2e} m"
            print(f"{r['tag']:<46} {r['drift_rel']*100:>13.4f}% {res:>18}")
    print("=" * 96)


def dt_scan(backend, integrator, close_loops, seconds, device, dts, eq_solref_factor=None):
    """能量漂移随 dt 的收敛阶——判断误差是不是积分器控制的。"""
    rows = []
    for dt in dts:
        kw = dict(integrator=integrator) if backend.startswith("mujoco") else {}
        if eq_solref_factor:
            kw["eq_solref"] = (eq_solref_factor * dt, 1.0)
        r = run(backend=backend, dt=dt, seconds=seconds, close_loops=close_loops,
                device=device, verbose=False, **kw)
        rows.append(r)
        print(f"  dt={dt*1000:>6.3f} ms -> drift {r['drift_rel']*100:>9.5f}%  "
              f"residual {r['residual_max']:.3e} m")
    print("  observed order (successive halvings):",
          ", ".join(f"{np.log2(rows[i]['drift_rel']/rows[i+1]['drift_rel']):.2f}"
                    for i in range(len(rows) - 1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mujoco",
                    choices=["mujoco", "mujoco_cpu", "featherstone", "semi_implicit"])
    ap.add_argument("--integrator", default="implicitfast", choices=["euler", "rk4", "implicitfast"])
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--open-loop", action="store_true", help="drop the loop constraint (tree only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dt-scan", action="store_true",
                    help="energy drift vs dt (convergence order)")
    a = ap.parse_args()

    from .paths import result_path

    if a.dt_scan:
        dts = [0.004, 0.002, 0.001, 0.0005, 0.00025]
        print(f"\n--- dt scan: {a.backend}/{a.integrator}, "
              f"{'loop' if not a.open_loop else 'tree'}, {a.seconds}s ---")
        rows = dt_scan(a.backend, a.integrator, not a.open_loop, a.seconds, a.device, dts,
                       eq_solref_factor=(2.0 if not a.open_loop else None))
        out = result_path(f"newton_four_bar_dtscan_{a.backend}_{a.integrator}.npz")
        np.savez(out, dts=np.array(dts),
                 drift=np.array([r["drift_rel"] for r in rows]),
                 residual=np.array([r["residual_max"] for r in rows]))
        print(f"saved -> {out}")
        return

    if a.sweep:
        rows = sweep(a.seconds, a.device, dt=a.dt)
        out = result_path("newton_four_bar_sweep.npz")
        np.savez(out, **{f"{i}_{r.get('tag','?')}": r.get("energy", np.array([]))
                         for i, r in enumerate(rows)},
                 summary=json.dumps([{k: v for k, v in r.items()
                                      if k in ("tag", "backend", "closed", "drift_rel",
                                               "residual_max", "E0", "wall", "error")}
                                     for r in rows]))
        print(f"saved -> {out}")
        return

    kw = {}
    if a.backend.startswith("mujoco"):
        kw["integrator"] = a.integrator
    r = run(backend=a.backend, dt=a.dt, seconds=a.seconds,
            close_loops=not a.open_loop, device=a.device, **kw)
    out = result_path(f"newton_four_bar_{a.backend}.npz")
    np.savez(out, t=r["t"], energy=r["energy"], residual=r["residual"], E0=r["E0"])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

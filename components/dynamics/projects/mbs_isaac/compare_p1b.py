"""
P1b 汇总：PhysX vs Newton vs 参考解，把"能量漂移"翻译成"轨迹误差"。

三条线放在同一根轴上（曲柄角 theta）：
  * 参考解  reference_four_bar.py     最小坐标 + DOP853 rtol=1e-12（float64）
  * PhysX   validate_closed_loop.py   Isaac 6.0，减缩坐标树 + 最大坐标闭环约束
  * Newton  validate_closed_loop_newton.py   同一份 spec，MuJoCo 后端 + equality 约束

前置：先各跑一次生成 npz
  run_mbs.bat validate_closed_loop 32 4 0.002          (需要 Isaac，约 1 分钟)
  python -m mbs_isaac.reference_four_bar               (秒级)
本脚本会自己现跑 Newton 那几个算例。
"""
from __future__ import annotations

import math

import numpy as np

THETA0 = math.pi / 2


def _to_theta(q, ref_theta_early):
    """关节坐标 q(t) -> 世界系曲柄角 theta(t)。

    关节角的**正方向**取决于关节帧的定向约定，两个引擎不保证一致，靠猜是不行的。
    这里用数据本身定：取前 50 步，看 theta0+q 和 theta0-q 哪个跟参考解同向。
    两个引擎各自定出来的符号如果不一致，说明确实有约定差异，会打印出来。
    """
    n = min(50, len(q))
    d_ref = ref_theta_early[:n] - THETA0
    err_p = np.abs((THETA0 + q[:n]) - THETA0 - d_ref).sum()
    err_m = np.abs((THETA0 - q[:n]) - THETA0 - d_ref).sum()
    sign = 1.0 if err_p <= err_m else -1.0
    return THETA0 + sign * q, sign


def main():
    from .paths import result_path
    from .validate_closed_loop_newton import run

    ref = np.load(result_path("four_bar_reference.npz"))
    t_ref, th_ref = ref["t"], ref["theta"]

    sources = []

    physx = np.load(result_path("four_bar_validation.npz"))
    if "joint_q" not in physx:
        raise SystemExit("four_bar_validation.npz has no joint_q — re-run "
                         "`run_mbs.bat validate_closed_loop 32 4 0.002` with the current script")
    E0p = float(physx["E0"])
    sources.append(dict(
        name="PhysX (Isaac 6.0, 32/4 iters)", t=physx["t"], q=physx["joint_q"][:, 0],
        drift=float(np.abs(physx["energy"] - E0p).max() / abs(E0p)),
        residual=float(physx["drift"].max())))

    dt = 0.002
    # 收紧闭环约束：solref 是约束的时间常数（MuJoCo 内部钳到 >= 2·dt），
    # solimp[0] 是约束阻抗的下限 d0，越接近 1 约束越"硬"。默认 (0.02, 0.9)
    # 对 2 ms 步长来说太软 —— 这是 Newton 侧唯一需要调的两个旋钮。
    HARD = dict(eq_solref=(2 * dt, 1.0), eq_solimp=(0.999, 0.99999, 1e-5, 0.5, 2.0))
    newton_cases = [
        ("Newton mjWarp implicitfast (default)", dict(backend="mujoco", integrator="implicitfast")),
        ("Newton mjWarp RK4 (default eq)", dict(backend="mujoco", integrator="rk4")),
        ("Newton mjWarp RK4 + hard eq", dict(backend="mujoco", integrator="rk4", **HARD)),
        ("Newton MuJoCo-C RK4 f64 + hard eq", dict(backend="mujoco_cpu", integrator="rk4", **HARD)),
    ]
    for name, kw in newton_cases:
        r = run(dt=dt, seconds=float(t_ref[-1]), close_loops=True, verbose=False, **kw)
        sources.append(dict(name=name, t=r["t"], q=r["joint_q"][:, 0],
                            drift=r["drift_rel"], residual=r["residual_max"]))

    print("\n" + "=" * 104)
    print("P1b — four-bar, 5 s, undamped/undriven, dt = 2 ms")
    print("reference: 1-DoF Lagrange + DOP853 (rtol 1e-12, float64); its own energy drift is 7.6e-12")
    print("=" * 104)
    hdr = (f"{'engine / config':<38} {'E drift':>10} {'loop resid':>12} "
           f"{'|dtheta| 1s':>12} {'2s':>10} {'5s':>10} {'RMS':>10}")
    print(hdr)
    print("-" * 104)
    rows = []
    for s in sources:
        th, sign = _to_theta(s["q"], np.interp(s["t"], t_ref, th_ref))
        th_r = np.interp(s["t"], t_ref, th_ref)
        err = np.abs(th - th_r)

        def at(h):
            return err[min(int(h / (s["t"][1] - s["t"][0])) - 1, len(err) - 1)]

        rms = float(np.sqrt((err ** 2).mean()))
        rows.append(dict(name=s["name"], sign=sign, err=err, t=s["t"], theta=th,
                         drift=s["drift"], residual=s["residual"], rms=rms))
        print(f"{s['name']:<38} {s['drift']*100:>9.4f}% {s['residual']:>10.2e} m "
              f"{at(1.0):>12.2e} {at(2.0):>10.2e} {at(5.0):>10.2e} {rms:>10.2e}")
    print("=" * 104)
    signs = {r["name"]: r["sign"] for r in rows}
    print("joint-angle sign convention resolved from data:",
          {k: ("+q" if v > 0 else "-q") for k, v in signs.items()})
    print("units: |dtheta| in rad, absolute crank-angle error against the reference solution")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(t_ref, th_ref, "k-", lw=2.0, label="reference (1-DoF Lagrange, rtol 1e-12)")
        for r in rows:
            ax[0].plot(r["t"], r["theta"], lw=1.0, label=r["name"])
        ax[0].set_ylabel("crank angle theta [rad]")
        ax[0].legend(fontsize=8)
        ax[0].set_title("Four-bar released from rest, 5 s, dt = 2 ms")
        ax[0].grid(alpha=0.3)
        for r in rows:
            ax[1].semilogy(r["t"], np.maximum(r["err"], 1e-12), lw=1.0, label=r["name"])
        ax[1].set_ylabel("|theta - theta_ref| [rad]")
        ax[1].set_xlabel("t [s]")
        ax[1].grid(alpha=0.3, which="both")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = result_path("p1b_four_bar_comparison.png")
        fig.savefig(out, dpi=140)
        print(f"saved -> {out}")
    except ImportError:
        print("(matplotlib not available, skipped the plot)")

    np.savez(result_path("p1b_summary.npz"),
             **{f"theta_{i}": r["theta"] for i, r in enumerate(rows)},
             **{f"err_{i}": r["err"] for i, r in enumerate(rows)},
             names=np.array([r["name"] for r in rows]),
             drift=np.array([r["drift"] for r in rows]),
             residual=np.array([r["residual"] for r in rows]),
             rms=np.array([r["rms"] for r in rows]))


if __name__ == "__main__":
    main()

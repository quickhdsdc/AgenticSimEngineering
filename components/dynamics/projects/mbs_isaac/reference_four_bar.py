"""
四杆机构的**参考解**：1 自由度拉格朗日方程 + 变步长误差控制积分（float64）。

【为什么需要这个】
能量漂移只是精度的**代用指标**。世界模型真正要的是轨迹本身：曲柄角 theta(t)。
把两个引擎和一个"工程级 MBS 求解器该给出的答案"放在同一根轴上，才能把
"PhysX 差 7%"翻译成"5 秒后曲柄角差多少弧度"。

这个参考解扮演传统 MBS 求解器的角色，做法与 ADAMS/SIMPACK 同源：
  * 用**最小坐标**（机构自由度就是 1，取曲柄角 theta）——闭环约束由此
    **恒等满足**，不存在约束漂移，残差恒为 0
  * 拉格朗日方程 M(theta)·theta'' + ½M'(theta)·theta'^2 + V'(theta) = 0
  * DOP853 变步长变阶，rtol=atol=1e-12，全程 float64

几何/惯量参数与 closed_loop.four_bar() 完全一致，跑的是同一个物理系统。

【踩过的坑】M、V 的导数最初用中心差分算，结果 rtol=1e-12 的积分器直接卡死：
差分的舍入噪声约 eps/h ≈ 1e-10，比容差大两个数量级，自适应步长于是无限收缩。
容差比数值噪声还小的时候，**必须给解析导数**。下面 u/v（一阶）和 up/vp（二阶）
就是对闭环方程逐次求导解出来的。

自校验：积分完看能量守恒到多少位。参考解自己守不住能量，就不配当参考。
"""
from __future__ import annotations

import math

import numpy as np


class FourBarLagrange:
    """四杆机构的最小坐标模型。theta = 曲柄相对 +X 的转角（XZ 平面，绕 +Y）。

    杆件方向角：曲柄 theta、连杆 beta、摇杆 gamma。闭环方程（平面复数形式）
        a e^{i theta} + b e^{i beta} = d + c e^{i gamma}
    对 theta 求一阶、二阶导，就得到 u=dbeta/dtheta, v=dgamma/dtheta 及其导数
    满足的两个 2x2 线性方程组（系数矩阵相同，只有右端项不同）。
    """

    def __init__(self, a=0.3, b=1.0, c=0.8, d=1.0, mass_per_m=1.0, g=9.81):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.g = g
        self.m = {"crank": mass_per_m * a, "coupler": mass_per_m * b, "rocker": mass_per_m * c}
        self.L = {"crank": a, "coupler": b, "rocker": c}
        # 绕质心、绕 +Y 的转动惯量：均匀细杆 m L^2 / 12（与 closed_loop._rod 一致）
        self.I = {k: self.m[k] * self.L[k] ** 2 / 12.0 for k in self.m}

    # ---------------------------------------------------------- kinematics --
    def angles(self, theta):
        """闭环位置分析 -> (beta, gamma)。分支取法与 closed_loop.four_bar 相同。"""
        a, b, c, d = self.a, self.b, self.c, self.d
        Bx, Bz = a * math.cos(theta), a * math.sin(theta)
        dx, dz = d - Bx, -Bz
        dist = math.hypot(dx, dz)
        if dist > b + c or dist < abs(b - c):
            raise ValueError(f"four-bar cannot be assembled at theta={theta}")
        t = (b * b - c * c + dist * dist) / (2 * dist)
        h = math.sqrt(max(b * b - t * t, 0.0))
        Mx, Mz = Bx + t * dx / dist, Bz + t * dz / dist
        Cx, Cz = Mx - h * dz / dist, Mz + h * dx / dist
        return math.atan2(Cz - Bz, Cx - Bx), math.atan2(Cz, Cx - d)

    def derivs(self, theta):
        """返回 (beta, gamma, u, v, up, vp)，u=dbeta/dtheta，up=d2beta/dtheta2。"""
        a, b, c = self.a, self.b, self.c
        be, ga = self.angles(theta)
        sT, cT = math.sin(theta), math.cos(theta)
        sB, cB = math.sin(be), math.cos(be)
        sG, cG = math.sin(ga), math.cos(ga)
        # 一阶：[-b sB,  c sG][u]   [ a sT]
        #       [ b cB, -c cG][v] = [-a cT]
        A = np.array([[-b * sB, c * sG], [b * cB, -c * cG]])
        u, v = np.linalg.solve(A, np.array([a * sT, -a * cT]))
        # 二阶：同一个系数矩阵，右端来自把一阶方程再对 theta 求导
        rhs2 = np.array([a * cT + b * cB * u * u - c * cG * v * v,
                         a * sT + b * sB * u * u - c * sG * v * v])
        up, vp = np.linalg.solve(A, rhs2)
        return be, ga, u, v, up, vp

    def _com_and_derivs(self, theta):
        """三根杆的质心位置及其对 theta 的一、二阶导（平面 xz）。"""
        a, b, c, d = self.a, self.b, self.c, self.d
        be, ga, u, v, up, vp = self.derivs(theta)
        eT = np.array([math.cos(theta), math.sin(theta)])
        nT = np.array([-math.sin(theta), math.cos(theta)])
        eB = np.array([math.cos(be), math.sin(be)])
        nB = np.array([-math.sin(be), math.cos(be)])
        eG = np.array([math.cos(ga), math.sin(ga)])
        nG = np.array([-math.sin(ga), math.cos(ga)])
        B = a * eT
        Dv = np.array([d, 0.0])

        com = {"crank": 0.5 * a * eT, "coupler": B + 0.5 * b * eB, "rocker": Dv + 0.5 * c * eG}
        d1 = {"crank": 0.5 * a * nT,
              "coupler": a * nT + 0.5 * b * u * nB,
              "rocker": 0.5 * c * v * nG}
        d2 = {"crank": -0.5 * a * eT,
              "coupler": -a * eT + 0.5 * b * (up * nB - u * u * eB),
              "rocker": 0.5 * c * (vp * nG - v * v * eG)}
        dang = {"crank": 1.0, "coupler": u, "rocker": v}
        d2ang = {"crank": 0.0, "coupler": up, "rocker": vp}
        return com, d1, d2, dang, d2ang

    # ------------------------------------------------------------ dynamics --
    def M_V(self, theta):
        """返回 (M, dM/dtheta, V, dV/dtheta)。T = ½ M(theta) theta'^2。"""
        com, d1, d2, dang, d2ang = self._com_and_derivs(theta)
        M = dM = V = dV = 0.0
        for k in self.m:
            m, I = self.m[k], self.I[k]
            M += m * float(d1[k] @ d1[k]) + I * dang[k] ** 2
            dM += 2.0 * (m * float(d1[k] @ d2[k]) + I * dang[k] * d2ang[k])
            V += m * self.g * com[k][1]
            dV += m * self.g * d1[k][1]
        return M, dM, V, dV

    def energy(self, theta, dtheta):
        M, _, V, _ = self.M_V(theta)
        return 0.5 * M * dtheta ** 2 + V

    def rhs(self, _t, y):
        theta, dtheta = y
        M, dM, _, dV = self.M_V(theta)
        return [dtheta, -(0.5 * dM * dtheta ** 2 + dV) / M]

    def solve(self, theta0=math.pi / 2, dtheta0=0.0, seconds=5.0, n_out=2500,
              rtol=1e-12, atol=1e-12):
        from scipy.integrate import solve_ivp

        t_eval = np.linspace(seconds / n_out, seconds, n_out)
        sol = solve_ivp(self.rhs, (0.0, seconds), [theta0, dtheta0], method="DOP853",
                        t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        E0 = self.energy(theta0, dtheta0)
        E = np.array([self.energy(th, dth) for th, dth in sol.y.T])
        return dict(t=sol.t, theta=sol.y[0], dtheta=sol.y[1], E=E, E0=E0,
                    drift_rel=float(np.abs(E - E0).max() / abs(E0)), nfev=sol.nfev)


def main():
    from .paths import result_path

    fb = FourBarLagrange()
    ref = fb.solve()
    print("reference: 1-DoF Lagrange, DOP853 rtol=atol=1e-12, float64")
    print(f"  E0 = {ref['E0']:.9f} J   (PhysX/Newton report 8.885017 J for the same spec)")
    print(f"  max |energy drift| = {ref['drift_rel']*100:.3e} %   <- self-check")
    print(f"  theta sweep = {ref['theta'].max()-ref['theta'].min():.4f} rad, "
          f"theta(5 s) = {ref['theta'][-1]:.9f} rad, rhs evals = {ref['nfev']}")
    out = result_path("four_bar_reference.npz")
    np.savez(out, t=ref["t"], theta=ref["theta"], dtheta=ref["dtheta"],
             E=ref["E"], E0=ref["E0"])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

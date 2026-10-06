"""projects/control_planning/ilqr.py 的测试 —— 直接运行：``python tests/test_ilqr.py``。

验证教程第 04 章 04.2 的 iLQR 教学实现（Jacobson & Mayne 1970；Tassa et al.,
ICRA 2012）：离散动力学雅可比的解析/中心差分一致性、线性二次特例与 Riccati
代数解的交叉验证（教程 05 章 (5.5)/(5.6)）、代价单调下降、路标点走廊任务、
标称轨迹的动力学精确性、正则化 (4.14) 的自适应行为。全部确定性，无数值噪声。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ilqr import (  # noqa: E402
    ReachCost,
    double_integrator_jacobians,
    double_integrator_step,
    ilqr_solve,
    numeric_dynamics_jacobians,
    riccati_fixed_point,
    riccati_lqr,
)

DT = 0.1
GOAL = np.array([2.0, 1.0])
GAP = np.array([1.0, 1.4])  # "走廊缺口"路标点（教学场景：先绕后到）


def _regulation_cost(w_u: float = 0.3, w_goal: float = 2.0, w_vf: float = 1.0) -> ReachCost:
    """纯二次调节代价（LQR 特例）：l = xᵀQx + uᵀRu，goal = 0。"""
    return ReachCost(w_u=w_u, goal=np.zeros(2), w_goal=w_goal, w_vf=w_vf)


def test_dynamics_jacobians_match_central_difference():
    """解析 (A, B) 与中心差分一致（1e-5 门限：差分误差 ~1e-10，实现错误 > 1e-3）。"""
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=4)
        u = rng.normal(size=2)
        A_a, B_a = double_integrator_jacobians(x, u, DT)
        A_n, B_n = numeric_dynamics_jacobians(double_integrator_step, x, u, DT)
        assert np.allclose(A_a, A_n, atol=1e-5), np.abs(A_a - A_n).max()
        assert np.allclose(B_a, B_n, atol=1e-5), np.abs(B_a - B_n).max()
    # 一步转移与雅可比的定义自洽：f(x+dx, u+du) ≈ f(x,u) + A dx + B du。
    x0 = np.array([0.3, -0.4, 0.2, 0.1])
    u0 = np.array([0.5, -0.2])
    dx = np.array([0.01, -0.02, 0.03, 0.01])
    du = np.array([-0.02, 0.03])
    pred = double_integrator_step(x0, u0, DT) + double_integrator_jacobians(x0, u0, DT)[0] @ dx \
        + double_integrator_jacobians(x0, u0, DT)[1] @ du
    real = double_integrator_step(x0 + dx, u0 + du, DT)
    assert np.allclose(pred, real, atol=1e-5)


def test_lqr_special_case_matches_riccati():
    """线性 + 二次特例下 iLQR 必须精确复现 Riccati 解（教程 05.1 ⑤ 的论证）。

    增益 K_fb 逐时刻等于 Riccati 递推 (5.5) 的 K 的负值（iLQR 的 K 是
    δu = k + Kδx 的符号约定，Riccati 的 u* = -Kx），最优代价 = x₀ᵀP₀x₀。
    本问题下 iLQR 的一切展开都是精确的——取 μ₀ = 1e-9 关闭阻尼，一次
    前向回滚即达最优；代价（R = w_u·I、Q = 0、P_f = diag(w)）的 2 倍
    约定见 ReachCost 的 l_uu = 2R / l_fxx = 2P_f。
    """
    cost = _regulation_cost()
    n = 30
    x0 = np.array([0.5, -0.3, 0.2, -0.1])
    res = ilqr_solve(x0, np.zeros((n, 2)), cost, dt=DT, max_iters=10, mu0=1e-9)
    assert res.converged
    A, B = double_integrator_jacobians(np.zeros(4), np.zeros(2), DT)
    p_diag = np.diag([cost.w_goal, cost.w_goal, cost.w_vf, cost.w_vf])
    p_list, k_list = riccati_lqr(
        A, B, np.zeros((4, 4)), cost.w_u * np.eye(2), n_steps=n, p_f=p_diag
    )
    for i in range(n):
        # iLQR 第 i 步增益 = -Riccati 第 i 步增益（k_list 按反向时间存）。
        assert np.allclose(res.K_fb[i], -k_list[n - 1 - i], atol=1e-8), i
    j_star = float(x0 @ p_list[-1] @ x0)  # J* = x₀ᵀP₀x₀
    assert abs(res.costs[-1] - j_star) < 1e-8 * max(1.0, abs(j_star))
    print(f"\n[lqr] iLQR cost {res.costs[-1]:.8f} vs Riccati J* {j_star:.8f}")


def test_riccati_fixed_point_matches_finite_horizon():
    """有限时域 Riccati 的解当时域拉长后收敛到代数 Riccati 方程 (5.6) 的定点。"""
    A, B = double_integrator_jacobians(np.zeros(4), np.zeros(2), DT)
    Q = np.diag([1.0, 1.0, 0.5, 0.5])
    R = 0.2 * np.eye(2)
    _, k_long = riccati_lqr(A, B, Q, R, n_steps=300)
    p_inf = riccati_fixed_point(A, B, Q, R)
    # 用定常增益 P^∞ 反推有限时域末端的 P：应与长时域 P₀ 一致。
    K_inf = np.linalg.solve(R + B.T @ p_inf @ B, B.T @ p_inf @ A)
    p_check = Q + A.T @ p_inf @ A - A.T @ p_inf @ B @ K_inf
    assert np.allclose(p_check, p_inf, atol=1e-10)  # (5.6) 的不动点性
    assert np.allclose(k_long[-1], K_inf, atol=1e-8)  # 长时域首步增益 → K^∞


def test_cost_monotone_decreasing():
    """每次迭代的代价单调下降（教程 (4.13) 的接受判据）：至少前 3 迭代严格降。"""
    cost = ReachCost(w_u=0.05, goal=GOAL, w_goal=200.0, w_vf=5.0)
    res = ilqr_solve(np.zeros(4), np.zeros((40, 2)), cost, dt=DT, max_iters=30)
    assert res.n_iters >= 3
    for k in range(3):
        assert res.costs[k + 1] < res.costs[k], (k, res.costs[:4])
    assert res.converged
    # 增益比（实际/预期下降）：初期因 μ 阻尼略低于 1（阻尼步的保守性），
    # 随 μ 驰豫收敛到 1（实测 [0.45, 0.64, …, 1.02]）。
    assert all(r > 0.3 for r in res.ratio_hist)  # 只接受真下降
    assert abs(res.ratio_hist[-1] - 1.0) < 0.15  # 收敛处预期模型可信
    print(f"\n[ilqr] costs {res.costs[0]:.1f} -> {res.costs[-1]:.4f} in {res.n_iters} iters")


def test_waypoint_mission_threads_the_gap():
    """路标点代价把轨迹拉过"缺口"：中途贴近路标、终端到达目标。"""
    n = 60
    cost = ReachCost(
        w_u=0.05, goal=GOAL, w_goal=200.0, w_vf=5.0,
        waypoint=(n // 2, GAP), w_wp=200.0,
    )
    res = ilqr_solve(np.array([0.5, 0.5, 0.0, 0.0]), np.zeros((n, 2)), cost, dt=DT)
    assert res.converged
    assert np.linalg.norm(res.x_traj[-1, :2] - GOAL) < 0.05
    assert np.linalg.norm(res.x_traj[n // 2, :2] - GAP) < 0.15
    print(f"\n[waypoint] mid err {np.linalg.norm(res.x_traj[n // 2, :2] - GAP):.4f} m, "
          f"terminal err {np.linalg.norm(res.x_traj[-1, :2] - GOAL):.4f} m")


def test_rollout_satisfies_dynamics_exactly():
    """标称轨迹精确满足动力学（教程 (4.7) 的"标称精确在轨"前提）。"""
    cost = _regulation_cost()
    res = ilqr_solve(np.array([0.2, 0.1, -0.1, 0.2]), np.zeros((25, 2)), cost, dt=DT)
    for i in range(res.u_traj.shape[0]):
        expect = double_integrator_step(res.x_traj[i], res.u_traj[i], DT)
        assert np.allclose(res.x_traj[i + 1], expect, atol=1e-12)


def test_regularization_adapts():
    """正则化 (4.14) 的两侧行为：良性问题 μ 驰豫到下界；强阻尼步仍下降。"""
    cost = _regulation_cost()
    res = ilqr_solve(np.array([0.4, 0.2, 0.0, 0.0]), np.zeros((20, 2)), cost, dt=DT)
    # 良性问题：μ 只被放松（×0.5 → 下界 1e-9），历史单调不升。
    assert all(res.mu_hist[i + 1] <= res.mu_hist[i] + 1e-15
               for i in range(len(res.mu_hist) - 1))
    assert res.mu_hist[-1] < 1.0  # 从初值 1.0 被放松
    # 强阻尼（μ0 = 1e6）：单步仍严格下降——阻尼逆保证下降方向（(4.14) 依据）。
    res_damped = ilqr_solve(
        np.array([0.4, 0.2, 0.0, 0.0]), np.zeros((20, 2)), cost, dt=DT,
        max_iters=1, mu0=1e6, mu_max=1e12,
    )
    assert res_damped.costs[1] < res_damped.costs[0]


if __name__ == "__main__":
    # 单点测试：python tests/test_X.py <测试名子串>；不带参数 = 全部测试。
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and pat in k]
    if not fns:
        print(f"没有匹配 '{pat}' 的测试；可用测试：")
        for k in sorted(globals()):
            if k.startswith("test_"):
                print(" ", k)
        sys.exit(1)
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)

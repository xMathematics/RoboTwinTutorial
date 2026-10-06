"""projects/control_planning/osc_arm.py 的测试 —— 直接运行：``python tests/test_osc_arm.py``。

验证教程第 06 章的平面 2R 臂教学实现（Khatib 1987 / Hogan 1985 口径）：
正运动学的工作空间界、解析雅可比与中心差分的一致性（教程 06 章对 J 一致性
的要求）、DLS 逆运动学的收敛与奇异位形有界性（第 02 章 (2.7)–(2.10) 的阻尼）、
质量阵正定性与重力补偿闭合、任务空间阻抗控制 τ = Jᵀ(K e + D ė) 的公式一致性
与"从偏移位形拉回锚点"（教程 (6.13)/(6.14) 的闭环行为）。全部确定性。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osc_arm import (  # noqa: E402
    PlanarArm,
    dls_ik,
    impedance_torque,
    numeric_jacobian,
    simulate_impedance,
)

ARM = PlanarArm()
X_D = np.array([1.2, 0.5])
K_IMP = 80.0 * np.eye(2)
D_IMP = 30.0 * np.eye(2)  # 量级按 (6.19) 的 D_c = 2√(MK)（M_eff ~ 2–3 kg）取


def test_fk_workspace_limits():
    """正运动学：伸直位形达到最大可达半径，任意位形的末端不出工作空间圆。"""
    assert np.allclose(ARM.fk(np.array([0.0, 0.0])), [ARM.l1 + ARM.l2, 0.0])
    rng = np.random.default_rng(2)
    max_reach = ARM.l1 + ARM.l2
    for _ in range(50):
        q = rng.uniform(-np.pi, np.pi, 2)
        r = float(np.linalg.norm(ARM.fk(q)))
        assert r <= max_reach + 1e-12
        assert r >= abs(ARM.l1 - ARM.l2) - 1e-12  # 最小可达半径（内环）


def test_jacobian_matches_central_difference():
    """解析雅可比与中心差分一致（1e-6 门限；差分误差 ~1e-10）。"""
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(-np.pi, np.pi, 2)
        J_a = ARM.jacobian(q)
        J_n = numeric_jacobian(ARM, q, eps=1e-6)
        assert np.allclose(J_a, J_n, atol=1e-6), np.abs(J_a - J_n).max()
    # 一阶自洽：f(q + dq) ≈ f(q) + J dq（余量 = 二阶曲率项 O(‖dq‖²) ~ 2e-4）。
    q = np.array([0.4, -0.7])
    dq = np.array([0.01, -0.02])
    pred = ARM.fk(q) + ARM.jacobian(q) @ dq
    real = ARM.fk(q + dq)
    assert np.allclose(pred, real, atol=1e-3)


def test_dls_ik_converges_to_target():
    """DLS 逆运动学收敛到目标位姿（1e-4 门限；实测 2.8e-10，6 次迭代）。"""
    q, err, hist = dls_ik(ARM, np.array([0.3, 0.6]), X_D)
    assert err < 1e-4, err
    assert len(hist) <= 200
    assert np.allclose(ARM.fk(q), X_D, atol=1e-4)
    print(f"\n[ik] err {err:.2e} in {len(hist)} iters")


def test_dls_ik_bounded_near_singular():
    """奇异位形附近 DLS 不发散：从伸直奇异点 (0,0) 出发步长有界且可收敛。

    恰好径向的目标分量在该位形不可观（J 秩 1，列空间只有切向）——
    教程 07.2 ⑤ 同款的"失效边界"教学案例：滤波/迭代停留在原地而非发散。
    """
    # 带切向分量的目标：可逃逸奇异并收敛（实测 7 迭代到 6e-10）。
    q, err, hist = dls_ik(ARM, np.array([0.0, 0.0]), np.array([1.5, 0.1]))
    assert err < 1e-4 and np.all(np.isfinite(q))
    # 纯径向目标：迭代保持有界（无 NaN/inf、步长有限），停在原地。
    q_r, err_r, hist_r = dls_ik(
        ARM, np.array([0.0, 0.0]), np.array([1.7, 0.0]), n_iters=50
    )
    assert np.all(np.isfinite(q_r))
    assert err_r < 0.2  # 不可观方向上不推进（目标在可达集内但该位形无法启动）
    assert np.allclose(q_r, np.array([0.0, 0.0]), atol=1e-6)  # 零步长：驻留在奇异点
    print(f"\n[ik-singular] tangential err {err:.1e} ({len(hist)} iters), "
          f"radial stays at q0 (err {err_r:.3f}, {len(hist_r)} iters)")


def test_mass_matrix_positive_definite_and_gravity_roundtrip():
    """M(q) ≻ 0（教程 (6.2) 前提）；重力补偿闭合：τ = g(q) 时 q̈ = 0。"""
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 2)
        M = ARM.mass_matrix(q)
        assert np.allclose(M, M.T)
        assert np.linalg.eigvalsh(M).min() > 0.0
        qdd = ARM.dynamics(q, np.zeros(2), ARM.gravity(q))
        assert np.allclose(qdd, 0.0, atol=1e-12)  # 静止 + 重力补偿 → 无加速度


def test_impedance_torque_matches_formula():
    """阻抗律 τ = Jᵀ(K e + D ė) + g(q) 的逐项一致性（教程 (6.12) + (6.4) 的映射）。"""
    q = np.array([0.5, -0.3])
    dq = np.array([0.4, 0.2])
    dx_d = np.array([0.1, -0.05])
    tau = impedance_torque(ARM, q, dq, X_D, dx_d, K_IMP, D_IMP)
    x_dot = ARM.jacobian(q) @ dq
    e = X_D - ARM.fk(q)
    expect = ARM.jacobian(q).T @ (K_IMP @ e + D_IMP @ (dx_d - x_dot)) + ARM.gravity(q)
    assert np.allclose(tau, expect, atol=1e-12)
    # 关闭重力补偿时恰好差一项 g(q)。
    tau_free = impedance_torque(ARM, q, dq, X_D, dx_d, K_IMP, D_IMP, gravity_comp=False)
    assert np.allclose(tau - tau_free, ARM.gravity(q), atol=1e-12)


def test_impedance_pulls_back_to_anchor():
    """从偏移位形释放：阻抗闭环把末端拉回锚点（教程 (6.13) 的自由空间口径）。

    实测：0.06/-0.08 rad 的初始偏移在 4 s 内收敛到 4.0e-07 m。
    """
    q_ref, _, _ = dls_ik(ARM, np.array([0.5, 0.8]), X_D)
    q0 = q_ref + np.array([0.06, -0.08])
    q_traj, x_traj = simulate_impedance(
        ARM, q0, X_D, K_IMP, D_IMP, dt=1e-3, n_steps=4000
    )
    assert x_traj.shape == (4001, 2)
    final_err = float(np.linalg.norm(x_traj[-1] - X_D))
    assert final_err < 1e-4, final_err
    # 末段误差小于首段（确实是"拉回"而非漂移）。
    assert final_err < np.linalg.norm(x_traj[0] - X_D) / 100.0
    print(f"\n[impedance] initial err {np.linalg.norm(x_traj[0] - X_D):.4f} m "
          f"-> final {final_err:.2e} m in 4 s")


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

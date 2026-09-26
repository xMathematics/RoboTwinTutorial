"""``projects/slam/vins`` 的测试 —— 直接运行：``python tests/test_vins.py``。

校验教程第 10 章 §10.3 紧耦合视觉-惯性捆绑（论文：VINS-Mono, Qin, Li & Shen,
IEEE T-RO 2018；IMU 因子按 Forster et al., T-RO 2017 Eq.(45)+(48)）：

1. 解析重投影雅可比 vs 有限差分（穿过求解器实际使用的同一回缩；
   符号/约定检查）；
2. 真值状态处 IMU 因子残差应为纯白化传感器噪声（对该离散场景 −½ g Δt²
   测量模型是精确的）；
3. 联合 VI 捆绑从**故意错误尺度**的初始化（平移/速度 × 0.5、偏置清零、
   DLT 三角化路标）恢复：帧间距离比 ∈ [0.9, 1.1]（尺度比 = 估计/真值的
   帧间位移之比，全部 ≈ 1 即绝对公制尺度被 IMU 因子恢复）且 SE(3) 对齐后
   轨迹 ATE < 0.1 m；
4. 同一张因子图把 IMU 因子权重置零后停留在错误尺度 —— IMU 提供公制尺度
   的数值演示（教程 §10.3 第 3 步，单目规范自由度/gauge freedom：纯视觉
   代价对 ``x → s·x`` 严格不变，尺度方向无梯度；实测比值漂移到 ≈ 0.39-0.42）。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lie import se3_exp  # noqa: E402
from vins import (  # noqa: E402
    ImuFactor,
    ReprojectionFactor,
    align_se3,
    build_vins_bundle,
    simulate_vi_scene,
)

SEED = 7


def _position_rmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """位置 RMSE：``√(mean_i ‖p_est,i − p_ref,i‖²)``，单位与输入相同 [m]。"""
    err = np.asarray(estimated) - np.asarray(reference)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _interframe_ratios(est_states, gt_states):
    """所有关键帧对上的比值 ‖p_j − p_i‖_est / ‖p_j − p_i‖_gt。

    尺度比：1 = 估计的帧间位移与真值等长（尺度正确）；整体 ≈ s 表示估计带
    尺度因子 s。SE(3) 对齐（消除 4 自由度规范）不改变帧间距离，故该量在
    对齐前后不变，是尺度可观测性的直接探针。
    """
    p_est = np.array([s.T_wb[:3, 3] for s in est_states])
    p_gt = np.array([s.T_wb[:3, 3] for s in gt_states])
    return [
        float(
            np.linalg.norm(p_est[j] - p_est[i]) / np.linalg.norm(p_gt[j] - p_gt[i])
        )
        for i in range(len(p_gt))
        for j in range(i + 1, len(p_gt))
    ]


def test_reprojection_jacobian_matches_finite_difference():
    """式 5.11 推导适配到机体系位姿回缩：解析 vs 有限差分。"""
    sim = simulate_vi_scene(seed=SEED)
    f = ReprojectionFactor(1, 5, sim.observations[1, 5], sim.sigma_z)
    T = sim.init_states[1].T_wb
    X = sim.init_landmarks[5]
    J = f.jacobian(T, X)
    eps = 1e-6  # 中心差分：舍入噪声 ∝ 1/eps、截断误差 ∝ eps²，取折中
    J_num = np.zeros((2, 9))
    for c in range(6):  # 位姿列：T_wb ← Exp(δξ) T_wb（求解器的更新方式）
        d = np.zeros(6)
        d[c] = eps
        J_num[:, c] = (
            f.residual(se3_exp(d) @ T, X) - f.residual(se3_exp(-d) @ T, X)
        ) / (2.0 * eps)
    for c in range(3):  # 路标列
        d = np.zeros(3)
        d[c] = eps
        J_num[:, 6 + c] = (f.residual(T, X + d) - f.residual(T, X - d)) / (2.0 * eps)
    err = float(np.abs(J - J_num).max())
    assert err < 1e-7, err
    print(f"\n[reproj-jacobian] max |analytic − finite difference| = {err:.2e}")


def test_imu_factor_residual_small_at_truth():
    """真值状态处的残差必须是纯白化传感器噪声。

    场景由与测量模型*相同的离散积分格式*（paper Eq.(31)-(32)）生成，因此
    任何模型失配都会叠加在 N(0, 1) 白化噪声上显形（15 维 → E‖e‖ ≈ 3.7）。
    """
    sim = simulate_vi_scene(seed=SEED)
    bundle = build_vins_bundle(sim)
    norms = []
    for f in bundle._imu_factors:
        e = f.residual(sim.gt_states[f.i], sim.gt_states[f.j])
        assert e.shape == (ImuFactor.dim,)
        norms.append(float(np.linalg.norm(e)))
    # 门限理由：15 维白化残差范数近似服从 χ₁₅ 分布，期望 E‖e‖ ≈ 3.7；
    # 门限 8.0 ≈ 2 倍期望，只为排除系统性模型失配（单次噪声实现的范数
    # 波动 ~ ±1 不会触发）。
    assert max(norms) < 8.0, norms
    print(f"\n[imu-residual@truth] whitened residual norms: "
          f"{np.array2string(np.array(norms), precision=3)} (E‖e‖ ≈ 3.7)")


def test_vi_bundle_recovers_metric_scale():
    """联合 VI 优化：错误尺度的初始化 → 恢复正确尺度 + 低 ATE。"""
    sim = simulate_vi_scene(seed=SEED)
    init_ratios = _interframe_ratios(sim.init_states, sim.gt_states)
    bundle = build_vins_bundle(sim)
    res = bundle.solve(n_iters=150)
    assert res.converged, (res.n_iters, res.cost)

    ratios = _interframe_ratios(res.states, sim.gt_states)
    # 门限理由：尺度比 ∈ (0.9, 1.1) 的含义是估计的帧间位移与真值之比全部
    # 接近 1，即绝对（公制）尺度已由 IMU 因子恢复；10% 容差对应本仿真噪声
    # 水平（σ_z = 4e-3 的归一化坐标观测 + IMU 噪声 + 零偏估计残差）下的
    # 恢复精度。
    assert all(0.9 < r < 1.1 for r in ratios), ratios

    p_est = np.array([s.T_wb[:3, 3] for s in res.states])
    p_gt = np.array([s.T_wb[:3, 3] for s in sim.gt_states])
    R_a, t_a, p_aligned = align_se3(p_est, p_gt)
    ate = _position_rmse(p_aligned, p_gt)
    # 门限理由：单目 VIO 仍有 4 自由度规范（全局 yaw + 平移），align_se3
    # 消除后 ATE 应远小于轨迹长度（~3.4 m）；< 0.1 m 说明位姿/速度/零偏
    # 联合估计收敛到噪声水平。
    assert ate < 0.1, ate

    # 路标与轨迹共享同一规范变换（对齐参数一并施加到路标上）。
    lm_aligned = (R_a @ res.landmarks.T).T + t_a
    lm_rmse = _position_rmse(lm_aligned, sim.gt_landmarks)

    b_est = np.array([[s.b_g, s.b_a] for s in res.states])  # (K, 2, 3)
    print(f"\n[vi-bundle] init scale ratios: "
          f"{np.array2string(np.array(init_ratios), precision=3)}")
    print(f"[vi-bundle] est. scale ratios: {np.array2string(np.array(ratios), precision=4)}")
    print(f"[vi-bundle] ATE RMSE = {ate * 100:.2f} cm  "
          f"(landmarks {lm_rmse * 100:.2f} cm), "
          f"iters {res.n_iters}, cost {res.cost:.3e}")
    print(f"[vi-bundle] mean bias est: bg {np.array2string(b_est[:, 0].mean(0), precision=4)}"
          f" vs true {np.array2string(sim.bias_true[:3], precision=4)} | "
          f"ba {np.array2string(b_est[:, 1].mean(0), precision=4)}"
          f" vs true {np.array2string(sim.bias_true[3:], precision=4)}")


def test_pure_visual_run_loses_scale():
    """IMU 因子权重置零后，尺度规范自由度（scale gauge freedom）保持自由。

    纯视觉代价对 ``x → s·x`` 严格不变（透视除法消去尺度），尺度方向梯度
    恒为零，优化器无法恢复尺度：初始尺度比 ≈ 0.5，实测停留在 ≈ 0.39-0.42
    （并非冻结在 0.5，而是沿无约束规范方向的数值漂移），离 1 仍然很远 ——
    这就是单目尺度不可观测的数值对应物（教程 §10.3 第 3 步）：尺度信息
    只可能来自 IMU 因子。
    """
    sim = simulate_vi_scene(seed=SEED)
    bundle = build_vins_bundle(sim)
    res = bundle.solve(n_iters=150, imu_weight=0.0)
    ratios = _interframe_ratios(res.states, sim.gt_states)
    init_ratio = float(
        np.linalg.norm(
            np.array([s.T_wb[:3, 3] for s in sim.init_states])[1]
            - np.array([s.T_wb[:3, 3] for s in sim.init_states])[0]
        )
        / np.linalg.norm(
            np.array([s.T_wb[:3, 3] for s in sim.gt_states])[1]
            - np.array([s.T_wb[:3, 3] for s in sim.gt_states])[0]
        )
    )
    # 门限理由：只要明显偏离 1（< 0.75）即证明尺度未被恢复 —— 实测比值
    # 漂移到 ≈ 0.39-0.42（初始 ≈ 0.5）：纯视觉代价在尺度方向无梯度，
    # 优化器沿规范自由度数值漂移；任何远离 1 的值都说明尺度只能来自 IMU。
    assert max(ratios) < 0.75, ratios
    print(f"\n[pure-visual] scale ratios: {np.array2string(np.array(ratios), precision=4)}"
          f" (init ≈ {init_ratio:.3f}) — IMU factor weight was 0, scale not recovered")


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

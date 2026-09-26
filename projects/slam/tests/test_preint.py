"""``projects/slam/preint`` 的测试 —— 直接运行：``python tests/test_preint.py``。

校验教程第 10 章 §10.2 预积分教学实现（论文：Forster et al., "On-Manifold
Preintegration", IEEE T-RO 2017, Eq.(27)-(47)、(59)-(69)）：

1. 零噪声一致性：ΔR̃/Δṽ/Δp̃ 与 (a) 在解析时变轨迹上对 Eq.(37) 的直接双重
   循环求值、(b) 常 ω/a 下的闭式（Rodrigues 分解的旋转级数）比对 —— 两者
   均 < 1e-9（≈ 机器精度：递推与定义逐项一致，任何下标/顺序错误都会被
   立刻放大）；
2. 测量模型闭合：经 Eq.(38) 模型（含 −½ g Δt² 重力项）从状态 i 预测状态 j，
   与同一批采样的独立世界系离散积分**精确**一致 —— 期望精确一致的原因是
   真值本身就是用同一离散格式生成的，任何模型/符号不一致都会在此暴露；
3. 协方差正确性：[δφ; δv; δp] 的 Monte-Carlo（200 次定种子）经验协方差与
   线性传播的 Σ 对角线相符（线性化近似 —— 对测量噪声一阶）。30% 相对偏差
   门限的理由：n = 200 时样本 std 自身的相对标准误约 1/√(2n) ≈ 5%，再叠加
   一阶线性化带来的系统性偏差，30% 只检验"同量级一致"而非精确相等；
4. 一阶偏置修正：``correct(δb)`` 与在 ``b + δb`` 处完整重积分之差为
   O(δb²)（δb 减半残差约缩为 1/4 的二次收敛 + 绝对误差 < 1e-3 —— 门限对应
   二阶项量级：|δb^g| ≈ 0.03 rad/s、|δb^a| ≈ 0.10 m/s² 时 C·|δb|² ~ 1e-3），
   且 9×6 偏置雅可比与中心有限差分一致。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lie import hat, so3_exp, so3_left_jacobian, so3_log  # noqa: E402
from preint import ImuParams, Preintegration, so3_right_jacobian  # noqa: E402

G = np.array([0.0, 0.0, -9.81])          # 重力矢量，指向下方（paper Eq.(31)）
RATE = 100.0
DT = 1.0 / RATE
BIAS = np.array([0.01, -0.02, 0.015, -0.05, 0.03, 0.02])


def _varying_truth(n: int) -> tuple[np.ndarray, np.ndarray]:
    """解析（陀螺仪, 加速度计）真值剖面：平滑且随时间非恒定。

    Args:
        n: 采样点数。

    Returns:
        ``(gyro, accel)``：各为 (n, 3)，单位分别为 rad/s 与 m/s²。
    """
    t = np.arange(n) * DT
    gyro = np.stack(
        [0.3 * np.sin(2.0 * t + 0.4), 0.5 * np.cos(1.7 * t), 0.2 + 0.4 * t], axis=1
    )
    accel = np.stack(
        [0.8 * np.sin(1.3 * t), 0.6 + 0.5 * np.cos(2.1 * t), -0.7 * np.sin(0.9 * t)],
        axis=1,
    )
    return gyro, accel


def _make_samples(
    times: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    bias: np.ndarray,
    rng: np.random.Generator | None = None,
    params: ImuParams | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """(t, gyro, accel) 测量 = 真值 + 偏置（+ 白噪声）。

    单样本噪声 std = σ_d/√Δt：这样一段区间上积分出的噪声方差为 σ_d²·Δt，
    与递推中使用的 Σ_η 一致（paper §V，离散/连续噪声密度的换算关系）。

    Args:
        times: (n,) 时间戳 [s]。
        gyro, accel: (n, 3) 真值角速度 [rad/s] 与比力 [m/s²]。
        bias: (6,) 偏置 [b^g; b^a]，单位 [rad/s; m/s²]。
        rng: 给定时注入高斯白噪声；None 表示无噪声。
        params: 传感器模型（取 σ 与 rate）。

    Returns:
        长度 n 的 ``(t, w, a)`` 列表（测量值）。
    """
    params = ImuParams(rate=RATE) if params is None else params
    out = []
    for k, t in enumerate(times):
        w = gyro[k] + bias[:3]
        a = accel[k] + bias[3:]
        if rng is not None:
            w = w + rng.normal(scale=params.sigma_g / np.sqrt(DT), size=3)
            a = a + rng.normal(scale=params.sigma_a / np.sqrt(DT), size=3)
        out.append((float(t), w, a))
    return out


def _eq37_double_loop(
    samples: list[tuple[float, np.ndarray, np.ndarray]], bias: np.ndarray, rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按论文 Eq.(37) 的字面双重循环求值。

    独立于 :class:`Preintegration` 内部的增量更新：ΔR̃_ik 由一条独立的
    旋转连乘链重建（R_acc，不借用预积分类的内部状态），Δṽ_ik 每一项都用
    内层循环从零重新累加 —— 因此增量递推中的任何顺序 bug 都无处藏身。
    """
    n = len(samples)
    dts = np.append(np.diff([s[0] for s in samples]), 1.0 / rate)
    R_acc = [np.eye(3)]  # R_acc[k] = ΔR̃_{i,k} = 前 k 个因子的连乘
    for k in range(n):
        R_acc.append(R_acc[k] @ so3_exp((samples[k][1] - bias[:3]) * dts[k]))
    d_v = np.zeros(3)
    d_p = np.zeros(3)
    for k in range(n):
        a_k = samples[k][2] - bias[3:]
        v_ik = np.zeros(3)  # Δṽ_ik，由内层循环重新累加
        for m in range(k):
            v_ik = v_ik + R_acc[m] @ (samples[m][2] - bias[3:]) * dts[m]
        d_v = d_v + R_acc[k] @ a_k * dts[k]
        d_p = d_p + v_ik * dts[k] + 0.5 * R_acc[k] @ a_k * dts[k] ** 2
    return R_acc[n], d_v, d_p


def test_right_jacobian_identity():
    """J_r(φ) = J_l(−φ) = J_l(φ)^T 必须精确成立（paper Eq.(8)）。"""
    rng = np.random.default_rng(3)
    for _ in range(20):
        phi = rng.normal(size=3) * 2.0
        Jr = so3_right_jacobian(phi)
        assert np.allclose(Jr, so3_left_jacobian(-phi), atol=1e-12)
        assert np.allclose(Jr, so3_left_jacobian(phi).T, atol=1e-12)


def _rotation_series_sum(E: np.ndarray, n: int) -> np.ndarray:
    """旋转矩阵 ``E = Exp(θ û)`` 的幂级数精确和 ``Σ_{k=0}^{n-1} E^k``。

    依据：令 ``P = ûûᵀ``、``W = hat(û)``，Rodrigues 分解给出
    ``E^k = P + cos(kθ)(I − P) + sin(kθ) W``，故求和为
    ``n P + [Σ_k cos kθ](I − P) + [Σ_k sin kθ] W``，其中半角和
    ``Σ cos kθ = sin(nθ/2)cos((n−1)θ/2)/sin(θ/2)``、
    ``Σ sin kθ = sin(nθ/2)sin((n−1)θ/2)/sin(θ/2)``。（教科书几何形式
    ``(I − E^N)(I − E)^{-1}`` 在此不可用：旋转沿其转轴总有特征值 1，
    ``I − E`` 奇异。）

    Args:
        E: (3, 3) 旋转矩阵 ``Exp(θ û)``。
        n: 求和项数（可为 0，返回零矩阵）。

    Returns:
        (3, 3) 幂级数和。
    """
    if n == 0:
        return np.zeros((3, 3))
    phi = so3_log(E)
    theta = float(np.linalg.norm(phi))
    u = phi / theta
    P = np.outer(u, u)
    W = hat(u)
    a = np.sin(n * theta / 2.0) / np.sin(theta / 2.0)
    c = a * np.cos((n - 1) * theta / 2.0)
    s = a * np.sin((n - 1) * theta / 2.0)
    return n * P + c * (np.eye(3) - P) + s * W


def test_zero_noise_matches_double_loop_and_closed_form():
    """ΔR̃/Δṽ/Δp̃ 与双重循环 Eq.(37) 及常 ω/a 闭式比对。"""
    n = 120
    times = np.arange(n) * DT
    gyro, accel = _varying_truth(n)
    samples = _make_samples(times, gyro, accel, BIAS)
    pre = Preintegration(samples, BIAS, ImuParams(rate=RATE))
    ref_r, ref_v, ref_p = _eq37_double_loop(samples, BIAS, RATE)
    assert np.allclose(pre.delta_R, ref_r, atol=1e-9)
    assert np.allclose(pre.delta_v, ref_v, atol=1e-9)
    assert np.allclose(pre.delta_p, ref_p, atol=1e-9)

    # 常 ω/a 闭式：所有因子互易可交换，E = Exp(ω Δt)，故
    # ΔR̃ = E^N、Δṽ = (Σ_k E^k) a Δt、
    # Δp̃ = (Σ_k Σ_{m<k} E^m) a Δt² + ½ (Σ_k E^k) a Δt²  （paper Eq.(37)）。
    n2, w0, a0 = 80, np.array([0.4, -0.25, 0.6]), np.array([0.9, -0.4, 0.3])
    samples2 = _make_samples(
        np.arange(n2) * DT, np.tile(w0, (n2, 1)), np.tile(a0, (n2, 1)), BIAS
    )
    pre2 = Preintegration(samples2, BIAS, ImuParams(rate=RATE))
    E = so3_exp(w0 * DT)
    s_total = _rotation_series_sum(E, n2)              # Σ_{k<N} E^k
    s_nested = sum(_rotation_series_sum(E, k) for k in range(n2))  # Σ_k Σ_{m<k}
    d_r = np.linalg.matrix_power(E, n2)
    d_v = s_total @ a0 * DT
    d_p = s_nested @ a0 * DT**2 + 0.5 * s_total @ a0 * DT**2
    assert np.allclose(pre2.delta_R, d_r, atol=1e-9)
    assert np.allclose(pre2.delta_v, d_v, atol=1e-9)
    assert np.allclose(pre2.delta_p, d_p, atol=1e-9)
    assert abs(pre2.duration - n2 * DT) < 1e-12
    print(f"\n[zero-noise] varying-profile max |Δ| vs double loop: "
          f"R {np.abs(pre.delta_R - ref_r).max():.2e}, "
          f"v {np.abs(pre.delta_v - ref_v).max():.2e}, "
          f"p {np.abs(pre.delta_p - ref_p).max():.2e}")


def test_measurement_model_closes_world_frame_integration():
    """Eq.(38) 预测（−½ g Δt² 模型）== 同一批采样的独立世界系离散积分
    （paper Eq.(31)-(32) 格式）。"""
    rng = np.random.default_rng(11)
    n = 90
    times = np.arange(n) * DT
    gyro, accel = _varying_truth(n)
    samples = _make_samples(times, gyro, accel, BIAS)  # 无噪声、带偏置

    R_i = so3_exp(rng.normal(size=3) * 0.5)
    v_i = rng.normal(size=3) * 0.8
    p_i = rng.normal(size=3) * 0.5
    R, v, p = R_i.copy(), v_i.copy(), p_i.copy()
    for k in range(n):
        a_body = samples[k][2] - BIAS[3:]          # 真实机体系比力
        w_body = samples[k][1] - BIAS[:3]
        acc_w = G + R @ a_body                     # Eq.(31)：v̇ = Ra + g
        p = p + v * DT + 0.5 * acc_w * DT**2       # 世界系格式：用旧 R、v
        v = v + acc_w * DT
        R = R @ so3_exp(w_body * DT)

    pre = Preintegration(samples, BIAS, ImuParams(rate=RATE))
    R_j, v_j, p_j = pre.predict(R_i, v_i, p_i)
    assert np.allclose(R_j, R, atol=1e-10)
    assert np.allclose(v_j, v, atol=1e-10)
    assert np.allclose(p_j, p, atol=1e-10)
    # 直接读出测量模型：Δp̃ = R_i^T (p_j − p_i − v_i Δt − ½ g Δt²)
    # （paper Eq.(38) / 教程 (10.5) 中的 −½ g Δt² 项；符号来自 g 指向下、
    # z 轴向上的约定）。
    d_p_model = R_i.T @ (p - p_i - v_i * pre.duration - 0.5 * G * pre.duration**2)
    assert np.allclose(d_p_model, pre.delta_p, atol=1e-10)
    print(f"\n[model-closure] max errors: R {np.abs(R_j - R).max():.2e}, "
          f"v {np.abs(v_j - v).max():.2e}, p {np.abs(p_j - p).max():.2e}")


def test_covariance_matches_monte_carlo():
    """[δφ; δv; δp] 的经验协方差 vs 线性传播的 Σ。

    该递推是非线性噪声映射的一阶（线性化）近似，因此在 200 次运行约 5% 的
    Monte-Carlo 标准误（样本 std 的相对标准误 ≈ 1/√(2n) = 1/√400）之上还会
    叠加几个百分点的系统偏差；30% 的逐对角门限只检验"同量级一致"，不要求
    精确相等。
    """
    params = ImuParams(sigma_g=2.0e-3, sigma_a=2.0e-2, rate=RATE)  # 放大噪声
    n = 100
    times = np.arange(n) * DT
    gyro, accel = _varying_truth(n)
    zero_bias = np.zeros(6)
    nominal = _make_samples(times, gyro, accel, zero_bias)
    pre0 = Preintegration(nominal, zero_bias, params)
    sig_ref = np.sqrt(np.diag(pre0.cov))

    n_runs = 200
    rng = np.random.default_rng(42)
    deltas = np.empty((n_runs, 9))
    for r in range(n_runs):
        noisy = _make_samples(times, gyro, accel, zero_bias, rng=rng, params=params)
        pre = Preintegration(noisy, zero_bias, params, propagate_uncertainty=False)
        # 噪声约定（paper Eq.(35)-(36)）：ΔR̃_noisy = ΔR̃_0 Exp(δφ^)，δφ 表达
        # 在系 j 中；δv、δp 是系 i 中的量。
        d_phi = so3_log(pre0.delta_R.T @ pre.delta_R)
        deltas[r] = np.concatenate(
            [d_phi, pre.delta_v - pre0.delta_v, pre.delta_p - pre0.delta_p]
        )
    cov_emp = np.cov(deltas, rowvar=False)
    std_emp = np.sqrt(np.diag(cov_emp))
    rel = np.abs(std_emp - sig_ref) / sig_ref
    assert rel.max() < 0.30, (rel, sig_ref, std_emp)
    print(f"\n[covariance] Σ diagonal (propagated): {np.array2string(sig_ref, precision=4)}")
    print(f"[covariance] Monte-Carlo std (200 runs): {np.array2string(std_emp, precision=4)}")
    print(f"[covariance] max relative deviation: {rel.max():.3f}")


def test_bias_correction_first_order():
    """correct(δb) 与在 b + δb 处完整重积分必须一致到 O(δb²)。"""
    n = 100
    times = np.arange(n) * DT
    gyro, accel = _varying_truth(n)
    samples = _make_samples(times, gyro, accel, BIAS)
    pre = Preintegration(samples, BIAS, ImuParams(rate=RATE))

    rng = np.random.default_rng(5)
    dg = rng.normal(size=3)
    dg /= np.linalg.norm(dg)
    da = rng.normal(size=3)
    da /= np.linalg.norm(da)
    delta = np.concatenate([0.03 * dg, 0.10 * da])  # 中等幅度的扰动

    def errors(db: np.ndarray) -> tuple[float, float, float]:
        full = Preintegration(samples, BIAS + db, ImuParams(rate=RATE))
        c_r, c_v, c_p = pre.correct(db)
        e_r = float(np.linalg.norm(so3_log(c_r.T @ full.delta_R)))
        e_v = float(np.linalg.norm(full.delta_v - c_v))
        e_p = float(np.linalg.norm(full.delta_p - c_p))
        return e_r, e_v, e_p

    e_r, e_v, e_p = errors(delta)
    h_r, h_v, h_p = errors(0.5 * delta)
    # 一阶自洽性：线性修正后的残差是 O(δb²) —— δb 减半残差须（近似）缩为
    # 1/4（测试用"减半后 < 原值一半"的宽松门限）—— 且绝对残差保持在 1e-3
    # 以下（门限对应二阶项 C·|δb|² 的量级）。
    assert e_r < 1e-3 and e_v < 1e-3 and e_p < 1e-3, (e_r, e_v, e_p)
    assert h_r < 0.5 * e_r and h_v < 0.5 * e_v and h_p < 0.5 * e_p, (
        (e_r, h_r), (e_v, h_v), (e_p, h_p)
    )
    print(f"\n[bias-correction] |δb|: g {np.linalg.norm(delta[:3]):.3f} rad/s, "
          f"a {np.linalg.norm(delta[3:]):.3f} m/s²")
    print(f"[bias-correction] errors at δb:  R {e_r:.2e} rad, v {e_v:.2e} m/s, "
          f"p {e_p:.2e} m")
    print(f"[bias-correction] errors at δb/2: R {h_r:.2e} rad, v {h_v:.2e} m/s, "
          f"p {h_p:.2e} m  (≈ ¼ of the above = O(δb²))")


def test_bias_jacobians_match_finite_differences():
    """9×6 j_bias（切空间约定）vs 完整重积分的中心差分：
    ΔR̃(b+δb) = ΔR̃ Exp((J δb^g)^)，Δṽ/Δp̃ 对 δb 仿射。"""
    n = 100
    times = np.arange(n) * DT
    gyro, accel = _varying_truth(n)
    samples = _make_samples(times, gyro, accel, BIAS)
    pre = Preintegration(samples, BIAS, ImuParams(rate=RATE))
    eps = 1e-6
    for m in range(6):
        e_m = np.zeros(6)
        e_m[m] = 1.0
        plus = Preintegration(samples, BIAS + eps * e_m, ImuParams(rate=RATE))
        minus = Preintegration(samples, BIAS - eps * e_m, ImuParams(rate=RATE))
        # 旋转部分：Log(ΔR̃^T ΔR̃(b ± ε e_m)) = ± J_R[:, m] ε（Eq.(44) 形式）。
        col = np.concatenate(
            [
                0.5 / eps
                * (
                    so3_log(pre.delta_R.T @ plus.delta_R)
                    - so3_log(pre.delta_R.T @ minus.delta_R)
                ),
                (plus.delta_v - minus.delta_v) / (2.0 * eps),
                (plus.delta_p - minus.delta_p) / (2.0 * eps),
            ]
        )
        err = float(np.abs(col - pre.j_bias[:, m]).max())
        scale = float(np.abs(pre.j_bias[:, m]).max()) + 1e-12
        assert err / scale < 1e-7, (m, err, scale)
    print(f"\n[bias-jacobian] max relative column error vs finite differences: "
          f"{err / scale:.2e}")


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

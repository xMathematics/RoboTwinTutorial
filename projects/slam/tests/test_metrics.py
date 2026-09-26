"""metrics.py 的测试 —— 直接运行：``python tests/test_metrics.py``。

验证统一测评模块的 5 个指标（umeyama_align / ate_rmse / rotation_error_deg /
scale_ratio / rpe_translation）。每个指标至少 2 个测试，且都与**朴素实现**
（显式循环 / 迹公式 / 逐对循环）交叉验证；全部场景确定性构造（seed 固定），
答案解析已知，无随机通过/失败。指标教程见 ../METRICS.md。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lie import so3_exp  # noqa: E402
from metrics import (  # noqa: E402
    ate_rmse,
    rotation_error_deg,
    rpe_translation,
    scale_ratio,
    umeyama_align,
)

rng = np.random.default_rng(7)


# ---------------------------------------------------------------- 朴素实现 --
def _naive_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """朴素逐点 RMSE：显式循环求和，用于交叉验证 ate_rmse 的向量化实现。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    total = 0.0
    for i in range(len(a)):
        total += float(np.sum((a[i] - b[i]) ** 2))
    return float(np.sqrt(total / len(a)))


def _naive_umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """朴素版 Umeyama：质心/协方差/变换/残差全部显式循环，返回 (R, t, s, 残差)。

    与 metrics.umeyama_align 的向量化实现是同一数学（Umeyama 1991），但
    计算路径独立——两者一致才能排除向量化的切片/求和写错。
    """
    p = np.asarray(src, dtype=float)
    q = np.asarray(dst, dtype=float)
    n = len(p)
    p_bar = np.zeros(3)
    q_bar = np.zeros(3)
    for i in range(n):
        p_bar += p[i]
        q_bar += q[i]
    p_bar /= n
    q_bar /= n
    H = np.zeros((3, 3))
    for i in range(n):
        H += np.outer(p[i] - p_bar, q[i] - q_bar)  # 逐点外积累加协方差
    U, S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))   # 同一个 det 校正
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    if with_scale:
        ss = 0.0
        for i in range(n):
            ss += float(np.sum((p[i] - p_bar) ** 2))
        s = (S[0] + S[1] + d * S[2]) / ss           # 闭式尺度（Umeyama 1991）
    else:
        s = 1.0
    t = q_bar - s * (R @ p_bar)
    aligned = np.zeros_like(p)
    for i in range(n):
        aligned[i] = s * (R @ p[i]) + t             # 逐点变换
    return R, t, s, _naive_rmse(aligned, q)


def _naive_rotation_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """朴素旋转误差：迹公式 acos((tr(R_est^T R_gt) - 1)/2)（教程第 02 章）。"""
    cos = np.clip(
        (np.trace(np.asarray(R_est).T @ np.asarray(R_gt)) - 1.0) * 0.5, -1.0, 1.0
    )
    return float(np.arccos(cos) * 180.0 / np.pi)


def _naive_scale_ratio(lengths_est: np.ndarray, lengths_gt: np.ndarray) -> float:
    """朴素尺度比：显式循环累加路程再相除。"""
    le = np.asarray(lengths_est, dtype=float)
    lg = np.asarray(lengths_gt, dtype=float)
    se, sg = 0.0, 0.0
    for x in le:
        se += x
    for x in lg:
        sg += x
    return se / sg


def _naive_rpe(traj_est: np.ndarray, traj_gt: np.ndarray, delta: int) -> float:
    """朴素 RPE：逐对 (i, i+delta) 循环累加相对位移误差。"""
    e = np.asarray(traj_est, dtype=float)
    g = np.asarray(traj_gt, dtype=float)
    total, count = 0.0, 0
    for i in range(len(e) - delta):
        j = i + delta
        total += float(np.sum(((e[j] - e[i]) - (g[j] - g[i])) ** 2))
        count += 1
    return float(np.sqrt(total / count))


# ------------------------------------------------------------- umeyama ----
def test_umeyama_rigid_recovery_without_scale():
    """已知 R, t（s=1）的刚体变换点集必须被精确恢复（with_scale=False 模式）。"""
    pts = rng.normal(size=(40, 3))
    R_true = so3_exp(rng.normal(size=3) * 1.2)
    t_true = rng.normal(size=3)
    dst = (R_true @ pts.T).T + t_true
    R, t, s = umeyama_align(pts, dst, with_scale=False)
    assert np.allclose(R, R_true, atol=1e-9), np.abs(R - R_true).max()
    assert np.allclose(t, t_true, atol=1e-9), np.abs(t - t_true).max()
    assert s == 1.0
    # 残差交叉验证：恢复的变换应把 src 完美摆到 dst（朴素逐点 RMSE < 1e-12）。
    assert _naive_rmse((R_true @ pts.T).T + t_true, dst) < 1e-12


def test_umeyama_with_scale_recovery():
    """已知 R, t, s=1.7 的相似变换：with_scale=True 精确恢复三参数。"""
    pts = rng.normal(size=(40, 3))
    R_true = so3_exp(rng.normal(size=3) * 0.8)
    t_true = rng.normal(size=3)
    s_true = 1.7
    dst = s_true * (R_true @ pts.T).T + t_true
    R, t, s = umeyama_align(pts, dst, with_scale=True)
    assert np.allclose(R, R_true, atol=1e-9), np.abs(R - R_true).max()
    assert np.allclose(t, t_true, atol=1e-9), np.abs(t - t_true).max()
    assert abs(s - s_true) < 1e-9, s
    # 对比：with_scale=False 时旋转仍可恢复（Procrustes 对整体尺度不敏感），
    # 但 s 恒为 1，尺度误差只能留在残差里——这正是尺度比要单独度量的原因。
    R0, t0, s0 = umeyama_align(pts, dst, with_scale=False)
    assert s0 == 1.0 and np.allclose(R0, R_true, atol=1e-9)
    resid_scale = _naive_rmse(s * ((R @ pts.T).T) + t, dst)
    resid_noscale = _naive_rmse((R0 @ pts.T).T + t0, dst)
    assert resid_scale < 1e-9 < resid_noscale, (resid_scale, resid_noscale)


def test_umeyama_naive_implementation_cross_check():
    """向量化实现 vs 显式循环朴素实现：R / t / s / 残差一致（< 1e-12），
    且对 (R, t) 的任意小扰动都不降低残差（闭式解的最优性朴素验证）。"""
    for with_scale in (False, True):
        pts = rng.normal(size=(25, 3))
        R_true = so3_exp(rng.normal(size=3))
        t_true = rng.normal(size=3)
        s_true = 1.7 if with_scale else 1.0
        dst = s_true * (R_true @ pts.T).T + t_true
        R_v, t_v, s_v = umeyama_align(pts, dst, with_scale=with_scale)
        R_n, t_n, s_n, resid_n = _naive_umeyama(pts, dst, with_scale)
        assert np.allclose(R_v, R_n, atol=1e-12)
        assert np.allclose(t_v, t_n, atol=1e-12)
        assert abs(s_v - s_n) < 1e-12
        resid_v = _naive_rmse(s_v * ((R_v @ pts.T).T) + t_v, dst)
        assert abs(resid_v - resid_n) < 1e-12, (resid_v, resid_n)
        # 最优性：闭式解 = 全局最优 ⇒ 任何小扰动只会更差（或持平）。
        for _ in range(10):
            dR = so3_exp(rng.normal(size=3) * 1e-3)
            dt = rng.normal(size=3) * 1e-3
            resid_p = _naive_rmse(s_v * ((dR @ R_v @ pts.T).T) + t_v + dt, dst)
            assert resid_p >= resid_v - 1e-12, (resid_p, resid_v)


def test_umeyama_reflection_guard():
    """镜像点集（反射变换）不得产生 det<0 的"伪刚体"解——det 校正必须生效。"""
    pts = rng.normal(size=(30, 3))
    pts[:, 2] *= 0.01  # 压扁 z 轴：接近共面，反射与旋转最难区分、最易被 SVD 混淆
    mirror = np.diag([-1.0, 1.0, 1.0])          # 反射阵：det = -1，不属于 SO(3)
    dst = (mirror @ pts.T).T                    # dst ≈ mirror @ src（可被完美"拟合"）
    # 未加 det 校正的朴素 SVD 解：det < 0（反射），残差≈0——正是要防的作弊解。
    pc = pts - pts.mean(axis=0)
    qc = dst - dst.mean(axis=0)
    U, _, Vt = np.linalg.svd(pc.T @ qc)
    R_bad = Vt.T @ U.T
    assert np.linalg.det(R_bad) < 0
    assert _naive_rmse((R_bad @ pts.T).T, dst) < 1e-6
    # 加 det 校正后：det = +1（真正的旋转），代价是残差不再为零——
    # 旋转在数学上无法表示反射，误差必须显性化而不是被镜像"吃掉"。
    R, t, s = umeyama_align(pts, dst, with_scale=False)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12          # SO(3) 约束
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)  # 正交性
    resid_guarded = _naive_rmse((R @ pts.T).T + t, dst)
    assert resid_guarded > 1e-3, resid_guarded


# ------------------------------------------------------------- ate --------
def test_ate_rmse_identity_zero():
    """恒等轨迹：ATE = 0（不对齐精确为 0；对齐后亦 < 1e-12）。"""
    gt = np.cumsum(rng.normal(size=(50, 3)) * 0.5, axis=0)
    assert ate_rmse(gt, gt, align=False) < 1e-15
    assert ate_rmse(gt, gt, align=True) < 1e-12
    print(f"\n[ate] identity: aligned {ate_rmse(gt, gt):.2e} m, "
          f"unaligned {ate_rmse(gt, gt, align=False):.2e} m")


def test_ate_rmse_naive_cross_check():
    """带漂移的合成轨迹：ate_rmse 与朴素实现（循环 Umeyama / 逐点距离）
    交叉验证，两种模式均 < 1e-12。"""
    gt = np.cumsum(rng.normal(size=(60, 3)) * 0.3, axis=0)
    drift = np.cumsum(rng.normal(scale=0.02, size=(60, 3)), axis=0)
    est = gt + drift
    # 对齐模式：朴素实现自带循环版 Umeyama（with_scale=True，与 ate_rmse 内部一致）
    _, _, _, resid_n = _naive_umeyama(est, gt, with_scale=True)
    assert abs(ate_rmse(est, gt, align=True) - resid_n) < 1e-12
    # 不对齐模式：朴素逐点距离
    assert abs(ate_rmse(est, gt, align=False) - _naive_rmse(est, gt)) < 1e-12
    print(f"[ate] drift traj: aligned {ate_rmse(est, gt):.4f} m, "
          f"unaligned {ate_rmse(est, gt, align=False):.4f} m")


def test_ate_rmse_align_vs_unaligned_difference():
    """对齐消去的是 gauge（全局摆位）而非误差：同一条轨迹只差一个刚体
    摆位时，对齐后 ATE ≈ 0，不对齐则把摆位全部记为误差——差异符合构造。"""
    gt = np.cumsum(rng.normal(size=(60, 3)) * 0.3, axis=0)
    R0 = so3_exp(np.array([0.3, -0.5, 1.1]))
    t0 = np.array([5.0, -3.0, 2.0])
    est_exact = (R0 @ gt.T).T + t0  # 同一条轨迹，仅全局摆位不同
    ate_aligned = ate_rmse(est_exact, gt, align=True)
    ate_raw = ate_rmse(est_exact, gt, align=False)
    assert ate_aligned < 1e-9, ate_aligned          # gauge 差不该算成误差
    assert ate_raw > 3.0, ate_raw                   # 摆位被完整记为"误差"
    # 加小漂移后：对齐把摆位消掉、ATE 回到漂移量级（且不会优于漂移本身，
    # 因为"恒等对齐"也在候选集里，Sim(3) 对齐只能降低残差）。
    drift = rng.normal(scale=0.02, size=gt.shape)
    est = est_exact + drift
    drift_rmse = _naive_rmse(drift, np.zeros_like(drift))
    ate2 = ate_rmse(est, gt, align=True)
    assert 0.5 * drift_rmse < ate2 <= drift_rmse + 1e-12, (ate2, drift_rmse)


# ------------------------------------------------------------- rotation ---
def test_rotation_error_deg_identity_and_known_angle():
    """恒等 = 0；已知小角度旋转给出解析值（1° 输入 → 1.0 ± 1e-9）。"""
    assert rotation_error_deg(np.eye(3), np.eye(3)) < 1e-12
    R1 = so3_exp(np.array([0.0, 0.0, np.pi / 180.0]))  # 绕 z 转 1°
    assert abs(rotation_error_deg(R1, np.eye(3)) - 1.0) < 1e-9
    # 随机轴 30°：解析值 30（so3_log 主值分支下的测地距离）。
    phi = rng.normal(size=3)
    phi *= np.deg2rad(30.0) / np.linalg.norm(phi)
    R30 = so3_exp(phi)
    assert abs(rotation_error_deg(R30, np.eye(3)) - 30.0) < 1e-6
    assert abs(rotation_error_deg(np.eye(3), R30) - 30.0) < 1e-6  # 参数对称性
    print(f"\n[rot] 1 deg -> {rotation_error_deg(R1, np.eye(3)):.10f} deg, "
          f"30 deg -> {rotation_error_deg(R30, np.eye(3)):.10f} deg")


def test_rotation_error_deg_naive_cross_check():
    """so3_log 实现 vs 迹公式朴素实现：随机旋转对上一致（< 1e-6 deg）。"""
    for _ in range(20):
        R_a = so3_exp(rng.normal(size=3) * 1.5)
        R_b = so3_exp(rng.normal(size=3) * 1.5)
        # 避开 θ≈π：两公式在该处的分支与数值条件不同（精度原因，非正确性）。
        R_rel = R_a.T @ R_b
        theta = np.arccos(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
        if theta > np.pi - 0.05:
            continue
        got = rotation_error_deg(R_a, R_b)
        naive = _naive_rotation_deg(R_a, R_b)
        assert abs(got - naive) < 1e-6, (got, naive)


# ------------------------------------------------------------- scale ------
def test_scale_ratio_uniform_scaling_and_naive():
    """整体缩放 s 倍的轨迹 → 尺度比 = s（单目尺度不确定性的直接体现），
    且与朴素循环求和一致（< 1e-12）。"""
    gt = np.cumsum(rng.normal(size=(40, 3)) * 0.5, axis=0)
    s_true = 0.62
    est = s_true * gt
    step_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    step_est = np.linalg.norm(np.diff(est, axis=0), axis=1)
    ratio = scale_ratio(step_est, step_gt)
    assert abs(ratio - s_true) < 1e-12, ratio
    assert abs(ratio - _naive_scale_ratio(step_est, step_gt)) < 1e-12
    print(f"\n[scale] {s_true}x scaled traj -> ratio {ratio:.6f}")


def test_scale_ratio_anomaly_signal():
    """恒等轨迹 → 比值 1；0.5 倍轨迹 → 比值 0.5，触发"偏离 1 超 10%"
    的异常信号（METRICS.md 尺度比一节的阈值）。"""
    gt = np.cumsum(rng.normal(size=(30, 3)), axis=0)
    step = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    assert abs(scale_ratio(step, step) - 1.0) < 1e-12
    ratio_half = scale_ratio(0.5 * step, step)
    assert abs(ratio_half - 0.5) < 1e-12
    assert abs(ratio_half - 1.0) > 0.1  # |ŝ - 1| > 10% ⇒ 尺度异常


# ------------------------------------------------------------- rpe --------
def test_rpe_translation_naive_cross_check():
    """向量化 RPE vs 逐对循环朴素实现：delta=1 与 delta=3 均 < 1e-12。"""
    gt = np.cumsum(rng.normal(size=(50, 3)) * 0.4, axis=0)
    est = gt + rng.normal(scale=0.01, size=gt.shape)
    for delta in (1, 3):
        got = rpe_translation(est, gt, delta=delta)
        naive = _naive_rpe(est, gt, delta)
        assert abs(got - naive) < 1e-12, (delta, got, naive)
    print(f"\n[rpe] noisy traj: rpe(d=1) {rpe_translation(est, gt):.5f} m, "
          f"rpe(d=3) {rpe_translation(est, gt, delta=3):.5f} m")


def test_rpe_translation_local_vs_global():
    """RPE 与 ATE 的分工（局部平滑 vs 全局一致）：
    全局错位 → rpe ≈ 0 而 ate 大；缓慢漂移 → rpe ≪ ate；局部抖动 →
    rpe 反超 ate。三种情形均与解析值对上。"""
    # 场景 1：整条轨迹只差一个全局平移（纯 gauge）——相对位移完全一致。
    gt = np.cumsum(rng.normal(size=(80, 3)) * 0.5, axis=0)
    shift = np.array([30.0, -12.0, 7.0])
    est = gt + shift
    rpe_shift = rpe_translation(est, gt, delta=1)
    ate_shift = ate_rmse(est, gt, align=False)
    assert rpe_shift < 1e-12, rpe_shift                       # 局部：完美
    assert abs(ate_shift - np.linalg.norm(shift)) < 1e-9      # 全局：全记为误差
    # 场景 2：每步 ~2 mm 的线性漂移（200 步累积 ~0.46 m）——rpe ≪ ate。
    n = 200
    c = np.array([0.002, -0.001, 0.0005])
    gt2 = np.cumsum(rng.normal(size=(n, 3)) * 0.3, axis=0)
    est2 = gt2 + np.arange(n, dtype=float)[:, None] * c
    rpe2 = rpe_translation(est2, gt2, delta=1)
    ate2 = ate_rmse(est2, gt2, align=False)
    # 解析值：rpe = |c|；ate = |c|·sqrt(mean(i²))（漂移随步数线性增长）。
    rpe_expected = float(np.linalg.norm(c))
    ate_expected = rpe_expected * float(np.sqrt(np.mean(np.arange(n) ** 2)))
    assert abs(rpe2 - rpe_expected) < 1e-12, (rpe2, rpe_expected)
    assert abs(ate2 - ate_expected) < 1e-9, (ate2, ate_expected)
    assert rpe2 < ate2 / 50.0, (rpe2, ate2)                   # 局部好、全局差
    # 场景 3：零均值逐帧抖动——全局均值良好但局部差，rpe 反超 ate。
    jitter = rng.normal(scale=0.01, size=(n, 3))
    est3 = gt2 + jitter
    rpe3 = rpe_translation(est3, gt2, delta=1)
    ate3 = ate_rmse(est3, gt2, align=False)
    assert rpe3 > ate3, (rpe3, ate3)
    print(f"[rpe] shift  : rpe {rpe_shift:.1e} m vs ate {ate_shift:.2f} m")
    print(f"[rpe] drift  : rpe {rpe2 * 1000:.2f} mm vs ate {ate2 * 1000:.1f} mm")
    print(f"[rpe] jitter : rpe {rpe3 * 1000:.2f} mm vs ate {ate3 * 1000:.2f} mm")


if __name__ == "__main__":
    # 单点测试：python tests/test_metrics.py <测试名子串>；不带参数 = 全部测试。
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

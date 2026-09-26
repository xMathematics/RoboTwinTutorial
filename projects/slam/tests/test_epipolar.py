"""projects/slam/epipolar 的测试——直跑: ``python tests/test_epipolar.py``.

全合成双视图基准: 随机 3D 点 + 已知 ``T_21`` 生成无噪声匹配像素，流水线
必须还原输入——E/F（八点法）、经手性检验的唯一 ``(R, t)``、重投影误差
~1e-6 px 量级的三角化点、以及经 DLT PnP + Gauss-Newton 精化恢复的相机位姿。
式号 (5.x) 指向教程第 05 章。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.camera import make_intrinsics  # noqa: E402
from core.lie import se3_exp, so3_log, transform_points  # noqa: E402
from epipolar import (  # noqa: E402
    decompose_E,
    eight_point,
    essential_from_rt,
    fundamental_from_E,
    pnp_dlt,
    pnp_refine,
    reprojection_jacobian,
    triangulate,
)

DEG = np.pi / 180.0
CAM = make_intrinsics(500.0, 480.0, 320.0, 240.0)
K = CAM.matrix()
# T_21 取为一次左 SE(3) 扰动: 保证 t_z != 0（pnp_dlt 的尺度符号约定不被
# 退化特例掩盖）、且场景位于两相机前方。
T1 = np.eye(4)
T2 = se3_exp(np.array([0.6, -0.1, 0.25, 0.35, 0.2, -0.12]))
R21, T21 = T2[:3, :3], T2[:3, 3]


def _two_view_scene(n_points: int = 60, seed: int = 7):
    """随机生成相机前方的 3D 点 -> 无噪声像素匹配 (u1, u2)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.5, 1.5, n_points)
    y1 = rng.uniform(-1.5, 1.5, n_points)
    z1 = rng.uniform(3.0, 9.0, n_points)
    X1 = np.column_stack([x1, y1, z1])          # world == cam-1 frame (T1 = I)
    X2 = transform_points(T2, X1)
    assert np.all(X2[:, 2] > 0.0), "scene must be in front of camera 2"
    u1 = CAM.project(X1)
    u2 = CAM.project(X2)
    return X1, u1, u2


def _rel_error(A: np.ndarray, B: np.ndarray) -> float:
    """A 相对 B 的 Frobenius 误差（先吸收最优整体尺度——E/F 尺度不定）."""
    alpha = float(np.sum(A * B) / np.sum(B * B))
    return float(np.linalg.norm(A - alpha * B) / np.linalg.norm(B))


def test_hartley_normalization_properties():
    from epipolar.epipolar import _normalize_points_hartley

    rng = np.random.default_rng(3)
    u = rng.uniform(0.0, 640.0, size=(200, 2))
    n, T = _normalize_points_hartley(u)
    assert np.allclose(n.mean(axis=0), 0.0, atol=1e-12)   # 质心在原点
    rms = float(np.sqrt(np.mean(np.sum(n * n, axis=1))))
    assert abs(rms - np.sqrt(2.0)) < 1e-12                # RMS 距离 = sqrt(2)
    hom = np.column_stack([u, np.ones(len(u))])
    assert np.allclose((T @ hom.T).T[:, :2], n, atol=1e-12)


def test_essential_and_fundamental_satisfied_by_data():
    """eight_point 必须返回 E ~ hat(t)R 与 F ~ K^-T E K^-1（式 5.3/5.4）."""
    _, u1, u2 = _two_view_scene()
    E, F = eight_point(u1, u2, K)
    E_true = essential_from_rt(R21, T21)
    F_true = fundamental_from_E(E_true, K)
    err_e = _rel_error(E, E_true)
    err_f = _rel_error(F, F_true)
    assert err_e < 1e-8, err_e   # 数据无噪声、八点法为线性估计，误差应在机器精度量级
    assert err_f < 1e-8, err_f   # F 由投影后的 E 经 (5.4) 严格变换而来，应同样精确
    # 对极约束在像素上成立: u2^T F u1 ~ 0 (式 5.1 经 5.4)
    hom1 = np.column_stack([u1, np.ones(len(u1))])
    hom2 = np.column_stack([u2, np.ones(len(u2))])
    epi = np.einsum("ki,ij,kj->k", hom2, F, hom1)
    scale = np.linalg.norm(F) * np.linalg.norm(hom1, axis=1) * np.linalg.norm(
        hom2, axis=1
    )
    assert np.max(np.abs(epi) / scale) < 1e-8
    print(f"\n[epipolar] |E - a*E_true|/|E_true| = {err_e:.2e}, "
          f"F: {err_f:.2e}")


def test_decompose_E_recovers_unique_pose():
    """手性检验必须选出唯一候选: R 误差 < 0.5 度，t 方向对齐."""
    _, u1, u2 = _two_view_scene()
    E, _ = eight_point(u1, u2, K)
    R, t = decompose_E(E, u1, u2, K)
    rot_err = float(np.linalg.norm(so3_log(R.T @ R21)))   # rad
    t_angle = float(
        np.arccos(
            np.clip(
                float(t @ T21) / (np.linalg.norm(t) * np.linalg.norm(T21)),
                -1.0,
                1.0,
            )
        )
    )
    assert rot_err < 0.5 * DEG, rot_err   # 无噪声下误差应近机器精度，0.5 度为宽松上界
    assert t_angle < 0.5 * DEG, t_angle   # 单目 t 只有方向可观，比对方向夹角
    print(f"[decompose_E] rotation error = {np.degrees(rot_err):.2e} deg, "
          f"t direction angle = {np.degrees(t_angle):.2e} deg")


def test_triangulate_reprojection_error():
    """DLT 三角化必须把观测像素重投影到 ~1e-6 px 精度."""
    X1, u1, u2 = _two_view_scene()
    X_tri = triangulate(T1, T2, u1, u2, K=K)
    err1 = np.linalg.norm(CAM.project(transform_points(T1, X_tri)) - u1, axis=1)
    err2 = np.linalg.norm(CAM.project(transform_points(T2, X_tri)) - u2, axis=1)
    assert err1.max() < 1e-6, err1.max()
    assert err2.max() < 1e-6, err2.max()
    # 结构也应逐点还原（同一实验里 E 分解 + 三角化的串联自检）
    assert np.linalg.norm(X_tri - X1).max() < 1e-6
    print(f"[triangulate] max reprojection err: cam1 {err1.max():.2e} px, "
          f"cam2 {err2.max():.2e} px")


def test_reprojection_jacobian_matches_finite_differences():
    """式 5.11 的解析雅可比必须与左扰动有限差分一致（符号约定校验）."""
    rng = np.random.default_rng(11)
    T = se3_exp(rng.normal(size=6) * 0.2)
    Xw = rng.normal(size=(8, 3)) * 0.5 + np.array([0.0, 0.0, 5.0])
    Xc = transform_points(T, Xw)
    u = CAM.project(Xc)

    def residual(xi):
        return (u - CAM.project(transform_points(se3_exp(xi) @ T, Xw))).reshape(-1)

    eps = 1e-7
    J_num = np.zeros((16, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        J_num[:, i] = (residual(d) - residual(-d)) / (2.0 * eps)
    J_ana = reprojection_jacobian(Xc, CAM.fx, CAM.fy)
    err = float(np.abs(J_ana - J_num).max())
    assert err < 1e-6, err   # 中心差分误差 O(eps^2)+舍入，eps=1e-7 时约 1e-8~1e-7，1e-6 为宽松上界
    print(f"[jacobian] max |analytic - finite difference| = {err:.2e}")


def test_pnp_dlt_gives_pose_initialization():
    """DLT PnP (式 5.10) 应给出接近真值的线性初值（非线性精化前的量级）."""
    X1, _, u2 = _two_view_scene()
    T0 = pnp_dlt(X1, u2, K)
    rot_err = float(np.linalg.norm(so3_log(T0[:3, :3].T @ R21)))
    trans_err = float(np.linalg.norm(T0[:3, 3] - T21))
    assert rot_err < 1e-3, rot_err   # 线性 DLT 对噪声敏感，但无噪声数据下已接近真值
    assert trans_err < 1e-3, trans_err
    assert np.all(transform_points(T0, X1)[:, 2] > 0.0)   # 深度为正
    print(f"[pnp_dlt] init rotation err = {rot_err:.2e} rad, "
          f"translation err = {trans_err:.2e} m")


def test_pnp_refine_converges_to_truth():
    """流形 GN 精化: DLT 初值与带扰动初值都必须收敛到 < 1e-6 位姿误差."""
    X1, _, u2 = _two_view_scene()
    T0 = pnp_dlt(X1, u2, K)
    T = pnp_refine(X1, u2, K, T0)
    rot_err = float(np.linalg.norm(so3_log(T[:3, :3].T @ R21)))
    trans_err = float(np.linalg.norm(T[:3, 3] - T21))
    assert rot_err < 1e-6, rot_err   # 无噪声 + GN 二次收敛：位姿应恢复到机器精度
    assert trans_err < 1e-6, trans_err

    rng = np.random.default_rng(5)
    T0_bad = se3_exp(rng.normal(size=6) * 0.03) @ T2   # 左扰动真值当"坏初值"
    T0_bad[:3, 3] = T0_bad[:3, 3] + rng.normal(scale=0.05, size=3)
    T2_est = pnp_refine(X1, u2, K, T0_bad)
    rot_err2 = float(np.linalg.norm(so3_log(T2_est[:3, :3].T @ R21)))
    trans_err2 = float(np.linalg.norm(T2_est[:3, 3] - T21))
    assert rot_err2 < 1e-6, rot_err2   # 同上：小的坏初值也应被拉回真值（收敛域足够）
    assert trans_err2 < 1e-6, trans_err2
    print(f"[pnp_refine] from DLT init: rot {rot_err:.2e} rad, "
          f"trans {trans_err:.2e} m; from perturbed init: "
          f"rot {rot_err2:.2e} rad, trans {trans_err2:.2e} m")


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

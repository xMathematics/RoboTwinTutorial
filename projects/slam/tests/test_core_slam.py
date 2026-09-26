"""projects/slam/core 的测试 —— 直接运行：``python tests/test_core_slam.py``。

验证第 02 章数学（hat/vee、Rodrigues、对数映射、雅可比、SE(3)）、第 04 章
投影往返一致性，以及第 08 章 Gauss-Newton 求解器。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.camera import make_intrinsics  # noqa: E402
from core.lie import (  # noqa: E402
    hat,
    se3_exp,
    se3_log,
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
    transform_points,
    vee,
)
from core.solver import gauss_newton  # noqa: E402

rng = np.random.default_rng(7)


def test_hat_vee_roundtrip():
    phi = rng.normal(size=3)
    assert np.allclose(vee(hat(phi)), phi, atol=1e-12)
    # hat(phi) @ v == phi x v（Eq. 2.14 的定义）。
    v = rng.normal(size=3)
    assert np.allclose(hat(phi) @ v, np.cross(phi, v), atol=1e-12)


def test_so3_exp_identity_and_norm_preserving():
    assert np.allclose(so3_exp(np.zeros(3)), np.eye(3), atol=1e-12)
    R = so3_exp(rng.normal(size=3) * 2.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)   # 正交性（Eq. 2.5）
    assert abs(np.linalg.det(R) - 1.0) < 1e-10           # 行列式 = +1（Eq. 2.6）


def test_so3_exp_log_roundtrip():
    for _ in range(50):
        phi = rng.normal(size=3) * 2.5  # theta < pi 几乎必然；过大的样本在下方重试
        if np.linalg.norm(phi) > np.pi - 1e-3:
            continue
        R = so3_exp(phi)
        phi2 = so3_log(R)
        assert np.allclose(phi, phi2, atol=1e-8), (phi, phi2)


def test_so3_exp_small_angle_series():
    # exp(极小 phi) ~ I + hat(phi) + hat(phi)^2 / 2（级数分支）。
    phi = np.array([1e-9, -2e-9, 5e-10])
    K = hat(phi)
    assert np.allclose(so3_exp(phi), np.eye(3) + K + K @ K * 0.5, atol=1e-15)


def test_left_jacobian_bch_consistency():
    """小扰动 dl 下 exp((phi + dl)^) ≈ exp((Jl dl)^) exp(phi^)（Eq. 2.20）。"""
    for _ in range(20):
        phi = rng.normal(size=3) * 1.5
        dl = rng.normal(size=3) * 1e-6
        lhs = so3_exp(phi + dl)
        rhs = so3_exp(so3_left_jacobian(phi) @ dl) @ so3_exp(phi)
        assert np.allclose(lhs, rhs, atol=1e-8)
    # Jl 与其逆相乘应为单位阵（Eq. 2.22-2.24）。
    phi = rng.normal(size=3)
    assert np.allclose(
        so3_left_jacobian(phi) @ so3_left_jacobian_inverse(phi),
        np.eye(3),
        atol=1e-10,
    )


def test_se3_exp_log_roundtrip():
    for _ in range(20):
        xi = rng.normal(size=6)
        T = se3_exp(xi)
        assert np.allclose(T[3], [0, 0, 0, 1], atol=1e-12)
        assert np.allclose(se3_log(T), xi, atol=1e-8)


def test_transform_points_rigid():
    T = se3_exp(rng.normal(size=6))
    pts = rng.normal(size=(10, 3))
    out = transform_points(T, pts)
    d_in = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d_out = np.linalg.norm(out[:, None] - out[None], axis=-1)
    assert np.allclose(d_in, d_out, atol=1e-9)  # 刚性：变换不改变两两点距（教程第 02 章 §2.1）


def test_camera_project_unproject_roundtrip():
    cam = make_intrinsics(500.0, 480.0, 320.0, 240.0)
    pts = rng.normal(size=(20, 3)) * 0.5 + np.array([0.0, 0.0, 4.0])
    pts[:, 2] = np.abs(pts[:, 2]) + 1.0  # 保证正深度
    uv = cam.project(pts)
    depth = np.full(len(pts), pts[:, 2])
    pts2 = cam.unproject(uv, depth)
    assert np.allclose(pts, pts2, atol=1e-9)
    # 矩阵形式：s*[u,v,1]^T = K [X,Y,Z]^T（Eq. 4.4-4.5）。
    K = cam.matrix()
    s = np.stack([uv[:, 0], uv[:, 1], np.ones(len(uv))], axis=1)
    assert np.allclose(np.cross(s * pts[:, 2:3], pts @ K.T), 0, atol=1e-8)


def test_gauss_newton_circle_fit():
    """GN 必须能拟合圆（非线性残差，教程第 08 章 §8.2）。"""
    a_true, b_true, r_true = 1.2, -0.7, 2.5
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pts = np.stack(
        [a_true + r_true * np.cos(theta), b_true + r_true * np.sin(theta)], axis=1
    )
    pts += rng.normal(scale=1e-6, size=pts.shape)

    def residual(x):
        return np.linalg.norm(pts - x[:2], axis=1) - x[2]

    def jacobian(x):
        d = x[:2] - pts   # d|p-c|/dc = (c - p)/n（符号很关键！）
        n = np.linalg.norm(d, axis=1, keepdims=True)
        J = np.concatenate([d / n, -np.ones((len(pts), 1))], axis=1)
        return J

    # 好的初值（质心 + 平均半径）：纯 GN 从它收敛；
    # 退化的零初值则需要 LM —— 见下一个测试。
    c0 = pts.mean(axis=0)
    r0 = np.linalg.norm(pts - c0, axis=1).mean()
    sol = gauss_newton(residual, jacobian, x0=np.array([*c0, r0]), n_iters=50)
    assert np.allclose(sol.x, [a_true, b_true, r_true], atol=1e-5), sol.x


def test_lm_recovers_from_bad_init():
    """加阻尼（LM）后，连退化的 x0=0 也能逃出纯 Gauss-Newton 会陷入的
    局部极小（教程第 08 章 §8.3，Eq. 8.8）。"""
    a_true, b_true, r_true = 1.2, -0.7, 2.5
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pts = np.stack(
        [a_true + r_true * np.cos(theta), b_true + r_true * np.sin(theta)], axis=1
    )
    pts += rng.normal(scale=1e-6, size=pts.shape)

    def residual(x):
        return np.linalg.norm(pts - x[:2], axis=1) - x[2]

    def jacobian(x):
        d = x[:2] - pts
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return np.concatenate([d / n, -np.ones((len(pts), 1))], axis=1)

    sol = gauss_newton(residual, jacobian, x0=np.zeros(3), n_iters=200, lm_lambda=1.0)
    assert np.allclose(sol.x, [a_true, b_true, r_true], atol=1e-4), sol.x


def test_gauss_newton_huber_robust_to_outlier():
    """单个粗大外点不得把直线拟合拖偏（教程第 08 章 §8.5）。"""
    x = np.linspace(0, 1, 30)
    y = 2.0 * x + 1.0 + rng.normal(scale=1e-3, size=30)
    y[5] += 50.0  # 粗大外点

    def residual(p):
        return p[0] * x + p[1] - y

    def jacobian(p):
        return np.stack([x, np.ones_like(x)], axis=1)

    sol = gauss_newton(residual, jacobian, x0=np.zeros(2), n_iters=30,
                       robust_delta=0.1)
    assert np.allclose(sol.x, [2.0, 1.0], atol=0.05), sol.x


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

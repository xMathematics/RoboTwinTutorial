"""projects/slam/direct 的测试——直跑：``python tests/test_direct.py``。

验证第 06 章教学实现（LSD-SLAM 式半稠密直接法）：合成纹理平面场景及其
正向 warp 渲染、手写双线性采样器、解析左扰动投影雅可比 (5.11)/(6.5) 对
有限差分、完整光度残差雅可比链，以及半稠密 Gauss-Newton 对齐的端到端
位姿恢复。位姿精度目标（平移 L2 < 1e-3 m、旋转 < 1e-3 rad）可达，原因：
渲染与跟踪残差在 O(0.1) 灰度的双线性重建底限内自洽（同一套几何与采样
器），优化器因此收敛到真值位姿、误差远低于容差。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lie import se3_exp, se3_log, transform_points  # noqa: E402
from direct import (  # noqa: E402
    bilinear_sample,
    make_textured_plane,
    photometric_jacobian,
    photometric_residuals,
    projection_jacobian,
    select_gradient_pixels,
    semi_dense_align,
    warp_and_sample,
)

# 共享场景：真值小运动 + 小初始位姿扰动（约 0.05 m 平移 + 0.02 rad 旋转，
# 依据：教程 06.2 ③ "光度误差非凸，需较好初值"——初值太远直接法不收敛）。
XI_GT = np.array([0.06, -0.03, 0.04, 0.006, -0.012, 0.009])
XI_PERT = np.array([0.03, -0.02, 0.03, 0.012, -0.008, 0.01])
SCENE_SEED = 7
POSE_TOL = 1e-3


def _scene():
    """渲染共享的两视图平面场景（确定性，seed 固定）。"""
    return make_textured_plane(seed=SCENE_SEED, t_gt=XI_GT[:3], r_gt=XI_GT[3:])


def _reference_points(I1, depth, cam, pixels):
    """所选像素的固定参考系 3D 点（深度固定，单位 m）。"""
    depth_sel = bilinear_sample(depth, pixels)
    return cam.unproject(pixels, depth_sel)


def _project(cam, q):
    """用原始 (5.11) 公式投影 (N, 3) 相机系点（不经过 cam.project）。"""
    q = np.asarray(q, dtype=float).reshape(-1, 3)
    return np.stack(
        [cam.fx * q[:, 0] / q[:, 2] + cam.cx, cam.fy * q[:, 1] / q[:, 2] + cam.cy],
        axis=-1,
    )


def rng_subset(n: int, k: int, seed: int) -> list[int]:
    """``range(n)`` 的确定性有序子集，含 ``k`` 个元素。"""
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())


def test_make_textured_plane_is_deterministic_and_consistent():
    I1a, I2a, T_gt_a, cam, depth_a = _scene()
    I1b, I2b, T_gt_b, _, depth_b = make_textured_plane(seed=SCENE_SEED)
    # 带 seed 生成：重跑逐元素相同（确定性）。
    assert np.array_equal(I1a, I1b) and np.array_equal(I2a, I2b)
    assert np.array_equal(depth_a, depth_b) and np.array_equal(T_gt_a, T_gt_b)
    # 形状与合理深度（z0 = 3 附近的轻微深度变化）。
    assert I1a.shape == I2a.shape == depth_a.shape == (180, 240)
    assert 2.8 < depth_a.min() and depth_a.max() < 3.2
    # 小运动必须真的改变图像（场景非平凡）：若两帧几乎一样，说明 warp 或
    # 渲染有错，后续一切测试都失去意义，故期望差值显著大于噪声。
    assert np.abs(I2a - I1a).max() > 10.0
    # 相机内参：focal 300，主点在图像中心（ch.04）。
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == (300.0, 300.0, 120.0, 90.0)


def test_bilinear_sample_weights_reproduce_exact_values():
    img = np.arange(20, dtype=float).reshape(4, 5)
    # 整数坐标精确返回纹素值（权重退化为 0/1）。
    uv = np.array([[0.0, 0.0], [4.0, 3.0], [2.0, 1.0]])
    assert np.allclose(bilinear_sample(img, uv), img[[0, 3, 1], [0, 4, 2]])
    # 胞元中点 = 2x2 胞元均值——为什么期望均值：手写权重
    # ((1-ax)(1-ay), ax(1-ay), (1-ax)ay, ax·ay) 在 ax = ay = 0.5 时四个
    # 全为 0.25。
    mid = bilinear_sample(img, np.array([[2.5, 1.5]]))[0]
    assert np.isclose(mid, img[1:3, 2:4].mean())
    # 双线性插值对线性图像精确复现（可微性的基础）：ramp(u, v) = 2u + 3v
    # 在胞元内是双线性函数本身，采样必须无误差还原。
    ramp = 2.0 * np.arange(6)[None, :] + 3.0 * np.arange(5)[:, None]
    uv = np.array([[1.3, 2.7], [4.2, 0.6]])
    assert np.allclose(bilinear_sample(ramp, uv), 2.0 * uv[:, 0] + 3.0 * uv[:, 1],
                       atol=1e-12)


def test_projection_jacobian_matches_finite_differences():
    """(6.5) 的几何因子：d π(exp(δξ) q) / dδξ 对解析 2x6 (5.11)。"""
    I1, _, T_gt, cam, depth = _scene()
    pixels = select_gradient_pixels(I1)
    points = _reference_points(I1, depth, cam, pixels)
    q = transform_points(T_gt, points)

    J = projection_jacobian(q, cam)
    eps = 1e-6
    for i in rng_subset(len(q), 40, seed=11):
        J_fd = np.zeros((2, 6))
        for k in range(6):
            d = np.zeros(6)
            d[k] = eps
            J_fd[:, k] = (
                _project(cam, transform_points(se3_exp(d), q[i][None])[0])
                - _project(cam, transform_points(se3_exp(-d), q[i][None])[0])
            ) / (2.0 * eps)
        assert np.allclose(J[i], J_fd, rtol=1e-6, atol=1e-5), (i, J[i], J_fd)
    # 逐元素钉死 (5.11) 的符号约定（平移 1-3 列、旋转 4-6 列；依据：教程
    # (5.11) 下方的符号约定说明）——防止列序/符号写反而数值碰巧接近。
    x, y, z = q[0]
    expected_u = [
        cam.fx / z, 0.0, -cam.fx * x / z**2,
        -cam.fx * x * y / z**2, cam.fx * (1.0 + x * x / z**2), -cam.fx * y / z,
    ]
    expected_v = [
        0.0, cam.fy / z, -cam.fy * y / z**2,
        -cam.fy * (1.0 + y * y / z**2), cam.fy * x * y / z**2, cam.fy * x / z,
    ]
    assert np.allclose(J[0, 0], expected_u, atol=1e-9)
    assert np.allclose(J[0, 1], expected_v, atol=1e-9)


def _warp_valid_points(I1, points, cam, T, guard=3.0):
    """warp 后仍留在图像内 ``guard`` px 的点——此带之外双线性采样在边界
    饱和、雅可比退化（对齐器经 ``guard`` 丢弃这类像素）。"""
    uv, _ = warp_and_sample(I1, points, cam, T)
    h, w = I1.shape
    keep = (
        (uv[:, 0] > guard) & (uv[:, 0] < w - 1 - guard)
        & (uv[:, 1] > guard) & (uv[:, 1] < h - 1 - guard)
    )
    return points[keep]


def test_full_residual_jacobian_chain_on_linear_image():
    """完整链 (6.5) 在线性图像上必须精确：线性图像的 np.gradient 无差分
    误差、双线性采样又精确复现线性函数，所以解析雅可比须与有限差分一致
    到机器精度——这验证的是链结构本身（图像梯度 × 几何雅可比），与纹理
    无关。"""
    I1, _, T_gt, cam, depth = _scene()
    pixels = select_gradient_pixels(I1)
    points = _reference_points(I1, depth, cam, pixels)
    points = _warp_valid_points(I1, points, cam, T_gt)
    ramp = 2.0 * np.arange(I1.shape[1])[None, :] + 1.5 * np.arange(I1.shape[0])[:, None]

    J = photometric_jacobian(ramp, points, cam, T_gt)
    eps = 1e-6
    for i in rng_subset(len(points), 30, seed=5):
        J_fd = np.zeros(6)
        for k in range(6):
            d = np.zeros(6)
            d[k] = eps
            vp = bilinear_sample(ramp, cam.project(transform_points(se3_exp(d) @ T_gt, points[i][None])))
            vm = bilinear_sample(ramp, cam.project(transform_points(se3_exp(-d) @ T_gt, points[i][None])))
            J_fd[k] = (vp[0] - vm[0]) / (2.0 * eps)
        assert np.allclose(J[i], J_fd, rtol=1e-7, atol=1e-6), (i, J[i], J_fd)


def test_full_residual_jacobian_on_textured_frame():
    """在渲染纹理上雅可比只到 np.gradient 离散化精度：中心差分是图像梯度
    的近似（依据：教程 (6.5) 第一步 "图像离散，工程上以有限差分近似"），
    且在分片双线性纹理的胞元边界（kink）处两者相差 O(梯度跳变)。几何
    因子本身是机器精确的（上一个测试），故差异全部来自梯度离散化。"""
    I1, I2, T_gt, cam, depth = _scene()
    pixels = select_gradient_pixels(I1)
    points = _reference_points(I1, depth, cam, pixels)
    uv_t, _ = warp_and_sample(I1, points, cam, T_gt)
    keep = (
        (uv_t[:, 0] > 3.0) & (uv_t[:, 0] < I1.shape[1] - 4)
        & (uv_t[:, 1] > 3.0) & (uv_t[:, 1] < I1.shape[0] - 4)
    )
    points, pixels = points[keep], pixels[keep]
    i1_vals = bilinear_sample(I1, pixels)

    def residual(x):
        _, vals = warp_and_sample(I2, points, cam, se3_exp(x) @ T_gt)
        return vals - i1_vals

    J = photometric_jacobian(I2, points, cam, T_gt)
    eps = 1e-6
    idx = rng_subset(len(points), 300, seed=3)
    J_fd = np.zeros((len(idx), 6))
    for j, i in enumerate(idx):
        for k in range(6):
            d = np.zeros(6)
            d[k] = eps
            J_fd[j, k] = (residual(d)[i] - residual(-d)[i]) / (2.0 * eps)
    rel = np.abs(J[idx] - J_fd) / np.maximum(np.abs(J_fd), 1e-9)
    # 中位数一致到 1e-3 水平；kink 像素（落在胞元边界附近的少数采样点）
    # 构成相对误差的尾部——故只对中位数与"85% 以上行 < 0.25"设阈值。
    assert np.median(rel) < 0.01, np.median(rel)
    assert (rel < 0.25).mean() > 0.85, (rel < 0.25).mean()


def test_semi_dense_align_recovers_pose():
    """端到端：从扰动初值（~0.05 m，~0.02 rad）出发，半稠密 G-N 必须把
    真值位姿恢复到 1e-3 以内（平移 L2 与旋转 so3_log 范数）。I2 乘以精确
    1.0 的增益——纯光度情形（无曝光变化；曝光是 photoba 的事）。"""
    I1, I2, T_gt, cam, depth = _scene()
    I2_gain1 = I2 * 1.0
    T0 = se3_exp(XI_PERT) @ T_gt

    pixels = select_gradient_pixels(I1)
    e0 = photometric_residuals(I1, I2_gain1, depth, cam, T0, pixels)
    T_est = semi_dense_align(I1, I2_gain1, depth, cam, T0, n_iters=20)
    e1 = photometric_residuals(I1, I2_gain1, depth, cam, T_est, pixels)
    assert np.sqrt(np.mean(e1**2)) < 0.5 * np.sqrt(np.mean(e0**2))  # 代价下降：
    # 为什么期望至少减半：G-N 每步都使 (6.4) 残差 RMS 严格下降，20 步足够
    # 把 ~0.05 m 初值误差拉到 O(重建底限)，残差应远小于初值的一半。

    err = se3_log(T_est @ np.linalg.inv(T_gt))
    dt, dr = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
    print(f"\n[direct] pose error: translation {dt:.3e} m, rotation {dr:.3e} rad")
    assert dt < POSE_TOL, dt
    assert dr < POSE_TOL, dr


def test_select_gradient_pixels_is_semi_dense():
    """筛选保留约 top_frac 的内部像素、全部高于分位阈值、且绝不取边界环
    （半稠密像素选取，依据：教程 06.2 第三步）。"""
    I1, _, _, _, _ = _scene()
    top_frac = 0.3
    pixels = select_gradient_pixels(I1, top_frac=top_frac)
    grad_v, grad_u = np.gradient(I1)
    magnitude = np.hypot(grad_u, grad_v)
    interior = magnitude[2:-2, 2:-2]
    # 数量与请求比例吻合（梯度并列时分位数边界略有出入，故留 2% 容差）。
    assert abs(len(pixels) - interior.size * top_frac) < 0.02 * interior.size
    sel_mag = magnitude[pixels[:, 1].astype(int), pixels[:, 0].astype(int)]
    assert sel_mag.min() >= np.quantile(interior, 1.0 - top_frac) - 1e-9
    assert pixels[:, 0].min() >= 2 and pixels[:, 0].max() <= I1.shape[1] - 3
    assert pixels[:, 1].min() >= 2 and pixels[:, 1].max() <= I1.shape[0] - 3
    # 弱纹理像素（低梯度）恰好是被丢弃的那批——更小的 top_frac 只保留更大
    # 的梯度幅值（信息量判据，依据：教程 06.1 第五步）。
    few = select_gradient_pixels(I1, top_frac=0.05)
    few_mag = magnitude[few[:, 1].astype(int), few[:, 0].astype(int)]
    assert few_mag.min() > sel_mag.min()


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

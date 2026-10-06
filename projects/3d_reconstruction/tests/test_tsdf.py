"""projects/3d_reconstruction/tsdf 的测试 —— 直接运行：``python tests/test_tsdf.py``。

验证 TSDF 体素网格与融合（教程第 04 章）：单球多视角积分后 TSDF 零交叉
位置 ≈ 半径（±1 体素）、权重单调性、空视角不改变场、截断 ±1（(4.3)）、
逆方差权重 (4.2)/(4.8) 的解析验证、三线性插值 (4.13)/(1.15) 与
深度->点云反投影的闭环。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene import (  # noqa: E402
    make_sphere_scene,
    make_intrinsics,
    make_orbit_poses,
    render_dataset,
)
from tsdf import TSDFVolume, depth_to_point_cloud  # noqa: E402

BBOX = np.full(3, -1.5), np.full(3, 1.5)
RES = 40                     # 体素数（voxel = 3.0/40 = 0.075 m）
VOXEL = 3.0 / RES


def _integrate_sphere(n_views=10, noise=0.0, seed=0):
    """单球场景 + n 视角深度 -> 已积分的 TSDFVolume（测试共用）。"""
    scene = make_sphere_scene(1.0)
    data = render_dataset(scene, n_views=n_views, img_size=32, focal=30.0,
                          depth_noise_coeff=noise, seed=seed)
    vol = TSDFVolume(*BBOX, resolution=RES)
    for T_wc, depth in zip(data["poses"], data["depths"]):
        vol.integrate(depth, data["K"], T_wc)
    return vol


def _first_zero_crossing(vol, axis_pts):
    """沿一维采样线找符号变号点（线性插值细化）。"""
    sd = vol.signed_distance(axis_pts)
    sign = np.signbit(sd)
    idx = np.nonzero(sign[:-1] != sign[1:])[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    t0, t1 = axis_pts[i, 0], axis_pts[i + 1, 0]
    return t0 - sd[i] * (t1 - t0) / (sd[i + 1] - sd[i])


def test_zero_crossing_near_radius():
    """单球多视角积分后 TSDF 零交叉位置 ≈ 半径（容差 ±1 体素）。"""
    vol = _integrate_sphere(n_views=10)
    pts = np.column_stack([np.linspace(-1.5, 0.0, 300), np.zeros(300), np.zeros(300)])
    x_hit = _first_zero_crossing(vol, pts)
    assert x_hit is not None
    # 零水平集应落在 -1.0 ± 1 个体素内（积分无噪 + 线性插值应远优于该界）。
    assert abs(abs(x_hit) - 1.0) <= VOXEL, (x_hit, VOXEL)
    print(f"\n[tsdf] zero crossing at |x| = {abs(x_hit):.4f} m (r=1.0, voxel={VOXEL:.4f})")


def test_weights_never_decrease():
    """权重单调性：再次融合后每体素权重不减；带内 > 0、远处保持 0。"""
    vol = _integrate_sphere(n_views=2)
    w_before = vol.weight.copy()
    f_before = vol.tsdf.copy()
    scene = make_sphere_scene(1.0)
    data = render_dataset(scene, n_views=2, img_size=32, focal=30.0, seed=3)
    updated = vol.integrate(data["depths"][0], data["K"], data["poses"][0])
    assert updated > 0
    # (4.8)：W <- W + w，逐体素单调不减。
    assert np.all(vol.weight >= w_before - 1e-15)
    assert np.allclose(vol.weight - w_before, vol.weight - w_before)  # 平凡自检
    # 结构：表面带内（|sdf| 小）权重为正，远离场景的角落保持 0（带外不写入）。
    n = vol.resolution
    axis = (np.arange(n) + 0.5) * vol.voxel_size + vol.origin[0]
    corner = vol.weight[0, 0, 0]  # 包围盒角落远离球面
    assert corner == 0.0
    center_slice = vol.weight[n // 2, n // 2, :]  # 过球心的 z 列：带内应有权重
    assert center_slice.max() > 0.0


def test_empty_depth_leaves_field_unchanged():
    """空视角（深度全 inf）不改变场：tsdf/weight 数组逐元素相等。"""
    vol = _integrate_sphere(n_views=2)
    f0, w0 = vol.tsdf.copy(), vol.weight.copy()
    empty = np.full((32, 32), np.inf)
    # 用一个合法位姿（相机在轨道上）但深度全无效。
    T_wc = make_orbit_poses(1, radius=1.6, elevation=0.35)[0]
    K = make_intrinsics(30.0, 30.0, 15.5, 15.5)
    updated = vol.integrate(empty, K, T_wc)
    assert updated == 0
    assert np.array_equal(vol.tsdf, f0)
    assert np.array_equal(vol.weight, w0)


def test_tsdf_bounded_by_one():
    """截断 ±1（(4.3)）：有观测（W>0）的体素 |F| <= 1，带外饱和 |F| = 1。"""
    vol = _integrate_sphere(n_views=8)
    seen = vol.weight > 0.0
    assert seen.any()
    assert np.abs(vol.tsdf[seen]).max() <= 1.0 + 1e-12
    # 自由空间带外体素保持初始 +1（饱和），零水平集只由带内线性区决定。
    assert vol.tsdf[0, 0, 0] == 1.0


def test_inverse_variance_weighting():
    """逆方差加权 (4.2)/(4.8)：两帧深度不同的常数深度图 -> 融合值与
    w = 1/z^2 的解析递归一致，且与"等权平均"显著不同（权重确实来自 1/z^2）。"""
    res = 16
    mu = 0.08
    vol = TSDFVolume(np.full(3, -0.8), np.full(3, 0.8), resolution=res, mu=mu)
    K = make_intrinsics(50.0, 50.0, 7.5, 7.5)     # 16x16 图，中心 (7.5, 7.5)
    T_wc = np.eye(4)                              # 相机在 z = -1.0 看向世界 +z
    T_wc[:3, 3] = np.array([0.0, 0.0, -1.0])
    D1, D2 = 1.00, 1.05                           # 两帧常数深度（权重 1/D^2）
    vol.integrate(np.full((16, 16), D1), K, T_wc, mu=mu)
    w1 = vol.weight.copy()
    f1 = vol.tsdf.copy()
    vol.integrate(np.full((16, 16), D2), K, T_wc, mu=mu)
    band = (w1 > 0) & (vol.weight > w1 + 1e-12)   # 两帧都写入的体素
    assert band.any()
    # 解析：体素世界 z = z_v -> 相机系深度 z_v + 1；帧 i 的观测
    # s_i = D_i - (z_v + 1)，f_i = clip(s_i/mu)，w_i = 1/D_i^2（(4.2)）。
    n = res
    z_v = (np.arange(n) + 0.5) * vol.voxel_size + vol.origin[2]      # (n,)
    z_grid = np.broadcast_to(z_v[:, None, None], (n, n, n))
    f1_obs = np.clip((D1 - (z_grid + 1.0)) / mu, -1.0, 1.0)
    f2_obs = np.clip((D2 - (z_grid + 1.0)) / mu, -1.0, 1.0)
    wgt1, wgt2 = 1.0 / D1 ** 2, 1.0 / D2 ** 2
    f_expect = (wgt1 * f1_obs + wgt2 * f2_obs) / (wgt1 + wgt2)        # (4.8) 展开
    assert np.allclose(vol.tsdf[band], f_expect[band], atol=1e-12)
    assert np.allclose(w1[band], wgt1, atol=1e-12)                    # 首帧 W = w
    # 对照：等权平均会给出不同结果 -> 权重确实随观测深度 z 变化。
    f_equal = 0.5 * (f1_obs + f2_obs)
    assert not np.allclose(vol.tsdf[band], f_equal[band], atol=1e-6)


def test_trilinear_interpolation_corners_midpoint():
    """三线性插值 (4.13)/(1.15)：体素中心 = 存储值、棱中点 = 两端均值、盒外 = +1。"""
    vol = TSDFVolume(np.full(3, -0.8), np.full(3, 0.8), resolution=16)
    # 手工填充一个线性场 F = z（体素中心处）。
    n = vol.resolution
    zz = (np.arange(n) + 0.5) * vol.voxel_size + vol.origin[2]
    vol.tsdf[:] = zz[:, None, None] * np.ones(n)[None, None, :]
    vol.weight[:] = 1.0
    voxel = vol.voxel_size
    centers = vol.voxel_centers()
    assert np.allclose(vol.sample_tsdf(centers), vol.tsdf.ravel(), atol=1e-12)
    # x 向棱中点：两端体素均值。
    a = centers[n * n * 8 + n * 8 + 8]             # 体素 (8, 8, 8)
    mid = a + np.array([0.5 * voxel, 0.0, 0.0])    # 与 (9, 8, 8) 的中点
    v0 = vol.tsdf[8, 8, 8]
    v1 = vol.tsdf[8, 8, 9]
    assert np.isclose(vol.sample_tsdf(mid[None])[0], 0.5 * (v0 + v1), atol=1e-12)
    # 包围盒外 -> +1（外部饱和语义）。
    assert vol.sample_tsdf(np.array([[5.0, 5.0, 5.0]]))[0] == 1.0


def test_signed_distance_query_sign_and_clamp():
    """截断符号距离查询：带内线性恢复 s = F·mu（(4.3) 反缩放）、带外饱和 ±mu、符号正确。

    用手工构造的平面带场（F = clip(x/mu)，|x| <= mu 处 W > 0）做解析断言，
    避免多视角渲染在小图像下的稀疏覆盖干扰。
    """
    res, mu = 32, 0.1
    vol = TSDFVolume(np.full(3, -0.8), np.full(3, 0.8), resolution=res, mu=mu)
    centers = vol.voxel_centers()
    x = centers[:, 0]
    vol.tsdf[:] = np.clip(x / mu, -1.0, 1.0).reshape(res, res, res)  # x 为最后一轴
    vol.weight[:] = (np.abs(x) <= mu).astype(float).reshape(res, res, res)
    q = vol.signed_distance(np.array([
        [0.05, 0.0, 0.0],     # 带内右侧：s = +0.05
        [-0.05, 0.0, 0.0],    # 带内左侧：s = -0.05（表面 x=0 的内侧）
        [0.20, 0.0, 0.0],     # 带外右侧：饱和 +mu
        [-0.20, 0.0, 0.0],    # 带外左侧：饱和 -mu
        [0.0, 0.0, 0.0],      # 表面上：0
    ]))
    assert np.allclose(q, [0.05, -0.05, mu, -mu, 0.0], atol=1e-12), q


def test_depth_to_point_cloud_roundtrip():
    """深度->点云：反投影点经 K、T_cw 再投影，深度/像素与原图一致。"""
    scene = make_sphere_scene(1.0)
    data = render_dataset(scene, n_views=1, img_size=32, focal=30.0,
                          depth_noise_coeff=0.0, seed=0)
    depth, K, T_wc = data["depths"][0], data["K"], data["poses"][0]
    pts = depth_to_point_cloud(depth, K, T_wc)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert len(pts) == int(np.isfinite(depth).sum())
    # 投回相机系：z 应精确等于原深度，像素回到 (u, v)。
    T_cw = np.linalg.inv(T_wc)
    cam = pts @ T_cw[:3, :3].T + T_cw[:3, 3]
    assert np.allclose(cam[:, 2], depth[np.isfinite(depth)], atol=1e-9)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu = np.rint(fx * cam[:, 0] / cam[:, 2] + cx).astype(int)
    vv = np.rint(fy * cam[:, 1] / cam[:, 2] + cy).astype(int)
    vs, us = np.nonzero(np.isfinite(depth))
    order = np.lexsort((us, vs))                   # 行主序对齐
    assert np.array_equal(vv[order], vs) and np.array_equal(uu[order], us)


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

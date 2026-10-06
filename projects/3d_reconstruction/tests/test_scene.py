"""projects/3d_reconstruction/scene 的测试 —— 直接运行：``python tests/test_scene.py``。

验证合成场景模块（教程第 01 章 SDF 约定 + 第 06 章 Eikonal + 4.3 节光线投射）：
三个 SDF 基元的解析距离与最近点、min 并集规则、球追踪渲染深度与已知 SDF
零点的一致性（采样光线验证）、轨道位姿的 look-at 闭环、数据集的形状与
确定性，以及解析 SDF 满足 Eikonal 方程 (6.1) 的数值验证。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene import (  # noqa: E402
    Box,
    Plane,
    Sphere,
    make_demo_scene,
    make_intrinsics,
    make_orbit_poses,
    make_sphere_scene,
    render_dataset,
    render_depth,
    sample_scene_surface,
)


def test_sphere_sdf_values_and_sign():
    """球 SDF：表面 0、径向线性（外正内负，教程 (1.13)）。"""
    sph = Sphere(np.array([0.2, -0.1, 0.3]), 0.5)
    p = np.array([
        [0.2, -0.1, 1.0],     # 顶点正上方 0.7 处：sdf = 0.2
        [0.2, -0.1, 0.3],     # 球心：sdf = -0.5
        [0.2, -0.1, 0.3 + 0.5],  # 球面：sdf = 0
        [0.2 + 0.5 / np.sqrt(2), -0.1 + 0.5 / np.sqrt(2), 0.3],  # 45 度斜向球面
    ])
    got = sph.sdf(p)
    assert np.allclose(got, [0.2, -0.5, 0.0, 0.0], atol=1e-12), got
    # 径向线性：从球心沿任意方向走 t，sdf 应为 t - r（Eikonal (6.1) 的解析形态）。
    rng = np.random.default_rng(3)
    d = rng.normal(size=(20, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ts = np.linspace(0.0, 1.2, 7)
    pts = sph.center + ts[None, :, None] * d[:, None, :]
    expect = ts[None, :] - 0.5
    assert np.allclose(sph.sdf(pts.reshape(-1, 3)).reshape(20, 7), expect, atol=1e-12)


def test_box_sdf_exact_distance():
    """盒子 SDF：盒外为精确欧氏距离、盒内为负深度、棱上为 0。"""
    box = Box(np.zeros(3), np.array([1.0, 0.5, 0.25]))
    p = np.array([
        [2.0, 0.0, 0.0],    # +x 面外 1.0
        [1.5, 0.5, 0.75],   # 角外：对角距离 sqrt(0.25 + 0 + 0.25)
        [0.3, 0.1, 0.0],    # 盒内：最近面 z=0.25 -> -0.25
        [1.0, 0.2, 0.1],    # +x 面上：0
    ])
    got = box.sdf(p)
    assert np.allclose(got, [1.0, np.sqrt(0.5), -0.25, 0.0], atol=1e-12), got


def test_plane_sdf_sign_and_norm():
    """平面 SDF：n̂·(p - x0)，两侧符号相反、距面距离精确。"""
    plane = Plane(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]))
    p = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -1.0], [5.0, 2.0, -1.0]])
    got = plane.sdf(p)
    assert np.allclose(got, [np.sqrt(2.0), -np.sqrt(2.0), 0.0], atol=1e-12), got
    # 最近点 = 垂直投影：到最近点的距离应与 |sdf| 一致。
    cp = plane.closest_point(p)
    dist = np.linalg.norm(p - cp, axis=1)
    assert np.allclose(dist, np.abs(got), atol=1e-12)
    assert np.allclose(plane.sdf(cp), 0.0, atol=1e-12)  # 最近点在平面上


def test_union_scene_min_rule():
    """场景 SDF = 各基元 min（并集外正内负）；最近点归属 |sdf| 更小的基元。"""
    scene = make_demo_scene()
    p = np.array([
        [0.0, 0.0, 0.4],      # 球外缘：球 sdf 0.05 < 盒 sdf -> 球主导
        [0.6, 0.0, 0.2],      # 盒顶面：盒 sdf 0.05 < 球 sdf -> 盒主导
        [0.0, 0.0, 0.0],      # 球心：负值
    ])
    got = scene.sdf(p)
    expect = np.minimum(
        Sphere(np.zeros(3), 0.35).sdf(p),
        Box(np.array([0.6, 0.0, 0.0]), np.full(3, 0.15)).sdf(p),
    )
    assert np.allclose(got, expect, atol=1e-12)


def test_closest_point_on_primitives():
    """最近点：球沿径向投影到球面、盒外截断/盒内推面、平面垂投。"""
    sph = Sphere(np.zeros(3), 1.0)
    p = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    cp = sph.closest_point(p)
    assert np.allclose(cp, [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], atol=1e-12)
    assert np.allclose(np.linalg.norm(cp, axis=1), 1.0, atol=1e-12)

    box = Box(np.zeros(3), np.array([1.0, 1.0, 1.0]))
    p_in = np.array([[0.9, 0.5, -0.5]])           # 盒内：沿 x 推到 +x 面
    assert np.allclose(box.closest_point(p_in), [[1.0, 0.5, -0.5]], atol=1e-12)
    p_out = np.array([[2.0, 3.0, 0.5]])           # 盒外角向：截断到角 (1,1,0.5)
    assert np.allclose(box.closest_point(p_out), [[1.0, 1.0, 0.5]], atol=1e-12)


def test_eikonal_property_of_analytic_sdf():
    """解析 SDF 的梯度模长 ≈ 1（Eikonal 方程，教程 (6.1)；中轴附近除外）。"""
    scene = make_demo_scene()
    rng = np.random.default_rng(11)
    lo, hi = scene.bbox_lo, scene.bbox_hi
    pts = lo + rng.random((400, 3)) * (hi - lo)
    # 丢掉 |sdf| 很小的点（差分跨零点）与 |sdf| 接近 0.05/0.10 的缝（两基元
    # 距离并列处梯度不唯一），保留距离语义良好的区域。
    s = scene.sdf(pts)
    keep = (np.abs(s) > 0.02) & (np.abs(np.abs(s) - 0.10) > 0.01)
    pts = pts[keep]
    h = 1e-4
    grad = np.stack([
        (scene.sdf(pts + h * np.eye(3)[i]) - scene.sdf(pts - h * np.eye(3)[i])) / (2 * h)
        for i in range(3)
    ], axis=1)
    norm = np.linalg.norm(grad, axis=1)
    assert np.allclose(norm, 1.0, atol=5e-4), (norm.min(), norm.max())


def test_depth_center_pixel_analytic():
    """深度图与已知 SDF 零点一致（中心像素）：光轴过球心时深度 = d - r。"""
    scene = make_sphere_scene(0.35)
    for el in (0.0, 0.35):
        poses = make_orbit_poses(1, radius=1.6, elevation=el)
        T_wc = poses[0]
        K = make_intrinsics(50.0, 50.0, 27.5, 27.5)
        depth = render_depth(scene, K, T_wc, 56, 56)
        expected = np.linalg.norm(T_wc[:3, 3]) - 0.35
        assert abs(depth[27, 27] - expected) < 2e-3, (depth[27, 27], expected)


def test_depth_zero_level_consistency():
    """深度图与已知 SDF 零点一致（采样光线验证）：随机像素反投影点的 |sdf| 极小。"""
    scene = make_demo_scene()
    K = make_intrinsics(50.0, 50.0, 27.5, 27.5)
    T_wc = make_orbit_poses(1, radius=1.6, elevation=0.25)[0]
    depth = render_depth(scene, K, T_wc, 56, 56)
    rng = np.random.default_rng(5)
    vs, us = np.nonzero(np.isfinite(depth))
    pick = rng.choice(len(vs), size=120, replace=False)
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
    for v, u in zip(vs[pick], us[pick]):
        # 反投影：深度 = 相机系 z，光线未归一化方向 z 分量为 1，
        # 故命中点 p = o + z·[(u-cx)/fx, (v-cy)/fy, 1]（针孔投影的逆）。
        d_unnorm = np.array([(u - 27.5) / 50.0, (v - 27.5) / 50.0, 1.0])
        hit = depth[v, u] * d_unnorm @ R_wc.T + t_wc
        assert abs(scene.sdf(hit[None])[0]) < 1e-3, (v, u, scene.sdf(hit[None])[0])
    # 无回波像素（+inf）确实存在且指向天空/包围盒外。
    assert np.isinf(depth).any()
    assert depth[np.isfinite(depth)].min() > 0


def test_orbit_poses_look_at_origin():
    """轨道位姿：光轴 (Z 列) 精确指向原点、旋转正交且 det=+1、T 闭环。"""
    poses = make_orbit_poses(8, radius=1.3, elevation=0.5)
    for T_wc in poses:
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        forward = R[:, 2]                              # 相机 +z 在世界系的方向
        expect = -t / np.linalg.norm(t)                # 指向原点的单位矢量
        assert np.allclose(forward, expect, atol=1e-12)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        # 闭环：把原点经 T_cw 变换应落在相机系 +z 轴上（x=y=0, z=|t|）。
        T_cw = np.linalg.inv(T_wc)
        o_cam = (T_cw @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        assert np.allclose(o_cam, [0.0, 0.0, np.linalg.norm(t)], atol=1e-12)


def test_render_dataset_shapes_deterministic():
    """数据集：形状/视角数正确；同 seed 两次渲染逐像素一致（确定性）。"""
    scene = make_demo_scene()
    d1 = render_dataset(scene, n_views=6, img_size=32, focal=40.0, seed=0)
    d2 = render_dataset(scene, n_views=6, img_size=32, focal=40.0, seed=0)
    assert len(d1["depths"]) == len(d1["poses"]) == 6
    assert d1["depths"][0].shape == (32, 32)
    assert d1["K"].shape == (3, 3) and d1["K"][0, 0] == 40.0
    for a, b in zip(d1["depths"], d2["depths"]):
        assert np.array_equal(a, b)                    # 逐 bit 一致
    d3 = render_dataset(scene, n_views=6, img_size=32, focal=40.0, seed=1)
    assert not np.array_equal(d1["depths"][0], d3["depths"][0])  # 换 seed 噪声变化


def test_surface_samples_lie_on_surface():
    """真值表面采样：所有样本 |sdf| ≈ 0（解析投影后精确落面）。"""
    scene = make_demo_scene()
    pts = sample_scene_surface(scene, 500, band=0.05, seed=2)
    assert pts.shape == (500, 3)
    s = scene.sdf(pts)
    assert np.abs(s).max() < 1e-9, np.abs(s).max()
    # 样本分布在两块物体上（球面 + 盒面），且都在场景包围盒内。
    on_sphere = np.abs(np.linalg.norm(pts, axis=1) - 0.35) < 1e-6
    assert 0.2 < on_sphere.mean() < 0.9
    assert np.all(pts >= scene.bbox_lo - 1e-9)
    assert np.all(pts <= scene.bbox_hi + 1e-9)


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

"""projects/3d_reconstruction/plane_sweep 的测试 —— 直接运行：``python tests/test_plane_sweep.py``。

验证平面扫描立体的教学代理（教程 03 章 (3.1)-(3.5)、(3.12)）：单应矩阵与
刚体投影的解析一致性、世界空间纹理的跨视角一致性（render_rgb）、无噪纹理下
深度恢复的相对误差（< 5%）、置信度在遮挡边界/弱纹理区的显著衰减（与已知
球体几何对照）、以及多视角融合点云对单视角的 Chamfer 优势。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util as _ilu

# 本项目 metrics.py 按路径加载（唯一模块键）——仓库内 4 个项目各有 metrics.py
# （CONSTRAINTS §4.7），合并收集（仓库根 pytest / VS Code Testing）时
# sys.modules["metrics"] 会被先导入者占据，故不能裸 `import metrics`。
_mpath = Path(__file__).resolve().parents[1] / "metrics.py"
_mspec = _ilu.spec_from_file_location(f"_metrics_{_mpath.parent.name}", _mpath)
metrics = _ilu.module_from_spec(_mspec)
_mspec.loader.exec_module(metrics)

chamfer_distance = metrics.chamfer_distance
sample_mesh_surface = metrics.sample_mesh_surface

from plane_sweep import (  # noqa: E402
    plane_homography,
    relative_pose,
    run_plane_sweep,
    warp_to_ref,
)
from scene import (  # noqa: E402
    Scene,
    Sphere,
    checker_texture,
    make_intrinsics,
    make_orbit_poses,
    render_depth,
    render_rgb,
    sample_scene_surface,
    value_noise_texture,
)
from tsdf import TSDFVolume, depth_to_point_cloud  # noqa: E402
from marching import marching_tetrahedra  # noqa: E402

# 测试相机/场景口径（解析量均由此推出）。
IMG_SIZE, FOCAL, ORBIT_R, SPHERE_R = 64, 60.0, 1.4, 0.35


def _sphere_scene() -> Scene:
    """单球测试场景（解析轮廓/深度，遮挡边界 = 投影轮廓线）。"""
    return Scene(primitives=(Sphere(np.zeros(3), SPHERE_R),),
                 bbox_lo=-np.full(3, SPHERE_R + 1.0),
                 bbox_hi=np.full(3, SPHERE_R + 1.0))


def _setup(texture_fn, n_views: int = 24):
    """环绕双相机口径：内参 + 24 视角位姿 + RGB 渲染（纹理可替换）。"""
    poses = make_orbit_poses(n_views, radius=ORBIT_R, elevation=0.25)
    k = make_intrinsics(FOCAL, FOCAL, (IMG_SIZE - 1) / 2.0, (IMG_SIZE - 1) / 2.0)
    images = [render_rgb(_sphere_scene(), k, t, IMG_SIZE, IMG_SIZE, texture_fn)
              for t in poses]
    return k, poses, images


# ------------------------------------------------------- 单应与相对位姿 ----
def test_homography_matches_rigid_projection_on_plane():
    """(3.1)-(3.2)：相对位姿的刚体一致性 + 平面上单应映射 == 直接投影（1e-9）。"""
    k = make_intrinsics(60.0, 60.0, 31.5, 31.5)
    poses = make_orbit_poses(12, radius=3.0, elevation=0.25)
    t_ref, t_src = poses[0], poses[1]
    r_rel, t_rel = relative_pose(t_ref, t_src)

    # 刚体一致性：R_rel X + t_rel 与"经世界系中转"逐点一致（T_wc 相乘消元）。
    rng = np.random.default_rng(0)
    points_cam = rng.normal(size=(30, 3))
    via_world = ((points_cam @ t_ref[:3, :3].T + t_ref[:3, 3] - t_src[:3, 3])
                 @ t_src[:3, :3])
    assert np.abs(points_cam @ r_rel.T + t_rel - via_world).max() < 1e-12

    # 平面 z = d 上的点：H(d) 映射 == 经刚体变换后的针孔投影（教程 3.1 第五步）。
    depth = 2.0
    hom = plane_homography(k, k, r_rel, t_rel, depth)
    on_plane = np.column_stack([rng.uniform(-0.5, 0.5, 30),
                                rng.uniform(-0.5, 0.5, 30), np.full(30, depth)])
    uv_ref = (k @ on_plane.T).T
    uv_ref = uv_ref[:, :2] / uv_ref[:, 2:]
    world = on_plane @ t_ref[:3, :3].T + t_ref[:3, 3]
    cam_src = (world - t_src[:3, 3]) @ t_src[:3, :3]
    uv_src = (k @ cam_src.T).T
    uv_src = uv_src[:, :2] / uv_src[:, 2:]
    homo = np.column_stack([uv_ref, np.ones(30)]) @ hom.T
    uv_h = homo[:, :2] / homo[:, 2:3]
    assert np.abs(uv_h - uv_src).max() < 1e-9


# ------------------------------------------------- 纹理与跨视角一致性 ------
def test_texture_rendering_world_space_consistency():
    """render_rgb：同一表面点跨视角同色（平面扫描的前提）、无回波像素为黑。"""
    scene = _sphere_scene()
    k = make_intrinsics(FOCAL, FOCAL, (IMG_SIZE - 1) / 2.0, (IMG_SIZE - 1) / 2.0)
    poses = make_orbit_poses(24, radius=ORBIT_R, elevation=0.25)
    img0 = render_rgb(scene, k, poses[0], IMG_SIZE, IMG_SIZE, checker_texture)
    img1 = render_rgb(scene, k, poses[23], IMG_SIZE, IMG_SIZE, checker_texture)

    # 两视角（方位差 ±15°）都能看到的表面点：取两相机方向角平分线方向的球面点。
    d0 = poses[0][:3, 3] / np.linalg.norm(poses[0][:3, 3])
    d1 = poses[23][:3, 3] / np.linalg.norm(poses[23][:3, 3])
    bisector = (d0 + d1) / np.linalg.norm(d0 + d1)
    surf = SPHERE_R * bisector
    for img, t in ((img0, poses[0]), (img1, poses[23])):
        cam = (surf - t[:3, 3]) @ t[:3, :3]        # 世界 -> 相机
        u = FOCAL * cam[0] / cam[2] + (IMG_SIZE - 1) / 2.0
        v = FOCAL * cam[1] / cam[2] + (IMG_SIZE - 1) / 2.0
        ui, vi = int(round(u)), int(round(v))
        assert 0 <= ui < IMG_SIZE and 0 <= vi < IMG_SIZE
        assert np.abs(img[vi, ui] - checker_texture(surf[None])[0]).max() < 0.02, \
            "同一世界点的颜色随视角改变（纹理不是世界坐标函数）"
    # 无回波像素（天空）为黑；值噪声纹理确定性且值域合法。
    gt0 = render_depth(scene, k, poses[0], IMG_SIZE, IMG_SIZE)
    assert np.all(img0[~np.isfinite(gt0)] == 0.0)
    nv = value_noise_texture(surf[None], seed=0)
    assert np.array_equal(nv, value_noise_texture(surf[None], seed=0))
    assert 0.0 <= nv.min() and nv.max() <= 1.0


# --------------------------------------------------------- 深度恢复精度 ---
def test_depth_recovery_relative_error():
    """无噪纹理 + 已知位姿：恢复深度与真值的平均相对误差 < 5%（任务验收线）。"""
    k, poses, images = _setup(lambda p: checker_texture(p, cell=0.08))
    gt = render_depth(_sphere_scene(), k, poses[0], IMG_SIZE, IMG_SIZE)
    out = run_plane_sweep(images[0], [images[1], images[23]], k, poses[0],
                          [poses[1], poses[23]], depth_min=0.85, depth_max=2.0,
                          n_depths=60, window=7)
    m = np.isfinite(gt)
    rel = np.abs(out["depth"][m] - gt[m]) / gt[m]
    assert rel.mean() < 0.05, rel.mean()
    print(f"\n[plane_sweep] 深度恢复：mean rel err = {rel.mean():.4f} "
          f"(median {np.median(rel):.4f}, n={int(m.sum())})")


# ------------------------------------------------------- 置信度退化实验 ---
def test_confidence_drops_at_occlusion_boundary():
    """遮挡边界（投影轮廓线）处置信度显著低于球面内部（与解析几何对照）。"""
    k, poses, images = _setup(lambda p: checker_texture(p, cell=0.08))
    gt = render_depth(_sphere_scene(), k, poses[0], IMG_SIZE, IMG_SIZE)
    out = run_plane_sweep(images[0], [images[1], images[23]], k, poses[0],
                          [poses[1], poses[23]], depth_min=0.85, depth_max=2.0,
                          n_depths=60, window=7)
    # 解析轮廓半径（像素）：rho = f·tan(asin(r/d))；轮廓带 = |radius - rho| <= 1.5 px。
    dist = np.linalg.norm(poses[0][:3, 3])
    rho = FOCAL * np.tan(np.arcsin(SPHERE_R / dist))
    cx = cy = (IMG_SIZE - 1) / 2.0
    vv, uu = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    radius = np.hypot(uu - cx, vv - cy)
    m = np.isfinite(gt)
    limb = m & (np.abs(radius - rho) <= 1.5)
    interior = m & (radius <= 0.6 * rho)
    conf = out["confidence"]
    assert interior.sum() > 200 and limb.sum() > 30
    # 内部置信度显著更高：绝对差 > 0.015 且轮廓带低于内部的一半（确定性种子，
    # 实测 内部 ≈ 0.028 / 轮廓带 ≈ 0.009，比值 ≈ 3.3×）。
    assert conf[interior].mean() > 0.02, conf[interior].mean()
    assert conf[limb].mean() < 0.5 * conf[interior].mean(), \
        (conf[interior].mean(), conf[limb].mean())
    print(f"\n[plane_sweep] 置信度：内部 {conf[interior].mean():.4f} vs "
          f"轮廓带 {conf[limb].mean():.4f}")


def test_confidence_drops_with_weak_texture():
    """弱纹理（常色纹理）-> 窗口无常方差 -> NCC 无判别力 -> 置信度全零。"""
    strong_k, strong_poses, strong = _setup(lambda p: checker_texture(p, cell=0.08))
    weak_k, weak_poses, weak = _setup(lambda p: np.full((len(p), 3), 0.5))
    kw = dict(depth_min=0.85, depth_max=2.0, n_depths=60, window=7)
    out_strong = run_plane_sweep(strong[0], [strong[1], strong[23]],
                                 strong_k, strong_poses[0],
                                 [strong_poses[1], strong_poses[23]], **kw)
    out_weak = run_plane_sweep(weak[0], [weak[1], weak[23]],
                               weak_k, weak_poses[0],
                               [weak_poses[1], weak_poses[23]], **kw)
    assert out_weak["confidence"].max() >= 0.0            # 形状/取值合法
    gt = render_depth(_sphere_scene(), strong_k, strong_poses[0], IMG_SIZE, IMG_SIZE)
    m = np.isfinite(gt)
    # 球盘深处的窗口完全落在常色区域内（std = 0 -> 全部深度并列最差代价 -> 0）。
    cx = cy = (IMG_SIZE - 1) / 2.0
    vv, uu = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    radius = np.hypot(uu - cx, vv - cy)
    dist = np.linalg.norm(strong_poses[0][:3, 3])
    rho = FOCAL * np.tan(np.arcsin(SPHERE_R / dist))
    deep = m & (radius <= 0.6 * rho - 4.0)
    assert out_weak["confidence"][deep].max() < 1e-12
    assert out_strong["confidence"][deep].mean() > 0.02
    print(f"\n[plane_sweep] 弱纹理 vs 强纹理（球盘内部平均置信度）："
          f"{out_weak['confidence'][deep].mean():.4f} vs {out_strong['confidence'][deep].mean():.4f}")


# ------------------------------------------------------- 融合 vs 单视角 ---
def test_fused_point_cloud_beats_single_view():
    """端到端：12 参考视角的扫描深度 -> TSDF 融合点云 Chamfer 优于单视角（(M.1)）。

    融合前按置信度过滤（二名代价差 <= 0.015 的像素置 +inf，MVSNet 用概率体
    P(d) 做同一件事）——低置信度的遮挡边界/背景像素会往 TSDF 里注入幽灵
    深度；单视角基线用同一张过滤后的深度图（公平对照覆盖缺口）。
    """
    scene = _sphere_scene()
    k, poses, images = _setup(lambda p: checker_texture(p, cell=0.08))
    refs = list(range(0, 24, 2))                          # 12 个参考视角（30° 步进）
    conf_tau = 0.015
    depth_maps = {}
    for r_idx in refs:
        srcs = [(r_idx - 1) % 24, (r_idx + 1) % 24]       # ±15° 邻图
        out = run_plane_sweep(images[r_idx], [images[s] for s in srcs], k,
                              poses[r_idx], [poses[s] for s in srcs],
                              depth_min=0.85, depth_max=2.0, n_depths=48, window=7)
        depth = out["depth"].copy()
        depth[out["confidence"] <= conf_tau] = np.inf     # 置信度过滤（+inf = 无效）
        depth_maps[r_idx] = depth

    gt = sample_scene_surface(scene, 4000, band=0.05, seed=0)
    # 融合路线：(3.12) 反投影 -> 第 04 章 (4.8) TSDF 融合 -> MT 提取 -> 采样。
    vol = TSDFVolume(np.full(3, -1.0), np.full(3, 1.0), resolution=48)
    for r_idx in refs:
        vol.integrate(depth_maps[r_idx], k, poses[r_idx])
    mesh = marching_tetrahedra(vol.extraction_field(),
                               vol.origin + 0.5 * vol.voxel_size, vol.voxel_size)
    rec_fused = sample_mesh_surface(mesh.vertices, mesh.triangles, 4000, seed=0)
    cd_fused = chamfer_distance(rec_fused, gt)
    # 单视角基线：同一张（过滤后的）扫描深度图的 (3.12) 反投影。
    single = depth_to_point_cloud(depth_maps[refs[0]], k, poses[refs[0]])
    cd_single = chamfer_distance(single, gt)
    assert cd_fused < cd_single, (cd_fused, cd_single)
    print(f"\n[plane_sweep] Chamfer：融合 {cd_fused:.6f} m^2 vs 单视角 {cd_single:.6f} m^2")


if __name__ == "__main__":
    # 单点测试：python tests/test_plane_sweep.py <测试名子串>；不带参数 = 全部测试。
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

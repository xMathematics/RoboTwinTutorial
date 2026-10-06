"""projects/3d_reconstruction/poisson 的测试 —— 直接运行：``python tests/test_poisson.py``。

验证 Poisson 表面重建教学代理（教程 (5.3)-(5.8)，推导教程见 ../TUTORIAL.md §5）：
FFT 求解器与离散拉普拉斯的解析对恢复（1e-10 级）、单位球点云 + 解析外向法向的
重建半径偏差（< 0.05）、隐式场的指示函数尺度（内 1 外 0）、法向噪声的误差
单调性、以及与 TSDF 路线（tsdf + marching）在同一球场景上的 Chamfer 对比。
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

from poisson import (  # noqa: E402
    implicit_from_points,
    reconstruct_mesh,
    solve_poisson_fft,
    _trilinear,
)
from scene import make_sphere_scene, render_dataset, sample_scene_surface  # noqa: E402
from tsdf import TSDFVolume  # noqa: E402
from marching import marching_tetrahedra  # noqa: E402


def _unit_sphere_points(n_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """单位球面均匀采样点 + 解析外向法向（法向 = 位置方向，Eikonal (6.1)）。"""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction, direction.copy()


def _discrete_laplacian(field: np.ndarray, h: float) -> np.ndarray:
    """周期边界中心差分离散拉普拉斯（与 solve_poisson_fft 的特征值配套）。"""
    out = np.zeros_like(field)
    for k in range(3):
        out += (np.roll(field, 1, axis=k) - 2.0 * field
                + np.roll(field, -1, axis=k)) / (h * h)
    return out


# ------------------------------------------------------------- FFT 求解器 --
def test_fft_solver_recovers_discrete_laplacian_pair():
    """解析对恢复：χ₀ = 三轴正弦积的离散拉普拉斯作右端，解回 χ₀ 到 1e-10。

    用**离散**拉普拉斯（周期卷积）构造右端而非连续公式——求解器的特征值
    (2cos(2πm/N)−2)/h² 本身就是离散算子，二者须精确互逆（连续式的 O(h²)
    截断差属另一回事，不进断言）。
    """
    n, h = 24, 1.0 / 24
    ax = np.arange(n) / n
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    chi0 = (np.sin(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
            * np.sin(2 * np.pi * zz))
    rhs = _discrete_laplacian(chi0, h)
    assert abs(rhs.mean()) < 1e-12                      # 可解性条件（零均值）
    chi = solve_poisson_fft(rhs, h)
    expect = chi0 - chi0.mean()                          # 解取均值零的代表
    assert np.abs(chi - expect).max() < 1e-10
    # 混频模式（不同频率叠加）同理，排除"只对单一模式凑巧"。
    chi1 = np.sin(2 * np.pi * 3 * xx) * np.cos(2 * np.pi * 2 * yy) * np.ones_like(zz)
    rhs1 = _discrete_laplacian(chi1, h)
    chi1_sol = solve_poisson_fft(rhs1, h)
    assert np.abs(chi1_sol - (chi1 - chi1.mean())).max() < 1e-10


# ------------------------------------------------------- 球体重建精度 ------
def test_sphere_reconstruction_radius_deviation():
    """单位球点云 + 解析法向 -> 重建网格顶点平均半径偏差 < 0.05（任务验收线）。"""
    pts, nrms = _unit_sphere_points(2000, seed=0)
    mesh = reconstruct_mesh(pts, nrms, resolution=40, pad_fraction=0.3,
                            sigma_voxels=1.5)
    assert len(mesh.triangles) > 100                     # 提取出的是真网格
    radius_err = np.abs(np.linalg.norm(mesh.vertices, axis=1) - 1.0)
    assert radius_err.mean() < 0.05, radius_err.mean()
    # 网格应水密地包住球心：球心到最近顶点距离显著大于半径偏差（无破面漏 interior）。
    print(f"\n[poisson] sphere V={len(mesh.vertices)} T={len(mesh.triangles)} "
          f"mean|r-1|={radius_err.mean():.4f} max={radius_err.max():.4f}")


def test_implicit_field_indicator_jump():
    """隐式场是 (5.3) 指示函数的教学版：内部 ≈ 1、外部 ≈ 0、γ ≈ 0.5。"""
    pts, nrms = _unit_sphere_points(2000, seed=0)
    chi, origin, voxel, isolevel = implicit_from_points(
        pts, nrms, resolution=40, pad_fraction=0.3, sigma_voxels=1.5)
    queries = np.array([[0.0, 0.0, 0.0],     # 球心（内部深处）
                        [0.5, 0.0, 0.0],     # 内部近表面
                        [1.8, 0.0, 0.0]])    # 外部（网格 pad 内）
    vals = _trilinear(chi, origin, voxel, queries)
    assert vals[0] > 0.8, vals                            # 内部平台 ≈ 1
    assert vals[1] > 0.8, vals
    assert abs(vals[2]) < 0.2, vals                       # 外部平台 ≈ 0
    assert 0.3 < isolevel < 0.7, isolevel                 # γ ≈ 0.5（(5.8)）
    print(f"\n[poisson] chi = {np.round(vals, 3)}, isolevel = {isolevel:.3f}")


def test_normal_noise_degrades_accuracy_monotonic():
    """法向扰动越大重建越差（单调性 sanity）：噪声 σ = 0/0.35/0.7 误差递增。"""
    pts, nrms = _unit_sphere_points(2000, seed=0)
    rng = np.random.default_rng(3)
    errors = []
    for sigma in (0.0, 0.35, 0.7):
        noisy = nrms + rng.normal(0.0, sigma, size=nrms.shape)
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        mesh = reconstruct_mesh(pts, noisy, resolution=32, pad_fraction=0.3,
                                sigma_voxels=1.5)
        errors.append(float(np.abs(np.linalg.norm(mesh.vertices, axis=1) - 1.0).mean()))
    assert errors[0] < 0.05
    assert errors[0] < errors[1] < errors[2], errors
    print(f"\n[poisson] radius err vs normal noise: {np.round(errors, 4)}")


# ------------------------------------------------------- 与 TSDF 路线对比 --
def test_chamfer_against_tsdf_route_on_sphere():
    """同一球场景上 Poisson 与 TSDF（tsdf+MT）两条路线的 Chamfer 对比输出。

    两者都应把球面重建到网格离散化水平（Chamfer < 5e-3 m²）；本测试只断言
    各自的健康上限并打印对比数字（谁更优取决于分辨率与数据 cleanliness，
    不作优劣断言——Poisson 拿到的是无噪声解析法向，TSDF 拿到的是干净深度，
    两条路线的误差来源不同，见 TUTORIAL.md §3.6）。
    """
    scene = make_sphere_scene(1.0)
    gt = sample_scene_surface(scene, 4000, band=0.05, seed=0)
    normals = gt / np.linalg.norm(gt, axis=1, keepdims=True)

    # 路线一：Poisson（点云 + 解析法向 -> 隐式场 -> MT 提取）。
    mesh_p = reconstruct_mesh(gt, normals, resolution=40, pad_fraction=0.3,
                              sigma_voxels=1.5)
    rec_p = sample_mesh_surface(mesh_p.vertices, mesh_p.triangles, 4000, seed=0)
    cd_p = chamfer_distance(rec_p, gt)

    # 路线二：TSDF 融合（24 视角无噪深度 -> (4.8) 融合 -> MT 提取）。
    data = render_dataset(scene, n_views=16, img_size=48, focal=45.0,
                          radius=3.0, depth_noise_coeff=0.0, seed=0)
    vol = TSDFVolume(np.full(3, -1.8), np.full(3, 1.8), resolution=48)
    for t_wc, depth in zip(data["poses"], data["depths"]):
        vol.integrate(depth, data["K"], t_wc)
    mesh_t = marching_tetrahedra(vol.extraction_field(),
                                 vol.origin + 0.5 * vol.voxel_size, vol.voxel_size)
    rec_t = sample_mesh_surface(mesh_t.vertices, mesh_t.triangles, 4000, seed=0)
    cd_t = chamfer_distance(rec_t, gt)

    assert cd_p < 5e-3, cd_p                             # 体素 0.04 m 的离散化水平
    assert cd_t < 5e-3, cd_t
    print(f"\n[poisson] 球场景 Chamfer 对比：Poisson {cd_p:.6f} m^2 vs "
          f"TSDF+MT {cd_t:.6f} m^2")


if __name__ == "__main__":
    # 单点测试：python tests/test_poisson.py <测试名子串>；不带参数 = 全部测试。
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

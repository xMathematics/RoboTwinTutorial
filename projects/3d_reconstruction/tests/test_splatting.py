"""projects/3d_reconstruction/splatting 的测试 —— 直接运行：``python tests/test_splatting.py``。

验证 3DGS 前向泼溅的教学代理（教程 08 章 (8.4)-(8.10)）：单高斯的屏幕 alpha
与解析式逐像素一致（含 EWA 低通项）、投影协方差 (8.8) 的闭式交叉验证、
深度排序的 alpha 合成权重单调（前景遮挡后景 + 完全遮挡 + 输入乱序不变）、
球面高斯云的轮廓半径与解析值一致（±2%）、空/全裁剪场景全透明。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene import make_orbit_poses  # noqa: E402
from splatting import (  # noqa: E402
    LOW_PASS_PX2,
    SplatCloud,
    project_splats,
    sphere_splat_cloud,
    splat_render,
    tangent_basis,
)


def _axis_splat(position: np.ndarray, sigma: float, opacity: float,
                color: np.ndarray) -> SplatCloud:
    """正对相机（法向 −z）的单基元云：切基与像平面平行，便于闭式计算。"""
    position = np.asarray(position, dtype=float)[None]
    return SplatCloud(
        positions=position,
        tangent_u=np.array([[1.0, 0.0, 0.0]]),
        tangent_v=np.array([[0.0, 1.0, 0.0]]),
        sigma=np.array([sigma]),
        opacity=np.array([opacity]),
        colors=color[None],
    )


def _simple_camera(focal: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """相机在原点、+z 为光轴的恒等位姿（T_wc = I）与居中内参。"""
    k = np.array([[focal, 0.0, (size - 1) / 2.0],
                  [0.0, focal, (size - 1) / 2.0],
                  [0.0, 0.0, 1.0]])
    return k, np.eye(4)


# ------------------------------------------------------- 投影与解析 alpha --
def test_single_splat_alpha_matches_analytic():
    """单高斯（轴上、切基平行像平面）：alpha 图 == 解析 2D 高斯（1e-9 级）。

    相机系中心 (0,0,1)、f=30：J = diag(30, 30)，Σ' = σ_px²I + 低通项，
    α(u,v) = o·exp(−((u−c)²+(v−c)²)/(2(σ_px²+0.3)))——逐像素闭式对照。
    """
    size, focal, sigma_m, opacity = 61, 30.0, 0.3, 1.0
    k, t_wc = _simple_camera(focal, size)
    cloud = _axis_splat(np.array([0.0, 0.0, 1.0]), sigma_m, opacity,
                        np.array([1.0, 0.0, 0.0]))
    out = splat_render(cloud, k, t_wc, size, size)
    sigma_px2 = (focal * sigma_m / 1.0) ** 2 + LOW_PASS_PX2
    c = (size - 1) / 2.0
    vv, uu = np.mgrid[0:size, 0:size]
    expect = opacity * np.exp(-((uu - c) ** 2 + (vv - c) ** 2) / (2.0 * sigma_px2))
    assert np.abs(out["alpha_total"] - expect).max() < 1e-9
    # 中心 = 不透明度本身（exp(0) = 1）；1σ_px 处 = e^{-1/2}（各向同性圆对称）。
    assert abs(out["alpha_total"][int(c), int(c)] - opacity) < 1e-9
    off = int(round(focal * sigma_m))
    # 1σ_px 处（低通项并入解析值）：α = o·exp(−1/2·off²/(σ_px²+低通))。
    assert abs(out["alpha_total"][int(c), int(c) + off]
               - opacity * np.exp(-off * off / (2.0 * sigma_px2))) < 1e-9
    assert abs(out["alpha_total"][int(c) + off, int(c)]
               - out["alpha_total"][int(c), int(c) + off]) < 1e-12
    # 单基元全前景：透射率 = 1 − α。
    assert np.abs(out["transmittance"] - (1.0 - expect)).max() < 1e-12


def test_project_splats_matches_ewa_closed_form():
    """(8.6)/(8.8)：离轴倾斜切基的 Σ' 与 JWΣWᵀJᵀ 手工展开一致（1e-12）。"""
    mu_cam = np.array([[0.5, 0.2, 2.0]])
    u = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2.0)
    v = np.array([[-1.0, 1.0, 0.0]]) / np.sqrt(2.0)
    assert abs(np.dot(u[0], v[0])) < 1e-15              # 切基正交
    sigma = np.array([0.1])
    k = np.array([[50.0, 0.0, 31.5], [0.0, 50.0, 31.5], [0.0, 0.0, 1.0]])
    mu2, cov2 = project_splats(mu_cam, u, v, sigma, k)
    # 手工展开 (8.6) 雅可比 + (8.8) 协方差传播（含低通项）。
    fx = fy = 50.0
    j = np.array([[fx / 2.0, 0.0, -fx * 0.5 / 4.0],
                  [0.0, fy / 2.0, -fy * 0.2 / 4.0]])
    a_u, a_v = j @ u[0], j @ v[0]
    expect = 0.01 * (np.outer(a_u, a_u) + np.outer(a_v, a_v)) \
        + LOW_PASS_PX2 * np.eye(2)
    assert np.abs(cov2[0] - expect).max() < 1e-12
    assert abs(mu2[0, 0] - (fx * 0.5 / 2.0 + 31.5)) < 1e-12  # 针孔投影 π (8.5)（含主点）
    assert abs(mu2[0, 1] - (fy * 0.2 / 2.0 + 31.5)) < 1e-12
    # 切平面基元的协方差是半正定的（(8.3) 的屏幕版：特征值 > 低通项）。
    eig = np.linalg.eigvalsh(cov2[0])
    assert eig.min() >= LOW_PASS_PX2 - 1e-12


# ------------------------------------------------------- 排序与 alpha 合成 --
def test_depth_sorting_front_occludes_back():
    """同轴双高斯：合成色 == 闭式 α₁c₁+(1−α₁)α₂c₂（(8.10)）；输入乱序不变。

    完全不透明的前景（o=1）让透射率归零——后景合成权重 α₂(1−α₁) 随透射率
    单调衰减到 0（前景遮挡后景）。
    """
    size, focal = 33, 30.0
    k, t_wc = _simple_camera(focal, size)
    red, blue = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    center = (size - 1) // 2

    def render(front_z: float, front_o: float, back_z: float,
               swap: bool = False) -> dict:
        front = _axis_splat(np.array([0.0, 0.0, front_z]), 0.2, front_o, red)
        back = _axis_splat(np.array([0.0, 0.0, back_z]), 0.2, 0.8, blue)
        cloud = SplatCloud(
            positions=np.concatenate([back.positions, front.positions])
            if swap else np.concatenate([front.positions, back.positions]),
            tangent_u=np.concatenate([back.tangent_u, front.tangent_u]) if swap
            else np.concatenate([front.tangent_u, back.tangent_u]),
            tangent_v=np.concatenate([back.tangent_v, front.tangent_v]) if swap
            else np.concatenate([front.tangent_v, back.tangent_v]),
            sigma=np.concatenate([back.sigma, front.sigma]) if swap
            else np.concatenate([front.sigma, back.sigma]),
            opacity=np.concatenate([back.opacity, front.opacity]) if swap
            else np.concatenate([front.opacity, back.opacity]),
            colors=np.concatenate([back.colors, front.colors]) if swap
            else np.concatenate([front.colors, back.colors]),
        )
        return splat_render(cloud, k, t_wc, size, size)

    # 部分遮挡闭式解：C = 0.6·red + (1−0.6)·0.8·blue（两个高斯中心同投到像素中心）。
    out = render(1.0, 0.6, 1.5)
    expect = 0.6 * red + 0.4 * 0.8 * blue
    assert np.abs(out["image"][center, center] - expect).max() < 1e-9
    out_swapped = render(1.0, 0.6, 1.5, swap=True)
    assert np.abs(out_swapped["image"][center, center] - expect).max() < 1e-9, \
        "深度排序失效：输入顺序改变了合成结果"
    # 完全遮挡：o=1 的前景吃掉全部透射率，后景贡献严格为 0。
    out_full = render(1.0, 1.0, 1.5)
    assert np.abs(out_full["image"][center, center] - red).max() < 1e-9
    assert out_full["transmittance"][center, center] == 0.0


# ------------------------------------------------- 球面轮廓与空场景 --------
def test_sphere_silhouette_radius_matches_analytic():
    """球面高斯云渲染的 alpha=0.5 轮廓半径 == f·tan(asin(r/d))（±2%）。

    设计注记：alpha 轮廓 = 表面轮廓 + EWA 低通地板（Σ' 对角 +0.3 px²，教程
    8.2 ⑤(b)）带来的 ~1.2 px 恒定外扩——与 σ 无关（边缘朝向的 splat 径向
    宽度被低通项垫底）。故用长焦（f=400）把解析轮廓半径抬到 ~90 px，使
    fringe 的相对量 < 2%；σ_px = 0.8、相邻间距 2.2σ 保证盘内饱和。
    """
    focal, size, radius, dist = 400.0, 200, 0.35, 1.6
    sigma = 0.8 * (dist - radius) / focal              # σ_px ≈ 0.8
    spacing = 2.2 * sigma                              # 相邻间距（重叠充分）
    n_splats = int(np.ceil(4.0 * np.pi * radius * radius / spacing ** 2))
    cloud = sphere_splat_cloud(radius=radius, n_splats=n_splats, sigma=sigma,
                               opacity=0.5)
    k = np.array([[focal, 0.0, (size - 1) / 2.0],
                  [0.0, focal, (size - 1) / 2.0],
                  [0.0, 0.0, 1.0]])
    t_wc = make_orbit_poses(1, radius=dist, elevation=0.0)[0]  # 光轴过球心
    out = splat_render(cloud, k, t_wc, size, size)
    alpha = out["alpha_total"]
    mask = alpha >= 0.5
    assert mask.sum() > 1000                           # 球盘确实被点亮
    # 解析轮廓半径：rho = f·tan(asin(r/d))。
    rho = focal * np.tan(np.arcsin(radius / dist))
    c = (size - 1) / 2.0
    vv, uu = np.mgrid[0:size, 0:size]
    rad = np.hypot(uu - c, vv - c)
    assert alpha[rad < 0.6 * rho].min() > 0.5           # 盘内饱和（基元重叠充分）
    assert alpha[rad > 1.4 * rho].max() < 1e-6          # 盘外全透明
    # 轮廓半径（亚像素）：360 个方向沿径向找 alpha 穿越 0.5 的位置（线性插值
    # 消掉像素中心 ~0.5 px 的量化偏置），取均值与解析值比（±2%）。
    thetas = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    steps = np.arange(0.0, 2.0 * rho, 0.1)
    dirs = np.column_stack([np.cos(thetas), np.sin(thetas)])
    samples = dirs[:, None, :] * steps[None, :, None] + np.array([c, c])
    si = np.clip(np.rint(samples[..., 0]).astype(int), 0, size - 1)
    ti = np.clip(np.rint(samples[..., 1]).astype(int), 0, size - 1)
    profile = alpha[ti, si]                             # (360, S) 径向 alpha 剖面
    below = profile < 0.5
    first = np.argmax(below, axis=1)                    # 首个 < 0.5 的下标
    valid = below.any(axis=1) & (first > 0)
    rows = np.nonzero(valid)[0]
    a_out = profile[rows, first[valid]]
    a_in = profile[rows, first[valid] - 1]
    frac = (a_in - 0.5) / np.maximum(a_in - a_out, 1e-12)
    crossings = steps[first[valid] - 1] + 0.1 * frac
    mean_r = float(crossings.mean())
    assert crossings.std() < 0.6, crossings.std()       # 轮廓的圆度（各方向一致）
    assert abs(mean_r - rho) / rho < 0.02, (mean_r, rho)
    print(f"\n[splatting] 轮廓半径：渲染 {mean_r:.3f} px vs 解析 {rho:.3f} px "
          f"({100 * abs(mean_r - rho) / rho:.2f}%)")
    # 同输入两次渲染逐位一致（确定性）。
    again = splat_render(cloud, k, t_wc, size, size)
    assert np.array_equal(again["image"], out["image"])


def test_empty_and_behind_camera_fully_transparent():
    """空场景 / 基元全在相机后方：图像全黑、透射率全 1（(8.10) 的空和）。"""
    size, focal = 16, 30.0
    k, t_wc = _simple_camera(focal, size)
    empty = SplatCloud(positions=np.zeros((0, 3)), tangent_u=np.zeros((0, 3)),
                       tangent_v=np.zeros((0, 3)), sigma=np.zeros(0),
                       opacity=np.zeros(0), colors=np.zeros((0, 3)))
    out = splat_render(empty, k, t_wc, size, size)
    assert out["image"].shape == (size, size, 3)
    assert np.all(out["image"] == 0.0)
    assert np.all(out["transmittance"] == 1.0)
    assert np.all(out["alpha_total"] == 0.0)
    # 全部基元在相机后方（z < 0 < Z_NEAR）同样全透明。
    behind = _axis_splat(np.array([0.0, 0.0, -1.0]), 0.2, 1.0,
                         np.array([1.0, 1.0, 1.0]))
    out_b = splat_render(behind, k, t_wc, size, size)
    assert np.all(out_b["transmittance"] == 1.0)
    assert np.all(out_b["image"] == 0.0)
    # tangent_basis：正交单位基且法向分量参与叉积闭环（球面铺云的前置）。
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    u, v = tangent_basis(normals)
    for i in range(3):
        assert abs(np.dot(normals[i], u[i])) < 1e-12
        assert abs(np.dot(u[i], v[i])) < 1e-12
        assert abs(np.linalg.norm(u[i]) - 1.0) < 1e-12


if __name__ == "__main__":
    # 单点测试：python tests/test_splatting.py <测试名子串>；不带参数 = 全部测试。
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

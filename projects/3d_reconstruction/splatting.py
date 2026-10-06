"""3DGS 前向泼溅渲染的教学代理（splatting）—— 教程第 08 章 8.1/8.2 节的教学实现。

函数流水线
----------
本模块实现 3D Gaussian Splatting（Kerbl et al., SIGGRAPH 2023）**渲染侧**的
前向管线：给定一组带切平面协方差、不透明度与颜色的 2D 高斯基元（贴在
``scene`` 的球面等表面上有解析切平面），投影成像：

    tangent_basis       （法向 -> 切平面正交基 u, v；协方差 Σ = σ²(uuᵀ+vvᵀ)，(8.2)）
        -> project_splats    （刚体变换 (8.4) -> 局部仿射雅可比 (8.5)-(8.6) ->
                              协方差传播 (8.7)-(8.8)：Σ' = JWΣWᵀJᵀ + 低通项）
        -> splat_render      （按深度排序（近 -> 远）-> 逐像素 2D 高斯求值 (8.9)
                              -> alpha 合成 C = Σ c_k α_k Π_{j<k}(1−α_j) (8.10)）

依赖方向：本模块不依赖库内其他模块（仅 numpy）；数据侧的球面基元可用
``scene.Sphere`` 的解析法向构造；``tests/test_splatting.py`` 直接驱动。

输入/输出：输入基元数组——位置 (M, 3)（单位 m）、切基 (M, 3) 各两支（单位
向量）、σ (M,)（单位 m）、不透明度 (M,)（[0,1]）、颜色 (M, 3)（[0,1]）；内参
K (3, 3)（px）、外参 T_wc (4, 4)。输出：RGB 图 (H, W, 3)、透射率 T (H, W)
与总 alpha = 1 − T (H, W)（无量纲）。全部确定性，无文件 I/O。

教学简化（与原文的差异，README 同款声明）
------------------------------------------
本模块**只做前向泼溅渲染**。3DGS 的三大优化侧组件全部不做：① 可微光栅化
（反向传播需要缓存 α 的 tile 化回扫与解析梯度，依赖自动微分框架）；
② 自适应致密化（位置梯度驱动的分裂/克隆/剪枝与不透明度重置，教程 8.3，
论文经验规则）；③ 球谐（SH）视角相关颜色（本模块颜色常量）。基元限定为
**各向同性 2D 高斯**（切平面内 σ²I，无各向异性缩放矩阵），tile 化光栅化
简化为"逐高斯全图求值 + Python 循环"——保结构正确性，不保论文性能。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SplatCloud",
    "project_splats",
    "sphere_splat_cloud",
    "splat_render",
    "tangent_basis",
]

# EWA 低通项（px^2，加在 2D 协方差对角上）：亚像素高斯在放大时的闪烁/走样
# 抑制（教程 8.2 ⑤(b) 的实现注记；同时保证边缘朝向的 splat 协方差可逆）。
LOW_PASS_PX2 = 0.3
# 相机系最近深度（m）：更近/相机后方的基元不参与渲染（视锥裁剪）。
Z_NEAR = 0.05
# 前向合成的透射率终止阈值：所有像素 T 低于它即提前结束（遮挡后不浪费求值）。
T_MIN = 1e-4
# 每个 splat 的求值半径（投影协方差最大特征值平方根的倍数）：bbox 之外的
# alpha < e^{-12.5}，截断误差远低于全部断言阈值——tile 化光栅化的教学版
# （教程 8.2 实现要点：按 footprint 复制到覆盖的 tile，避免全图求值）。
EVAL_RADIUS_SIGMAS = 5.0


@dataclass(frozen=True)
class SplatCloud:
    """2D 高斯基元集合（贴表面：位置 + 切平面协方差 + 不透明度 + 颜色）。

    Attributes:
        positions: (M, 3) 基元中心世界坐标，单位 m。
        tangent_u: (M, 3) 切平面第一基（单位向量，与 v、法向正交）。
        tangent_v: (M, 3) 切平面第二基。协方差 Σ = σ²(uuᵀ + vvᵀ)（(8.2) 的
            各向同性切平面特例，教程 8.1）。
        sigma: (M,) 高斯标准差，单位 m，取值 > 0。
        opacity: (M,) 不透明度 o_k，无量纲，[0, 1]。
        colors: (M, 3) RGB 颜色，[0, 1]（教学简化：常量色，非球谐 SH）。
    """

    positions: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    sigma: np.ndarray
    opacity: np.ndarray
    colors: np.ndarray


def tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """单位法向 -> 切平面正交基 (u, v)（u = normalize(up × n) 的稳定版本）。

    辅助轴取与法向夹角最大的坐标轴（避免 n ∥ up 时退化），再
    u = normalize(aux × n)、v = n × u（右手正交，u ⊥ v ⊥ n）。

    Args:
        normals: (M, 3) 法向（内部按模长归一）。

    Returns:
        (tangent_u, tangent_v)：各 (M, 3) 单位向量。
    """
    n = np.asarray(normals, dtype=float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    # 每行选 |n·axis| 最小的坐标轴作辅助轴（叉积模长 = sqrt(1 - (n·axis)^2) 最大）。
    aux = np.eye(3)[np.argmin(np.abs(n), axis=-1)]
    u = np.cross(aux, n)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v = np.cross(n, u)
    return u, v


def sphere_splat_cloud(
    radius: float = 0.35,
    n_splats: int = 400,
    sigma: float | None = None,
    opacity: float = 0.9,
    color: tuple[float, float, float] = (0.85, 0.85, 0.85),
) -> SplatCloud:
    """在球面上均匀铺一层切平面 2D 高斯（Fibonacci 螺旋，确定性无随机）。

    位置 = 球面 Fibonacci 均匀点（相邻间距 ≈ sqrt(4πr²/M)），法向 = 径向，
    协方差由 :func:`tangent_basis` 的切平面给出；σ 默认取 0.5×间距（高斯
    相邻重叠 ~2σ，保证 alpha 合成后球盘内部饱和、轮廓光滑）。

    Args:
        radius: 球半径，单位 m。
        n_splats: 基元数，取值 >= 4。
        sigma: 高斯标准差，单位 m；None 取 0.5×Fibonacci 间距。
        opacity: 全体基元的不透明度，[0, 1]。
        color: 全体基元的 RGB 颜色（教学简化：常量色）。

    Returns:
        cloud: :class:`SplatCloud`。
    """
    if n_splats < 4:
        raise ValueError(f"n_splats 须 >= 4，当前 {n_splats}")
    idx = np.arange(n_splats)
    z = 1.0 - (2.0 * idx + 1.0) / n_splats            # 均匀覆盖 [-1, 1] 的黄金螺旋
    ring = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    az = idx * np.pi * (3.0 - np.sqrt(5.0))           # 黄金角
    normals = np.column_stack([ring * np.cos(az), ring * np.sin(az), z])
    spacing = np.sqrt(4.0 * np.pi * radius * radius / n_splats)
    sig = 0.5 * spacing if sigma is None else float(sigma)
    u, v = tangent_basis(normals)
    return SplatCloud(
        positions=radius * normals,
        tangent_u=u,
        tangent_v=v,
        sigma=np.full(n_splats, sig),
        opacity=np.full(n_splats, float(opacity)),
        colors=np.tile(np.asarray(color, dtype=float), (n_splats, 1)),
    )


def project_splats(
    mu_cam: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    sigma: np.ndarray,
    k_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """局部仿射近似（EWA）投影：3D 切平面高斯 -> 屏幕 2D 高斯（(8.5)-(8.8)）。

    雅可比在各高斯自己的中心处取值（(8.6)），协方差经 (8.7) 的引理传播：
    Σ' = J W Σ Wᵀ Jᵀ = σ²(a₁a₁ᵀ + a₂a₂ᵀ)，a_i = J (W t_i)；再加 EWA 低通项
    0.3 px² 保可逆（教程 8.2 ⑤(b)）。

    Args:
        mu_cam: (M, 3) 基元中心的相机系坐标（(8.4) 刚体变换后），单位 m。
        tangent_u, tangent_v: (M, 3) 相机系切基（世界系切基经 W 旋转）。
        sigma: (M,) 标准差，单位 m。
        k_matrix: (3, 3) 内参，px。

    Returns:
        (mu2, cov2)：mu2 (M, 2) 屏幕中心（px）、cov2 (M, 2, 2) 2D 协方差
        （px²，已含低通项）。
    """
    k_matrix = np.asarray(k_matrix, dtype=float)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    big_x, big_y, z = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]
    # (8.6)：J = [[fx/Z, 0, −fx X/Z²], [0, fy/Z, −fy Y/Z²]]。
    jacobian = np.stack([
        np.stack([fx / z, np.zeros_like(z), -fx * big_x / (z * z)], axis=-1),
        np.stack([np.zeros_like(z), fy / z, -fy * big_y / (z * z)], axis=-1),
    ], axis=1)                                        # (M, 2, 3)
    a_u = np.einsum("kij,kj->ki", jacobian, tangent_u)  # (M, 2) = J @ u
    a_v = np.einsum("kij,kj->ki", jacobian, tangent_v)
    sig2 = (sigma * sigma)[:, None, None]
    cov2 = sig2 * (a_u[..., None] * a_u[:, None, :]
                   + a_v[..., None] * a_v[:, None, :])
    cov2 += LOW_PASS_PX2 * np.eye(2)
    mu2 = np.column_stack([fx * big_x / z + k_matrix[0, 2],
                           fy * big_y / z + k_matrix[1, 2]])  # 针孔投影 π (8.5)
    return mu2, cov2


def splat_render(
    cloud: SplatCloud,
    k_matrix: np.ndarray,
    t_wc: np.ndarray,
    width: int,
    height: int,
) -> dict:
    """前向泼溅渲染：深度排序 + 逐像素 (8.9) 求值 + (8.10) alpha 合成。

    基元按相机系 z 从近到远排序（CUDA 实现的全局排序的教学版），逐基元
    合成 C ← C + c_k·α_k·T、T ← T·(1−α_k)（(8.10) 的前向形式）；全体像素
    T < T_MIN 时提前终止（完全遮挡后不再求值）。相机后方（z < Z_NEAR）的
    基元被裁剪。

    Args:
        cloud: :class:`SplatCloud` 基元集合。
        k_matrix: (3, 3) 内参，px。
        t_wc: (4, 4) 相机 -> 世界位姿，单位 m / rad。
        width, height: 图像宽/高，单位 px。

    Returns:
        dict：
        - ``"image"``: (height, width, 3) RGB 合成图，[0, 1]（空场景为黑）；
        - ``"transmittance"``: (height, width) 透射率 T，[0, 1]（空场景为 1）；
        - ``"alpha_total"``: (height, width) = 1 − T（累计不透明度）。
    """
    k_matrix = np.asarray(k_matrix, dtype=float)
    t_wc = np.asarray(t_wc, dtype=float)
    r_wc, t_vec = t_wc[:3, :3], t_wc[:3, 3]
    positions = np.asarray(cloud.positions, dtype=float).reshape(-1, 3)
    mu_cam = (positions - t_vec) @ r_wc                # (8.4) 世界 -> 相机（精确）

    image = np.zeros((height, width, 3))
    transmittance = np.ones((height, width))
    # 视锥裁剪 + 深度排序（近 -> 远；argsort 稳定，同深度保持输入序）。
    front = mu_cam[:, 2] > Z_NEAR
    if not front.any():
        return {"image": image, "transmittance": transmittance,
                "alpha_total": 1.0 - transmittance}
    order = np.argsort(mu_cam[front, 2], kind="stable")
    idx = np.nonzero(front)[0][order]

    tangent_u = np.asarray(cloud.tangent_u, dtype=float).reshape(-1, 3)[idx] @ r_wc
    tangent_v = np.asarray(cloud.tangent_v, dtype=float).reshape(-1, 3)[idx] @ r_wc
    mu2, cov2 = project_splats(mu_cam[idx], tangent_u, tangent_v,
                               np.asarray(cloud.sigma, dtype=float)[idx], k_matrix)
    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]
    # 2×2 逆的解析式（[[d,−b],[−c,a]]/det）。
    inv00, inv01 = cov2[:, 1, 1] / det, -cov2[:, 0, 1] / det
    inv11 = cov2[:, 0, 0] / det
    # 每个 splat 的求值半径：2D 协方差最大特征值的平方根（px）× 截断倍数。
    half_trace = 0.5 * (cov2[:, 0, 0] + cov2[:, 1, 1])
    eig_max = half_trace + np.sqrt(np.maximum(half_trace * half_trace - det, 0.0))
    radius_px = EVAL_RADIUS_SIGMAS * np.sqrt(np.maximum(eig_max, 1e-12))

    colors = np.asarray(cloud.colors, dtype=float).reshape(-1, 3)[idx]
    opacities = np.asarray(cloud.opacity, dtype=float)[idx]
    for m in range(len(idx)):
        # bbox 截断求值（tile 化光栅化的教学版，教程 8.2 实现要点）：
        # bbox 外 alpha < exp(-EVAL_RADIUS_SIGMAS²/2)，远低于全部断言阈值。
        u0 = max(int(mu2[m, 0] - radius_px[m]), 0)
        u1 = min(int(mu2[m, 0] + radius_px[m]) + 1, width)
        v0 = max(int(mu2[m, 1] - radius_px[m]), 0)
        v1 = min(int(mu2[m, 1] + radius_px[m]) + 1, height)
        if u0 >= u1 or v0 >= v1:
            continue
        du = np.arange(u0, u1, dtype=float) - mu2[m, 0]
        dv = np.arange(v0, v1, dtype=float) - mu2[m, 1]
        expo = (inv00[m] * du[None, :] ** 2
                + 2.0 * inv01[m] * dv[:, None] * du[None, :]
                + inv11[m] * dv[:, None] ** 2)
        # (8.9)：α = o·exp(−½ qᵀΣ'⁻¹q)。
        alpha = opacities[m] * np.exp(-0.5 * expo)
        contrib = alpha * transmittance[v0:v1, u0:u1]   # 该基元对 bbox 内像素的权重
        image[v0:v1, u0:u1] += contrib[..., None] * colors[m]
        transmittance[v0:v1, u0:u1] *= 1.0 - alpha      # (8.10) 的透射率递推
        if m % 64 == 63 and transmittance.min() < T_MIN:
            break
    return {"image": image, "transmittance": transmittance,
            "alpha_total": 1.0 - transmittance}

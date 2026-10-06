"""Poisson 表面重建的教学代理（poisson）—— 教程第 05 章 5.3 节的教学实现。

函数流水线
----------
本模块承接带法向的表面点集（来自 ``scene.sample_scene_surface`` 的解析采样，
或任意外点法向点云），产出隐式场并交给 ``marching``（marching.py）提取网格：

    rasterize_normal_field （外向法向用高斯核涂抹成平滑向量场 V，
                             原文 §3 式 (2)；场在表面处归一到单位模长）
        -> divergence          （V 的中心差分离散散度 = Poisson 方程右端，(5.6)）
        -> solve_poisson_fft   （FFT 频域对角化拉普拉斯，解 ∇²χ̃ = ∇·V，(5.6)/(5.7)）
        -> implicit_from_points（包围盒 + 求解 + 归一到 (5.3) 指示函数尺度
                                 + 样本处等值面水平 γ（原文 §4.4））
        -> reconstruct_mesh    （γ − χ 交给 marching_tetrahedra 提取（5.1)/(5.2)，
                                 外正内负约定 (1.13)，即 (5.8) 的 0.5 等值面）

依赖方向：依赖 ``marching``（网格提取，输入场/原点/体素的数据约定见其
docstring）；``tests/test_poisson.py`` 直接驱动（评测指标从 ``metrics`` 按
路径加载器取用，见该文件头部说明——仓库多项目同名 metrics.py，禁止裸 import）。

输入/输出：输入点集 (M, 3)（单位 m）与外向法向 (M, 3)（单位向量）；输出
隐式场 χ (n, n, n)（无量纲，样本定义在 origin + (i,j,k)·voxel 的格点上，
数组布局 (x, y, z)，交给 marching 前转置为其 (z, y, x) 约定）、等值面水平
γ（标量）与 :class:`marching.TriangleMesh`。

教学简化（与原文的差异，README 同款声明）
------------------------------------------
Kazhdan, Bolitho & Hoppe, *"Poisson Surface Reconstruction"*, SGP 2006 用
**自适应八叉树**（表面附近加密、别处粗网格，原文 §4.1）+ **多级求解器**
（multigrid 式由粗到细精化，原文 §4.3）在稀疏线性系统上解 (5.7)；本教学实现
改为**均匀网格 + FFT 周边界的频域求解**——拉普拉斯在周期边界下于 Fourier
域对角化（特征值 Σ_axis (2cos(2πm/N) − 2)/h²），一次 FFT 除法即得解，代价
是：分辨率均匀（内存 O(n³)）、边界按周期处理（依赖"点集离边界足够远、V 在
边界处衰减为 0"的 padding 约定）。两点原文行为的对应：等值面水平按原文
§4.4 取"样本点处 χ 的均值"γ；指示函数的 0/1 尺度由"解的两平台（内/外）
归一"给出（原文通过补丁面积权重 a_i 达到同一效果）。法向取向约定与仓库
一致取**外向**（原文用内法向，二者只差 (5.4) 的一个符号：∇χ = −n δ_S；
外向法向解出的场外高内低，取反后即 (5.3) 的内 1 外 0 指示函数）。
"""
from __future__ import annotations

import numpy as np

from marching import TriangleMesh, marching_tetrahedra

__all__ = [
    "divergence",
    "implicit_from_points",
    "rasterize_normal_field",
    "reconstruct_mesh",
    "solve_poisson_fft",
]

# 法向涂抹核的截断半径（σ 的倍数）：高斯在 3σ 外 < 1%，再远的质量可忽略。
KERNEL_RADIUS_SIGMAS = 3.0
# 指示函数尺度归一的分位数（内/外两平台的稳健估计，避开边界与离群格点）。
_PLATFORM_LO, _PLATFORM_HI = 0.05, 0.95


def rasterize_normal_field(
    points: np.ndarray,
    normals: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    resolution: int,
    sigma_voxels: float = 1.0,
) -> np.ndarray:
    """把带法向点集涂抹成平滑向量场 V（教程 (5.5) 上文的构造；原文 §3 式 (2)）。

    每个样本的外向法向 n_i 乘高斯核 K_σ(q − p_i) 累加到周围格点（截断半径
    3σ；核宽取"窄到不抹平数据、宽到盖住采样间隔"的折中，原文 §3 讨论，
    故 sigma_voxels 应与采样间距同量级）；之后把场在样本处归一到单位模长
    （场幅值只决定解的整体尺度，等值面提取对该尺度不变）。按 (5.4) 的符号
    约定（∇χ = −n δ_S，χ 内 1 外 0），此处 V ≈ +n̂ δ_S 的平滑延拓 = −∇χ，
    解出的 χ̃ 外高内低，取反后即 (5.3) 指示函数（见 :func:`implicit_from_points`）。

    Args:
        points: (M, 3) 表面采样点，单位 m。
        normals: (M, 3) 外向法向（内部按模长归一），单位向量。
        origin: (3,) 格点 (0, 0, 0) 的世界坐标（场样本位于 origin + 下标·voxel）。
        voxel_size: 格点间距，单位 m，取值 > 0。
        resolution: 每轴格点数 n，取值 >= 2。
        sigma_voxels: 核宽（以体素为单位），取值 > 0。

    Returns:
        vec_field: (n, n, n, 3) 向量场，数组轴序 (x, y, z)、末轴为 (x, y, z)
                   分量，无量纲；表面处单位模长、远离表面处衰减为 0。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    nrms = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(pts) != len(nrms):
        raise ValueError(f"点数 {len(pts)} 与法向数 {len(nrms)} 不一致")
    if len(pts) == 0:
        raise ValueError("空点集无法涂抹向量场")
    norms = np.linalg.norm(nrms, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("存在零法向")
    nrms = nrms / norms
    n = int(resolution)
    h = float(voxel_size)
    sigma = sigma_voxels * h

    # 每点的基准格点下标（最近格点），核窗口 [-r, r]^3 逐偏移累加。
    # 数组布局 (x, y, z)：展平下标 = (ix·n + iy)·n + iz（x 最慢、z 最快）。
    base = np.rint((pts - np.asarray(origin, dtype=float)) / h).astype(np.int64)
    flat_base = (base[:, 0] * n + base[:, 1]) * n + base[:, 2]
    radius = int(np.ceil(KERNEL_RADIUS_SIGMAS * sigma_voxels))
    offsets = np.arange(-radius, radius + 1)
    comps = [np.zeros(n * n * n) for _ in range(3)]
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                step = np.array([dx, dy, dz])
                in_grid = np.all((base + step >= 0) & (base + step < n), axis=-1)
                if not in_grid.any():
                    continue
                # 核权重 = exp(-||p - node||^2 / 2σ^2)（未归一化高斯，整体尺度后置）。
                node_pos = np.asarray(origin) + (base[in_grid] + step) * h
                diff = pts[in_grid] - node_pos
                wgt = np.exp(-np.einsum("ij,ij->i", diff, diff) / (2.0 * sigma * sigma))
                flat = flat_base[in_grid] + (dx * n + dy) * n + dz
                for comp in range(3):
                    comps[comp] += np.bincount(
                        flat, weights=wgt * nrms[in_grid, comp], minlength=n * n * n)

    field = np.stack(comps, axis=-1).reshape(n, n, n, 3)
    # 表面处单位模长归一：样本插值位置的 |V| 均值缩放到 1
    # （等效原文式 (2) 的补丁面积权重 a_i，只改解的整体尺度、不改等值面）。
    mag = np.linalg.norm(_trilinear(field, origin, h, pts), axis=-1).mean()
    if mag <= 0:
        raise ValueError("涂抹后的向量场为零场：检查点集是否落在网格内")
    return field / mag


def divergence(vec_field: np.ndarray, voxel_size: float) -> np.ndarray:
    """向量场的离散散度 ∇·V（Poisson 方程右端，教程 (5.6)）。

    三轴中心差分（``np.gradient``，边界一阶单侧）后按 (x, y, z) 分量求和；
    V 在边界处已衰减为 0（涂抹核的局部性），故右端均值 ≈ 0，满足周期
    泊松方程的可解性条件（``solve_poisson_fft`` 的前提）。

    Args:
        vec_field: (n, n, n, 3) 向量场，轴序 (x, y, z)
                   （:func:`rasterize_normal_field` 的输出）。
        voxel_size: 格点间距，单位 m。

    Returns:
        rhs: (n, n, n) 散度场，无量纲（与 χ 同尺度）。
    """
    grad = [np.gradient(vec_field[..., k], float(voxel_size), axis=k)
            for k in range(3)]
    return grad[0] + grad[1] + grad[2]


def solve_poisson_fft(rhs: np.ndarray, voxel_size: float) -> np.ndarray:
    """FFT 频域求解周期边界泊松方程 ∇²χ̃ = rhs（教程 (5.6)/(5.7) 的教学代理）。

    周期网格上拉普拉斯的特征函数是平面波、特征值
    λ(m) = Σ_axis [2cos(2πm_axis/N) − 2]/h²（中心差分二阶导的频域形式），
    故解在 Fourier 域是逐频率的除法：χ̃̂ = rhŝ / λ——原文 (5.7) 的稀疏对称
    L 矩阵在周期边界下恰被 FFT 对角化。零频（λ = 0）对应"解差一个常数"：
    可解性要求 rhs 零均值，零频分量置 0（取均值零的代表解）。

    Args:
        rhs: (n, n, n) 方程右端（:func:`divergence` 的输出，须近似零均值）。
        voxel_size: 格点间距，单位 m。

    Returns:
        chi: (n, n, n) 实值隐式场，无量纲，均值 0。
    """
    rhs = np.asarray(rhs, dtype=float)
    if rhs.ndim != 3 or min(rhs.shape) < 2:
        raise ValueError(f"rhs 须为 (n, n, n)、n >= 2，当前 shape {rhs.shape}")
    h = float(voxel_size)
    # 每轴一维特征值（m = 0..size-1 与 fft 频率序一致）。
    axis_vals = [(2.0 * np.cos(2.0 * np.pi * np.arange(size) / size) - 2.0) / (h * h)
                 for size in rhs.shape]
    lam = (axis_vals[0][:, None, None] + axis_vals[1][None, :, None]
           + axis_vals[2][None, None, :])
    rhs_hat = np.fft.fftn(rhs)
    chi_hat = np.zeros_like(rhs_hat)
    nonzero = lam != 0.0
    chi_hat[nonzero] = rhs_hat[nonzero] / lam[nonzero]   # 零频保持 0（均值零代表）
    return np.fft.ifftn(chi_hat).real


def _trilinear(field: np.ndarray, origin: np.ndarray, voxel_size: float,
               points: np.ndarray) -> np.ndarray:
    """场（标量或末轴带向量分量）的三线性插值（教程 (4.13)/(1.15) 同款凸组合）。

    Args:
        field: (n, n, n) 或 (n, n, n, 3)，轴序 (x, y, z)。
        origin: (3,) 场原点，单位 m。
        voxel_size: 格点间距，单位 m。
        points: (M, 3) 查询点，单位 m（界外按边界截断）。

    Returns:
        (M,) 或 (M, 3) 插值结果。
    """
    n = field.shape[0]
    g = (np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)) / voxel_size
    gc = np.clip(g, 0.0, n - 1.0)
    i0 = np.minimum(np.floor(gc).astype(int), n - 2)
    frac = gc - i0
    scalar = field.ndim == 3
    out = np.zeros((len(gc),) if scalar else (len(gc), 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = i0 + np.array([dx, dy, dz])
                wgt = ((frac[:, 0] if dx else 1.0 - frac[:, 0])
                       * (frac[:, 1] if dy else 1.0 - frac[:, 1])
                       * (frac[:, 2] if dz else 1.0 - frac[:, 2]))
                val = field[idx[:, 0], idx[:, 1], idx[:, 2]]
                out += wgt[:, None] * val if not scalar else wgt * val
    return out


def implicit_from_points(
    points: np.ndarray,
    normals: np.ndarray,
    resolution: int = 32,
    pad_fraction: float = 0.25,
    sigma_voxels: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """从带法向点集解出隐式场 χ 与等值面水平 γ（(5.3)→(5.8) 的完整链路）。

    网格自动包围点集（pad_fraction 的外边距保证涂抹核在边界处衰减、周期
    假设成立）。外向法向解出的原始场 χ̃ 外高内低，先归一到 (5.3) 指示函数
    尺度：χ = (q_hi − χ̃)/(q_hi − q_lo)，q_lo/q_hi 取场值的 5%/95% 分位数
    （内/外两平台的稳健估计）——归一后内部 ≈ 1、外部 ≈ 0（原文以补丁面积
    权重 a_i 达成同一效果）。等值面水平取原文 §4.4 的"样本点处 χ 的均值"
    γ（≈ 0.5，(5.8) 的跃迁中点）。

    Args:
        points: (M, 3) 表面采样点，单位 m。
        normals: (M, 3) 外向法向。
        resolution: 每轴格点数 n。
        pad_fraction: 包围盒外边距 = 该比例 × 点集最大边长（无量纲）。
        sigma_voxels: 涂抹核宽（体素数）。

    Returns:
        (chi, origin, voxel_size, isolevel) 四元组：chi (n, n, n) 无量纲、
        内 ≈ 1 外 ≈ 0、轴序 (x, y, z)；origin (3,) 单位 m；voxel_size 单位 m；
        isolevel 标量（无量纲）。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise ValueError("点集退化（最大边长为 0）")
    pad = pad_fraction * extent
    origin = lo - pad
    voxel = (extent + 2.0 * pad) / resolution
    vec_field = rasterize_normal_field(pts, normals, origin, voxel, resolution,
                                       sigma_voxels=sigma_voxels)
    raw = solve_poisson_fft(divergence(vec_field, voxel), voxel)
    # 归一到 (5.3) 指示函数尺度：外高内低 -> 内 1 外 0（两平台分位数归一）。
    q_lo, q_hi = np.quantile(raw, [_PLATFORM_LO, _PLATFORM_HI])
    if q_hi - q_lo <= 0:
        raise ValueError("解退化为常数场：检查法向是否取向一致")
    chi = (q_hi - raw) / (q_hi - q_lo)
    # 原文 §4.4：等值面水平 = 样本点处 χ 的均值（(5.8) 的跃迁中点）。
    isolevel = float(_trilinear(chi, origin, voxel, pts).mean())
    return chi, origin, voxel, isolevel


def reconstruct_mesh(
    points: np.ndarray,
    normals: np.ndarray,
    resolution: int = 32,
    pad_fraction: float = 0.25,
    sigma_voxels: float = 1.0,
) -> TriangleMesh:
    """重建三角网格：γ − χ 交给 :func:`marching.marching_tetrahedra` 提取。

    g = γ − χ 把指示函数（内 1 外 0）翻成 marching 要求的**外正内负**
    （(1.13)）：{χ = γ} = {g = 0}，即 (5.8) 的 0.5 等值面；顶点法向 =
    ∇g/‖∇g‖ = −∇χ 方向朝外，与 (5.2)/(1.3) 的水平集几何一致。数组布局：
    本模块场轴序为 (x, y, z)，marching 的采样约定是 (z, y, x)（其 docstring
    的轴序注记），故先转置再提取。

    Args:
        points: (M, 3) 表面采样点，单位 m。
        normals: (M, 3) 外向法向。
        resolution: 每轴格点数 n。
        pad_fraction: 包围盒外边距比例。
        sigma_voxels: 涂抹核宽（体素数）。

    Returns:
        mesh: :class:`marching.TriangleMesh`（顶点/三角形/外法向）。
    """
    chi, origin, voxel, isolevel = implicit_from_points(
        points, normals, resolution=resolution, pad_fraction=pad_fraction,
        sigma_voxels=sigma_voxels)
    field_zyx = np.ascontiguousarray(np.transpose(isolevel - chi, (2, 1, 0)))
    return marching_tetrahedra(field_zyx, origin, voxel)

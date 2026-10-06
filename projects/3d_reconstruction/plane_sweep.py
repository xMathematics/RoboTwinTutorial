"""平面扫描立体的最简教学代理（plane_sweep）—— 教程第 03 章 3.1/3.2 节的教学实现。

函数流水线
----------
本模块承接 ``scene`` 的程序化纹理 RGB 渲染（``scene.render_rgb`` +
``scene.checker_texture``，同一物理表面点跨视角同色），为参考视角逐像素
恢复深度，产出可进 ``tsdf.TSDFVolume`` 融合的深度图：

    relative_pose      （T_wc 相乘消去世界系 -> X' = R_rel X + t_rel，教程 3.1 ⑤）
        -> plane_homography   （深度 d 的前向平行平面上的像素映射
                                H(d) = K(R_rel − t_rel nᵀ/d)K⁻¹，教程 (3.1)-(3.2)）
        -> warp_to_ref        （邻图按 H(d) 双线性重采样到参考像素）
        -> run_plane_sweep    （逐深度 NCC 光度一致性 (3.4) -> 沿深度轴取优 ->
                                深度图 + 置信度（最优与次优代价差））
        -> 深度图去向：tsdf.depth_to_point_cloud / TSDFVolume.integrate
                       （教程 (3.12) 反投影 + 第 04 章 (4.8) 融合）

依赖方向：依赖 ``scene``（渲染 RGB 与深度图的口径定义在该模块）；
``tsdf`` 只在测试与文档示例中衔接（融合），本模块本身不 import。
``tests/test_plane_sweep.py`` 直接驱动（评测指标从 ``metrics`` 按路径
加载器取用——仓库多项目同名 metrics.py，禁止裸 import）。

输入/输出：输入为 RGB 图 (H, W, 3)（float [0,1]，无效像素黑色）、内参
K (3, 3)（px）、外参 T_wc (4, 4)（相机 -> 世界）；输出深度图 (H, W)（单位 m，
像素级深度假设）与置信度 (H, W)（无量纲 >= 0，二名代价差）。

教学简化（与原系统/MVSNet 的差异，README 同款声明）
----------------------------------------------------
COLMAP-MVS（Schönberger & Fischer, ECCV 2016）在平面扫描之上还有 patch
匹配的 (深度, 法向) 联合估计、几何一致性的像素级视图选择与深度图融合；
MVSNet（ECCV 2018）把同一几何（(3.2) 的单应 warp）搬进可微代价体，加
3D CNN 正则与软 argmin。本教学代理只保留**共享的几何内核**：前向平行
深度假设 + 单应 warp + NCC 光度一致性 + 逐像素取优——法向固定为主光轴
（前向平行假设，倾斜面有模型误差，教程 3.1 末），视图选择退化为"全部
邻图等权平均"，置信度用代价体积的最优/次优差近似 MVSNet 的 P(d) 峰度。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "plane_homography",
    "relative_pose",
    "run_plane_sweep",
    "warp_to_ref",
]

# NCC 的标准差下限（无量纲）：窗口近常数（弱纹理）时 NCC 无判别力，
# 直接给最差代价（教程 3.2 ③："弱纹理与反射面处无判别力"）。
NCC_STD_EPS = 1e-6
# 采样越界（源图中不存在）的窗口按无效处理，代价取最差值 1。
INVALID_COST = 1.0


def relative_pose(
    t_wc_ref: np.ndarray,
    t_wc_src: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """参考相机系 -> 源相机系的相对位姿（X' = R_rel X + t_rel，教程 3.1 ⑤）。

    由两条 T_wc 相乘消去世界系：X_world = R_wc_ref X + t_wc_ref，
    X_src = R_wc_srcᵀ(X_world − t_wc_src)，合并即得。

    Args:
        t_wc_ref: (4, 4) 参考相机 -> 世界位姿。
        t_wc_src: (4, 4) 源相机 -> 世界位姿。

    Returns:
        (r_rel, t_rel)：r_rel (3, 3)、t_rel (3,)，把参考系坐标映到源系。
    """
    t_wc_ref = np.asarray(t_wc_ref, dtype=float)
    t_wc_src = np.asarray(t_wc_src, dtype=float)
    r_cw_src = t_wc_src[:3, :3].T
    r_rel = r_cw_src @ t_wc_ref[:3, :3]
    t_rel = r_cw_src @ (t_wc_ref[:3, 3] - t_wc_src[:3, 3])
    return r_rel, t_rel


def plane_homography(
    k_ref: np.ndarray,
    k_src: np.ndarray,
    r_rel: np.ndarray,
    t_rel: np.ndarray,
    depth: float,
) -> np.ndarray:
    """深度假设 d 的前向平行平面的单应矩阵（教程 (3.1)-(3.2)）。

    平面取 n = −e₃（法向指向参考相机，教程 3.1 的记号约定），点法式
    nᵀX + d = 0 即 z = d；代入 (3.1) 的消元式：

        H(d) = K_src (R_rel − t_rel nᵀ / d) K_ref⁻¹，  û' ~ H(d) û，

    平面上的点经 H(d) 的映射与真实刚体投影逐点一致（教程 3.1 第五步的
    特例校验），倾斜面上的窗口则有前向平行模型误差（教程 3.1 末）。

    Args:
        k_ref, k_src: (3, 3) 参考/源相机内参，单位 px。
        r_rel, t_rel: :func:`relative_pose` 的输出。
        depth: 深度假设（平面沿参考相机光轴的深度），单位 m，取值 > 0。

    Returns:
        H: (3, 3) 单应矩阵（齐次像素坐标：û_src ~ H @ û_ref）。
    """
    k_ref = np.asarray(k_ref, dtype=float)
    k_src = np.asarray(k_src, dtype=float)
    n = np.array([0.0, 0.0, -1.0])               # 前向平行平面的单位法向（教程 3.1）
    return k_src @ (r_rel - np.outer(t_rel, n) / float(depth)) @ np.linalg.inv(k_ref)


def warp_to_ref(
    src_image: np.ndarray,
    homography: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """源图按单应 H 重采样到参考像素网格（教程 (3.2) 的 û' ~ H û + 双线性）。

    Args:
        src_image: (H_s, W_s, 3) 源图，float。
        homography: (3, 3) 单应矩阵（:func:`plane_homography` 的输出）。
        width, height: 参考图宽/高，单位 px。

    Returns:
        (warped, valid)：warped (height, width, 3) 重采样图（越界像素为 0）；
        valid (height, width) 布尔掩码（像素中心落在源图内为 True）。
    """
    src = np.asarray(src_image, dtype=float)
    src_h, src_w = src.shape[:2]
    us = np.arange(width, dtype=float)
    vs = np.arange(height, dtype=float)
    uu, vv = np.meshgrid(us, vs)                     # (H, W) 参考像素坐标
    homo = np.asarray(homography, dtype=float)
    denom = homo[2, 0] * uu + homo[2, 1] * vv + homo[2, 2]
    su = (homo[0, 0] * uu + homo[0, 1] * vv + homo[0, 2]) / denom
    sv = (homo[1, 0] * uu + homo[1, 1] * vv + homo[1, 2]) / denom
    in_bounds = (su >= 0.0) & (su <= src_w - 1.0) & (sv >= 0.0) & (sv <= src_h - 1.0)
    # 双线性采样：4 邻像素按小数部分加权（越界掩码在加权前记录，防污染邻域）。
    u0 = np.floor(su).astype(int)
    v0 = np.floor(sv).astype(int)
    u1 = np.minimum(u0 + 1, src_w - 1)
    v1 = np.minimum(v0 + 1, src_h - 1)
    u0c, v0c = np.clip(u0, 0, src_w - 1), np.clip(v0, 0, src_h - 1)
    wu = np.clip(su - u0c, 0.0, 1.0)[..., None]
    wv = np.clip(sv - v0c, 0.0, 1.0)[..., None]
    top = (1.0 - wu) * src[v0c, u0c] + wu * src[v0c, u1]
    bot = (1.0 - wu) * src[v1, u0c] + wu * src[v1, u1]
    warped = (1.0 - wv) * top + wv * bot
    return warped * in_bounds[..., None], in_bounds


def _window_ncc(
    ref_gray: np.ndarray,
    warped_gray: np.ndarray,
    valid: np.ndarray,
    window: int,
) -> np.ndarray:
    """逐像素窗口 NCC -> 光度代价 C = 1 − NCC（教程 (3.4)，灰度窗口版）。

    两个窗口各自减均值、除标准差后做内积（(3.4) 的离散形式）；对仿射光照
    变化 I' = aI + b 不变（(3.5)）。窗口不完整（含越界采样）或任一侧近常数
    （std < eps，弱纹理）时无判别力，代价取最差值 1。

    Args:
        ref_gray: (H, W) 参考图灰度。
        warped_gray: (H, W) 按 H(d) 重采样后的源图灰度。
        valid: (H, W) 采样有效掩码。
        window: 窗口边长（奇数），单位 px。

    Returns:
        cost: (H, W) 光度代价，无量纲，[0, 1]（1 = 无判别力/无效）。
    """
    height, width = ref_gray.shape
    cost = np.full((height, width), INVALID_COST)
    half = window // 2
    win = (window, window)
    ref_w = np.lib.stride_tricks.sliding_window_view(ref_gray, win)
    warp_w = np.lib.stride_tricks.sliding_window_view(warped_gray, win)
    # 窗口完整 = 中心窗口内全部采样有效（掩码窗取 min = 逻辑与）。
    valid_w = np.lib.stride_tricks.sliding_window_view(valid.astype(float), win)
    full = valid_w.min(axis=(-2, -1)) > 0.5
    ref_c = ref_w - ref_w.mean(axis=(-2, -1), keepdims=True)
    warp_c = warp_w - warp_w.mean(axis=(-2, -1), keepdims=True)
    ref_std = np.sqrt(np.einsum("ijkl,ijkl->ij", ref_c, ref_c))
    warp_std = np.sqrt(np.einsum("ijkl,ijkl->ij", warp_c, warp_c))
    ok = full & (ref_std > NCC_STD_EPS) & (warp_std > NCC_STD_EPS)
    ncc = np.einsum("ijkl,ijkl->ij", ref_c, warp_c) / np.maximum(ref_std * warp_std, NCC_STD_EPS)
    interior = np.where(ok, 1.0 - np.clip(ncc, -1.0, 1.0), INVALID_COST)
    cost[half:height - half, half:width - half] = interior
    return cost


def run_plane_sweep(
    ref_image: np.ndarray,
    src_images: list[np.ndarray],
    k_ref: np.ndarray,
    t_wc_ref: np.ndarray,
    t_wc_srcs: list[np.ndarray],
    depth_min: float,
    depth_max: float,
    n_depths: int = 48,
    window: int = 7,
) -> dict:
    """对参考图逐像素扫描深度假设：NCC 代价 -> 取优 -> 深度图 + 置信度。

    每个深度假设 d：邻图按 (3.2) 的 H(d) warp 到参考视角，逐像素算 NCC
    代价 (3.4) 并跨邻图取平均（全部视图等权——教学简化，COLMAP 的视图
    选择见教程 3.2 末）；沿深度轴取 argmin 得逐像素深度（MVSNet 的硬
    argmin 版本），置信度取次优与最优代价之差——真实深度处代价曲线单峰
    陡峭（二名差大），遮挡/弱纹理处平坦或并列（二名差趋 0），即 MVSNet
    概率体 P(d) 峰度的手工版（教程 3.3 ⑤(b) 的极限行为）。

    Args:
        ref_image: (H, W, 3) 参考图。
        src_images: 邻图列表，每个 (H, W, 3)。
        k_ref: (3, 3) 参考相机内参，px（邻图共用同一内参——
               ``scene.render_dataset`` 的单 K 口径）。
        t_wc_ref: (4, 4) 参考相机位姿。
        t_wc_srcs: 邻图位姿列表，各 (4, 4)。
        depth_min, depth_max: 深度假设区间，单位 m（要求 min < max）。
        n_depths: 假设数（区间上均匀采样）。
        window: NCC 窗口边长（奇数），px。

    Returns:
        dict：
        - ``"depth"``: (H, W) 最优假设深度图，单位 m；
        - ``"confidence"``: (H, W) 置信度 = 次优代价 − 最优代价，无量纲 >= 0；
        - ``"cost_volume"``: (n_depths, H, W) 聚合后的代价体积（诊断用）。
    """
    ref = np.asarray(ref_image, dtype=float)
    height, width = ref.shape[:2]
    if window % 2 == 0 or window < 3:
        raise ValueError(f"window 须为 >= 3 的奇数，当前 {window}")
    if not (0.0 < depth_min < depth_max):
        raise ValueError(f"要求 0 < depth_min < depth_max，当前 ({depth_min}, {depth_max})")
    depths = np.linspace(depth_min, depth_max, int(n_depths))
    ref_gray = ref.mean(axis=-1)

    cost_volume = np.zeros((len(depths), height, width))
    for i, (src, t_wc_src) in enumerate(zip(src_images, t_wc_srcs)):
        r_rel, t_rel = relative_pose(t_wc_ref, t_wc_src)
        src_gray = np.asarray(src, dtype=float).mean(axis=-1)
        for d_idx, depth in enumerate(depths):
            hom = plane_homography(k_ref, np.asarray(k_ref, dtype=float),
                                   r_rel, t_rel, float(depth))
            warped, valid = warp_to_ref(src, hom, width, height)
            per_view = _window_ncc(ref_gray, warped.mean(axis=-1), valid, window)
            if i == 0:
                cost_volume[d_idx] = per_view
            else:
                cost_volume[d_idx] += per_view
    cost_volume /= max(len(src_images), 1)           # 跨邻图平均（等权）

    best = np.argmin(cost_volume, axis=0)            # (H, W) 最优深度下标
    depth_map = depths[best]
    ordered = np.sort(cost_volume, axis=0)           # 沿深度轴排序取前二名
    confidence = ordered[1] - ordered[0]             # 二名代价差（>= 0，并列趋 0）
    return {"depth": depth_map, "confidence": confidence, "cost_volume": cost_volume}

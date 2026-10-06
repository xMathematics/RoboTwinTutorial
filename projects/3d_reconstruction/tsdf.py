"""TSDF 体素网格与多视角深度融合（tsdf）—— 教程第 04 章的教学实现。

函数流水线
----------
本模块承接 ``scene``（scene.py）产出的多视角深度数据，产出可供
``marching``（marching.py）提取网格的截断符号距离场：

    TSDFVolume(bbox, resolution) （世界包围盒 + 均匀体素网格，F=+1/W=0 初始化）
        -> integrate(depth, K, T_wc) （逐视角融合：投影体素中心 -> (4.1) 离散观测
                                        s ≈ z(u) - z_x -> (4.3) 截断 -> (4.8) 递归加权平均）
        -> sample_tsdf / signed_distance （任意点查询：三线性插值 (4.13)/(1.15) + 零点）
        -> depth_to_point_cloud （深度反投影点云，单视角基线/输入检查工具）

依赖方向：依赖 ``scene``（仅类型层面：深度图/内参/外参的口径定义在该模块）；
``marching`` 与 ``demo`` 消费本模块的 ``tsdf`` 数组；``tests/test_tsdf.py`` 直接驱动。

输入/输出：深度图 (H, W)（单位 m，无回波 = +inf）、内参 K (3, 3)（px）、
外参 T_wc (4, 4)（相机 -> 世界）；体素场 tsdf/weight 均为 (n, n, n)
（tsdf 无量纲、被 mu 归一到 [-1, 1]；weight 单位 1/m^2，取值 >= 0）。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "TSDFVolume",
    "depth_to_point_cloud",
]

# 相机系最小深度（m）：更近的体素投影被视作数值伪影丢弃（避免 z -> 0 除零）。
Z_NEAR = 1e-3


class TSDFVolume:
    """截断符号距离场（TSDF）体素网格（Curless & Levoy 1996；教程 (1.14)/(4.3)）。

    世界空间被剖分成 n x n x n 的均匀立方体素网格；每个体素存两个标量：

    - ``tsdf``：截断符号距离 F = clip(s / mu, -1, 1)，无量纲，[-1, 1]，
      零水平集即表面（(4.3)；外正内负约定同 (1.13)）；
    - ``weight``：累计权重 W = sum_k w_k，单位 1/m^2（w = 1/z^2，(4.2)），W=0 表示未见。

    融合按 (4.8) 的递归加权平均（Curless–Levoy 运行平均、KinectFusion 融合式）：
    F <- (W·F + w·clip(s/mu)) / (W + w)，W <- W + w，仅更新 |s| <= mu 的带内体素。

    Attributes:
        resolution: 每轴体素数 n（int）。
        voxel_size: 体素边长，单位 m（要求包围盒为立方）。
        origin: (3,) 网格原点 = bbox 最小角，单位 m（体素 (0,0,0) 的角点）。
        tsdf: (n, n, n) 截断符号距离场，无量纲，初始全 +1（外部）。
        weight: (n, n, n) 累计权重，单位 1/m^2，初始全 0（未知）。
        mu: 截断带宽，单位 m（默认 3 个体素；教程 4.1："mu 取几个体素尺度"）。
    """

    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        resolution: int,
        mu: float | None = None,
    ) -> None:
        """建立体素网格并把场初始化为"未知 = 外部"。

        Args:
            bbox_min: (3,) 世界包围盒最小角，单位 m。
            bbox_max: (3,) 世界包围盒最大角，单位 m（须为立方，边长一致）。
            resolution: 每轴体素数 n，取值 >= 2。
            mu: 截断带宽，单位 m；None 时取 3 * voxel_size。
        """
        lo = np.asarray(bbox_min, dtype=float)
        hi = np.asarray(bbox_max, dtype=float)
        sides = hi - lo
        if not np.allclose(sides, sides[0]):
            raise ValueError(f"包围盒须为立方（各边相等），当前边长 {sides}")
        if resolution < 2:
            raise ValueError(f"resolution 必须 >= 2，当前为 {resolution}")
        self.resolution = int(resolution)
        self.voxel_size = float(sides[0] / resolution)
        self.origin = lo
        # 教程 4.1：mu 取几个体素尺度的量级（与传感器噪声水平、体素分辨率匹配）。
        self.mu = float(mu) if mu is not None else 3.0 * self.voxel_size
        self.tsdf = np.ones((self.resolution,) * 3, dtype=float)   # (4.3) 带外饱和值 +1
        self.weight = np.zeros((self.resolution,) * 3, dtype=float)

    # -------------------------------------------------- 几何辅助 ----------
    def voxel_centers(self) -> np.ndarray:
        """全部体素中心的世界坐标。

        Returns:
            centers: (n^3, 3) 体素中心，单位 m（行主序：x 最快、z 最慢）。
        """
        n = self.resolution
        axis = (np.arange(n) + 0.5) * self.voxel_size   # (n,) 体素中心的单轴偏移
        zz, yy, xx = np.meshgrid(axis + self.origin[2],
                                 axis + self.origin[1],
                                 axis + self.origin[0], indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # -------------------------------------------------- 融合 --------------
    def integrate(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        mu: float | None = None,
    ) -> int:
        """融合一帧深度图（教程 (4.8) 递归加权平均；Curless & Levoy 1996 §4）。

        每个体素中心经位姿投到像素（依据：针孔投影 u = fx·X/Z + cx，SLAM
        教程 (5.10)），沿视线取 (4.1) 的离散观测 s ≈ z(u) - z_x；仅 |s| <= mu
        的带内体素参与更新（(4.3) 截断 + 教程 4.2 第六步"带外不写入"），
        权重取逆方差 w = 1/z(u)^2（(4.2)），更新式为 (4.8)：
        F <- (W·F + w·clip(s/mu, -1, 1)) / (W + w)，W <- W + w。

        Args:
            depth: (H, W) 深度图，单位 m，无回波 = +inf。
            K: (3, 3) 内参矩阵，单位 px。
            T_wc: (4, 4) 该帧相机 -> 世界位姿，单位 m / rad。
            mu: 本帧截断带宽，单位 m；None 用 ``self.mu``。

        Returns:
            updated: 本次被更新的体素数（int，诊断量；0 = 该帧无有效信息）。
        """
        mu = self.mu if mu is None else float(mu)
        depth = np.asarray(depth, dtype=float)
        K = np.asarray(K, dtype=float)
        T_wc = np.asarray(T_wc, dtype=float)
        T_cw = np.linalg.inv(T_wc)                      # 世界 -> 相机
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        height, width = depth.shape

        centers = self.voxel_centers()                  # (V, 3) 世界系体素中心
        cam = centers @ R_cw.T + t_cw                   # (V, 3) 相机系
        z = cam[:, 2]
        # 针孔投影 + 最近像素取整（依据：深度图是像素离散采样，最近邻对齐）。
        ui = np.rint(fx * cam[:, 0] / z + cx).astype(int)
        vi = np.rint(fy * cam[:, 1] / z + cy).astype(int)
        in_img = (z > Z_NEAR) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        ui_c = np.clip(ui, 0, width - 1)
        vi_c = np.clip(vi, 0, height - 1)
        zpix = depth[vi_c, ui_c]                        # (V,) 该像素的观测深度 z(u)
        valid = in_img & np.isfinite(zpix) & (zpix > 0.0)

        # (4.1) 离散观测：s ≈ z(u) - z_x（视线内深度差近似符号距离）。
        s = zpix - z
        band = valid & (np.abs(s) <= mu)                # (4.3)/教程 4.2：带外不写入
        if not band.any():
            return 0

        idx = np.nonzero(band)[0]
        # centers 按行主序 (k, j, i) 展平，与 self.tsdf.ravel() 的内存布局一致，
        # 因此体素中心的行号就是展平体素下标（免去重算网格下标）。
        flat = idx

        # (4.2)：w = 1/z^2（远距读数更不可信；z 取观测深度，恒 > 0）。
        w = 1.0 / (zpix[idx] ** 2)
        f_obs = np.clip(s[idx] / mu, -1.0, 1.0)         # (4.3) 截断归一化

        f_old = self.tsdf.ravel()[flat]
        w_old = self.weight.ravel()[flat]
        # (4.8) 递归加权平均（首帧 W=0 退化为直接写入）。
        w_new = w_old + w
        f_new = (w_old * f_old + w * f_obs) / w_new
        self.tsdf.ravel()[flat] = f_new
        self.weight.ravel()[flat] = w_new
        return int(len(idx))

    # -------------------------------------------------- 查询 --------------
    def sample_tsdf(self, points: np.ndarray) -> np.ndarray:
        """任意点的 TSDF 值：三线性插值（教程 (4.13)/(1.15)）。

        场在带内近似线性（教程 01.3 第 1 条），三线性插值是凸组合、
        跨体素连续。包围盒外的点按"外部"处理，返回 +1（(4.3) 饱和语义）。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M,) 插值后的 TSDF 值，无量纲，范围 [-1, 1]。
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        n = self.resolution
        # 场样本存于体素中心：连续网格坐标 g = (p - origin - 0.5·voxel) / voxel，
        # 整数 g 对应体素中心，有效范围 [0, n-1]（每轴）。
        g = (pts - self.origin - 0.5 * self.voxel_size) / self.voxel_size  # (M, 3)
        inside = np.all((g >= 0.0) & (g <= n - 1.0), axis=-1)
        # 边界点拉回 [0, n-1] 区间；g0 再截到 n-2 保证 idx1 = g0+1 不越界
        # （frac = 1.0 时权重整体落在 idx1 上，插值退化为中心值本身）。
        gc = np.clip(g, 0.0, n - 1.0)
        g0 = np.minimum(np.floor(gc).astype(int), n - 2)
        frac = gc - g0                                   # (M, 3) 每轴插值权重 alpha
        out = np.ones(len(pts))                         # 盒外默认 +1（外部）
        sel = np.nonzero(inside)[0]
        if len(sel):
            acc = np.zeros(len(sel))
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        corner = np.stack([
                            g0[sel, 0] + dx,
                            g0[sel, 1] + dy,
                            g0[sel, 2] + dz,
                        ], axis=-1)
                        wgt = ((frac[sel, 0] if dx else 1.0 - frac[sel, 0])
                               * (frac[sel, 1] if dy else 1.0 - frac[sel, 1])
                               * (frac[sel, 2] if dz else 1.0 - frac[sel, 2]))
                        acc += wgt * self.tsdf[corner[:, 2], corner[:, 1], corner[:, 0]]
            out[sel] = acc
        return out

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """任意点的截断符号距离（米）：带内 s = F·mu（(4.3) 反缩放），带外饱和 ±mu。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M,) 截断符号距离，单位 m，范围 [-mu, mu]；外正内负。
        """
        return self.sample_tsdf(points) * self.mu

    def extraction_field(self) -> np.ndarray:
        """供等值面提取的权重掩码场：未观测体素（W = 0）置 NaN。

        未知区以 F = +1、W = 0 初始化；若直接对原始场提取零等值面，
        "观测带内边界（F = -1）与未知区（+1）"的过渡会生成一层**幽灵内壁**
        （物体内部从未被任何视角观测）。提取前把 W = 0 的角点置 NaN，
        由 ``marching.marching_tetrahedra`` 跳过含 NaN 角点的单元——
        即"只对有观测支撑的体素提取"（KinectFusion 实践中的权重阈值提取）。

        Returns:
            (n, n, n) 场副本，无量纲；观测体素为 TSDF 值，未知体素为 NaN。
        """
        field = self.tsdf.copy()
        field[self.weight == 0.0] = np.nan
        return field


def depth_to_point_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
) -> np.ndarray:
    """单帧深度图反投影为世界系点云（针孔投影的逆，SLAM 教程 (5.10)）。

    Args:
        depth: (H, W) 深度图，单位 m，无回波 = +inf（无效像素被跳过）。
        K: (3, 3) 内参矩阵，单位 px。
        T_wc: (4, 4) 相机 -> 世界位姿，单位 m / rad。

    Returns:
        points: (M, 3) 世界系点云，单位 m，M = 深度有效的像素数；
                无有效像素时返回 (0, 3)。
    """
    depth = np.asarray(depth, dtype=float)
    K = np.asarray(K, dtype=float)
    T_wc = np.asarray(T_wc, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width, dtype=float),
                         np.arange(height, dtype=float))
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return np.zeros((0, 3))
    z = depth[valid]                                # (M,) 深度 = 相机系 z
    x = (us[valid] - cx) / fx * z                   # 反投影：X = (u-cx)/fx·Z
    y = (vs[valid] - cy) / fy * z
    cam = np.column_stack([x, y, z])                # (M, 3) 相机系
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
    return cam @ R_wc.T + t_wc                      # 相机 -> 世界

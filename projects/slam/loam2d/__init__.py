"""LOAM 思想的 2D 激光里程计与建图包（教程第 10 章系统观）.

Zhang & Singh, *LOAM: Lidar Odometry and Mapping in Real-time* (RSS 2014)
的教学实现，作为**显式声明的 2D 教学简化**把论文约简到平面 SE(2) 扫描仪：
Eq. (1) 基于 curvature（平滑度）的 edge/planar 特征提取、Eq. (2)-(3) 点到线
残差（解析 SE(2) 雅可比），经 :func:`core.solver.gauss_newton` 求解
（论文 Eq. (12) 的 LM），以及论文 §IV（Fig. 9）的 10Hz odometry / 1Hz
mapping 双速率（dual-rate）结构与 §VI 的协方差特征值直线对应（scan-to-map
配准）。

函数流水线（整体系统中的位置、输入/输出与依赖）
------------------------------------------------
1. :func:`make_room`：合成 2D 房间几何（:class:`Room`：线段墙/方柱 + 圆）；
2. :func:`render_scan`：从位姿射线求交渲染一帧极坐标扫描
   ``(ranges, angles)``——教学"数据集"来源；
3. :func:`smoothness` + :func:`extract_features`：对每帧扫描点算 Eq. (1)
   平滑度，按 §V-A 规则（阈值 + 非极大值抑制 + 配额）选出 edge/planar
   特征（:class:`FeatureSet`）；
4. :func:`point_line_residuals_jacobian`：Eq. (2) 点到线残差与解析雅可比；
5. :func:`register_scan_to_scan`（odometry，帧间）/ :func:`register_scan_to_map`
   （mapping，扫描-地图）：LM 配准，正规方程统一复用
   :mod:`core.solver` 的 :func:`core.solver.gauss_newton`——**本模块唯一的
   外部求解依赖**；
6. :func:`run_loam2d`：双速率主流程——逐帧 odometry + 每 ``mapping_every``
   帧 mapping 精化，输出 :class:`Loam2DOutput`（双速率轨迹与诊断）。

docstring 中全部式号指
papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf。
"""
from .loam2d import (
    FeatureSet,
    Loam2DOutput,
    RegistrationResult,
    Room,
    extract_features,
    make_room,
    point_line_residuals_jacobian,
    register_scan_to_map,
    register_scan_to_scan,
    render_scan,
    run_loam2d,
    scan_points,
    smoothness,
    wrap_angle,
)

__all__ = [
    "FeatureSet",
    "Loam2DOutput",
    "RegistrationResult",
    "Room",
    "extract_features",
    "make_room",
    "point_line_residuals_jacobian",
    "register_scan_to_map",
    "register_scan_to_scan",
    "render_scan",
    "run_loam2d",
    "scan_points",
    "smoothness",
    "wrap_angle",
]

"""对极几何与特征点法前端核心包（教程第 05 章）.

本包在系统中的位置: 特征提取与匹配之后的几何前端。流水线: 归一化八点法
(E/F) -> 本质矩阵分解 + 手性检验 -> DLT 三角化 -> DLT PnP + 流形
Gauss-Newton 精化；输入为两帧匹配像素 ``(u1, u2)`` 与内参 ``K``（PnP 支线
为 3D-2D 对应），输出 ``(E, F)``、唯一 ``(R, t)``、场景点与 ``T_cw``。
依赖 ``core.lie`` / ``core.solver`` / ``core.camera``；:func:`triangulate`
被 ``vins`` 复用做初始化路标三角化。式号 (5.x) 指向
``tutorials/slam/05_视觉里程计-i特征点法.md``，详见 ``epipolar.epipolar``
模块 docstring。ORB-SLAM（Mur-Artal et al., T-RO 2015）的单目初始化与回环
几何校验（第 09 章 (9.5)）即建立在这些机制之上。
"""
from .epipolar import (
    decompose_E,
    eight_point,
    essential_from_rt,
    fundamental_from_E,
    pnp_dlt,
    pnp_refine,
    reprojection_jacobian,
    triangulate,
)

__all__ = [
    "decompose_E",
    "eight_point",
    "essential_from_rt",
    "fundamental_from_E",
    "pnp_dlt",
    "pnp_refine",
    "reprojection_jacobian",
    "triangulate",
]

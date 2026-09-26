"""SLAM 教学实现库的共享基础层。

函数流水线
----------
本包是教程代码库的最底层：不依赖任何其他子模块，向上为各章算法模块提供
统一的数学与数值基础件。

- ``lie``    : SO(3)/SE(3) 指数/对数映射与左雅可比（教程第 02 章）。
- ``camera`` : 针孔相机投影/反投影（教程第 04 章）。
- ``solver`` : 带鲁棒核的 Gauss-Newton / Levenberg-Marquardt（教程第 08 章）。

输入/输出：纯内存计算——输入 numpy 数组（旋转矢量 (3,)、位姿 (4,4)、
点集 (N,3)、像素 (N,2) 等），输出对应形状的数组；无文件 I/O、无全局状态。

被谁调用（自底向上复用，实际 import 关系）：
- ``epipolar``  -> lie / camera / solver
- ``preint``    -> lie
- ``vins``      -> lie / solver
- ``direct``    -> lie / camera / solver
- ``droidlite`` -> lie / camera / solver
- ``photoba``   -> lie / camera / solver
- ``loam2d``    -> solver
（``fastslam`` 不依赖本包：2D 粒子滤波只需平面三角函数，自成一体。）

每个章节模块的 docstring 中式号均对应 ``tutorials/slam/`` 下的教程章节。
"""
from .camera import PinholeCamera, make_intrinsics
from .lie import (
    hat,
    se3_exp,
    se3_log,
    so3_exp,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
    so3_log,
    transform_points,
    vee,
)
from .solver import gauss_newton, huber_weights

__all__ = [
    "PinholeCamera",
    "make_intrinsics",
    "hat",
    "vee",
    "so3_exp",
    "so3_log",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse",
    "se3_exp",
    "se3_log",
    "transform_points",
    "gauss_newton",
    "huber_weights",
]

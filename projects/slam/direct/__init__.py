"""半稠密直接法（semi-dense direct method）包（教程第 06 章，LSD-SLAM 思想）。

函数流水线
----------
``make_textured_plane``（合成带纹理平面场景：I1、I2、真值位姿、参考帧深度图）
→ ``select_gradient_pixels``（梯度幅值前 ``top_frac`` 像素筛选）
→ ``photometric_residuals``（光度残差 photometric residual，教程 (6.4)）
→ ``semi_dense_align``（复用 :func:`core.solver.gauss_newton`，在左扰动
坐标上做 G-N 流形迭代，教程 (8.9)）。

对外暴露：合成纹理平面场景（``make_textured_plane`` / ``make_plane_scene``
/ :class:`PlaneScene`）、手写可微双线性采样器（:func:`bilinear_sample`）、
解析左扰动投影雅可比（:func:`projection_jacobian`，教程 (5.11)/(6.5)）与
半稠密光度对齐 :func:`semi_dense_align`。场景合成（``make_plane_scene``）
与雅可比（``projection_jacobian``）被 ``droidlite`` 复用。
"""
from .direct import (
    PlaneScene,
    bilinear_sample,
    make_plane_scene,
    make_textured_plane,
    photometric_jacobian,
    photometric_residuals,
    projection_jacobian,
    select_gradient_pixels,
    semi_dense_align,
    warp_and_sample,
)

__all__ = [
    "PlaneScene",
    "bilinear_sample",
    "make_plane_scene",
    "make_textured_plane",
    "photometric_jacobian",
    "photometric_residuals",
    "projection_jacobian",
    "select_gradient_pixels",
    "semi_dense_align",
    "warp_and_sample",
]

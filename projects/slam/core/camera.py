"""针孔相机投影（教程第 04 章，Eq. 4.1-4.5）。

函数流水线
----------
本模块提供 ``core`` 的几何投影基础件：``direct``、``droidlite``、``photoba``
复用 ``PinholeCamera`` 做重投影/反投影，``epipolar`` 复用 ``make_intrinsics``
构造内参；``fastslam`` 是 2D 距离-角度传感器，不用本模块。

投影模型：``s * [u, v, 1]^T = K [X, Y, Z]^T``，其中
``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``
（依据：相似三角形 + 齐次坐标，教程第 04 章 Eq. 4.1-4.5）。
输入为相机系点 (N, 3)（单位 m，深度 Z > 0），输出像素 (N, 2)（单位像素）；
反投影需额外给定深度 (N,)。
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["PinholeCamera", "make_intrinsics"]


@dataclass(frozen=True)
class PinholeCamera:
    """针孔相机内参，附投影 / 反投影辅助方法。

    Attributes:
        fx: x 方向焦距（单位像素），u 轴的尺度因子。
        fy: y 方向焦距（单位像素）。
        cx: 主点 u 坐标（单位像素），通常接近图像宽的一半。
        cy: 主点 v 坐标（单位像素）。
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """把相机系点 (N, 3) 投影为像素 (N, 2)。

        点必须有正深度 ``Z > 0``；深度为零会抛 ``ValueError``（深度正是
        投影这一步丢掉的信息，教程第 04 章 §4.1）。

        Args:
            points_cam: (N, 3) 相机系点 (X, Y, Z)，单位 m，要求 Z > 0。

        Returns:
            (N, 2) 像素坐标 (u, v)，单位像素。
        """
        pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        z = pts[:, 2]
        if np.any(z <= 0):
            raise ValueError("project() requires positive depth (Z > 0)")
        # 依据：Eq. 4.4-4.5，u = fx*X/Z + cx，v = fy*Y/Z + cy。
        u = self.fx * pts[:, 0] / z + self.cx
        v = self.fy * pts[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def unproject(self, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """已知深度的像素 (N, 2) 反投影回相机系 (N, 3)。

        Args:
            pixels: (N, 2) 像素坐标 (u, v)，单位像素。
            depth: (N,) 深度 Z，单位 m，须为正。

        Returns:
            (N, 3) 相机系点 (X, Y, Z)，单位 m。
        """
        uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
        d = np.asarray(depth, dtype=float).reshape(-1)
        # 依据：Eq. 4.4-4.5 的反解，X = (u - cx)*Z/fx，Y = (v - cy)*Z/fy。
        x = (uv[:, 0] - self.cx) * d / self.fx
        y = (uv[:, 1] - self.cy) * d / self.fy
        return np.stack([x, y, d], axis=1)

    def matrix(self) -> np.ndarray:
        """返回 (3, 3) 内参矩阵 ``K``。"""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> PinholeCamera:
    """构造 :class:`PinholeCamera`（位置参数的易读写法）。

    Args:
        fx, fy: 焦距（单位像素）。
        cx, cy: 主点（单位像素）。

    Returns:
        :class:`PinholeCamera` 实例。
    """
    return PinholeCamera(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

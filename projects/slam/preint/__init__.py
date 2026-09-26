"""IMU 预积分（preintegration）教学包（教程第 10 章 §10.2）。

函数流水线
----------
按 Forster, Carlone, Dellaert & Scaramuzza, *On-Manifold Preintegration for
Real-Time Visual-Inertial Odometry*, IEEE T-RO 2017
（papers/slam/classics/arXiv-1512.02363）实现的一阶预积分 IMU 测量：无重力
（gravity-free）增量 ΔR̃/Δṽ/Δp̃（论文 Eq.(37)）、其协方差递推（Eq.(46)-(47)）
与一阶偏置修正（Eq.(44)），对应教程第 10 章 §10.2 式 (10.1)-(10.8)。

输入/输出：原始 IMU 采样序列 ``(t, gyro, accel)`` 与偏置估计
``b̄ = [b̄^g; b̄^a]`` 进入 :class:`Preintegration`，输出三个预积分增量
（ΔR̃ 为 (3, 3) 旋转、Δṽ/Δp̃ 为 (3,) 矢量）、9x9 协方差 ``Σ_ij`` 与 9x6
偏置雅可比（:attr:`Preintegration.j_bias`）；:meth:`Preintegration.correct`
提供免重积分的一阶偏置修正。

依赖关系：依赖 ``core.lie`` 的左雅可比 :func:`core.lie.so3_left_jacobian`
构造 SO(3) 右雅可比（恒等式 ``J_r(φ) = J_l(−φ) = J_l(φ)^T``）；下游的
VINS 模块（``projects/slam/vins/``）把 :class:`Preintegration` 包装成紧耦合
因子图中的 IMU 因子。由 ``tests/test_preint.py`` 直接驱动。
"""
from .preint import ImuParams, Preintegration, so3_right_jacobian

__all__ = ["ImuParams", "Preintegration", "so3_right_jacobian"]

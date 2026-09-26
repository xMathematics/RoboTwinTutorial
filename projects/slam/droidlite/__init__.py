"""DROID-SLAM 递归稠密 BA 结构演示包（教程第 10 章 §10.4 前沿表）.

Teed & Deng, *DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D
Cameras* (NeurIPS 2021) 的教学实现，作为**结构演示（structural demo），
无任何学习组件**：论文的光流网络（Eq. (1) 相关体 + GRU 更新算子）被替换为
合成 GT 光流 + 高斯噪声（噪声幅度随迭代轮数按几何速率收缩，模拟 GRU 收敛），
而状态（SE(3) 位姿 + 逐像素逆深度）、Algorithm 1 的 [光流更新 <-> 稠密 BA]
递归交替、DBA 代价（Eq. (4)）及其 Schur complement（舒尔补）正规方程
（Eq. (5)，经 :func:`core.solver.gauss_newton` 稠密求解与等价交叉验证）
与论文一致。

函数流水线（整体系统中的位置、输入/输出与依赖）
------------------------------------------------
1. :func:`make_sequence`：合成 textured-plane 序列并生成 GT 位姿/逆深度/
   光流 + 固定底噪（:class:`SequenceData`）——结构演示的"数据集"；
2. :class:`DenseBA`（构造：绑定量测与初值）:meth:`DenseBA.solve` 跑
   Algorithm 1 的 R 轮递归，每轮依次：
   a. **Eq. (3)** 用当前 (位姿, 逆深度) 重新生成对应场/预测光流
      (:meth:`DenseBA.predicted_flow`)；
   b. **Eq. (4)** 光流修正：当轮测量（GT + ``sigma * decay^round`` 噪声）
      减预测得修正对应；
   c. **Eq. (5)** Schur 补 LM 增量（:meth:`DenseBA._schur_step`：深度块
      对角分块消元；解析雅可比复用 :func:`direct.projection_jacobian` 的
      左扰动投影雅可比；IRLS 权重复用 :func:`core.solver.huber_weights`；
      稠密 GN 回退路径 ``use_dense_gn=True`` 改由
      :func:`core.solver.gauss_newton` 解同一正规方程，tests 交叉验证两路
      数值一致）；
   d. **Eq. (2)** 回缩：位姿沿 SE(3) 指数回缩、逆深度向量加法更新。

依赖关系：``core.camera`` / ``core.lie`` / ``core.solver``（求解与鲁棒核）、
``direct``（:func:`direct.make_plane_scene` 场景合成、
:func:`direct.projection_jacobian` 投影雅可比）。

docstring 中全部式号指
papers/slam/frontier/arXiv-2108.10869_DROID-SLAM.pdf。
"""
from .droidlite import (
    DEFAULT_MOTIONS,
    DenseBA,
    DroidLiteResult,
    SequenceData,
    make_sequence,
)

__all__ = [
    "DEFAULT_MOTIONS",
    "DenseBA",
    "DroidLiteResult",
    "SequenceData",
    "make_sequence",
]

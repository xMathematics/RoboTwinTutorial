"""紧耦合视觉-惯性捆绑调整教学包（教程第 10 章 §10.3）。

函数流水线
----------
简化的 VINS-Mono（Qin, Li & Shen, IEEE T-RO 2018）：滑动窗口内的关键帧由
IMU 预积分因子（包装 ``preint.Preintegration``，Forster et al. T-RO 2017
Eq.(45)+(48)）与带 Huber 鲁棒化的单目归一化坐标重投影因子联合估计，外层用
Levenberg-Marquardt 循环配 SE(3) 流形位姿更新求解。内置的
:func:`simulate_vi_scene` 故意把初始化设成错误尺度，以演示教程 §10.3 第 3 步
的单目 + IMU 尺度可观性（scale observability）论证（VINS-Mono §V 的数值
对应物）。

输入/输出：仿真器 :func:`simulate_vi_scene` 输出 :class:`VISimulation`
（真值状态/路标、每区间 IMU 采样、归一化坐标观测、故意错误尺度的初始化）；
:func:`build_vins_bundle` 把它组装成 :class:`ViBundle` 因子图（式 (10.9)）；
:meth:`ViBundle.solve` 用 LM 流形求解并输出 :class:`ViResult`，配合
:func:`align_se3`（Kabsch/Umeyama 对齐）做规范消除后的轨迹评估。

依赖关系：复用 ``core.lie``（hat/so3_exp/so3_log/se3_exp/vee）、
``core.solver.huber_weights``（视觉残差的鲁棒权重）、
``epipolar.triangulate``（DLT 三角化初始路标）以及 ``preint``（IMU 预积分）。
由 ``tests/test_vins.py`` 直接驱动。
"""
from .vins import (
    ImuFactor,
    KeyframeState,
    ReprojectionFactor,
    VISimulation,
    ViBundle,
    ViResult,
    align_se3,
    build_vins_bundle,
    simulate_vi_scene,
)

__all__ = [
    "ImuFactor",
    "KeyframeState",
    "ReprojectionFactor",
    "VISimulation",
    "ViBundle",
    "ViResult",
    "align_se3",
    "build_vins_bundle",
    "simulate_vi_scene",
]

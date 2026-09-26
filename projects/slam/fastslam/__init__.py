"""FastSLAM 2D 教学包（教程第 07 章 §7.3）。

函数流水线
----------
Montemerlo, Thrun, Koller & Wegbreit, *FastSLAM* (AAAI 2002) 的
Rao-Blackwellized（Rao-Blackwell 化）粒子滤波教学实现：机器人路径上的
粒子滤波 × 每个路标独立的 2x2 EKF（论文 Eq. (4)，教程第 07 章 §7.3 的
Rao-Blackwell 分解式），外加一个矩形巡逻仿真器 :func:`simulate_rectangle`，
记录真值轨迹、真值地图、带噪里程计与带真值关联的距离-角度观测。

输入/输出：仿真器输入为随机种子与场景参数，输出 :class:`SimulationResult`
（各数组形状见其 docstring）；滤波器 :class:`FastSLAM2D` 逐步消化
里程计与观测，输出位姿估计与地图估计。

依赖关系：本包**不依赖 ``core``**——2D 平面场景的位姿推演与测量几何只需
numpy 三角函数，自成一体；与其他章节模块（``epipolar`` 等）无调用关系，
由 ``tests/test_fastslam.py`` 直接驱动。

docstring 中的式号均指向
papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf。
"""
from .fastslam import (
    ASSOCIATION_KNOWN,
    ASSOCIATION_NEAREST_NEIGHBOR,
    GATE_CHI2_2DOF_95,
    GATE_DEFAULT,
    FastSLAM2D,
    SimulationResult,
    range_bearing_jacobian,
    range_bearing_measurement,
    simulate_rectangle,
    wrap_angle,
)

__all__ = [
    "ASSOCIATION_KNOWN",
    "ASSOCIATION_NEAREST_NEIGHBOR",
    "GATE_CHI2_2DOF_95",
    "GATE_DEFAULT",
    "FastSLAM2D",
    "SimulationResult",
    "range_bearing_jacobian",
    "range_bearing_measurement",
    "simulate_rectangle",
    "wrap_angle",
]

"""PTAM 教学代理包（教程第 05/08/10 章系统观：双线程架构 + 关键帧局部 BA）.

Klein & Murray, *Parallel Tracking and Mapping for Small AR Workspaces*
(ISMAR 2007) 的教学实现：跟踪/建图双线程架构的教学模拟（交替阶段，见
:mod:`ptam.ptam` 模块 docstring 的显式声明）、论文 §6.2 的关键帧插入判据
（帧距 + 平移/视差阈值）、§5 的 motion-only 位姿更新（复用
:func:`core.solver.gauss_newton` 与 :func:`epipolar.reprojection_jacobian`）、
§6.3 Eq.(11) 的关键帧局部 BA（稠密联合 G-N + 流形重线性化），以及 §6.1
双帧初始化的教学替身（复用 :mod:`epipolar` 的八点法/手性分解/DLT 三角化）。

函数流水线（整体系统中的位置、输入/输出与依赖）
------------------------------------------------
1. :func:`simulate_planar_scene` / :func:`render_planar_frames`：带纹理的
   3D 平面场景 + 确定性相机轨迹 -> 逐帧带噪像素观测（教学"数据集"）；
2. :func:`track_frame`：motion-only G-N 跟踪（论文 §5 / Eq.(8)-(9)）；
3. :func:`ba_terms` / :func:`local_ba`：关键帧局部 BA 的残差/雅可比组装与
   求解（论文 §6.3 / Eq.(10)-(11) 的教学替身）；
4. :class:`Ptam`：系统主体——逐帧 :meth:`Ptam.process_frame` 交替执行
   跟踪与建图阶段，维护关键帧地图；:func:`run_ptam` 为整段场景的便捷
   驱动（输出 :class:`PtamResult`：轨迹/跟踪掩码/逐帧日志/地图）；
5. :func:`project_visible`、:func:`mean_parallax`、:func:`camera_center`：
   投影筛选、视差（关键帧判据物理量）与光心提取的工具件。

依赖方向：``core.lie``（SE(3) 指数映射/点变换）、``core.camera``（针孔
投影）、``core.solver``（GN/LM + Huber）与 ``epipolar``（八点法/手性分解/
DLT 三角化/PnP 雅可比——初始化与跟踪的几何核心同源）；指标评估由
``tests/test_ptam.py`` 经 :mod:`metrics` 完成。

docstring 中全部论文式号指 papers/slam/classics/PTAM_ISMAR2007_KleinMurray.pdf，
教程式号（(5.x)/(8.x)/§10.3）指 tutorials/slam/ 第 05/08/10 章。
"""
from .ptam import (
    BAResult,
    DEFAULT_XI,
    FrameLog,
    KeyFrame,
    PlanarScene,
    Ptam,
    PtamResult,
    TrackResult,
    ba_terms,
    camera_center,
    local_ba,
    mean_parallax,
    project_visible,
    render_planar_frames,
    run_ptam,
    simulate_planar_scene,
    track_frame,
)

__all__ = [
    "BAResult",
    "DEFAULT_XI",
    "FrameLog",
    "KeyFrame",
    "PlanarScene",
    "Ptam",
    "PtamResult",
    "TrackResult",
    "ba_terms",
    "camera_center",
    "local_ba",
    "mean_parallax",
    "project_visible",
    "render_planar_frames",
    "run_ptam",
    "simulate_planar_scene",
    "track_frame",
]

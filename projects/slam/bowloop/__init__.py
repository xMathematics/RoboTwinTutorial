"""词答回环检测与位姿图优化包（教程第 09 章）.

本包在系统中的位置: 前端输出关键帧描述子 -> 场景识别（回环候选检索与验证）
-> 位姿图一致性优化。流水线: ``Vocabulary``（离线 k-means 词表 + IDF ->
每帧 TF-IDF BoW 向量）-> ``InvertedIndex``（word -> 帧倒排索引）->
``detect_loop``（L1 评分 (9.3) + 时间一致性 -> 回环对）-> ``PoseGraph2D``
（SE(2) 位姿图 Gauss-Newton 优化 (9.8)，gauge fixing 锚定首节点）。输入为
各帧描述子 (N, D) 与位姿图节点/边，输出为确认的回环对列表与优化后位姿；
仅依赖 numpy。DBoW2 式词袋（视觉词表 + TF-IDF BoW 向量 + L1 评分 + 倒排索
引 + 时间一致性）。式号 (9.x) 指向 ``tutorials/slam/09_回环检测.md``，详见
``bowloop.bowloop`` 模块 docstring。ORB-SLAM §VII 的回环线程 = 本模块的
检索/验证 + Essential Graph 位姿图。
"""
from .bowloop import (
    InvertedIndex,
    PoseGraph2D,
    PoseGraphEdge,
    Vocabulary,
    detect_loop,
    score,
    se2_relative_jacobian,
    se2_relative_residual,
    wrap_angle,
)

__all__ = [
    "InvertedIndex",
    "PoseGraph2D",
    "PoseGraphEdge",
    "Vocabulary",
    "detect_loop",
    "score",
    "se2_relative_jacobian",
    "se2_relative_residual",
    "wrap_angle",
]

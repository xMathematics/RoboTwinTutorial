"""光度法光束平差（photometric bundle adjustment）包（教程第 06/08 章，DSO 思想）。

函数流水线
----------
滑窗光度 BA：联合优化每帧仿射亮度参数 (a_i, b_i) 与每点逆深度
（inverse depth），沿 Engel, Koltun & Cremers, "Direct Sparse Odometry"
（arXiv:1607.02565，论文 Eq. (2)-(9)），构建在共享的 ``core`` 求解器与
``direct`` 模块的几何之上。

``make_keyframe_scene`` / :class:`KeyframeScene`（复用 :mod:`direct` 的
纹理平面渲染 3 关键帧真值场景）→ :class:`PhotometricBA`（host 像素选取 /
参数布局 :meth:`PhotometricBA.parameter_layout` / 残差 (6.6) / 解析雅可比 /
LM + Nielsen 增益比（gain ratio）自适应 + 逆深度物理箱）→
:meth:`PhotometricBA.solve` → :class:`PhotometricBAResult`。
对应 DSO 论文式 (2)-(9) 与教程 (6.6)-(6.8)、(8.5)-(8.6)、(8.9)-(8.10)；
鲁棒核复用 :func:`core.solver.huber_weights`，投影雅可重复用
:func:`direct.projection_jacobian`。
"""
from .photoba import (
    KeyframeScene,
    PhotometricBA,
    PhotometricBAResult,
    make_keyframe_scene,
)

__all__ = ["KeyframeScene", "PhotometricBA", "PhotometricBAResult", "make_keyframe_scene"]

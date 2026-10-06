"""RoboTwin 迷你基准的统一测评模块（metrics）——全部指标的唯一定义处。

函数流水线
----------
本模块是评测流水线的最后一环（纯函数层）：``benchmark.run`` 产出的逐
回合成功记录经本模块的三个纯函数聚合成协议要求的口径——每任务成功率、
宏平均 + 最差任务成功率、泛化差距；``demo.py`` 的报告表与
``tests/test_metrics.py`` 的断言全部从这里取公式与实现。依赖方向：
``benchmark / tests / demo → 本模块``；本模块只依赖 numpy，不依赖任何
被测模块。

指标清单（公式编号 (M.x) 供 METRICS.md 与教程回引）
----------------------------------------------------
- :func:`success_rate`         成功率（RoboTwin 2.0 附录 G 的 R 口径）
- :func:`macro_mean_and_worst` 跨任务宏平均 + 最差任务成功率（双层报告）
- :func:`generalization_gap`   泛化差距 seen − unseen（Table 4 配置轴）

评测口径出处：RoboTwin 2.0（Bu et al., arXiv:2506.18088）附录 G
（成功率 = 对 M 次执行的成功指示变量取平均）、附录 L（50 任务 × 5 本体
逐格成功率的双层报告）；教程第 05 章 §5.6（"报均值更报最差"的五步论证）、
第 06 章 §6.5（seen/unseen 配置轴）、§6.6（基准 Table 5 的宏平均口径）。

全部函数为纯函数：不修改输入、无全局状态、输入为空时显式抛
:class:`ValueError`（早失败优于静默返回 NaN）。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["success_rate", "macro_mean_and_worst", "generalization_gap"]


def success_rate(results: Sequence[float] | np.ndarray) -> float:
    """成功率：成功指示变量的均值，见 (M.1)。

    $$\\mathrm{SR} = \\frac{1}{N}\\sum_{i=1}^{N} s_i, \\qquad s_i \\in \\{0, 1\\} \\tag{M.1}$$

    出处：RoboTwin 2.0（arXiv:2506.18088）附录 G——对 M 次仿真执行的成功
    指示变量取平均；教程 §5.6 "每任务 100 次 rollout、报告成功率" 的口径。
    本项目每回合 rollout 一次，``info["success"] ∈ {True, False}`` 即 $s_i$。

    Args:
        results: (N,) 成功指示序列——元素为 0/1（或 bool，或已聚合的
            成功率值 ∈ [0, 1]；两种输入同口径，函数不做区分）。

    Returns:
        成功率，标量 ∈ [0, 1]，无量纲。

    Raises:
        ValueError: 输入为空（0 回合的"成功率"无定义）。
    """
    arr = np.asarray(list(results), dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("success_rate：输入为空（至少需要 1 个回合结果）")
    # (M.1)：指示变量均值。
    return float(np.mean(arr))


def macro_mean_and_worst(
    per_task_success: Mapping[str, float] | Sequence[float],
    task_names: Sequence[str] | None = None,
) -> tuple[float, float, str | None]:
    """跨任务宏平均 + 最差任务成功率（双层报告的"报均值更报最差"），见 (M.2)。

    $$\\overline{\\mathrm{SR}}_{\\mathrm{macro}} = \\frac{1}{K}\\sum_{k=1}^{K} r_k,
    \\qquad
    \\mathrm{SR}_{\\mathrm{worst}} = \\min_{k}\\, r_k \\tag{M.2}$$

    为什么均值之外必须报最差：均值是分布的一阶摘要，"全面 60 分"与
    "一半满分一半零分"看起来一样；只有最差行能定位"策略到底在哪崩"
    （依据：教程 §5.6 知识点五步——均值不唯一决定分布；论文的实践：
    附录 L 公开全部 50 任务 × 5 本体逐格成功率，Table 4 把**最差配置**
    的相对提升写进摘要）。鲁棒性主张以最差情形检验的思想另见
    Rajeswaran et al., ICLR 2017（EPOpt）。

    Args:
        per_task_success: 每任务成功率——``dict[任务名, 成功率]``，或
            (K,) 数组/序列（此时传 ``task_names`` 给出任务名，否则名字
            为 None/自动编号）。
        task_names: (K,) 任务名（仅当第一个参数是序列时使用）。

    Returns:
        ``(macro_mean, worst, worst_task)``：宏平均 ∈ [0, 1]；最差任务
        成功率 ∈ [0, 1]；最差任务名（并列取字典序/输入序第一个，
        无名字时为 None）。

    Raises:
        ValueError: 输入为空。
    """
    if isinstance(per_task_success, Mapping):
        names = list(per_task_success.keys())
        values = [float(v) for v in per_task_success.values()]
    else:
        values = [float(v) for v in per_task_success]  # type: ignore[arg-type]
        names = list(task_names) if task_names is not None else []
    if len(values) == 0:
        raise ValueError("macro_mean_and_worst：输入为空（至少需要 1 个任务）")
    # (M.2)：宏平均（对任务一阶平均，每任务权重相同——不按回合数加权）；
    # 最差 = min。宏平均与逐任务简单平均一致正是"macro"的含义。
    macro = float(np.mean(values))
    worst_idx = int(np.argmin(values))
    worst = values[worst_idx]
    worst_name = names[worst_idx] if worst_idx < len(names) else None
    return macro, worst, worst_name


def generalization_gap(
    seen: Sequence[float] | np.ndarray, unseen: Sequence[float] | np.ndarray
) -> float:
    """泛化差距：seen 成功率 − unseen 成功率，见 (M.3)。

    $$\\mathrm{Gap} = \\frac{1}{K}\\sum_{k} r_k^{\\mathrm{seen}}
    \\; - \\; \\frac{1}{K}\\sum_{k} r_k^{\\mathrm{unseen}} \\tag{M.3}$$

    正值 = 策略在训练/标定分布内强、外推到未见配置变弱（过拟合到 seen
    分布的证据）；0 = 对该配置轴无泛化损失。出处：RoboTwin 2.0（arXiv:
    2506.18088）§4.4 Table 4 的 {seen, unseen} 背景 × {干净, 杂乱} 配置轴
    ——环境越未见、差距越大（最差配置 +367% 相对提升正来自这一轴）；
    教程 §5.6 ④（RT-1 的 seen/unseen 分层报告）与 §6.5（Sim-to-Real 的
    seen/unseen 解读）。本项目把该轴落在"目标位姿范围"上（``tasks.reset``
    的 ``split`` 参数）：seen = 标定工作空间内部，unseen = 外围一圈。

    Args:
        seen: (K,) seen 配置下的成功率（或 0/1 回合指示）。
        unseen: (K,) unseen 配置下的成功率，与 ``seen`` 逐任务对应。

    Returns:
        泛化差距，标量 ∈ [-1, 1]，无量纲；正值代表 unseen 变弱。

    Raises:
        ValueError: 任一输入为空，或两者长度不一致。
    """
    s_arr = np.asarray(list(seen), dtype=float).reshape(-1)
    u_arr = np.asarray(list(unseen), dtype=float).reshape(-1)
    if s_arr.size == 0 or u_arr.size == 0:
        raise ValueError("generalization_gap：输入为空（seen/unseen 各至少 1 项）")
    if s_arr.size != u_arr.size:
        raise ValueError(
            f"generalization_gap：seen/unseen 任务数不一致 {s_arr.size} vs {u_arr.size}"
        )
    # (M.3)：两个配置轴上的宏平均之差。
    return float(np.mean(s_arr) - np.mean(u_arr))

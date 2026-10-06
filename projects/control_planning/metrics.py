"""控制与规划统一测评模块（metrics）—— 全部指标的唯一定义处。

函数流水线
----------
本模块是评估流水线的最后一环，被各规划/控制模块的评估环节调用：
``rrt`` / ``ilqr`` / ``mpc_cem`` / ``cbf`` / ``ppo_lite`` 产出的轨迹与控制
序列，都应经本模块的纯函数评估。``tests/test_*.py`` 中的指标断言从这里取
公式与实现。依赖方向：``各模块 → tests/test_*.py → 本模块``；本模块不依赖
库内任何被测模块，只依赖 numpy。

输入/输出：输入轨迹 ``traj`` (N, 2)（平面位置序列，单位 m，逐时刻一一对应）、
控制序列 ``u_traj`` (N-1, 2)（加速度，单位 m/s²）、障碍列表
``obstacle_list``（每项 = ``(center (2,), radius)``，圆域障碍）；
输出标量指标（无量纲 / m / 混合单位，见各函数 docstring）。

指标清单（公式编号 (M.x) 供 METRICS.md 回引；CONSTRAINTS.md §4.7 指定的
control_planning 三指标基线 = 规划成功率 / 路径长度 / 轨迹代价）
----------------------------------------------------
- :func:`success_rate`     规划成功率：到达目标且全程无碰撞的比例 (M.1)
- :func:`path_length`      路径长度：相邻采样点折线段长之和 (M.2)
- :func:`trajectory_cost`  轨迹代价：控制功效 ∫‖u‖² dt + 平滑项 (M.3)

全部函数为纯函数：不修改输入、无全局状态，可安全地在测试断言中直接调用。
"""
import numpy as np

__all__ = [
    "success_rate",
    "path_length",
    "trajectory_cost",
]

#: 碰撞判定余量的下限：障碍距离恰好等于半径时视为相切（不碰撞），
#: 只在距离严格小于 ``radius - tol_collision`` 时判碰撞。
_TOL_COLLISION = 1e-12


def _as_position_traj(traj: np.ndarray) -> np.ndarray:
    """把轨迹输入规整为 (N, 2) 浮点数组（各指标的公共入口检查）。

    Args:
        traj: (N, 2) 或 (N, d) 位置序列（允许传入 (N, 4) 双积分器状态，
              自动取前两维位置分量），单位 m。

    Returns:
        (N, 2) float64 位置序列（视图/拷贝均不修改调用方数据）。

    Raises:
        ValueError: 轨迹少于 2 个采样点（无法定义路径长度/代价的时间网格）。
    """
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(
            f"轨迹须为 (N, 2) 及以上、N >= 2 的数组，当前形状 {arr.shape}"
        )
    return arr[:, :2]


def success_rate(
    trajs: list[np.ndarray] | np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    tol: float,
) -> float:
    """规划成功率：到达目标**且**全程无碰撞的轨迹占比，无量纲，见 (M.1)。

    这是 CONSTRAINTS.md §4.7 为 control_planning 指定的三指标基线之一，
    衡量规划/控制模块"任务完成质量"的复合主张：规划器（``rrt``）给出的
    路径必须无碰撞地伸入目标邻域（教程第 03 章 (3.1) 的可行性），MPC /
    CBF / 学习式控制（``mpc_cem`` / ``cbf`` / ``ppo_lite``）滚动执行的
    轨迹必须既到达又安全（教程第 05 章 (5.1) 的约束、第 07 章 (7.1) 的
    安全集）。两个条件缺一不可：只看到达会奖励"穿墙直达"的假成功。

    Args:
        trajs: 轨迹列表；每条为 (N_i, 2)（或 (N_i, 4) 自动取位置分量）
            的位置序列，单位 m。N_i 可各不相同。
        goal: (2,) 目标点，单位 m。
        obstacle_list: 圆域障碍列表，每项 ``(center (2,), radius)``，
            center 单位 m、radius 单位 m（点机器人模型：位置进入圆域即碰撞）。
        tol: 到达容差，单位 m；末端点与 goal 的欧氏距离 ≤ tol 即判"到达"。

    Returns:
        成功率，无量纲标量，取值 [0, 1] = 成功条数 / 总条数；空输入返回 0.0。
    """
    g = np.asarray(goal, dtype=float).reshape(2)
    trajs = list(trajs)
    if len(trajs) == 0:
        return 0.0
    n_success = 0
    for traj in trajs:
        pos = _as_position_traj(traj)  # (N, 2) 位置分量，单位 m
        # 条件一：到达。依据：(M.1) 的到达子条件——末端落在目标 tol-邻域。
        reached = float(np.linalg.norm(pos[-1] - g)) <= tol
        # 条件二：全程无碰撞。逐采样点检查点到每个圆心的距离是否 < 半径。
        # 依据：(M.1) 的安全子条件——点机器人模型下碰撞 = 位置进入圆域；
        # 采样点间距引起的穿隧漏检口径与教程第 03 章 (3.10) 相同，
        # 调用方应以足够密的采样（或保守半径）补偿。
        collision = False
        for center, radius in obstacle_list:
            c = np.asarray(center, dtype=float).reshape(2)
            d2 = np.sum((pos - c) ** 2, axis=1)  # (N,) 到圆心的距离平方
            if np.any(d2 < (radius - _TOL_COLLISION) ** 2):
                collision = True
                break
        if reached and not collision:
            n_success += 1
    # (M.1)：成功率 = 成功条数 / 总条数。
    return float(n_success) / float(len(trajs))


def path_length(traj: np.ndarray) -> float:
    """路径长度：相邻采样点欧氏距离之和（折线总长），单位 m，见 (M.2)。

    对应教程第 03 章 RRT* 的边代价定义（论文框架取欧氏长度，教程 (3.5)
    的逐边累加对象）：路径长度是"规划质量的几何度量"，与控制无关——
    同一任务的可行解可以任意长，最优解长度是路径规划文献的标准对比量。

    Args:
        traj: (N, 2) 位置序列（(N, 4) 自动取位置分量），单位 m，N ≥ 2。

    Returns:
        路径长度，标量，单位 m，取值 ≥ 0。
    """
    pos = _as_position_traj(traj)  # (N, 2)
    # (M.2)：Σ_i ‖p_{i+1} - p_i‖。diff 一次性给出所有相邻差分
    # （依据：向量化切片，与逐段累加循环等价，tests/test_metrics.py 有朴素版交叉验证）。
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)  # (N-1,)
    return float(np.sum(steps))


def trajectory_cost(
    traj: np.ndarray,
    u_traj: np.ndarray,
    dt: float = 1.0,
    w_smooth: float = 0.1,
) -> float:
    r"""轨迹代价：控制功效 + 控制平滑项，见 (M.3)。

    对应教程第 04 章最优控制目标 (4.1) 的离散版：阶段代价取控制功效
    $\ell_u = \|u\|^2$（能耗 / 力矩方量的标准代理），另加相邻控制差的
    平方作平滑惩罚——第 03 章 RRT* 折线的"拐点抖动"（教程 04.1 的问题
    场景）在控制序列上表现为 $\|u_{i+1} - u_i\|$ 大，平滑项将其显式计价。
    这是 CONSTRAINTS.md §4.7 三指标基线中"轨迹好不好走"的一支：
    ``path_length`` 管几何、本指标管执行代价。

    $$
    J = \\sum_{i=0}^{N-2} \\|u_i\\|^2\\,\\Delta t
      + w_{\\mathrm{smooth}} \\sum_{i=0}^{N-3} \\|u_{i+1} - u_i\\|^2\\,\\Delta t
    \\tag{M.3}
    $$

    Args:
        traj: (N, 2) 位置序列（定义时间网格长度；本指标不直接使用位置值，
            但要求 ``len(u_traj) == N - 1`` 以保证控制与状态同网格）。
        u_traj: (N-1, 2) 控制序列（加速度），单位 m/s²。
        dt: 采样周期，单位 s；积分按矩形法离散（每段代价 × dt）。
        w_smooth: 平滑项权重，无量纲；0.0 时退化为纯控制功效。

    Returns:
        轨迹代价，标量；量纲为 (m²/s⁴)·s = m²/s³（功效项）与平滑项同量纲
        加权，仅作同网格下的相对比较。

    Raises:
        ValueError: ``len(u_traj) != N - 1`` 或 ``dt <= 0``。
    """
    u = np.asarray(u_traj, dtype=float)
    if u.ndim != 2 or u.shape[1] < 2:
        raise ValueError(f"控制序列须为 (N-1, 2)，当前形状 {u.shape}")
    pos = _as_position_traj(traj)
    if u.shape[0] != pos.shape[0] - 1:
        raise ValueError(
            f"控制序列长度须为轨迹点数 - 1（{pos.shape[0] - 1}），当前 {u.shape[0]}"
        )
    if dt <= 0.0:
        raise ValueError(f"采样周期必须为正，当前 dt={dt}")
    # 功效项：Σ ‖u_i‖² Δt（依据：(M.3) 第一项，矩形法离散 ∫‖u‖²dt）。
    effort = float(np.sum(np.sum(u * u, axis=1)) * dt)
    # 平滑项：w · Σ ‖u_{i+1} - u_i‖² Δt（依据：(M.3) 第二项，相邻控制差）。
    du = np.diff(u, axis=0)  # (N-2, 2)
    smooth = float(np.sum(np.sum(du * du, axis=1)) * dt) * w_smooth
    return effort + smooth

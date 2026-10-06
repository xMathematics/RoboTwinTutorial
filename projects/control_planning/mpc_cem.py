"""CEM-MPC：滚动时域 + 交叉熵方法（PETS 已知动力学简化档，教学实现）。

函数流水线
----------
Chua, Calandra, McAllister, Levine, *Deep Reinforcement Learning in a
Handful of Trials using Probabilistic Dynamics Models*, NeurIPS 2018
（本地 papers/control_planning/frontier/arXiv-1805.12114_PETS.pdf）中
"CEM 采样优化 + MPC 滚动执行"路线的教学简化档：动力学已知（2D 双积分器，
复用 :func:`ilqr.double_integrator_step`），去掉概率集成模型——即论文
消融排序 PE > P > DE > D 里的确定性档（论文 §7.2），推导见教程第 05 章
05.1（滚动时域三步循环）与 05.3（CEM 迭代 (5.11)–(5.13)）。

调用链：:func:`cem_plan`（单周期 OCP 求解：采样 → 评分 (5.11)/(5.12) →
精英重分布 (5.13)）→ :func:`cem_mpc_rollout`（(5.1)/(5.2) 的滚动时域循环：
测量 → 重解 → 执行首步，含温启动与动力学噪声）→ :class:`MPCResult`，
评估经 ``metrics.success_rate`` / ``metrics.trajectory_cost``
（由 ``tests/test_mpc_cem.py`` 与 ``demo.py`` 驱动）。

依赖方向：``mpc_cem → ilqr``（被控对象同一台双积分器）与
``mpc_cem → rrt``（基准场景复用同一走廊障碍图；局部导入）。

CEM 单次求解为（教程 05.3 第五步）：

1. 对候选动作序列逐时刻维护独立高斯 ``N(mean_t, sigma_t²)``（初值宽方差）；
2. 采样 n_samples 条候选，按已知动力学确定性 rollout，累计目标代价 +
   障碍惩罚 + 控制功效作为负回报（(5.11) 的 μ；无集成模型故 (5.12)
   的方差惩罚项没有认知不确定性来源，取 λ = 0）；
3. 取最优 n_elite 条为精英集，按 (5.13) 用精英均值/方差重分布搜索
   分布（方差加下限防过早收缩）；
4. 返回精英均值的首步作控制——(5.2) 的 ``π_MPC(x) = u_0*(x)``。

约定
----
- 状态/控制形状：``x (4,)``、``u (2,)``，含义同 :mod:`ilqr`；
  候选批形状 ``(n_samples, horizon, 2)``。
- 随机性全部经 ``np.random.default_rng(seed)``（CEM 采样与执行噪声
  各自独立计流），同 seed 结果逐位可复现。

示例
----
>>> import numpy as np
>>> from mpc_cem import cem_mpc_rollout, make_reach_scene
>>> scene = make_reach_scene()
>>> res = cem_mpc_rollout(np.zeros(4), scene["goal"], scene["obstacles"], seed=0)
>>> res.final_pos_err < 0.2  # doctest: +SKIP
True
"""
from dataclasses import dataclass, field

import numpy as np

from ilqr import double_integrator_step

__all__ = [
    "CEMPlanResult",
    "MPCResult",
    "candidate_cost",
    "cem_mpc_rollout",
    "cem_plan",
    "make_reach_scene",
]


def make_reach_scene() -> dict:
    """构造 CEM-MPC 基准场景（demo 与测试共用，走廊口径与 :mod:`rrt` 一致）。

    Returns:
        dict：``goal (2,)`` 目标点（m）、``obstacles`` 圆障碍列表（与
        :func:`rrt.make_corridor_obstacles` 相同的墙体，单位 m）、
        ``bounds (4,)`` 场地边界。
    """
    from rrt import make_corridor_obstacles  # 局部导入：依赖方向见模块 docstring

    return {
        "goal": np.array([9.0, 6.0]),
        "obstacles": make_corridor_obstacles(),
        "bounds": (0.0, 10.0, 0.0, 10.0),
    }


def candidate_cost(
    u_batch: np.ndarray,
    x0: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    dt: float,
    w_goal: float = 1.0,
    w_u: float = 0.05,
    w_obs: float = 2000.0,
    r_safe: float = 0.6,
) -> np.ndarray:
    """一批候选动作序列的代价（(5.11) 评分的确定性档：回报取负）。

    代价 = Σ 目标距离² + Σ 控制功效 + Σ 障碍穿透惩罚——目标代价把
    "到得了"写成吸引力，障碍惩罚把 (5.1) 的状态约束 ``x ∈ X`` 以罚函数
    写进无梯度优化（约束直接进优化是 MPC 区别于固定增益的关键能力，
    教程 05.1 ③）。

    Args:
        u_batch: (S, H, 2) 候选动作序列（S 条、时域 H），单位 m/s²。
        x0: (4,) 规划起点状态。
        goal: (2,) 目标位置，单位 m。
        obstacle_list: 圆障碍列表，每项 ``(center (2,), radius)``，单位 m。
        dt: 步长，单位 s。
        w_goal / w_u: 目标 / 控制功效权重（无量纲缩放）。
        w_obs: 障碍惩罚权重（穿透量的平方放大，取大值硬压碰撞）。
        r_safe: 安全余量：位置距障碍圆心小于 radius + r_safe 即开始计罚，
            单位 m。

    Returns:
        (S,) 各候选的总代价（越小越好）。
    """
    u_batch = np.asarray(u_batch, dtype=float)
    n, h, _ = u_batch.shape
    goal = np.asarray(goal, dtype=float).reshape(2)
    xs = np.tile(np.asarray(x0, dtype=float).reshape(4), (n, 1))  # (S, 4)
    has_obs = len(obstacle_list) > 0
    if has_obs:
        centers = np.stack([np.asarray(c, dtype=float) for c, _ in obstacle_list])
        radii = np.asarray([r for _, r in obstacle_list], dtype=float)
    total = np.zeros(n)
    for k in range(h):
        # 向量化 rollout：整批候选同一步推进（(5.1) 的动力学约束逐段成立）。
        pos = xs[:, 0:2] + xs[:, 2:4] * dt
        vel = xs[:, 2:4] + u_batch[:, k, :] * dt
        xs = np.column_stack([pos, vel])
        # 目标代价：距离平方沿途累计（教程 (5.1) 的阶段代价 l）。
        total += w_goal * np.sum((pos - goal[None, :]) ** 2, axis=1) * dt
        # 控制功效（小权重：只压制无意义的抖动，(5.1) 的 u ∈ U 软化）。
        total += w_u * np.sum(u_batch[:, k, :] ** 2, axis=1) * dt
        if not has_obs:
            continue
        # 障碍惩罚：穿透量 max(0, radius + r_safe - dist) 的平方（凸增罚）。
        d = np.linalg.norm(pos[:, None, :] - centers[None, :, :], axis=2)  # (S, M)
        pen = np.maximum(0.0, (radii[None, :] + r_safe) - d)
        total += w_obs * np.sum(pen * pen, axis=1) * dt
    return total


@dataclass
class CEMPlanResult:
    """单周期 CEM 求解结果（DEBUG 变量表 M-1～M-3 的来源）。

    Attributes:
        u_seq: (H, 2) 精英均值动作序列（首步即 (5.2) 的执行量）。
        best_cost: 精英均值序列的代价（标量）。
        mean: (H, 2) 收敛后的搜索分布均值。
        sigma_trace: (n_iters,) 每轮精英协方差迹的逐时刻均值
            （(5.13) 的收缩仪表：应单调下降并停在 sigma_min 附近）。
        elite_cost_mean: (n_iters,) 每轮精英集平均代价（应逐轮下降）。
    """

    u_seq: np.ndarray
    best_cost: float
    mean: np.ndarray
    sigma_trace: np.ndarray
    elite_cost_mean: np.ndarray


def cem_plan(
    x0: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    dt: float,
    horizon: int = 14,
    n_samples: int = 48,
    n_elite: int = 10,
    n_iters: int = 8,
    mean_init: np.ndarray | None = None,
    sigma0: float = 1.5,
    sigma_min: float = 0.05,
    u_limit: float = 6.0,
    w_goal: float = 1.0,
    w_u: float = 0.05,
    w_obs: float = 2000.0,
    r_safe: float = 0.6,
    seed: int = 0,
) -> CEMPlanResult:
    """单周期 CEM 求解 (5.1)：采样 → 评分 → 精英重分布 (5.13)。

    交叉熵方法在动作序列空间维护逐时刻独立高斯分布：每轮采样
    ``n_samples`` 条候选、确定性 rollout 评分（(5.11)）、取最优
    ``n_elite`` 条按精英均值/方差重分布（(5.13)，方差加 ``sigma_min``
    下限防过早收缩）。无需动力学梯度——CEM 与第 04 章基于梯度的
    iLQR 构成"模型可微与否"的两条求解路线（教程 05.3 ③）。

    Args:
        x0: (4,) 当前状态（测量/估计值，(5.1) 的 ``x_0 = x_t``）。
        goal: (2,) 目标位置，单位 m。
        obstacle_list: 圆障碍列表，单位 m。
        dt: 步长，单位 s。
        horizon: 预测时域 H（(5.1) 的 N）。
        n_samples / n_elite: 每轮候选数 / 精英数（n_elite < n_samples）。
        n_iters: CEM 迭代轮数。
        mean_init: (H, 2) 搜索分布初值（温启动用：上一周期解的平移，
            教程 05.1 ③）；None 时取零序列。
        sigma0: 初始标准差（宽方差 = 初始均匀假设，(5.13) 前提）。
        sigma_min: 标准差下限（防过早收缩，(5.13) 行内注）。
        u_limit: 动作截断幅值，单位 m/s²（候选裁剪进可行箱）。
        w_goal / w_u / w_obs / r_safe: 代价权重与安全余量，语义见
            :func:`candidate_cost`（默认值按无碰撞到达调定，见 DEBUG.md）。
        seed: 采样种子。

    Returns:
        :class:`CEMPlanResult`。
    """
    if not 1 <= n_elite < n_samples:
        raise ValueError(f"要求 1 ≤ n_elite({n_elite}) < n_samples({n_samples})")
    x0 = np.asarray(x0, dtype=float).reshape(4)
    goal = np.asarray(goal, dtype=float).reshape(2)
    mean = (
        np.zeros((horizon, 2))
        if mean_init is None
        else np.asarray(mean_init, dtype=float).reshape(horizon, 2).copy()
    )
    sigma = np.full((horizon, 2), float(sigma0))
    rng = np.random.default_rng(seed)
    sigma_trace: list[float] = []
    elite_cost_mean: list[float] = []
    best_cost = float("inf")
    best_seq = mean.copy()
    for _ in range(n_iters):
        # ① 采样候选（截断进动作箱：u ∈ U 的盒约束，(5.1)）。
        u_batch = mean[None, :, :] + sigma[None, :, :] * rng.standard_normal(
            (n_samples, horizon, 2)
        )
        u_batch = np.clip(u_batch, -u_limit, u_limit)
        # ② 评分（(5.11) 的 μ；已知动力学 → 无 σ 惩罚项，λ = 0）。
        costs = candidate_cost(
            u_batch, x0, goal, obstacle_list, dt,
            w_goal=w_goal, w_u=w_u, w_obs=w_obs, r_safe=r_safe,
        )
        elite_idx = np.argsort(costs)[:n_elite]
        elites = u_batch[elite_idx]  # (E, H, 2)
        # ③ 精英重分布 (5.13)：μ ← mean(精英)，σ ← std(精英) 加下限。
        mean = elites.mean(axis=0)
        sigma = np.maximum(elites.std(axis=0), sigma_min)
        sigma_trace.append(float(np.mean(sigma**2)))  # 逐时刻方差迹的均值
        elite_cost_mean.append(float(costs[elite_idx].mean()))
        c_mean = float(
            candidate_cost(
                mean[None], x0, goal, obstacle_list, dt,
                w_goal=w_goal, w_u=w_u, w_obs=w_obs, r_safe=r_safe,
            )[0]
        )
        if c_mean < best_cost:
            best_cost = c_mean
            best_seq = mean.copy()
    return CEMPlanResult(
        u_seq=best_seq,
        best_cost=best_cost,
        mean=mean,
        sigma_trace=np.asarray(sigma_trace),
        elite_cost_mean=np.asarray(elite_cost_mean),
    )


@dataclass
class MPCResult:
    """滚动时域执行结果。

    Attributes:
        traj: (T+1, 4) 实际状态轨迹（含初态；执行噪声已含）。
        u_applied: (T, 2) 实际执行的首步控制。
        min_dist: 全程最小"障碍圆心距 - 半径"（净距，m；DEBUG 表 M-5）。
        final_pos_err: 末端位置到目标的欧氏误差（m）。
        plans: 每周期 CEMPlanResult 的精简诊断列表（均值序列代价）。
    """

    traj: np.ndarray
    u_applied: np.ndarray
    min_dist: float
    final_pos_err: float
    plans: list[float] = field(default_factory=list)


def cem_mpc_rollout(
    x0: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    n_steps: int = 80,
    dt: float = 0.15,
    horizon: int = 14,
    n_samples: int = 48,
    n_elite: int = 10,
    n_iters: int = 8,
    noise_std: float = 0.05,
    seed: int = 0,
) -> MPCResult:
    """滚动时域主循环（教程 05.1 的三步循环 + (5.2) 的状态反馈策略）。

    每个控制周期：测量当前状态（含执行噪声的 (5.1) ``x_0 = x_t``）→
    :func:`cem_plan` 重解 OCP（上一周期解平移作温启动初值，教程 05.1 ③）
    → 执行首步并把时域前移。执行时对控制注入高斯噪声——扰动由"每周期
    从实测状态重新最优"吸收，这正是 (5.2) 反馈内嵌于反复重解的机制。

    Args:
        x0: (4,) 初始状态。
        goal: (2,) 目标位置，单位 m。
        obstacle_list: 圆障碍列表，单位 m。
        n_steps: 执行步数 T。
        dt: 控制周期，单位 s。
        horizon / n_samples / n_elite / n_iters: 透传给 :func:`cem_plan`。
        noise_std: 执行噪声标准差（加在加速度上），单位 m/s²。
        seed: 随机种子（CEM 与噪声共用一条确定性随机流）。

    Returns:
        :class:`MPCResult`。
    """
    x = np.asarray(x0, dtype=float).reshape(4)
    goal = np.asarray(goal, dtype=float).reshape(2)
    traj = np.empty((n_steps + 1, 4))
    traj[0] = x
    u_applied = np.empty((n_steps, 2))
    plan_costs: list[float] = []
    warm_mean: np.ndarray | None = None
    rng = np.random.default_rng(seed)
    for t in range(n_steps):
        # ② 求解 OCP (5.1)：温启动 = 上一周期解左移一格（时域前移）。
        plan = cem_plan(
            x, goal, obstacle_list, dt,
            horizon=horizon, n_samples=n_samples, n_elite=n_elite,
            n_iters=n_iters, mean_init=warm_mean, seed=seed + t,
        )
        plan_costs.append(plan.best_cost)
        # ③ 执行首步（(5.2)：π_MPC(x) = u_0*；其余丢弃），时域前移。
        u = plan.u_seq[0]
        u_applied[t] = u
        warm_mean = np.vstack([plan.mean[1:], plan.mean[-1:]])
        x = double_integrator_step(
            x, u + noise_std * rng.standard_normal(2), dt
        )
        traj[t + 1] = x
    # 诊断量：全程最小净距与终端误差（指标取用见 METRICS.md）。
    if obstacle_list:
        centers = np.stack([np.asarray(c, dtype=float) for c, _ in obstacle_list])
        radii = np.asarray([r for _, r in obstacle_list], dtype=float)
        d = np.linalg.norm(traj[:, None, :2] - centers[None, :, :], axis=2) - radii
        min_dist = float(d.min())
    else:
        min_dist = float("inf")
    return MPCResult(
        traj=traj,
        u_applied=u_applied,
        min_dist=min_dist,
        final_pos_err=float(np.linalg.norm(traj[-1, :2] - goal)),
        plans=plan_costs,
    )

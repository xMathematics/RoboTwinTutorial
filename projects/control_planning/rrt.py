"""2D 采样式运动规划：RRT 与 RRT*（教学实现）。

函数流水线
----------
Karaman & Frazzoli, *Sampling-based Algorithms for Optimal Motion Planning*,
IJRR 2011（本地 papers/control_planning/classics/arXiv-1105.1186_RRTstar.pdf；
RRT* 见 Algorithm 1–3 与 Theorem 38）与 LaValle, TR 98-11（RRT 原始算法）的
教学实现，推导见教程第 03 章。规划对象为 2D 平面点机器人（构型空间 =
``bounds`` 矩形），障碍为圆域集合；即教程 (3.1) 中 ``C_free`` 由碰撞检测
隐式给定的设定。

本模块不依赖库内其他模块（仅 numpy + 标准库）。调用链：
:func:`plan_rrt` / :func:`plan_rrt_star` 的主循环逐步调用
:func:`steer`（教程 (3.3) 限步长转向）、:func:`segment_free`（增量碰撞检测，
(3.4) 的边可行性）、:func:`neighbor_radius`（RRT* 收缩半径，教程 (3.7)）与
:func:`gamma_constant`（Theorem 38 的 γ 下界常数）；输出 :class:`PlanResult`
（树数组 + 可行路径 + 代价），评估经 ``metrics.path_length`` /
``metrics.success_rate``（由 ``tests/test_rrt.py`` 与 ``demo.py`` 驱动）。

每轮迭代为（教程 03.1 的算法骨架）：

1. 采样 ``q_rand``：以概率 ``goal_bias`` 直接取目标点，否则在 bounds 内
   均匀采样（goal bias，教程 (3.11)：期望每 1/p_g 轮一次直连尝试）；
2. 树上最近邻 + :func:`steer` 限步长（(3.3)）得 ``q_new``；
3. 对候选边做 motion validation（(3.4)：按 ``max_edge_check_dist`` 逐点
   检查，穿隧口径见教程 (3.10)）；
4. RRT：直接以最近邻为父挂树；RRT*：在收缩半径邻域内先**选父**再
   **重布线**（(3.5)–(3.7)，代价沿树递归定义且逐节点单调不增）；
5. ``q_new`` 落入目标邻域（半径 ``goal_radius``）→ 回溯 parent 提取路径。

约定
----
- 构型 ``q = (x, y)``，单位 m；世界系右手平面系。
- 树以三个平行数组存储：``nodes (n, 2)``、``parent (n,)``（根为 -1）、
  ``cost (n,)``（节点代价 (3.5)：从根到该节点的欧氏折线长）。
- 碰撞判定：点到圆心距离严格小于 ``radius`` 时碰撞（相切不碰）；边检查
  按间距 ``max_edge_check_dist`` 采样——这是 (3.10) 的离散化口径，调用方
  应取其为最小障碍净距的若干分之一。

示例
----
>>> import numpy as np
>>> from rrt import plan_rrt_star
>>> res = plan_rrt_star(
...     start=np.array([0.5, 0.5]), goal=np.array([9.5, 9.5]),
...     obstacles=[(np.array([5.0, 5.0]), 1.5)],
...     bounds=(0.0, 10.0, 0.0, 10.0), seed=0, max_iters=4000)
>>> res.success, round(res.cost, 2)  # doctest: +SKIP
(True, 13.9)
"""
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PlanResult",
    "gamma_constant",
    "make_corridor_obstacles",
    "neighbor_radius",
    "plan_rrt",
    "plan_rrt_star",
    "segment_free",
    "segments_free_batch",
    "steer",
]

#: RRT* 的 γ 常数的安全放大系数：Theorem 38 只要求严格大于下界
#: (2(1+1/d))^{1/d}(μ(C_free)/ζ_d)^{1/d}；有限样本下取 2 倍让邻域稍宽，
#: 重布线机会更多（渐近性质不受影响，常数口径见 :func:`gamma_constant`）。
_GAMMA_SAFETY = 2.0

#: motion validation 的默认采样间距，单位 m。取值口径：小于场景中最小
#: 障碍净距（教程 (3.10)：漏检概率 = max(0, 1 - w/Δ)）。
DEFAULT_EDGE_CHECK_DIST = 0.05


@dataclass
class PlanResult:
    """规划结果（树 + 路径）。

    Attributes:
        nodes: (n, 2) 全部树节点构型，单位 m；第 0 行恒为起点。
        parent: (n,) 各节点的父节点下标，根为 -1。
        node_costs: (n,) 各节点的代价 (3.5)（根到该节点的折线欧氏长）。
        cost: 路径代价（目标节点的 (3.5) 代价）；规划失败为 nan。
        path: (K, 2) 可行路径（起点 → 目标邻域）；规划失败为 None。
        success: 是否在 max_iters 内找到可行路径。
        goal_idx: 目标邻域内节点下标；失败为 -1。
        n_iters: 实际执行的采样轮数。
    """

    nodes: np.ndarray
    parent: np.ndarray
    node_costs: np.ndarray
    cost: float
    path: np.ndarray | None
    success: bool
    goal_idx: int
    n_iters: int


def segment_free(
    p: np.ndarray,
    q: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    max_check_dist: float = DEFAULT_EDGE_CHECK_DIST,
) -> bool:
    """线段 ``p→q`` 是否全程无碰撞（增量碰撞检测，教程 (3.4)）。

    按 ``max_check_dist`` 的间距把线段切成若干检查点（motion validation，
    教程 03.3），对每个检查点与每个圆做精确"点-圆"距离判定。这是 (3.10)
    的离散化口径：检查点间距 Δ 下，长度 ≥ Δ 的障碍交叠段必被检出；
    调用方应保证 Δ 小于场景最小净距。

    Args:
        p: (2,) 线段起点，单位 m。
        q: (2,) 线段终点，单位 m。
        obstacle_list: 圆域障碍列表，每项 ``(center (2,), radius)``，单位 m。
        max_check_dist: 检查点间距，单位 m，必须 > 0。

    Returns:
        True 表示线段全部检查点都在所有障碍圆之外（含端点）。
    """
    return bool(
        segments_free_batch(
            np.asarray(p, dtype=float)[None, :],
            np.asarray(q, dtype=float)[None, :],
            obstacle_list,
            max_check_dist,
        )[0]
    )


def segments_free_batch(
    p_arr: np.ndarray,
    q_arr: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]] | np.ndarray,
    max_check_dist: float = DEFAULT_EDGE_CHECK_DIST,
) -> np.ndarray:
    """:func:`segment_free` 的批量版：一次判定 k 条线段（RRT* 邻域选父/重布线用）。

    Args:
        p_arr: (k, 2) 各线段起点，单位 m。
        q_arr: (k, 2) 各线段终点，单位 m。
        obstacle_list: 圆域障碍列表（每项 ``(center (2,), radius)``），或
            预处理的 (M, 3) 数组 ``[cx, cy, r²]``（内部热路径用，省去逐项
            asarray）。
        max_check_dist: 检查点间距，单位 m（按批内最长边布点，短边多查
            的内点仍在该边线段上，不改变判定结果）。

    Returns:
        (k,) 布尔数组：True = 对应线段无碰撞。
    """
    if max_check_dist <= 0.0:
        raise ValueError(f"max_check_dist 必须为正，当前 {max_check_dist}")
    p_arr = np.asarray(p_arr, dtype=float).reshape(-1, 2)
    q_arr = np.asarray(q_arr, dtype=float).reshape(-1, 2)
    k = p_arr.shape[0]
    diff = q_arr - p_arr  # (k, 2)
    lengths = np.sqrt((diff * diff).sum(axis=1))  # (k,)
    n_checks = int(np.ceil(lengths.max(initial=0.0) / max_check_dist)) + 1
    ts = np.linspace(0.0, 1.0, max(n_checks, 2))
    # (k, c, 2) 全部检查点（依据：等距采样的 motion validation，(3.10)）。
    pts = p_arr[:, None, :] + ts[None, :, None] * diff[:, None, :]
    free = np.ones(k, dtype=bool)
    obs = (
        obstacle_list
        if isinstance(obstacle_list, np.ndarray)
        else np.asarray(
            [
                [c[0], c[1], r * r]
                for c, r in (
                    (np.asarray(c, dtype=float).reshape(2), r) for c, r in obstacle_list
                )
            ]
        ).reshape(-1, 3)
    )
    for row in obs:
        d2 = (pts - row[:2]) ** 2
        d2 = d2[..., 0] + d2[..., 1]  # (k, c)
        # 严格小于半径判碰撞（相切不碰）；用距离平方比较避免开方。
        free &= ~np.any(d2 < row[2], axis=1)
    return free


def steer(near: np.ndarray, rand: np.ndarray, eta: float) -> np.ndarray:
    """限步长转向（教程 (3.3)）：从 ``near`` 朝 ``rand`` 方向至多走 η。

    Args:
        near: (2,) 最近树节点构型，单位 m。
        rand: (2,) 采样构型，单位 m。
        eta: 生长步长上限，单位 m，必须 > 0。

    Returns:
        (2,) 新节点构型 ``q_new = near + min(η, ‖rand-near‖)·(rand-near)/‖rand-near‖``。
    """
    if eta <= 0.0:
        raise ValueError(f"步长上限 eta 必须为正，当前 {eta}")
    diff = np.asarray(rand, dtype=float) - np.asarray(near, dtype=float)
    dist = float(np.linalg.norm(diff))
    if dist < 1e-12:  # 采样点与最近节点重合：新节点即该点（避免除零）。
        return np.asarray(near, dtype=float).copy()
    # 依据：(3.3) steer 定义——方向取单位向量，步长取 min(ε, 距离)。
    return np.asarray(near, dtype=float) + min(eta, dist) * (diff / dist)


def gamma_constant(
    bounds: tuple[float, float, float, float],
    obstacle_list: list[tuple[np.ndarray, float]],
    d: int = 2,
) -> float:
    """RRT* 收缩半径的 γ 常数（教程 (3.7)，Theorem 38 下界的放大版）。

    r(n) = min{ γ (log n / n)^{1/d}, η }，其中 γ 必须满足
    γ > (2(1+1/d))^{1/d} · (μ(C_free)/ζ_d)^{1/d}（Theorem 38；教程 (3.7)
    行内注）。本实现取下界的 :data:`_GAMMA_SAFETY` 倍：严格大于下界的
    要求在有限样本下更稳（邻域稍宽、重布线更充分；渐近最优性不受影响）。

    Args:
        bounds: (xmin, xmax, ymin, ymax) 构型空间边界，单位 m。
        obstacle_list: 圆域障碍列表，单位 m（用于从边界面积中扣除 μ(C_obs)）。
        d: 构型空间维数，默认 2。

    Returns:
        γ 常数，无量纲（量纲随 d 折算进半径公式）。
    """
    xmin, xmax, ymin, ymax = bounds
    mu_free = max((xmax - xmin) * (ymax - ymin), 1e-9)
    for center, radius in obstacle_list:
        # 依据：μ(C_free) = μ(C) - Σμ(C_obs)（障碍不相交假设；重叠时该式
        # 低估自由体积 → γ 稍大 → 邻域稍宽，安全侧误差）。
        mu_free = max(mu_free - np.pi * radius * radius, 1e-9)
    zeta_d = np.pi if d == 2 else None  # ζ_2 = π：2 维单位球（圆）面积
    if zeta_d is None:
        raise NotImplementedError("教学实现只支持 d = 2")
    base = (2.0 * (1.0 + 1.0 / d)) ** (1.0 / d) * (mu_free / zeta_d) ** (1.0 / d)
    return float(_GAMMA_SAFETY * base)


def neighbor_radius(n: int, gamma: float, eta: float, d: int = 2) -> float:
    """第 n 个样本的 RRT* 邻域半径（教程 (3.7)）。

    r(n) = min{ γ (log n / n)^{1/d}, η }：收缩速率恰卡在 (log n / n)^{1/d}
    ——比它快则覆盖失败（解次优）、比它慢则碰撞检测调用数不降（Lemma 42），
    折中推导见教程 03.2 第三段。

    Args:
        n: 当前样本数（树节点数），n ≥ 2。
        gamma: :func:`gamma_constant` 给出的 γ 常数。
        eta: 步长上限（半径的上截断），单位 m。
        d: 构型空间维数。

    Returns:
        邻域半径，单位 m。
    """
    if n < 2:
        return float(eta)  # log(1)=0 会使半径为 0：树只有根时无邻域可言
    # 依据：(3.7) —— min{γ (log n / n)^{1/d}, η}。
    return float(min(gamma * (np.log(n) / n) ** (1.0 / d), eta))


def make_corridor_obstacles() -> list[tuple[np.ndarray, float]]:
    """构造基准"走廊"障碍图：10x10 场地中间一道带缺口的墙。

    本场景是 demo 与各测试共用的基准环境：墙上一条对角可穿的缺口，
    足以让 RRT/RRT* 展示绕障、iLQR 展示走廊内平滑、CEM-MPC 展示
    障碍惩罚下的滚动到达。

    Returns:
        5 个圆域障碍，每项 ``(center (2,), radius)``，单位 m；
        墙体沿 x = 5 分布，缺口在 y ∈ (5.2, 6.8)。
    """
    # 沿 x=5 的圆链墙（缺口在 y=6 附近）：圆心 x 全为 5。
    wall = [(5.0, y) for y in (0.4, 2.2, 4.0, 7.8, 9.6)]
    return [(np.array(c), 1.0) for c in wall]


def _extract_path(parent: np.ndarray, goal_idx: int) -> np.ndarray:
    """从目标节点沿 parent 回溯到根，返回 (K, 2) 起点→目标的路径。

    Args:
        parent: (n,) 父节点下标数组，根为 -1。
        goal_idx: 目标节点下标。

    Returns:
        (K, 2) 路径（第 0 行为根/起点，末行为目标节点）。
    """
    seq: list[int] = []
    j = goal_idx
    while j != -1:  # 依据：parent 链以根的 -1 结束
        seq.append(int(j))
        j = int(parent[j])
    seq.reverse()  # 回溯得到的是目标→起点，反转成起点→目标
    return np.asarray(seq, dtype=int)


def _plan_core(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    bounds: tuple[float, float, float, float],
    seed: int,
    max_iters: int,
    eta: float,
    goal_bias: float,
    goal_radius: float,
    optimal: bool,
) -> PlanResult:
    """:func:`plan_rrt` / :func:`plan_rrt_star` 共享的主循环。

    两种模式共用同一随机数消费顺序（每轮恰好一次 goal-bias 判定 +
    一次均匀采样），保证同 seed 下 RRT 与 RRT* 生成完全相同的节点序列，
    两者的代价可比（tests/test_rrt.py 的最优性对照依赖这一点）。
    ``optimal=True`` 时启用 RRT* 的选父 + 重布线（(3.5)–(3.7)）。
    """
    xmin, xmax, ymin, ymax = bounds
    start = np.asarray(start, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(2)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"bounds 必须满足 xmin<xmax, ymin<ymax，当前 {bounds}")
    if not 0.0 <= goal_bias <= 1.0:
        raise ValueError(f"goal_bias 必须在 [0,1]，当前 {goal_bias}")
    # 树存储：预分配倍增缓冲（nodes_buf[:n] 为有效区），parent 用列表、
    # 代价用同步倍增的 numpy 缓冲（热路径避免逐轮 list→array 转换）。
    nodes_buf = np.empty((1024, 2))
    nodes_buf[0] = start
    n_nodes = 1
    parent = [-1]
    costs_buf = np.zeros(1024)
    children: dict[int, list[int]] = {}  # 父下标 → 子下标表（重布线子树传播用）
    # 障碍预处理成 (M, 3) 的 [cx, cy, r²]（碰撞检测热路径的公共开销）。
    obs_arr = np.asarray(
        [
            [np.asarray(c, dtype=float).reshape(2)[0],
             np.asarray(c, dtype=float).reshape(2)[1],
             float(r) * float(r)]
            for c, r in obstacle_list
        ],
        dtype=float,
    ).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    gamma = gamma_constant(bounds, obstacle_list)  # RRT* 的 γ（Theorem 38）
    goal_idx = -1
    n_iters_done = 0

    for it in range(max_iters):
        n_iters_done = it + 1
        # ① 采样 q_rand：goal bias（教程 (3.11)）或均匀采样。
        if rng.random() < goal_bias:
            q_rand = goal.copy()
        else:
            q_rand = np.array(
                [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]
            )
        # ② 最近邻（Voronoi 偏置，教程 03.1）+ steer 限步长（(3.3)）。
        view = nodes_buf[:n_nodes]
        d_rand = view - q_rand[None, :]
        i_near = int(np.argmin(np.sqrt((d_rand * d_rand).sum(axis=1))))
        q_near = view[i_near]
        diff_near = q_rand - q_near
        dist_near = float(np.sqrt(diff_near @ diff_near))
        step = eta if dist_near > eta else dist_near
        q_new = (
            q_near if dist_near < 1e-12 else q_near + step * diff_near / dist_near
        )  # 依据：(3.3) steer——方向取单位向量，步长取 min(ε, 距离)
        # ③ 增量碰撞检测（(3.4)）：最近邻边不可行则跳过本轮。
        if not segment_free(q_near, q_new, obstacle_list):
            continue
        if optimal:
            # ④a RRT* 选父：邻域（(3.7)）内代价最小且可连的节点作父。
            r_n = neighbor_radius(n_nodes, gamma, eta)
            diff = view - q_new[None, :]
            d_new = np.sqrt((diff * diff).sum(axis=1))  # (n,)
            cand = np.nonzero(d_new <= r_n)[0]
            if cand.size == 0:
                # 半径 (3.7) 收缩到比 steer 距离还小的时候，q_new 可能落在
                # 所有邻域之外——退回最近邻作父（其边已过碰撞检测），不重布线。
                i_best = i_near
                best_cost = float(costs_buf[i_near]) + float(d_new[i_near])
                rewire_idx: list[int] = []
            else:
                cand_cost = costs_buf[cand] + d_new[cand]  # (k,) 候选总代价
                feasible = segments_free_batch(
                    view[cand], np.repeat(q_new[None, :], len(cand), axis=0),
                    obs_arr,
                )
                # 依据：(3.5) 的候选比较——可行者中总代价最小者当选。
                cand_cost = np.where(feasible, cand_cost, np.inf)
                i_best = int(cand[int(np.argmin(cand_cost))])
                best_cost = float(cand_cost.min())
                # ④b RRT* 重布线：邻域内经 q_new 更便宜的节点改接（(3.6) 单调性）。
                # 先按"经新节点的总代价"找出所有潜在改善者，再一次性批量检边。
                through_new = best_cost + d_new  # 经 q_new 到每个邻域节点的代价
                rewire_idx = [
                    int(i_v)
                    for i_v in cand
                    if int(i_v) != i_best
                    and through_new[int(i_v)] < costs_buf[int(i_v)] - 1e-12
                ]
                if rewire_idx:
                    ok = segments_free_batch(
                        np.repeat(q_new[None, :], len(rewire_idx), axis=0),
                        view[np.asarray(rewire_idx)],
                        obs_arr,
                    )
                    rewire_idx = [i_v for i_v, f in zip(rewire_idx, ok) if f]
            # —— 挂树（缓冲区按需倍增）。
            if n_nodes == len(nodes_buf):
                nodes_buf = np.vstack([nodes_buf, np.empty_like(nodes_buf)])
                costs_buf = np.concatenate([costs_buf, np.zeros_like(costs_buf)])
            nodes_buf[n_nodes] = q_new
            i_new = n_nodes
            n_nodes += 1
            parent.append(i_best)
            costs_buf[i_new] = best_cost
            children.setdefault(i_best, []).append(i_new)
            for i_v in rewire_idx:
                old_parent = parent[i_v]
                children[old_parent].remove(i_v)  # 从旧父的子表摘除
                parent[i_v] = i_new  # 改接父节点
                children.setdefault(i_new, []).append(i_v)
                # 依据：(3.5) 的递归定义——祖先代价下降须逐级传给子树，
                # 否则后续选父/重布线的比较基于过期的代价。
                stack = [i_v]
                while stack:
                    j = stack.pop()
                    pj = parent[j]
                    djp = nodes_buf[j] - nodes_buf[pj]
                    costs_buf[j] = costs_buf[pj] + float(np.sqrt(djp @ djp))
                    stack.extend(children.get(j, []))
        else:
            # ④ RRT：直接以最近邻为父挂树（教程 03.1 算法骨架）。
            if n_nodes == len(nodes_buf):
                nodes_buf = np.vstack([nodes_buf, np.empty_like(nodes_buf)])
                costs_buf = np.concatenate([costs_buf, np.zeros_like(costs_buf)])
            nodes_buf[n_nodes] = q_new
            i_new = n_nodes
            n_nodes += 1
            parent.append(i_near)
            # 新边长恰为 steer 的实际步长 min(η, 距离)。
            costs_buf[i_new] = costs_buf[i_near] + step
        # ⑤ 目标判定：新节点落入目标邻域（半径 goal_radius 的圆）。
        if goal_idx == -1 and float(np.linalg.norm(q_new - goal)) <= goal_radius:
            goal_idx = i_new
            if not optimal:
                break  # RRT 取第一条可行解即返回；RRT* 继续迭代让代价下降

    idx_seq = (
        _extract_path(np.asarray(parent), goal_idx) if goal_idx != -1 else None
    )
    return PlanResult(
        nodes=nodes_buf[:n_nodes].copy(),
        parent=np.asarray(parent, dtype=int),
        node_costs=costs_buf[:n_nodes].copy(),
        cost=float(costs_buf[goal_idx]) if goal_idx != -1 else float("nan"),
        path=nodes_buf[:n_nodes][idx_seq].copy() if idx_seq is not None else None,
        success=goal_idx != -1,
        goal_idx=goal_idx,
        n_iters=n_iters_done,
    )


def plan_rrt(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    bounds: tuple[float, float, float, float],
    seed: int = 0,
    max_iters: int = 4000,
    eta: float = 0.5,
    goal_bias: float = 0.05,
    goal_radius: float = 0.3,
) -> PlanResult:
    """2D RRT 规划（LaValle TR 98-11；教程 03.1 算法骨架）。

    快速探索随机树：最近邻 + steer (3.3) + 增量碰撞检测 (3.4)，返回
    第一条伸入目标邻域的可行路径（不保证最优——教程 03.2 的问题场景）。

    Args:
        start: (2,) 起点构型，单位 m，须在 bounds 内且无碰撞。
        goal: (2,) 目标点，单位 m。
        obstacle_list: 圆域障碍列表，每项 ``(center (2,), radius)``，单位 m。
        bounds: (xmin, xmax, ymin, ymax) 采样边界，单位 m。
        seed: 随机种子（确定性：同 seed 同结果）。
        max_iters: 最大采样轮数。
        eta: steer 步长上限（(3.3) 的 ε），单位 m。
        goal_bias: goal bias 概率 p_g（(3.11)：期望等待 1/p_g 轮）。
        goal_radius: 目标邻域半径，单位 m。

    Returns:
        :class:`PlanResult`（success=False 时 path 为 None）。
    """
    return _plan_core(
        start, goal, obstacle_list, bounds, seed, max_iters,
        eta, goal_bias, goal_radius, optimal=False,
    )


def plan_rrt_star(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    bounds: tuple[float, float, float, float],
    seed: int = 0,
    max_iters: int = 4000,
    eta: float = 0.5,
    goal_bias: float = 0.05,
    goal_radius: float = 0.3,
) -> PlanResult:
    """2D RRT* 规划（Karaman & Frazzoli 2011, Theorem 38；教程 03.2）。

    RRT + 选父 + 重布线：邻域半径按 (3.7) 收缩，节点代价 (3.5) 逐节点
    单调不增 (3.6)——样本数 → ∞ 时解以概率 1 收敛到最优（渐近最优）。
    与 :func:`plan_rrt` 同 seed 时生成相同的节点序列（随机数消费顺序
    一致），故两者的解代价可直接对照。

    Args:
        参数与 :func:`plan_rrt` 完全一致。

    Returns:
        :class:`PlanResult`；``cost[goal_idx]`` 即最终路径代价。
    """
    return _plan_core(
        start, goal, obstacle_list, bounds, seed, max_iters,
        eta, goal_bias, goal_radius, optimal=True,
    )

"""mpnet_lite：MPNet 的学习式采样偏置规划（教学代理）。

函数流水线
----------
Qureshi, Simeonov, Bency, Yip, *Motion Planning Networks*, ICRA 2019
（本地 papers/control_planning/frontier/arXiv-1806.05767_MPNet.pdf；
Enet 的重建损失为论文式 (1)、Pnet 的 MSE 蒸馏为式 (2)、在线双向规划为
Algorithm 2、两级重规划为 Algorithm 3）与教程第 09 章 §09.1（采样分布
(9.1)、蒸馏目标 (9.2)、steerTo (9.3)、完备性继承 (9.4)）的教学代理，
逐式精读见 tutorials/control_planning/精读/MPNet_ICRA2019.md。设定为
2D 圆障碍环境中的点机器人（构型空间与碰撞判定口径同 :mod:`rrt`）。

核心思想与原文一致——**把采样偏置从"无信息的 goal bias (3.11)"升级为
"看得到障碍与目标"的网络**：输入 = (当前点, 目标点, 周围障碍的粗 SDF
编码)，输出 = 建议的下一采样点偏移；训练数据 = 随机环境里用
:func:`rrt.plan_rrt` 解出的路径中间点（在线生成、少量），损失即 (9.2)
的逐步 MSE 蒸馏；推理时以概率 p 用网络建议点代替 RRT 的均匀采样。
这一"学到的采样器喂给经典规划器"的用法取自原文 §VII-A 的自述（dropout
随机性"可直接为采样式规划器生成自适应样本"，即 neural RRT* [26] 谱系；
更早的学习采样分布路线为 Ichter et al., ICRA 2018，原文引文 [16]）。

与原文的差异（教学代理的简化边界，逐条声明）：

1. **环境编码**：原文 Enet 是点云自编码器（式 (1)，收缩正则、可泛化到
   未见工作空间）→ 本代理用固定粗 SDF 网格查表（:func:`sdf_grid`），
   无可学习编码器——"环境 → 向量"的接口形状保留，学习成分只留 Pnet；
2. **网络与随机化**：原文深层 PyTorch 网络 + dropout 隐式采样（(9.1)
   的分布来源）→ 单隐层 tanh MLP 的手写前向/反向（不依赖 torch），
   随机性改为"建议点 = 条件均值 + 高斯抖动"的显式采样器（原文 §IV-B.2
   "dropout 让输出在条件均值附近抖动"的可控版）与"以概率 p 混合均匀
   采样"共同提供；
3. **训练数据**：原文每类环境 110 个工作空间 × 5000 条 RRT* 专家路径的
   大规模离线蒸馏 → 本代理在线生成 40 个随机环境的 :func:`rrt.plan_rrt`
   首解路径（弧长重采样为等距中间点后逐步模仿）；
4. **在线规划**：原文 Algorithm 2 双向采样–连接循环 + Algorithm 3 两级
   重规划（神经重修复 / 回退 RRT*）→ 单树 RRT 的采样偏置，完整混合
   规划器不做——故 (9.4) 的完备性继承论证在此不重做（偏置支只是把
   均匀采样换成同为全空间支撑的混合分布，概率完备性平凡保留）。教学
   代理的条件节点取"树上离目标最近的节点"——Algorithm 2 每轮扩展树端
   （τ 的 end）在有效树（拒绝碰撞边）上的对应物；
5. **维度**：2D 点机器人（原文覆盖 2D/3D 点质量、刚体与 7-DOF Baxter）。

调用链：:func:`sdf_grid`（障碍 → 粗 SDF 编码，替代 Enet 的环境编码）→
:func:`make_random_env` + :func:`rrt.plan_rrt` + :func:`resample_path_by_arclength`
→ :func:`build_dataset`（在线专家数据 (9.2) 的 (x, y) 对）→
:func:`mse_loss_grad`（(9.2) 损失的手写前向/反向）→ :func:`train_mpnet`
（Adam 外层循环）→ :func:`plan_rrt_biased`（偏置采样 RRT 主循环：复用
:func:`rrt.steer` 的限步长 (3.3) 与 :func:`rrt.segment_free` 的增量碰撞
检测 (3.4)）。评估经 ``metrics``（由 ``tests/test_mpnet.py`` 驱动；metrics
按路径加载，见该文件头部注释）。

手写反向的梯度链（N 个样本、隐藏层宽 H、tanh 隐激活）：

- 前向：``z1 = x·W1 + b1``、``a1 = tanh(z1)``、``ŷ = a1·W2 + b2``；
- (9.2) 的损失 ``L = mean((ŷ - y)²)``（对全部 N·2 个分量取均值）→
  ``∂L/∂ŷ = (ŷ - y)/N``；
- 输出层线性：``∂L/∂W2 = a1ᵀ·∂L/∂ŷ``、``∂L/∂b2 = Σ ∂L/∂ŷ``；
- 隐层 tanh：``∂L/∂a1 = ∂L/∂ŷ·W2ᵀ``、``∂L/∂z1 = ∂L/∂a1 ⊙ (1 - a1²)``
  （tanh 导数 1 - tanh²）→ ``∂L/∂W1 = xᵀ·∂L/∂z1``、``∂L/∂b1 = Σ ∂L/∂z1``；
- 优化器 Adam（Kingma & Ba 2015；与 ``ppo_lite`` 同一套约定，但本模块
  自包含、不 import 库内其他学习模块）。

约定
----
- 构型 ``q = (x, y)``，单位 m；障碍为圆域列表 ``[(center (2,), radius)]``，
  口径与 :mod:`rrt` 完全一致（点机器人、点到圆心距离严格小于半径判碰、
  相切不碰）。
- MLP 输入布局 = ``(当前点/s, 目标点/s, SDF 编码)``：位置分量按工作空间
  对角线 s 归一化到 O(1)（网络数值卫生），编码维度 = k²（:func:`make_inputs`
  负责装配，全网唯一布局出处）。
- 确定性：所有随机性走 ``np.random.default_rng(seed)``（环境采样、专家
  规划、偏置规划各自计流），同 seed 逐位可复现；网络前向本身无随机性
  （差异 2：原文的随机性来自 dropout，本代理来自采样混合）。
- 数值：float64 全程。

示例
----
>>> import numpy as np
>>> from mpnet_lite import plan_rrt_biased, make_random_env, train_mpnet
>>> params, hist = train_mpnet(seed=0)          # 在线数据 + 训练（约数秒）
>>> start, goal, obstacles = make_random_env(seed=101)
>>> res = plan_rrt_biased(start, goal, obstacles, (0.0, 10.0, 0.0, 10.0),
...                       params, p_bias=0.5, seed=0)
>>> res.success  # doctest: +SKIP
True
"""
from dataclasses import dataclass

import numpy as np

from rrt import PlanResult, plan_rrt, segment_free, steer

__all__ = [
    "MLPParams",
    "build_dataset",
    "make_inputs",
    "make_random_env",
    "mlp_forward",
    "mse_loss",
    "mse_loss_grad",
    "plan_rrt_biased",
    "points_free",
    "propose_points",
    "resample_path_by_arclength",
    "sdf_grid",
    "train_mpnet",
]


# ------------------------------------------------------------------ 编码 --
def sdf_grid(
    obstacle_list: list[tuple[np.ndarray, float]],
    bounds: tuple[float, float, float, float],
    k: int = 16,
) -> np.ndarray:
    """障碍的粗 SDF 网格编码（替代 MPNet Enet 的环境编码，差异 1）。

    把 bounds 均分为 k×k 个格子，取每格中心处的"带符号净距"：
    ``d(c) = min_j (‖c - o_j‖ - r_j)``（圆域障碍下到自由边界的精确有符号
    距离；空障碍为 +∞），再除以工作空间对角线的 1/4 并截断到 [-1, 1]。
    负值 = 格心在障碍内，正值 = 净距（≥ 1 表示"远场"被截断饱和）。这是
    论文"环境 → 隐向量 Z"接口的固定特征版：无参数、不可学习，但保留了
    "障碍在哪、多近"的信息，供 MLP 从中读出局部绕行方向。

    Args:
        obstacle_list: 圆域障碍列表，每项 ``(center (2,), radius)``，单位 m。
        bounds: (xmin, xmax, ymin, ymax) 工作空间边界，单位 m。
        k: 每边格子数（编码维度 = k²），必须 ≥ 2。

    Returns:
        (k²,) float64 编码向量，取值 [-1, 1]；展平顺序 = 行主序
        （y 为外层、x 为内层，:func:`make_inputs` 与全网共用同一定义）。
    """
    if k < 2:
        raise ValueError(f"网格数 k 必须 ≥ 2，当前 {k}")
    xmin, xmax, ymin, ymax = bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"bounds 必须满足 xmin<xmax, ymin<ymax，当前 {bounds}")
    # 格心坐标（半格偏移，避免格心落在边界上）。
    xs = xmin + (np.arange(k) + 0.5) * (xmax - xmin) / k
    ys = ymin + (np.arange(k) + 0.5) * (ymax - ymin) / k
    gx, gy = np.meshgrid(xs, ys)  # (k, k) ×2，行主序展平后 y 外层
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (k², 2) 格心
    if len(obstacle_list) == 0:
        return np.ones(k * k)  # 无障碍：处处"远场自由"
    # 依据：圆域障碍下带符号距离 = min_j (‖c - o_j‖ - r_j)。
    d = np.full(k * k, np.inf)
    for center, radius in obstacle_list:
        c = np.asarray(center, dtype=float).reshape(2)
        dist = np.sqrt(np.sum((cells - c) ** 2, axis=1)) - float(radius)
        d = np.minimum(d, dist)
    # 归一化尺度：对角线的 1/4——近场（约 3.5 m @ 10×10 地图）线性保留，
    # 远场截断到 +1（编码容量集中在障碍附近，正是绕行决策需要的区域）。
    scale = 0.25 * float(np.hypot(xmax - xmin, ymax - ymin))
    return np.clip(d / scale, -1.0, 1.0)


def points_free(
    pts: np.ndarray, obstacle_list: list[tuple[np.ndarray, float]]
) -> np.ndarray:
    """逐点判定是否落在自由空间（点机器人口径，与 :mod:`rrt` 一致）。

    Args:
        pts: (N, 2) 待判定点，单位 m。
        obstacle_list: 圆域障碍列表，每项 ``(center (2,), radius)``，单位 m。

    Returns:
        (N,) 布尔数组：True = 该点与所有障碍圆的距离 ≥ 半径（相切不碰）。
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    free = np.ones(pts.shape[0], dtype=bool)
    for center, radius in obstacle_list:
        c = np.asarray(center, dtype=float).reshape(2)
        d2 = np.sum((pts - c) ** 2, axis=1)
        # 依据：rrt 的碰撞口径——距离严格小于半径判碰，相切不碰。
        free &= d2 >= float(radius) ** 2
    return free


# ------------------------------------------------------------ 环境与数据 --
def make_random_env(
    seed: int = 0,
    bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
    n_obsts: tuple[int, int] = (2, 5),
    r_range: tuple[float, float] = (0.8, 1.5),
    margin: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, float]]]:
    """采样一个随机训练/测试环境（起点、目标、2–5 个圆障碍）。

    障碍圆心取场地中部（离边界 ≥ 1.5 m），半径在 ``r_range`` 内均匀取值；
    起点取自左下角 2×2 方块、目标取自右上角 2×2 方块，均拒绝采样到与
    全部障碍净距 ≥ ``margin`` 的自由位置——保证起终点本身可行（同 (3.1)
    契约的 ``q_init, q_goal ∈ C_free``），而起终点连线是否被挡交给随机。

    Args:
        seed: 随机种子（确定性：同 seed 同环境）。
        bounds: 工作空间边界，单位 m。
        n_obsts: 障碍个数的 [下限, 上限]（均匀整数，含端点）。
        r_range: 障碍半径的 [下限, 上限]，单位 m。
        margin: 起终点与障碍的最小净距，单位 m。

    Returns:
        (start (2,), goal (2,), obstacles)——与 :mod:`rrt` 各函数的入参口径一致。
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    n = int(rng.integers(n_obsts[0], n_obsts[1] + 1))
    obstacles: list[tuple[np.ndarray, float]] = []
    for _ in range(n):
        c = np.array(
            [
                rng.uniform(xmin + 1.5, xmax - 1.5),
                rng.uniform(ymin + 1.5, ymax - 1.5),
            ]
        )
        obstacles.append((c, float(rng.uniform(r_range[0], r_range[1]))))

    def _free_point(lo: float, hi: float) -> np.ndarray:
        for _ in range(200):  # 拒绝采样；默认参数下命中率高，200 次必够
            p = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)])
            # 净距 ≥ r + margin（蕴含自由，且留 margin 安全余量，同 (3.1) 的
            # q_init, q_goal ∈ C_free 契约）。
            if all(np.linalg.norm(p - c) >= r + margin for c, r in obstacles):
                return p
        raise RuntimeError("拒绝采样超限：margin 过大或障碍过多，请放宽参数")

    start = _free_point(xmin + 0.5, xmin + 2.5)
    goal = _free_point(xmax - 2.5, xmax - 0.5)
    return start, goal, obstacles


def resample_path_by_arclength(path: np.ndarray, stride: float) -> np.ndarray:
    """把折线路径按弧长等距重采样（专家数据的等距中间点）。

    :func:`rrt.plan_rrt` 的路径点间距由 steer 步长与目标邻域决定、疏密
    不均；按弧长每 ``stride`` 取一点后，相邻中间点的**弧长间隔**恒定，
    折线偏移（弦长）模长不超过 stride——直线段处恰为 stride，弯折段因
    "弦长 < 弧长"而略短。(9.2) 的回归目标因此量纲有界：网络不必学
    "步幅"（推理时偏移还会被 :func:`propose_points` 按 clamp 截断），
    只需学"往哪个方向采样"。

    Args:
        path: (M, 2) 折线路径（M ≥ 2），单位 m。
        stride: 采样弧长间距，单位 m，必须 > 0。

    Returns:
        (K, 2) 重采样点：从 path[0] 起每隔 stride 一点，末点恒为 path[-1]。
    """
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if path.shape[0] < 2:
        raise ValueError(f"路径至少 2 个点，当前 {path.shape[0]}")
    if stride <= 0.0:
        raise ValueError(f"stride 必须为正，当前 {stride}")
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)  # (M-1,) 逐段长
    s = np.concatenate([[0.0], np.cumsum(seg)])  # (M,) 累计弧长
    total = float(s[-1])
    n_mid = int(np.floor(total / stride))
    targets = np.arange(n_mid + 1) * stride  # [0, stride, ..., n_mid·stride]
    if total - targets[-1] > 1e-9:
        targets = np.append(targets, total)  # 末点强制对齐 path[-1]
    # 依据：折线的弧长参数化逐段线性，np.interp 即精确取值。
    return np.column_stack(
        [np.interp(targets, s, path[:, 0]), np.interp(targets, s, path[:, 1])]
    )


def make_inputs(
    points: np.ndarray,
    goals: np.ndarray,
    enc: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """装配 MLP 输入批（全网唯一的输入布局出处）。

    布局 = ``(当前点/s, 目标点/s, SDF 编码)``：位置分量除以工作空间对角线
    s（归一化到 O(1)，tanh 隐层不饱和）；``goals``/``enc`` 允许传单份
    （同一目标/同一环境编码广播到批内全部点）。

    Args:
        points: (N, 2) 条件点（当前构型），单位 m。
        goals: (2,) 或 (N, 2) 目标点，单位 m。
        enc: (k²,) 或 (N, k²) SDF 编码（:func:`sdf_grid` 的输出）。
        bounds: 工作空间边界（只用于算归一化尺度 s）。

    Returns:
        (N, 4 + k²) float64 输入批。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    gl = np.asarray(goals, dtype=float).reshape(-1, 2)
    if gl.shape[0] == 1 and n > 1:
        gl = np.repeat(gl, n, axis=0)
    e = np.asarray(enc, dtype=float)
    if e.ndim == 1:
        e = e[None, :]
    if e.shape[0] == 1 and n > 1:
        e = np.repeat(e, n, axis=0)
    if gl.shape[0] != n or e.shape[0] != n:
        raise ValueError(
            f"批大小不一致：points {n}, goals {gl.shape[0]}, enc {e.shape[0]}"
        )
    xmin, xmax, ymin, ymax = bounds
    scale = float(np.hypot(xmax - xmin, ymax - ymin))
    return np.hstack([pts / scale, gl / scale, e])


def build_dataset(
    seed: int = 0,
    n_envs: int = 40,
    stride: float = 1.1,
    k: int = 16,
    bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
    max_iters: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """在线生成 (9.2) 的蒸馏数据集（差异 3：少量在线专家替代大规模离线）。

    逐个采样随机环境（:func:`make_random_env`）→ 用 :func:`rrt.plan_rrt`
    解一条可行路径（即"专家"，原文用 RRT* 近优路径，此处取首解即可演示
    机制）→ 弧长重采样 → 相邻中间点对给出一行数据：

    - 输入 = :func:`make_inputs`(当前中间点, goal, 环境编码)；
    - 目标 = 下一中间点 − 当前中间点（该往哪采样；弦长 ≤ stride）。

    Args:
        seed: 总种子（环境与规划共用一个随机源，同 seed 逐位复现）。
        n_envs: 尝试的环境数；规划失败的环境跳过（info 中计数）。
        stride: 弧长重采样间距，单位 m。
        k: SDF 编码每边格子数。
        bounds: 工作空间边界，单位 m。
        max_iters: 专家规划的单环境迭代预算。

    Returns:
        (x (S, 4+k²), y (S, 2), info)：输入批、目标偏移批与诊断 dict
        （``n_envs / n_solved / n_samples / stride``）。全部环境失败时
        抛 RuntimeError（默认参数下不会发生）。
    """
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_solved = 0
    for _ in range(n_envs):
        start, goal, obstacles = make_random_env(seed=int(rng.integers(2**31)))
        res = plan_rrt(
            start, goal, obstacles, bounds,
            seed=int(rng.integers(2**31)), max_iters=max_iters,
            eta=0.5, goal_bias=0.1,
        )
        if not res.success:
            continue
        n_solved += 1
        enc = sdf_grid(obstacles, bounds, k)
        pts = resample_path_by_arclength(res.path, stride)
        xs.append(make_inputs(pts[:-1], goal, enc, bounds))
        ys.append(pts[1:] - pts[:-1])  # (9.2)：专家的"下一步"偏移
    if n_solved == 0:
        raise RuntimeError("全部环境规划失败：请检查 max_iters 或障碍参数")
    info = {
        "n_envs": n_envs,
        "n_solved": n_solved,
        "n_samples": int(sum(len(a) for a in ys)),
        "stride": stride,
    }
    return np.vstack(xs), np.vstack(ys).reshape(-1, 2), info


# ------------------------------------------------------------------ MLP --
@dataclass
class MLPParams:
    """单隐层 tanh MLP 的参数（Pnet 的教学代理，差异 2）。

    前向：``y = tanh(x·W1 + b1)·W2 + b2``；输入布局见 :func:`make_inputs`，
    输出 (N, 2) = 建议的下一采样点偏移（(9.1) 的点预测版）。

    Attributes:
        W1: (d_in, H) 输入→隐层权重。
        b1: (H,) 隐层偏置。
        W2: (H, 2) 隐层→输出权重。
        b2: (2,) 输出偏置。
    """

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @staticmethod
    def init(d_in: int, hidden: int = 32, seed: int = 0, scale: float = 0.4) -> "MLPParams":
        """确定性初始化：小尺度高斯权重 + 零偏置（同 seed 逐位复现）。

        Args:
            d_in: 输入维数（= 4 + k²）。
            hidden: 隐层宽度 H。
            seed: 随机种子。
            scale: 权重初值标准差。

        Returns:
            :class:`MLPParams`。
        """
        rng = np.random.default_rng(seed)
        return MLPParams(
            W1=rng.normal(0.0, scale, size=(d_in, hidden)),
            b1=np.zeros(hidden),
            W2=rng.normal(0.0, scale, size=(hidden, 2)),
            b2=np.zeros(2),
        )


def mlp_forward(params: MLPParams, x: np.ndarray) -> np.ndarray:
    """MLP 前向（批）：x (N, d_in) → y (N, 2)，隐层 tanh。

    Args:
        params: 网络参数。
        x: (N, d_in) 输入批（:func:`make_inputs` 的输出）。

    Returns:
        (N, 2) 建议偏移批，单位 m。
    """
    x = np.asarray(x, dtype=float)
    a1 = np.tanh(x @ params.W1 + params.b1)  # (N, H)
    return a1 @ params.W2 + params.b2  # (N, 2)


def mse_loss(params: MLPParams, x: np.ndarray, y: np.ndarray) -> float:
    """(9.2) 蒸馏损失的标量版（tests 的中心差分对照用；梯度见 :func:`mse_loss_grad`）。"""
    y = np.asarray(y, dtype=float).reshape(-1, 2)
    return float(np.mean((mlp_forward(params, x) - y) ** 2))


def mse_loss_grad(
    params: MLPParams, x: np.ndarray, y: np.ndarray
) -> tuple[float, MLPParams]:
    """(9.2) 蒸馏损失的手写前向/反向。

    损失 ``L = mean((ŷ - y)²)``（对全部 N·2 个分量取均值），梯度链见模块
    docstring；正确性由 tests/test_mpnet.py 的中心差分对照钉住。

    Args:
        params: 网络参数。
        x: (N, d_in) 输入批。
        y: (N, 2) 目标偏移批（专家的下一步，单位 m）。

    Returns:
        (loss, grad)：标量损失与同布局的梯度对象（Adam 直接消费）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 2)
    # 前向（带缓存的展开版，供反向复用 a1）。
    z1 = x @ params.W1 + params.b1  # (N, H)
    a1 = np.tanh(z1)  # (N, H)
    y_hat = a1 @ params.W2 + params.b2  # (N, 2)
    diff = y_hat - y  # (N, 2)
    loss = float(np.mean(diff * diff))
    # 反向：∂L/∂ŷ = 2(ŷ - y) / (2N) = (ŷ - y)/N（分量均值口径）。
    g = diff / x.shape[0]  # (N, 2)
    grad_b2 = g.sum(axis=0)  # (2,)
    grad_W2 = a1.T @ g  # (H, 2)
    ga1 = g @ params.W2.T  # (N, H)
    gz1 = ga1 * (1.0 - a1 * a1)  # tanh 导数 1 - tanh²
    grad_b1 = gz1.sum(axis=0)  # (H,)
    grad_W1 = x.T @ gz1  # (d_in, H)
    return loss, MLPParams(grad_W1, grad_b1, grad_W2, grad_b2)


def _adam_step(
    param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, lr: float, t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单参数数组的 Adam 步（Kingma & Ba 2015；与 ppo_lite 同约定，自包含）。"""
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * grad * grad
    m_hat = m_new / (1.0 - beta1**t)
    v_hat = v_new / (1.0 - beta2**t)
    return param - lr * m_hat / (np.sqrt(v_hat) + adam_eps), m_new, v_new


def train_mpnet(
    seed: int = 0,
    n_envs: int = 40,
    epochs: int = 600,
    hidden: int = 32,
    lr: float = 0.01,
    stride: float = 1.1,
    k: int = 16,
    bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
) -> tuple[MLPParams, list[float]]:
    """训练偏置网络：在线数据 (9.2) + 全批量 Adam（教学规模）。

    超参为实测调定（seed=0）：损失从 ≈ 0.9（"瞎指路"的偏移方差量级）
    降到 ≈ 0.2 上下；测试预算（< 8 s）内取 20 环境 / 400 轮——数据生成
    与训练各占约一半时长。诊断 ``history`` 见 DEBUG.md 变量表 N-1。

    Args:
        seed: 总种子（数据与初始化共用，同 seed 逐位复现）。
        n_envs: 训练环境数（:func:`build_dataset`）。
        epochs: 全批量 Adam 轮数。
        hidden: 隐层宽度 H。
        lr: Adam 学习率。
        stride: 弧长重采样间距，单位 m。
        k: SDF 编码每边格子数。
        bounds: 工作空间边界，单位 m。

    Returns:
        (params, history)：训练后权重与损失曲线（长度 = epochs + 1：
        第 0 项为初始化参数的损失，其后为每轮更新**前**的损失，末项追加
        训练后的最终损失）。
    """
    x, y, _info = build_dataset(seed=seed, n_envs=n_envs, stride=stride, k=k, bounds=bounds)
    params = MLPParams.init(d_in=x.shape[1], hidden=hidden, seed=seed)
    # Adam 状态：每个参数数组一份（一阶/二阶矩）。
    state = {
        "W1": [np.zeros_like(params.W1), np.zeros_like(params.W1)],
        "b1": [np.zeros_like(params.b1), np.zeros_like(params.b1)],
        "W2": [np.zeros_like(params.W2), np.zeros_like(params.W2)],
        "b2": [np.zeros_like(params.b2), np.zeros_like(params.b2)],
    }
    history: list[float] = []
    for t in range(1, epochs + 1):
        loss, grad = mse_loss_grad(params, x, y)
        history.append(loss)
        w1, m1, v1 = _adam_step(params.W1, grad.W1, *state["W1"], lr, t)
        state["W1"] = [m1, v1]
        b1, m2, v2 = _adam_step(params.b1, grad.b1, *state["b1"], lr, t)
        state["b1"] = [m2, v2]
        w2, m3, v3 = _adam_step(params.W2, grad.W2, *state["W2"], lr, t)
        state["W2"] = [m3, v3]
        b2, m4, v4 = _adam_step(params.b2, grad.b2, *state["b2"], lr, t)
        state["b2"] = [m4, v4]
        params = MLPParams(w1, b1, w2, b2)
    history.append(mse_loss(params, x, y))  # 训练后最终损失
    return params, history


# ---------------------------------------------------------- 推理：偏置 RRT --
def propose_points(
    params: MLPParams,
    points: np.ndarray,
    goal: np.ndarray,
    enc: np.ndarray,
    bounds: tuple[float, float, float, float],
    clamp: float = 1.0,
) -> np.ndarray:
    """批量为条件点生成网络建议采样点（(9.1) 的点预测 + 模长截断）。

    建议点 = 条件点 + offset，offset 模长截断到 ``clamp``（网络只负责
    "指方向"，步幅交给调用方的 steer 语义）；出界建议裁回 bounds。

    Args:
        params: 训练后的偏置网络。
        points: (N, 2) 条件点（当前构型），单位 m。
        goal: (2,) 目标点，单位 m。
        enc: (k²,) 环境的 SDF 编码。
        bounds: 工作空间边界（裁剪出界建议），单位 m。
        clamp: 偏移模长上限，单位 m。

    Returns:
        (N, 2) 建议采样点，单位 m。
    """
    x = make_inputs(points, goal, enc, bounds)
    off = mlp_forward(params, x)  # (N, 2)
    nrm = np.linalg.norm(off, axis=1, keepdims=True)
    off = off * np.where(nrm > clamp, clamp / np.maximum(nrm, 1e-12), 1.0)
    prop = np.asarray(points, dtype=float).reshape(-1, 2) + off
    xmin, xmax, ymin, ymax = bounds
    return np.clip(prop, [xmin, ymin], [xmax, ymax])


def plan_rrt_biased(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    bounds: tuple[float, float, float, float],
    params: MLPParams | None,
    p_bias: float = 0.5,
    seed: int = 0,
    max_iters: int = 4000,
    eta: float = 0.5,
    goal_bias: float = 0.05,
    goal_radius: float = 0.3,
    k: int = 16,
    clamp: float = 1.0,
    jitter: float = 0.5,
) -> PlanResult:
    """带学习式采样偏置的 2D RRT（(9.1) 的教学代理用法）。

    主循环与 :func:`rrt.plan_rrt` 的 RRT 分支同构（最近邻 + steer (3.3) +
    增量碰撞检测 (3.4) + 目标邻域判定），唯一差异在采样分支：非 goal-bias
    轮以概率 ``p_bias`` 用网络建议点代替均匀采样——条件节点取当前树上
    **离目标最近**的节点（Algorithm 2 每轮扩展树端的对应物，见差异 4），
    网络在该节点处"看着障碍与目标"给出建议方向，再叠加高斯抖动（原文
    dropout 随机性的显式版，防止"同一方向反复撞墙"的局部卡死）。
    ``p_bias = 0`` 退化为均匀 RRT（同一段代码作对照臂，保证对比只差在
    偏置上）。

    每轮随机数消耗顺序（确定性口径，同 seed 逐位复现）：① ``rng.random()``
    goal-bias 判定；② 非 goal 轮 ``rng.random()`` 偏置判定；③ 偏置命中时
    ``rng.normal(0, jitter, 2)`` 的两个抖动 draw（条件节点取 argmin，无
    随机）；④ 均匀分支两次 ``rng.uniform``。消耗顺序与 :func:`rrt.plan_rrt`
    不同，故两函数不做逐轮对照、只做统计对照（tests/test_mpnet.py 的
    固定环境集平均迭代数）。

    Args:
        start: (2,) 起点构型，单位 m，须无碰撞。
        goal: (2,) 目标点，单位 m。
        obstacle_list: 圆域障碍列表，口径同 :mod:`rrt`。
        bounds: (xmin, xmax, ymin, ymax) 采样边界，单位 m。
        params: 训练后的偏置网络；``p_bias = 0`` 时可为 None。
        p_bias: 偏置概率（每轮用网络建议点代替均匀采样的概率），∈ [0, 1]。
        seed: 随机种子。
        max_iters: 最大采样轮数。
        eta: steer 步长上限（(3.3) 的 ε），单位 m。
        goal_bias: goal bias 概率（(3.11)，与 :func:`rrt.plan_rrt` 同义）。
        goal_radius: 目标邻域半径，单位 m。
        k: SDF 编码每边格子数（须与训练一致）。
        clamp: 网络偏移模长上限，单位 m（建议只指方向，步幅由 η 决定）。
        jitter: 建议点高斯抖动标准差，单位 m（0 关闭；对应原文 dropout
            的随机化职责——让重试逃出局部陷阱，§VII-A ③）。

    Returns:
        :class:`rrt.PlanResult`（与 :func:`rrt.plan_rrt` 同一结果结构）。

    Raises:
        ValueError: ``p_bias`` 越界，或 ``p_bias > 0`` 而 ``params`` 为 None。
    """
    if not 0.0 <= p_bias <= 1.0:
        raise ValueError(f"p_bias 必须在 [0,1]，当前 {p_bias}")
    if p_bias > 0.0 and params is None:
        raise ValueError("p_bias > 0 需要传入训练后的 params（偏置网络）")
    xmin, xmax, ymin, ymax = bounds
    start = np.asarray(start, dtype=float).reshape(2)
    goal = np.asarray(goal, dtype=float).reshape(2)
    # 环境编码一次、全程共用（对照 Enet 的"一次编码、千万次查询"分工）。
    enc = sdf_grid(obstacle_list, bounds, k)
    # 树存储：倍增缓冲（与 rrt._plan_core 同一套热路径手法）。
    nodes_buf = np.empty((1024, 2))
    nodes_buf[0] = start
    n_nodes = 1
    parent = [-1]
    costs_buf = np.zeros(1024)
    rng = np.random.default_rng(seed)
    goal_idx = -1
    n_iters_done = 0

    for it in range(max_iters):
        n_iters_done = it + 1
        # ① goal bias（教程 (3.11)）。
        if rng.random() < goal_bias:
            q_rand = goal.copy()
        else:
            # ② 学习偏置 vs 均匀采样（(9.1)：以 p 用 p_θ 的建议代替均匀）。
            if rng.random() < p_bias:
                # ③ 条件节点 = 树上离目标最近者（Algorithm 2 的"扩展树端"
                # 在有效树上的对应物；argmin 无随机），建议点加高斯抖动。
                i_c = int(np.argmin(
                    np.sqrt(((nodes_buf[:n_nodes] - goal[None, :]) ** 2).sum(axis=1))
                ))
                q_prop = propose_points(
                    params, nodes_buf[i_c][None, :], goal, enc, bounds, clamp
                )[0]
                q_rand = np.clip(
                    q_prop + rng.normal(0.0, jitter, 2), [xmin, ymin], [xmax, ymax]
                )
            else:
                # ④ 均匀采样（bounds 内独立均匀）。
                q_rand = np.array(
                    [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)]
                )
        # 最近邻 + steer 限步长（(3.3)，复用 rrt.steer）。
        view = nodes_buf[:n_nodes]
        d_rand = view - q_rand[None, :]
        i_near = int(np.argmin(np.sqrt((d_rand * d_rand).sum(axis=1))))
        q_near = view[i_near]
        q_new = steer(q_near, q_rand, eta)
        # 增量碰撞检测（(3.4)，复用 rrt.segment_free）。
        if not segment_free(q_near, q_new, obstacle_list):
            continue
        # 挂树（缓冲区按需倍增，代价递归同 rrt 的 (3.5)）。
        if n_nodes == len(nodes_buf):
            nodes_buf = np.vstack([nodes_buf, np.empty_like(nodes_buf)])
            costs_buf = np.concatenate([costs_buf, np.zeros_like(costs_buf)])
        nodes_buf[n_nodes] = q_new
        parent.append(i_near)
        costs_buf[n_nodes] = costs_buf[i_near] + float(
            np.linalg.norm(q_new - q_near)
        )
        n_nodes += 1
        # 目标判定：新节点落入目标邻域（与 rrt.plan_rrt 同口径）。
        if float(np.linalg.norm(q_new - goal)) <= goal_radius:
            goal_idx = n_nodes - 1
            break

    idx_seq: np.ndarray | None = None
    if goal_idx != -1:
        # 沿 parent 回溯提取路径（rrt._extract_path 为私有，此处就地实现）。
        seq: list[int] = []
        j = goal_idx
        while j != -1:
            seq.append(int(j))
            j = int(parent[j])
        seq.reverse()
        idx_seq = np.asarray(seq, dtype=int)
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

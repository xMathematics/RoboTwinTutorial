"""LOAM 思想的 2D 激光里程计与建图（教学实现）.

Teaching implementation of Zhang & Singh, ``LOAM: Lidar Odometry and Mapping
in Real-time``, RSS 2014 (papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf),
对应教程第 10 章的系统观：odometry 与 mapping 双算法（论文 Fig. 3 / §IV）。

**2D 教学简化（显式声明）**：论文面向 6-DOF 运动的 2 轴激光雷达；本模块把
传感器与运动都约简到平面——SE(2) 位姿 ``(x, y, theta)`` 的 2D 激光扫描仪。
论文的核心机制全部保留，逐条对应关系如下（式号均指论文编号）。

式号对照
--------
- Eq. (1)   平滑度 ``c_i``：半径 ``r`` 邻域内点的平均距离、按点的 range 归
  一化。``c`` 大 => 邻域两侧距离突变（**边缘点**）；``c`` 小 => 邻域共线
  （**平面点**）。该式在 2D 逐字适用（距离为平面欧氏距离）。
- §V-A      特征选取规则：``c`` 阈值 + "邻域内无已选点"的非极大值抑制 +
  配额（论文按 4 子区域各限 2 边缘/4 平面点，教学版简化为全局配额）；论文
  的"近平行表面/遮挡边界剔除"在 2D 中省略。
- Eq. (2)   边缘点到线距离：3D 叉积在 2D 退化为标量叉积——
  ``e = (q1 - X) x2 (q2 - q1) / ||q2 - q1||``，即到两最近邻目标点连线
  ``q1q2`` 的带符号垂距（``x2`` 为 2D 标量叉积）。
- Eq. (3)   平面点到平面距离：2D 中平面退化为直线（贴墙约束），与 Eq. (2)
  同型，仅特征类型不同。
- Eq. (9)-(11)  两类残差堆叠为 ``f_E``/``f_H`` 并最小化 ``sum ||f||^2``。
- Eq. (12)  LM 正规方程 ``(J^T J + lambda D^T D) dx = -J^T f``：复用
  :func:`core.solver.gauss_newton`（``lm_lambda > 0``，教学版略去列缩放
  ``D``；论文的 bisquare 权重以 Huber IRLS 等价替代）。
- §IV/Fig. 9  双速率结构：odometry 高频逐帧（论文 ~10Hz）以上一帧位姿为
  初值做帧间配准；mapping 低频每 ``mapping_every`` 帧（论文 ~1Hz，教学版
  1x/5x）把帧间结果对累积地图精化，位姿按 "mapping 位姿 ⊕ odometry 增量"
  组合输出。
- §VI       建图对应：对特征点邻域内的地图点做 2x2 协方差特征值分解——
  ``lambda1/lambda2`` 大 => 局部线状，直线过质心、方向为主特征向量；残差
  仍用 Eq. (2) 的点到线形式。论文的 10m 立方体 + KD-tree 简化为暴力最近
  邻 + 体素（质心）降采样。

Conventions
-----------
- 位姿 ``s = (x, y, theta)``（world->body）；扫描点机体极坐标
  ``(range, angle)``，笛卡尔 ``p = range * (cos angle, sin angle)``。
- 配准返回的位姿是"当前帧在目标帧（上一帧机体系 / 地图系）中的位姿"，即
  把当前帧机体点变换到目标系的变换 ``X = R(theta) p + t``。
- :func:`run_loam2d` 的输出轨迹以第 0 帧机体系为世界系。
"""
from dataclasses import dataclass

import numpy as np

from core.solver import gauss_newton

__all__ = [
    "FeatureSet",
    "Loam2DOutput",
    "RegistrationResult",
    "Room",
    "extract_features",
    "make_room",
    "point_line_residuals_jacobian",
    "register_scan_to_map",
    "register_scan_to_scan",
    "render_scan",
    "run_loam2d",
    "scan_points",
    "smoothness",
    "wrap_angle",
]

#: 交叉积为零（平行）时射线-线段相交判定为无效的阈值。
_PARALLEL_EPS = 1e-12
#: 命中距离的下限（排除传感器自身位置处的退化交点）。
_T_MIN = 1e-9


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """把角度卷绕到 ``(-pi, pi]``（帧间角度差、位姿输出统一使用）。"""
    a = np.asarray(angle, dtype=float)
    wrapped = (a + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def _cross2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """2D 标量叉积 ``p x q = p_x q_y - p_y q_x``（支持广播）。"""
    return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]


@dataclass(frozen=True)
class Room:
    """2D 房间的几何基元：线段（墙、方柱）与圆（圆墙、圆柱）。

    Attributes:
        segments: (M, 2, 2) 线段端点 ``[[x1, y1], [x2, y2]]``。
        circles: (C, 3) 每行 ``(cx, cy, radius)``。
    """

    segments: np.ndarray
    circles: np.ndarray


def make_room(
    shape: str = "rect",
    width: float = 20.0,
    height: float = 16.0,
    pillar_positions: tuple[tuple[float, float], ...] = (
        (-5.0, 4.0), (0.0, 4.5), (5.0, -4.0),
    ),
    pillar_size: float = 0.8,
    radius: float = 12.0,
) -> Room:
    """合成 2D 房间（矩形墙 + 若干方柱凸角，或无特征圆形房间）。

    ``shape="rect"``（默认）：``width x height`` 矩形四墙 + 若干轴对齐方柱
    ——方柱侧面提供平面特征、轮廓（视线掠过柱角处）提供强边缘特征，对应
    LOAM 论文强调的"边缘 + 平面"两类结构。
    ``shape="circle"``：半径 ``radius`` 的圆形房间，无任何边缘（解析圆，
    射线求交精确），用于复现"无特征走廊"的可观测性退化（见
    :func:`register_scan_to_scan` 与测试）。

    Args:
        shape: ``"rect"`` 或 ``"circle"``。
        width, height: 矩形房间边长（米）。
        pillar_positions: 方柱中心位置（米）。
        pillar_size: 方柱边长（米）。
        radius: 圆形房间半径（米）。

    Returns:
        :class:`Room`。

    Raises:
        ValueError: ``shape`` 非法。
    """
    if shape == "circle":
        return Room(
            segments=np.zeros((0, 2, 2)),
            circles=np.array([[0.0, 0.0, float(radius)]]),
        )
    if shape != "rect":
        raise ValueError(f"shape must be 'rect' or 'circle', got {shape!r}")

    x2, y2 = width / 2.0, height / 2.0
    walls = np.array(
        [
            [[-x2, -y2], [x2, -y2]],
            [[x2, -y2], [x2, y2]],
            [[x2, y2], [-x2, y2]],
            [[-x2, y2], [-x2, -y2]],
        ]
    )
    pillars = []
    half = pillar_size / 2.0
    for px, py in pillar_positions:
        pillars.append([[px - half, py - half], [px + half, py - half]])
        pillars.append([[px + half, py - half], [px + half, py + half]])
        pillars.append([[px + half, py + half], [px - half, py + half]])
        pillars.append([[px - half, py + half], [px - half, py - half]])
    return Room(segments=np.concatenate([walls, pillars]), circles=np.zeros((0, 3)))


def render_scan(
    room: Room,
    pose: np.ndarray,
    n_beams: int = 720,
    max_range: float = 30.0,
    range_noise_std: float = 0.0,
    angles: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从 ``pose`` 处射线求交合成一帧带噪 2D 激光扫描（极坐标）。

    对每条束线 ``o + t d``（``d`` 为机体角度方向）与全部线段/圆求最小正
    交点距离；随后叠加零均值高斯 range 噪声（仅作用于有效命中），并把超
    出 ``max_range`` 的命中丢弃为 ``inf``（无回波）。

    Args:
        room: :class:`Room` 几何。
        pose: (3,) 传感器位姿 ``(x, y, theta)``。
        n_beams: 束线数（``angles`` 为 None 时均匀覆盖 ``(-pi, pi)``）。
        max_range: 最大量程（米），超出视为无回波。
        range_noise_std: range 高斯噪声标准差（米）。
        angles: 可选 (n_beams,) 机体角度序列；默认均匀全向。
        seed: 噪声种子（确定性）。

    Returns:
        ``(ranges, angles)``：量程（无回波为 ``inf``）与对应机体角度。
    """
    pose = np.asarray(pose, dtype=float).reshape(3)
    if angles is None:
        angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    angles = np.asarray(angles, dtype=float)
    # 机体角度 -> 世界方向（激光雷达随机体转动，束线方向必须先转过 theta）。
    d = np.stack(
        [np.cos(angles + pose[2]), np.sin(angles + pose[2])], axis=1
    )  # (B, 2)
    origin = pose[:2]
    best = np.full(len(angles), np.inf)

    seg = np.asarray(room.segments, dtype=float).reshape(-1, 2, 2)
    if seg.shape[0]:
        a = seg[:, 0]                                    # (M, 2)
        e = seg[:, 1] - a                                # (M, 2) 边向量
        ao = (a - origin)[None, :, :]                    # (1, M, 2)
        dv = d[:, None, :]                               # (B, 1, 2)
        ev = e[None, :, :]                               # (1, M, 2)
        den = _cross2(dv, ev)                            # (B, M)
        safe_den = np.where(np.abs(den) > _PARALLEL_EPS, den, 1.0)
        t_hit = _cross2(ao, ev) / safe_den               # 沿束线参数
        u_hit = _cross2(ao, dv) / safe_den               # 沿线段参数
        valid = (
            (np.abs(den) > _PARALLEL_EPS)
            & (t_hit > _T_MIN)
            & (u_hit >= 0.0)
            & (u_hit <= 1.0)
        )
        best = np.minimum(best, np.where(valid, t_hit, np.inf).min(axis=1))

    circ = np.asarray(room.circles, dtype=float).reshape(-1, 3)
    for cx, cy, radius in circ:
        oc = origin - np.array([cx, cy])
        b = d @ oc                                       # (B,)
        disc = b * b - (oc @ oc - radius * radius)
        hit = disc >= 0.0
        sq = np.sqrt(np.maximum(disc, 0.0))
        t_near, t_far = -b - sq, -b + sq
        t = np.where(t_near > _T_MIN, t_near, np.where(t_far > _T_MIN, t_far, np.inf))
        best = np.minimum(best, np.where(hit, t, np.inf))

    if range_noise_std > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, range_noise_std, len(angles))
        best = best + np.where(np.isfinite(best), noise, 0.0)
    best = np.where(best > max_range, np.inf, best)
    return best, angles


def scan_points(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """极坐标扫描 -> 机体系笛卡尔点 (N, 2)（丢弃 ``inf``/NaN 无回波束）。"""
    r = np.asarray(ranges, dtype=float)
    a = np.asarray(angles, dtype=float)
    ok = np.isfinite(r)
    return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], axis=1)


def smoothness(points: np.ndarray, r: int = 5) -> np.ndarray:
    """LOAM Eq. (1) 的 2D 平滑度 ``c_i``（逐点）。

    ``c_i = (1 / (|S| * ||X_i||)) * sum_{j in S, j != i} ||X_i - X_j||``，
    ``S`` 为同一扫描中 ``i`` 两侧各 ``r`` 个连续点（扫描端处裁剪）；2D 中
    ``||X_i - X_j||`` 为平面欧氏距离、``||X_i||`` 即该点的 range（按距离
    归一化，远处采样间距更大）。直线/圆墙（均匀角采样）上 ``c_i`` 恒定，
    距离突变处（轮廓、遮挡沿）``c_i`` 显著增大。

    Args:
        points: (N, 2) 单帧机体系点（按扫描顺序排列）。
        r: 邻域单侧点数。

    Returns:
        (N,) 平滑度。
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    c = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - r), min(n, i + r + 1)
        neigh = np.concatenate([pts[lo:i], pts[i + 1: hi]], axis=0)
        # 依据：论文 Eq. (1)——邻域距离和按 |S| 与 range 双重归一化
        # （远处角采样间距更大，不归一化会高估远处的 c）。
        c[i] = (
            np.linalg.norm(neigh - pts[i], axis=1).sum()
            / (max(len(neigh), 1) * max(norms[i], 1e-12))
        )
    return c


def _select_by_smoothness(
    c: np.ndarray, r: int, threshold: float, descending: bool, cap: int
) -> np.ndarray:
    """按平滑度排序选取特征索引（阈值 + ``r`` 邻域非极大值抑制 + 配额）。

    对应论文 §V-A：从 ``c`` 最大（边缘）或最小（平面）处开始选取；某点被
    选中后其 ``r`` 邻域内的点不再参选（论文："None of its surrounding
    point is already selected"）；数量不超过配额（论文按子区域限额，此处
    全局配额）。
    """
    order = np.argsort(-c if descending else c)
    selected: list[int] = []
    for i in order:
        if c[i] < threshold if descending else c[i] > threshold:
            break  # 后续候选更不满足阈值（有序遍历）
        if any(abs(i - j) <= r for j in selected):
            continue  # 依据：论文 §V-A "邻域内无已选点"的 NMS——间距 <= r 不再参选
        selected.append(int(i))
        if len(selected) >= cap:
            break
    return np.array(sorted(selected), dtype=int)


@dataclass(frozen=True)
class FeatureSet:
    """一帧扫描的特征（全部在机体系，与位姿无关，可提取一次反复使用）。

    Attributes:
        points: (N, 2) 全部有效扫描点（按束序）。
        smoothness: (N,) 对应 :func:`smoothness` 值。
        edge_idx: (E,) 边缘点索引（按束序升序）。
        planar_idx: (P,) 平面点索引（按束序升序）。
    """

    points: np.ndarray
    smoothness: np.ndarray
    edge_idx: np.ndarray
    planar_idx: np.ndarray

    @property
    def edge_points(self) -> np.ndarray:
        """(E, 2) 边缘点坐标。"""
        return self.points[self.edge_idx]

    @property
    def planar_points(self) -> np.ndarray:
        """(P, 2) 平面点坐标。"""
        return self.points[self.planar_idx]

    @property
    def feature_points(self) -> np.ndarray:
        """(E+P, 2) 两类特征合并（建图入库用）。"""
        return np.vstack([self.edge_points, self.planar_points])


def extract_features(
    points: np.ndarray,
    r: int = 5,
    c_edge_min: float = 0.1,
    c_planar_max: float = 0.05,
    n_edge_max: int = 20,
    n_planar_max: int = 60,
) -> FeatureSet:
    """提取边缘/平面特征（LOAM §V-A 的 2D 版，见 :func:`smoothness`）。

    Args:
        points: (N, 2) 单帧机体系扫描点（按束序）。
        r: 平滑度邻域单侧点数（同时用作非极大值抑制半径）。
        c_edge_min: 边缘点平滑度下限。
        c_planar_max: 平面点平滑度上限。
        n_edge_max: 边缘点全局配额。
        n_planar_max: 平面点全局配额。

    Returns:
        :class:`FeatureSet`。
    """
    c = smoothness(points, r)
    edge = _select_by_smoothness(c, r, c_edge_min, descending=True, cap=n_edge_max)
    planar = _select_by_smoothness(
        c, r, c_planar_max, descending=False, cap=n_planar_max
    )
    return FeatureSet(points=points, smoothness=c, edge_idx=edge, planar_idx=planar)


def point_line_residuals_jacobian(
    pose: np.ndarray, points_body: np.ndarray, lines: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """点到线残差及其对位姿 ``(x, y, theta)`` 的解析雅可比。

    残差为 LOAM Eq. (2) 的 2D 标量叉积形式（带符号垂距）::

        e = ((q1 - X) x2 (q2 - q1)) / ||q2 - q1||,   X = R(theta) p + t

    记 ``n = (w_y, -w_x)/||w||``（``w = q2 - q1``，连线单位法向），则
    ``e = (X - q1) . n``，且（依据：2D 旋转导数 ``dR/dtheta = J R``，
    ``J = [[0, -1], [1, 0]]``，与 core.lie 的左扰动推导同源但更简单——
    SE(2) 增量本就是 3 维向量）::

        de/dt = n,      de/dtheta = n . (J (X - t))

    残差定义取"预测 - 观测"侧的负号约定与雅可比严格一致（有限差分在
    ``tests/test_loam2d.py`` 校验到机器精度）。

    Args:
        pose: (3,) 位姿 ``(x, y, theta)``。
        points_body: (M, 2) 当前帧机体系点。
        lines: (M, 2, 2) 每点的目标线段端点 ``[[q1], [q2]]``（目标系）。

    Returns:
        ``(e, J)``：残差 (M,) 与雅可比 (M, 3)，列序 ``(x, y, theta)``。
    """
    pose = np.asarray(pose, dtype=float).reshape(3)
    pts = np.asarray(points_body, dtype=float)
    lines = np.asarray(lines, dtype=float)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    X = pts @ R.T + pose[:2]                              # (M, 2)
    w = lines[:, 1] - lines[:, 0]                         # (M, 2)
    w_norm = np.maximum(np.linalg.norm(w, axis=1), 1e-9)
    n = np.stack([w[:, 1], -w[:, 0]], axis=1) / w_norm[:, None]
    e = np.sum((X - lines[:, 0]) * n, axis=1)
    Jrot = np.array([[0.0, -1.0], [1.0, 0.0]])
    dX_dtheta = (X - pose[:2]) @ Jrot.T                   # J (X - t), (M, 2)
    J = np.concatenate([n, np.sum(n * dX_dtheta, axis=1)[:, None]], axis=1)
    return e, J


@dataclass(frozen=True)
class RegistrationResult:
    """一次配准的结果与诊断量。

    Attributes:
        pose: (3,) 配准后的位姿（当前帧在目标系中）。
        cost: 终端鲁棒代价。
        n_iters: G-N/LM 实际迭代轮数。
        converged: 求解器收敛标志。
        condition_number: 终端线性化点处 ``J^T J`` 的条件数——可观测性
            诊断（无特征几何 => 某方向曲率趋零 => 条件数爆炸）。
        healthy: ``condition_number <= max_condition`` 且残差数 >= 3。
        n_residuals: 终端参与求解的残差条数。
    """

    pose: np.ndarray
    cost: float
    n_iters: int
    converged: bool
    condition_number: float
    healthy: bool
    n_residuals: int


def _solve_point_to_line(
    terms_fn,
    pose_init: np.ndarray,
    robust_delta: float | None,
    lm_lambda: float,
    n_iters: int,
    max_condition: float,
) -> RegistrationResult:
    """共享的 G-N/LM 求解收尾（Eq. (11)-(12)：堆叠残差 + 阻尼最小二乘）。

    ``terms_fn(pose) -> (e, J)`` 负责在给定线性化点重新查找对应并给出
    堆叠残差/雅可比；求解后用终端线性化点计算 ``J^T J`` 条件数作为
    可观测性诊断。
    """
    sol = gauss_newton(
        lambda x: terms_fn(x)[0],
        lambda x: terms_fn(x)[1],
        x0=np.asarray(pose_init, dtype=float).reshape(3),
        n_iters=n_iters,
        lm_lambda=lm_lambda,
        robust_delta=robust_delta,
    )
    pose = np.array([sol.x[0], sol.x[1], wrap_angle(sol.x[2])])
    e, J = terms_fn(pose)
    # 可观测性诊断：无特征几何（如圆形房间中的纯旋转）下，J^T J 的切向旋转
    # 方向曲率趋零 => 最小特征值 -> 0，条件数爆炸（inf = 该方向完全不可观
    # 测）；据此把配准判为 unhealthy。
    if len(e) >= 3:
        cond = float(np.linalg.cond(J.T @ J))
    else:
        cond = float("inf")
    healthy = bool(np.isfinite(cond) and cond <= max_condition)
    return RegistrationResult(
        pose=pose,
        cost=float(sol.cost),
        n_iters=int(sol.n_iters),
        converged=bool(sol.converged),
        condition_number=cond,
        healthy=healthy,
        n_residuals=int(len(e)),
    )


def _nearest_two_lines(
    pose: np.ndarray, points_body: np.ndarray, target_points: np.ndarray,
    max_corr_dist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """逐特征点取目标点集中最近 2 个同型点 -> 目标线段 + 距离门限掩码。

    对应论文 §V-B 的 odometry 对应查找（"find the two closest points ...
    then the edge line passes through them"；2D 中平面点同理）。
    """
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    X = points_body @ R.T + pose[:2]
    d2 = ((X[:, None, :] - target_points[None, :, :]) ** 2).sum(-1)
    idx = np.argpartition(d2, 1, axis=1)[:, :2]
    lines = target_points[idx]
    gate = d2[np.arange(len(X))[:, None], idx].max(axis=1) <= max_corr_dist**2
    return lines, gate


def register_scan_to_scan(
    features_curr: FeatureSet,
    points_prev: np.ndarray,
    features_prev: FeatureSet,
    pose_init: np.ndarray,
    max_corr_dist: float = 0.6,
    robust_delta: float | None = 0.05,
    lm_lambda: float = 1e-4,
    n_iters: int = 30,
    max_condition: float = 1e8,
) -> RegistrationResult:
    """帧间配准（odometry 核心）：当前帧特征 -> 上一帧点集的点到线配准。

    对每个边缘/平面特征点，用当前位姿估计变换到上一帧机体系后，在上一帧
    的**同型**特征中取最近 2 点构成目标线段（论文 §V-B），残差为
    :func:`point_line_residuals_jacobian` 的点到线形式；对应在每个非线性
    迭代处重新查找（ICP 惯例），距离超过 ``max_corr_dist`` 的对应被门限
    剔除（论文可靠性检查的 2D 简化），鲁棒核以 Huber IRLS 近似论文的
    bisquare 权重（Eq. (11) 后）。

    Args:
        features_curr: 当前帧 :class:`FeatureSet`（机体系）。
        points_prev: (N, 2) 上一帧全部机体系点。
        features_prev: 上一帧 :class:`FeatureSet`（提供同型对应目标）。
        pose_init: (3,) 位姿初值（帧间配准通常取零运动 = 上一帧位姿）。
        max_corr_dist: 对应距离门限（米）。
        robust_delta: Huber 核宽度（米）；``None`` 关闭鲁棒加权。
        lm_lambda: LM 阻尼初值（Eq. (12) 的 lambda）。
        n_iters: 最大非线性迭代轮数。
        max_condition: ``J^T J`` 条件数上限（超过则 ``healthy=False``）。

    Returns:
        :class:`RegistrationResult`。

    Raises:
        ValueError: 初值处无任何通过门限的同型对应。
    """
    pairs = (
        (features_curr.edge_points, features_prev.edge_points),
        (features_curr.planar_points, features_prev.planar_points),
    )

    def terms_fn(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e_list: list[np.ndarray] = []
        j_list: list[np.ndarray] = []
        for pts, target in pairs:
            if len(pts) == 0 or len(target) < 2:
                continue
            lines, gate = _nearest_two_lines(pose, pts, target, max_corr_dist)
            if not gate.any():
                continue
            e, J = point_line_residuals_jacobian(pose, pts, lines)
            e_list.append(e[gate])
            j_list.append(J[gate])
        if not e_list:
            raise ValueError("no gated correspondences at this pose")
        return np.concatenate(e_list), np.vstack(j_list)

    terms_fn(np.asarray(pose_init, dtype=float).reshape(3))  # 初值处必须可行
    return _solve_point_to_line(
        terms_fn, pose_init, robust_delta, lm_lambda, n_iters, max_condition
    )


def register_scan_to_map(
    features_curr: FeatureSet,
    map_points: np.ndarray,
    pose_init: np.ndarray,
    corr_radius: float = 0.6,
    min_line_points: int = 4,
    min_linearity: float = 30.0,
    robust_delta: float | None = 0.05,
    lm_lambda: float = 1e-4,
    n_iters: int = 30,
    max_condition: float = 1e8,
) -> RegistrationResult:
    """扫描-地图配准（mapping 核心）：协方差特征值直线对应 + 点到线精化。

    对应论文 §VI：对每个特征点（按当前位姿估计变换到地图系），取地图中
    ``corr_radius`` 邻域内的全部点，计算 2x2 协方差并做特征值分解——
    ``lambda1/lambda2 >= min_linearity`` 时判定局部线状，直线过质心、方向
    取主特征向量（2D 版的"edge line / planar patch"统一为线），随后用与
    :func:`register_scan_to_scan` 相同的点到线残差（论文：'the distances
    computed using the same formulations as (2) and (3)'）做 LM 精化。

    Args:
        features_curr: 当前帧 :class:`FeatureSet`（机体系）。
        map_points: (M, 2) 累积地图点（目标系 = 第 0 帧机体系）。
        pose_init: (3,) 位姿初值（通常为 odometry 组合的预测位姿）。
        corr_radius: 地图邻域半径（米）。
        min_line_points: 拟合直线所需的最少邻域点数。
        min_linearity: 线状判定阈值 ``lambda1/lambda2``。
        robust_delta: Huber 核宽度（米）；``None`` 关闭鲁棒加权。
        lm_lambda: LM 阻尼初值。
        n_iters: 最大非线性迭代轮数。
        max_condition: ``J^T J`` 条件数上限。

    Returns:
        :class:`RegistrationResult`。

    Raises:
        ValueError: 地图点不足或初值处无有效对应。
    """
    map_points = np.asarray(map_points, dtype=float).reshape(-1, 2)
    if len(map_points) < min_line_points:
        raise ValueError("map has too few points for line correspondence")

    def terms_fn(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        pts_all = features_curr.feature_points
        X = pts_all @ R.T + pose[:2]
        d2 = ((X[:, None, :] - map_points[None, :, :]) ** 2).sum(-1)
        e_list: list[np.ndarray] = []
        j_list: list[np.ndarray] = []
        for i in range(len(pts_all)):
            nb = map_points[d2[i] <= corr_radius**2]
            if len(nb) < min_line_points:
                continue
            # 依据：论文 §VI——邻域点 2x2 协方差特征值分解做直线对应：
            # lambda1/lambda2 大 => 局部呈线状，直线过质心、方向取主特征向量。
            mu = nb.mean(axis=0)
            cov = (nb - mu).T @ (nb - mu) / len(nb)
            w, v = np.linalg.eigh(cov)          # 升序: w[0] <= w[1]
            if w[0] <= 1e-12 or w[1] / w[0] < min_linearity:
                continue
            direction = v[:, 1]                  # 主特征向量 = 线方向
            lines = np.array([[mu, mu + direction]])
            e, J = point_line_residuals_jacobian(pose, pts_all[i: i + 1], lines)
            e_list.append(e[0])
            j_list.append(J[0])
        if not e_list:
            raise ValueError("no map correspondences passed the linearity gate")
        return np.array(e_list), np.vstack(j_list)

    terms_fn(np.asarray(pose_init, dtype=float).reshape(3))
    return _solve_point_to_line(
        terms_fn, pose_init, robust_delta, lm_lambda, n_iters, max_condition
    )


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SE(2) 位姿复合 ``a (+) b``：先施加 ``a`` 再施加 ``b``。"""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        [
            a[0] + c * b[0] - s * b[1],
            a[1] + s * b[0] + c * b[1],
            wrap_angle(a[2] + b[2]),
        ]
    )


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """体素栅格降采样（论文 §VI 的 voxel grid filter）：逐体素取质心。"""
    if len(points) == 0:
        return points
    # 依据：论文 §VI 的 voxel points filter——每体素以质心代替内部点
    # （保持表面位置基本不变，同时控制地图点数）。
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_cells = int(inverse.max()) + 1
    sums = np.zeros((n_cells, 2))
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse, minlength=n_cells)
    return sums / counts[:, None]


@dataclass(frozen=True)
class Loam2DOutput:
    """:func:`run_loam2d` 的输出（双速率轨迹 + 诊断）。

    Attributes:
        poses_odom: (T, 3) 纯帧间 odometry 轨迹（无地图精化，第 0 帧为原点）。
        poses_map: (T, 3) mapping 精化后的轨迹（第 0 帧为原点）。
        key_indices: 执行了 mapping 精化的帧号（升序）。
        map_points: (M, 2) 终端累积地图点（第 0 帧机体系）。
        odom_results: 每帧 :class:`RegistrationResult`（t>=1）。
        map_results: 每个关键帧的 :class:`RegistrationResult`。
    """

    poses_odom: np.ndarray
    poses_map: np.ndarray
    key_indices: np.ndarray
    map_points: np.ndarray
    odom_results: tuple[RegistrationResult, ...]
    map_results: tuple[RegistrationResult, ...]


def run_loam2d(
    ranges_list: list[np.ndarray] | tuple[np.ndarray, ...],
    angles: np.ndarray,
    mapping_every: int = 5,
    r: int = 5,
    c_edge_min: float = 0.1,
    c_planar_max: float = 0.05,
    n_edge_max: int = 20,
    n_planar_max: int = 60,
    max_corr_dist: float = 0.6,
    corr_radius: float = 0.6,
    voxel: float = 0.1,
    robust_delta: float | None = 0.05,
    max_condition: float = 1e8,
) -> Loam2DOutput:
    """LOAM 双速率主流程（论文 §IV/Fig. 9，教学版 1x/5x）。

    每帧扫描依次：

    1. **odometry（高频）**：把当前帧特征对上一帧点集做帧间配准（初值取
       零运动 = "上一帧位姿"，即论文的恒速模型在相邻帧近似下的简化），得
       帧间增量 ``Delta_t``，组合出 odometry 轨迹；
    2. **mapping（低频，每 ``mapping_every`` 帧）**：把当前帧特征对累积
       地图（全部历史帧特征点按 map 轨迹变换入库、体素降采样）做扫描-
       地图精化，重设该帧的 map 位姿（对应论文 "the mapping algorithm
       matches and registers ... by optimizing the lidar pose"）；
    3. 地图入库：当前帧特征点按（精化后的）map 位姿变换后并入地图。

    Args:
        ranges_list: T 帧量程序列（每帧 (n_beams,)，无回波为 ``inf``）。
        angles: (n_beams,) 机体角度（各帧相同）。
        mapping_every: mapping 运行周期（每多少帧一次）。
        r, c_edge_min, c_planar_max, n_edge_max, n_planar_max:
            特征提取参数（见 :func:`extract_features`）。
        max_corr_dist: 帧间对应门限（米）。
        corr_radius: 地图邻域半径（米）。
        voxel: 地图体素边长（米）。
        robust_delta: Huber 核宽度（米）。
        max_condition: 配准健康判定的条件数上限。

    Returns:
        :class:`Loam2DOutput`。

    Raises:
        ValueError: 扫描帧数 < 1 或 ``mapping_every < 1``。
    """
    if len(ranges_list) < 1:
        raise ValueError("need at least one scan")
    if mapping_every < 1:
        raise ValueError("mapping_every must be >= 1")

    scans = [scan_points(np.asarray(rr, dtype=float), angles) for rr in ranges_list]
    feats = [
        extract_features(s, r=r, c_edge_min=c_edge_min, c_planar_max=c_planar_max,
                         n_edge_max=n_edge_max, n_planar_max=n_planar_max)
        for s in scans
    ]
    n_frames = len(scans)
    poses_odom = np.zeros((n_frames, 3))
    poses_map = np.zeros((n_frames, 3))
    odom_results: list[RegistrationResult] = []
    map_results: list[RegistrationResult] = []
    key_indices: list[int] = []

    map_points = _voxel_downsample(feats[0].feature_points, voxel)
    for t in range(1, n_frames):
        res = register_scan_to_scan(
            feats[t], scans[t - 1], feats[t - 1], pose_init=np.zeros(3),
            max_corr_dist=max_corr_dist, robust_delta=robust_delta,
            max_condition=max_condition,
        )
        odom_results.append(res)
        poses_odom[t] = _compose(poses_odom[t - 1], res.pose)
        poses_map[t] = _compose(poses_map[t - 1], res.pose)
        if t % mapping_every == 0:
            refined = register_scan_to_map(
                feats[t], map_points, pose_init=poses_map[t],
                corr_radius=corr_radius, robust_delta=robust_delta,
                max_condition=max_condition,
            )
            map_results.append(refined)
            poses_map[t] = refined.pose
            key_indices.append(t)
        world = feats[t].feature_points @ _rot(poses_map[t]).T + poses_map[t][:2]
        map_points = _voxel_downsample(
            np.vstack([map_points, world]), voxel
        )

    return Loam2DOutput(
        poses_odom=poses_odom,
        poses_map=poses_map,
        key_indices=np.array(key_indices, dtype=int),
        map_points=map_points,
        odom_results=tuple(odom_results),
        map_results=tuple(map_results),
    )


def _rot(pose: np.ndarray) -> np.ndarray:
    """SE(2) 旋转矩阵 ``R(theta)``。"""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([[c, -s], [s, c]])

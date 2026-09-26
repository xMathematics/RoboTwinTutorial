"""FastSLAM 2D：Rao-Blackwellized 粒子滤波（教学实现）。

函数流水线
----------
Montemerlo, Thrun, Koller & Wegbreit, ``FastSLAM: A Factored Solution to the
Simultaneous Localization and Mapping Problem``, AAAI 2002
（papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf）的教学实现，
推导见教程第 07 章 §7.3：SLAM 后验分解为"机器人路径上的粒子滤波 × 每个路标
独立的 2x2 EKF"（论文 Eq. (4)，精确条件分解 —— 教程第 07 章 §7.3 的
Rao-Blackwell 分解式），以此替代 EKF-SLAM 联合 K 路标协方差每步 O(K^2) 的
更新代价。

本模块不依赖 ``core`` 或库内其他子模块（仅 numpy + 标准库）。调用链：
:func:`simulate_rectangle` 产生里程计/观测/真值关联 -> :class:`FastSLAM2D`
逐步消化 -> :meth:`FastSLAM2D.pose_estimate` / :meth:`FastSLAM2D.map_estimate`
输出位姿与地图估计；由 ``tests/test_fastslam.py`` 直接驱动。

每个时间步的递推为：

1. 从运动模型采样每个粒子的位姿（论文 Eq. (6)）；
2. 按粒子做数据关联 —— 已知关联，或在平方马氏距离门限下的极大似然最近邻
   （论文 Eq. (12)）；
3. 对被关联路标做 EKF 更新（论文 Eq. (9)）；未被观测的路标保持其后验不变
   （Eq. (10)）；首次观测按逆测量模型初始化新的高斯；
4. 重要性权重乘以测量似然 ``N(z; z_hat, S)``（论文 Eq. (7)，其中协方差用
   Eq. (11) 的 EKF 近似）；
5. 一旦有效样本数 ``N_eff = 1 / sum_i w_i^2`` 低于
   ``resample_fraction * N``，执行一次系统重采样（systematic resampling）。

约定
----
- 位姿 ``s = (x, y, theta)``，路标 ``theta_k = (lx, ly)``：均在平面世界系
  （论文 "SLAM Problem Definition" 一节）；x/y 单位 m，theta 单位 rad。
- 里程计 ``u = (dx, dy, dtheta)`` 是*机体系*（body-frame）增量：作用于 ``s``
  得 ``x + dx*cos(theta) - dy*sin(theta)``、
  ``y + dx*sin(theta) + dy*cos(theta)``、``wrap(theta + dtheta)``。运动噪声
  是该增量上的 i.i.d. 高斯（论文 Eq. (1) 取高斯律），即噪声活在机体系而非
  世界系。
- 测量 ``z = (r, phi)``：距离 + 相对*机器人朝向*的方位角，
  ``phi = atan2(ly - y, lx - x) - theta``（论文 Eq. (2)）；r 单位 m、
  phi 单位 rad。
- 所有粒子从同一已知位姿、零先验不确定性出发（调用
  :meth:`FastSLAM2D.initialize`）；多样性来自后续步 Eq. (6) 的运动噪声。

示例
----
>>> sim = simulate_rectangle(seed=0)
>>> slam = FastSLAM2D(n_particles=50, n_landmarks=20, seed=0)
>>> slam.initialize(sim.gt_trajectory[0])
>>> for t in range(sim.odometry.shape[0]):
...     k = sim.n_obs[t]
...     slam.step(sim.odometry[t], sim.observations[t, :k],
...               sim.associations[t, :k])  # doctest: +SKIP
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

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

_LOG_2PI = float(np.log(2.0 * np.pi))
# r^2 的下限：当某路标估计与机器人位置重合时，方位角雅可比 (-dy/r^2, dx/r^2)
# 不会除零（下限 1e-9 m^2）。
_RANGE_FLOOR = 1e-9
#: 平方马氏距离门限，卡方(2) 分布 95% 分位数（论文 Eq. (12)）。教科书惯用值，
#: 但*实际偏紧*：合法重观测的 d^2 还包含当前位姿采样本身的误差，而 Eq. (6)
#: 的运动模型提议忽略了这一项，因此 d^2 会超出理想卡方(2) 尺度，95% 门限
#: 会把百分之几的好测量拒之门外 —— 每次拒绝都复制出一个已知路标。
GATE_CHI2_2DOF_95 = 5.991
#: 默认平方马氏距离门限。门限必须隔开两个尺度：(a) *合法*重观测的 d^2 ——
#: 卡方(2) 加位姿采样误差项，在本模块基准噪声下 N=50 时保持在 ~20 以下；
#: (b) *不同*路标的 d^2，``(间距 / sigma)^2 >= (2 m / 0.3 m)^2 ~ 44``
#: （默认仿真器）。30 落在该窗口内且两侧都有余量；论文本身也强调接受门限
#: 需要"仔细斟酌所有常数"（Eq. (12) 之后的讨论）。
GATE_DEFAULT = 30.0
#: 数据关联模式：关联量 ``n_t`` 由调用方给出。
ASSOCIATION_KNOWN = "known"
#: 数据关联模式：按粒子的门控最近邻（论文 Eq. (12)）。
ASSOCIATION_NEAREST_NEIGHBOR = "nearest_neighbor"


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """把角度约束回 ``(-pi, pi]``。

    施加于每个方位角差，使测量新息不会在分支割线处跳变 ``2*pi``
    （测量模型，论文 Eq. (2)）。

    Args:
        angle: 标量或数组形式的角度，单位 rad。

    Returns:
        约束后的角度，形状与输入相同（标量输入返回 float），取值 (-pi, pi]。
    """
    a = np.asarray(angle, dtype=float)
    wrapped = np.arctan2(np.sin(a), np.cos(a))
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def _rb_from_delta(
    dx: np.ndarray, dy: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """标量与批量辅助函数共用的距离-方位角核心。

    Args:
        dx, dy: 路标减机器人的偏移量，可广播形状 ``(...)``，单位 m。
        theta: 机器人朝向，可广播到 ``(...)``，单位 rad。

    Returns:
        ``(z_hat, H)``，其中 ``z_hat (..., 2) = (r, phi)``（r 单位 m、phi
        单位 rad），以及解析雅可比 ``H (..., 2, 2) = dh/d(lx, ly)``，
        ``H = [[dx/r, dy/r], [-dy/r^2, dx/r^2]]``。
    """
    r_sq = np.maximum(dx * dx + dy * dy, _RANGE_FLOOR)
    r = np.sqrt(r_sq)
    phi = wrap_angle(np.arctan2(dy, dx) - theta)
    z_hat = np.stack(np.broadcast_arrays(r, phi), axis=-1)
    H = np.stack(
        [
            np.stack([dx / r, dy / r], axis=-1),
            np.stack([-dy / r_sq, dx / r_sq], axis=-1),
        ],
        axis=-2,
    )
    return z_hat, H


def range_bearing_measurement(pose: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """预测测量 ``h(s, theta_k)``（论文 Eq. (2)；教程第 07 章 §7.3）。

    Args:
        pose: (3,) 机器人位姿 ``(x, y, theta)``，x/y 单位 m、theta 单位 rad。
        landmark: (2,) 路标位置 ``(lx, ly)``，单位 m。

    Returns:
        (2,) 预测 ``z_hat = (r, phi)``：方位角相对机器人朝向且约束回
        ``(-pi, pi]``；r 单位 m、phi 单位 rad。
    """
    pose = np.asarray(pose, dtype=float).reshape(3)
    landmark = np.asarray(landmark, dtype=float).reshape(2)
    z_hat, _ = _rb_from_delta(
        landmark[0] - pose[0], landmark[1] - pose[1], pose[2]
    )
    return z_hat


def range_bearing_jacobian(pose: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """解析测量雅可比 ``H = dh/d(lx, ly)``，形状 (2, 2)。

    行对应 (距离, 方位角)，列对应 (lx, ly)：
    ``H = [[dx/r, dy/r], [-dy/r^2, dx/r^2]]``，``(dx, dy)`` 是路标减机器人的
    偏移。用于按粒子的 EKF 更新（论文 Eq. (9)）与似然协方差
    ``S = H Sigma H^T + R``（Eq. (11)）；教程第 07 章 §7.3。

    Args:
        pose: (3,) 机器人位姿 ``(x, y, theta)``，x/y 单位 m、theta 单位 rad。
        landmark: (2,) 路标位置 ``(lx, ly)``，单位 m。

    Returns:
        (2, 2) 雅可比矩阵（距离行无量纲；方位角行单位 rad/m）。
    """
    pose = np.asarray(pose, dtype=float).reshape(3)
    landmark = np.asarray(landmark, dtype=float).reshape(2)
    _, H = _rb_from_delta(
        landmark[0] - pose[0], landmark[1] - pose[1], pose[2]
    )
    return H


def _invert_2x2(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """批量 2x2 矩阵的闭式求逆。

    Args:
        mat: (..., 2, 2) 矩阵；须非奇异（``S = H Sigma H^T + R`` 总非奇异，
            因为测量噪声 ``R`` 正定）。

    Returns:
        ``(inverse, determinant)``，形状分别为 (..., 2, 2) 与 (...)。
    """
    a = mat[..., 0, 0]
    b = mat[..., 0, 1]
    c = mat[..., 1, 0]
    d = mat[..., 1, 1]
    det = a * d - b * c
    inv = np.empty_like(mat)
    inv[..., 0, 0] = d
    inv[..., 0, 1] = -b
    inv[..., 1, 0] = -c
    inv[..., 1, 1] = a
    inv /= det[..., None, None]
    return inv, det


@dataclass(frozen=True)
class SimulationResult:
    """:func:`simulate_rectangle` 产出的真值加传感器日志。

    Attributes:
        gt_trajectory: (T+1, 3) 真值位姿 ``s_0..s_T``（论文 "SLAM Problem
            Definition" 一节）；x/y 单位 m、theta 单位 rad。
        gt_landmarks: (M, 2) 真值路标地图（Eq. (3) 的真值），单位 m。
        odometry: (T, 3) 带噪*机体系*增量 ``u_t``（论文 Eq. (1)），单位
            (m, m, rad)；``odometry[t]`` 把机器人从 ``gt_trajectory[t]``
            推到 ``gt_trajectory[t+1]``。
        observations: (T, K_max, 2) 距离-方位角测量 ``z_t``（论文 Eq. (2)），
            单位 (m, rad)；``observations[t]`` 在 ``gt_trajectory[t+1]`` 处
            取得（即施加 ``odometry[t]`` 之后）；各步有效数之外以 NaN 填充。
        associations: (T, K_max) 每条测量的真值关联 ``n_t``（Eq. (2) 的
            ``n_t``），填充处为 ``-1``；使 :class:`FastSLAM2D` 的已知关联
            模式可被精确检验。
        n_obs: (T,) 每步有效测量条数。
    """

    gt_trajectory: np.ndarray
    gt_landmarks: np.ndarray
    odometry: np.ndarray
    observations: np.ndarray
    associations: np.ndarray
    n_obs: np.ndarray


def simulate_rectangle(
    n_steps: int = 60,
    width: float = 10.0,
    height: float = 8.0,
    grid_cols: int = 5,
    grid_rows: int = 4,
    margin: float = 1.0,
    max_range: float = 5.0,
    odom_noise_std: Sequence[float] = (0.04, 0.04, 0.02),
    obs_noise_std: Sequence[float] = (0.15, 0.05),
    seed: int = 0,
) -> SimulationResult:
    """矩形巡逻 + 均匀路标网格（2D FastSLAM 基准场景）。

    机器人从原点角落出发，沿 ``width x height`` 矩形逆时针以 ``n_steps``
    等弧长步巡逻；朝向取当地边界方向，因此闭环回到起点。路标布置在距矩形
    边内缩 ``margin`` 的均匀 ``grid_cols x grid_rows`` 网格上（默认
    5x4 = 20 个路标、间距 2 m —— 间距远大于测量噪声是 Eq. (12) 门控最近邻
    关联可靠的前提）。里程计是真值路径的精确机体系增量加高斯噪声（概率
    运动模型，论文 Eq. (1)）；测量对 ``max_range`` 内的所有路标（理想
    360 度传感器）给出带高斯距离/方位角噪声的观测（测量模型，论文
    Eq. (2)）。真值关联 ``n_t``（Eq. (2)）被记录下来供 :class:`FastSLAM2D`
    的已知关联模式检验；最近邻模式忽略它、必须自行恢复关联（Eq. (12)）。

    Args:
        n_steps: 里程计步数（产生 ``n_steps + 1`` 个真值位姿）。
        width, height: 矩形边长，单位 m。
        grid_cols, grid_rows: 路标网格分辨率（M = cols * rows）。
        margin: 路标网格相对矩形边的内缩量，单位 m。
        max_range: 传感器量程上限，单位 m。
        odom_noise_std: (3,) 每步机体系噪声标准差 (m, m, rad)。
        obs_noise_std: (2,) 测量噪声标准差 (距离 m, 方位角 rad)。
        seed: 随机数种子，保证完全可复现。

    Returns:
        :class:`SimulationResult`，含真值与传感器日志。
    """
    rng = np.random.default_rng(seed)

    perimeter = 2.0 * (width + height)
    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]]
    )
    seg_len = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    # 按弧长把周长等分为 n_steps 段；seg 定位每步落在哪条边上，along 为边内位置。
    arc = np.mod(np.linspace(0.0, perimeter, n_steps + 1), perimeter)
    seg = np.clip(np.searchsorted(cum_len, arc, side="right") - 1, 0, 3)
    along = arc - cum_len[seg]
    direction = (corners[seg + 1] - corners[seg]) / seg_len[seg][:, None]
    xy = corners[seg] + along[:, None] * direction
    theta = np.arctan2(direction[:, 1], direction[:, 0])
    gt_trajectory = np.concatenate([xy, theta[:, None]], axis=1)

    grid_x, grid_y = np.meshgrid(
        np.linspace(margin, width - margin, grid_cols),
        np.linspace(margin, height - margin, grid_rows),
    )
    gt_landmarks = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # 真值世界系位移旋转到机体系，得到"无噪里程计"（依据：模块约定的
    # body-frame 旋转，即论文 Eq. (1) 的均值部分）。
    d_world = np.diff(gt_trajectory[:, :2], axis=0)
    c = np.cos(gt_trajectory[:-1, 2])
    s = np.sin(gt_trajectory[:-1, 2])
    odom = np.stack(
        [
            c * d_world[:, 0] + s * d_world[:, 1],
            -s * d_world[:, 0] + c * d_world[:, 1],
            wrap_angle(np.diff(gt_trajectory[:, 2])),
        ],
        axis=1,
    )
    # 加性高斯运动噪声（依据：论文 Eq. (1) 取高斯律）。
    odom = odom + rng.normal(
        scale=np.asarray(odom_noise_std, dtype=float), size=odom.shape
    )

    # 测量模型（依据：论文 Eq. (2)）：量程内每个路标给一条 (r, phi) 观测。
    obs_rows: list[np.ndarray] = []
    assoc_rows: list[np.ndarray] = []
    for t in range(1, n_steps + 1):
        pose = gt_trajectory[t]
        dxy = gt_landmarks - pose[None, :2]
        r = np.hypot(dxy[:, 0], dxy[:, 1])
        phi = wrap_angle(np.arctan2(dxy[:, 1], dxy[:, 0]) - pose[2])
        in_range = np.nonzero(r <= max_range)[0]
        z_r = r[in_range] + rng.normal(0.0, obs_noise_std[0], in_range.size)
        z_phi = wrap_angle(
            phi[in_range] + rng.normal(0.0, obs_noise_std[1], in_range.size)
        )
        obs_rows.append(np.stack([z_r, z_phi], axis=1))
        assoc_rows.append(in_range.astype(int))
    k_max = max([z.shape[0] for z in obs_rows] + [1])
    # 用 NaN/-1 填充成定长 (T, K_max, 2) / (T, K_max) 数组。
    observations = np.full((n_steps, k_max, 2), np.nan)
    associations = np.full((n_steps, k_max), -1, dtype=int)
    n_obs = np.array([z.shape[0] for z in obs_rows], dtype=int)
    for t, (z, a) in enumerate(zip(obs_rows, assoc_rows)):
        observations[t, : z.shape[0]] = z
        associations[t, : a.size] = a

    return SimulationResult(
        gt_trajectory=gt_trajectory,
        gt_landmarks=gt_landmarks,
        odometry=odom,
        observations=observations,
        associations=associations,
        n_obs=n_obs,
    )


class FastSLAM2D:
    """2D 距离-方位角 SLAM 的 Rao-Blackwellized 粒子滤波器。

    每个粒子 ``m`` 持有一个位姿采样和 ``K`` 个独立的 2x2 路标 EKF
    （论文 Eq. (5), (8)；教程第 07 章 §7.3 的 Rao-Blackwell 分解式）：

        ``s_t^[m], mu_1^[m], Sigma_1^[m], ..., mu_K^[m], Sigma_K^[m]``

    存放在定长数组 ``poses (N, 3)``、``landmark_means (N, K, 2)``、
    ``landmark_covs (N, K, 2, 2)``、观测记账掩码 ``observed (N, K)`` 与
    对数重要性权重 ``log_weights (N,)`` 中 —— 不做逐步的列表扩容。一次
    :meth:`step` 实现模块 docstring 列出的递推（论文 Eq. (6), (12), (9)-(11),
    (7), 重采样）。

    Attributes:
        n_particles: 粒子数 N。
        n_landmarks: 地图容量 K（路标槽位数，首次观测时填充）。
        motion_noise_std: (3,) 机体系运动噪声标准差 (m, m, rad)，即运动模型
            的 ``R``（论文 Eq. (1)）。
        obs_noise_std: (2,) 测量噪声标准差 (距离 m, 方位角 rad)，即测量模型
            的 ``R``（论文 Eq. (2)）。
        association: 数据关联模式，``"known"`` 或 ``"nearest_neighbor"``。
        mahalanobis_gate: 平方马氏距离门限 d^2 的阈值（仅最近邻模式使用）。
        resample_fraction: 重采样触发比例（N_eff 低于 N * 该值时重采样）。
        poses: (N, 3) 每粒子的位姿采样 (x, y, theta)，单位 (m, m, rad)。
        landmark_means: (N, K, 2) 每粒子每路标的高斯均值 (lx, ly)，单位 m；
            未初始化槽位为 0（是否有效以 ``observed`` 为准）。
        landmark_covs: (N, K, 2, 2) 每粒子每路标的 2x2 协方差，单位 m^2。
        observed: (N, K) 布尔掩码，标记该粒子是否已初始化该路标槽位。
        log_weights: (N,) 对数重要性权重，初始为均匀 ``-log(N)``。
        n_resamples: 累计触发系统重采样的次数（调试观察量）。
    """

    def __init__(
        self,
        n_particles: int = 50,
        n_landmarks: int = 20,
        motion_noise_std: Sequence[float] = (0.04, 0.04, 0.02),
        obs_noise_std: Sequence[float] = (0.15, 0.05),
        association: str = ASSOCIATION_KNOWN,
        mahalanobis_gate: float = GATE_DEFAULT,
        resample_fraction: float = 0.3,
        seed: int = 0,
    ) -> None:
        """初始化空粒子与滤波器超参数。

        Args:
            n_particles: 粒子数 ``M``（论文 Eq. (5)）。
            n_landmarks: 地图容量 ``K``；槽位在首次观测时填充。
            motion_noise_std: (3,) 机体系运动噪声标准差 (m, m, rad)，即运动
                模型的 ``R``（论文 Eq. (1)）。
            obs_noise_std: (2,) 测量噪声标准差 (距离 m, 方位角 rad)，即测量
                模型的 ``R``（论文 Eq. (2)）。
            association: ``"known"``（每条测量给出关联）或
                ``"nearest_neighbor"``（门控 ML 关联，论文 Eq. (12)）。
            mahalanobis_gate: 最近邻模式下对 ``d^2 = nu^T S^-1 nu`` 的门限，
                默认 :data:`GATE_DEFAULT`；取值背后的双尺度分析见其 docstring。
            resample_fraction: 当 ``N_eff < resample_fraction * n_particles``
                时重采样；取 0.3 而非教科书的 0.5，因为每次重采样本身也注入
                选择噪声，且 N=50 时一旦测量携带信息，它所防的退化就很少发生。
            seed: 运动噪声与重采样的 RNG 种子（保证确定性）。
        """
        if association not in (ASSOCIATION_KNOWN, ASSOCIATION_NEAREST_NEIGHBOR):
            raise ValueError(
                f"association must be '{ASSOCIATION_KNOWN}' or "
                f"'{ASSOCIATION_NEAREST_NEIGHBOR}', got {association!r}"
            )
        if n_particles < 1 or n_landmarks < 1:
            raise ValueError("n_particles and n_landmarks must be >= 1")
        if not 0.0 < resample_fraction <= 1.0:
            raise ValueError("resample_fraction must lie in (0, 1]")

        self.n_particles = int(n_particles)
        self.n_landmarks = int(n_landmarks)
        self.motion_noise_std = np.asarray(motion_noise_std, dtype=float).reshape(3)
        self.obs_noise_std = np.asarray(obs_noise_std, dtype=float).reshape(2)
        self._obs_cov = np.diag(self.obs_noise_std**2)
        self.association = association
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.resample_fraction = float(resample_fraction)
        self._rng = np.random.default_rng(seed)
        self.n_resamples = 0

        n, k = self.n_particles, self.n_landmarks
        self.poses = np.zeros((n, 3))
        self.landmark_means = np.zeros((n, k, 2))
        self.landmark_covs = np.zeros((n, k, 2, 2))
        self.observed = np.zeros((n, k), dtype=bool)
        self.log_weights = np.full(n, -np.log(n))

    def initialize(self, pose: Sequence[float]) -> None:
        """把滤波器重置为 ``pose`` 处的 delta 先验。

        所有粒子从同一已知位姿、零不确定性出发，地图清空、权重恢复均匀。
        滤波器所需的粒子多样性由后续步 Eq. (6) 的运动噪声产生。

        Args:
            pose: (3,) 已知起始位姿 ``(x, y, theta)``，单位 (m, m, rad)。
        """
        self.poses[:] = np.asarray(pose, dtype=float).reshape(3)
        self.landmark_means[:] = 0.0
        self.landmark_covs[:] = 0.0
        self.observed[:] = False
        self.log_weights[:] = -np.log(self.n_particles)

    def step(
        self,
        odom: np.ndarray,
        z: np.ndarray | None = None,
        assoc: np.ndarray | None = None,
    ) -> None:
        """一步 FastSLAM 递推（教程第 07 章 §7.3）。

        采样位姿（Eq. (6)），处理本步测量（关联 Eq. (12)、EKF 更新 Eq. (9)、
        权重 Eq. (7)/(11)），并在 ``N_eff`` 低于阈值时重采样。

        Args:
            odom: (3,) 机体系里程计增量 ``u_t``（论文 Eq. (1)），单位
                (m, m, rad)。
            z: (K_t, 2) 在*新*位姿处取得的距离-方位角测量（论文 Eq. (2)），
                单位 (m, rad)；没有观测时传 ``None`` 或空数组。
            assoc: (K_t,) 已知关联 ``n_t``；``"known"`` 模式必需，
                ``"nearest_neighbor"`` 模式忽略。
        """
        self._motion_update(odom)
        if z is not None and len(z) > 0:
            self._measurement_update(z, assoc)
        if self.effective_sample_size() < self.resample_fraction * self.n_particles:
            self.resample()

    def weights(self) -> np.ndarray:
        """归一化重要性权重 ``w_t^[m]``（论文 Eq. (7)）。

        从存储的对数权重出发，用 max-shift 技巧保证数值稳定。

        Returns:
            (N,) 非负权重，和为 1。
        """
        shifted = self.log_weights - self.log_weights.max()
        w = np.exp(shifted)
        return w / w.sum()

    def effective_sample_size(self) -> float:
        """有效样本数 ``N_eff = 1 / sum_i w_i^2``。

        标准的粒子退化诊断量，用于触发重采样（教程第 03 章 §3.3；
        :meth:`step` 内部使用）。权重均匀时 N_eff = N，塌缩到单粒子时
        N_eff = 1。
        """
        w = self.weights()
        return float(1.0 / np.sum(w * w))

    def resample(self) -> None:
        """就地执行系统（低方差）重采样。

        以单个均匀数 ``u`` 画有序网格 ``u_i = (i + u) / N``，在权重 CDF 上
        查找选中粒子；这使每个粒子的复制数与 ``N * w_i`` 相差不超过 1
        （教程第 03 章 §3.3）。被选粒子替换原集合，权重重置为均匀 ``1/N``，
        因此重采样后 ``sum(w) = 1`` 且 ``max(w) = 1/N``。
        """
        n = self.n_particles
        cdf = np.cumsum(self.weights())
        cdf[-1] = 1.0  # 防 CDF 尾部浮点舍入导致 searchsorted 取到越界下标
        positions = (np.arange(n) + self._rng.uniform()) / n
        idx = np.searchsorted(cdf, positions, side="left")
        self.poses = self.poses[idx]
        self.landmark_means = self.landmark_means[idx]
        self.landmark_covs = self.landmark_covs[idx]
        self.observed = self.observed[idx]
        self.log_weights = np.full(n, -np.log(n))
        self.n_resamples += 1

    def pose_estimate(self) -> np.ndarray:
        """粒子集上的加权平均位姿 ``(x, y, theta)``。

        ``theta`` 用圆均值 ``atan2(sum w sin, sum w cos)``，使估计不受
        ``(-pi, pi]`` 分支割线影响（粒子跨 ±pi 分布时算术平均会出错）。

        Returns:
            (3,) 位姿估计 (x, y, theta)，单位 (m, m, rad)。
        """
        w = self.weights()
        xy = w @ self.poses[:, :2]
        sin_theta = w @ np.sin(self.poses[:, 2])
        cos_theta = w @ np.cos(self.poses[:, 2])
        return np.array([xy[0], xy[1], float(np.arctan2(sin_theta, cos_theta))])

    def map_estimate(self) -> np.ndarray:
        """粒子上的加权平均路标地图。

        槽位 ``k`` 在已观测它的粒子上按权重 ``w_m`` 平均 ``mu_k^[m]``。
        已知关联模式下槽位与真值路标对齐；最近邻模式下槽位顺序是逐粒子的，
        评分前需先用匹配步骤把估计与真值对齐。

        Returns:
            (K, 2) 路标位置，单位 m；从未被任何粒子观测的槽位为 NaN。
        """
        w_obs = self.weights()[:, None] * self.observed
        denom = w_obs.sum(axis=0)
        numer = np.einsum("nk,nkc->kc", w_obs, self.landmark_means)
        map_est = np.full((self.n_landmarks, 2), np.nan)
        valid = denom > 0
        map_est[valid] = numer[valid] / denom[valid, None]
        return map_est

    def best_particle(self) -> int:
        """最可能粒子的下标（归一化权重最大者）。"""
        return int(np.argmax(self.weights()))

    def _motion_update(self, odom: np.ndarray) -> None:
        """采样 ``s_t^[m] ~ p(s_t | u_t, s_{t-1}^[m])``（论文 Eq. (6)）。

        机体系增量 ``u = (dx, dy, dtheta)`` 按当前朝向旋转到世界系；
        ``motion_noise_std`` 的 i.i.d. 高斯噪声加在机体系上（模块约定）。
        """
        o = np.asarray(odom, dtype=float).reshape(3)
        c = np.cos(self.poses[:, 2])
        s = np.sin(self.poses[:, 2])
        noise = self._rng.normal(
            scale=self.motion_noise_std, size=(self.n_particles, 3)
        )
        # 依据：模块约定的 body-frame 旋转（论文 Eq. (6) 的均值部分），
        # 噪声按约定加在机体系各分量上。
        self.poses[:, 0] += c * o[0] - s * o[1] + noise[:, 0]
        self.poses[:, 1] += s * o[0] + c * o[1] + noise[:, 1]
        self.poses[:, 2] = wrap_angle(self.poses[:, 2] + o[2] + noise[:, 2])

    def _measurement_update(self, z: np.ndarray, assoc: np.ndarray | None) -> None:
        """处理当前步的每条测量（论文 Eq. (9)-(12)）。

        Args:
            z: (K_t, 2) 堆叠的距离-方位角测量，单位 (m, rad)。
            assoc: (K_t,) ``"known"`` 模式的关联；最近邻模式下为 ``None``。

        Raises:
            ValueError: ``"known"`` 模式缺少配套的 ``assoc`` 数组时抛出。
        """
        z = np.asarray(z, dtype=float).reshape(-1, 2)
        if self.association == ASSOCIATION_KNOWN:
            if assoc is None:
                raise ValueError(
                    "association='known' requires the assoc argument"
                )
            assoc = np.asarray(assoc, dtype=int).reshape(-1)
            if assoc.shape[0] != z.shape[0]:
                raise ValueError("assoc and z must have the same length")
            for k in range(z.shape[0]):
                self._update_known(int(assoc[k]), z[k])
        else:
            for k in range(z.shape[0]):
                self._update_nearest(z[k])

    def _update_known(self, slot: int, z: np.ndarray) -> None:
        """每粒子已知关联 ``n_t = slot``。

        已观测的粒子做论文 Eq. (9) 的 EKF 更新；其余粒子按此次首次观测
        初始化路标（Eq. (10) 的逆命题 —— Eq. (10) 说的是已估路标保持不变）。
        """
        updated = np.nonzero(self.observed[:, slot])[0]
        initialized = np.nonzero(~self.observed[:, slot])[0]
        if updated.size:
            self._ekf_update_slots(updated, np.full(updated.size, slot), z)
        if initialized.size:
            self._initialize_slots(initialized, np.full(initialized.size, slot), z)

    def _update_nearest(self, z: np.ndarray) -> None:
        """按粒子的门控最近邻关联（论文 Eq. (12)）。

        对每个粒子，计算其到每个*已观测*路标的平方马氏距离
        ``d^2 = nu^T S^-1 nu``（nu 为测量新息，S 为似然协方差），ML 选择即
        ``argmin_j d^2``（FastSLAM 按粒子估计关联，不同于 EKF-SLAM 一次定死、
        无法回退）。

        - ``d^2 <= mahalanobis_gate``：对该路标做 EKF 更新；
        - 否则若该粒子还有空槽：在此初始化新路标（测量描述的是此前未见过的
          路标，即论文的地图增广规则）；
        - 否则：把 ML 似然 ``N(z; z_hat_j*, S_j*)`` 折入权重（论文 Eq. (7),
          (11)）但地图不动 —— 无法解释的测量必须在重采样时*惩罚*该粒子，
          绝不能被融合进一个与之矛盾的高斯。
        """
        dx = self.landmark_means[..., 0] - self.poses[:, None, 0]
        dy = self.landmark_means[..., 1] - self.poses[:, None, 1]
        r_sq = np.maximum(dx * dx + dy * dy, _RANGE_FLOOR)
        r = np.sqrt(r_sq)
        bearing = np.arctan2(dy, dx) - self.poses[:, None, 2]
        nu = np.stack([z[0] - r, wrap_angle(z[1] - bearing)], axis=-1)
        H = np.stack(
            [
                np.stack([dx / r, dy / r], axis=-1),
                np.stack([-dy / r_sq, dx / r_sq], axis=-1),
            ],
            axis=-2,
        )
        Ht = np.swapaxes(H, -1, -2)  # 转置末两维的 (2, 2) 块，保留 (N, M) 批量维
        S = H @ self.landmark_covs @ Ht + self._obs_cov
        S_inv, _ = _invert_2x2(S)
        d2 = np.einsum("...i,...ij,...j->...", nu, S_inv, nu)  # 平方马氏距离 d^2
        d2 = np.where(self.observed, d2, np.inf)  # 未初始化槽位不参与关联

        j_star = np.argmin(d2, axis=1)  # 每粒子的 ML 关联槽位
        d_min = d2[np.arange(self.n_particles), j_star]
        has_free = ~self.observed.all(axis=1)
        free_slot = np.argmax(~self.observed, axis=1)  # 第一个未观测槽位

        gated_in = d_min <= self.mahalanobis_gate
        do_init = ~gated_in & has_free
        p_ini = np.nonzero(do_init)[0]
        if p_ini.size:
            self._initialize_slots(p_ini, free_slot[p_ini], z)
        p_upd = np.nonzero(gated_in)[0]
        if p_upd.size:
            self._ekf_update_slots(p_upd, j_star[p_upd], z)
        p_unexplained = np.nonzero(~gated_in & ~has_free)[0]
        if p_unexplained.size:
            self._weight_only_slots(p_unexplained, j_star[p_unexplained], z)

    def _ekf_update_slots(
        self, p_idx: np.ndarray, s_idx: np.ndarray, z: np.ndarray
    ) -> None:
        """对选中粒子的一套路标做 EKF 更新（论文 Eq. (9)）。

        新息 ``nu = z - h(mu)``（方位角差已约束回 (-pi, pi]），增益
        ``K = Sigma H^T S^-1``，``S = H Sigma H^T + R``：

            ``mu <- mu + K nu``，  ``Sigma <- (I - K H) Sigma``。

        同一条测量似然 ``N(z; z_hat, S)`` —— 即 Eq. (11) 积分的 EKF 近似 ——
        被折入对数重要性权重（论文 Eq. (7)）。

        Args:
            p_idx: (n,) 待更新的粒子下标。
            s_idx: (n,) 每粒子对应的路标槽位（与 ``p_idx`` 等长）。
            z: (2,) 该条测量 (距离 m, 方位角 rad)。
        """
        mu = self.landmark_means[p_idx, s_idx]
        cov = self.landmark_covs[p_idx, s_idx]
        pose = self.poses[p_idx]
        z_hat, H = _rb_from_delta(
            mu[:, 0] - pose[:, 0], mu[:, 1] - pose[:, 1], pose[:, 2]
        )
        nu = np.stack(
            [z[0] - z_hat[:, 0], wrap_angle(z[1] - z_hat[:, 1])], axis=1
        )
        Ht = np.swapaxes(H, 1, 2)
        S = H @ cov @ Ht + self._obs_cov
        S_inv, det_S = _invert_2x2(S)
        gain = cov @ Ht @ S_inv
        mu_new = mu + np.matmul(gain, nu[..., None])[..., 0]
        cov_new = (np.eye(2) - gain @ H) @ cov
        cov_new = 0.5 * (cov_new + np.swapaxes(cov_new, 1, 2))  # 强制对称，防数值漂移

        self.landmark_means[p_idx, s_idx] = mu_new
        self.landmark_covs[p_idx, s_idx] = cov_new
        d2 = np.einsum("ni,nij,nj->n", nu, S_inv, nu)  # 平方马氏距离 d^2
        # 依据：Eq. (7)/(11) —— 对数似然 -0.5*(d^2 + log|2*pi*S|) 折入 log 权重；
        # det 下限 1e-300 防 log(0)。
        self.log_weights[p_idx] -= 0.5 * (
            d2 + 2.0 * _LOG_2PI + np.log(np.maximum(det_S, 1e-300))
        )

    def _weight_only_slots(
        self, p_idx: np.ndarray, s_idx: np.ndarray, z: np.ndarray
    ) -> None:
        """只把 ML 似然折入权重，不做 EKF 更新。

        与 :meth:`_ekf_update_slots` 相同的似然因子（论文 Eq. (7) 配
        Eq. (11) 近似），但路标高斯保持不变。用于 ML 路标越过门限且无空槽
        的情形：粒子保住自己的（低）似然，让重采样去淘汰它，而矛盾的测量
        永远不会被融进地图。
        """
        mu = self.landmark_means[p_idx, s_idx]
        cov = self.landmark_covs[p_idx, s_idx]
        pose = self.poses[p_idx]
        z_hat, H = _rb_from_delta(
            mu[:, 0] - pose[:, 0], mu[:, 1] - pose[:, 1], pose[:, 2]
        )
        nu = np.stack(
            [z[0] - z_hat[:, 0], wrap_angle(z[1] - z_hat[:, 1])], axis=1
        )
        S = H @ cov @ np.swapaxes(H, -1, -2) + self._obs_cov
        S_inv, det_S = _invert_2x2(S)
        d2 = np.einsum("ni,nij,nj->n", nu, S_inv, nu)
        # 依据：Eq. (7)/(11) —— 仅更新权重，路标高斯不动。
        self.log_weights[p_idx] -= 0.5 * (
            d2 + 2.0 * _LOG_2PI + np.log(np.maximum(det_S, 1e-300))
        )

    def _initialize_slots(
        self, p_idx: np.ndarray, s_idx: np.ndarray, z: np.ndarray
    ) -> None:
        """首次观测：按逆测量模型初始化路标高斯。

        在粒子位姿处反解测量模型，
        ``mu = (x + r cos(theta + phi), y + r sin(theta + phi))``，并把测量
        噪声经逆雅可比
        ``J = d(lx, ly)/d(r, phi) = [[cos, -r sin], [sin, r cos]]``
        传播，得 ``Sigma = J R J^T``。新初始化路标的新息恒为零，因此
        Eq. (11) 的似然因子退化为（近）常数归一化项 ``N(0; S)``；该槽位
        标记为已观测，后续观测由此路由到 :meth:`_ekf_update_slots` 的
        EKF 更新（论文 Eq. (9)）。
        """
        pose = self.poses[p_idx]
        r = float(z[0])
        ang = pose[:, 2] + float(z[1])
        c = np.cos(ang)
        s = np.sin(ang)
        mu = np.stack([pose[:, 0] + r * c, pose[:, 1] + r * s], axis=1)
        J = np.stack(
            [np.stack([c, -r * s], axis=1), np.stack([s, r * c], axis=1)], axis=1
        )
        cov = J @ self._obs_cov @ np.swapaxes(J, 1, 2)
        cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))  # 强制对称，防数值漂移
        det_S = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] * cov[..., 1, 0]

        self.landmark_means[p_idx, s_idx] = mu
        self.landmark_covs[p_idx, s_idx] = cov
        self.observed[p_idx, s_idx] = True
        # 依据：Eq. (11) 在 nu=0 时只剩归一化项 -0.5*log|2*pi*S|。
        self.log_weights[p_idx] -= 0.5 * (
            2.0 * _LOG_2PI + np.log(np.maximum(det_S, 1e-300))
        )

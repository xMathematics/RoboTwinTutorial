"""递归稠密 BA 结构演示（DROID-SLAM 教学实现，无学习组件）.

**Structural demo, no learned components**（结构演示，首行声明）：本模块把
Teed & Deng, ``DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D
Cameras``, NeurIPS 2021 (papers/slam/frontier/arXiv-2108.10869_DROID-SLAM.pdf)
的"循环更新算子 <-> 可微稠密 BA 层"交替结构抽出来教学演示——**无任何学习
组件**：论文的光流网络（Eq. (1) 全对相关体 + GRU 更新算子，RAFT 血统）被
替换为合成 GT 光流 + 高斯噪声，且噪声幅度随迭代轮数按几何速率收缩，以模拟
GRU 隐状态反馈使光流估计逐步收敛到定点（论文 §3.2 "with the expectation of
converging to a fixed point"）；论文的逐像素置信度权重 ``w_ij`` 取全 1
（Huber 核可选替代）。位姿/逐像素逆深度的状态、递归交替与重投影目标与论
文逐式对应。

式号对照（均指 DROID-SLAM 论文编号）
------------------------------------
- Eq. (1)  相关体 ``C``（光流来源）：替换为 :func:`make_sequence` 给出的合
  成 GT 光流 + 高斯噪声（首行声明的结构性替换）。
- Eq. (2)  深度回缩 ``d^(k+1) = Delta d^(k) + d^(k)``（向量加法）：
  :meth:`DenseBA.solve` 每轮的 ``rho <- rho + drho``（位姿以 SE(3) 指数回
  缩，等价于论文 "retraction on the SE3 manifold"）。
- Eq. (3)  对应场 ``p_ij = pi(G_ij . pi^{-1}(p_i, d_i))``、
  ``G_ij = G_j . G_i^{-1}``：:meth:`DenseBA.predicted_flow`——每轮开始时用
  当前 (位姿, 逆深度) 估计重新生成"更新光流"（Algorithm 1 的 Reset p_ij）。
- Eq. (4)  DBA 目标 ``E = sum ||r_ij * p_ij - pi(G'_ij . pi^{-1}(p_i,
  d'_i))||^2_{Sigma_ij}``：观测对应 = 宿主像素 + 当轮光流测量；修正项
  ``r_ij`` 在教学版中 = 当轮光流测量 - 当前预测光流（故 ``p* = p_pred +
  r_ij = p_i + f^{meas}``，逐项对应）；残差 ``e = 观测像素 - 预测投影``，
  解析雅可比复用 :mod:`direct` 的左扰动投影雅可比（教程 (5.11)/(6.5)，
  有限差分校验见 ``tests/test_droidlite.py``）。
- Eq. (5)  稠密 BA 的 Schur 补正规方程 ``Delta xi = [B - E C^{-1} E^T]^{-1}
  (v - E C^{-1} w)``（``C`` 对角、含阻尼 lambda）：教学版用稠密
  :func:`core.solver.gauss_newton` 解同一正规方程（教程第 08 章 (8.5)），
  不实现 Schur 稀疏求解；每个残差只触达一个位姿块与一个逆深度列（深度块
  对角，与论文的 Schur 结构来源一致）。
- Algorithm 1（附录）  ``[重置对应场 -> 光流修正 -> DBA 求解 -> 回缩]`` 的
  R 轮循环：:meth:`DenseBA.solve` 的主循环。

敏感度（实验记录，tests/test_droidlite.py 打印）
------------------------------------------------
光流噪声 sigma 增大时终端位姿/逆深度 RMSE 近似线性增大（线性化区的直接后
果：BA 解对光流观测是线性的）——测试以同一噪声底的不同 sigma 复现该趋势。

规范自由度（gauge）
--------------------
重投影误差对全体位姿的整体左乘不变（教程第 08 章 BA 规范自由度；论文以阻
尼最小二乘隐式固定）。教学版显式锚定第 0 帧位姿 ``G_0 = I``（第 0 帧为世
界系，也是所有点的宿主帧），其余位姿以左扰动坐标 ``T_k(x) = exp(xi_k^)
T_k^0`` 进入向量正规方程（教程 (8.9)-(8.10)）。

Conventions
-----------
- ``T_k``：world(= 第 0 帧机体系)-> 第 k 帧机体系的 (4, 4) 变换；
  ``X_cam = R @ X_world + t``（教程第 02 章记号）。
- 逆深度 ``rho_i = 1 / Z_i`` 挂在第 0 帧像素 ``p_i`` 的射线上，
  ``X = rho * K^{-1}[u, v, 1]^T``（DSO/VINS 逆深度参数化，教程 (10.10)）。
- "光流" ``f_ki = pi(T_k (rho_i ray_i)) - p_i``：第 0 帧像素到第 k 帧的
  几何对应位移；观测 = GT 光流 + 噪声。
"""
from dataclasses import dataclass

import numpy as np

from core.camera import PinholeCamera
from core.lie import se3_exp, se3_log, transform_points
from core.solver import gauss_newton, huber_weights
from direct import make_plane_scene, projection_jacobian

__all__ = ["DenseBA", "DroidLiteResult", "SequenceData", "make_sequence"]

#: 默认 GT 运动（6 维 xi = (tau, phi)，见 core.lie.se3_exp；平移在前），
#: 提供充分的平移视差使逐像素逆深度可观（纯旋转不产生深度信息）。
DEFAULT_MOTIONS = (
    np.array([0.09, 0.03, 0.02, 0.004, 0.008, -0.006]),
    np.array([0.16, 0.01, 0.05, -0.006, 0.012, 0.004]),
    np.array([0.24, -0.02, 0.03, 0.005, -0.009, 0.007]),
)


@dataclass(frozen=True)
class SequenceData:
    """合成序列：GT 位姿/深度/光流 + 噪声底（结构演示的"数据集"）。

    Attributes:
        images: K 帧 (H, W) 渲染图（真实系统中送入 Eq. (1) 相关体，此处仅
            作场景具体化）。
        cam: 针孔内参。
        poses_gt: K 个 (4, 4) GT 位姿（``poses_gt[0] = I``，世界系）。
        host_pixels: (N, 2) 宿主（第 0 帧）像素网格。
        rays: (N, 3) 宿主射线 ``K^{-1}[u, v, 1]^T``（第三分量恒 1）。
        rho_gt: (N,) 宿主像素处 GT 逆深度。
        flows_gt: (K-1, N, 2) GT 光流（第 0 帧 -> 第 k 帧的像素位移）。
        valid: (K-1, N) 观测有效掩码（GT 投影落在图像内）。
        flow_noise_base: (K-1, N, 2) 固定 N(0,1) 底噪（同种子可复现，便于
            不同 sigma 的受控对比）。
        flow_noise_std: 光流噪声标准差（像素）。
        grid: 每边网格数（``N = grid**2``）。
    """

    images: list[np.ndarray]
    cam: PinholeCamera
    poses_gt: list[np.ndarray]
    host_pixels: np.ndarray
    rays: np.ndarray
    rho_gt: np.ndarray
    flows_gt: np.ndarray
    valid: np.ndarray
    flow_noise_base: np.ndarray
    flow_noise_std: float
    grid: int


def make_sequence(
    n_frames: int = 4,
    width: int = 240,
    height: int = 180,
    focal: float = 300.0,
    grid: int = 32,
    z0: float = 3.0,
    plane_slope: tuple[float, float] = (0.05, -0.03),
    motions: tuple[np.ndarray, ...] | None = None,
    flow_noise_std: float = 0.3,
    margin: int = 14,
    seed: int = 7,
) -> SequenceData:
    """合成 textured-plane 序列并生成带噪 GT 光流（教学"数据集"）。

    场景复用 :mod:`direct` 的平面合成器（粗随机纹理 + 双线性上采样、正向
    warp 渲染），GT 深度由平面方程解析给出。光流由 GT 位姿与逐像素深度精
    确投影生成（第 0 帧像素 -> 第 k 帧），叠加零均值高斯噪声：
    ``flow_noise_base`` 为固定 N(0,1) 底噪、``flow_noise_std`` 为幅度——
    二者分离使不同噪声水平共享同一噪声实现（受控对比）。

    Args:
        n_frames: 帧数 K（第 0 帧 = 宿主帧，``motions`` 需给 K-1 条）。
        width, height, focal: 图像几何。
        grid: 宿主像素网格每边点数（DROID 的逐像素稠密逆深度在教学版降采
            样到 ``grid x grid``）。
        z0: 平面平均深度（米）。
        plane_slope: 平面倾斜 ``(s_u, s_v)``（见 :mod:`direct`）。
        motions: K-1 个 (6,) GT 运动 xi；默认 :data:`DEFAULT_MOTIONS`。
        flow_noise_std: 光流高斯噪声标准差（像素）。
        margin: 宿主网格与有效掩码的图像边距（像素）。
        seed: 纹理与噪声种子（确定性）。

    Returns:
        :class:`SequenceData`。

    Raises:
        ValueError: 帧数与运动条数不一致，或 ``n_frames < 2``。
    """
    if n_frames < 2:
        raise ValueError("need at least 2 frames")
    motions = DEFAULT_MOTIONS if motions is None else tuple(motions)
    if len(motions) != n_frames - 1:
        raise ValueError(f"expected {n_frames - 1} motions, got {len(motions)}")

    scene = make_plane_scene(
        width=width, height=height, focal=focal, z0=z0,
        plane_slope=plane_slope, seed=seed,
    )
    cam = scene.cam
    poses_gt = [np.eye(4)] + [se3_exp(np.asarray(m, dtype=float)) for m in motions]

    xs = np.linspace(margin, width - 1 - margin, grid)
    ys = np.linspace(margin, height - 1 - margin, grid)
    uu, vv = np.meshgrid(xs, ys)
    pixels = np.stack([uu.ravel(), vv.ravel()], axis=1)
    rays = np.stack(
        [
            (pixels[:, 0] - cam.cx) / cam.fx,
            (pixels[:, 1] - cam.cy) / cam.fy,
            np.ones(len(pixels)),
        ],
        axis=1,
    )
    rho_gt = 1.0 / scene.plane_depth(pixels[:, 0], pixels[:, 1])

    k_minus = n_frames - 1
    flows_gt = np.empty((k_minus, len(pixels), 2))
    valid = np.zeros((k_minus, len(pixels)), dtype=bool)
    for k in range(1, n_frames):
        q = transform_points(poses_gt[k], rays / rho_gt[:, None])
        uv = cam.project(q)
        flows_gt[k - 1] = uv - pixels
        valid[k - 1] = (
            (uv[:, 0] > 1.0) & (uv[:, 0] < width - 2.0)
            & (uv[:, 1] > 1.0) & (uv[:, 1] < height - 2.0)
        )

    rng = np.random.default_rng(seed)
    noise_base = rng.normal(size=(k_minus, len(pixels), 2))
    images = [scene.texture] + [scene.render_target(T) for T in poses_gt[1:]]
    return SequenceData(
        images=images,
        cam=cam,
        poses_gt=poses_gt,
        host_pixels=pixels,
        rays=rays,
        rho_gt=rho_gt,
        flows_gt=flows_gt,
        valid=valid,
        flow_noise_base=noise_base,
        flow_noise_std=float(flow_noise_std),
        grid=grid,
    )


@dataclass(frozen=True)
class DroidLiteResult:
    """:meth:`DenseBA.solve` 的输出（轨迹、地图与逐轮诊断）。

    Attributes:
        poses: K 个 (4, 4) 优化位姿（第 0 帧固定为 I）。
        inverse_depths: (N,) 优化逆深度。
        pose_rmse: (R+1,) 逐轮位姿 RMSE（``se3_log`` 6 维范数，第 0 项为
            初值；1 rad 与 1 m 等权，教学指标）。
        depth_rmse: (R+1,) 逐轮逆深度 RMSE（1/m）。
        flow_cost: (R+1,) 逐轮光流代价（第 0 项为初值处）。
        flow_residual: (M, 2) 终端逐对应光流残差（像素，对 GT 光流）。
        n_rounds: 递归轮数 R。
    """

    poses: list[np.ndarray]
    inverse_depths: np.ndarray
    pose_rmse: np.ndarray
    depth_rmse: np.ndarray
    flow_cost: np.ndarray
    flow_residual: np.ndarray
    n_rounds: int


class DenseBA:
    """DROID 式递归稠密 BA（Eq. (2)-(5) + Algorithm 1 的无学习结构演示）。

    变量 = 全体位姿 ``T_1..T_{K-1}``（左扰动坐标，第 0 帧锚定）+ 宿主网格
    每像素逆深度 ``rho_i``。:meth:`solve` 每轮执行（Algorithm 1）：

    1. **更新光流（Eq. (3)）**：用当前 (位姿, 逆深度) 重新生成对应场
       ``p_ij``（:meth:`predicted_flow`）；
    2. **光流修正（Eq. (4) 的 ``r_ij * p_ij``）**：当轮光流测量
       ``f_gt + sigma * decay^r * noise``（模拟 GRU 收敛的合成调度）减去
       当前预测作为修正项，得到修正对应 ``p* = p_pred + r_ij``；
    3. **稠密 BA 增量（Eq. (4)-(5)）**：对重投影残差做一次 LM 步——按论
       文 Eq. (5) 的 Schur 补正规方程显式消元（深度块对角 =>
       ``[B - E C^{-1} E^T + lambda I] Delta xi = v - E C^{-1} w``、
       ``Delta rho = C^{-1}(w - E^T Delta xi)``，:meth:`_solve_ba_increment`），
       解析雅可比见 :meth:`jacobian_matrix`；稠密整体正规方程的等价解由
       :func:`core.solver.gauss_newton` 在 ``tests/test_droidlite.py`` 中交
       叉验证（Schur 消元与稠密求解数值一致）；
    4. **回缩（Eq. (2)）**：``T_k <- exp(Delta xi_k^) T_k``、
       ``rho <- rho + Delta rho``，进入下一轮。

    Args:
        seq: :class:`SequenceData`（GT 与噪声底）。
        poses_init: K 个 (4, 4) 初值位姿（``poses_init[0]`` 被忽略，恒取
            ``I`` 以锚定规范自由度）。
        rho_init: 初值逆深度：标量（全体常数——"逆深度置常数"演示）或 (N,)。
        flow_decay: 光流噪声逐轮收缩率 ``sigma * flow_decay^round``
            （GRU 收敛调度的教学替身；< 1 收敛，= 1 不细化）。
        inner_iters: 每轮 BA 增量的内层 LM 迭代上限。
        lm_lambda: LM 阻尼（论文按像素置信度给 ``C`` 加阻尼，教学版取标量，
            作用于 Eq. (5) 的 ``C`` 与消元后的位姿块）。
        huber_delta: 可选 Huber 核宽度（像素）；``None`` = 纯最小二乘
            （论文以置信度 ``w_ij`` 加权，教学版取全 1）。
        ba_tol: 内层迭代的相对代价下降阈值（提前收敛判定）。

    Raises:
        ValueError: 初值形状与序列不一致或参数非法。
    """

    def __init__(
        self,
        seq: SequenceData,
        poses_init: list[np.ndarray],
        rho_init: float | np.ndarray,
        flow_decay: float = 0.5,
        inner_iters: int = 4,
        lm_lambda: float = 1e-6,
        huber_delta: float | None = None,
        ba_tol: float = 1e-8,
        use_dense_gn: bool = False,
    ) -> None:
        k = len(seq.poses_gt)
        if len(poses_init) != k:
            raise ValueError(f"expected {k} init poses, got {len(poses_init)}")
        self._seq = seq
        self._k = k
        self._k_minus = k - 1
        self._n = len(seq.host_pixels)
        self._poses0 = [np.eye(4)] + [
            np.asarray(T, dtype=float).reshape(4, 4) for T in poses_init[1:]
        ]
        self._rho0 = (
            np.full(self._n, float(rho_init))
            if np.isscalar(rho_init)
            else np.asarray(rho_init, dtype=float).reshape(self._n)
        )
        if not 0.0 < flow_decay <= 1.0:
            raise ValueError("flow_decay must lie in (0, 1]")
        self._flow_decay = float(flow_decay)
        self._inner_iters = int(inner_iters)
        self._lm_lambda = float(lm_lambda)
        self._huber_delta = huber_delta
        self._ba_tol = float(ba_tol)
        self._use_dense_gn = bool(use_dense_gn)

        # 预展开有效 (帧 k, 像素 i) 对：每对贡献 u、v 两行，行序与残差堆叠
        # 一致（u0, v0, u1, v1, ...）。``_pix0`` 为各帧像素块的起点，
        # ``_row0 = 2 * _pix0`` 为残差行的起点；每个残差只触达一个逆深度
        # 列 => 深度块对角（论文 Eq. (5) 中 Schur 消元的结构基础）。
        ks, ii = np.nonzero(seq.valid)
        self._ks, self._ii = ks, ii
        self._m = len(ks)
        counts = np.bincount(ks, minlength=self._k_minus)
        self._pix0 = np.concatenate([[0], np.cumsum(counts)])
        self._row0 = 2 * self._pix0
        self._sel = [ii[ks == k] for k in range(self._k_minus)]
        self._dim = 6 * self._k_minus + self._n

    # ------------------------------------------------------------------ #
    # Eq. (3)：由当前估计生成对应场（correspondence field，预测光流）      #
    # ------------------------------------------------------------------ #
    def predicted_flow(self, poses: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
        """用当前估计生成对应场/预测光流 (K-1, N, 2)（Eq. (3)、Alg. 1 复位）。

        Args:
            poses: K 个 (4, 4) 当前位姿（第 0 帧为宿主/世界系）。
            rho: (N,) 当前逆深度。

        Returns:
            (K-1, N, 2) 预测光流 ``pi(G_k . pi^{-1}(p_i, d_i)) - p_i``。
        """
        seq = self._seq
        out = np.empty((self._k_minus, self._n, 2))
        for k in range(1, self._k):
            q = transform_points(poses[k], seq.rays / rho[:, None])
            out[k - 1] = seq.cam.project(q) - seq.host_pixels
        return out

    # ------------------------------------------------------------------ #
    # Eq. (4)：稠密 BA（dense BA）增量的残差与解析雅可比                   #
    # ------------------------------------------------------------------ #
    def residuals(
        self, x: np.ndarray, poses: list[np.ndarray], rho: np.ndarray,
        obs_pixels: np.ndarray,
    ) -> np.ndarray:
        """堆叠重投影残差 (2M,)：``e = p*_{ki} - pi(T_k(x) (rho ray_i))``。

        Args:
            x: (dim,) 增量状态 ``[xi_1..xi_{K-1} | drho_1..N]``。
            poses: K 个 (4, 4) 线性化点（本轮入口位姿）。
            rho: (N,) 本轮入口逆深度。
            obs_pixels: (M, 2) 修正对应 ``p*``（按 (k, i) 有效对展开，顺序
                与 :attr:`_ks`/`_ii` 一致）。
        """
        seq = self._seq
        dxi = x[: 6 * self._k_minus].reshape(self._k_minus, 6)
        drho = x[6 * self._k_minus:]
        rho_x = rho + drho
        out = np.empty(2 * self._m)
        for k in range(self._k_minus):
            sel = self._sel[k]
            T_k = se3_exp(dxi[k]) @ poses[k + 1]
            q = transform_points(T_k, seq.rays[sel] / rho_x[sel, None])
            e = obs_pixels[self._pix0[k]: self._pix0[k + 1]] - seq.cam.project(q)
            out[self._row0[k]: self._row0[k + 1]] = e.reshape(-1)
        return out

    def jacobian_matrix(
        self, x: np.ndarray, poses: list[np.ndarray], rho: np.ndarray
    ) -> np.ndarray:
        """解析雅可比 (2M, dim)（Eq. (4) 残差对 (Delta xi, Delta rho)）。

        位姿列 = ``-G(q)``（:func:`direct.projection_jacobian` 的左扰动
        2x6；残差取"观测 - 预测"故整体变号）；逆深度列：由
        ``q = R_k ray_i / rho_i + t_k`` 得 ``dq/drho_i = -(R_k ray_i)/rho_i^2``，
        故 ``de/drho_i = +(dpi/dq)(R_k ray_i)/rho_i^2``（残差"观测-预测"），
        且第 (k, i) 对残差只触达列 ``6(K-1) + i``——深度块对角。

        Args:
            x, poses, rho: 与 :meth:`residuals` 相同（雅可比与观测无关）。
        """
        seq = self._seq
        dxi = x[: 6 * self._k_minus].reshape(self._k_minus, 6)
        drho = x[6 * self._k_minus:]
        rho_x = rho + drho
        J = np.zeros((2 * self._m, self._dim))
        for k in range(self._k_minus):
            sel = self._sel[k]
            T_k = se3_exp(dxi[k]) @ poses[k + 1]
            q = transform_points(T_k, seq.rays[sel] / rho_x[sel, None])
            G = projection_jacobian(q, seq.cam)                # (Mk, 2, 6)
            J[self._row0[k]: self._row0[k + 1]: 2, 6 * k: 6 * k + 6] = -G[:, 0, :]
            J[self._row0[k] + 1: self._row0[k + 1]: 2, 6 * k: 6 * k + 6] = -G[:, 1, :]
            ray_t = seq.rays[sel] @ T_k[:3, :3].T              # R_k ray_i
            d_uv_drho = np.einsum("nij,nj->ni", G[:, :, :3], ray_t)
            d_uv_drho /= rho_x[sel, None] ** 2
            # u、v 两行都只依赖 rho_i => 列索引按像素重复两次（行序 u, v）。
            J[np.arange(self._row0[k], self._row0[k + 1]),
              6 * self._k_minus + np.repeat(sel, 2)] = d_uv_drho.reshape(-1)
        return J

    # ------------------------------------------------------------------ #
    # Eq. (5)：稠密 BA 增量的 Schur complement（舒尔补）LM 步              #
    # ------------------------------------------------------------------ #
    def _weights(self, e: np.ndarray) -> np.ndarray:
        """IRLS 权重：Huber（复用 :func:`core.solver.huber_weights`）或全 1。"""
        if self._huber_delta is None:
            return np.ones_like(e)
        return huber_weights(e, self._huber_delta)

    def _schur_step(
        self,
        x: np.ndarray,
        poses: list[np.ndarray],
        rho: np.ndarray,
        obs_pixels: np.ndarray,
        lam: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """一步 Eq. (5) 的 Schur 补正规方程（返回 ``(dx, e_new, w_new, cost)``）。

        正规方程按块写为
        ``[[B, E], [E^T, C]] [dxi; drho] = -[gx; gd]``，其中 ``B``、``E``、
        ``C`` 分别为加权雅可比的位姿-位姿、位姿-深度、深度-深度块；``C``
        为对角（每个残差只触达一个深度列，Eq. (5) 的结构基础），故消元给

        ``dxi = [B - E C^{-1} E^T + lambda I]^{-1} (-(gx - E C^{-1} gd))``、
        ``drho = C^{-1} (-(gd + E^T dxi))``——与论文 Eq. (5) 逐项对应
        （论文记号 ``v = -gx``、``w = -gd``；论文把逐像素置信度阻尼放进
        ``C``，教学版用标量 LM 阻尼）。稠密整体正规方程的等价解由
        :func:`core.solver.gauss_newton` 在 ``tests/test_droidlite.py``
        中交叉验证。
        """
        e = self.residuals(x, poses, rho, obs_pixels)
        J = self.jacobian_matrix(x, poses, rho)
        w = self._weights(e)
        e_w = w * e
        Jw = w[:, None] * J
        k1 = 6 * self._k_minus
        Jx, Jd = Jw[:, :k1], Jw[:, k1:]
        B = Jx.T @ Jx                                    # (6(K-1), 6(K-1))
        E = Jx.T @ Jd                                    # (6(K-1), N)
        C = np.einsum("ij,ij->j", Jd, Jd) + lam          # 深度块（对角）
        gx, gd = Jx.T @ e_w, Jd.T @ e_w
        A = B - (E / C) @ E.T + lam * np.eye(k1)
        dxi = np.linalg.solve(A, -(gx - E @ (gd / C)))
        drho = -(gd + E.T @ dxi) / C
        e_new = self.residuals(x + np.concatenate([dxi, drho]), poses, rho, obs_pixels)
        w_new = self._weights(e_new)
        dx = np.concatenate([dxi, drho])
        return dx, e_new, w_new, float(np.sum(w_new * e_new**2))

    def _solve_ba_increment(
        self, obs_pixels: np.ndarray, poses: list[np.ndarray], rho: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """本轮 BA 增量的内层 LM 循环（接受/拒绝 + 相对代价下降终止）。

        ``use_dense_gn=False``（默认）走 Eq. (5) 的 Schur 补消元；置 True
        时改由 :func:`core.solver.gauss_newton` 解完整稠密正规方程——同一
        目标的稠密等价解（``tests/test_droidlite.py`` 验证两条路径收敛到
        同一状态），代价是每步 O(dim^3) 而非消元后的 O(6(K-1)^3)。
        """
        if self._use_dense_gn:
            sol = gauss_newton(
                lambda x: self.residuals(x, poses, rho, obs_pixels),
                lambda x: self.jacobian_matrix(x, poses, rho),
                x0=np.zeros(self._dim),
                n_iters=self._inner_iters,
                lm_lambda=self._lm_lambda,
                robust_delta=self._huber_delta,
            )
            return sol.x, float(sol.cost)
        x = np.zeros(self._dim)
        e = self.residuals(x, poses, rho, obs_pixels)
        w = self._weights(e)
        cost = float(np.sum(w * e * e))
        lam = self._lm_lambda
        for _ in range(self._inner_iters):
            dx, e_new, w_new, cost_new = self._schur_step(
                x, poses, rho, obs_pixels, lam
            )
            if not np.isfinite(cost_new) or cost_new >= cost:
                lam = max(lam * 10.0, 1e-12)             # 拒绝、加大阻尼
                if lam > 1e10:
                    break
                continue
            lam = max(lam / 10.0, 1e-12)                 # 接受、放松阻尼
            improvement = cost - cost_new
            x, e, w, cost = x + dx, e_new, w_new, cost_new
            if improvement <= self._ba_tol * max(cost, 1e-300):
                break
        return x, cost

    # ------------------------------------------------------------------ #
    # Algorithm 1：[光流更新 <-> 稠密 BA] 的递归主循环                     #
    # ------------------------------------------------------------------ #
    def solve(self, n_rounds: int = 6) -> DroidLiteResult:
        """跑 R 轮 [更新光流 -> 修正对应 -> 稠密 BA 增量 -> 回缩]（Alg. 1）。

        Args:
            n_rounds: 递归轮数 R。

        Returns:
            :class:`DroidLiteResult`（含逐轮 RMSE，供单调性检查）。
        """
        seq = self._seq
        poses = [T.copy() for T in self._poses0]
        rho = self._rho0.copy()

        pose_rmse = [self._pose_rmse(poses)]
        depth_rmse = [self._depth_rmse(rho)]
        flow_cost = [self._flow_cost(poses, rho)]
        for r in range(1, n_rounds + 1):
            # 依据：GRU 收敛调度——噪声幅度按 decay^round 几何收缩，模拟论文
            # GRU 更新算子的隐状态反馈使光流逐步收敛到定点（论文 §3.2）。
            sigma_r = self._flow_decay ** r
            flow_pred = self.predicted_flow(poses, rho)        # Eq. (3)
            # Eq. (4) 修正项 r_ij = 当轮光流测量 - 当前预测；修正对应
            # p* = p_pred + r_ij = p_i + f^{meas}（见模块 docstring）。
            flow_meas = (seq.flows_gt
                         + sigma_r * seq.flow_noise_std * seq.flow_noise_base)
            obs = flow_pred + (flow_meas - flow_pred)
            obs_pixels = seq.host_pixels[self._ii] + obs[self._ks, self._ii]
            x, cost = self._solve_ba_increment(obs_pixels, poses, rho)
            dxi = x[: 6 * self._k_minus].reshape(self._k_minus, 6)
            drho = x[6 * self._k_minus:]
            poses = [poses[0]] + [                             # Eq. (2) 回缩
                se3_exp(dxi[k]) @ poses[k + 1] for k in range(self._k_minus)
            ]
            rho = rho + drho
            pose_rmse.append(self._pose_rmse(poses))
            depth_rmse.append(self._depth_rmse(rho))
            flow_cost.append(cost)

        # 终端光流残差（对干净 GT 光流，供收敛质量诊断）。
        gt_px = seq.host_pixels[self._ii] + seq.flows_gt[self._ks, self._ii]
        flow_res = np.empty((self._m, 2))
        for k in range(self._k_minus):
            sel = self._sel[k]
            q = transform_points(poses[k + 1], seq.rays[sel] / rho[sel, None])
            flow_res[self._ks == k] = gt_px[self._ks == k] - seq.cam.project(q)
        return DroidLiteResult(
            poses=poses,
            inverse_depths=rho,
            pose_rmse=np.array(pose_rmse),
            depth_rmse=np.array(depth_rmse),
            flow_cost=np.array(flow_cost),
            flow_residual=flow_res,
            n_rounds=n_rounds,
        )

    # ------------------------------------------------------------------ #
    # 内部工具                                                            #
    # ------------------------------------------------------------------ #
    def _pose_rmse(self, poses: list[np.ndarray]) -> float:
        """位姿 RMSE：``sqrt(mean_k ||Log(T_k^gt^{-1} T_k)||^2)``（6 维）。"""
        errs = []
        for k in range(1, self._k):
            xi = se3_log(np.linalg.inv(self._seq.poses_gt[k]) @ poses[k])
            errs.append(xi @ xi)
        return float(np.sqrt(np.mean(errs)))

    def _depth_rmse(self, rho: np.ndarray) -> float:
        """逆深度 RMSE：``sqrt(mean (rho - rho_gt)^2)``（1/m）。"""
        d = rho - self._seq.rho_gt
        return float(np.sqrt(np.mean(d * d)))

    def _flow_cost(self, poses: list[np.ndarray], rho: np.ndarray) -> float:
        """初值处的光流代价（与 gauss_newton 的无 1/2 约定一致）。"""
        flow = self.predicted_flow(poses, rho)
        obs = self._seq.flows_gt + self._seq.flow_noise_std * self._seq.flow_noise_base
        resid = (obs - flow)[self._ks, self._ii]
        return float(np.sum(resid * resid))

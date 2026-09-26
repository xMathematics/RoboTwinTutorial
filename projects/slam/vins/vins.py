"""紧耦合视觉-惯性捆绑调整（VINS-Mono 简化教学实现）。

函数流水线
----------
教程第 10 章 §10.3（式 (10.9)-(10.11)）的教学实现：滑动窗口内关键帧的状态
``x_k = (T_wb, v, b^g, b^a)`` 由两类因子联合估计——IMU 预积分因子（第 10 章
§10.2；Forster et al., T-RO 2017 论文 Eq.(45)+(48)）与单目归一化坐标重投影
因子（第 08 章重投影误差 (8.1)）——外层用 Levenberg-Marquardt 在 SE(3)
流形上迭代求解（第 08 章 (8.5)-(8.8)；回缩/retraction 即论文 Eq.(21)）。

完整调用链：:func:`simulate_vi_scene`（合成走廊飞行场景，故意给出错误尺度
的初始化）→ :func:`build_vins_bundle`（组装 (10.9) 的因子图）→
:meth:`ViBundle.solve`（LM 流形求解：IMU 因子数值雅可比 + 视觉因子解析
雅可比）→ :func:`align_se3`（Kabsch/Umeyama 对齐后评估轨迹）。场景数据由
:class:`VISimulation` 承载，结果由 :class:`ViResult` 承载。复用
``core.lie`` / ``core.solver.huber_weights`` / ``epipolar.triangulate``。
由 ``tests/test_vins.py`` 直接驱动。

论文：T. Qin, P. Li, S. Shen, "VINS-Mono: A Robust and Versatile Monocular
Visual-Inertial State Estimator", IEEE T-RO 2018
（``papers/slam/classics/arXiv-1708.03852_VINS-Mono.pdf``）。

尺度可观性（教程 §10.3 第 3 步）
--------------------------------
视觉代价对世界系整体缩放 ``x → s·x`` 不变（透视除法消去尺度），因此*纯视觉*
捆绑存在尺度规范自由度（scale gauge freedom）。IMU 因子残差 (10.10) 由带
物理单位的加速度计读数与*已知*重力 ``g`` 构成，打破了尺度规范：缩放轨迹
不改变视觉项、却改变 IMU 残差，联合问题因此钉住绝对尺度（以及横滚/俯仰），
只剩单目 VIO 的 4 自由度规范（全局 yaw + 全局平移）。
:func:`simulate_vi_scene` 故意把初始状态设成错误尺度以演示恢复过程；
``ViBundle.solve(imu_weight=0)`` 复现退化的纯视觉运行。

教学简化（相对 VINS-Mono）
--------------------------
- 路标是全局 XYZ 点（非逆深度参数化），由初始位姿经 DLT 三角化
  （:func:`epipolar.triangulate`）得到。
- IMU 因子雅可比是**数值**的（中心差分穿过求解器实际更新所用的同一回缩）；
  重投影雅可比是解析的。取舍讨论见 :meth:`ImuFactor.jacobian`。
- 相机-IMU 外参为单位阵（相机系 == IMU/机体系），且重力矢量假设已知
  （VINS-Mono 在线估计重力方向）。

约定
----
- ``T_wb`` 是机体系（-相机系）到世界系的变换：``x_w = R_wb x_b + p_wb``，
  即 ``x_body = R_wb^T (x_w − p_wb)``；``R_wb = T_wb[:3, :3]``。
- 每个关键帧的切向量 (15,)：
  ``[δτ(3), δφ(3), δv(3), δb^g(3), δb^a(3)]``，回缩为
  ``T_wb ← Exp(δξ^) T_wb``（**左**乘，``δξ = (δτ, δφ)`` 按
  :func:`core.lie.se3_exp` 的参数顺序），``v, b ← v + δv, b + δb`` —— 与
  第 08 章 (8.7) 及 :func:`epipolar.pnp_refine` 相同的流形更新。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.lie import hat, se3_exp, so3_exp, so3_log, vee
from core.solver import huber_weights
from epipolar import triangulate
from preint import ImuParams, Preintegration

__all__ = [
    "KeyframeState",
    "ImuFactor",
    "ReprojectionFactor",
    "ViBundle",
    "ViResult",
    "VISimulation",
    "align_se3",
    "build_vins_bundle",
    "simulate_vi_scene",
]


def _retract_state(state: "KeyframeState", dxi: np.ndarray) -> "KeyframeState":
    """按 15 维切向量步回缩（retract）单个关键帧状态（位姿**左乘**更新）。"""
    dxi = np.asarray(dxi, dtype=float).reshape(15)
    return KeyframeState(
        T_wb=se3_exp(dxi[:6]) @ state.T_wb,
        v=state.v + dxi[6:9],
        b_g=state.b_g + dxi[9:12],
        b_a=state.b_a + dxi[12:15],
    )


def _numeric_jacobian(residual_fn, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """残差映射在切空间上的中心差分（central-difference）雅可比。

    ``residual_fn`` 需接受切向量并在内部完成回缩，因此求出的导数正是求解器
    线性化的那个流形残差的导数（教学取舍，讨论见 :meth:`ImuFactor.jacobian`）。

    Args:
        residual_fn: 切向量 ``x -> e`` 的残差函数（内部自行回缩）。
        x0: (n,) 求导点（切向量）。
        eps: 中心差分布长（默认 1e-6：舍入噪声 ∝ 1/eps、截断误差 ∝ eps²）。

    Returns:
        (m, n) 雅可比矩阵：m 为残差维数，n 为切向量维数。
    """
    x0 = np.asarray(x0, dtype=float)
    e0 = np.asarray(residual_fn(x0), dtype=float)
    J = np.empty((e0.size, x0.size))
    for i in range(x0.size):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (
            np.asarray(residual_fn(xp), dtype=float)
            - np.asarray(residual_fn(xm), dtype=float)
        ) / (2.0 * eps)
    return J


@dataclass
class KeyframeState:
    """单个关键帧：SE(3) 机体系位姿、世界系速度与 IMU 零偏。"""

    T_wb: np.ndarray  # (4, 4) 机体系→世界系变换
    v: np.ndarray     # (3,) 世界系速度 [m/s]
    b_g: np.ndarray   # (3,) 陀螺仪零偏 [rad/s]
    b_a: np.ndarray   # (3,) 加速度计零偏 [m/s²]


class ImuFactor:
    """关键帧 ``i`` 与 ``j`` 之间的 IMU 预积分因子。

    包装由该区间原始 IMU 采样构建的 :class:`preint.Preintegration`。
    残差（15 维，已白化）：三个预积分测量模型残差（paper Eq.(45)，
    教程 (10.10)；旋转残差经 ``Log`` 映射写入向量空间 —— 依据：第 02 章
    (2.25)，SO(3) 残差必须先经 ``Log`` 提升，才能被高斯分布加权）外加零偏
    随机游走残差（paper Eq.(48)，``Σ^{bgd/bad} = σ_b² Δt_ij I``）::

        r_ΔR = Log( (ΔR̃_ij Exp(J δb^g))^T R_i^T R_j ),   δb = b_i − b̄,
        r_Δv = R_i^T (v_j − v_i − g Δt_ij) − (Δṽ_ij + J_v δb),
        r_Δp = R_i^T (p_j − p_i − v_i Δt_ij − ½ g Δt_ij²) − (Δp̃_ij + J_p δb),
        r_b  = [b_j − b_i] / (σ_b √Δt_ij),

    全部经 9×9 预积分协方差 ``Σ_ij`` 加权（Cholesky 白化；零偏随机游走部分
    使用其自身的对角协方差）。
    """

    dim = 15

    def __init__(self, i: int, j: int, preint: Preintegration) -> None:
        """缓存预积分结果、Σ_ij 的 Cholesky 因子与零偏随机游走标准差。"""
        self.i = int(i)
        self.j = int(j)
        self.preint = preint
        # Σ_ij 的 Cholesky 因子：白化把马氏范数 ‖e‖²_Σ 化为普通最小二乘代价
        # （第 08 章 (8.2)）。
        self._chol = np.linalg.cholesky(preint.cov)
        p = preint.params
        walk = preint.duration
        self._bias_std = np.concatenate(
            [np.full(3, p.sigma_bg * np.sqrt(walk)),
             np.full(3, p.sigma_ba * np.sqrt(walk))]
        )

    def residual(self, si: KeyframeState, sj: KeyframeState) -> np.ndarray:
        """给定两端点状态处的白化 (15,) 残差。

        Args:
            si: 关键帧 ``i`` 的状态估计（:class:`KeyframeState`）。
            sj: 关键帧 ``j`` 的状态估计（:class:`KeyframeState`）。

        Returns:
            (15,) ``[r_ΔR; r_Δv; r_Δp; r_b]``：前 9 维经 ``Σ_ij`` 的 Cholesky
            白化（各分量近似 N(0, 1)），后 6 维为零偏随机游走白化残差。
        """
        b_bar = self.preint.bias0
        delta_b = np.concatenate([si.b_g - b_bar[:3], si.b_a - b_bar[3:]])
        d_r, d_v, d_p = self.preint.correct(delta_b)  # paper Eq.(44) 更新
        g = self.preint.params.g
        T = self.preint.duration
        R_i = si.T_wb[:3, :3]
        R_j = sj.T_wb[:3, :3]
        r_r = so3_log(d_r.T @ R_i.T @ R_j)
        r_v = R_i.T @ (sj.v - si.v - g * T) - d_v
        p_i = si.T_wb[:3, 3]
        p_j = sj.T_wb[:3, 3]
        r_p = R_i.T @ (p_j - p_i - si.v * T - 0.5 * g * T**2) - d_p
        e9 = np.concatenate([r_r, r_v, r_p])
        r_b = np.concatenate([sj.b_g - si.b_g, sj.b_a - si.b_a]) / self._bias_std
        return np.concatenate([np.linalg.solve(self._chol, e9), r_b])

    def jacobian(
        self, si: KeyframeState, sj: KeyframeState
    ) -> tuple[np.ndarray, np.ndarray]:
        """对两端点切向量的雅可比（数值）。

        Returns:
            ``(J, idx)``：``J`` 形状 (15, 30)，列 ``[0:15]`` 对应关键帧 ``i``
            的切向量、``[15:30]`` 对应关键帧 ``j`` 的切向量；``idx`` 为二者在
            全局参数矢量中的下标。

        数值 vs 解析的取舍（教学取舍）：论文推导解析残差雅可比（附录 IX-C，
        Eq.(70)-(81)）是为了避免以 IMU 高采样率反复对状态线性化；这里残差
        很廉价（预积分已缓存）且窗口只有 4 个关键帧，因此每次外层迭代每因子
        60 次中心差分求值只花毫秒级时间，还免去了手推分块矩阵的符号/记账
        风险。求导**穿过求解器实际更新所用的同一回缩**（:func:`_retract_state`），
        这正是它们构成外层 LM 循环的正确流形雅可比的原因。
        """
        base = np.zeros(30)

        def residual_of(dx: np.ndarray) -> np.ndarray:
            return self.residual(
                _retract_state(si, dx[:15]), _retract_state(sj, dx[15:])
            )

        J = _numeric_jacobian(residual_of, base)
        idx = np.concatenate(
            [np.arange(15 * self.i, 15 * self.i + 15),
             np.arange(15 * self.j, 15 * self.j + 15)]
        )
        return J, idx


class ReprojectionFactor:
    """单目路标重投影因子（归一化坐标）。

    残差（经像素/归一化坐标标准差 ``σ`` 白化）：
    ``e = (z_obs − z_pred) / σ``，其中 ``z_pred = [x_c/z_c, y_c/z_c]``、
    ``x_c = R_wb^T (X − p_wb)`` —— 第 08 章重投影误差 (8.1) 取 ``K = I``
    （归一化平面观测，即第 04 章 (4.5) 的反投影）。

    解析雅可比沿用 :func:`epipolar.reprojection_jacobian`（式 5.11）的推导并
    适配到本模块的状态参数化。彼处位姿是世界系→相机系、左扰动；此处状态是
    机体系→世界系 ``T_wb ← Exp(δξ^) T_wb``（``δξ = (δτ, δφ)``，即求解器的
    回缩）。一阶近似下（依据：``Exp(δφ) ≈ I + δφ^`` 与叉积搬运恒等式
    ``Rᵀ(u × v) = (Rᵀu) × (Rᵀv)``）左扰动位姿带着相机中心一起动，
    ``p' = p + δτ + δφ × p``，于是相机系坐标变为

        x_c' = R'ᵀ(X − p') = x_c − Rᵀδτ + hat(x_c + Rᵀp) Rᵀ δφ,

    再由 ``e = (z_obs − z_pred)/σ``、``z_pred = [x_c/z_c, y_c/z_c]`` 得::

        ∂e/∂[δτ, δφ] = [ J_p Rᵀ , −J_p hat(x_c + Rᵀp) Rᵀ ],
        ∂e/∂X        = −J_p Rᵀ,      J_p = [[1/z, 0, −x/z²], [0, 1/z, −y/z²]].

    相对 epipolar 模块的分块 ``[−J_p, J_p hat(x_c)]``（其世界系→相机系约定），
    这里是同一个投影雅可比复合上重参数化 ``δ → −Rᵀδ`` 以及左乘机体系→世界系
    回缩引入的相机中心项。外点由 :meth:`ViBundle.solve` 施加的 Huber IRLS
    权重（:func:`core.solver.huber_weights`）处理。
    """

    dim = 2

    def __init__(self, k: int, lm: int, z: Sequence[float], sigma: float) -> None:
        """记录观测所属关键帧 ``k``、路标 ``lm``、归一化坐标观测与噪声标准差。"""
        self.k = int(k)
        self.lm = int(lm)
        self.z = np.asarray(z, dtype=float).reshape(2)
        self.sigma = float(sigma)

    @staticmethod
    def _project(Xc: np.ndarray) -> tuple[np.ndarray, float]:
        """归一化平面投影，带深度退化保护。

        Args:
            Xc: (3,) 相机系下的点，单位 m。

        Returns:
            ``(z_pred, zz)``：z_pred 为 (2,) 归一化坐标 ``[x/z, y/z]``，
            zz 为所用深度（标量，m）。
        """
        zz = float(Xc[2])
        if abs(zz) < 1e-9:  # 教学场景的几何不会触发此分支
            zz = 1e-9 if zz >= 0.0 else -1e-9
        return Xc[:2] / zz, zz

    def residual(self, T_wb: np.ndarray, X: np.ndarray) -> np.ndarray:
        """白化 (2,) 重投影残差。

        Args:
            T_wb: (4, 4) 关键帧位姿（机体系→世界系）。
            X: (3,) 路标全局坐标 [m]。

        Returns:
            (2,) ``(z_obs − z_pred) / σ``，各分量以 σ 为单位。
        """
        x_c = T_wb[:3, :3].T @ (X - T_wb[:3, 3])
        z_pred, _ = self._project(x_c)
        return (self.z - z_pred) / self.sigma

    def jacobian(self, T_wb: np.ndarray, X: np.ndarray) -> np.ndarray:
        """解析 (2, 9) 雅可比，列 ``[位姿切向量 (6), 路标 (3)]``。

        Args:
            T_wb: (4, 4) 关键帧位姿（机体系→世界系）。
            X: (3,) 路标全局坐标 [m]。

        Returns:
            (2, 9) 雅可比（已除以 ``σ`` 白化；单位为 1/σ 每米 / 每弧度）。
        """
        R = T_wb[:3, :3]
        x_c = R.T @ (X - T_wb[:3, 3])
        z_pred, zz = self._project(x_c)
        J_p = np.array(
            [[1.0 / zz, 0.0, -z_pred[0] / zz], [0.0, 1.0 / zz, -z_pred[1] / zz]]
        )
        x_c0 = x_c + R.T @ T_wb[:3, 3]  # 左回缩把相机中心一并带入
        J_pose = np.hstack([J_p @ R.T, -J_p @ hat(x_c0) @ R.T])
        J_lm = -J_p @ R.T
        return np.hstack([J_pose, J_lm]) / self.sigma


@dataclass
class ViResult:
    """:meth:`ViBundle.solve` 的结果捆绑。"""

    states: list[KeyframeState]  # 优化后的关键帧状态（深拷贝）
    landmarks: np.ndarray        # (L, 3) 优化后的路标全局 XYZ [m]
    cost: float          # 最终的鲁棒（白化）代价
    n_iters: int         # 外层 LM 实际迭代次数
    converged: bool      # 代价下降是否满足 tol 收敛判据


class ViBundle:
    """滑动窗口关键帧 + XYZ 路标上的因子图。

    因子：:class:`ImuFactor`（相邻关键帧之间，第 10 章 (10.9) 的第一个
    求和项）与 :class:`ReprojectionFactor`（第二个求和项）。(10.11) 的边缘化
    先验（marginalization prior）不在本教学实现的固定 4 关键帧窗口范围内。
    """

    def __init__(
        self, states: Sequence[KeyframeState], landmarks: np.ndarray
    ) -> None:
        """深拷贝初始关键帧状态与初始路标，建立空的因子列表。"""
        self.states: list[KeyframeState] = [
            KeyframeState(
                s.T_wb.copy(), np.asarray(s.v, float).copy(),
                np.asarray(s.b_g, float).copy(), np.asarray(s.b_a, float).copy(),
            )
            for s in states
        ]
        self.landmarks = np.asarray(landmarks, dtype=float).copy()
        self._imu_factors: list[ImuFactor] = []
        self._vis_factors: list[ReprojectionFactor] = []

    # ------------------------------------------------------------------ #
    def add_imu_factor(self, i: int, j: int, preint: Preintegration) -> None:
        """在关键帧 ``i`` 与 ``j`` 之间加入一个 IMU 预积分因子。"""
        self._imu_factors.append(ImuFactor(i, j, preint))

    def add_observation(
        self, k: int, lm: int, z: Sequence[float], sigma: float
    ) -> None:
        """加入关键帧 ``k`` 对路标 ``lm`` 的单目归一化坐标观测。"""
        self._vis_factors.append(ReprojectionFactor(k, lm, z, sigma))

    # ------------------------------------------------------------------ #
    @property
    def n_params(self) -> int:
        """堆叠切向量的总长度（位姿 + 速度 + 零偏 + 路标）。"""
        return 15 * len(self.states) + 3 * self.landmarks.shape[0]

    def _residuals(
        self, states: list[KeyframeState], landmarks: np.ndarray,
        use_imu: bool, imu_weight: float,
    ) -> tuple[np.ndarray, int]:
        """堆叠的白化残差（IMU 行在前，视觉行在后）。

        Returns:
            ``(e, n_imu_rows)``：e 为 (m,) 堆叠残差；n_imu_rows 为 IMU 行数，
            视觉行从下标 ``n_imu_rows`` 开始（供 Huber 加权区分）。
        """
        parts: list[np.ndarray] = []
        if use_imu:
            for f in self._imu_factors:
                parts.append(imu_weight * f.residual(states[f.i], states[f.j]))
        n_imu_rows = int(sum(p.size for p in parts))
        for f in self._vis_factors:
            parts.append(f.residual(states[f.k].T_wb, landmarks[f.lm]))
        return np.concatenate(parts), n_imu_rows

    @staticmethod
    def _cost(e: np.ndarray, n_imu_rows: int, huber_delta: float) -> tuple[float, np.ndarray]:
        """鲁棒代价 ``‖e‖²_W``：视觉行乘 Huber IRLS 权重。

        Args:
            e: (m,) 堆叠白化残差。
            n_imu_rows: 前 ``n_imu_rows`` 行为 IMU 行（不加 Huber 权重）。
            huber_delta: 白化视觉残差上的 Huber 阈值。

        Returns:
            (标量代价, (m,) IRLS 权重矢量 w)。
        """
        w = np.ones_like(e)
        if e.size > n_imu_rows:
            w[n_imu_rows:] = huber_weights(e[n_imu_rows:], huber_delta)
        return float(np.sum(w * e * e)), w

    def _linearize(
        self, states: list[KeyframeState], landmarks: np.ndarray,
        use_imu: bool, imu_weight: float,
    ) -> np.ndarray:
        """堆叠残差对切向量的稠密雅可比 (m, n_params)。

        IMU 因子块排在前部（``use_imu`` 为真时）；每个视觉因子贡献 2 行，
        写入其位姿切向量的 6 列与所观测路标的 3 列。
        """
        n_kf = len(states)
        J = np.zeros((sum(f.dim for f in self._imu_factors if use_imu)
                      + 2 * len(self._vis_factors), self.n_params))
        row = 0
        if use_imu:
            for f in self._imu_factors:
                J_f, idx = f.jacobian(states[f.i], states[f.j])
                J[row:row + f.dim, idx] = imu_weight * J_f
                row += f.dim
        cols_vis = np.arange(15 * n_kf, 15 * n_kf + 3 * self.landmarks.shape[0])
        for f in self._vis_factors:
            J_f = f.jacobian(states[f.k].T_wb, landmarks[f.lm])
            cols = np.concatenate(
                [np.arange(15 * f.k, 15 * f.k + 6), cols_vis[3 * f.lm:3 * f.lm + 3]]
            )
            J[row:row + 2, cols] = J_f
            row += 2
        return J

    def _retract(
        self, dx: np.ndarray
    ) -> tuple[list[KeyframeState], np.ndarray]:
        """施加整步切向量更新（位姿走流形回缩，矢量直接相加）。

        Returns:
            (新的关键帧状态列表, 新的路标数组 (L, 3))。
        """
        new_states = [
            _retract_state(s, dx[15 * k:15 * k + 15])
            for k, s in enumerate(self.states)
        ]
        cut = 15 * len(self.states)
        new_lm = self.landmarks + dx[cut:].reshape(self.landmarks.shape)
        return new_states, new_lm

    # ------------------------------------------------------------------ #
    def solve(
        self,
        n_iters: int = 120,
        lm_lambda: float = 1e-6,
        tol: float = 1e-10,
        imu_weight: float = 1.0,
        huber_delta: float = 3.0,
        verbose: bool = False,
    ) -> ViResult:
        """SE(3) 流形上的外层 Levenberg-Marquardt（第 08 章 (8.5)-(8.8)）。

        每次外层迭代在当前状态重新线性化（IMU 数值雅可比、视觉解析雅可比），
        求解阻尼正规方程 ``(Jᵀ W J + λ diag(JᵀWJ)) δx = −Jᵀ W e``（Marquardt
        缩放：堆叠系统的列混合了 [m] 与 [rad] 量纲，故阻尼按 H 的对角元缩放
        而非单一 λI），且只在鲁棒代价下降时接受该步（接受则 λ ← λ/10、拒绝则
        ×10，即 :func:`core.solver.gauss_newton` 的自适应调度）。位姿经回缩
        ``T_wb ← Exp(δξ^) T_wb`` 更新 —— 绝不用矢量加法，那会离开 SO(3)
        （第 02 章）。

        Args:
            n_iters: 外层迭代次数上限（默认 120）。
            lm_lambda: 初始 LM 阻尼 λ（默认 1e-6）。
            tol: 相对代价下降的收敛阈（默认 1e-10）。
            imu_weight: IMU 因子的全局权重；``0`` 复现纯视觉退化运行
                （尺度规范自由，教程 §10.3）。
            huber_delta: *白化*视觉残差上的 Huber 阈值（默认 3.0）。
            verbose: True 时打印每次被接受迭代的代价与 λ。

        Returns:
            :class:`ViResult`：优化后的状态与路标、最终代价、迭代数、收敛标志。
        """
        if imu_weight < 0.0:
            raise ValueError("imu_weight must be non-negative")
        use_imu = imu_weight > 0.0
        lam = float(lm_lambda)
        e, n_imu_rows = self._residuals(self.states, self.landmarks, use_imu, imu_weight)
        cost, w = self._cost(e, n_imu_rows, huber_delta)
        converged = False
        iters = 0
        for iters in range(1, n_iters + 1):
            J = self._linearize(self.states, self.landmarks, use_imu, imu_weight)
            H = J.T @ (w[:, None] * J)
            grad = J.T @ (w * e)
            damp = np.maximum(np.diag(H), 1e-12)
            accepted = False
            for _ in range(60):
                H_damped = H + lam * damp + 1e-12 * np.eye(self.n_params)
                try:
                    dx = -np.linalg.solve(H_damped, grad)
                except np.linalg.LinAlgError:
                    dx = -np.linalg.lstsq(H_damped, grad, rcond=None)[0]
                cand_states, cand_lm = self._retract(dx)
                e_c, n_imu_c = self._residuals(
                    cand_states, cand_lm, use_imu, imu_weight
                )
                cost_c, w_c = self._cost(e_c, n_imu_c, huber_delta)
                if np.isfinite(cost_c) and cost_c < cost:
                    self.states, self.landmarks = cand_states, cand_lm
                    e, w, prev_cost, cost = e_c, w_c, cost, cost_c
                    lam = max(lam / 10.0, 1e-14)
                    accepted = True
                    break
                lam = min(lam * 10.0, 1e14)
                if lam >= 1e14:
                    break
            if not accepted:
                break
            if verbose:
                print(f"  [LM] iter {iters:3d}  cost {cost:.6e}  lam {lam:.1e}")
            if prev_cost - cost < tol * max(1.0, cost):
                converged = True
                break
        return ViResult(
            states=[
                KeyframeState(s.T_wb.copy(), s.v.copy(), s.b_g.copy(), s.b_a.copy())
                for s in self.states
            ],
            landmarks=self.landmarks.copy(),
            cost=cost,
            n_iters=iters,
            converged=converged,
        )


def align_se3(
    points_est: np.ndarray, points_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch/Umeyama SE(3) 对齐：``points_ref ≈ R @ points_est + t``。

    单目 VIO 保有 4 自由度规范（全局 yaw + 全局平移），估计器无法钉住它们，
    因此对轨迹评分前必须先做此对齐（教程 §10.3 第 3 步）。依据：正交
    Procrustes 问题 ``max_R tr(Rᵀ H)``（``H = Σ(p−p̄)(q−q̄)ᵀ``）由 SVD 求解
    ``H = U S Vᵀ → R = V diag(1, 1, sign det) Uᵀ``（Kabsch 1976;
    Umeyama 1991）。

    Args:
        points_est: (N, 3) 估计点集（如估计轨迹位置），单位 m。
        points_ref: (N, 3) 参考点集（如真值轨迹位置），单位 m。

    Returns:
        ``(R, t, aligned_est)``：R 为 (3, 3) 旋转（无量纲）、t 为 (3,) 平移
        [m]、aligned_est 为 (N, 3) 对齐后的估计点集 [m]。
    """
    pe = np.asarray(points_est, dtype=float).reshape(-1, 3)
    pr = np.asarray(points_ref, dtype=float).reshape(-1, 3)
    ce, cr = pe.mean(axis=0), pr.mean(axis=0)
    H = (pe - ce).T @ (pr - cr)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cr - R @ ce
    return R, t, (R @ pe.T).T + t


@dataclass
class VISimulation:
    """:func:`simulate_vi_scene` 合成的场景数据捆绑。"""

    params: ImuParams
    gt_states: list[KeyframeState]          # 真值关键帧状态（世界系离散积分生成）
    gt_landmarks: np.ndarray                     # (L, 3) 真值路标全局 XYZ [m]
    imu_samples: list[list[tuple[float, np.ndarray, np.ndarray]]]  # 每个关键帧区间一段 (t, gyro, accel)
    observations: np.ndarray                     # (K, L, 2) 归一化坐标观测（带噪）
    sigma_z: float                               # 观测噪声标准差（归一化坐标，无量纲）
    init_states: list[KeyframeState]             # 故意错误尺度的初始化
    init_landmarks: np.ndarray                   # 由初始位姿三角化的路标 [m]
    bias_true: np.ndarray                        # (6,) 真值零偏 [b^g; b^a]，单位 [rad/s; m/s²]


def simulate_vi_scene(
    seed: int = 0,
    n_keyframes: int = 4,
    interval: float = 0.8,
    imu_rate: float = 100.0,
    n_landmarks: int = 40,
    sigma_z: float = 4.0e-3,
    init_scale: float = 0.5,
    init_rot_perturb: float = 0.02,
) -> VISimulation:
    """合成一段带单目特征与带偏 IMU 的走廊飞行。

    机体系沿 ``p(t) = [1.2 t, 0.8 sin(0.8 t), 0.3 sin(0.9 t)]`` [m] 平移
    （速度 ≈ 1.4 m/s），同时偏航 ``±0.35 rad``；相机（= 机体系，外参单位阵）
    沿机体系 z 轴观察，轴约定 ``[右; 下; 前]``，路标块位于前方 6.5–11 m。
    每个区间的 IMU 采样携带常值真偏置加白噪声（噪声密度 ``σ`` 见
    :class:`ImuParams`，单样本标准差 ``σ/√Δt``）。

    真值关键帧状态由**同一批真实读数的世界系离散积分**定义（paper
    Eq.(31)-(32) 格式），因此与预积分测量模型严格自洽 —— 剩下的唯一误差是
    传感器噪声与零偏，而它们正是估计器所建模的对象。

    关键设计：返回的 ``init_states`` 是**故意错误尺度**的 —— 平移与速度缩小
    ``init_scale``（0.5）倍、零偏清零（旋转加小的定种子扰动）；路标由这些
    错误位姿 DLT 三角化，因此初始猜测是一个自洽的*纯视觉*解、只是尺度错误
    —— 把 ``init_scale → 1`` 恢复出来只能依靠 IMU 因子（教程 §10.3
    尺度可观性）。

    Args:
        seed: 随机种子（默认 0；噪声与场景随机性全部由它确定）。
        n_keyframes: 关键帧数（默认 4）。
        interval: 相邻关键帧间隔 [s]（默认 0.8）。
        imu_rate: IMU 采样率 [Hz]（默认 100）。
        n_landmarks: 路标数（默认 40）。
        sigma_z: 归一化坐标观测噪声标准差（默认 4.0e-3）。
        init_scale: 初始化的尺度缩放因子（默认 0.5，< 1 表示初始轨迹偏短）。
        init_rot_perturb: 初始化旋转扰动的标准差 [rad]（默认 0.02）。

    Returns:
        :class:`VISimulation`：真值状态/路标、每区间 IMU 采样、观测、错误
        尺度的初始化与三角化路标、真值零偏。
    """
    params = ImuParams(rate=imu_rate)
    rng = np.random.default_rng(seed)
    dt = 1.0 / imu_rate
    g = params.g
    bg_true = np.array([0.01, -0.008, 0.005])
    ba_true = np.array([0.08, -0.05, 0.06])

    def R_of(t: float) -> np.ndarray:
        # R_wb：列向量为世界系中的机体系坐标轴 ——
        # x（右）、y（下）、z（前，即相机光轴）。
        yaw, pitch = 0.35 * np.sin(0.7 * t), 0.05 * np.sin(0.5 * t)
        f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch),
                      np.sin(pitch)])
        d0 = np.array([0.0, 0.0, -1.0])
        r = np.cross(d0, f)
        r /= np.linalg.norm(r)
        d = np.cross(f, r)
        return np.vstack([r, d, f]).T

    def p_of(t: float) -> np.ndarray:
        return np.array([1.2 * t, 0.8 * np.sin(0.8 * t), 0.3 * np.sin(0.9 * t)])

    def v_of(t: float) -> np.ndarray:
        return np.array([1.2, 0.64 * np.cos(0.8 * t), 0.27 * np.cos(0.9 * t)])

    def a_of(t: float) -> np.ndarray:
        return np.array([0.0, -0.512 * np.sin(0.8 * t), -0.243 * np.sin(0.9 * t)])

    def w_of(t: float) -> np.ndarray:
        """机体系角速度 ω = vee(Rᵀ Ṙ)；Ṙ 用 O(h²) 中心差分
        （误差 ≈ 1e-8 rad/s，相对 0.007 rad/s 量级的采样噪声可忽略）。"""
        h = 1e-4
        Rd = (R_of(t + h) - R_of(t - h)) / (2.0 * h)
        return vee(R_of(t).T @ Rd)

    def a_body_of(t: float) -> np.ndarray:
        """真实比力（paper Eq.(28) 移项）：Rᵀ(p̈ − g)。"""
        return R_of(t).T @ (a_of(t) - g)

    # 真值关键帧：由世界系离散积分生成。
    gt_states: list[KeyframeState] = []
    T0 = np.eye(4)
    T0[:3, :3] = R_of(0.0)
    T0[:3, 3] = p_of(0.0)
    gt_states.append(KeyframeState(T0, v_of(0.0), bg_true.copy(), ba_true.copy()))
    imu_samples: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
    n_per = int(round(interval * imu_rate))
    for i in range(n_keyframes - 1):
        s = gt_states[-1]
        R, v, p = s.T_wb[:3, :3].copy(), s.v.copy(), s.T_wb[:3, 3].copy()
        samples: list[tuple[float, np.ndarray, np.ndarray]] = []
        for k in range(n_per):
            t = i * interval + k * dt
            w_true = w_of(t)
            a_b = a_body_of(t)
            w_m = w_true + bg_true + rng.normal(scale=params.sigma_g / np.sqrt(dt), size=3)
            a_m = a_b + ba_true + rng.normal(scale=params.sigma_a / np.sqrt(dt), size=3)
            samples.append((float(t), w_m, a_m))
            acc_w = g + R @ a_b          # 真实的世界系加速度
            p = p + v * dt + 0.5 * acc_w * dt**2
            v = v + acc_w * dt
            R = R @ so3_exp(w_true * dt)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        gt_states.append(KeyframeState(T, v, bg_true.copy(), ba_true.copy()))
        imu_samples.append(samples)

    # 轨迹前方的路标块 + 带噪归一化观测。
    gt_landmarks = np.column_stack(
        [rng.uniform(7.0, 11.0, n_landmarks),
         rng.uniform(-1.3, 1.3, n_landmarks),
         rng.uniform(-1.2, 1.2, n_landmarks)]
    )
    observations = np.empty((n_keyframes, n_landmarks, 2))
    for k, s in enumerate(gt_states):
        R = s.T_wb[:3, :3]
        x_c = (gt_landmarks - s.T_wb[:3, 3]) @ R  # (X−p) @ R 的行 = Rᵀ(X−p)
        if np.any(x_c[:, 2] < 0.3) or np.any(np.abs(x_c[:, :2] / x_c[:, 2:3]) > 1.0):
            raise RuntimeError("scene geometry violates visibility assumptions")
        observations[k] = (
            x_c[:, :2] / x_c[:, 2:3] + rng.normal(scale=sigma_z, size=(n_landmarks, 2))
        )

    # 故意错误尺度的初始化 + 由其 DLT 三角化路标。
    init_states: list[KeyframeState] = []
    for s in gt_states:
        T = np.eye(4)
        T[:3, :3] = s.T_wb[:3, :3] @ so3_exp(
            rng.normal(scale=init_rot_perturb, size=3)
        )
        T[:3, 3] = s.T_wb[:3, 3] * init_scale
        init_states.append(
            KeyframeState(T, s.v * init_scale, np.zeros(3), np.zeros(3))
        )
    T_cw0 = np.linalg.inv(init_states[0].T_wb)
    T_cw1 = np.linalg.inv(init_states[1].T_wb)
    init_landmarks = triangulate(
        T_cw0, T_cw1, observations[0], observations[1], K=None
    )
    if not np.all(np.isfinite(init_landmarks)):
        raise RuntimeError("landmark triangulation from the init poses failed")

    return VISimulation(
        params=params,
        gt_states=gt_states,
        gt_landmarks=gt_landmarks,
        imu_samples=imu_samples,
        observations=observations,
        sigma_z=sigma_z,
        init_states=init_states,
        init_landmarks=init_landmarks,
        bias_true=np.concatenate([bg_true, ba_true]),
    )


def build_vins_bundle(sim: VISimulation) -> ViBundle:
    """由 :class:`VISimulation` 组装 §10.3 的因子图（式 (10.9)）。

    IMU 因子建立在相邻关键帧之间，初始偏置估计取零（因子的一阶 δb 修正会
    吸收后续的偏置更新，paper Eq.(44)）；视觉因子连接每个关键帧与每个路标。

    Args:
        sim: :func:`simulate_vi_scene` 产出的场景数据。

    Returns:
        组装完毕、尚未求解的 :class:`ViBundle`。
    """
    bundle = ViBundle(sim.init_states, sim.init_landmarks)
    for i, samples in enumerate(sim.imu_samples):
        bundle.add_imu_factor(i, i + 1, Preintegration(samples, np.zeros(6), sim.params))
    for k in range(len(sim.gt_states)):
        for lm in range(sim.gt_landmarks.shape[0]):
            bundle.add_observation(k, lm, sim.observations[k, lm], sim.sigma_z)
    return bundle

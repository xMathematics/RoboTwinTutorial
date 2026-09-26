"""SO(3) 流形上的 IMU 预积分（preintegration，Forster et al., T-RO 2017）。

函数流水线
----------
教程第 10 章 §10.2（式 (10.1)-(10.8)）的教学实现；论文：C. Forster,
L. Carlone, F. Dellaert, D. Scaramuzza, "On-Manifold Preintegration for
Real-Time Visual-Inertial Odometry", IEEE T-RO 2017
（``papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf``）。本文件
docstring 中 ``Eq.(x)`` 均指该论文的式号，``(10.x)`` 指教程第 10 章
（``tutorials/slam/10_建图与系统实战.md``）的式号。

流水线：:class:`ImuParams`（传感器噪声参数与采样率）→ :class:`Preintegration`
（输入一段 ``(t, gyro, accel)`` 原始采样与偏置估计 ``b̄``，逐样本递推无重力
增量 ΔR̃/Δṽ/Δp̃、协方差 Σ 与偏置雅可比；:meth:`Preintegration.correct` 做
免重积分的一阶偏置修正，:meth:`Preintegration.predict` 做零噪声状态预测）
→ 由 ``vins`` 模块的 :class:`vins.ImuFactor` 包装成紧耦合因子图中的 IMU
因子。依赖 ``core.lie`` 的左雅可比（SO(3) 右雅可比由恒等式
``J_r(φ) = J_l(−φ) = J_l(φ)^T`` 得到）。由 ``tests/test_preint.py`` 直接驱动。

测量模型（论文 Eq.(27)，教程 (10.1)）
------------------------------------
陀螺仪/加速度计读数携带慢变偏置 ``b`` 与加性白噪声 ``η``::

    ω̂_k = ω_k + b^g_k + η^g_k,      â_k = a_k + b^a_k + η^a_k.

在两个相机关键帧 :math:`i \\to j` 之间，三个预积分测量（论文 Eq.(37)，
教程第 4 步）是*无重力*积分::

    ΔR̃_ij = Π_k Exp((ω̂_k − b^g) Δt_k),
    Δṽ_ij = Σ_k ΔR̃_ik (â_k − b^a) Δt_k,
    Δp̃_ij = Σ_k [ Δṽ_ik Δt_k + ½ ΔR̃_ik (â_k − b^a) Δt_k² ].

为什么这里没有重力（务必理清）：这些相对量在*定义*（论文 Eq.(33)，
教程 (10.4)）中就是
``Δp_ij ≜ R_i^T (p_j − p_i − v_i Δt_ij − ½ g Δt_ij²)`` —— 重力项在定义里
就被减掉了，使右端只依赖 IMU 读数本身。重力只在**测量模型**（论文
Eq.(38)，教程 (10.5)）中、即预积分测量与状态比对时才重新出现::

    ΔR̃_ij = R_i^T R_j Exp(δφ_ij),
    Δṽ_ij = R_i^T (v_j − v_i − g Δt_ij) + δv_ij,
    Δp̃_ij = R_i^T (p_j − p_i − v_i Δt_ij − ½ g Δt_ij²) + δp_ij,

其中 ``g`` 为指向**下方**的重力加速度矢量（默认 ``[0, 0, −9.81]``），因此
位置模型里是 ``−½ g Δt²``（符号约定：世界系 z 轴向上、g 指向下；静止时
加速度计读 ``R^T(−g) = [0, 0, +9.81]``，见 :class:`ImuParams.gravity`）。
零噪声下反解该测量模型即得状态预测 :meth:`Preintegration.predict`。

右乘更新（偏置独立、可增量更新）
--------------------------------
``ΔR̃ ← ΔR̃ Exp((ω̂ − b^g) Δt)`` 把每个新采样**从右侧**乘入。
依据 (tutorial §10.2 第 3 步, paper Eq.(32)→(33))：``R_i^T R_j`` 可望远镜式
（telescoping）分解为逐样本指数的连乘；由于每个新因子都从右侧乘入，先前
累积的项永不改变，因此三个积分可在 IMU 数据到达时增量更新，且与 ``i`` 处
的状态无关（世界系积分 (10.3) 的"初值锚定"问题就此解决）。偏置在 ``[i, j]``
上视为常值（paper Eq.(34)）；之后的偏置更新经 :meth:`Preintegration.correct`
一阶施加。

噪声约定
--------
遵循论文 Eq.(35)-(36)：真值转移等于预积分测量右乘一个小噪声，
``ΔR_true = ΔR̃ Exp(−δφ^)``（``δφ`` 表达在系 *j* 中），
``Δv_true = Δṽ − δv``、``Δp_true = Δp̃ − δp``（``δv, δp`` 表达在系 *i* 中）。
``η^Δ = [δφ; δv; δp]`` 的 9×9 协方差 ``Σ_ij`` 逐样本按线性递推传播
（paper Eq.(46)-(47) / 紧凑形式 (62)-(63)，教程 (10.7)；与 EKF 预测同构，
第 03 章 (3.9)）::

    Σ ← A Σ A^T + B Σ_η B^T,

    A = [[ Exp(φ̃)^T        0           0    ],
         [ −ΔR̃ (a)^ Δt      I           0    ],
         [ −½ ΔR̃ (a)^ Δt²   I Δt        I    ]],
    B = [[ J_r(φ̃)   0          ],
         [ 0        ΔR̃         ],
         [ 0        ½ ΔR̃ Δt    ]],
    Σ_η = diag(σ_g² Δt I₃, σ_a² Δt I₃),

其中 ``φ̃ = (ω̂ − b^g) Δt``、``a = â − b^a``，``J_r`` 为 SO(3) 右雅可比
（paper Eq.(8)；``J_r(φ) = J_l(−φ) = J_l(φ)^T``，故由
:func:`core.lie.so3_left_jacobian` 得到）。这些矩阵来自噪声的一阶（线性化）
递推（paper Eq.(59)-(61)）：``δφ⁺ = Exp(φ̃)^T δφ + J_r η^g Δt``、
``δv⁺ = δv − ΔR̃ (a)^ δφ Δt + ΔR̃ η^a Δt``、``δp⁺ = δp + δv Δt − ½ΔR̃(a)^δφΔt² +
½ΔR̃ η^a Δt²``。

偏置雅可比（论文附录 IX-B，Eq.(44)/(67)-(69)，教程 (10.8)）
----------------------------------------------------------
偏置扰动 ``b ← b + δb`` 时，预积分测量一阶变化为::

    ΔR̃(b + δb) ≈ ΔR̃ Exp((J_R δb^g)^),
    Δṽ(b + δb) ≈ Δṽ + J_vg δb^g + J_va δb^a,
    Δp̃(b + δb) ≈ Δp̃ + J_pg δb^g + J_pa δb^a,

其中（paper 附录 IX-B，符号按论文原样）::

    ∂ΔR̃_ij/∂b^g = −Σ_k ΔR̃_{k+1,j}^T J_r^k Δt,
    ∂Δṽ_ij/∂b^g  = −Σ_k ΔR̃_ik (â_k − b^a)^ (∂ΔR̃_ik/∂b^g) Δt,   ∂Δṽ/∂b^a = −Σ_k ΔR̃_ik Δt,
    ∂Δp̃/∂b^g    = Σ_k [∂Δṽ_ik/∂b^g Δt − ½ΔR̃_ik (â_k−b^a)^ (∂ΔR̃_ik/∂b^g) Δt²],
    ∂Δp̃/∂b^a    = Σ_k [∂Δṽ_ik/∂b^a Δt − ½ΔR̃_ik Δt²],

全部可增量计算（存入 :attr:`Preintegration.j_bias`，一个作用于
``δb = [δb^g; δb^a]`` 的 9×6 矩阵）。:meth:`Preintegration.correct` 按
Eq.(44) 施加更新而无需重新积分。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.lie import hat, so3_exp, so3_left_jacobian

__all__ = ["ImuParams", "Preintegration", "so3_right_jacobian"]


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """SO(3) 的右雅可比（right Jacobian），paper Eq.(8)。

    ``J_r(φ) = I − (1−cosθ)/θ² φ^ + (θ−sinθ)/θ³ (φ^)²``。依据：``J_l`` 的
    闭式（core.lie Eq. 2.20-2.22）满足恒等式 ``J_r(φ) = J_l(−φ) = J_l(φ)^T``
    （线性项变号、 ``(φ^)²`` 项不变号），故直接取 ``−φ`` 的左雅可比。

    Args:
        phi: (3,) 旋转向量 ``φ``（轴角/李代数形式），单位 rad。

    Returns:
        (3, 3) 右雅可比矩阵 ``J_r(φ)``，无量纲。
    """
    return so3_left_jacobian(-np.asarray(phi, dtype=float).reshape(3))


@dataclass(frozen=True)
class ImuParams:
    """IMU 噪声参数与采样率。

    Attributes:
        sigma_g:  陀螺仪连续时间白噪声密度 [rad/(s·√Hz)]（即 rad/√s）：
                  角速度测量噪声的谱密度；换算到单样本的噪声标准差为
                  ``σ_g/√Δt``（见 ``rate`` 条目）。
        sigma_a:  加速度计连续时间白噪声密度 [m/(s²·√Hz)]（即 (m/s²)/√s）：
                  比力测量噪声的谱密度；单样本标准差 ``σ_a/√Δt``。
        sigma_bg: 陀螺仪零偏连续时间随机游走密度 [rad/(s²·√Hz)]
                  （即 (rad/s)/√s）：零偏 ``b^g`` 的方差按 ``σ_bg² Δt`` 增长。
        sigma_ba: 加速度计零偏连续时间随机游走密度 [m/(s³·√Hz)]
                  （即 (m/s²)/√s）：零偏 ``b^a`` 的方差按 ``σ_ba² Δt`` 增长。
        gravity:  世界系中的重力加速度矢量，指向**下方**（paper Eq.(31) 约定：
                  静止时加速度计读 ``R^T(−g) = [0, 0, +9.81]``）；测量模型
                  （paper Eq.(38)，教程 (10.5)）中的 ``−½ g Δt²`` 用的正是
                  这个符号。
        rate:     IMU 采样率 [Hz]；用于补足最后一个样本的区间时长，并把连续
                  噪声密度换算成逐样本/逐区间协方差（``Var[η^d] = σ² Δt``，
                  paper §V；等价地，单样本测量噪声的标准差为 ``σ/√Δt``）。

    默认值取论文 §VIII-A 的仿真参数（σ_gd = 7e-4,
    σ_ad = 0.019, σ_bgd = 4e-4, σ_bad = 0.012）。
    """

    sigma_g: float = 7.0e-4
    sigma_a: float = 1.9e-2
    sigma_bg: float = 4.0e-4
    sigma_ba: float = 1.2e-2
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    rate: float = 200.0

    @property
    def g(self) -> np.ndarray:
        """重力矢量，(3,) 数组（世界系，指向下方）。"""
        return np.asarray(self.gravity, dtype=float).reshape(3)

    @property
    def dt(self) -> float:
        """标称采样间隔 ``1 / rate`` [s]。"""
        return 1.0 / float(self.rate)


class Preintegration:
    """两个关键帧 i → j 之间的预积分 IMU 测量（preintegrated measurements）。

    Args:
        imu_samples: ``(t, gyro, accel)`` 序列；``t`` 为 float 时间戳 [s]，
            ``gyro, accel`` 为 (3,) 的角速度 [rad/s] 与比力 [m/s²] 测量
            （Eq.(27)：真值 + 偏置 + 噪声）。第 ``k`` 个样本覆盖区间
            ``[t_k, t_{k+1})``；最后一个样本覆盖 ``1/rate``（均匀采样假设），
            因此按 ``rate`` 对 ``[t_i, t_j)`` 采样时总时长恰为 ``t_j − t_i``。
        bias0: (6,) 预积分期间使用的偏置估计 ``b̄ = [b̄^g; b̄^a]``，单位
            [rad/s; m/s²]（paper Eq.(34)：在 ``[i, j]`` 上视为常值）。
        params: 传感器模型（:class:`ImuParams`），缺省 ``ImuParams()``。
        propagate_uncertainty: 为 True（默认）时同时传播 9×9 协方差
            :attr:`cov` 与 9×6 偏置雅可比 :attr:`j_bias`；廉价的
            Monte-Carlo / 纯预测运行可置 False。

    Attributes（命名与论文符号对应）:
        delta_R: (3, 3) ``ΔR̃_ij``（Eq.(37)，右乘增量），无量纲（旋转矩阵）。
        delta_v: (3,) ``Δṽ_ij``（无重力速度积分），单位 m/s。
        delta_p: (3,) ``Δp̃_ij``（无重力位置积分），单位 m。
        cov:     (9, 9) ``η^Δ = [δφ; δv; δp]`` 的协方差 ``Σ_ij``
            （Eq.(46)-(47)；δφ 单位 rad、δv 单位 m/s、δp 单位 m）。
        j_bias:  (9, 6) 偏置雅可比 ``[∂ΔR̃/∂b^g, 0; ∂Δṽ/∂b, ∂Δp̃/∂b]``，
            分块顺序 ``[[R,0],[v_g,v_a],[p_g,p_a]]``（附录 IX-B）。
        bias0, params, duration, n_samples: 与传入一致 / 由其导出
            （duration 为总时长 [s]，n_samples 为样本数）。

    Raises:
        ValueError: 采样为空或时间区间非正时抛出。
    """

    def __init__(
        self,
        imu_samples: Sequence[tuple[float, Sequence[float], Sequence[float]]],
        bias0: Sequence[float],
        params: ImuParams | None = None,
        *,
        propagate_uncertainty: bool = True,
    ) -> None:
        self.params = ImuParams() if params is None else params
        self.bias0 = np.asarray(bias0, dtype=float).reshape(6).copy()
        samples = [
            (
                float(t),
                np.asarray(w, dtype=float).reshape(3),
                np.asarray(a, dtype=float).reshape(3),
            )
            for (t, w, a) in imu_samples
        ]
        if not samples:
            raise ValueError("imu_samples must contain at least one sample")
        dts = [samples[k + 1][0] - samples[k][0] for k in range(len(samples) - 1)]
        dts.append(self.params.dt)  # 最后一个区间时长按标称采样率补足（均匀采样假设）
        if any(d <= 0.0 for d in dts):
            raise ValueError("IMU sample timestamps must be strictly increasing")
        self.n_samples = len(samples)
        self.duration = float(sum(dts))

        bg, ba = self.bias0[:3], self.bias0[3:]
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.cov = np.zeros((9, 9))
        self.j_bias = np.zeros((9, 6))

        for (t_k, w_k, a_k), dt in zip(samples, dts):  # noqa: B007 (t_k 未使用)
            w = w_k - bg                      # 去偏置读数（Eq.(27) 的逆向使用）
            a = a_k - ba
            phi = w * dt                      # φ̃_k
            Jr = so3_right_jacobian(phi)
            E = so3_exp(phi)                  # Exp(φ̃_k)
            R_a = self.delta_R @ a            # ΔR̃_ik a_k（更新前的旋转）

            if propagate_uncertainty:
                self._propagate_covariance(E, Jr, self.delta_R, a, dt)
                self._propagate_bias_jacobians(E, Jr, self.delta_R, a, dt)

            # 均值更新，paper Eq.(37) / 教程 (10.4)：无重力积分；Δp̃ 用更新前的
            # Δṽ_ik（旧值），ΔR̃ 的更新放在最后。
            self.delta_p = self.delta_p + self.delta_v * dt + 0.5 * R_a * dt**2
            self.delta_v = self.delta_v + R_a * dt
            self.delta_R = self.delta_R @ E

    # ------------------------------------------------------------------ #
    # 增量传播辅助（论文附录 IX-A / IX-B）                                 #
    # ------------------------------------------------------------------ #
    def _propagate_covariance(
        self, E: np.ndarray, Jr: np.ndarray, R: np.ndarray, a: np.ndarray, dt: float
    ) -> None:
        """递推一步 ``Σ ← A Σ A^T + B Σ_η B^T``（paper Eq.(46)-(47)）。"""
        A = np.zeros((9, 9))
        A[0:3, 0:3] = E.T
        A[3:6, 0:3] = -R @ hat(a) * dt
        A[3:6, 3:6] = np.eye(3)
        A[6:9, 0:3] = -0.5 * R @ hat(a) * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        A[6:9, 6:9] = np.eye(3)
        B = np.zeros((9, 6))
        # B 的各块即噪声一阶递推（paper Eq.(59)-(61)）的系数：角度行 J_r、
        # 速度行 ΔR̃、位置行 ½ΔR̃·Δt（对应 δp⁺ 中的噪声项 ½ΔR̃ η^a Δt²，
        # 即 ½ΔR̃ · (η^{ad} Δt) · Δt，配合 Σ_η 的 σ² Δt 归一）。
        B[0:3, 0:3] = Jr            # J_r η^{gd} Δt   = J_r · (η^{gd} Δt)
        B[3:6, 3:6] = R             # ΔR̃ η^{ad} Δt   = ΔR̃ · (η^{ad} Δt)
        B[6:9, 3:6] = 0.5 * R * dt  # ½ ΔR̃ η^{ad} Δt² = ½ΔR̃ · (η^{ad} Δt) · Δt
        sigma_eta = np.diag(
            np.concatenate(
                [np.full(3, self.params.sigma_g**2 * dt),
                 np.full(3, self.params.sigma_a**2 * dt)]
            )
        )
        cov = A @ self.cov @ A.T + B @ sigma_eta @ B.T
        self.cov = 0.5 * (cov + cov.T)  # 强制精确对称（数值舍入防护）

    def _propagate_bias_jacobians(
        self, E: np.ndarray, Jr: np.ndarray, R: np.ndarray, a: np.ndarray, dt: float
    ) -> None:
        """增量式附录 IX-B 递推的一步（Eq.(67)-(69)）。"""
        R_a_hat_JR = R @ hat(a) @ self.j_bias[0:3, 0:3]  # ΔR̃ (a)^ ∂ΔR̃/∂b^g
        j_R = E.T @ self.j_bias[0:3, 0:3] - Jr * dt      # ∂ΔR̃/∂b^g（切空间形式）
        j_vg = self.j_bias[3:6, 0:3] - R_a_hat_JR * dt
        j_va = self.j_bias[3:6, 3:6] - R * dt
        j_pg = self.j_bias[6:9, 0:3] + self.j_bias[3:6, 0:3] * dt - 0.5 * R_a_hat_JR * dt**2
        j_pa = self.j_bias[6:9, 3:6] + self.j_bias[3:6, 3:6] * dt - 0.5 * R * dt**2
        self.j_bias = np.block([[j_R, np.zeros((3, 3))], [j_vg, j_va], [j_pg, j_pa]])

    # ------------------------------------------------------------------ #
    # 公开的模型辅助方法                                                  #
    # ------------------------------------------------------------------ #
    def correct(
        self, bias_delta: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """一阶偏置修正，paper Eq.(44) / 教程 (10.8)。

        给定偏置更新 ``b̄ ← b̄ + δb``，不重新积分即返回修正后的预积分测量
        ``(ΔR̃_corr, Δṽ_corr, Δp̃_corr)``::

            ΔR̃(b̄ + δb) ≈ ΔR̃ Exp((∂ΔR̃/∂b^g δb^g)^),
            Δṽ(b̄ + δb) ≈ Δṽ + ∂Δṽ/∂b δb,      Δp̃(b̄ + δb) ≈ Δp̃ + ∂Δp̃/∂b δb,

        即论文 §VI-C 的 a-posteriori 更新（对小 ``δb`` 有效；雅可比已在积分
        期间预存于 :attr:`j_bias`，因此修正本身是 O(1) 计算量，修正残差为
        O(δb²)）。

        Args:
            bias_delta: (6,) 偏置增量 ``δb = [δb^g; δb^a]``，单位
                [rad/s; m/s²]。

        Returns:
            ``(ΔR̃_corr, Δṽ_corr, Δp̃_corr)``：形状分别为 (3, 3)、(3,)、(3,)，
            单位/约定同 :attr:`delta_R` / :attr:`delta_v` / :attr:`delta_p`。
        """
        db = np.asarray(bias_delta, dtype=float).reshape(6)
        d_r = self.delta_R @ so3_exp(self.j_bias[0:3, 0:3] @ db[:3])
        d_v = self.delta_v + self.j_bias[3:6] @ db
        d_p = self.delta_p + self.j_bias[6:9] @ db
        return d_r, d_v, d_p

    def predict(
        self, R_i: np.ndarray, v_i: np.ndarray, p_i: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """零噪声状态预测 ``i → j``（paper Eq.(38) 在零噪声处）。

        反解测量模型（教程 (10.5)）—— 注意重力项出现在**这里**，而不在预积分
        递推里::

            R_j = R_i ΔR̃,
            v_j = v_i + g Δt_ij + R_i Δṽ,
            p_j = p_i + v_i Δt_ij + ½ g Δt_ij² + R_i Δp̃.

        若给定非零的噪声实现或更新后的偏置，把 ``correct(...)`` 的输出代入
        同样的公式即可。

        Args:
            R_i: (3, 3) 关键帧 i 的姿态（机体系→世界系），无量纲。
            v_i: (3,) 关键帧 i 的世界系速度 [m/s]。
            p_i: (3,) 关键帧 i 的世界系位置 [m]。

        Returns:
            ``(R_j, v_j, p_j)``：预测的关键帧 j 状态，形状/单位与输入一致。
        """
        g, T = self.params.g, self.duration
        R_j = R_i @ self.delta_R
        v_j = v_i + g * T + R_i @ self.delta_v
        p_j = p_i + v_i * T + 0.5 * g * T**2 + R_i @ self.delta_p
        return R_j, v_j, p_j

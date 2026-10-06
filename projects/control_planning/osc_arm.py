"""平面 2R 机械臂：正运动学 / 解析雅可比 / DLS 逆运动学 / 任务空间阻抗控制。

函数流水线
----------
Khatib, *A Unified Approach for Motion and Force Control of Robotic
Manipulators: The Operational Space Formulation*, IEEE JRA 1987 与 Hogan,
*Impedance Control: An Approach to Manipulation, Part I–III*, ASME J-DSC
1985 的教学实现（两文未收录本地 PDF，见论文库 README 延伸阅读），推导见
教程第 06 章：本模块覆盖 (6.3) 的正运动学与速度映射、(6.2) 的关节动力学、
(6.12)/(6.13) 的阻抗律与闭环误差系统；DLS 逆运动学对应第 02 章
(2.7)–(2.10) 的阻尼最小二乘伪逆（与 (6.7) 动力学一致伪逆的关系在
:func:`dls_ik` docstring 说明）。

调用链：:class:`PlanarArm`（FK / 解析雅可比 / M、C、g 动力学）
→ :func:`dls_ik`（阻尼最小二乘 IK，奇异位形不发散）
→ :func:`impedance_torque`（τ = Jᵀ(K e + D ė) + g(q)，(6.12) 的力矩级实现）
→ :func:`simulate_impedance`（关节动力学闭环仿真，验证 (6.13) 的
"从偏移位形拉回目标"）。数值雅可比对照由 tests/test_osc_arm.py 以
中心差分交叉验证（教程 06 章对 J 一致性的要求）。

依赖方向：本模块不依赖库内其他模块（仅 numpy）。

约定
----
- 关节角 ``q = (θ1, θ2)`` 单位 rad；任务坐标 ``x = (x, y)`` 单位 m
  （末端位置，2 维任务参数化——教程 06 章记号说明的"局部任务参数化"）。
- 连杆几何：l1/l2 连杆长、lc1/l2 质心距关节距离（m）；m1/m2 连杆质量
  （kg）；I1/I2 绕质心转动惯量（kg·m²）。重力沿 -y。
- 动力学采用标准 2R 闭式（Craig《机器人学导论》ch6 的平面 2R 特例，
  教程 (6.2) 的 Lagrange 建模出处）：

  $$M(q)\\ddot q + C(q,\\dot q)\\dot q + g(q) = \\tau$$

示例
----
>>> import numpy as np
>>> from osc_arm import PlanarArm, dls_ik
>>> arm = PlanarArm()
>>> q, err, _ = dls_ik(arm, np.array([0.3, 0.6]), np.array([0.8, 0.5]))
>>> err < 1e-6  # doctest: +SKIP
True
"""
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PlanarArm",
    "dls_ik",
    "impedance_torque",
    "numeric_jacobian",
    "simulate_impedance",
]


@dataclass
class PlanarArm:
    """平面 2R 机械臂（几何 + 质量参数 + 运动学/动力学闭式）。

    Attributes:
        l1, l2: 连杆长度，单位 m。
        m1, m2: 连杆质量，单位 kg。
        lc1, lc2: 质心到各自近端关节的距离，单位 m。
        I1, I2: 连杆绕质心的转动惯量，单位 kg·m²。
    """

    l1: float = 1.0
    l2: float = 0.8
    m1: float = 1.2
    m2: float = 0.9
    lc1: float = 0.5
    lc2: float = 0.4
    I1: float = 0.1
    I2: float = 0.05

    def fk(self, q: np.ndarray) -> np.ndarray:
        """正运动学（教程 (6.3) 的 ``x = f(q)``）：末端位置 (2,)，单位 m。

        Args:
            q: (2,) 关节角 ``[θ1, θ2]``，单位 rad。

        Returns:
            (2,) 末端位置 ``[l1c1 + l2c12, l1s1 + l2s12]``（c12 = cos(θ1+θ2)）。
        """
        q = np.asarray(q, dtype=float).reshape(2)
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c12, s12 = np.cos(q[0] + q[1]), np.sin(q[0] + q[1])
        return np.array([self.l1 * c1 + self.l2 * c12, self.l1 * s1 + self.l2 * s12])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """解析位置雅可比 J = ∂f/∂q（教程 (6.3) 的速度映射 J(q)q̇）。

        Args:
            q: (2,) 关节角，单位 rad。

        Returns:
            (2, 2) 雅可比 ``[[-l1s1 - l2s12, -l2s12], [l1c1 + l2c12, l2c12]]``。
        """
        q = np.asarray(q, dtype=float).reshape(2)
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array(
            [
                [-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                [self.l1 * c1 + self.l2 * c12, self.l2 * c12],
            ]
        )

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """关节空间惯性阵 M(q)（教程 (6.2)，M ≻ 0）。

        Args:
            q: (2,) 关节角，单位 rad。

        Returns:
            (2, 2) 对称正定惯性阵，单位 kg·m²。
        """
        q = np.asarray(q, dtype=float).reshape(2)
        c2 = np.cos(q[1])
        # 依据：Craig ch6 平面 2R 的标准闭式（Lagrange 建模）。
        m11 = self.m1 * self.lc1**2 + self.I1 + self.m2 * (
            self.l1**2 + self.lc2**2 + 2.0 * self.l1 * self.lc2 * c2
        ) + self.I2
        m12 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * c2) + self.I2
        m22 = self.m2 * self.lc2**2 + self.I2
        return np.array([[m11, m12], [m12, m22]])

    def coriolis_torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """科氏/离心力矩 C(q, q̇)q̇（教程 (6.2) 的 C q̇ 合并记法）。

        Args:
            q: (2,) 关节角，单位 rad。
            dq: (2,) 关节角速度，单位 rad/s。

        Returns:
            (2,) 科氏与离心力矩，单位 N·m。
        """
        q = np.asarray(q, dtype=float).reshape(2)
        dq = np.asarray(dq, dtype=float).reshape(2)
        h = -self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        # 依据：Christoffel 记号的平面 2R 闭式（Craig ch6）。
        c_vec = np.array(
            [
                h * (2.0 * dq[0] * dq[1] + dq[1] * dq[1]),
                -h * (dq[0] * dq[0]),
            ]
        )
        return c_vec

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """重力力矩 g(q)（教程 (6.2) 的 g(q)；重力沿 -y）。

        Args:
            q: (2,) 关节角，单位 rad。

        Returns:
            (2,) 重力补偿力矩，单位 N·m。
        """
        q = np.asarray(q, dtype=float).reshape(2)
        g = 9.81
        c1 = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        tau1 = (self.m1 * self.lc1 + self.m2 * self.l1) * g * c1
        tau1 += self.m2 * self.lc2 * g * c12
        tau2 = self.m2 * self.lc2 * g * c12
        return np.array([tau1, tau2])

    def dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """关节动力学正解（教程 (6.2)）：给 τ 解 q̈。

        Args:
            q: (2,) 关节角，单位 rad。
            dq: (2,) 关节角速度，单位 rad/s。
            tau: (2,) 关节力矩，单位 N·m。

        Returns:
            (2,) 关节角加速度，单位 rad/s²。
        """
        M = self.mass_matrix(q)
        rhs = np.asarray(tau, dtype=float).reshape(2)
        rhs = rhs - self.coriolis_torque(q, dq) - self.gravity(q)
        # 依据：(6.2) 移项 + M ≻ 0 可逆（教程 (6.2) 后注）。
        return np.linalg.solve(M, rhs)


def numeric_jacobian(arm: PlanarArm, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """中心差分数值雅可比（教学对照基准：解析 J 的一致性验证）。

    Args:
        arm: :class:`PlanarArm`。
        q: (2,) 展开点关节角，单位 rad。
        eps: 差分步长，单位 rad。

    Returns:
        (2, 2) 数值雅可比。
    """
    q = np.asarray(q, dtype=float).reshape(2)
    J = np.zeros((2, 2))
    for j in range(2):
        qp, qm = q.copy(), q.copy()
        qp[j] += eps
        qm[j] -= eps
        J[:, j] = (arm.fk(qp) - arm.fk(qm)) / (2.0 * eps)
    return J


def dls_ik(
    arm: PlanarArm,
    q0: np.ndarray,
    x_target: np.ndarray,
    n_iters: int = 200,
    tol: float = 1e-8,
    damping: float = 0.05,
) -> tuple[np.ndarray, float, list[float]]:
    """阻尼最小二乘（DLS）逆运动学（教程第 02 章 (2.7)–(2.10) 的迭代形式）。

    每步取 ``Δq = Jᵀ(JJᵀ + λ²I)⁻¹ e``——最小范数伪逆 (2.10) 加阻尼 λ：
    接近奇异位形（‖J‖ 小的方向）时 Δq 被 λ 压住而**不发散**；与第 06 章
    (6.7) 动力学一致伪逆的差别在最小化的范数（速度范数 vs 动能），
    本函数只求位置级 IK，用速度范数版即可。

    Args:
        arm: :class:`PlanarArm`。
        q0: (2,) 初始关节角，单位 rad。
        x_target: (2,) 目标末端位置，单位 m（须在工作空间内）。
        n_iters: 最大迭代数。
        tol: 位置误差收敛阈值，单位 m。
        damping: 阻尼 λ（DLS 的正则化，单位 m 换算的等效量）。

    Returns:
        (q, final_err, err_hist)：解出的关节角 (2,)、最终位置误差（m）、
        逐迭代误差历史。
    """
    q = np.asarray(q0, dtype=float).reshape(2).copy()
    x_target = np.asarray(x_target, dtype=float).reshape(2)
    err_hist: list[float] = []
    err = float(np.linalg.norm(arm.fk(q) - x_target))
    for _ in range(n_iters):
        if err < tol:
            break
        e = x_target - arm.fk(q)  # (2,) 位置误差
        J = arm.jacobian(q)  # (2, 2)
        # DLS 增量：Δq = Jᵀ(JJᵀ + λ²I)⁻¹e（依据：(2.7)–(2.10) 加阻尼）。
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(2), e)
        q = q + dq
        err = float(np.linalg.norm(arm.fk(q) - x_target))
        err_hist.append(err)
    return q, err, err_hist


def impedance_torque(
    arm: PlanarArm,
    q: np.ndarray,
    dq: np.ndarray,
    x_d: np.ndarray,
    dx_d: np.ndarray,
    k_imp: np.ndarray,
    d_imp: np.ndarray,
    gravity_comp: bool = True,
) -> np.ndarray:
    """任务空间阻抗控制律 ``τ = Jᵀ(K e + D ė) (+ g(q))``（教程 (6.12) 的实现）。

    Hogan 阻抗口径：把末端渲染成"锚在 x_d 的弹簧-阻尼"，期望加速度
    ``ẍ_des = M⁻¹(K e + D ė)`` 的力矩级实现取最简形式——经 (6.4) 的
    虚功映射 τ = JᵀF 落到关节（Khatib (6.1) 的 τ = JᵀF 桥）。完整
    Khatib 版本还会做 (6.10) 的 Λ/μ/p 补偿；教学实现以关节空间重力
    补偿 g(q) 替代 p 项（消除静态下垂，使 (6.16) 的稳态柔顺关系干净），
    惯性/科氏失配的影响在 tests/test_osc_arm.py 的容差里体现。

    Args:
        arm: :class:`PlanarArm`。
        q: (2,) 当前关节角，单位 rad。
        dq: (2,) 当前关节角速度，单位 rad/s。
        x_d: (2,) 任务空间锚点（期望末端位置），单位 m。
        dx_d: (2,) 锚点速度（静止目标取 0），单位 m/s。
        k_imp: (2, 2) 刚度阵 K（N/m 量纲）；常传对角阵。
        d_imp: (2, 2) 阻尼阵 D（N·s/m 量纲）；临界阻尼参照 (6.19)
            ``D_c = 2√(MK)`` 选取。
        gravity_comp: True 时叠加关节空间重力补偿 g(q)。

    Returns:
        (2,) 关节力矩指令，单位 N·m。
    """
    q = np.asarray(q, dtype=float).reshape(2)
    dq = np.asarray(dq, dtype=float).reshape(2)
    x_d = np.asarray(x_d, dtype=float).reshape(2)
    dx_d = np.asarray(dx_d, dtype=float).reshape(2)
    k_imp = np.asarray(k_imp, dtype=float).reshape(2, 2)
    d_imp = np.asarray(d_imp, dtype=float).reshape(2, 2)
    # 末端实际速度 ẋ = J q̇（(6.3)；阻抗律的 D ė 项需要它）。
    x_dot = arm.jacobian(q) @ dq
    e = x_d - arm.fk(q)  # (2,) 位置偏差（教程 06.2 第五步的 e）
    e_dot = dx_d - x_dot  # (2,) 速度偏差
    # 依据：(6.12) 移项的弹簧-阻尼力 + (6.4) 的虚功映射 τ = JᵀF。
    tau = arm.jacobian(q).T @ (k_imp @ e + d_imp @ e_dot)
    if gravity_comp:
        tau = tau + arm.gravity(q)
    return tau


def simulate_impedance(
    arm: PlanarArm,
    q0: np.ndarray,
    x_d: np.ndarray,
    k_imp: np.ndarray,
    d_imp: np.ndarray,
    dt: float = 1e-3,
    n_steps: int = 4000,
    dq0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """阻抗闭环仿真（教程 (6.13)：Λ ë + D ė + K e = 0 的自由空间口径）。

    半隐式欧拉积分（先更新 dq 再更新 q——对刚性二阶系统比显式欧拉稳），
    力矩每步由 :func:`impedance_torque` 重算。用于验证阻抗控制把偏移
    位形拉回锚点（(6.14) 的 Lyapunov 单调下降 → (6.16) 的稳态零偏差）。

    Args:
        arm: :class:`PlanarArm`。
        q0: (2,) 初始关节角（含相对 x_d 的偏移），单位 rad。
        x_d: (2,) 锚点，单位 m。
        k_imp / d_imp: 刚度/阻尼阵 (2, 2)。
        dt: 积分步长，单位 s。
        n_steps: 步数 T。
        dq0: (2,) 初始关节角速度，单位 rad/s；None 取 0。

    Returns:
        (q_traj (T+1, 2), x_traj (T+1, 2))：关节与末端轨迹。
    """
    q = np.asarray(q0, dtype=float).reshape(2).copy()
    dq = np.zeros(2) if dq0 is None else np.asarray(dq0, dtype=float).reshape(2).copy()
    x_d = np.asarray(x_d, dtype=float).reshape(2)
    q_traj = np.empty((n_steps + 1, 2))
    x_traj = np.empty((n_steps + 1, 2))
    q_traj[0] = q
    x_traj[0] = arm.fk(q)
    for t in range(n_steps):
        tau = impedance_torque(arm, q, dq, x_d, np.zeros(2), k_imp, d_imp)
        qdd = arm.dynamics(q, dq, tau)
        dq = dq + dt * qdd  # 半隐式欧拉：先速度后位置（数值稳定性依据）
        q = q + dt * dq
        q_traj[t + 1] = q
        x_traj[t + 1] = arm.fk(q)
    return q_traj, x_traj

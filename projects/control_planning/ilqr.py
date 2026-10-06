"""iLQR：迭代线性二次调节器（2D 双积分器轨迹优化，教学实现）。

函数流水线
----------
Jacobson & Mayne, *Differential Dynamic Programming*, 1970（DDP 源头）与
Tassa, Erez, Todorov, ICRA 2012（iLQG，控制受限/正则化的工程形态）的
教学实现，推导见教程第 04 章 04.2（压轴节）。被控对象为 2D 双积分器
（点质量）：状态 ``x = [px, py, vx, vy]``、控制 ``u = [ax, ay]``，离散
动力学为精确欧拉（线性系统 + 分段常值输入下无离散化误差）——简单到
可手推、复杂到能演示"反向递推 + 前向回滚"的全部算法结构（教程 (4.5)）。

调用链：:func:`double_integrator_step` / :func:`double_integrator_jacobians`
（(4.5) 的 f 与解析 f_x、f_u；数值版 :func:`numeric_dynamics_jacobians`
用中心差分，被 tests/test_ilqr.py 交叉验证）→ :class:`ReachCost`（阶段/终端
代价及其解析 1-2 阶导数，教程 (4.1) 的离散形态）→ :func:`ilqr_solve`
（反向递推 (4.8)–(4.12) 求仿射反馈 (4.11)，前向回滚 (4.13) + 接受判定，
正则化 (4.14) 自适应）→ :class:`ILQRResult`。线性二次特例由
:func:`riccati_lqr`（教程 05 章 (5.5) 的 Riccati 递推）与
:func:`riccati_fixed_point`（(5.6) 代数 Riccati 定点）交叉验证。

iLQR 一次迭代为：

1. **反向扫**：沿标称轨迹逐级展开 Q 函数 (4.7)，取其四块 (4.8)–(4.9)，
   解出前馈 ``k`` 与反馈增益 ``K``（(4.11)；``Q_uu`` 不正定时用阻尼逆
   (4.14)）并累计预期下降 ΔV；
2. **前向扫**：以步长 α 回滚新轨迹（(4.13)），按"实际下降逼近预期下降"
   的 Armijo 型判据接受/拒绝（Tassa 2012/2014 口径）；
3. 拒绝则 α 减半重试、再失败则增大正则化 μ 重做反向扫；接受且增益比
   接近 1 时减小 μ——信赖域逻辑，与 LM 阻尼同构（教程 (4.14) 行内注）。

约定
----
- 状态/控制的形状：``x (4,) = [px, py, vx, vy]``（m, m/s）、
  ``u (2,) = [ax, ay]``（m/s²）；轨迹批形状见 :class:`ILQRResult`。
- 所有随机性（无）：本模块完全确定性，无需 seed。

示例
----
>>> import numpy as np
>>> from ilqr import ReachCost, ilqr_solve, double_integrator_step
>>> cost = ReachCost(w_u=0.1, goal=np.array([2.0, 1.0]), w_goal=100.0)
>>> res = ilqr_solve(np.zeros(4), np.zeros((40, 2)), cost, dt=0.1)
>>> end = res.x_traj[-1]
>>> float(np.linalg.norm(end[:2] - cost.goal)) < 0.05  # doctest: +SKIP
True
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ILQRResult",
    "ReachCost",
    "double_integrator_jacobians",
    "double_integrator_step",
    "ilqr_solve",
    "numeric_dynamics_jacobians",
    "riccati_fixed_point",
    "riccati_lqr",
]


# --------------------------------------------------------------- 被控对象 --
def double_integrator_step(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """2D 双积分器一步离散动力学（教程 (4.5) 的 ``x_{i+1} = f(x_i, u_i)``）。

    欧拉离散对本系统**精确**：动力学线性、输入分段常值时一步转移与
    连续时间解逐位一致——标称轨迹的"精确在轨"（教程 (4.7) 推导前提）
    因此无离散化残差。

    Args:
        x: (4,) 状态 ``[px, py, vx, vy]``，单位 m, m/s。
        u: (2,) 控制 ``[ax, ay]``，单位 m/s²。
        dt: 步长，单位 s，> 0。

    Returns:
        (4,) 下一状态 ``[px + vx·dt, py + vy·dt, vx + ax·dt, vy + ay·dt]``。
    """
    x = np.asarray(x, dtype=float).reshape(4)
    u = np.asarray(u, dtype=float).reshape(2)
    nx = np.empty(4)
    nx[0] = x[0] + x[2] * dt
    nx[1] = x[1] + x[3] * dt
    nx[2] = x[2] + u[0] * dt
    nx[3] = x[3] + u[1] * dt
    return nx


def double_integrator_jacobians(
    x: np.ndarray, u: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """离散动力学的解析雅可比 ``(A, B) = (∂f/∂x, ∂f/∂u)``（教程 (4.5)）。

    线性系统下雅可比与 ``(x, u)`` 无关——解析式一眼可写，正适合作为
    :func:`numeric_dynamics_jacobians` 的交叉验证基准。

    Args:
        x: (4,) 状态（占位：雅可比不依赖它，保持统一签名）。
        u: (2,) 控制（同上）。
        dt: 步长，单位 s。

    Returns:
        A: (4, 4) 状态雅可比 ``[[I, dt·I], [0, I]]``。
        B: (4, 2) 控制雅可比 ``[[0], [dt·I]]``。
    """
    del x, u  # 线性系统：雅可比为常数，参数仅为签名统一
    A = np.eye(4)
    A[0, 2] = dt
    A[1, 3] = dt
    B = np.zeros((4, 2))
    B[2, 0] = dt
    B[3, 1] = dt
    return A, B


def numeric_dynamics_jacobians(
    step_fn, x: np.ndarray, u: np.ndarray, dt: float, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """数值雅可比：对任意 ``step_fn(x, u, dt)`` 做中心差分（教学对照基准）。

    中心差分误差 O(eps²)——eps 取 1e-6 时对光滑 f 的截断误差 ~1e-14、
    舍入误差 ~1e-10，足以分辨"雅可比写错"（通常 > 1e-3）与实现正确。

    Args:
        step_fn: 形如 :func:`double_integrator_step` 的 ``(x, u, dt) -> x'``。
        x: (4,) 展开点状态。
        u: (2,) 展开点控制。
        dt: 步长，单位 s。
        eps: 差分步长。

    Returns:
        (A (4, 4), B (4, 2)) 中心差分雅可比。
    """
    x = np.asarray(x, dtype=float).reshape(4)
    u = np.asarray(u, dtype=float).reshape(2)
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    for j in range(4):  # 中心差分 ∂f/∂x_j
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        A[:, j] = (step_fn(xp, u, dt) - step_fn(xm, u, dt)) / (2.0 * eps)
    for j in range(2):  # 中心差分 ∂f/∂u_j
        up, um = u.copy(), u.copy()
        up[j] += eps
        um[j] -= eps
        B[:, j] = (step_fn(x, up, dt) - step_fn(x, um, dt)) / (2.0 * eps)
    return A, B


# --------------------------------------------------------------- 代价结构 --
@dataclass
class ReachCost:
    """双积分器"到达 + 中途路标点"代价（教程 (4.1) 的离散二次形态）。

    阶段代价 ``l_i(x, u) = w_u‖u‖² + [i == 路标步]·w_wp‖p - 路标‖²``，
    终端代价 ``l_f(x) = w_goal‖p - goal‖² + w_vf‖v‖²``——控制功效、
    路标牵引与终端到达分别对应教程 04.1 "能耗 / 走廊 / 目标"的三类
    desiderata。全部为 x、u 的二次型，1-2 阶导数解析可写（反向递推
    (4.8)–(4.9) 直接取用，无近似）。

    Attributes:
        w_u: 控制功效权重，无量纲（> 0 保证 Q_uu 正定）。
        goal: (2,) 终端目标位置，单位 m。
        w_goal: 终端位置权重。
        w_vf: 终端速度权重（压终端速度，m²/s² 量纲的权重）。
        waypoint: 可选 ``(步下标 i_wp, 位置 (2,))`` 路标点；None 表示无。
        w_wp: 路标点权重。
    """

    w_u: float
    goal: np.ndarray
    w_goal: float
    w_vf: float = 1.0
    waypoint: tuple[int, np.ndarray] | None = None
    w_wp: float = 100.0

    def stage(self, i: int, x: np.ndarray, u: np.ndarray) -> tuple:
        """第 i 步阶段代价及其 1-2 阶导数（教程 (4.8)–(4.9) 的 ℓ 侧）。

        Args:
            i: 步下标（用于判定是否落在路标步上）。
            x: (4,) 状态。
            u: (2,) 控制。

        Returns:
            ``(l, l_x (4,), l_u (2,), l_xx (4,4), l_uu (2,2), l_ux (2,4))``；
            本代价不含 x-u 交叉项，l_ux 恒为 0。
        """
        x = np.asarray(x, dtype=float).reshape(4)
        u = np.asarray(u, dtype=float).reshape(2)
        # 控制功效项：w_u ‖u‖²（依据：定义行；导数 2 w_u u）。
        l = self.w_u * float(u @ u)
        l_u = 2.0 * self.w_u * u
        l_xx = np.zeros((4, 4))
        l_uu = 2.0 * self.w_u * np.eye(2)
        if self.waypoint is not None and i == self.waypoint[0]:
            wp = np.asarray(self.waypoint[1], dtype=float).reshape(2)
            e = x[:2] - wp  # (2,) 位置对路标的偏差
            l += self.w_wp * float(e @ e)
            l_x = np.zeros(4)
            l_x[:2] = 2.0 * self.w_wp * e
            l_xx[:2, :2] = 2.0 * self.w_wp * np.eye(2)
        else:
            l_x = np.zeros(4)
        l_ux = np.zeros((2, 4))
        return l, l_x, l_u, l_xx, l_uu, l_ux

    def terminal(self, x: np.ndarray) -> tuple:
        """终端代价及其 1-2 阶导数（横截条件 ℓ_{f,x}，教程 (4.4) 末式）。

        Args:
            x: (4,) 终端状态。

        Returns:
            ``(l_f, l_fx (4,), l_fxx (4,4))``。
        """
        x = np.asarray(x, dtype=float).reshape(4)
        ep = x[:2] - self.goal  # (2,) 终端位置偏差
        l = self.w_goal * float(ep @ ep) + self.w_vf * float(x[2:] @ x[2:])
        l_fx = np.zeros(4)
        l_fx[:2] = 2.0 * self.w_goal * ep
        l_fx[2:] = 2.0 * self.w_vf * x[2:]
        l_fxx = 2.0 * np.diag([self.w_goal, self.w_goal, self.w_vf, self.w_vf])
        return l, l_fx, l_fxx


# ------------------------------------------------- Riccati 交叉验证基准 --
def riccati_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    n_steps: int,
    p_f: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """有限时域 Riccati 反向递推（教程 05 章 (5.4)–(5.5)）——iLQR 的线性特例。

    对线性动力学 ``x+ = A x + B u``、二次代价 ``xᵀQx + uᵀRu``，DP 递推
    (5.3) 的二次型不变性给出 ``u* = -K_k x`` 与 ``P_k`` 递推——这正是
    iLQR 反向递推 (4.8)–(4.12) 在"线性 + 二次"下的精确特例（教程 05.1
    ⑤ 的论证）。本函数作为 iLQR 的交叉验证基准（tests/test_ilqr.py）。

    Args:
        A: (n, n) 状态转移阵。
        B: (n, m) 控制阵。
        Q: (n, n) 状态二次代价阵（代价取 xᵀQx，Q ⪰ 0）。
        R: (m, m) 控制二次代价阵（代价取 uᵀRu，R ≻ 0）。
        n_steps: 时域步数 N。
        p_f: (n, n) 终端值函数阵 V_f = xᵀP_f x；None 时取零阵。

    Returns:
        (P_list, K_list)：各含 N 个数组，``P_list[i] (n, n)`` 为第 i 步值
        函数阵、``K_list[i] (m, n)`` 为第 i 步增益（i = 0..N-1）。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    P = np.zeros_like(Q) if p_f is None else np.asarray(p_f, dtype=float).copy()
    p_list: list[np.ndarray] = []
    k_list: list[np.ndarray] = []
    for _ in range(n_steps):
        # 驻点条件 (5.4)：(R + BᵀPB) u = -BᵀPA x → u* = -K x。
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        # Riccati 递推 (5.5)：P_k = Q + AᵀPA - AᵀPB(R + BᵀPB)⁻¹BᵀPA。
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        P = 0.5 * (P + P.T)  # 对称化（数值卫生，理论值本就对称）
        p_list.append(P.copy())
        k_list.append(K)
    return p_list, k_list


def riccati_fixed_point(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iters: int = 100000,
    tol: float = 1e-12,
) -> np.ndarray:
    """代数 Riccati 方程 (ARE) 的定点迭代（教程 05 章 (5.6)）。

    无限时域 LQR 的值函数阵 P^∞ 是递推 (5.5) 的不动点——迭代至
    ‖P_{k+1} - P_k‖ < tol 即得 ARE 的解（可稳定/可检测条件下的标准结果，
    教程 05.1 第四步）。用于检验"MPC/LQR 等价" (5.7) 的数值口径。

    Args:
        A/B/Q/R: 同 :func:`riccati_lqr`。
        max_iters: 最大迭代数。
        tol: 收敛阈值（Frobenius 范数）。

    Returns:
        (n, n) ARE 的定常解 P^∞。
    """
    P = np.asarray(Q, dtype=float).copy()
    for _ in range(max_iters):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        P_next = 0.5 * (P_next + P_next.T)
        if float(np.linalg.norm(P_next - P)) < tol:
            return P_next
        P = P_next
    return P


# ------------------------------------------------------------- iLQR 主程 --
@dataclass
class ILQRResult:
    """iLQR 求解结果。

    Attributes:
        x_traj: (N+1, 4) 标称状态轨迹（含初态）。
        u_traj: (N, 2) 标称控制序列。
        k_ff: (N, 2) 前馈修正（(4.11) 的 k）——最后次反向扫的值。
        K_fb: (N, 2, 4) 反馈增益（(4.11) 的 K）——最后次反向扫的值。
        costs: (iters+1,) 每次迭代的轨迹代价（第 0 项为初值）。
        mu_hist: (n_updates,) 正则化系数 μ 的更新历史（(4.14)）。
        ratio_hist: (n_accept,) 接受步的实际/预期下降比（增益比，
            Armijo 口径的仪表，DEBUG.md 变量表 I-4）。
        converged: 是否满足收敛判据。
        n_iters: 实际迭代数。
    """

    x_traj: np.ndarray
    u_traj: np.ndarray
    k_ff: np.ndarray
    K_fb: np.ndarray
    costs: np.ndarray
    mu_hist: list[float] = field(default_factory=list)
    ratio_hist: list[float] = field(default_factory=list)
    converged: bool = False
    n_iters: int = 0


def _rollout_cost(
    x0: np.ndarray,
    u_seq: np.ndarray,
    cost: ReachCost,
    dt: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """给定控制序列的完整轨迹与总代价（教程 (4.5) 的 J 离散求和）。"""
    n = u_seq.shape[0]
    xs = np.empty((n + 1, 4))
    xs[0] = x0
    total = 0.0
    for i in range(n):
        xs[i + 1] = double_integrator_step(xs[i], u_seq[i], dt)
        l, _, _, _, _, _ = cost.stage(i, xs[i], u_seq[i])
        total += l
    lf, _, _ = cost.terminal(xs[n])
    return total + lf, xs, u_seq


def ilqr_solve(
    x0: np.ndarray,
    u_init: np.ndarray,
    cost: ReachCost,
    dt: float,
    max_iters: int = 50,
    tol: float = 1e-6,
    mu0: float = 1.0,
    mu_max: float = 1e10,
) -> ILQRResult:
    """iLQR 主循环：反向递推 (4.8)–(4.12) → 前向回滚 (4.13) → 接受判定。

    每次迭代先用阻尼逆 (4.14) 解出仿射反馈策略 δu = k + K·δx（预期下降
    ΔV = -Σ½ kᵀQ_uu k，见教程 (4.12) 下方），再以步长 α ∈ {1, 1/2, …}
    回滚；接受判据为 Tassa 2012/2014 的 Armijo 型条件（实际下降须为正，
    且与预期下降的比值被记录为增益比仪表）。α 全被拒时增大 μ 重做反向
    扫——"二次近似 + 阻尼信赖域"（教程 (4.14) 行内注，与 LM 同构）。

    Args:
        x0: (4,) 初始状态。
        u_init: (N, 2) 控制序列初值（第 03 章路径经时间分配充当）。
        cost: :class:`ReachCost` 代价结构。
        dt: 离散步长，单位 s。
        max_iters: 最大迭代数。
        tol: 代价相对收敛阈值（相对 ‖J‖ 的比例）。
        mu0: 正则化初值。
        mu_max: 正则化上限（超过即判失败退出）。

    Returns:
        :class:`ILQRResult`。
    """
    x0 = np.asarray(x0, dtype=float).reshape(4)
    u_seq = np.asarray(u_init, dtype=float).reshape(-1, 2).copy()
    n = u_seq.shape[0]
    if n == 0:
        raise ValueError("u_init 不能为空（时域 N ≥ 1）")
    j_curr, xs, _ = _rollout_cost(x0, u_seq, cost, dt)
    costs = [j_curr]
    mu = float(mu0)
    mu_hist: list[float] = []
    ratio_hist: list[float] = []
    k_ff = np.zeros((n, 2))
    k_fb = np.zeros((n, 2, 4))
    converged = False

    for _ in range(max_iters):
        # ---------------- 反向扫（(4.8)–(4.12)；Q_uu 不正定时 (4.14) 阻尼）--
        while True:  # μ 自适应循环：不正定 → 增大 μ 重来（信赖域收紧）
            _, lfx, lfxx = cost.terminal(xs[n])
            v_x, v_xx = lfx.copy(), lfxx.copy()
            k_ff = np.zeros((n, 2))
            k_fb = np.zeros((n, 2, 4))
            d_v = 0.0  # 预期下降 ΔV = Σ (kᵀQ_u + ½kᵀQ_uu k)（α=1 时）
            regularized = False
            for i in range(n - 1, -1, -1):
                l, l_x, l_u, l_xx, l_uu, l_ux = cost.stage(i, xs[i], u_seq[i])
                # 离散雅可比（线性系统：解析式；教程 (4.5) 后注）。
                A, B = double_integrator_jacobians(xs[i], u_seq[i], dt)
                # Q 函数四块（(4.8)–(4.9)；iLQR 只取一阶动力学展开）。
                q_x = l_x + A.T @ v_x
                q_u = l_u + B.T @ v_x
                q_xx = l_xx + A.T @ v_xx @ A
                q_ux = l_ux + B.T @ v_xx @ A
                q_uu = l_uu + B.T @ v_xx @ B
                # 阻尼逆 (4.14)：(Q_uu + μI)⁻¹——最小特征值 ≥ μ 保证可解
                # 且更新方向为下降方向（依据：教程 (4.14) 行内注）。
                q_uu_reg = q_uu + mu * np.eye(2)
                try:
                    # Cholesky 试分解 = 最便宜的正定判定（失败即不正定）。
                    np.linalg.cholesky(q_uu_reg)
                except np.linalg.LinAlgError:
                    mu = min(mu * 10.0, mu_max)  # 不正定：μ ×10 收紧信赖域
                    mu_hist.append(mu)
                    if mu >= mu_max:
                        break
                    regularized = True
                    break
                # 仿射反馈 (4.11)：k = -Q_uu⁻¹Q_u，K = -Q_uu⁻¹Q_ux。
                sol_u = np.linalg.solve(q_uu_reg, np.column_stack([q_u, q_ux]))
                k_i = -sol_u[:, 0]
                k_i_full = -sol_u[:, 1:]
                k_ff[i] = k_i
                k_fb[i] = k_i_full.reshape(2, 4)
                # 预期下降 ΔV_i = kᵀQ_u + ½kᵀQ_uu k（(4.12) 下方的配方法）。
                # 用正则化后的 Q_uu_reg：k 本就是该阻尼模型的最优步（Tassa
                # 2012 的 ΔV 口径）——否则强阻尼步的增益比会被系统性低估。
                d_v += float(k_i @ q_u) + 0.5 * float(k_i @ q_uu_reg @ k_i)
                # 值函数更新 (4.12)：V_x = Q_x - KᵀQ_uu k，V_xx = Q_xx - KᵀQ_uu K。
                v_x = q_x - k_i_full.T @ q_uu @ k_i
                v_xx = q_xx - k_i_full.T @ q_uu @ k_i_full
            if not regularized:
                break
            if mu >= mu_max:
                break

        # ---------------- 前向回滚（(4.13)）+ Armijo 型接受判定 ------------
        accepted = False
        alpha = 1.0
        while alpha >= 1.0 / 64.0:
            xs_new = np.empty_like(xs)
            xs_new[0] = x0
            u_new = np.empty_like(u_seq)
            for i in range(n):
                # 依据：(4.13)——前馈缩 α、反馈项随实际偏离自适应。
                du = alpha * k_ff[i] + k_fb[i] @ (xs_new[i] - xs[i])
                u_new[i] = u_seq[i] + du
                xs_new[i + 1] = double_integrator_step(xs_new[i], u_new[i], dt)
            j_new, _, _ = _rollout_cost(x0, u_new, cost, dt)
            decrease = j_curr - j_new
            expected = -d_v  # 预期下降 ≥ 0（d_v ≤ 0，Q_uu ≻ 0 时配方法保证）
            # Armijo 型接受（Tassa 2012/2014 口径）：实际下降须为正；
            # 增益比 = 实际/预期 记入仪表（DEBUG.md 变量表 I-4）。
            if decrease > 0.0:
                ratio = decrease / expected if expected > 1e-300 else 1.0
                ratio_hist.append(ratio)
                xs, u_seq, j_curr = xs_new, u_new, j_new
                accepted = True
                break
            alpha *= 0.5  # 依据：(4.13) 的线搜索——步长减半重试
        costs.append(j_curr)
        # 增益比接近 1（预测可信）→ 放松 μ；线搜索被砍短 → 收紧 μ。
        if accepted and ratio_hist and ratio_hist[-1] > 0.5 and mu > 1e-9:
            mu = max(mu * 0.5, 1e-9)
            mu_hist.append(mu)
        elif not accepted:
            mu = min(mu * 10.0, mu_max)
            mu_hist.append(mu)
            if mu >= mu_max:
                break
        rel = abs(costs[-2] - costs[-1]) / max(1.0, abs(costs[-1]))
        if rel < tol:
            converged = True
            break

    return ILQRResult(
        x_traj=xs,
        u_traj=u_seq,
        k_ff=k_ff,
        K_fb=k_fb,
        costs=np.asarray(costs),
        mu_hist=mu_hist,
        ratio_hist=ratio_hist,
        converged=converged,
        n_iters=len(costs) - 1,
    )

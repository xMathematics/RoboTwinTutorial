"""CBF 安全滤波器：控制屏障函数 QP（教学实现）。

函数流水线
----------
Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada, *Control Barrier
Functions: Theory and Applications*, ECC 2019（本地
papers/control_planning/classics/arXiv-1903.11199_CBF-Survey.pdf）的
CBF-QP 安全滤波教学实现（综述 Definition 2 / (10) / Theorem 2），推导见
教程第 07 章 07.1–07.2（压轴节）。被控对象为 2D 双积分器（控制仿射系统
(7.6)：f(x) = [vx, vy, 0, 0]、g = [[0,0],[0,0],[1,0],[0,1]]），障碍为
圆域集合；屏障 h 取"距离 + 速度阻尼"的一阶提升——原始距离屏障对加速度
输入相对阶为 2（教程 07.2 ⑤ 的失效边界），按综述 §IV 指数 CBF 的口径
把状态提升到速度级恢复相对阶 1。

调用链：:func:`barrier_affine`（单障碍的 h、L_f h、L_g h，教程 (7.7) 李导数
展开）→ :func:`cbf_filter`（QP (7.9)：单约束闭式解 (7.13)，多约束投影迭代）
→ :func:`simulate_filtered`（"无滤波撞入 / 有滤波保持 h ≥ -tol"的对照仿真）
→ :class:`CBFSimResult`。被 ``tests/test_cbf.py`` 与 ``demo.py`` 驱动。

QP (7.9) 的求解（教程 07.2 ⑤）：

1. 把约束写成 ``A u + b ≥ 0``（A 各行 = L_g h_i，b_i = L_f h_i + α(h_i)）；
2. **单约束/批量闭式**：(7.13) 的批量推广
   ``u* = u_des + Aᵀ (A Aᵀ + εI)⁻¹ max(0, -(A u_des + b))``——
   εI 保证 Gram 阵可逆（约束行共线/零行时退化安全）；
3. **多约束投影迭代**：闭式更新对相互冲突的行只是近似，之后做 POCS
   （逐行向半空间投影的循环迭代）压残余违反到机器精度。

约定
----
- 状态 ``x = [px, py, vx, vy]``（m, m/s）、控制 ``u = [ax, ay]``（m/s²）。
- 安全集 (7.1)：``h(x) ≥ 0``；h 的速度阻尼系数 ``kappa`` 单位 s
  （把速度失配折算成等效距离）。
- 离散口径注明：连续条件 (7.4) 经欧拉离散在线性被控对象上**精确**成立
  （双积分器 + 分段常值 u 无离散化误差），唯一近似来自 h 的非线性在
  dt 网格上的二阶余量——tests/test_cbf.py 的容差即按此标注。

示例
----
>>> import numpy as np
>>> from cbf import cbf_filter
>>> x = np.array([0.0, 0.0, 1.0, 0.0])           # 向 +x 匀速
>>> u_des = np.array([1.0, 0.0])                 # 想继续加速
>>> u_safe, info = cbf_filter(u_des, x, [(np.array([2.0, 0.0]), 0.5)])
>>> float(info["h"][0]) > 0                        # doctest: +SKIP
True
"""
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CBFSimResult",
    "affine_fields",
    "barrier_affine",
    "cbf_filter",
    "min_obstacle_distance",
    "project_halfspaces",
    "simulate_filtered",
]


def affine_fields(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """双积分器的控制仿射分解 ``ẋ = f(x) + g(x) u``（教程 (7.6)）。

    Args:
        x: (4,) 状态 ``[px, py, vx, vy]``。

    Returns:
        f: (4,) 漂移场 ``[vx, vy, 0, 0]``。
        g: (4, 2) 输入场（加速度直接进速度通道）。
    """
    x = np.asarray(x, dtype=float).reshape(4)
    f = np.array([x[2], x[3], 0.0, 0.0])
    g = np.zeros((4, 2))
    g[2, 0] = 1.0
    g[3, 1] = 1.0
    return f, g


def barrier_affine(
    x: np.ndarray,
    center: np.ndarray,
    radius: float,
    kappa: float = 0.4,
) -> tuple[float, float, np.ndarray]:
    """单圆障碍的屏障 h 及其李导数（教程 (7.7) 的逐障碍实现）。

    屏障取"距离 + 速度阻尼"的一阶提升（综述 §IV 指数 CBF 的实用形式）：

    $$h(x) = \\big(\\|p - c\\| - r\\big) + \\kappa\\, v^{\\top} n, \\quad
    n = \\frac{p - c}{\\|p - c\\|}$$

    原始距离屏障 ‖p-c‖ - r 对加速度输入的相对阶为 2（ḣ = vᵀn 不含 u，
    教程 07.2 ⑤ 的失效边界）；加上 κ vᵀn 后 ḣ 含 κ nᵀu，相对阶恢复 1，
    (7.7) 的李导数展开才给出 u 的线性约束。安全语义：距离裕量为正时
    h > 0；朝障碍高速逼近（vᵀn < 0）按 κ·s 量纲折算提前消耗裕量。

    Args:
        x: (4,) 状态。
        center: (2,) 障碍圆心，单位 m。
        radius: 障碍半径，单位 m。
        kappa: 速度阻尼系数，单位 s，> 0。

    Returns:
        (h, L_fh, L_g h)：屏障值（m）、漂移李导数（m/s）、输入李导数
        (2,)（行向量语义：L_g h · u 进入 (7.8) 的约束）。
    """
    x = np.asarray(x, dtype=float).reshape(4)
    c = np.asarray(center, dtype=float).reshape(2)
    p, v = x[:2], x[2:]
    diff = p - c
    d = float(np.linalg.norm(diff))
    n = diff / max(d, 1e-9)  # 单位法向（d→0 时退化保护，正常场景不触发）
    vn = float(v @ n)
    # h = (d - r) + κ vᵀn（m；κ 的 s 量纲把 m/s 折算成 m）。
    h = (d - radius) + kappa * vn
    # L_f h = ḋ + κ d(vᵀn)/dt|_f = vᵀn + κ vᵀ(I - nnᵀ)v / d
    # （依据：dn/dt = (I - nnᵀ)ṗ/d，链式法则；教程 (7.7) 第一项）。
    v_perp = v - vn * n  # v 的切向分量
    l_fh = vn + kappa * float(v_perp @ v_perp) / max(d, 1e-9)
    # L_g h = ∇h · g = κ nᵀ（依据：(7.7) 第二项；∇h/∂v = κn 打到速度通道）。
    l_gh = kappa * n
    return h, l_fh, l_gh


def project_halfspaces(
    u_des: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """批量半空间投影的闭式解——(7.13) 的多行推广（(L_gh L_ghᵀ + εI) 可逆）。

    对约束 ``A u + b ≥ 0``（可行集 = 半空间之交），单行的 KKT 闭式解
    (7.13) 是到该半空间的正交投影；多行时一次批量更新

    $$u^+ = u_{des} + A^{\\top}\\big(AA^{\\top} + \\varepsilon I\\big)^{-1}
    \\max\\big(0,\\ -(Au_{des} + b)\\big)$$

    在"被违反的行互不冲突"时即精确解，行冲突时是良好初值（之后由
    :func:`cbf_filter` 的投影迭代收尾）。εI 使 Gram 阵恒可逆——行共线
    （多障碍同向）或零行（‖L_g h‖ = 0 的失效边界）时退化安全。

    Args:
        u_des: (m,) 期望控制，单位 m/s²。
        A: (k, m) 约束行（每行一个 L_g h_i）。
        b: (k,) 约束常数（L_f h_i + α(h_i)）。
        eps: Gram 阵正则化 ε，> 0。

    Returns:
        (m,) 投影后的控制。
    """
    u_des = np.asarray(u_des, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1, u_des.shape[0])
    b = np.asarray(b, dtype=float).reshape(-1)
    viol = np.maximum(0.0, -(A @ u_des + b))  # (k,) 各行的违反量
    # (7.13) 批量推广：u⁺ = u_des + Aᵀ(AAᵀ + εI)⁻¹ viol（εI 防奇异）。
    gram = A @ A.T + eps * np.eye(A.shape[0])
    return u_des + A.T @ np.linalg.solve(gram, viol)


def cbf_filter(
    u_des: np.ndarray,
    x: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    alpha: float = 8.0,
    kappa: float = 0.4,
    n_proj_iters: int = 10,
    eps: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    """CBF-QP 安全滤波（教程 (7.9)）：把 u_des 掰进安全控制集 K_cbf (7.8)。

    每个约束一行：``L_f h_i + L_g h_i · u ≥ -α(h_i)``（class-K 取线性
    α(r) = alpha·r，(7.4)/(7.5) 的指数回拉口径）。求解两段：批量闭式
    投影（(7.13) 推广，见 :func:`project_halfspaces`）+ POCS 投影迭代
    压残余违反。u_des 已安全时修改量为零——"最小侵入"（综述 §II.C）。

    Args:
        u_des: (2,) 期望控制（iLQR/MPC/阻抗/RL 策略皆可，教程 07.2 ①）。
        x: (4,) 当前状态。
        obstacle_list: 圆障碍列表，单位 m。
        alpha: 线性 class-K 系数 γ（(7.5) 的回拉速率，1/s）。
        kappa: 屏障速度阻尼系数，单位 s。
        n_proj_iters: POCS 投影迭代轮数。
        eps: Gram 正则化。

    Returns:
        (u_safe, info)：
        u_safe: (2,) 滤波后控制。
        info: 诊断 dict——``h (k,)`` 滤波前屏障值、``A (k,2)``、
        ``b (k,)``、``viol_before``（u_des 的最大约束违反）、
        ``viol_after``（u_safe 的最大约束违反，应 ≤ 数值容差）、
        ``du_norm``（修改量范数，"最小侵入"仪表）。
    """
    u_des = np.asarray(u_des, dtype=float).reshape(2)
    rows_a, rows_b, h_vals = [], [], []
    for center, radius in obstacle_list:
        h, l_fh, l_gh = barrier_affine(x, center, radius, kappa)
        # 约束行：L_g h · u ≥ -(L_f h + α h)，即 A u + b ≥ 0（(7.9) 移项）。
        rows_a.append(l_gh)
        rows_b.append(l_fh + alpha * h)
        h_vals.append(h)
    A = np.stack(rows_a)  # (k, 2)
    b = np.asarray(rows_b)  # (k,)
    u = project_halfspaces(u_des, A, b, eps)
    # POCS 投影迭代：逐行把违反压到 0（行冲突时批量更新只是近似；
    # 半空间之交为凸集，POCS 收敛到交集中一点）。
    for _ in range(n_proj_iters):
        viol = -(A @ u + b)  # (k,) 当前违反量
        if float(viol.max()) <= 1e-12:
            break
        for i in range(A.shape[0]):
            viol_i = -(A[i] @ u + b[i])
            if viol_i > 0.0:
                # 单行闭式投影（(7.13)：u ← u + a_iᵀ(a_i a_iᵀ + ε)⁻¹ viol_i）。
                u = u + A[i] * viol_i / (float(A[i] @ A[i]) + eps)
    info = {
        "h": np.asarray(h_vals),
        "A": A,
        "b": b,
        "viol_before": float(np.maximum(0.0, -(A @ u_des + b)).max()),
        "viol_after": float(np.maximum(0.0, -(A @ u + b)).max()),
        "du_norm": float(np.linalg.norm(u - u_des)),
    }
    return u, info


def min_obstacle_distance(
    traj: np.ndarray, obstacle_list: list[tuple[np.ndarray, float]]
) -> float:
    """轨迹全程的最小障碍净距（min 距离 - 半径），单位 m。

    Args:
        traj: (N, 4) 或 (N, 2) 状态/位置轨迹（自动取位置分量）。
        obstacle_list: 圆障碍列表，单位 m。

    Returns:
        全程最小净距；负值 = 发生穿透（碰撞深度）。
    """
    arr = np.asarray(traj, dtype=float)
    pos = arr[:, :2]
    centers = np.stack([np.asarray(c, dtype=float) for c, _ in obstacle_list])
    radii = np.asarray([r for _, r in obstacle_list], dtype=float)
    d = np.linalg.norm(pos[:, None, :] - centers[None, :, :], axis=2) - radii
    return float(d.min())


@dataclass
class CBFSimResult:
    """对照仿真结果。

    Attributes:
        traj: (T+1, 4) 状态轨迹（双积分器欧拉 = 精确积分）。
        u_applied: (T, 2) 实际执行的控制（滤波后或原始）。
        min_h: 全程最小屏障值（多障碍取逐时刻最小值的最小）。
        min_dist: 全程最小障碍净距（m）。
        u_des_seq: (T, 2) 每步的期望控制（对照用）。
        viol_resid: (T,) 每步滤波后的约束残余违反（投影质量仪表）。
    """

    traj: np.ndarray
    u_applied: np.ndarray
    min_h: float
    min_dist: float
    u_des_seq: np.ndarray
    viol_resid: np.ndarray = field(default_factory=lambda: np.zeros(0))


def simulate_filtered(
    x0: np.ndarray,
    u_des: np.ndarray,
    obstacle_list: list[tuple[np.ndarray, float]],
    dt: float,
    n_steps: int,
    filtered: bool = True,
    alpha: float = 8.0,
    kappa: float = 0.4,
) -> CBFSimResult:
    """在恒定期望控制下仿真"滤波 vs 不滤波"（教程 07.2 的动机场景）。

    双积分器 + 分段常值输入下欧拉积分**精确**（线性系统），故屏障值
    序列的非线性余量只剩 h 本身的曲率项——(7.4) 连续条件的离散容差
    在 tests/test_cbf.py 中按 O(dt²·ḧ) 标注。

    Args:
        x0: (4,) 初始状态。
        u_des: (2,) 恒定期望控制（如"直冲障碍"的加速度指令）。
        obstacle_list: 圆障碍列表，单位 m。
        dt: 步长，单位 s。
        n_steps: 步数 T。
        filtered: True 时每步经 :func:`cbf_filter`（安全层），False 时
            原样执行 u_des（对照）。
        alpha / kappa: 透传给滤波器。

    Returns:
        :class:`CBFSimResult`。
    """
    x = np.asarray(x0, dtype=float).reshape(4)
    u_des = np.asarray(u_des, dtype=float).reshape(2)
    traj = np.empty((n_steps + 1, 4))
    traj[0] = x
    u_seq = np.empty((n_steps, 2))
    u_des_seq = np.empty((n_steps, 2))
    viol_resid = np.zeros(n_steps)
    for t in range(n_steps):
        u = u_des.copy()
        if filtered:
            u, info = cbf_filter(u_des, x, obstacle_list, alpha, kappa)
            viol_resid[t] = info["viol_after"]
        u_seq[t] = u
        u_des_seq[t] = u_des
        # 精确欧拉一步（线性被控对象：无离散化误差，模块 docstring 注）。
        f, g = affine_fields(x)
        x = x + dt * (f + g @ u)
        traj[t + 1] = x
    # 全程屏障值：对每时刻状态重算各障碍 h 取最小（评测口径，不进控制）。
    h_all = np.empty((n_steps + 1, len(obstacle_list)))
    for i, (center, radius) in enumerate(obstacle_list):
        for t in range(n_steps + 1):
            h_all[t, i] = barrier_affine(traj[t], center, radius, kappa)[0]
    return CBFSimResult(
        traj=traj,
        u_applied=u_seq,
        min_h=float(h_all.min()),
        min_dist=min_obstacle_distance(traj, obstacle_list),
        u_des_seq=u_des_seq,
        viol_resid=viol_resid,
    )

"""对极几何与特征点法前端核心（教程第 05 章：视觉里程计 I：特征点法）.

函数流水线: 本模块是特征点法单目前端的*几何核心*，处于"特征提取与匹配之后、
后端优化之前"。输入为两帧匹配像素 ``(u1, u2)`` 与内参 ``K``（PnP 支线为
3D-2D 对应 ``(X, u)``），输出本质/基本矩阵 ``(E, F)``、唯一相对位姿
``(R, t)``、三角化场景点与相机位姿 ``T_cw``。调用链:
``eight_point``（归一化八点法，内部用 :func:`_normalize_points_hartley`）
-> ``decompose_E``（4 候选分解 + 手性检验，内部调用 :func:`triangulate`）
-> ``triangulate``（DLT）；PnP 支线: ``pnp_dlt``（线性初值）->
``pnp_refine``（流形 Gauss-Newton，复用 ``core.solver.gauss_newton`` 与解析
雅可比 :func:`reprojection_jacobian`）。依赖: ``core.lie``（``hat`` /
``se3_exp`` / ``so3_log`` / ``transform_points``）、``core.solver``
（``gauss_newton``）、``core.camera``（``make_intrinsics``）；本模块的
:func:`triangulate` 被 ``vins`` 复用做初始化路标三角化。

特征点法单目前端几何核心的教学实现。PTAM (Klein & Murray, ISMAR 2007,
papers/slam/classics/PTAM_ISMAR2007_KleinMurray.pdf) 固定了"前端跟踪 / 后端
建图"双线程结构; ORB-SLAM (Mur-Artal, Martinez Montiel & Tardos, IEEE T-RO
2015, papers/slam/classics/arXiv-1502.00956_ORB-SLAM.pdf) 的单目初始化与回
环几何校验（第 09 章 (9.5)）都建立在本模块实现的机制上: 由匹配点对估计本质
/基本矩阵（归一化八点法）、分解出唯一相对位姿（手性检验）、DLT 三角化与
PnP（流形 Gauss-Newton 精化复用 ``core.solver.gauss_newton``）.

式号约定 (5.x)
--------------
指向 ``tutorials/slam/05_视觉里程计-i特征点法.md``（该章撰写中，定稿后如有
调整以定稿为准）。其中 (5.1) 对极约束与 (5.2) 八点法线性方程组已被第 09 章
回引，编号确定；其余编号按本章草案：

- (5.1)  对极约束 ``x2^T E x1 = 0``（归一化坐标）
- (5.2)  八点法线性方程组 ``A e = 0``
- (5.3)  本质矩阵构造 ``E = hat(t) R``
- (5.4)  基本矩阵 ``F = K^{-T} E K^{-1}``
- (5.5)  Hartley 各向同性归一化
- (5.6)  本质流形投影（奇异值约束 diag(sigma, sigma, 0)）
- (5.7)  本质矩阵 4 候选分解
- (5.8)  手性检验（cheirality，深度符号）
- (5.9)  DLT 三角化
- (5.10) DLT 求解 PnP（位姿初值）
- (5.11) 重投影残差与解析雅可比（PnP 流形精化）

记号约定
-----------
- ``T_cw``：world-to-camera 齐次变换（教程第 02 章记号），
  ``x_cam = R @ x_world + t``。
- ``u1`` 为第一帧（参考帧）像素，``u2`` 为第二帧像素；估计的运动是相对变换
  ``x2 = R x1 + t``（frame-1 -> frame-2，即 ``T_21``）。
- ``E`` 作用在归一化相机坐标（``K^{-1} [u, v, 1]^T``）上，``F`` 作用在像素
  上（式 5.1 与 5.4 的定义域）；单目尺度不可观，``t`` 只有方向有意义。
"""
import numpy as np

from core.camera import make_intrinsics
from core.lie import hat, se3_exp, so3_log, transform_points
from core.solver import gauss_newton

__all__ = [
    "essential_from_rt",
    "fundamental_from_E",
    "eight_point",
    "decompose_E",
    "triangulate",
    "pnp_dlt",
    "pnp_refine",
    "reprojection_jacobian",
]

#: 90 度旋转 (Hartley & Zissman 本质分解的标准构造矩阵, 式 5.7)。
_W_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _as_pixels(u: np.ndarray) -> np.ndarray:
    """把 (N, 2) 像素数组（或任意可重排成两列的形状）转成 float (N, 2)."""
    return np.asarray(u, dtype=float).reshape(-1, 2)


def _to_normalized(u: np.ndarray, K: np.ndarray) -> np.ndarray:
    """像素 (N, 2) -> 归一化相机坐标 (N, 2).

    ``x = K^{-1} [u, v, 1]^T`` 的前两维（第 04 章 (4.5) 的反投影方向，深度
    仍未知——这正是三角化要补的信息）。
    """
    K = np.asarray(K, dtype=float).reshape(3, 3)
    pix = _as_pixels(u)
    hom = np.column_stack([pix, np.ones(len(pix))])
    return (np.linalg.inv(K) @ hom.T).T[:, :2]


def _normalize_points_hartley(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley 各向同性归一化 (式 5.5)。

    平移到质心、再缩放到 RMS 距离 = sqrt(2)。依据: Hartley & Zissman,
    *Multiple View Geometry in Computer Vision* (2nd ed.) 第 11 章归一化
    八点法——像素量级 (数百) 的坐标使 (5.2) 数据矩阵的奇异值相差多个数量
    级、最小奇异向量被数值噪声淹没; 各向同性归一化把条件数压回 O(1)，是
    八点法数值稳定的前提。原文建议缩放到平均距离 sqrt(2)，此处取同量级的
    RMS 距离 = sqrt(2)（二者均把尺度固定在 O(1)，估计出的 E 随后经 (5.6)
    重投影，残余尺度差异被吸收）。

    Args:
        u: (N, 2) 像素坐标。

    Returns:
        ``(u_norm, T)``: 归一化坐标 (N, 2) 与齐次变换 ``T (3, 3)``，
        ``[u_norm; 1] = T @ [u; 1]``。
    """
    u = _as_pixels(u)
    centroid = u.mean(axis=0)
    centered = u - centroid
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    scale = np.sqrt(2.0) / max(rms, 1e-12)
    normalized = scale * centered
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return normalized, T


def essential_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """相对位姿 -> 本质矩阵 ``E = hat(t) @ R`` (式 5.3, 用 core.lie.hat).

    依据: 对 ``x2 = R x1 + t``，``E x1 = hat(t) R x1 = t x (x2 - t) = t x x2``
    与 ``x2`` 垂直，故对极约束 (式 5.1) ``x2^T E x1 = 0`` 对真实对应成立；
    误匹配不共面（不同 3D 点），约束失效——这正是回环几何校验（第 09 章
    (9.5)）与误匹配剔除的判据。

    Args:
        R: (3, 3) 相对旋转（frame-1 -> frame-2）。
        t: (3,) 相对平移（frame-1 光心在 frame-2 坐标系下的位置）。

    Returns:
        (3, 3) 本质矩阵（作用在归一化坐标上）。
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    return hat(t) @ R


def fundamental_from_E(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """本质矩阵 -> 基本矩阵 ``F = K^{-T} E K^{-1}`` (式 5.4).

    依据: 归一化坐标与像素由 ``x = K^{-1} [u, v, 1]^T`` 相联（第 04 章），
    代入 (5.1) 得 ``u2^T (K^{-T} E K^{-1}) u1 = 0``。

    Args:
        E: (3, 3) 本质矩阵。
        K: (3, 3) 内参矩阵。

    Returns:
        (3, 3) 基本矩阵（作用在像素上）。
    """
    E = np.asarray(E, dtype=float).reshape(3, 3)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    K_inv = np.linalg.inv(K)
    return K_inv.T @ E @ K_inv


def eight_point(
    u1: np.ndarray, u2: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """归一化八点法: 由 >= 8 对匹配估计 (E, F) (式 5.2, 5.5, 5.6).

    流程与依据:

    1. Hartley 归一化 (式 5.5，见 :func:`_normalize_points_hartley`)：作用在
       *像素*上——像素量级 (数百) 的坐标使 (5.2) 数据矩阵的奇异值相差多个
       数量级、最小奇异向量被数值噪声淹没，各向同性归一化把条件数压回
       O(1)（Hartley & Zissman 第 11 章归一化八点法）；
    2. 组装 ``A e = 0`` (式 5.2): 每对点一行
       ``[a*c, a*d, a, b*c, b*d, b, c, d, 1]``，其中 ``(a, b) = n2``、
       ``(c, d) = n1``——由 ``n2^T F_n n1 = sum_ij F_ij n2_i n1_j = 0`` 按
       ``e = vec(F_n)``（行优先）展开；最小二乘解
       ``min ||Ae||^2 s.t. ||e|| = 1`` 由瑞利商理论取 ``A`` 的最小奇异向量
       （Eckart-Young 定理; Hartley & Zissman Alg. 11.1）；
    3. 反归一化到像素 ``F = T2^T F_n T1``，再经内参升维 ``E = K^T F K``
       （式 5.4 的逆变换）——归一化必须先"原路返回"再升维: 若把 Hartley
       仿射留在归一化坐标上直接投影本质流形，得到的会是 ``A2 E A1^T``
       （一般不再是 ``hat(t) R`` 形式）的最近投影，真解被破坏；
    4. SVD 投影回本质流形 (式 5.6): ``E = hat(t) R`` 的奇异值必为
       ``(sigma, sigma, 0)``——``hat(t)`` 的奇异值为 ``(|t|, |t|, 0)`` 且旋
       转不改变奇异值；取 ``U diag(1, 1, 0) V^T`` 即 Frobenius 范数下最近的
       本质矩阵（Eckart-Young；整体尺度不可观，故取 1）；
    5. ``F = K^{-T} E K^{-1}`` (式 5.4) 与投影后的 ``E`` 严格一致。

    Args:
        u1: (N, 2) 第一帧像素（N >= 8）。
        u2: (N, 2) 第二帧像素，与 ``u1`` 一一对应。
        K: (3, 3) 内参矩阵（两帧相同）。

    Returns:
        ``(E, F)``: 本质矩阵（归一化坐标）与基本矩阵（像素）。

    Raises:
        ValueError: 匹配数不足 8 或两数组形状不一致。
    """
    u1 = _as_pixels(u1)
    u2 = _as_pixels(u2)
    if u1.shape != u2.shape:
        raise ValueError("u1 and u2 must have the same shape")
    if len(u1) < 8:
        raise ValueError("eight_point needs at least 8 point pairs")
    K = np.asarray(K, dtype=float).reshape(3, 3)

    n1, T1h = _normalize_points_hartley(u1)
    n2, T2h = _normalize_points_hartley(u2)
    a, b = n2[:, 0], n2[:, 1]
    c, d = n1[:, 0], n1[:, 1]
    A = np.stack(
        [a * c, a * d, a, b * c, b * d, b, c, d, np.ones(len(u1))], axis=1
    )
    _, _, vt = np.linalg.svd(A)
    F_n = vt[-1].reshape(3, 3)

    F_raw = T2h.T @ F_n @ T1h     # 反归一化回像素空间
    E = K.T @ F_raw @ K           # 升维到归一化相机坐标（式 5.4 之逆）

    # (5.6): 投影回本质流形 diag(sigma, sigma, 0) —— 见 docstring 第 4 步。
    U, _, Vt = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    F = fundamental_from_E(E, K)
    return E, F


def decompose_E(
    E: np.ndarray, u1: np.ndarray, u2: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """本质矩阵分解 + 手性检验, 返回唯一相对位姿 (R, t) (式 5.7, 5.8).

    标准构造 (式 5.7): ``E = U diag(s) V^T``，取
    ``R1 = U W V^T``、``R2 = U W^T V^T``（``W`` 为绕 z 轴 90 度旋转）与
    ``t = +/- U[:, 2]``，共 4 个候选 (Longuet-Higgins 1981; Hartley &
    Zissman 第 12 章)。手性检验 (式 5.8): 用候选 (R, t) 构造 ``T_21`` 对全
    部匹配做三角化，统计在两帧中深度均为正的点数——物理上真实的运动使所有
    场景点位于两个相机前方，错误候选会把部分点投到相机背后，故正深度计数最
    大的候选即唯一解（cheirality 依据: Longuet-Higgins 1981; ORB-SLAM 单目
    初始化同样以该检验从 4 解中择一）。E 的整体符号被 SVD 吸收进 U/V，4 个
    候选集合不变。

    Args:
        E: (3, 3) 本质矩阵（归一化坐标）。
        u1: (N, 2) 第一帧像素。
        u2: (N, 2) 第二帧像素。
        K: (3, 3) 内参矩阵。

    Returns:
        ``(R, t)``: (3, 3) 旋转与 (3,) 平移，满足 ``x2 = R x1 + t``；
        t 的尺度不定（单目），只有方向有意义。
    """
    E = np.asarray(E, dtype=float).reshape(3, 3)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    u1 = _as_pixels(u1)
    u2 = _as_pixels(u2)

    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0.0:  # 保证 det(R) = +1 可达（式 5.7 的符号约定）
        U = -U
    if np.linalg.det(Vt) < 0.0:
        Vt = -Vt
    R1 = U @ _W_90 @ Vt
    R2 = U @ _W_90.T @ Vt
    t = U[:, 2]

    n1 = _to_normalized(u1, K)
    n2 = _to_normalized(u2, K)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_in_front = -1
    for R_c, t_c in ((R1, t), (R1, -t), (R2, t), (R2, -t)):
        T2 = np.eye(4)
        T2[:3, :3] = R_c
        T2[:3, 3] = t_c
        X1 = triangulate(np.eye(4), T2, n1, n2)  # 相机 1 坐标系下的场景点
        z1 = X1[:, 2]
        z2 = transform_points(T2, X1)[:, 2]
        in_front = int(np.count_nonzero((z1 > 0.0) & (z2 > 0.0)))
        if in_front > best_in_front:
            best_in_front = in_front
            best = (R_c, t_c)
    assert best is not None  # 循环必然给 best 赋值
    return best


def _projection_matrix(T: np.ndarray, K: np.ndarray | None) -> np.ndarray:
    """(3, 4) 投影矩阵: ``P = K @ T[:3, :4]``（K 为 None 时用单位内参）。"""
    T = np.asarray(T, dtype=float)
    T34 = T[:3, :4] if T.shape == (4, 4) else T.reshape(3, 4)
    if K is None:
        return T34
    return np.asarray(K, dtype=float).reshape(3, 3) @ T34


def triangulate(
    T1: np.ndarray,
    T2: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    K: np.ndarray | None = None,
) -> np.ndarray:
    """DLT 三角化 (式 5.9): 两帧观测 -> 场景点世界坐标.

    每个点的两帧观测给出 4 条齐次线性方程（由 ``x ~ P X`` 的叉积展开，每
    帧取两个独立分量）::

        row_u:  u * P[2] - P[0]
        row_v:  v * P[2] - P[1]

    逐点堆叠成该点的 ``A_k X~ = 0``（4 x 4），解取 ``A_k`` 的最小奇异向量
    （齐次最小二乘，Eckart-Young；Hartley & Zissman 三角化一章），再除以第
    四分量去齐次化；齐次向量整体变号等价，取 ``w > 0`` 与"深度为正"的约定
    一致。N 个点相互独立，按批 (N, 4, 4) 一次 SVD 求解。

    Args:
        T1, T2: 两相机的 world-to-camera 变换 (4, 4) 或 (3, 4)。
        u1, u2: (N, 2) 观测；``K`` 给定时为像素，否则为归一化坐标。
        K: (3, 3) 内参矩阵；``None`` 表示 ``u`` 已是归一化坐标。

    Returns:
        (N, 3) 场景点在世界系（``T1``/``T2`` 的参考系）下的坐标。
    """
    P1 = _projection_matrix(T1, K)
    P2 = _projection_matrix(T2, K)
    u1 = _as_pixels(u1)
    u2 = _as_pixels(u2)
    if u1.shape != u2.shape:
        raise ValueError("u1 and u2 must have the same shape")

    A = np.empty((len(u1), 4, 4))
    for row, (P, u) in enumerate(((P1, u1), (P2, u2))):
        base = 2 * row
        A[:, base + 0, :] = u[:, 0][:, None] * P[2] - P[0]
        A[:, base + 1, :] = u[:, 1][:, None] * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    X_h = vt[:, -1, :]                      # (N, 4) 最小奇异向量
    X_h = np.where(X_h[:, 3:4] < 0.0, -X_h, X_h)
    return X_h[:, :3] / X_h[:, 3:]


def pnp_dlt(X: np.ndarray, u: np.ndarray, K: np.ndarray) -> np.ndarray:
    """DLT 求解 PnP, 返回位姿初值 T_cw (式 5.10).

    线性化 ``P = K [R | t]``：每个 3D-2D 对给出两条方程（由 ``u ~ P X``
    展开，12 个齐次未知数 ``vec(P)``）::

        row_u: [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        row_v: [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    解 ``A p = 0`` 取最小奇异向量（齐次最小二乘）。P 只定到非零尺度
    ``P ~ s K [R | t]``：由 ``det(P[:, :3]) = s^3 det(K)``（det(R) = +1）的
    符号先固定 ``s > 0``（必要时整体变号），再由
    ``K^{-1} P = s [R | t]`` 的旋转块 SVD 正交化得到 ``R``，尺度
    ``s = trace(R^T K^{-1} P[:, :3]) / 3``，``t = (K^{-1} P)[:, 3] / s``。
    结果作为 :func:`pnp_refine` 的线性化初值（线性方法对噪声敏感，闭环还
    需非线性精化）。

    Args:
        X: (N, 3) 世界系 3D 点（N >= 6，非退化、不共面过相机光心）。
        u: (N, 2) 对应像素。
        K: (3, 3) 内参矩阵。

    Returns:
        (4, 4) 位姿初值 ``T_cw``。

    Raises:
        ValueError: 点数不足 6 或 X/u 长度不一致。
    """
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    u = _as_pixels(u)
    if len(X) != len(u):
        raise ValueError("X and u must have the same length")
    if len(X) < 6:
        raise ValueError("pnp_dlt needs at least 6 points")
    K = np.asarray(K, dtype=float).reshape(3, 3)

    n = len(X)
    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = np.column_stack([X, np.ones(n)])
    A[1::2, 4:8] = np.column_stack([X, np.ones(n)])
    A[0::2, 8:11] = -u[:, 0][:, None] * X
    A[0::2, 11] = -u[:, 0]
    A[1::2, 8:11] = -u[:, 1][:, None] * X
    A[1::2, 11] = -u[:, 1]
    _, _, vt = np.linalg.svd(A)
    P = vt[-1].reshape(3, 4)
    if np.linalg.det(P[:, :3]) < 0.0:  # 固定尺度符号 s > 0（见 docstring）
        P = -P

    M = np.linalg.inv(K) @ P  # = s [R | t], s > 0
    U, _, Vt = np.linalg.svd(M[:, :3])
    det_fix = np.linalg.det(U @ Vt)
    R = U @ np.diag([1.0, 1.0, det_fix]) @ Vt  # 投影回 SO(3)
    scale = float(np.trace(R.T @ M[:, :3])) / 3.0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = M[:, 3] / scale
    return T


def reprojection_jacobian(
    points_cam: np.ndarray, fx: float, fy: float
) -> np.ndarray:
    """重投影残差的解析雅可比, 每点 (2, 6), 堆叠为 (2N, 6) (式 5.11).

    残差取 ``e = u - proj(T X)``（第 08 章 (8.1) 重投影误差，观测 - 预测），
    位姿左扰动 ``T <- exp(delta_xi^) T``（``xi = (tau, phi)``, 平移在前，
    与 :func:`core.lie.se3_exp` 一致）。扰动后的相机系点为

        ``X' -> X' + delta_tau + delta_phi x X' = X' + delta_tau - hat(X') delta_phi``

    （依据: ``exp(delta_xi^) X' = X' + hat(delta_phi) X' + delta_tau`` 取一
    阶），于是 ``d(proj)/d delta_xi = d(proj)/dX' @ [I, -hat(X')]``，代回
    ``e = u - proj`` 得每点

        ``d e / d delta_xi = [ -d(proj)/dX' | d(proj)/dX' @ hat(X') ]``

    即::

        [[-fx/Z', 0,  fx X'/Z'^2,  fx X'Y'/Z'^2, -fx(1+X'^2/Z'^2),  fx Y'/Z'],
         [0, -fy/Z',  fy Y'/Z'^2,  fy(1+Y'^2/Z'^2), -fy X'Y'/Z'^2, -fy X'/Z']]

    其中 ``[X', Y', Z']`` 为变换后相机系坐标。注意: 常见资料中平移块取
    ``+fx/Z'`` 的版本对应残差定义为"预测 - 观测"——两块符号必须随残差定义
    同时翻转，混用会使 GN 在平移与旋转上互相打架而发散（实现与测试以有限
    差分校验过）。

    Args:
        points_cam: (N, 3) 当前位姿下的相机系点（Z > 0）。
        fx, fy: 焦距（像素）。

    Returns:
        (2N, 6) 雅可比，行序与 ``e = (u_0, v_0, u_1, v_1, ...)`` 一致。
    """
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)  # 防退化（正常观测 Z > 0）
    z2 = z * z
    zeros = np.zeros_like(x)

    du_dtau = np.stack([fx / z, zeros, -fx * x / z2], axis=1)
    dv_dtau = np.stack([zeros, fy / z, -fy * y / z2], axis=1)
    du_dphi = np.stack(
        [fx * x * y / z2, -fx * (1.0 + x * x / z2), fx * y / z], axis=1
    )
    dv_dphi = np.stack(
        [fy * (1.0 + y * y / z2), -fy * x * y / z2, -fy * x / z], axis=1
    )

    J = np.empty((2 * len(pts), 6))
    J[0::2, 0:3] = -du_dtau
    J[0::2, 3:6] = du_dphi
    J[1::2, 0:3] = -dv_dtau
    J[1::2, 3:6] = dv_dphi
    return J


def pnp_refine(
    X: np.ndarray,
    u: np.ndarray,
    K: np.ndarray,
    T0: np.ndarray,
    n_outer: int = 8,
    tol: float = 1e-12,
) -> np.ndarray:
    """PnP 流形 Gauss-Newton 精化 (式 5.11; 第 08 章 (8.3)/(8.5)/(8.7)).

    GN 作用于 6 维扰动向量（复用 :func:`core.solver.gauss_newton` 的正规方
    程 (8.5)）：每轮外层迭代把当前位姿固定为线性化点 ``T0``（第 08 章
    (8.3) 的流形扰动），残差与雅可比按 :func:`reprojection_jacobian` 展开；
    解出 ``delta_xi`` 后按流形更新 ``T <- exp(delta_xi^) T``（第 08 章
    (8.7)，用 :func:`core.lie.se3_exp`）并重新线性化，直至扰动范数或代价
    下降低于 ``tol``。线性化点固定 + 流形更新保证了旋转矩阵的正交性不被破
    坏（第 02 章: SO(3)/SE(3) 不是向量空间，直接加增量会离开流形）。

    Args:
        X: (N, 3) 世界系 3D 点。
        u: (N, 2) 对应像素。
        K: (3, 3) 内参矩阵。
        T0: (4, 4) 位姿初值（通常来自 :func:`pnp_dlt`）。
        n_outer: 外层重线性化的最大轮数。
        tol: 收敛阈值（扰动范数 / 代价下降）。

    Returns:
        (4, 4) 精化后的位姿 ``T_cw``。
    """
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    u = _as_pixels(u)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    cam = make_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    T = np.asarray(T0, dtype=float).reshape(4, 4).copy()

    prev_cost = np.inf
    for _ in range(n_outer):
        sol = gauss_newton(
            residual_fn=lambda xi, T_lin=T: (
                u - cam.project(transform_points(se3_exp(xi) @ T_lin, X))
            ).reshape(-1),
            jacobian_fn=lambda xi, T_lin=T: reprojection_jacobian(
                transform_points(se3_exp(xi) @ T_lin, X), cam.fx, cam.fy
            ),
            x0=np.zeros(6),
            n_iters=50,
            tol=tol,
        )
        T = se3_exp(sol.x) @ T
        if np.linalg.norm(sol.x) < tol or prev_cost - sol.cost < tol:
            break
        prev_cost = sol.cost
    return T

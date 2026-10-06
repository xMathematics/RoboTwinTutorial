"""PTAM（Klein & Murray, ISMAR 2007）的教学代理：跟踪/建图两阶段 + 关键帧局部 BA.

Teaching proxy of Klein & Murray, ``Parallel Tracking and Mapping for Small
AR Workspaces``, ISMAR 2007
(papers/slam/classics/PTAM_ISMAR2007_KleinMurray.pdf)。PTAM 首次把"逐帧跟踪
相机位姿"（tracking 线程，论文 §5）与"关键帧地图 + 光束法平差"（mapping
线程，论文 §6）拆成两个并行线程，是现代视觉 SLAM 架构（ORB-SLAM 三线程，
教程第 10 章 10.1）的直接源头。

**2D-3D 教学简化（显式声明）**：真实系统在图像金字塔上做 8x8 patch 的 SSD
匹配；本模块的世界是"稀疏 3D 路标 + 针孔相机"的 2D-3D 简化世界——路标-观测
的数据关联由仿真器按路标索引直接给出（patch 匹配的教学替身），测量为带
高斯噪声的像素坐标。论文的核心机制全部保留，逐条对应见下。

式号对照（论文式号 + 教程式号）
--------------------------------
- §5 / Eq.(8)-(9)  跟踪线程的 motion-only 位姿更新（论文以 Tukey M-估计
  加权）：:func:`track_frame`——**地图固定、只优化当前位姿**，残差为重投影
  误差（教程 (8.1)），G-N 在 SE(3) 左扰动坐标上迭代（教程 (8.9) 流形更新、
  (8.10) 雅可比），鲁棒核以 Huber IRLS 等价替代 Tukey；正规方程复用
  :func:`core.solver.gauss_newton`，雅可比复用
  :func:`epipolar.reprojection_jacobian`。
- §6.1  初始化（论文：五点立体 + RANSAC 恢复相对位姿并三角化出基图）：
  教学版复用 :mod:`epipolar` 的归一化八点法（教程 (5.1)-(5.2)）+ 4 候选
  分解手性检验（教程 (5.7)）+ DLT 三角化（教程 (5.9)）——见
  :meth:`Ptam._try_initialize`。单目尺度不可观：论文用主平面规范化 +
  "初始化平移 = 10 cm"约定折算米制，教学版由 E 分解的单位平移定标，
  尺度差交由评估的 Sim(3) 对齐吸收（教程 §10.3 的规范自由度）。
- §6.2  关键帧插入判据（论文：距上一关键帧 > 20 帧且相机离最近关键帧的
  距离超过随所观测特征平均深度伸缩的阈值）：:meth:`Ptam.process_frame`
  取 ``frames_since_kf >= key_min_frames`` 且（平移距离 >
  ``key_min_dist_frac * 跟踪点中位深度`` 或 平均视差 >=
  ``key_min_parallax_deg``）。新点初始化（论文：极线搜索 + 三角化）简化为
  "与最近关键帧的公共未建图观测直接 DLT 三角化"——位姿已知时极线搜索只是
  把二维匹配降为一维，教学版关联已知，故省去搜索只留三角化。
- §6.3 / Eq.(10)-(11)  建图线程的 BA：论文 Eq.(10) 的全 BA 目标即教程
  (8.2) 的 BA 标准型（第一关键帧固定 datum 消去整体 gauge）；地图变大后
  论文默认改做 Eq.(11) 的**局部 BA**——可调关键帧集 X（最新 + 最近的
  4 个）、固定集 Y、可见点集 Z，且 Z 中点用其**全部**观测（含 Y 的旧
  观测作常量约束）。:func:`local_ba` 教学版按同一三分法实现（X = 最近
  ``local_ba_k`` 个关键帧、Y = 观测过 Z 的其余关键帧——位姿冻结、只贡献
  点块残差，把窗口"钉"在既有地图上抑制尺度/形状漂移；Z 的截断与求解是
  仅有的教学差异，见下）： 论文用稀疏 LM + Schur 补求解，教学版把
  [位姿增量 | 点增量] 堆成一个稠密矢量交给 :func:`core.solver.gauss_newton`
  联合求解（同一目标函数 (8.2)/(8.5)；点数封顶 ``max_ba_points``、窗口小
  时稠密解代价可忽略，Schur 稀疏版见 ``droidlite`` 的 ``_schur_step``）；
   规范锚定取窗口内最老关键帧位姿固定（窗口含第 0 关键帧时即论文的
  datum 约定）。
- §4  地图表示（M 个世界系点 + N 个关键帧 SE(3) 位姿）：:class:`Ptam` 的
  ``points`` (M, 3) 与 ``keyframes``（各存 T_cw 与逐点像素观测）。

双线程与交替阶段（重要声明）
----------------------------
论文：跟踪线程逐帧（~30 Hz）**从不等待**建图线程；建图线程在共享地图上
异步运行（收关键帧 -> 更新数据关联 -> 局部 BA -> 整合新关键帧/新点，§6.4）。
教学代理把两个线程压成**交替阶段**：跟踪（每帧）-> 触发关键帧判据时插入
建图阶段（插关键帧 -> 三角化新点 -> 局部 BA）。异步并行的**接口**语义
保留：局部 BA 只写地图（关键帧位姿与路标），跟踪永远以"当前地图"为参照、
以上一帧位姿为初值（论文用匀速运动模型预测，教学版简化为零运动——帧间
运动远小于跟踪收敛域时二者等价）。

Conventions
-----------
- ``T_cw``：world-to-camera（教程第 02 章记号），``x_cam = R x_world + t``；
  相机光心的世界系坐标为 ``-R^T t``（:func:`camera_center`）。
- 路标 ``X_w`` (M, 3) 世界系坐标（单位 m）；像素 (V, 2)（单位 px）。
- 世界系 = 第 0 个关键帧的相机系（其 ``T_cw = I`` 永不优化，即论文的
  "第一关键帧 datum"，同时消去 BA 的整体规范自由度）。
- 未建图路标在 ``Ptam.points`` 中以 NaN 行占位（行号 = 路标全局 id，
  ``point_status`` 给出已建图掩码）——一个路标在**第二个**观测到它的
  关键帧入图时才获得深度（论文 §6.2：单帧定不了深度）。
"""
from dataclasses import dataclass

import numpy as np

from core.camera import PinholeCamera, make_intrinsics
from core.lie import se3_exp, transform_points
from core.solver import gauss_newton, huber_weights
from epipolar import decompose_E, eight_point, reprojection_jacobian, triangulate

__all__ = [
    "BAResult",
    "DEFAULT_XI",
    "FrameLog",
    "KeyFrame",
    "PlanarScene",
    "Ptam",
    "PtamResult",
    "TrackResult",
    "ba_terms",
    "camera_center",
    "local_ba",
    "mean_parallax",
    "project_visible",
    "render_planar_frames",
    "run_ptam",
    "simulate_planar_scene",
    "track_frame",
]

#: 默认 GT 运动（6 维 xi = (tau, phi)，平移在前，见 core.lie.se3_exp）：
#: 每帧前向 6 cm + 横向 4.5 cm + 轻微旋转——平移视差是三角化与 BA 的
#: 可观测性来源（纯旋转不产生深度信息）。
DEFAULT_XI = np.array([0.06, 0.045, 0.0, 0.002, 0.005, -0.002])


# --------------------------------------------------------------------- #
# 基础几何工具                                                            #
# --------------------------------------------------------------------- #
def camera_center(T_cw: np.ndarray) -> np.ndarray:
    """(4, 4) world-to-camera 位姿 -> 相机光心的世界系坐标 (3,)。

    依据：光心是相机系原点，``X_world = R^T (0 - t) = -R^T t``（教程第 02 章
    位姿变换的逆方向）。
    """
    T_cw = np.asarray(T_cw, dtype=float).reshape(4, 4)
    return -T_cw[:3, :3].T @ T_cw[:3, 3]


def project_visible(
    cam: PinholeCamera,
    T_cw: np.ndarray,
    points_w: np.ndarray,
    width: int,
    height: int,
    margin: float = 8.0,
    min_depth: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把世界点投到相机并做可见性筛选，返回 ``(uv, ids, depth)``。

    只保留深度 ``Z > min_depth`` 且落在图像内（留 ``margin`` 边距）的点——
    对应真实系统"投影预测 + 视野检查"的候选路标筛选（论文 §5 跟踪前的
    特征投影步骤）。

    Returns:
        ``(uv (V, 2), ids (V,), depth (V,))``：可见点的像素、全局路标 id 与
        相机系深度。
    """
    q = transform_points(T_cw, points_w)
    z = q[:, 2]
    uv = np.full((len(q), 2), np.nan)
    fwd = z > min_depth
    if fwd.any():
        uv[fwd] = cam.project(q[fwd])
    inside = (
        fwd
        & (uv[:, 0] > margin) & (uv[:, 0] < width - margin)
        & (uv[:, 1] > margin) & (uv[:, 1] < height - margin)
    )
    ids = np.nonzero(inside)[0]
    return uv[ids], ids, z[ids]


def mean_parallax(cam: PinholeCamera, uv1: np.ndarray, uv2: np.ndarray) -> float:
    """同一批路标在两帧中的平均视差（度）。

    视差 = 两帧归一化射线的平均夹角：``x = K^{-1}[u, v, 1]`` 归一化后取
    ``mean(arccos(x1 . x2))``（教程第 04/05 章的归一化坐标；视差是三角化
    可观测性的直接来源，也是关键帧判据的物理量，论文 §6.2）。
    """
    uv1 = np.asarray(uv1, dtype=float).reshape(-1, 2)
    uv2 = np.asarray(uv2, dtype=float).reshape(-1, 2)

    def rays(uv: np.ndarray) -> np.ndarray:
        x = (uv[:, 0] - cam.cx) / cam.fx
        y = (uv[:, 1] - cam.cy) / cam.fy
        n = np.sqrt(x * x + y * y + 1.0)
        return np.stack([x / n, y / n, 1.0 / n], axis=1)

    r1, r2 = rays(uv1), rays(uv2)
    cos = np.clip(np.sum(r1 * r2, axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def _project_safe(cam: PinholeCamera, q: np.ndarray) -> np.ndarray:
    """带深度保护的针孔投影。

    ``Z <= 0`` 的点钳到小正深度：投影值变得很大 => 残差很大，交由代价
    接受判据与 Huber 核处理，而不让 :meth:`PinholeCamera.project` 的
    ``Z > 0`` 检查在优化的试探步中抛异常（优化中间步允许越过深度边界）。
    """
    z = np.where(q[:, 2] < 1e-6, 1e-6, q[:, 2])
    return np.stack(
        [cam.fx * q[:, 0] / z + cam.cx, cam.fy * q[:, 1] / z + cam.cy], axis=1
    )


# --------------------------------------------------------------------- #
# 跟踪线程：motion-only 位姿更新（论文 §5，Eq.(8)-(9) 的教学替身）         #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrackResult:
    """:func:`track_frame` 的输出与诊断。

    Attributes:
        T_cw: (4, 4) 优化后的 world-to-camera 位姿。
        n_obs: 进入求解的 3D-2D 对应数。
        inlier_ratio: 重投影距离 < ``gate_px`` 的对应比例（跟踪质量指标，
            论文 §5.6 用"成功测量比例"驱动失败恢复）。
        reproj_rmse: 内点重投影 RMSE（px）。
        converged: G-N 收敛标志。
    """

    T_cw: np.ndarray
    n_obs: int
    inlier_ratio: float
    reproj_rmse: float
    converged: bool


def track_frame(
    cam: PinholeCamera,
    X_w: np.ndarray,
    uv: np.ndarray,
    T_init: np.ndarray,
    gate_px: float = 4.0,
    n_outer: int = 3,
    tol: float = 1e-10,
) -> TrackResult:
    """motion-only 跟踪：固定地图路标，G-N 精化当前帧位姿（论文 §5）。

    残差 ``e = u - proj(T X)``（教程 (8.1)，观测 - 预测），变量只有当前位姿
    的 6 维左扰动 ``xi``（教程 (8.9)-(8.10)）：每轮外层迭代在当前位姿处重新
    线性化、解阻尼正规方程（教程 (8.5)-(8.8)，经
    :func:`core.solver.gauss_newton`，Huber 核宽度取 ``gate_px``）后流形回缩
    ``T <- exp(xi^) T``——与论文 Eq.(8)-(9) 的"迭代位姿更新 + 鲁棒加权"
    同构（Tukey 以 Huber 等价替代）。

    Args:
        cam: 针孔内参。
        X_w: (N, 3) 参与跟踪的地图路标（世界系）。
        uv: (N, 2) 对应像素测量。
        T_init: (4, 4) 位姿初值（上一帧位姿，零运动预测）。
        gate_px: 内点门限（px），同时用作 Huber 核宽度。
        n_outer: 外层重线性化轮数上限。
        tol: 扰动范数 / 代价下降的收敛阈值。

    Returns:
        :class:`TrackResult`。
    """
    X_w = np.asarray(X_w, dtype=float).reshape(-1, 3)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    T = np.asarray(T_init, dtype=float).reshape(4, 4).copy()

    prev_cost = np.inf
    converged = False
    for _ in range(n_outer):
        sol = gauss_newton(
            residual_fn=lambda xi, T_lin=T: (
                uv - _project_safe(cam, transform_points(se3_exp(xi) @ T_lin, X_w))
            ).reshape(-1),
            jacobian_fn=lambda xi, T_lin=T: reprojection_jacobian(
                transform_points(se3_exp(xi) @ T_lin, X_w), cam.fx, cam.fy
            ),
            x0=np.zeros(6),
            n_iters=20,
            tol=tol,
            robust_delta=gate_px,
        )
        T = se3_exp(sol.x) @ T
        if np.linalg.norm(sol.x) < tol or prev_cost - sol.cost < tol:
            converged = True
            break
        prev_cost = sol.cost

    # 内点统计：重投影距离 < gate_px 的比例与内点 RMSE（论文 §5.6 跟踪质量）。
    q = transform_points(T, X_w)
    resid = uv - _project_safe(cam, q)
    rnorm = np.linalg.norm(resid, axis=1)
    inlier = rnorm < gate_px
    rmse = (
        float(np.sqrt(np.mean(rnorm[inlier] ** 2))) if inlier.any() else float("nan")
    )
    return TrackResult(
        T_cw=T,
        n_obs=len(X_w),
        inlier_ratio=float(inlier.mean()) if len(inlier) else 0.0,
        reproj_rmse=rmse,
        converged=bool(converged),
    )


# --------------------------------------------------------------------- #
# 建图线程：关键帧局部 BA（论文 §6.3，Eq.(10)-(11) 的教学替身）             #
# --------------------------------------------------------------------- #
@dataclass
class KeyFrame:
    """关键帧：地图位姿节点 + 该帧全部像素观测（论文 §4 地图表示）。

    **可变**数据类：局部 BA 会就地更新 ``T_cw``（建图线程对共享地图的写入）。

    Attributes:
        frame: 插入时的帧号。
        T_cw: (4, 4) world-to-camera 位姿（第 0 个关键帧恒为 I——论文的
            "第一关键帧 datum"，消去 BA 的整体规范自由度）。
        ids: (V,) 该帧观测的路标全局 id（跨帧一致；教学替身——patch 匹配
            给出的数据关联直接按 id 对上）。
        uv: (V, 2) 对应像素测量（px，含噪声）。
    """

    frame: int
    T_cw: np.ndarray
    ids: np.ndarray
    uv: np.ndarray


@dataclass(frozen=True)
class BAResult:
    """:func:`local_ba` 的输出诊断。

    Attributes:
        rmse_before / rmse_after: 窗口观测重投影 RMSE（px），BA 前 / 后。
        cost_after: BA 后的加权限鲁棒代价 sum(w e^2)（px^2，w 为 Huber 权）。
        n_outer: 实际执行的外层重线性化轮数。
        converged: 是否以微小增量提前收敛。
    """

    rmse_before: float
    rmse_after: float
    cost_after: float
    n_outer: int
    converged: bool


def _ba_build(
    cam: PinholeCamera,
    win_obs: list[tuple[np.ndarray, np.ndarray]],
    fixed_obs: list[tuple[np.ndarray, np.ndarray]],
    fixed_T: list[np.ndarray],
    z_arr: np.ndarray,
) -> tuple:
    """组装局部 BA 的观测组与闭式残差/雅可比工厂（内部共享件）。

    Args:
        win_obs: 窗口每个关键帧在 Z 内的观测 ``(ids, uv)``（已过滤）——
            位姿参与优化（第 0 帧为规范锚点除外）。
        fixed_obs: 非窗口关键帧对 Z 的观测（论文 Eq.(11) 的固定关键帧集
            Y）——位姿固定，只贡献点块的残差与雅可比：它把窗口内的点
            "钉"在既有地图上，抑制窗口相对冻结地图的尺度/形状漂移（论文
            原文："points in Z use **all** their measurements"）。
        fixed_T: 与 ``fixed_obs`` 一一对应的固定位姿（BA 期间为常量，
            构建时拷贝捕获）。
        z_arr: (P,) 参与优化的路标全局 id（升序，已建图）。

    Returns:
        ``(make_fns, dim, n_rows)``：``make_fns(T_lin, X_lin)`` 返回与
        :func:`core.solver.gauss_newton` 对接的 ``(residual_fn, jacobian_fn)``；
        ``T_lin`` 为 W 个线性化点位姿、``X_lin`` 为 (P, 3) 线性化点坐标。
        变量布局 ``x = [xi_1..xi_{W-1} (6(W-1)) | dX (3P)]``：窗口内第 0 个
        关键帧为规范锚点（位姿固定，对应论文的 datum），点在 R^3 上直接加。
    """
    n_free = len(win_obs) - 1
    n_pts = len(z_arr)
    z_col = {int(p): j for j, p in enumerate(z_arr)}
    t_fixed = [np.asarray(T, dtype=float).copy() for T in fixed_T]

    # 收集观测组：窗口组（位姿可动，帧序号 b）与固定组（位姿常量，取自
    # t_fixed）——组结构与线性化点在重线性化轮间不变。
    win_groups: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    fixed_groups: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    row = 0
    for b, (ids, uv) in enumerate(win_obs):
        cols = np.array([z_col[int(i)] for i in ids], dtype=int)
        win_groups.append((b, cols, uv, row, row + 2 * len(cols)))
        row += 2 * len(cols)
    for k, (ids, uv) in enumerate(fixed_obs):
        cols = np.array([z_col[int(i)] for i in ids], dtype=int)
        fixed_groups.append((k, cols, uv, row, row + 2 * len(cols)))
        row += 2 * len(cols)
    dim = 6 * n_free + 3 * n_pts

    def make_fns(T_lin: list, X_lin: np.ndarray) -> tuple:
        def residual_fn(x: np.ndarray) -> np.ndarray:
            xi = x[: 6 * n_free].reshape(n_free, 6)
            dx = x[6 * n_free:].reshape(n_pts, 3)
            # 依据：教程 (8.9)——位姿走左扰动流形回缩，点在 R^3 上直接加。
            poses = [T_lin[0]] + [
                se3_exp(xi[k]) @ T_lin[k + 1] for k in range(n_free)
            ]
            pts = X_lin + dx
            e = np.empty(row)
            for _, cols, uv, r0, r1 in win_groups:
                q = transform_points(poses[_], pts[cols])
                e[r0:r1] = (uv - _project_safe(cam, q)).reshape(-1)
            for k, cols, uv, r0, r1 in fixed_groups:
                q = transform_points(t_fixed[k], pts[cols])
                e[r0:r1] = (uv - _project_safe(cam, q)).reshape(-1)
            return e

        def jacobian_fn(x: np.ndarray) -> np.ndarray:
            xi = x[: 6 * n_free].reshape(n_free, 6)
            poses = [T_lin[0]] + [
                se3_exp(xi[k]) @ T_lin[k + 1] for k in range(n_free)
            ]
            pts = X_lin + x[6 * n_free:].reshape(n_pts, 3)
            J = np.zeros((row, dim))

            def fill_point_block(T: np.ndarray, q: np.ndarray, cols: np.ndarray,
                                 r0: int, r1: int) -> None:
                # 点块：de/dX = -(dpi/dq) @ R_cw（dq/dX = R 的链式因子；
                # dpi/dq 为投影导数，教程 (5.11)/(8.10) 的因子二）。
                z = np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])
                dpi = np.stack(
                    [
                        np.stack(
                            [cam.fx / z, np.zeros_like(z), -cam.fx * q[:, 0] / z**2],
                            axis=1,
                        ),
                        np.stack(
                            [np.zeros_like(z), cam.fy / z, -cam.fy * q[:, 1] / z**2],
                            axis=1,
                        ),
                    ],
                    axis=1,
                )                                                    # (n, 2, 3)
                j_pt = -np.einsum("nij,jk->nik", dpi, T[:3, :3])     # (n, 2, 3)
                # 残差行按 (u0, v0, u1, v1, ...) 排列，每行取同一路标的 3 列。
                row_idx = np.repeat(np.arange(r0, r1)[:, None], 3, axis=1)
                col_idx = np.repeat(
                    6 * n_free + 3 * cols[:, None] + np.arange(3)[None, :], 2, axis=0
                )
                J[row_idx, col_idx] = j_pt.reshape(-1, 3)

            for b, cols, _, r0, r1 in win_groups:
                q = transform_points(poses[b], pts[cols])
                # 位姿块（教程 (8.10) 左扰动雅可比；锚点帧 b=0 无位姿列）：
                if b > 0:
                    J[r0:r1, 6 * (b - 1): 6 * b] = reprojection_jacobian(
                        q, cam.fx, cam.fy
                    )
                fill_point_block(poses[b], q, cols, r0, r1)
            for k, cols, _, r0, r1 in fixed_groups:
                q = transform_points(t_fixed[k], pts[cols])
                fill_point_block(t_fixed[k], q, cols, r0, r1)
            return J

        return residual_fn, jacobian_fn

    return make_fns, dim, row


def ba_terms(
    cam: PinholeCamera,
    keyframes: list[KeyFrame],
    points: np.ndarray,
    window: list[int],
) -> tuple:
    """组装局部 BA 的堆叠残差 / 雅可比（教程 (8.2) 目标、(8.10) 雅可比）。

    调试与教学检查入口（:func:`local_ba` 内部同样经此路径的共享件）：在
    *当前*关键帧位姿与路标处展开 ``e(x) ≈ e(0) + J x``，供雅可比的有限
    差分交叉验证（tests/test_ptam.py）。可见点集 Z = 任一窗口关键帧观测过
    且已建图（坐标有限）的路标。

    Returns:
        ``(residual_fn, jacobian_fn, dim, n_rows)``；变量布局见
        :func:`_ba_build`。
    """
    window = list(window)
    win_set = set(window)
    z_seen: set[int] = set()
    for ki in window:
        z_seen.update(int(i) for i in keyframes[ki].ids)
    z_arr = np.array(sorted(z_seen), dtype=int)
    z_arr = z_arr[z_arr < len(points)]                        # 防越界（未建图占位）
    z_arr = z_arr[np.isfinite(points[z_arr]).all(axis=1)]     # 只留已建图路标
    z_set = set(z_arr.tolist())
    win_obs, fixed_obs, fixed_T = [], [], []
    for ki, kf in enumerate(keyframes):
        sel = np.array([int(i) in z_set for i in kf.ids], dtype=bool)
        if not sel.any():
            continue
        if ki in win_set:
            win_obs.append((kf.ids[sel], kf.uv[sel]))
        else:
            fixed_obs.append((kf.ids[sel], kf.uv[sel]))
            fixed_T.append(kf.T_cw)
    make_fns, dim, n_rows = _ba_build(cam, win_obs, fixed_obs, fixed_T, z_arr)
    t_lin = [np.asarray(keyframes[ki].T_cw, dtype=float).copy()
             for ki in window]
    x_lin = points[z_arr].copy()
    res_fn, jac_fn = make_fns(t_lin, x_lin)
    return res_fn, jac_fn, dim, n_rows


def local_ba(
    cam: PinholeCamera,
    keyframes: list[KeyFrame],
    points: np.ndarray,
    window: list[int],
    max_ba_points: int = 120,
    huber_delta: float = 2.0,
    lm_lambda: float = 1e-4,
    n_outer: int = 2,
    inner_iters: int = 6,
    tol: float = 1e-10,
) -> BAResult:
    """关键帧局部 BA（论文 §6.3 Eq.(11) 的教学替身，差异见模块 docstring）。

    对窗口内最近 ``len(window)`` 个关键帧的位姿与它们的可见路标联合最小化
    重投影误差（教程 (8.2)，Huber 鲁棒核）：稠密联合 G-N/LM（教程
    (8.5)-(8.8)，经 :func:`core.solver.gauss_newton`）+ 外层流形重线性化
    （``T <- exp(xi^) T``、``X <- X + dX``，教程 (8.9)）。**就地更新**传入的
    ``keyframes[ki].T_cw`` 与 ``points``——对应论文建图线程对共享地图的
    写入（跟踪线程随后以更新后的地图为参照）。可见点数超过
    ``max_ba_points`` 时按路标 id 升序截断（确定性）。

    Args:
        cam: 针孔内参。
        keyframes: 全部关键帧（窗口取其子集）。
        points: (M, 3) 全量路标（NaN 行 = 未建图）。
        window: 参与优化的关键帧下标（升序），第 0 个为规范锚点。
        max_ba_points: 点数上限（控制稠密解规模）。
        huber_delta: Huber 核宽度（px）。
        lm_lambda: LM 阻尼初值（教程 (8.8)）。
        n_outer: 外层重线性化轮数。
        inner_iters: 每轮 G-N 的内层迭代上限。
        tol: 收敛阈值。

    Returns:
        :class:`BAResult`。

    Raises:
        ValueError: 窗口内关键帧数 < 2。
    """
    window = list(window)
    if len(window) < 2:
        raise ValueError("local BA needs at least 2 keyframes in the window")

    # 可见点集 Z（已建图）+ 确定性截断；全部关键帧对 Z 的观测进入 BA
    # （窗口帧 = 论文的 X 集合，其余 = 固定集合 Y），观测按 Z 过滤成副本，
    # 不改关键帧自身的全部观测记录。
    z_seen: set[int] = set()
    for ki in window:
        z_seen.update(int(i) for i in keyframes[ki].ids)
    z_arr = np.array(sorted(z_seen), dtype=int)
    # 只留已建图路标（id 在界内且坐标有限；关键帧可观测到尚未建图的路标）。
    z_arr = z_arr[z_arr < len(points)]
    z_arr = z_arr[np.isfinite(points[z_arr]).all(axis=1)]
    if len(z_arr) > max_ba_points:
        z_arr = z_arr[:max_ba_points]                          # id 升序截断（确定性）
    z_set = set(z_arr.tolist())
    # 按关键帧收集 Z 观测（副本，不改关键帧自身的全部观测记录）：窗口帧
    # = 论文 Eq.(11) 的 X 集合，其余帧 = 固定集合 Y（位姿冻结、只贡献点块）。
    win_idx = [ki for ki in window
               if any(int(i) in z_set for i in keyframes[ki].ids)]
    if len(win_idx) < 2:
        # 极端退化（窗口内不足两个含 Z 观测的关键帧）：无可优化结构，原样返回。
        rmse = float("nan")
        if z_set:
            e_all = []
            for ki in win_idx:
                kf = keyframes[ki]
                sel = np.array([int(i) in z_set for i in kf.ids], dtype=bool)
                q = transform_points(kf.T_cw, points[kf.ids[sel].astype(int)])
                e_all.append((kf.uv[sel] - _project_safe(cam, q)).reshape(-1))
            e_cat = np.concatenate(e_all) if e_all else np.zeros(0)
            if len(e_cat):
                e2 = e_cat.reshape(-1, 2)
                rmse = float(np.sqrt(np.mean(np.sum(e2 * e2, axis=1))))
        return BAResult(rmse_before=rmse, rmse_after=rmse, cost_after=0.0,
                        n_outer=0, converged=True)
    win_set = set(win_idx)
    win_obs, fixed_obs, fixed_T = [], [], []
    for ki, kf in enumerate(keyframes):
        sel = np.array([int(i) in z_set for i in kf.ids], dtype=bool)
        if not sel.any():
            continue
        if ki in win_set:
            win_obs.append((kf.ids[sel], kf.uv[sel]))
        else:
            fixed_obs.append((kf.ids[sel], kf.uv[sel]))
            fixed_T.append(kf.T_cw)

    make_fns, dim, _ = _ba_build(cam, win_obs, fixed_obs, fixed_T, z_arr)
    t_work = [np.asarray(keyframes[ki].T_cw, dtype=float).copy() for ki in win_idx]
    x_work = points[z_arr].copy()

    def _rmse(t_lin: list, x_lin: np.ndarray) -> float:
        f, _ = make_fns(t_lin, x_lin)
        e = f(np.zeros(dim)).reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))

    rmse0 = _rmse(t_work, x_work)
    converged = False
    n_free = len(win_idx) - 1
    used = 0
    for used in range(1, n_outer + 1):
        f, jf = make_fns(t_work, x_work)
        sol = gauss_newton(
            f, jf,
            x0=np.zeros(dim),
            n_iters=inner_iters,
            lm_lambda=lm_lambda,
            robust_delta=huber_delta,
        )
        # 流形回缩（教程 (8.9)）：位姿左乘增量指数映射、点加增量。
        xi = sol.x[: 6 * n_free].reshape(n_free, 6)
        t_work = [t_work[0]] + [
            se3_exp(xi[k]) @ t_work[k + 1] for k in range(n_free)
        ]
        x_work = x_work + sol.x[6 * n_free:].reshape(len(z_arr), 3)
        if np.linalg.norm(sol.x) < tol or sol.cost < tol:
            converged = True
            break
    rmse1 = _rmse(t_work, x_work)

    # 写回共享地图（建图线程的"整合"步骤，论文 §6.4）。
    for b, ki in enumerate(win_idx):
        keyframes[ki].T_cw = t_work[b]
    points[z_arr] = x_work
    f_final, _ = make_fns(t_work, x_work)
    e_fin = f_final(np.zeros(dim))
    w_fin = huber_weights(e_fin, huber_delta)
    return BAResult(
        rmse_before=rmse0,
        rmse_after=rmse1,
        cost_after=float(np.sum(w_fin * e_fin * e_fin)),
        n_outer=int(used),
        converged=bool(converged),
    )


# --------------------------------------------------------------------- #
# 2D 平面场景仿真器（教学"数据集"：纹理平面 + 确定性种子）                  #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanarScene:
    """合成平面场景：GT 路标/位姿 + 逐帧带噪像素观测（教学"数据集"）。

    Attributes:
        cam: 针孔内参。
        width, height: 图像尺寸（px）。
        points: (M, 3) 世界系路标（带纹理平面 ``z = z0 + s_u x + s_v y`` 上
            的均匀采样点，m）。
        texture: (M,) 每路标灰度（[0, 1]；教学"纹理"——真实系统据其做
            patch 匹配，教学版关联按 id 直接给出，不参与几何）。
        poses_gt: T 个 (4, 4) 真值 ``T_cw``（第 0 帧为 I，世界系锚点）。
        observations: 每帧 ``(uv (V, 2), ids (V,))``——可见路标的带噪像素
            与全局 id（关联已知的 2D-3D 测量，patch 匹配的教学替身）。
    """

    cam: PinholeCamera
    width: int
    height: int
    points: np.ndarray
    texture: np.ndarray
    poses_gt: list[np.ndarray]
    observations: list[tuple[np.ndarray, np.ndarray]]


def render_planar_frames(
    cam: PinholeCamera,
    points_w: np.ndarray,
    poses: list[np.ndarray],
    width: int,
    height: int,
    pixel_noise_std: float = 0.5,
    seed: int = 0,
    margin: float = 8.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按 GT 位姿渲染逐帧观测：可见点投影 + 高斯像素噪声（确定性 seed）。

    Args:
        cam: 针孔内参。
        points_w: (M, 3) 世界系路标。
        poses: T 个 (4, 4) 真值 ``T_cw``。
        width, height: 图像尺寸（px）。
        pixel_noise_std: 像素高斯噪声标准差（px）；0 给出无噪测量。
        seed: 噪声种子（确定性，逐帧共用一条随机流）。
        margin: 可见性边距（px）。

    Returns:
        每帧 ``(uv (V, 2), ids (V,))`` 列表。
    """
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for T in poses:
        uv, ids, _ = project_visible(cam, T, points_w, width, height, margin=margin)
        if len(uv) and pixel_noise_std > 0.0:
            uv = uv + rng.normal(0.0, pixel_noise_std, uv.shape)
        out.append((uv, ids))
    return out


def simulate_planar_scene(
    n_frames: int = 36,
    width: int = 320,
    height: int = 240,
    focal: float = 300.0,
    n_points: int = 140,
    z0: float = 4.0,
    plane_slope: tuple[float, float] = (0.15, -0.1),
    x_range: tuple[float, float] = (-2.5, 4.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    xi: np.ndarray | None = None,
    poses: list[np.ndarray] | None = None,
    pixel_noise_std: float = 0.5,
    seed: int = 0,
) -> PlanarScene:
    """合成带纹理的 3D 平面场景与相机轨迹（教学"数据集"）。

    路标采自平面 ``z = z0 + s_u x + s_v y``（"纹理"为逐点灰度），相机沿
    :data:`DEFAULT_XI` 的增量链运动（前向 + 横向 + 轻微旋转——平移视差使
    路标深度可观；增量左乘 = 在世界系中表达，相机朝 +x/+z 前进，故路标
    ``x_range`` 默认偏向运动前方，保证全程有足量共视路标）。观测 = 可见
    路标投影 + 高斯像素噪声，关联按 id 给出（模块 docstring 的教学替身
    声明）。全部随机性定种子、可复现。

    Args:
        n_frames: 帧数 T（``poses`` 给定时被忽略）。
        width, height, focal: 图像几何。
        n_points: 路标数 M。
        z0: 平面平均深度（m）。
        plane_slope: 平面倾斜 ``(s_u, s_v)``。
        x_range, y_range: 路标采样范围（m）。
        xi: 每帧 SE(3) 增量 (6,)；默认 :data:`DEFAULT_XI`。
        poses: 显式给定 GT 位姿列表（跳过增量链，用于构造静止段等退化
            轨迹）。
        pixel_noise_std: 像素噪声标准差（px）。
        seed: 纹理与噪声种子（确定性）。

    Returns:
        :class:`PlanarScene`。
    """
    rng = np.random.default_rng(seed)
    cam = make_intrinsics(focal, focal, width / 2.0, height / 2.0)
    x = rng.uniform(x_range[0], x_range[1], n_points)
    y = rng.uniform(y_range[0], y_range[1], n_points)
    z = z0 + plane_slope[0] * x + plane_slope[1] * y
    points = np.stack([x, y, z], axis=1)
    texture = rng.uniform(0.0, 1.0, n_points)

    if poses is None:
        step = DEFAULT_XI if xi is None else np.asarray(xi, dtype=float).reshape(6)
        poses = [np.eye(4)]
        for _ in range(n_frames - 1):
            poses.append(se3_exp(step) @ poses[-1])
    observations = render_planar_frames(
        cam, points, poses, width, height, pixel_noise_std, seed
    )
    return PlanarScene(
        cam=cam,
        width=width,
        height=height,
        points=points,
        texture=texture,
        poses_gt=list(poses),
        observations=observations,
    )


# --------------------------------------------------------------------- #
# 系统主体：跟踪 / 建图交替阶段（论文双线程的教学模拟）                     #
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameLog:
    """单帧处理日志（:meth:`Ptam.process_frame` 的逐帧诊断）。

    Attributes:
        frame: 帧号（从 0 计）。
        status: ``"anchor"``（锚定第 0 关键帧）| ``"initialized"``（完成
            双帧初始化）| ``"awaiting-init"``（视差不足，等待）|
            ``"tracked"`` | ``"tracked+keyframe"``（本帧插入关键帧并建图）|
            ``"too-few-correspondences"`` | ``"no-measurements"``（无纹理
            帧的空观测）。
        pose: (4, 4) 本帧位姿（跟踪失败/无观测时保持上一帧位姿）。
        n_meas: 本帧测量数；n_tracked: 进入跟踪求解的对应数。
        inlier_ratio: 跟踪内点率（重投影 < gate_px 的比例）。
        reproj_rmse: 跟踪内点重投影 RMSE（px）。
        keyframe: 本帧是否插入关键帧。
        ba_rmse_before / ba_rmse_after: 本帧局部 BA 前 / 后的窗口重投影
            RMSE（px；未跑 BA 为 NaN）。
        parallax_deg: 与最近关键帧的平均视差（度；无跟踪为 NaN）。
    """

    frame: int
    status: str
    pose: np.ndarray
    n_meas: int
    n_tracked: int
    inlier_ratio: float
    reproj_rmse: float
    keyframe: bool
    ba_rmse_before: float
    ba_rmse_after: float
    parallax_deg: float


@dataclass(frozen=True)
class PtamResult:
    """:func:`run_ptam` 的输出（轨迹、关键帧与最终地图）。

    Attributes:
        poses: (T, 4, 4) 每帧位姿（未跟踪帧保持上一帧/单位位姿，
            配合 ``tracked`` 掩码使用）。
        tracked: (T,) 布尔掩码：该帧是否完成了一次有效跟踪求解。
        logs: 逐帧 :class:`FrameLog`。
        keyframe_frames: 插入关键帧的帧号（升序）。
        points: (M', 3) 终端已建图路标（世界系，NaN 行已剔除）。
        n_mapped: 已建图路标数。
    """

    poses: np.ndarray
    tracked: np.ndarray
    logs: tuple[FrameLog, ...]
    keyframe_frames: np.ndarray
    points: np.ndarray
    n_mapped: int


class Ptam:
    """PTAM 教学代理系统：跟踪（每帧）与建图（关键帧触发）交替阶段。

    状态 = 关键帧列表（位姿节点 + 观测）+ 路标数组（NaN 占位未建图）。
    逐帧调用 :meth:`process_frame`：

    1. **跟踪**（论文 §5）：与最近关键帧的公共观测中已建图者构成 3D-2D
       对应，:func:`track_frame` 做 motion-only G-N（零运动初值）；
    2. **关键帧判据**（论文 §6.2）：帧距 + （平移/中位深度 或 视差）；
    3. **建图**（论文 §6）：插入关键帧 -> 与最近关键帧三角化新点 ->
       :func:`local_ba` 局部 BA（窗口 = 最近 ``local_ba_k`` 个关键帧）。

    初始化（论文 §6.1）：第 0 个有观测的帧锚定为第 0 关键帧（T = I）；
    其后第一个视差足够的帧经八点法 + 手性分解 + DLT 三角化（复用
    :mod:`epipolar`）完成地图初始化。无观测帧（无纹理）与静止帧被安全
    跳过：不崩溃、不插关键帧（对应论文 §5.6 "不干净帧不入图"的防线）。

    Args:
        cam: 针孔内参。
        width, height: 图像尺寸（px，可见性筛选用）。
        init_min_parallax_deg: 初始化所需的最小平均视差（度）。
        init_min_common: 初始化所需的最小公共观测数（八点法需 >= 8）。
        key_min_frames: 两次关键帧的最小帧距（论文为 > 20 帧，教学版缩短）。
        key_min_parallax_deg: 关键帧视差阈值（度）。
        key_min_dist_frac: 关键帧平移阈值 = 该系数 x 跟踪点中位深度
            （论文 §6.2 "最小距离随平均深度伸缩"）。
        local_ba_k: 局部 BA 窗口大小（论文为最新 + 最近 4 个 = 5）。
        max_ba_points: 局部 BA 点数上限。
        gate_px: 跟踪内点门限（px，兼 Huber 核宽度）。
        min_track_obs: 有效跟踪所需的最少 3D-2D 对应数。
    """

    def __init__(
        self,
        cam: PinholeCamera,
        width: int = 320,
        height: int = 240,
        init_min_parallax_deg: float = 4.0,
        init_min_common: int = 12,
        key_min_frames: int = 3,
        key_min_parallax_deg: float = 3.0,
        key_min_dist_frac: float = 0.1,
        local_ba_k: int = 4,
        max_ba_points: int = 120,
        gate_px: float = 4.0,
        min_track_obs: int = 6,
    ) -> None:
        self.cam = cam
        self.width = int(width)
        self.height = int(height)
        self.init_min_parallax_deg = float(init_min_parallax_deg)
        self.init_min_common = int(init_min_common)
        self.key_min_frames = int(key_min_frames)
        self.key_min_parallax_deg = float(key_min_parallax_deg)
        self.key_min_dist_frac = float(key_min_dist_frac)
        self.local_ba_k = int(local_ba_k)
        self.max_ba_points = int(max_ba_points)
        self.gate_px = float(gate_px)
        self.min_track_obs = int(min_track_obs)

        self.keyframes: list[KeyFrame] = []
        self.points = np.zeros((0, 3))                 # 行号 = 路标全局 id
        self.point_status = np.zeros(0, dtype=bool)    # 已建图掩码
        self.logs: list[FrameLog] = []
        self.last_pose = np.eye(4)
        self.frame_idx = -1
        self.frames_since_kf = 0

    # ------------------------------------------------------------ 内部 --
    def _ensure_capacity(self, max_id: int) -> None:
        """路标数组按需扩容（新行 = NaN / 未建图）。"""
        if len(self.points) <= max_id:
            new_len = max(max_id + 1, 2 * len(self.points))
            pad = new_len - len(self.points)
            self.points = np.vstack(
                [self.points, np.full((pad, 3), np.nan)]
            )
            self.point_status = np.concatenate(
                [self.point_status, np.zeros(pad, dtype=bool)]
            )

    def _append_keyframe(self, frame: int, T_cw: np.ndarray,
                         uv: np.ndarray, ids: np.ndarray) -> None:
        if len(ids):
            self._ensure_capacity(int(ids.max()))   # 观测可先于建图出现（占位）
        self.keyframes.append(KeyFrame(frame=frame, T_cw=T_cw, ids=ids, uv=uv))
        self.frames_since_kf = 0

    def _try_initialize(self, uv: np.ndarray, ids: np.ndarray) -> tuple[bool, np.ndarray]:
        """双帧初始化（论文 §6.1 的教学替身，复用 :mod:`epipolar`）。

        与锚定关键帧的公共观测足够多且视差足够时：归一化八点法（教程
        (5.1)-(5.2)）-> 4 候选分解手性检验（教程 (5.7)）-> DLT 三角化
        （教程 (5.9)）建基图；尺度由 E 分解的单位平移约定（单目尺度
        不可观，评估端以 Sim(3) 对齐吸收）。

        Returns:
            ``(ok, T_cw)``：成功时 ``T_cw`` 为新关键帧位姿并把公共观测
            写入地图；失败时 ``(False, I)``。
        """
        anchor = self.keyframes[0]
        common, ia, ib = np.intersect1d(anchor.ids, ids, return_indices=True)
        if len(common) < self.init_min_common:
            return False, np.eye(4)
        uv_a, uv_b = anchor.uv[ia], uv[ib]
        if mean_parallax(self.cam, uv_a, uv_b) < self.init_min_parallax_deg:
            return False, np.eye(4)

        # 依据：教程 (5.1)-(5.2)/(5.7)/(5.9)——2D-2D 相对位姿 + 三角化基图
        # （论文 §6.1 五点立体的教学替身；RANSAC 省略：教学数据无外点）。
        E, _ = eight_point(uv_a, uv_b, self.cam.matrix())
        R, t = decompose_E(E, uv_a, uv_b, self.cam.matrix())
        T_new = np.eye(4)
        T_new[:3, :3] = R
        T_new[:3, 3] = t
        X0 = triangulate(np.eye(4), T_new, uv_a, uv_b, K=self.cam.matrix())
        # 手性过滤：两个关键帧中深度均为正才入库（教程 (5.8) 的思想）。
        keep = (X0[:, 2] > 0.2) & (transform_points(T_new, X0)[:, 2] > 0.2)
        if keep.sum() < 8:
            return False, np.eye(4)
        self._ensure_capacity(int(common.max()))
        pids = common[keep]
        self.points[pids] = X0[keep]
        self.point_status[pids] = True
        self._append_keyframe(self.frame_idx, T_new, uv, ids)
        # 初始化后立即做一次两关键帧局部 BA，把基图与相对位姿拉紧（§6.3）。
        local_ba(
            self.cam, self.keyframes, self.points,
            list(range(len(self.keyframes))),
            max_ba_points=self.max_ba_points, huber_delta=self.gate_px,
        )
        return True, self.keyframes[-1].T_cw

    def _triangulate_new_points(self, pose: np.ndarray, uv: np.ndarray,
                                ids: np.ndarray) -> int:
        """新关键帧入图后：与最近关键帧的公共未建图观测三角化新点（§6.2）。

        论文用极线搜索找对应再三角化；教学版关联已知，直接 DLT
        （教程 (5.9)），位姿来自当前估计。返回新建图路标数。
        """
        prev = self.keyframes[-2]
        common, ia, ib = np.intersect1d(prev.ids, ids, return_indices=True)
        if len(common) == 0:
            return 0
        unmapped = ~self.point_status[common]
        if not unmapped.any():
            return 0
        pids = common[unmapped]
        uv_prev, uv_new = prev.uv[ia][unmapped], uv[ib][unmapped]
        T_rel = pose @ np.linalg.inv(prev.T_cw)      # prev 相机系 -> 新帧相机系
        X_prev = triangulate(np.eye(4), T_rel, uv_prev, uv_new, K=self.cam.matrix())
        keep = (X_prev[:, 2] > 0.2) & (transform_points(T_rel, X_prev)[:, 2] > 0.2)
        if not keep.any():
            return 0
        self._ensure_capacity(int(pids.max()))
        self.points[pids[keep]] = transform_points(
            np.linalg.inv(prev.T_cw), X_prev[keep]   # prev 相机系 -> 世界系
        )
        self.point_status[pids[keep]] = True
        return int(keep.sum())

    # ------------------------------------------------------------ 主流程 --
    def process_frame(self, uv: np.ndarray, ids: np.ndarray) -> FrameLog:
        """处理一帧：跟踪 ->（触发时）插关键帧 + 三角化新点 + 局部 BA。

        Args:
            uv: (V, 2) 本帧像素测量（可为空——无纹理帧）。
            ids: (V,) 对应路标全局 id。

        Returns:
            :class:`FrameLog`（逐帧诊断，供测试与 DEBUG 观察）。
        """
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        ids = np.asarray(ids, dtype=int).reshape(-1)
        self.frame_idx += 1
        self.frames_since_kf += 1
        pose = self.last_pose
        status = "tracked"
        n_tracked = 0
        inlier_ratio = 0.0
        reproj = float("nan")
        parallax = float("nan")
        kf_ins = False
        ba_before = ba_after = float("nan")

        if len(uv) == 0:
            # 无纹理帧：特征检测落空——保持上一帧位姿、不入图（§5.6 的
            # "不干净帧禁止作为关键帧入图"防线的教学版）。
            status = "no-measurements"
        elif not self.keyframes:
            # 第 0 个有观测的帧：锚定关键帧（论文 §6.1 第一次按键），T = I。
            self._append_keyframe(self.frame_idx, np.eye(4), uv, ids)
            status, pose = "anchor", np.eye(4)
        elif len(self.keyframes) == 1:
            ok, pose = self._try_initialize(uv, ids)
            status = "initialized" if ok else "awaiting-init"
        else:
            # ---- 跟踪（论文 §5）：新帧 x 最近关键帧的公共已建图观测 ----
            near = self.keyframes[-1]
            common, ia, ib = np.intersect1d(near.ids, ids, return_indices=True)
            mapped = self.point_status[common] if len(common) else np.zeros(0, bool)
            sel = mapped
            if int(sel.sum()) < self.min_track_obs:
                status = "too-few-correspondences"     # 保持上一帧位姿，不崩溃
            else:
                X = self.points[common[sel]]
                uv_near, uv_new = near.uv[ia][sel], uv[ib][sel]
                res = track_frame(
                    self.cam, X, uv_new, self.last_pose, gate_px=self.gate_px
                )
                pose = res.T_cw
                n_tracked = res.n_obs
                inlier_ratio = res.inlier_ratio
                reproj = res.reproj_rmse
                parallax = mean_parallax(self.cam, uv_near, uv_new)
                # ---- 关键帧判据（论文 §6.2）：帧距 + (平移/中位深度 或 视差) ----
                center = camera_center(pose)
                dist = min(
                    float(np.linalg.norm(center - camera_center(kf.T_cw)))
                    for kf in self.keyframes
                )
                med_depth = float(np.median(transform_points(pose, X)[:, 2]))
                want_kf = (
                    self.frames_since_kf >= self.key_min_frames
                    and (
                        dist > self.key_min_dist_frac * med_depth
                        or parallax >= self.key_min_parallax_deg
                    )
                )
                if want_kf:
                    # ---- 建图阶段（论文 §6，教学版与跟踪交替执行）----
                    self._append_keyframe(self.frame_idx, pose, uv, ids)
                    self._triangulate_new_points(pose, uv, ids)
                    window = list(
                        range(max(0, len(self.keyframes) - self.local_ba_k),
                              len(self.keyframes))
                    )
                    ba = local_ba(
                        self.cam, self.keyframes, self.points, window,
                        max_ba_points=self.max_ba_points,
                        huber_delta=self.gate_px,
                    )
                    pose = self.keyframes[-1].T_cw    # BA 精化后的本帧位姿
                    ba_before, ba_after = ba.rmse_before, ba.rmse_after
                    kf_ins = True
                    status = "tracked+keyframe"
        self.last_pose = pose
        log = FrameLog(
            frame=self.frame_idx,
            status=status,
            pose=pose.copy(),
            n_meas=len(uv),
            n_tracked=int(n_tracked),
            inlier_ratio=float(inlier_ratio),
            reproj_rmse=float(reproj),
            keyframe=kf_ins,
            ba_rmse_before=ba_before,
            ba_rmse_after=ba_after,
            parallax_deg=float(parallax),
        )
        self.logs.append(log)
        return log


def run_ptam(scene: PlanarScene, **kwargs) -> PtamResult:
    """在合成场景上跑完整 PTAM 教学流水线（逐帧 :meth:`Ptam.process_frame`）。

    Args:
        scene: :class:`PlanarScene`（GT 与逐帧观测）。
        **kwargs: 透传给 :class:`Ptam` 构造器。

    Returns:
        :class:`PtamResult`（轨迹、跟踪掩码、逐帧日志、关键帧与地图）。
    """
    system = Ptam(scene.cam, width=scene.width, height=scene.height, **kwargs)
    t_total = len(scene.observations)
    poses = np.tile(np.eye(4), (t_total, 1, 1))
    tracked = np.zeros(t_total, dtype=bool)
    for k, (uv, ids) in enumerate(scene.observations):
        log = system.process_frame(uv, ids)
        poses[k] = log.pose
        tracked[k] = log.n_tracked >= system.min_track_obs
    mapped = system.points[system.point_status]
    return PtamResult(
        poses=poses,
        tracked=tracked,
        logs=tuple(system.logs),
        keyframe_frames=np.array([kf.frame for kf in system.keyframes], dtype=int),
        points=mapped,
        n_mapped=int(len(mapped)),
    )

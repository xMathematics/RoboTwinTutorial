"""SLAM 统一测评模块（metrics）—— 全部指标的唯一定义处。

函数流水线
----------
本模块是评估流水线的最后一环，被各模块的评估/测试环节调用：
``fastslam``、``vins``、``loam2d``、``droidlite``、``photoba`` 等模块产出
的轨迹与地图，都应经本模块的纯函数与真值比较。``tests/test_*.py`` 中的
指标断言从这里取公式与实现（``vins.align_se3`` 与各测试文件里的
``_position_rmse`` 是同一公式的就地实现，属历史遗留，逐步统一到本模块）。
依赖方向：``各模块 → tests/test_*.py → 本模块 → core.lie``（旋转误差复用
:func:`core.lie.so3_log`）；本模块不依赖任何被测模块，只依赖 numpy。

输入/输出：输入估计轨迹 ``traj_est`` (N, 3) 与真值 ``traj_gt`` (N, 3)
（单位 m，逐帧一一对应）、旋转矩阵 (3, 3)、相邻帧步长 (N-1,)（单位 m）；
输出对齐参数 ``(R (3,3), t (3,), s)`` 或标量指标（单位 m / deg / 无量纲）。

指标清单（公式编号 (M.x) 供 METRICS.md 与教程回引）
----------------------------------------------------
- :func:`umeyama_align`      Umeyama/Kabsch 相似对齐（Umeyama 1991）
- :func:`ate_rmse`           绝对轨迹误差 RMSE（Sturm et al., 2012 的 ATE）
- :func:`rotation_error_deg` 旋转误差（教程第 02 章对数映射）
- :func:`scale_ratio`        尺度比（单目尺度可观性，教程第 10 章 §10.3）
- :func:`rpe_translation`    相对位姿误差·平移（Sturm et al., 2012 的 RPE）

全部函数为纯函数：不修改输入、无全局状态，可安全地在测试断言中直接调用。
"""
import numpy as np

from core.lie import so3_log

__all__ = [
    "umeyama_align",
    "ate_rmse",
    "rotation_error_deg",
    "scale_ratio",
    "rpe_translation",
]


def umeyama_align(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Umeyama/Kabsch 最小二乘相似对齐：求 ``(R, t, s)`` 使 ``dst ≈ s·R@src + t``。

    这是 ATE 评估的第一步（Sturm et al., 2012：先按 Umeyama 对齐轨迹再算
    误差），也是一切"估计结果与真值只差一个规范自由度（gauge freedom，
    教程第 09 章 Sim(3) 回环、第 10 章 §10.3 尺度可观性）"情形的公共工具。

    推导梗概（Umeyama, "Least-squares estimation of transformation
    parameters between two point patterns", IEEE TPAMI 13(4), 1991, §III）：
    最小化 $\sum_i \|q_i - sRp_i - t\|^2$。对 $t$ 求导置零（依据：凸二次的
    驻点条件）得 $t = \bar q - sR\bar p$；代回消去 $t$ 后，问题化为正交
    Procrustes 问题 + 一维标量，均有闭式解（步骤见行内注释）。

    Args:
        src: (N, 3) 源点集（如估计轨迹位置），单位 m。
        dst: (N, 3) 目标点集（如真值轨迹位置），单位 m，与 ``src`` 逐点对应。
        with_scale: 是否估计尺度 ``s``；False 时固定 s=1（纯 Kabsch/SE(3)
            对齐，适合尺度已可观的 VIO）；True 时估计 s（Sim(3) 对齐，
            适合带任意尺度的纯单目结果）。

    Returns:
        R: (3, 3) 旋转矩阵，det = +1（第 3 步 det 校正保证，见行内注释）。
        t: (3,) 平移矢量，单位 m。
        s: 尺度因子，无量纲；``with_scale=False`` 时恒为 1.0。
    """
    p = np.asarray(src, dtype=float).reshape(-1, 3)
    q = np.asarray(dst, dtype=float).reshape(-1, 3)
    if len(p) != len(q):
        raise ValueError(f"src/dst 点数不一致: {len(p)} vs {len(q)}")
    if len(p) < 3:
        raise ValueError("至少需要 3 个点才能唯一确定 SO(3) 旋转")
    # 第 1 步：中心化。依据：Umeyama 1991 §III 第一步——平移与旋转在
    # 最小二乘中耦合，先减去质心把平移解耦出去（q̄, p̄ 为质心）。
    p_bar = p.mean(axis=0)
    q_bar = q.mean(axis=0)
    pc = p - p_bar
    qc = q - q_bar
    # 第 2 步：互协方差 H = Σ p'_i q'_i^T（Umeyama 记号 Σ_pq 的未归一化版，
    # 差一个常数因子 1/N，不影响 SVD 的方向、只按比例改奇异值）。
    H = pc.T @ qc
    # 第 3 步：SVD 分解 H = U Λ V^T，并做 det 校正防反射。
    U, S, Vt = np.linalg.svd(H)
    # d = sign(det(V U^T))：把"最优正交阵"折叠回 SO(3)（det=+1）。依据：
    # Umeyama 1991 的反射校正——SVD 给出的无约束最优解可能是反射（镜像，
    # det=-1），不是刚体旋转；校正把符号翻转吸收到最小奇异值方向上。
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    if with_scale:
        # 第 4 步：尺度闭式解 s = tr(D Λ)/σ_p² = (S₁+S₂+d·S₃)/Σ‖p'_i‖²。
        # 依据：Umeyama 1991——对 s 求导置零；σ_p² = mean(‖p'_i‖²) 为源点集
        # 均方半径，D = diag(1, 1, d)（与第 3 步的 det 校正保持一致）。
        s = float((S @ np.array([1.0, 1.0, d])) / np.sum(pc * pc))
    else:
        s = 1.0
    # 第 5 步：平移由第 1 步的驻点条件回代 t = q̄ - s·R·p̄。
    t = q_bar - s * (R @ p_bar)
    return R, t, s


def ate_rmse(traj_est: np.ndarray, traj_gt: np.ndarray, align: bool = True) -> float:
    """绝对轨迹误差（ATE）RMSE，单位 m。

    定义（Sturm, Engelhard, Endres, Burgard, Cremers, "A Benchmark for the
    Evaluation of RGB-D SLAM Systems", IROS 2012, §IV-A；即本项目文档所称
    Sturm et al. 2012）：估计轨迹与真值轨迹逐帧配对后，先（可选）用带尺度
    的 Umeyama 对齐把估计轨迹摆到真值坐标系（Sturm et al. 2012 的做法，
    以兼容单目系统的任意尺度），再取逐点欧氏误差的均方根，见 (M.2)。

    为什么必须对齐：估计器存在不可观的规范自由度（gauge freedom）——纯单目
    共 7 自由度（Sim(3)：3 旋转 + 3 平移 + 1 尺度；第 05 章三角化的尺度
    不确定性、第 09 章 Sim(3) 回环）、单目+IMU 剩 4 自由度（全局 yaw +
    全局平移，教程第 10 章 §10.3 第 3 步可观性论证）。这些自由度上估计器
    "取哪个值都与观测一致"，不对齐就把与精度无关的全局摆位差记成了误差。
    对齐后评估的是**估计轨迹的形状**，7 自由度 gauge 已消去——因此
    ``align=True`` 是 ATE 的默认用法；``align=False`` 仅用于两轨迹已知
    处于同一坐标系（如消融对比同一 gauge 下的两个估计器）。

    Args:
        traj_est: (N, 3) 估计轨迹位置，单位 m。
        traj_gt: (N, 3) 真值轨迹位置，单位 m，与 ``traj_est`` 逐帧一一对应。
        align: True 时先做带尺度的 Umeyama 对齐（默认）；False 时逐点
            直接比较。

    Returns:
        ATE RMSE，标量，单位 m，取值 ≥ 0。
    """
    e = np.asarray(traj_est, dtype=float).reshape(-1, 3)
    g = np.asarray(traj_gt, dtype=float).reshape(-1, 3)
    if len(e) != len(g):
        raise ValueError(f"轨迹帧数不一致: {len(e)} vs {len(g)}")
    if align:
        # 依据：(M.2) 的对齐步骤——带尺度的 Umeyama（Sturm et al. 2012）。
        R, t, s = umeyama_align(e, g, with_scale=True)
        e = s * (e @ R.T) + t  # 等价于逐行 s·(R @ p_i) + t
    err = e - g
    # RMSE：sqrt(mean_i ‖e_i - g_i‖²)，单位 m（依据：(M.2) 定义）。
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    r"""旋转误差（度）：$\|\log(R_{est}^\top R_{gt})\| \cdot 180/\pi$。

    相对旋转 $R_{rel} = R_{est}^\top R_{gt}$ 的转角即两个姿态在 SO(3) 上
    的最短测地距离：对数映射把旋转矩阵映回旋转矢量 $\varphi$，其范数
    $\|\varphi\| = \theta$ 就是转角（教程第 02 章；实现复用
    :func:`core.lie.so3_log`，其闭式推导见视觉 SLAM 十四讲 Eq. 4.19-4.22）。

    Args:
        R_est: (3, 3) 估计旋转矩阵，无量纲，det = +1。
        R_gt: (3, 3) 真值旋转矩阵，无量纲，det = +1。

    Returns:
        旋转误差，标量，单位 deg，取值范围 [0, 180]（SO(3) 测地距离的上限
        是 π，由 :func:`core.lie.so3_log` 的主值分支保证）。
    """
    R_est = np.asarray(R_est, dtype=float).reshape(3, 3)
    R_gt = np.asarray(R_gt, dtype=float).reshape(3, 3)
    # 相对旋转：把 est 姿态转到 gt 姿态所需的旋转（依据：R_est^T R_gt 作用
    # 于 est 坐标系下的矢量即得 gt 坐标系下的同一矢量）。
    R_rel = R_est.T @ R_gt
    # ‖log(R_rel)‖·180/π：对数映射的范数即转角（教程第 02 章 Eq. 2.15-2.19）。
    return float(np.linalg.norm(so3_log(R_rel)) * 180.0 / np.pi)


def scale_ratio(lengths_est: np.ndarray, lengths_gt: np.ndarray) -> float:
    """尺度比：估计轨迹与真值轨迹"相邻帧距离之和"的比值，无量纲，见 (M.4)。

    单目视觉恢复的轨迹整体带一个任意尺度因子（第 05 章三角化的尺度不确定
    性；教程第 10 章 §10.3：纯单目 7 自由度 gauge 中的一维）。该比值是
    整条轨迹尺度合理性的单一度量——单目+IMU 系统用它检验 IMU 是否把尺度
    钉住（尺度可观性论证见教程第 10 章 §10.3 第 3 步），理想值 1；本项目
    vins 模块的尺度测试（``tests/test_vins.py`` 的 ``[pure-visual]`` 打印）
    就是它的逐对版本：IMU 权重清零后比值跌到 0.38-0.42，即尺度不可观。

    Args:
        lengths_est: (N-1,) 估计轨迹的相邻帧距离（步长），单位 m；可由
            ``np.linalg.norm(np.diff(traj, axis=0), axis=1)`` 从轨迹得到。
        lengths_gt: (N-1,) 真值轨迹的相邻帧距离，单位 m，与 ``lengths_est``
            逐段对应。

    Returns:
        尺度比，无量纲标量；≈1 表示尺度正确，>1 估计偏大，<1 估计偏小。
    """
    le = np.asarray(lengths_est, dtype=float).reshape(-1)
    lg = np.asarray(lengths_gt, dtype=float).reshape(-1)
    if len(le) != len(lg):
        raise ValueError(f"步长段数不一致: {len(le)} vs {len(lg)}")
    denom = float(np.sum(lg))
    if denom <= 0.0:
        # 真值轨迹静止（总路程为 0）时尺度比无定义——早失败优于静默给出 inf。
        raise ValueError("真值步长之和必须为正（静止轨迹无法度量尺度）")
    # (M.4)：Σ‖Δp^est‖ / Σ‖Δp^gt‖——路程之比即整体尺度估计。
    return float(np.sum(le) / denom)


def rpe_translation(
    traj_est: np.ndarray, traj_gt: np.ndarray, delta: int = 1
) -> float:
    """相对位姿误差（RPE）的平移分量 RMSE，单位 m，见 (M.5)。

    与 ATE 的区别（Sturm et al., 2012, §IV-B）：ATE 比较**绝对位置**，
    全局漂移会随时间累积进误差；RPE 比较**间隔 ``delta`` 帧的相对位移**，
    只看局部运动是否准确——一条整体缓慢漂移但局部平滑的轨迹 RPE 小而
    ATE 大；反之局部抖动、全局恰好贴合的轨迹 ATE 小而 RPE 大。相对量对
    刚体规范自由度（gauge）不变，因此 RPE **无需对齐**。完整 RPE 还含
    旋转分量（Sturm et al. 2012 定义在 SE(3) 上）；本函数取平移分量，
    因为 (N, 3) 轨迹只含位置——需要旋转分量时请对 (4, 4) 位姿序列用
    :func:`core.lie.se3_log` 自行扩展。

    Args:
        traj_est: (N, 3) 估计轨迹位置，单位 m。
        traj_gt: (N, 3) 真值轨迹位置，单位 m，与 ``traj_est`` 逐帧对应。
        delta: 间隔帧数，取值 [1, N-1]；1 = 逐帧局部误差，越大越接近 ATE
            的"长程"视角。

    Returns:
        RPE 平移分量 RMSE，标量，单位 m，取值 ≥ 0。
    """
    e = np.asarray(traj_est, dtype=float).reshape(-1, 3)
    g = np.asarray(traj_gt, dtype=float).reshape(-1, 3)
    if len(e) != len(g):
        raise ValueError(f"轨迹帧数不一致: {len(e)} vs {len(g)}")
    if not 1 <= delta < len(e):
        raise ValueError(f"delta 必须在 [1, {len(e) - 1}] 内，当前为 {delta}")
    # (M.5)：全部 (i, i+delta) 对上，相对位移之差的 RMSE。
    # e[delta:] - e[:-delta] 一次性给出所有间隔 delta 的相对位移（依据：
    # 向量化切片，与逐对循环等价，tests/test_metrics.py 有朴素版交叉验证）。
    rel_est = e[delta:] - e[:-delta]  # (N-delta, 3) 估计轨迹的相对位移
    rel_gt = g[delta:] - g[:-delta]   # (N-delta, 3) 真值轨迹的相对位移
    err = rel_est - rel_gt
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

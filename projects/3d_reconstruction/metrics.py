"""3D 重建统一测评模块（metrics）—— 全部指标的唯一定义处。

函数流水线
----------
本模块是评估流水线的最后一环：``scene``（解析真值表面点）、``tsdf + marching``
（重建网格）的产物在此与真值比较，``demo.py`` 与 ``tests/test_metrics.py``
直接调用：

    sample_mesh_surface(v, f, n)   （重建网格 -> 均匀表面点，面积加权 (M.4)）
        -> chamfer_distance(P, Q)            （对称表面距离 (M.1)）
        -> accuracy_completion(est, gt)      （两个单向距离 (M.2)，DTU 口径）
        -> f_score(est, gt, tau)             （阈值化 P/R 的调和平均 (M.3)）

依赖方向：本模块不依赖任何被测模块，只依赖 numpy（纯函数，无状态无 I/O）。

输入/输出：输入点集 (N, 3) / (M, 3)（单位 m；Chamfer/Accuracy 内部把距离
平方后平均，输出单位 m^2；F-score 的阈值 tau 单位 m）；输出标量指标或
采样点 (n, 3)。公式编号 (M.x) 供 METRICS.md 与教程回引。

指标清单
--------
- :func:`chamfer_distance`      对称 Chamfer 距离（Barrow et al., 1977 起源）
- :func:`accuracy_completion`   DTU 精度/完整度（Jensen et al., 2014）
- :func:`f_score`               F-score@tau（Knapitsch et al., 2017，DTU/Tanks & Temples）
- :func:`sample_mesh_surface`   面积加权三角形表面采样（评测的公共前置）

全部函数为纯函数：不修改输入、无全局状态，可安全地在测试断言中直接调用。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_completion",
    "chamfer_distance",
    "f_score",
    "sample_mesh_surface",
]


def _pairwise_min_dist_sq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """src 每个点到 dst 最近点的平方距离（分块计算，控制内存）。

    Args:
        src: (N, 3) 查询点集，单位 m。
        dst: (M, 3) 参考点集，单位 m（须非空）。

    Returns:
        (N,) 每行的 min_q ||p_q - d_q||^2，单位 m^2。
    """
    if len(dst) == 0:
        raise ValueError("参考点集为空：距离无定义")
    out = np.empty(len(src))
    chunk = max(1, int(4_000_000 // max(len(dst), 1)))  # 每块距离矩阵 ~ 数十 MB 上限
    for s in range(0, len(src), chunk):
        diff = src[s:s + chunk, None, :] - dst[None, :, :]   # (b, M, 3)
        out[s:s + chunk] = np.min(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
    return out


def chamfer_distance(points_p: np.ndarray, points_q: np.ndarray) -> float:
    """对称 Chamfer 距离：两个点集间"平均最近点平方距离"的平均，见 (M.1)。

    $$d_{CD}(P, Q) = \\frac{1}{2}\\Bigl(\\frac{1}{|P|}\\sum_{p}\\min_{q}\\|p-q\\|^2
    + \\frac{1}{|Q|}\\sum_{q}\\min_{p}\\|q-p\\|^2\\Bigr) \\tag{M.1}$$

    起源于 Barrow et al. (1977) 的参数拟合距离；取平方均值是形状重建评测的
    常用口径（如 Tatarchenko et al., CVPR 2019）。对称项缺一不可：只有
    P->Q 方向会奖励"重建点都贴着真值"却漏掉真值的大部分（稀疏作弊）；
    只有 Q->P 方向会奖励"覆盖全"却允许大量离谱外点。它同时惩罚
    **精度**（第一项）与**完整度**（第二项），是 (M.2) 两个单向项的合成。
    教程语境：衡量第 04 章融合 (4.8) + 第 05 章提取 (5.1)/(5.2) 重建出的
    表面点与真值表面的整体偏差。

    Args:
        points_p: (N, 3) 点集 P，单位 m（如重建网格表面采样点）。
        points_q: (M, 3) 点集 Q，单位 m（如真值表面采样点）。

    Returns:
        对称 Chamfer 距离，标量，单位 m^2，取值 >= 0；两集合重合时为 0。

    Raises:
        ValueError: 任一点集为空或两维数不一致。
    """
    p = np.asarray(points_p, dtype=float).reshape(-1, 3)
    q = np.asarray(points_q, dtype=float).reshape(-1, 3)
    if len(p) == 0 or len(q) == 0:
        raise ValueError("Chamfer 距离要求两个非空点集")
    # (M.1)：两个方向的"平均最近点平方距离"各算一遍再平均。
    d_pq = float(np.mean(_pairwise_min_dist_sq(p, q)))
    d_qp = float(np.mean(_pairwise_min_dist_sq(q, p)))
    return 0.5 * (d_pq + d_qp)


def accuracy_completion(
    est_pts: np.ndarray, gt_pts: np.ndarray
) -> tuple[float, float]:
    """DTU 口径的精度（accuracy）与完整度（completion），见 (M.2)。

    $$\\mathrm{acc} = \\frac{1}{|E|}\\sum_{e \\in E}\\min_{g \\in G}\\|e-g\\|^2,
    \\qquad
    \\mathrm{comp} = \\frac{1}{|G|}\\sum_{g \\in G}\\min_{e \\in E}\\|g-e\\|^2
    \\tag{M.2}$$

    **accuracy 衡量估计的多余程度**：重建点离最近真值点多远——重建了
    真值上不存在的几何（噪声、飞点、虚假面片）会推高它，而漏掉半个物体
    不影响它；**completion 衡量缺失程度**：真值点离最近重建点多远——
    漏测（遮挡、视角不足）会推高它，而多重建垃圾不影响它。二者合起来
    才是 (M.1) 的完整画面。出处：DTU 评测协议（R. Jensen et al.,
    *"Large Scale Multi-view Stereopsis Evaluation"*, CVPR 2014，第 3 节
    accuracy / completion 定义；平方距离口径沿用其 MATLAB 实现）。
    教程语境：第 03 章 MVS 与第 04/05 章重建管线的评测即此协议。

    Args:
        est_pts: (N, 3) 估计（重建）表面点，单位 m。
        gt_pts: (M, 3) 真值表面点，单位 m。

    Returns:
        (accuracy, completion) 二元组，单位均为 m^2，取值 >= 0。

    Raises:
        ValueError: 任一点集为空。
    """
    est = np.asarray(est_pts, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt_pts, dtype=float).reshape(-1, 3)
    if len(est) == 0 or len(gt) == 0:
        raise ValueError("accuracy/completion 要求两个非空点集")
    accuracy = float(np.mean(_pairwise_min_dist_sq(est, gt)))   # est -> gt：多余程度
    completion = float(np.mean(_pairwise_min_dist_sq(gt, est)))  # gt -> est：缺失程度
    return accuracy, completion


def f_score(
    est_pts: np.ndarray, gt_pts: np.ndarray, tau: float
) -> float:
    """F-score@tau：阈值化查准率与查全率的调和平均，见 (M.3)。

    $$P = \\frac{1}{|E|}\\#\\{e : \\min_g\\|e-g\\| \\le \\tau\\},\\quad
    R = \\frac{1}{|G|}\\#\\{g : \\min_e\\|g-e\\| \\le \\tau\\},\\quad
    F_\\tau = \\frac{2PR}{P+R} \\tag{M.3}$$

    （P+R = 0 时约定 F = 0。）tau 是"距离多近算对"的工程判定线：
    (M.2) 的平方距离对少数离谱外点敏感（一个大误差平方后淹没全部好点），
    F-score 把误差二值化后只问"多少比例的点在容差内"，对离群更稳健、
    也更贴近应用判断（重建点要么能用要么不能用）。

    出处：DTU F-score 与 Tanks and Temples 基准（A. Knapitsch, T. Aanæs,
    R. Jensen, R. Koch, *"A Benchmark and Comparison of Point Cloud
    Registration Algorithms"*, IJCV 2021 [T&T 版本为 ACM TOG 2017]；
    DTU F@tau 常用 tau = 5/10/20 mm）。教程语境：为第 05 章 (5.1) 提取
    与第 04 章 (4.8) 融合的产物提供"容差内比例"式评测。

    Args:
        est_pts: (N, 3) 估计表面点，单位 m。
        gt_pts: (M, 3) 真值表面点，单位 m。
        tau: 距离容差，单位 m，取值 > 0（与点云尺度匹配，如 0.02 = 20 mm）。

    Returns:
        F-score@tau，无量纲标量，取值 [0, 1]；est 与 gt 重合时为 1。

    Raises:
        ValueError: 点集为空或 tau <= 0。
    """
    est = np.asarray(est_pts, dtype=float).reshape(-1, 3)
    gt = np.asarray(gt_pts, dtype=float).reshape(-1, 3)
    if len(est) == 0 or len(gt) == 0:
        raise ValueError("f_score 要求两个非空点集")
    if tau <= 0.0:
        raise ValueError(f"tau 必须为正（单位 m），当前为 {tau}")
    # 平方距离与 tau^2 比较，免去开方（等价于欧氏距离 <= tau）。
    precision = float(np.mean(_pairwise_min_dist_sq(est, gt) <= tau * tau))
    recall = float(np.mean(_pairwise_min_dist_sq(gt, est) <= tau * tau))
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def sample_mesh_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """三角网格表面均匀采样 n 个点（按三角形面积加权），见 (M.4)。

    两步采样（标准蒙特卡洛表面积分）：
    1. **选三角形**：以面积 $A_t = \\tfrac{1}{2}\\|(v_1-v_0)\\times(v_2-v_0)\\|$
       为权重的离散分布抽 n 个三角形（``rng.choice`` 的 p 参数）；
    2. **面内均匀落点**：重心坐标 $u, v \\sim U(0,1)$，
       $p = (1-\\sqrt{u})\\,v_0 + \\sqrt{u}(1-v)\\,v_1 + \\sqrt{u}\\,v\\,v_2$
       （带 $\\sqrt{u}$ 的 striped 折叠使落点在三角形上**均匀**而非重心聚心）。

    作为 (M.1)-(M.3) 的公共前置：网格是连续表面，逐点指标都先把它
    离散成等密度点云（每单位面积样本数相同 <=> 面积加权选面）。

    Args:
        vertices: (V, 3) 网格顶点，单位 m。
        triangles: (T, 3) 三角形顶点索引（int，指向 ``vertices`` 行）。
        n_samples: 采样点数，取值 >= 1。
        seed: 随机种子（确定性输出）。

    Returns:
        points: (n_samples, 3) 表面采样点，单位 m，全部落在网格表面上。

    Raises:
        ValueError: 网格为空或总面积为 0（退化网格）。
    """
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        raise ValueError("网格没有三角形，无法采样")
    tri = v[f]                                        # (T, 3, 3) 三个顶点
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=-1)            # 2 倍面积，m^2
    total = float(area2.sum())
    if total <= 0.0:
        raise ValueError("网格总面积为 0（退化网格），无法采样")
    rng = np.random.default_rng(seed)
    # 第 1 步：面积加权选面（A_t / ΣA 为抽中概率，均匀表面密度）。
    face_idx = rng.choice(len(f), size=n_samples, p=area2 / total)
    # 第 2 步：面内重心坐标均匀落点（sqrt 折叠，见函数 docstring）。
    u = rng.random(n_samples)
    v_ = rng.random(n_samples)
    su = np.sqrt(u)
    picked = tri[face_idx]                            # (n, 3, 3)
    points = ((1.0 - su)[:, None] * picked[:, 0]
              + (su * (1.0 - v_))[:, None] * picked[:, 1]
              + (su * v_)[:, None] * picked[:, 2])
    return points

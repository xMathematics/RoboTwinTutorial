"""Gauss-Newton / Levenberg-Marquardt 求解器与鲁棒核。

函数流水线
----------
实现教程第 08 章（后端 II：图优化与 BA，Eq. 8.1-8.8）的非线性最小二乘机制。
被 ``epipolar``、``direct``、``droidlite``、``photoba``、``loam2d`` 复用
（各自传入残差/雅可比函数调 :func:`gauss_newton`），``vins`` 复用
:func:`huber_weights` 做鲁棒加权。

输入：残差函数 ``x -> e`` (m,)、雅可比函数 ``x -> J`` (m, n)、初值 (n,)；
输出 :class:`GNSolution`（最优参数 (n,)、最终代价、迭代数、收敛标志）。
正规方程 ``(J^T W J + lambda I) dx = -J^T W e`` 在*矢量*参数化上求解；
住在流形上的状态（如位姿）由调用方处理：传入关于扰动的残差函数，每接受
一步后围绕 ``T <- exp(dx)^T`` 重新线性化（Eq. 8.7，流形更新）。
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["gauss_newton", "huber_weights", "GNSolution"]


@dataclass
class GNSolution:
    """:func:`gauss_newton` 的结果打包。

    Attributes:
        x: (n,) 优化后的参数矢量。
        cost: 最终（加权限）鲁棒代价 sum(w * e^2)。
        n_iters: 实际执行的迭代次数。
        converged: 是否在 ``tol`` 内收敛；False 也可能是耗尽 ``n_iters``。
    """

    x: np.ndarray            # 优化后的参数矢量 (n,)
    cost: float              # 最终的加权限鲁棒代价
    n_iters: int             # 实际执行的迭代次数
    converged: bool


def huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    """Huber 核的 IRLS 权重（教程第 08 章 §8.5，鲁棒代价）。

    ``w(r) = 1`` 若 ``|r| <= delta``，否则 ``delta / |r|`` —— 等价于在迭代
    重加权最小二乘（IRLS）下最小化 Huber 损失：大残差被降权，外点不再以
    平方代价主导正规方程。

    Args:
        residual: (m,) 或任意形状的残差数组，单位与残差本身一致。
        delta: Huber 阈值，与残差同单位；以内二次、以外线性。

    Returns:
        与 ``residual`` 同形状的权重，取值范围 (0, 1]。
    """
    r = np.abs(np.asarray(residual, dtype=float))
    return np.minimum(1.0, delta / np.maximum(r, 1e-12))


def gauss_newton(
    residual_fn,
    jacobian_fn,
    x0: np.ndarray,
    n_iters: int = 20,
    lm_lambda: float = 0.0,
    tol: float = 1e-10,
    robust_delta: float | None = None,
) -> GNSolution:
    """稠密 Gauss-Newton（``lm_lambda > 0`` 时为 Levenberg-Marquardt）。

    Args:
        residual_fn:   ``x -> e``，形状 (m,) —— 堆叠的残差矢量。
        jacobian_fn:   ``x -> J``，形状 (m, n) —— ``de/dx`` 在 ``x`` 处的值。
        x0:            (n,) 初始参数矢量（仅限矢量空间状态；位姿请在函数外
                       包一层流形更新）。
        n_iters:       最大迭代次数。
        lm_lambda:     LM 阻尼 ``lambda``（0 -> 纯 Gauss-Newton，Eq. 8.6）；
                       按代价是否被接受自适应缩放（Eq. 8.8 的思想）。
        tol:           代价下降量低于此值即停止。
        robust_delta:  若给定，每迭代施加 Huber IRLS 权重。

    Returns:
        :class:`GNSolution`。
    """
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    lam = float(lm_lambda)

    def cost_of(e: np.ndarray, w: np.ndarray) -> float:
        # 加权代价 sum(w_i * e_i^2)：Huber IRLS 下等价于 Huber 损失（教程 §8.5）。
        return float(np.sum(w * e * e))

    e = np.asarray(residual_fn(x), dtype=float).reshape(-1)
    w = (
        huber_weights(e, robust_delta)
        if robust_delta is not None
        else np.ones_like(e)
    )
    cost = cost_of(e, w)
    converged = False
    iters = 0
    for iters in range(1, n_iters + 1):
        # 依据：Eq. 8.5-8.6 —— 组装正规方程 H dx = -g，其中 H = J^T W J。
        J = np.asarray(jacobian_fn(x), dtype=float).reshape(len(e), -1)
        H = J.T @ (w[:, None] * J)
        g = J.T @ (w * e)
        # LM 阻尼项 lambda*I：保证正定、限制步长（依据：Eq. 8.8）。
        H_damped = H + lam * np.eye(H.shape[0])
        try:
            dx = -np.linalg.solve(H_damped, g)
        except np.linalg.LinAlgError:
            # H_damped 数值奇异时退化为最小二乘解。
            dx = -np.linalg.lstsq(H_damped, g, rcond=None)[0]
        x_new = x + dx
        e_new = np.asarray(residual_fn(x_new), dtype=float).reshape(-1)
        w_new = (
            huber_weights(e_new, robust_delta)
            if robust_delta is not None
            else np.ones_like(e_new)
        )
        cost_new = cost_of(e_new, w_new)
        if not np.isfinite(cost_new) or (lam > 0 and cost_new >= cost):
            # 依据：Eq. 8.8 —— 新代价未改善，拒绝该步并增大阻尼（更信任线性化）。
            lam = max(lam * 10.0, 1e-12)          # 拒绝：阻尼 x10
            if lam > 1e12:
                break
            continue
        if lam > 0:
            # 依据：Eq. 8.8 —— 该步被接受，减小阻尼（更接近纯 Gauss-Newton）。
            lam = max(lam / 10.0, 1e-12)          # 接受：阻尼 /10
        improvement = cost - cost_new
        x, e, w, cost = x_new, e_new, w_new, cost_new
        if improvement < tol:
            converged = True
            break
    else:
        converged = False  # n_iters 次迭代耗尽仍未达到 tol
    return GNSolution(x=x, cost=cost, n_iters=iters, converged=converged)

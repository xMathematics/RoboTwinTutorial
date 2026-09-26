"""SO(3)/SE(3) 流形上的刚体运动工具（全库共享的位姿数学基础）。

函数流水线
----------
本模块是教程代码库的最底层之一，被 ``epipolar``（对极几何）、``preint``
（IMU 预积分）、``vins``（视觉惯性紧耦合）、``direct``、``droidlite``、
``photoba`` 等模块复用（位姿更新 ``T <- exp(dx)^T``、点云变换、李代数扰动
传播都靠它）；``fastslam`` 不依赖本模块。本模块自身只依赖 numpy。

实现教程第 02 章（三维刚体运动：旋转与位姿）推导的全部工具：hat/vee、
Rodrigues 公式（Eq. 2.15-2.19）、闭式对数映射、左雅可比及其逆（Eq. 2.20-2.24，
BCH 一阶项）、SE(3) 指数/对数映射、点集刚体变换。

输入/输出：输入旋转矢量 ``phi`` (3,)（单位 rad）、位姿 ``T`` (4,4)（旋转
无量纲、平移单位 m）、点集 (N,3)（单位 m）；输出对应形状的 numpy 数组。

约定
----
- 旋转矩阵 ``R`` 是形状 (3, 3) 的 numpy 数组，作用为 ``x_cam = R @ x_world``
  （world-to-camera 方向）。
- 李代数元素是 3 维矢量 ``phi``；``phi_hat = hat(phi)`` 是反对称矩阵，满足
  ``phi_hat @ v == cross(phi, v)``（Eq. 2.13-2.14）。
- ``se3_exp(xi)`` 接受 ``xi = (tau, phi)`` 形状 (6,)，其中平移部分 ``tau``
  是"雅可比前"量：``t = Jl(phi) @ tau``（Murray et al. 1994, Sec. 2.4；
  视觉 SLAM 十四讲 Eq. 4.24 采用同一约定）。
"""
import numpy as np

__all__ = [
    "hat",
    "vee",
    "so3_exp",
    "so3_log",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse",
    "se3_exp",
    "se3_log",
    "transform_points",
]

_SMALL = 1e-8  # 小角度阈值（rad）：低于它走级数分支，避免 sin(theta)/theta 的 0/0 数值病态


def hat(phi: np.ndarray) -> np.ndarray:
    """R^3 矢量 -> 反对称矩阵（Eq. 2.14）。

    把旋转矢量写成矩阵形式，使 ``phi_hat @ v`` 等价于叉积 ``phi x v``，
    是一切李代数计算的入口。

    Args:
        phi: (3,) 旋转矢量，单位 rad。

    Returns:
        (3, 3) 反对称矩阵 ``phi_hat``，无量纲。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    x, y, z = phi
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(phi_hat: np.ndarray) -> np.ndarray:
    """反对称矩阵 -> 矢量（:func:`hat` 的逆运算）。

    从反对称矩阵的三个独立元素各取半差恢复 (3,) 矢量，单位 rad。
    """
    m = np.asarray(phi_hat, dtype=float)
    return np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) * 0.5


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues 公式：计算 ``exp(phi_hat)``（Eq. 2.15-2.19）。

    ``exp(phi_hat) = I + sin(theta)/theta * phi_hat
                     + (1 - cos(theta))/theta^2 * phi_hat^2``，
    依据 ``(phi_hat)^3 = -theta^2 * phi_hat``（Eq. 2.16）把矩阵指数的无穷
    级数收拢为三项闭式。

    Args:
        phi: (3,) 旋转矢量，单位 rad；``theta = |phi|`` 为转角。

    Returns:
        (3, 3) 旋转矩阵，无量纲。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < _SMALL:
        # 小角度级数分支：sin(theta)/theta -> 1、(1-cos)/theta^2 -> 1/2，
        # 只保留前两项避免三角系数的 0/0（依据：Eq. 2.15 级数截断）。
        return np.eye(3) + K + K @ K * 0.5
    return (
        np.eye(3)
        + np.sin(theta) / theta * K
        + (1.0 - np.cos(theta)) / (theta * theta) * (K @ K)
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """闭式对数映射：``log(R)`` -> 旋转矢量。

    ``0 < theta < pi`` 时 ``phi = theta / (2 sin theta) * vee(R - R^T)``；
    ``theta`` 极小时退化为级数 ``vee(R - R^T)/2``；``theta ~ pi`` 时
    sin(theta) -> 0，改由对角元提取转轴（视觉 SLAM 十四讲 Eq. 4.19-4.22）。
    """
    R = np.asarray(R, dtype=float)
    # 转角由迹恢复：trace(R) = 1 + 2*cos(theta)（依据：教程第 02 章 Eq. 2.5 附近）。
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < _SMALL:
        return vee(R - R.T) * 0.5
    if np.pi - theta < 1e-4:
        # theta 接近 pi：sin(theta) -> 0 使 vee(R - R^T) 的系数发散，
        # 改从主对角元提取转轴（依据：视觉 SLAM 十四讲 Eq. 4.21-4.22）。
        A = np.where(np.abs(np.diag(R) + 1.0) > 1e-6)[0]
        if A.size == 0:
            return np.zeros(3)
        i = int(A[0])
        axis = R[i] + R[:, i]
        axis[i] -= 1.0
        axis = axis / np.linalg.norm(axis)
        return axis * np.pi
    return vee(R - R.T) * (theta / (2.0 * np.sin(theta)))


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """SO(3) 左雅可比（Eq. 2.20-2.22）。

    ``Jl = I + (1 - cos t)/t^2 * K + (t - sin t)/t^3 * K^2``，其中
    ``K = hat(phi)``、``t = |phi|``；``t -> 0`` 时退化为级数
    ``I + K/2 + K^2/6``。用于 SE(3) 指数映射的平移部分 ``t = Jl @ tau``，
    以及 ``preint`` 等模块的 BCH 一阶扰动传播。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < 1e-6:
        # 小角度级数分支（依据：Eq. 2.21 系数的泰勒展开）。
        return np.eye(3) + 0.5 * K + K @ K / 6.0
    st, ct = np.sin(theta), np.cos(theta)
    A = (1.0 - ct) / (theta * theta)
    B = (theta - st) / (theta**3)
    return np.eye(3) + A * K + B * (K @ K)


def so3_left_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """左雅可比的逆（BCH 一阶近似，Eq. 2.22-2.24）。

    ``Jl^{-1} = I - K/2 + (1/t^2 - (1 + cos t)/(2 t sin t)) * K^2``；
    ``t -> 0`` 时标量系数趋于 ``1/12 + t^2/720``（洛必达展开，避免 0/0）。
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < 1e-4:
        # 小角度级数分支：c2 = 1/12 + t^2/720 + O(t^4)。
        c2 = 1.0 / 12.0 + theta * theta / 720.0
        return np.eye(3) - 0.5 * K + c2 * (K @ K)
    st, ct = np.sin(theta), np.cos(theta)
    c2 = 1.0 / (theta * theta) - (1.0 + ct) / (2.0 * theta * st)
    return np.eye(3) - 0.5 * K + c2 * (K @ K)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) 指数映射：``xi = (tau, phi)`` -> 齐次变换矩阵。

    ``T = [[R, t], [0, 1]]``，其中 ``R = so3_exp(phi)``、``t = Jl(phi) @ tau``。

    Args:
        xi: (6,) 李代数矢量 ``(tau, phi)``，``tau`` 单位 m、``phi`` 单位 rad。

    Returns:
        (4, 4) 齐次变换矩阵（旋转无量纲、平移单位 m）。
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    tau, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = so3_left_jacobian(phi) @ tau
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) 对数映射：齐次变换矩阵 -> ``xi = (tau, phi)``，形状 (6,)。

    先对旋转块取 ``phi = so3_log(R)``，再左乘 ``Jl^{-1}`` 恢复"雅可比前"
    平移 ``tau = Jl^{-1}(phi) @ t``——恰为 :func:`se3_exp` 的逆。
    """
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    phi = so3_log(R)
    tau = so3_left_jacobian_inverse(phi) @ t
    return np.concatenate([tau, phi])


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """用 (4, 4) 齐次变换矩阵变换 (N, 3) 点集。

    Args:
        T: (4, 4) 齐次变换矩阵（:func:`se3_exp` 的输出）。
        points: (N, 3) 点集，单位 m。

    Returns:
        (N, 3) 变换后的点集 ``p' = R p + t``，单位 m。
    """
    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]

"""平面 2 连杆双臂的运动学（FK / IK / 雅可比）——迷你基准的机器人本体层。

函数流水线
----------
本模块是 ``projects/robotwin`` 的最底层：``tasks.py`` 用真值连杆长做
FK 演化世界并把关节伺服到目标；``policies.py`` 用自己的模型连杆长解
IK 把"想要的末端点位"翻译成关节目标（标定失准就从这一步进入）。
依赖方向：本模块不依赖库内其他模块（仅 numpy + 标准库）；
``tasks / policies → 本模块 → dr.WorldParams（仅取连杆长字段）``。

运动学约定（与教程第 02 章 §2.1 的关节/连杆模型一致）
----------------------------------------------------
- 每条臂 = 2 个旋转关节的平面开链（平面 2R 臂），基座固定在桌面系：
  左肩 ``BASES[0] = (0, +0.25)``、右肩 ``BASES[1] = (0, -0.25)``（单位 m），
  双臂 = 两条同构臂共享桌面坐标系（教程 §2.2 的双臂布局的 2D 化简）。
- 正运动学（FK，教程 (2.1) 的平面版）：
  ``x = bx + L1·cos(q1) + L2·cos(q1+q2)``，``y = by + L1·sin(q1) + L2·sin(q1+q2)``。
- 逆运动学（IK）：由余弦定理解出 ``q2`` 的两支（肘向 elbow ±1），再回代
  得 ``q1``（几何法双解，Siciliano et al. 2009 §2.13；教程 §2.1.2 ② 的
  "对候选位姿解 IK 做可达性初筛"即此操作）。
- 关节限位：``q1 ∈ [-1.2, 2.4] rad``、``q2 ∈ [-2.6, -0.15] rad``——限位
  天然只在肘向 ``elbow=-1`` 的一支内可行（2D 世界的"肘部工作姿态"），
  ``elbow=+1`` 一支保留在 :func:`ik_both` 供教学对照与测试。

输入/输出：输入关节角 (2,) 或双臂状态 (4,)（单位 rad）、目标点 (2,)
（单位 m，桌面系）；输出末端位置 (2,) / 双臂末端 (2, 2)、关节解 (2,)
或双臂 (4,)、雅可比 (2, 2)（单位 m/rad）。
"""
from __future__ import annotations

import numpy as np

from dr import NOMINAL_LINK_LEFT, NOMINAL_LINK_RIGHT

__all__ = [
    "LEFT",
    "RIGHT",
    "BASES",
    "Q1_LIMITS",
    "Q2_LIMITS",
    "link_fk",
    "link_ik",
    "ik_both",
    "link_jacobian",
    "numerical_jacobian",
    "BimanualArm2D",
]

#: 臂侧索引：0 = 左臂，1 = 右臂（双臂状态向量的分块顺序）。
LEFT: int = 0
RIGHT: int = 1
#: 双臂基座（肩部）在桌面系的位置，单位 m；布局 = 教程 §2.2 双臂的 2D 化简。
BASES: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.25), (0.0, -0.25))
#: 关节限位 (下, 上)，单位 rad；q2 上限为负 ⇒ 只有 elbow=-1 一支可行。
#: q2 下限 −2.9（≈ −166°，深弯肘）是 2D 教学约定：允许臂在肩前小半径
#: 区域工作（无自碰撞模型，深弯无代价）；−2.6 会让拖曳式推物的 IK 在
#: r ≈ 0.21 m 内圈无解。
Q1_LIMITS: tuple[float, float] = (-1.2, 2.4)
Q2_LIMITS: tuple[float, float] = (-2.9, -0.15)


def link_fk(
    links: tuple[float, float],
    q: np.ndarray,
    base: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """单臂正运动学 FK：关节角 → 末端位置（教程 (2.1) 的平面版）。

    解析式（依据：平面开链逐节旋转叠加）：
    ``x = bx + L1·cos(q1) + L2·cos(q1+q2)``，
    ``y = by + L1·sin(q1) + L2·sin(q1+q2)``。

    Args:
        links: (L1, L2) 两节连杆长，单位 m。
        q: (2,) 关节角 (q1, q2)，单位 rad。
        base: 基座（肩部）位置，单位 m，默认原点。

    Returns:
        (2,) 末端位置 (x, y)，单位 m。
    """
    q = np.asarray(q, dtype=float).reshape(2)
    l1, l2 = float(links[0]), float(links[1])
    bx, by = float(base[0]), float(base[1])
    return np.array(
        [
            bx + l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
            by + l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]),
        ]
    )


def ik_both(
    links: tuple[float, float],
    target: np.ndarray,
    base: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """单臂解析 IK 的原始双解（不做限位检查）——肘向 +1 与 -1 两支。

    几何法（依据：余弦定理 + 二连杆三角形的内/外角关系，Siciliano et
    al. 2009 §2.13；教程 §2.1.2 ②）：
    1. 目标相对基座的半径 ``r = ‖target − base‖``；
    2. ``cos(q2) = (r² − L1² − L2²) / (2·L1·L2)``——|cos q2| > 1 时目标
       在可达工作空间外（两支都无解）；
    3. ``q2 = ±arccos(cos q2)``（肘向两支），回代
       ``q1 = atan2(y, x) − atan2(L2·sin q2, L1 + L2·cos q2)``。

    Args:
        links: (L1, L2) 两节连杆长，单位 m。
        target: (2,) 目标末端位置，单位 m（桌面系）。
        base: 基座位置，单位 m。

    Returns:
        ``(q_elbow_up, q_elbow_down)``：两支关节解各 (2,)（单位 rad）；
        目标不可达时两支均为 ``None``。
    """
    target = np.asarray(target, dtype=float).reshape(2)
    l1, l2 = float(links[0]), float(links[1])
    dx = target[0] - base[0]
    dy = target[1] - base[1]
    r_sq = dx * dx + dy * dy
    cos_q2 = (r_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cos_q2 < -1.0 or cos_q2 > 1.0:
        # 可达工作空间之外（依据：|cos q2| ≤ 1 是两圆相交的必要条件）。
        return None, None
    solutions: list[np.ndarray | None] = []
    for elbow in (1.0, -1.0):  # +1 = q2 取正的一支，-1 = q2 取负的一支
        q2 = elbow * np.arccos(cos_q2)
        # 回代 q1：目标方位角减去"基座→腕心"的内角（依据：几何法第二步）。
        q1 = np.arctan2(dy, dx) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        solutions.append(np.array([q1, q2]))
    return solutions[0], solutions[1]


def link_ik(
    links: tuple[float, float],
    target: np.ndarray,
    base: tuple[float, float] = (0.0, 0.0),
    elbow: float = -1.0,
    joint_limits: bool = True,
) -> np.ndarray | None:
    """单臂解析 IK：选肘向 + 过限位检查（策略与环境共用的入口）。

    Args:
        links: (L1, L2) 两节连杆长，单位 m。
        target: (2,) 目标末端位置，单位 m。
        base: 基座位置，单位 m。
        elbow: 肘向符号，``+1`` 或 ``-1``（默认 -1：与 :data:`Q2_LIMITS`
            的负值约定一致，是本基准的实际工作肘向）。
        joint_limits: True 时对解做 :data:`Q1_LIMITS` / :data:`Q2_LIMITS`
            检查，超限返回 ``None``（依据：教程 (2.2)——IK 可行 = 有解
            且不撞限位）。

    Returns:
        (2,) 关节角 (q1, q2)（单位 rad）；不可达或超限位时 ``None``。
    """
    plus, minus = ik_both(links, target, base)
    sol = plus if elbow > 0 else minus
    if sol is None:
        return None
    if joint_limits:
        if not (Q1_LIMITS[0] <= sol[0] <= Q1_LIMITS[1]):
            return None
        if not (Q2_LIMITS[0] <= sol[1] <= Q2_LIMITS[1]):
            return None
    return sol


def link_jacobian(
    links: tuple[float, float],
    q: np.ndarray,
) -> np.ndarray:
    """单臂解析雅可比 ``J = ∂(x, y)/∂(q1, q2)``，形状 (2, 2)。

    对 FK 解析求导（依据：链式法则逐项求导）：
    ``J = [[−L1·s1 − L2·s12, −L2·s12], [L1·c1 + L2·c12, L2·c12]]``，
    其中 ``s1 = sin q1``、``s12 = sin(q1+q2)``。用于数值-解析交叉验证与
    教学展示（速度映射 ``ẋ = J·q̇``，教程 §2.1.2 第三步的奇异位形讨论
    即 J 降秩）。

    Args:
        links: (L1, L2) 两节连杆长，单位 m。
        q: (2,) 关节角，单位 rad。

    Returns:
        (2, 2) 雅可比矩阵，单位 m/rad。
    """
    q = np.asarray(q, dtype=float).reshape(2)
    l1, l2 = float(links[0]), float(links[1])
    s1, c1 = np.sin(q[0]), np.cos(q[0])
    s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
    return np.array(
        [
            [-l1 * s1 - l2 * s12, -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12],
        ]
    )


def numerical_jacobian(
    fun,
    q: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """中心差分数值雅可比——与 :func:`link_jacobian` 交叉验证的参照实现。

    ``J[:, i] ≈ (f(q + eps·e_i) − f(q − eps·e_i)) / (2·eps)``（依据：中心
    差分截断误差 O(eps²)，eps=1e-6 时总误差 ~1e-9，远小于断言容差 1e-6）。

    Args:
        fun: 可调用 ``f(q: (2,)) -> (2,)``（如 FK）。
        q: (2,) 求导点。
        eps: 差分步长，默认 1e-6。

    Returns:
        (2, 2) 数值雅可比，行 = 输出分量，列 = 输入分量。
    """
    q = np.asarray(q, dtype=float).reshape(2)
    jac = np.zeros((2, 2))
    for i in range(2):
        q_p = q.copy()
        q_m = q.copy()
        q_p[i] += eps
        q_m[i] -= eps
        jac[:, i] = (np.asarray(fun(q_p)) - np.asarray(fun(q_m))) / (2.0 * eps)
    return jac


class BimanualArm2D:
    """双臂 = 左右两条固定基座的平面 2 连杆臂，状态 = 双臂关节角 (4,)。

    布局（教程 §2.2 双臂的 2D 化简）：左肩 :data:`BASES`[0] = (0, +0.25)、
    右肩 = (0, -0.25)，两条臂同构、共享桌面坐标系。状态向量
    ``q = [qL1, qL2, qR1, qR2]``（单位 rad）按左右分块。

    Attributes:
        links: ((L1, L2), (L1, L2)) 左/右臂连杆长，单位 m（来自
            :class:`dr.WorldParams`，即"世界真实运动学"）。
        q: (4,) 双臂关节角状态，单位 rad。
    """

    def __init__(
        self,
        left_link: tuple[float, float] = NOMINAL_LINK_LEFT,
        right_link: tuple[float, float] = NOMINAL_LINK_RIGHT,
        q0: np.ndarray | None = None,
    ) -> None:
        """初始化双臂运动学模型与状态。

        Args:
            left_link: (L1, L2) 左臂连杆长，单位 m；默认名义值。
            right_link: (L1, L2) 右臂连杆长，单位 m；默认名义值。
            q0: (4,) 初始双臂关节角，单位 rad；默认每臂 ``(0.9, -1.8)``
                （收拢的"home"姿态，末端位于 (≈0.50, ±0.33) m）。
        """
        self.links: tuple[tuple[float, float], tuple[float, float]] = (
            tuple(float(v) for v in left_link),
            tuple(float(v) for v in right_link),
        )
        if q0 is None:
            q0 = np.array([0.9, -1.8, 0.9, -1.8])
        self.q = np.asarray(q0, dtype=float).reshape(4)

    def get_state(self) -> np.ndarray:
        """返回双臂关节角状态 (4,) = [qL1, qL2, qR1, qR2]，单位 rad。"""
        return self.q.copy()

    def set_state(self, q: np.ndarray) -> None:
        """就地设置双臂关节角状态。

        Args:
            q: (4,) 关节角 [qL1, qL2, qR1, qR2]，单位 rad。
        """
        self.q = np.asarray(q, dtype=float).reshape(4).copy()

    def _slice(self, side: int, q: np.ndarray | None) -> np.ndarray:
        """取某一臂的 (2,) 关节角分块（q 为 None 时用当前状态）。"""
        qq = self.q if q is None else np.asarray(q, dtype=float).reshape(4)
        return qq[2 * side: 2 * side + 2]

    def fk(self, side: int, q: np.ndarray | None = None) -> np.ndarray:
        """某一臂的末端位置 (2,)，单位 m。

        Args:
            side: 臂侧索引，:data:`LEFT`(0) 或 :data:`RIGHT`(1)。
            q: (4,) 双臂状态；``None`` 用当前状态。
        """
        return link_fk(self.links[side], self._slice(side, q), BASES[side])

    def end_effectors(self, q: np.ndarray | None = None) -> np.ndarray:
        """双臂末端位置 (2, 2)，行 = [左, 右]，列 = (x, y)，单位 m。

        Args:
            q: (4,) 双臂状态；``None`` 用当前状态。
        """
        return np.stack([self.fk(LEFT, q), self.fk(RIGHT, q)], axis=0)

    def ik(
        self,
        side: int,
        target: np.ndarray,
        elbow: float = -1.0,
        joint_limits: bool = True,
    ) -> np.ndarray | None:
        """某一臂的解析 IK：目标点 → (2,) 关节角；不可达/超限位为 None。

        Args:
            side: 臂侧索引。
            target: (2,) 目标末端位置，单位 m（桌面系）。
            elbow: 肘向符号（默认 -1 = 工作肘向）。
            joint_limits: 是否做限位检查（默认 True）。
        """
        return link_ik(self.links[side], target, BASES[side], elbow, joint_limits)

    def jacobian(self, side: int, q: np.ndarray | None = None) -> np.ndarray:
        """某一臂的解析雅可比 (2, 2)，单位 m/rad。

        Args:
            side: 臂侧索引。
            q: (4,) 双臂状态；``None`` 用当前状态。
        """
        return link_jacobian(self.links[side], self._slice(side, q))

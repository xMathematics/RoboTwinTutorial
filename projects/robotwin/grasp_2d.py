"""2D 平行夹爪的力闭合抓取质量评估与候选排序——DexGraspNet 的教学代理。

函数流水线
----------
DexGraspNet（Wang et al., NeurIPS 2022, arXiv:2210.02697；本地精读
[tutorials/robotwin/精读/DexGraspNet_NeurIPS2022.md](../../tutorials/robotwin/精读/DexGraspNet_NeurIPS2022.md)；
教程第 09 章 §9.2 阅读清单、[3D 重建第 10 章 §10.3](../../tutorials/3d_reconstruction/10_机器人场景中的重建实战与选型.md)
式 (10.4)(10.5)）用"可微力闭合能量优化 + 物理校验筛选"合成 1.32M 灵巧手
抓取；其质量评估的解析核心是**力闭合（force closure）判据**与
**Ferrari–Canny L1 质量**（精读式 (1) 的判据定位与式 (7) 的 $Q_1$）。
本模块把这套"候选生成 → 力闭合判据 → 质量排序"的思想落到 **2D 平行
夹爪**的最小可跑版本。

调用链：:class:`Disk` / :class:`ConvexPolygon`（凸物体：接触点 + 内法向）→
:class:`Grasp2D`（候选抓取 = 接触点对 + 拟合轴）→
:func:`friction_wrenches`（单接触摩擦锥的 elementary wrench 集合）→
:func:`wrench_hull_analysis`（wrench 凸包的支撑平面枚举：原点内点判定 +
L1 质量）→ :func:`is_force_closure` / :func:`grasp_quality` →
:func:`polygon_grasp_candidates` / :func:`disk_grasp_candidates`（网格系统
采样候选）→ :func:`rank_candidates` / :func:`best_grasp`（按质量排序取
最优）；:func:`l1_by_support_sampling` 是 L1 的独立交叉验证口径（支撑
函数方向采样）。被 ``tests/test_grasp_2d.py`` 驱动；与 ``tasks.py``
pick_place 的联动 sanity（最优抓取点满足吸附条件）也在该测试里。
依赖方向：本模块只依赖 numpy（不 import 库内其他模块）——判据是纯几何。

理论依据（docstring 推导，式号供教程/精读回引）
----------------------------------------------
- **力闭合判据（2D 形式）**：接触点 $p_i$ 沿摩擦锥内方向 $f_i$（与内法向
  夹角 $\\le \\phi = \\arctan\\mu$）施力，产生的 2D wrench（力旋量）为
  $w_i = (f_i,\\ p_i \\times f_i) \\in \\mathbb{R}^3$（第三维 = 绕原点力矩）。
  力闭合 $\\iff$ 原点是各接触 elementary wrench 集合并集凸包的**内点**：
  $0 \\in \\mathrm{int}\\,\\mathrm{Conv}(\\bigcup_i \\mathcal{W}_i)$——即
  存在一组锥内力使合力与合力矩同时为零、且扰动可阻。这正是 Ferrari &
  Canny 1992 判据（精读 §4.3：存在性判据 $0 \\in \\mathrm{int}\\,
  \\mathrm{Conv}(\\bigcup_i \\mathcal{W}_i)$，即 3D 重建教程式 (10.4)
  的 2D 形式），也是 DexGraspNet 式 (1) 力闭合能量 $E_{fc} = \\lVert Gc
  \\rVert_2$ 的 0/1 判据版（(1) 是它的可微松弛，精读 §4.3）。
  两接触情形的解析特例即 Nguyen 1988：**接触点连线必须落在两个摩擦锥
  内**（对径 antipodal——法向共线反向——是摩擦最充裕的特例，也是圆盘
  上 L1 最大的构型）；偏移对径抓取的力闭合边界可手算：连线倾角
  $\\arctan(c/2a) \\le \\phi = \\arctan\\mu$（$c$ = 偏移量、$2a$ = 接触
  距），测试里有该边界的两侧用例。
- **L1 质量（Ferrari–Canny）**：$Q_1$ = 以原点为中心、能放进 wrench
  凸包的最大球半径 = 原点到凸包各支撑平面距离的最小值（精读式 (7)：
  $Q_1$ = wrench 凸包的内切球半径，即 3D 重建教程式 (10.5)）。凸包由
  半空间交给出 ⇒ 内切半径 = 支撑平面距离的最小值（沿棱/顶点相切的
  支撑平面距离 ≥ 内切半径，故取 min 恰为 $Q_1$）。DexGraspNet 只用
  $Q_1$ 做**评测**（原文 §IV：生成管线不显式优化它）；本模块同口径：
  判据/质量只用于候选排序。

与原文的差异（教学化简，皆有意识为之）
------------------------------------
1. **手与维度**：ShadowHand 28 维位姿 $(T, R, \\theta)$ + 140 接触候选
   → 2D 平行夹爪：候选 = 一对接触点 + 拟合轴（夹爪闭合轴 = 两接触点
   连线），6 参数的抓取假设化简为"点对"。
2. **合成方式**：原文是可微能量梯度优化 6000 步（式 (3)(6)）；这里是
   网格系统采样候选 + 判据排序——对应其管线"**大批生成—物理校验筛选**"
   （教程 04 章 §4.3 的同构概括）里的"筛"端，不实现可微优化端。
3. **校验协议**：原文用 Isaac Gym 6 个重力方向 × 100 步物理仿真；
   这里用力闭合内点判据本身做解析校验（2D 无重力方向的歧义，判据即
   充要），摩擦锥用 $m$ 条射线离散（原文 3D 锥为连续约束）。

确定性约定：全部函数纯确定性（无随机数——网格候选；原文的随机初始化/
采样由"系统网格"替代），同输入同输出逐位一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "N_RAYS",
    "Disk",
    "ConvexPolygon",
    "Grasp2D",
    "friction_wrenches",
    "wrench_hull_analysis",
    "grasp_wrenches",
    "is_force_closure",
    "grasp_quality",
    "l1_by_support_sampling",
    "polygon_grasp_candidates",
    "disk_grasp_candidates",
    "rank_candidates",
    "best_grasp",
]

#: 每个接触摩擦锥的离散射线数 m（锥角 $\\phi = \\arctan\\mu$ 内均匀栅格；
#: 教学离散化——3D 原文的锥为连续约束；2 接触 × m 射线 = 2m 个
#: elementary wrench，m=8 时凸包枚举 O(m⁴) ≈ 微秒级）。
N_RAYS: int = 8

#: 支撑平面判定的数值容差（wrench 量级 ~0.1-1，绝对容差即可）。
_HULL_TOL: float = 1e-9


# ---------------------------------------------------------------- 物体 --
@dataclass(frozen=True)
class Disk:
    """圆盘物体（2D）：圆心 + 半径；边界点法向 = 径向。

    Attributes:
        center: (2,) 圆心，单位 m（桌面系）。
        radius: 半径，标量，单位 m，> 0。
    """

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        # 构造即校验（与 dr.WorldParams 同款"早失败"约定）。
        if self.center.shape != (2,):
            raise ValueError(f"center 须是 (2,)，当前 {self.center.shape}")
        if not (float(self.radius) > 0.0):
            raise ValueError(f"radius 须 > 0，当前 {self.radius}")

    def contact(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """边界上方位角 theta 处的接触点与**内**法向（指向圆心，径向）。

        Args:
            theta: 方位角，rad（0 = +x 方向）。

        Returns:
            ``(point (2,), normal (2,))``，单位 m / 单位矢量。
        """
        radial = np.array([np.cos(theta), np.sin(theta)])
        point = np.asarray(self.center, dtype=float) + float(self.radius) * radial
        return point, -radial  # 内法向 = 径向反向（指向圆心）


@dataclass(frozen=True)
class ConvexPolygon:
    """凸多边形物体（2D）：顶点按**逆时针（CCW）**序给出。

    构造即校验：顶点数 ≥ 3、有向面积为正（CCW）、逐边叉积 ≥ 0（凸），
    违反抛 ValueError（早失败优于在判据里吃到翻折的几何）。

    Attributes:
        vertices: (n, 2) 顶点，单位 m，CCW 序。
    """

    vertices: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
            raise ValueError(f"vertices 须是 (n≥3, 2)，当前 {v.shape}")
        area2 = float(np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                             - v[:, 1] * np.roll(v[:, 0], -1)))
        if area2 <= 0.0:
            raise ValueError("顶点须按逆时针（CCW）序给出（有向面积 > 0）")
        edges = np.roll(v, -1, axis=0) - v  # (n, 2) 逐边向量
        crosses = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if np.any(crosses < -1e-12):
            raise ValueError("顶点不构成凸多边形（存在左转/凹角）")

    def contact(self, edge: int, u: float) -> tuple[np.ndarray, np.ndarray]:
        """第 edge 条边上参数 u ∈ [0, 1] 处的接触点与**内**法向。

        CCW 多边形的边向量 d = (dx, dy) 的内法向 = d 左转 90° =
        (−dy, dx)/‖d‖（依据：CCW 环绕时内部在边的左侧）。

        Args:
            edge: 边序号 0..n−1（第 i 条边 = vertices[i] → vertices[i+1]）。
            u: 边上参数（0 = 起点，1 = 终点）。

        Returns:
            ``(point (2,), normal (2,))``，单位 m / 单位矢量。
        """
        v = np.asarray(self.vertices, dtype=float)
        a, b = v[edge % len(v)], v[(edge + 1) % len(v)]
        d = b - a
        point = a + float(u) * d
        normal = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-12)
        return point, normal


# ---------------------------------------------------------------- 抓取 --
@dataclass(frozen=True)
class Grasp2D:
    """一个候选抓取 = 接触点对 + 两点的**内**法向（DexGraspNet 抓取位姿
    $g = (T, R, \\theta)$ 的 2D 平行夹爪化简：闭合轴 + 张开宽度即可确定）。

    Attributes:
        points: (2, 2) 两接触点，行 = [p1, p2]，单位 m。
        normals: (2, 2) 两接触的内法向（单位矢量，指向物体内部）。
    """

    points: np.ndarray
    normals: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.points).shape != (2, 2) or np.asarray(self.normals).shape != (2, 2):
            raise ValueError("points / normals 须是 (2, 2)")
        nrm = np.linalg.norm(np.asarray(self.normals, dtype=float), axis=1)
        if np.any(np.abs(nrm - 1.0) > 1e-9):
            raise ValueError("内法向必须是单位矢量")

    @property
    def axis(self) -> np.ndarray:
        """拟合轴（夹爪闭合轴）的单位方向 (2,) = (p2 − p1)/‖·‖。

        依据：平行夹爪沿该轴对合，两接触点必须落在爪指接触面上——
        "接触点对 + 拟合轴"即 2D 抓取假设的全部内容。
        """
        d = self.points[1] - self.points[0]
        return d / max(float(np.linalg.norm(d)), 1e-12)

    @property
    def width(self) -> float:
        """张开宽度 = ‖p2 − p1‖，单位 m（对应平行夹爪的开口量）。"""
        return float(np.linalg.norm(self.points[1] - self.points[0]))


# ------------------------------------------------------------ 力旋量 --
def friction_wrenches(point: np.ndarray, normal: np.ndarray, mu: float,
                      n_rays: int = N_RAYS) -> np.ndarray:
    """单接触摩擦锥的 elementary wrench 集合 (n_rays, 3)。

    力方向 = 内法向绕 $\\theta \\in $ [−φ, +φ] 均匀栅格旋转（φ = arctan μ，
    库仑摩擦锥的 2D 离散），幅值取 1；wrench $w = (f,\\ p \\times f)$，
    第三维 = 绕原点力矩 $p_x f_y − p_y f_x$（精读 §4.2 的
    $w_i = (c_i, x_i \\times c_i)$ 的 2D 形式）。

    Args:
        point: (2,) 接触点，单位 m。
        normal: (2,) 内法向（单位矢量）。
        mu: 摩擦系数 μ ≥ 0（无量纲）；μ = 0 → 全部射线与法向重合。
        n_rays: 锥内射线数 m（≥ 2）。

    Returns:
        (n_rays, 3) wrench 数组（行 = 各射线）。
    """
    if mu < 0.0:
        raise ValueError(f"摩擦系数须 ≥ 0，当前 {mu}")
    point = np.asarray(point, dtype=float).reshape(2)
    normal = np.asarray(normal, dtype=float).reshape(2)
    phi = float(np.arctan(mu))
    theta = np.linspace(-phi, phi, int(n_rays))  # (m,) 锥内栅格
    c, s = np.cos(theta), np.sin(theta)
    fx = c * normal[0] - s * normal[1]  # R(θ)·n 的 x 分量
    fy = s * normal[0] + c * normal[1]  # R(θ)·n 的 y 分量
    tau = point[0] * fy - point[1] * fx  # p × f（2D 叉积的标量）
    return np.stack([fx, fy, tau], axis=1)


def grasp_wrenches(grasp: Grasp2D, mu: float, n_rays: int = N_RAYS) -> np.ndarray:
    """一个抓取两接触的 elementary wrench 并集 (2·n_rays, 3)。"""
    return np.vstack([
        friction_wrenches(grasp.points[i], grasp.normals[i], mu, n_rays)
        for i in (0, 1)
    ])


def wrench_hull_analysis(wrenches: np.ndarray) -> tuple[bool, float]:
    """wrench 凸包的力闭合判定 + L1 质量（支撑平面枚举，纯 numpy）。

    原理（见模块 docstring 的判据推导）：2m 个 elementary wrench 的凸包
    由其**支撑平面**（半空间表示）完全决定——枚举全部不共面三元组
    （m ≤ 16 时 C(16,3) = 560 个，O(m⁴) 教学规模足够），满足"其余点全部
    在其一侧"的三元组即支撑平面；原点严格在凸包内 $\\iff$ 每个支撑
    平面都把原点包在内侧（有向距离 > 0）；L1 = 距离的最小值（内切球
    半径，Ferrari–Canny $Q_1$；沿棱/顶点相切的支撑平面距离 ≥ 内切半径，
    故 min 不受非面元支撑平面影响）。

    Args:
        wrenches: (m, 3) elementary wrench 集合（行可任意重复/乱序）。

    Returns:
        ``(inside, l1)``：inside = 原点是否为凸包**内点**（力闭合）；
        l1 = 内切半径（力闭合不成立时返回 0.0）。
    """
    w = np.asarray(wrenches, dtype=float).reshape(-1, 3)
    m = w.shape[0]
    triples = np.array(list(combinations(range(m), 3)), dtype=int)  # (T, 3)
    if triples.size == 0:
        return False, 0.0
    a = w[triples[:, 0]]  # (T, 3) 平面过点
    nu = np.cross(w[triples[:, 1]] - a, w[triples[:, 2]] - a)  # (T, 3) 法向
    nn = np.linalg.norm(nu, axis=1)  # (T,)
    ok = nn > 1e-12  # 滤掉共面/重复点的退化三元组
    if not np.any(ok):
        return False, 0.0
    nu_hat = nu / np.where(ok, nn, 1.0)[:, None]
    # 全部点相对平面的有符号距离（沿 +ν̂ 为正）：(T, m)。
    s = ((w[None, :, :] - a[:, None, :]) * nu_hat[:, None, :]).sum(axis=2)
    case_plus = ok & (s.min(axis=1) >= -_HULL_TOL)  # 凸包在 +ν̂ 侧
    case_minus = ok & (s.max(axis=1) <= _HULL_TOL)  # 凸包在 −ν̂ 侧
    an = (a * nu_hat).sum(axis=1)  # (T,) a·ν̂ = 原点到平面的带号距离
    # "原点在内侧"的距离：case_plus 时内侧 = +ν̂ 方向 → d = −a·ν̂；反之 +a·ν̂。
    d_in = np.where(case_plus, -an, an)
    supporting = case_plus | case_minus
    if not np.any(supporting):
        return False, 0.0
    d_support = d_in[supporting]
    if not bool(d_support.min() > _HULL_TOL):
        return False, 0.0  # 有支撑平面切过/越过原点 → 非内点（力闭合不成立）
    return True, float(d_support.min())


def is_force_closure(grasp: Grasp2D, mu: float, n_rays: int = N_RAYS) -> bool:
    """抓取在摩擦系数 μ 下是否力闭合（原点在摩擦锥 wrench 凸包内）。"""
    inside, _l1 = wrench_hull_analysis(grasp_wrenches(grasp, mu, n_rays))
    return inside


def grasp_quality(grasp: Grasp2D, mu: float, n_rays: int = N_RAYS) -> float:
    """抓取的 L1 质量（Ferrari–Canny $Q_1$ 的 2D 版，单位 = wrench 量纲）。

    = 摩擦锥 wrench 凸包以原点为中心的内切球半径（可阻最小扰动的范数
    下界）；力闭合不成立时返回 0.0（协议：质量非负，0 = 不可行）。

    Args:
        grasp: :class:`Grasp2D` 候选。
        mu: 摩擦系数。
        n_rays: 锥离散射线数。

    Returns:
        标量 ≥ 0。
    """
    inside, l1 = wrench_hull_analysis(grasp_wrenches(grasp, mu, n_rays))
    return l1 if inside else 0.0


def l1_by_support_sampling(wrenches: np.ndarray, n_dirs: int = 4000) -> float:
    """L1 的独立交叉验证口径：支撑函数在 Fibonacci 球面栅格上的最小值。

    $Q_1 = \\min_{\\lVert d \\rVert = 1} h(d)$，$h(d) = \\max_i \\langle d, w_i \\rangle$
    （支撑函数；原点居中内切球半径的等价刻画）。栅格最小值 ≥ 真值
    （在子集上取 min），误差 ≤ 网格间距 × 最大 wrench 范数——只用于与
    :func:`wrench_hull_analysis` 交叉验证（tests 有一致性断言），不是
    主实现。

    Args:
        wrenches: (m, 3) elementary wrench 集合。
        n_dirs: 球面方向数（Fibonacci 格点，确定性）。

    Returns:
        标量（可为负 = 原点在凸包外，此时与力闭合判定一致）。
    """
    w = np.asarray(wrenches, dtype=float).reshape(-1, 3)
    i = np.arange(int(n_dirs))
    z = 1.0 - (2.0 * i + 1.0) / n_dirs  # (n,) 均匀纬度
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    az = np.pi * (1.0 + np.sqrt(5.0)) * i  # 黄金角经度
    dirs = np.stack([r * np.cos(az), r * np.sin(az), z], axis=1)  # (n, 3)
    return float((dirs @ w.T).max(axis=1).min())


# ------------------------------------------------------------ 候选生成 --
def polygon_grasp_candidates(poly: ConvexPolygon,
                             u_grid: tuple[float, ...] = (0.25, 0.5, 0.75)) -> list[Grasp2D]:
    """凸多边形的候选抓取：每边 u_grid 参数点 × 两两异边配对（确定性网格）。

    Args:
        poly: :class:`ConvexPolygon` 物体。
        u_grid: 每条边上的采样参数（缺省 3 点 → n 边 3n 个接触点，
            C(3n,2) − n·C(3,2) 个候选；正方形 54 个）。

    Returns:
        list[:class:`Grasp2D`]，生成顺序确定（边序 × u 序）。
    """
    n_edges = len(np.asarray(poly.vertices))
    contacts = [poly.contact(e, u) for e in range(n_edges) for u in u_grid]
    grasps: list[Grasp2D] = []
    for (i, j) in combinations(range(len(contacts)), 2):
        ei, _ = divmod(i, len(u_grid))  # 边序号（同边点对不构成抓取）
        ej, _ = divmod(j, len(u_grid))
        if ei == ej:
            continue
        grasps.append(Grasp2D(
            points=np.stack([contacts[i][0], contacts[j][0]], axis=0),
            normals=np.stack([contacts[i][1], contacts[j][1]], axis=0),
        ))
    return grasps


def disk_grasp_candidates(disk: Disk, n_angles: int = 12) -> list[Grasp2D]:
    """圆盘的候选抓取：边界 n_angles 等分角点两两配对（确定性网格）。"""
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    contacts = [disk.contact(t) for t in thetas]
    grasps: list[Grasp2D] = []
    for (i, j) in combinations(range(len(contacts)), 2):
        grasps.append(Grasp2D(
            points=np.stack([contacts[i][0], contacts[j][0]], axis=0),
            normals=np.stack([contacts[i][1], contacts[j][1]], axis=0),
        ))
    return grasps


def rank_candidates(candidates: list[Grasp2D], mu: float,
                    n_rays: int = N_RAYS) -> list[tuple[float, Grasp2D]]:
    """按 L1 质量降序排列候选（并列质量保持生成序——Python 稳定排序）。

    Args:
        candidates: 候选抓取列表。
        mu / n_rays: 摩擦系数与锥离散射线数。

    Returns:
        list[(质量, 候选)]，质量降序；不可行抓取质量为 0.0 排在最后。
    """
    scored = [(grasp_quality(g, mu, n_rays), g) for g in candidates]
    return sorted(scored, key=lambda t: -t[0])


def best_grasp(candidates: list[Grasp2D], mu: float,
               n_rays: int = N_RAYS) -> Grasp2D:
    """质量最优的候选（并列取生成序第一个，保证确定性）——对应原文
    "1.32M 候选 → 校验/排序"流程的筛端。"""
    return rank_candidates(candidates, mu, n_rays)[0][1]

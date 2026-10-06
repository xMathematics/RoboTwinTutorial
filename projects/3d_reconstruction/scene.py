"""合成场景与深度渲染（scene）—— 3D 重建教学实现的数据生成前端。

函数流水线
----------
本模块是重建管线的第 0 环（数据侧），被 ``tsdf``（第 04 章融合）与
``marching``（第 05 章提取）下游消费：

    Sphere / Box / Plane （SDF 基元：外正内负，教程 (1.13)）
        -> Scene （min 并集，教程 (1.2) 的零水平集语义；
                   Scene.closest_point 给出解析最近点，供真值表面采样）
        -> render_depth （球追踪 sphere tracing，Hart 1996；
                          教程 4.3 节"光线投射"的自适应步长版）
        -> make_orbit_poses / render_dataset （N 视角外参 + 内参 + 带噪深度图）
        -> sample_scene_surface （包围盒拒绝采样 + 解析投影 = 评测用真值表面点）
        -> checker_texture / value_noise_texture / render_rgb
                   （世界坐标程序化纹理 + RGB 渲染，供 plane_sweep 的
                     平面扫描光度匹配使用，教程第 03 章）

依赖方向：本模块不依赖库内其他模块（仅 numpy + 标准库）；
``tsdf.py``、``demo.py`` 与 ``tests/test_scene.py`` 直接调用本模块。

输入/输出：输入为内存数组——点集 (M, 3)（单位 m）、内参 K (3, 3)（单位 px）、
外参 T_wc (4, 4)（相机 -> 世界，x_world = R @ x_cam + t）；输出 SDF 值 (M,)
（单位 m，外正内负）、深度图 (H, W)（单位 m，无回波 = +inf）与数据集 dict。
全部确定性（固定 seed 或无随机），无文件 I/O。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Box",
    "Plane",
    "Scene",
    "Sphere",
    "checker_texture",
    "make_demo_scene",
    "make_intrinsics",
    "make_orbit_poses",
    "make_sphere_scene",
    "render_dataset",
    "render_depth",
    "render_rgb",
    "sample_scene_surface",
    "value_noise_texture",
]

# 球追踪的收敛容差（m）：|sdf| 低于它即认为光线命中表面（深度分辨率量级）。
HIT_EPS = 1e-4
# 球追踪的最大光行距离（m）：超过即判"无回波"（场景包围球直径量级即可）。
T_MAX = 4.0
# 球追踪的最大步数：步长下界由 SDF 的 Lipschitz-1 性质（教程 (6.1)）保证，
# 实践中 100+ 步足够从包围球外走到表面。
MAX_STEPS = 160


@dataclass(frozen=True)
class Sphere:
    """球 SDF 基元：s(p) = ||p - c|| - r（外正内负，教程 (1.13) 的解析特例）。

    Attributes:
        center: (3,) 球心世界坐标，单位 m。
        radius: 球半径，标量，单位 m，取值 > 0。
    """

    center: np.ndarray
    radius: float

    def sdf(self, points: np.ndarray) -> np.ndarray:
        """点到球面的带符号距离。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M,) SDF 值，单位 m：球外为正、球内为负、球面上为 0（(1.13)）。
        """
        d = np.asarray(points, dtype=float) - np.asarray(self.center, dtype=float)
        return np.linalg.norm(d, axis=-1) - self.radius

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        """球面上离查询点最近的点（沿径向投影，球面的解析最近点）。

        Args:
            points: (M, 3) 查询点集，单位 m（约定不含球心本身）。

        Returns:
            (M, 3) 球面上的最近点，单位 m。
        """
        d = np.asarray(points, dtype=float) - np.asarray(self.center, dtype=float)
        # 沿 (p - c) 方向归一化后走 r：||cp - c|| = r，且 cp 在 p 的径向射线上。
        direction = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return np.asarray(self.center, dtype=float) + self.radius * direction


@dataclass(frozen=True)
class Box:
    """轴对齐盒子（AABB）SDF 基元：盒外取欧氏距离、盒内取 -到最近面距离。

    距离公式（精确 SDF，A. G. "iq"，iker.vui 论坛式；教程 (1.13) 外正内负）：
    q = |p - c| - h；盒外 s = ||max(q, 0)||，盒内 s = min(max(qx, qy, qz), 0)。

    Attributes:
        center: (3,) 盒中心世界坐标，单位 m。
        half_size: (3,) 沿 x/y/z 的半边长，单位 m，各分量 > 0。
    """

    center: np.ndarray
    half_size: np.ndarray

    def sdf(self, points: np.ndarray) -> np.ndarray:
        """点到盒表面的带符号距离（盒外为精确欧氏距离，盒内为负深度）。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M,) SDF 值，单位 m：盒外正、盒内负、表面上为 0。
        """
        q = np.abs(np.asarray(points, dtype=float) - np.asarray(self.center, dtype=float))
        q -= np.asarray(self.half_size, dtype=float)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)   # 盒外：超出分量构成的距离
        inside = np.minimum(np.max(q, axis=-1), 0.0)            # 盒内：到最近面的负距离
        return outside + inside

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        """盒表面上离查询点最近的点。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M, 3) 盒表面上的最近点，单位 m。
        """
        c = np.asarray(self.center, dtype=float)
        h = np.asarray(self.half_size, dtype=float)
        q = np.asarray(points, dtype=float) - c
        # 盒外：各轴截断到 [-h, h] 恰落在表面上（至少一轴贴边）。
        outside_cp = c + np.clip(q, -h, h)
        # 盒内：沿"离面最近"的轴推到该面（h - |q| 最小的轴 = 最薄的方向）。
        inside = np.all(np.abs(q) <= h, axis=-1)
        axis = np.argmin(h - np.abs(q), axis=-1)
        inside_cp = q.copy()
        inside_cp[np.arange(len(q)), axis] = np.sign(q[np.arange(len(q)), axis]) * h[axis]
        inside_cp += c
        return np.where(inside[:, None], inside_cp, outside_cp)


@dataclass(frozen=True)
class Plane:
    """无限平面 SDF 基元：s(p) = n̂·(p - x0)（过点 x0、单位法向 n̂）。

    与教程 (1.3)/01 章的点到平面带符号距离一致（法向朝外时外正内负）。

    Attributes:
        point: (3,) 平面上一点的世界坐标，单位 m。
        normal: (3,) 平面法向（内部自动归一化），单位 m/|n| 任意。
    """

    point: np.ndarray
    normal: np.ndarray

    @property
    def unit_normal(self) -> np.ndarray:
        """归一化后的法向 (3,)，无量纲，模长 1。"""
        n = np.asarray(self.normal, dtype=float)
        return n / np.linalg.norm(n)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        """点到平面的带符号距离：s = n̂·(p - x0)。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M,) SDF 值，单位 m：法向一侧为正、另一侧为负、平面上为 0。
        """
        d = np.asarray(points, dtype=float) - np.asarray(self.point, dtype=float)
        return d @ self.unit_normal

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        """平面上离查询点最近的点（沿法向垂直投影）。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M, 3) 平面上的最近点，单位 m。
        """
        s = self.sdf(points)
        return np.asarray(points, dtype=float) - s[:, None] * self.unit_normal[None, :]


@dataclass(frozen=True)
class Scene:
    """SDF 基元的 min 并集场景（教程 (1.2)：表面 = 零水平集）。

    Attributes:
        primitives: 基元元组，每个成员提供 ``sdf(points)`` 与
            ``closest_point(points)`` 两个方法。
        bbox_lo: (3,) 场景包围盒下角，单位 m（覆盖全部基元）。
        bbox_hi: (3,) 场景包围盒上角，单位 m。
    """

    primitives: tuple
    bbox_lo: np.ndarray
    bbox_hi: np.ndarray

    def sdf(self, points: np.ndarray) -> np.ndarray:
        """场景 SDF：各基元 SDF 的逐点最小值（并集的外正内负距离场）。

        min 并集对"外部"是精确的（最近表面 = 各基元最近者）；
        对"内部"是保守近似——本实现全部基元互不重叠，故处处精确。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M,) 场景 SDF 值，单位 m，外正内负。
        """
        return np.min(np.stack([prim.sdf(points) for prim in self.primitives], axis=0), axis=0)

    def closest_point(self, points: np.ndarray) -> np.ndarray:
        """并集表面上离查询点最近的点（各基元最近点中 |sdf| 最小者）。

        基元互不重叠时上式精确；表面拼缝处（两个 |sdf| 几乎并列）存在
        1e-9 量级的选择歧义，对评测无影响（``sample_scene_surface``
        会丢弃此类样本）。

        Args:
            points: (M, 3) 查询点集，单位 m。

        Returns:
            (M, 3) 并集表面上的最近点，单位 m。
        """
        sdfs = np.stack([prim.sdf(points) for prim in self.primitives], axis=0)  # (P, M)
        cps = np.stack([prim.closest_point(points) for prim in self.primitives],
                       axis=0)                        # (P, M, 3) 各基元最近点
        owner = np.argmin(np.abs(sdfs), axis=0)       # (M,) 谁的表面更近
        idx = np.arange(len(points))
        return cps[owner, idx]


def make_sphere_scene(radius: float = 1.0) -> Scene:
    """单球测试场景（教程 (1.13)/(6.1) 的解析验证基准）。

    Args:
        radius: 球半径，单位 m，默认 1.0。

    Returns:
        Scene：单个 :class:`Sphere`，包围盒 [-2, 2]^3。
    """
    return Scene(
        primitives=(Sphere(np.zeros(3), radius),),
        bbox_lo=-np.full(3, radius + 1.0),
        bbox_hi=np.full(3, radius + 1.0),
    )


def make_demo_scene() -> Scene:
    """演示场景：单位为米的"球 + 盒"双物体（互不重叠，便于解析真值）。

    布局：球 r=0.35 @ 原点；盒半边长 0.15 @ (0.60, 0, 0)。
    两物体表面最近间距 0.10 m > 截断带宽 mu（``tsdf.py`` 默认），
    保证 TSDF 融合不会把两个物体粘连。
    """
    return Scene(
        primitives=(
            Sphere(np.zeros(3), 0.35),
            Box(np.array([0.60, 0.0, 0.0]), np.full(3, 0.15)),
        ),
        bbox_lo=np.array([-0.35, -0.35, -0.35]),
        bbox_hi=np.array([0.75, 0.35, 0.35]),
    )


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """针孔相机内参矩阵 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]。

    与 SLAM 教程 (4.1)/(5.10) 的针孔投影 u = fx·X/Z + cx 同一口径
    （本实现按任务要求自包含，不复用 projects/slam）。

    Args:
        fx, fy: 焦距，单位 px。
        cx, cy: 主点（光轴与像平面交点的像素坐标），单位 px。

    Returns:
        K: (3, 3) 内参矩阵，无量纲元素（px）。
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def make_orbit_poses(
    n_views: int,
    radius: float = 1.6,
    elevation: float = 0.35,
    elevations: list[float] | None = None,
) -> list[np.ndarray]:
    """环绕轨道相机位姿：n 个视角均匀分布在水平圆上、俯仰角循环取值、看向原点。

    相机坐标系约定（CV 惯例）：z 轴指向前方（光轴）、x 向右、y 向下；
    T_wc 把相机系点映射到世界系（x_world = R @ x_cam + t）。每个位姿
    由 look-at 构造：光轴 (Z 列) 指向原点，世界 +z 为向上参考方向。

    Args:
        n_views: 视角数，取值 >= 1。
        radius: 轨道半径，单位 m（须大于场景包围球半径，保证相机在外部）。
        elevation: 默认俯仰角，单位 rad，> 0 为俯视（``elevations`` 缺省时
            全部视角都用它——单一轨道环）。
        elevations: 俯仰角列表，单位 rad（如 [0.35, -0.3] = 上下双环），
            第 k 个视角用 ``elevations[k % len(elevations)]``；双环覆盖
            朝下/朝上表面，避免单环对"环平面以下盲区"的系统性缺测。

    Returns:
        T_wc 列表，长度 n_views，每个 (4, 4)，单位 m / rad。
    """
    ring = elevations if elevations is not None else [elevation]
    poses: list[np.ndarray] = []
    up = np.array([0.0, 0.0, 1.0])  # 世界向上方向
    for k in range(n_views):
        az = 2.0 * np.pi * k / n_views
        el = ring[k % len(ring)]
        eye = np.array([
            radius * np.cos(el) * np.cos(az),
            radius * np.cos(el) * np.sin(az),
            radius * np.sin(el),
        ])
        forward = -eye / np.linalg.norm(eye)          # 光轴：从相机指向原点
        right = np.cross(forward, up)                 # 右方向 = forward × up（观察者右手）
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)               # 下方向补全正交基（CV 系 y 向下）
        down /= np.linalg.norm(down)
        R_wc = np.column_stack([right, down, forward])  # 列 = 相机轴在世界系的方向
        T_wc = np.eye(4)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = eye
        poses.append(T_wc)
    return poses


def render_depth(
    scene: Scene,
    K: np.ndarray,
    T_wc: np.ndarray,
    width: int,
    height: int,
    hit_eps: float = HIT_EPS,
    t_max: float = T_MAX,
    max_steps: int = MAX_STEPS,
) -> np.ndarray:
    """球追踪（sphere tracing）渲染一张深度图。

    每像素发一条光线 d_cam = K^{-1}[u, v, 1]（z 分量归一为 1 后整体归一化），
    从相机中心 t=0 起步反复 t <- t + s(p)，p = o + t·d̂：由 SDF 的
    Lipschitz-1 性质（教程 (6.1) Eikonal），步长 s 永不跳过表面——这是
    教程 4.3 节"体素尺度步进光线投射"的自适应版本（Hart, 1996）。
    命中（|s| < hit_eps）后深度取相机系 z：depth = t · d̂_z。

    Args:
        scene: :class:`Scene`，被渲染的场景。
        K: (3, 3) 内参矩阵，单位 px。
        T_wc: (4, 4) 相机 -> 世界位姿，单位 m / rad。
        width, height: 图像宽/高，单位 px。
        hit_eps: 命中容差，单位 m。
        t_max: 最大光行距离，单位 m，超过按无回波处理。
        max_steps: 最大追踪步数。

    Returns:
        depth: (height, width) 深度图，单位 m，取值 (0, t_max]；
               无回波像素为 +inf（调用方按 isfinite 判有效性）。
    """
    K = np.asarray(K, dtype=float)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    T_wc = np.asarray(T_wc, dtype=float)
    R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]

    # 像素光线方向（相机系，未归一化时 z 分量 = 1，即针孔投影的逆）。
    us = np.arange(width, dtype=float)
    vs = np.arange(height, dtype=float)
    uu, vv = np.meshgrid(us, vs)                       # (H, W) 各
    d_cam = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    d_cam = d_cam.reshape(-1, 3)                       # (HW, 3)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_world = d_cam @ R_wc.T                           # (HW, 3) 方向旋转不带平移

    t = np.zeros(len(d_world))                         # 沿光线的弧长参数，m
    depth = np.full(len(d_world), np.inf)
    active = np.ones(len(d_world), dtype=bool)         # 仍在追踪的光线
    for _ in range(max_steps):
        if not active.any():
            break
        p = t[active, None] * d_world[active] + t_wc   # 当前采样点 (A, 3)
        s = scene.sdf(p)                               # (A,) 步长 = 当前 SDF 值
        idx = np.nonzero(active)[0]
        hit = np.abs(s) < hit_eps
        depth[idx[hit]] = t[idx[hit]] * d_cam[idx[hit], 2]  # 相机系 z = t·d̂_z
        exceed = (t[idx] + s) > t_max
        active[idx[hit | exceed]] = False              # 命中 / 飞出界：停止
        t[idx] += s                                    # 球追踪核心步
    return depth.reshape(height, width)


def render_dataset(
    scene: Scene,
    n_views: int = 24,
    img_size: int = 56,
    focal: float = 50.0,
    radius: float = 1.6,
    elevation: float = 0.35,
    elevations: tuple[float, ...] | list[float] | None = (0.35, -0.30),
    depth_noise_coeff: float = 0.002,
    seed: int = 0,
) -> dict:
    """生成 N 视角 RGB-D 数据集（本教学实现深度即数据，无需颜色）。

    深度噪声按结构光/双目的视差误差模型 sigma(z) = c·z^2 注入
    （教程 (4.2)：dz = -z^2/(fb)·d_delta 的误差传播），这正是 TSDF
    逆方差权重 w ∝ 1/z^2 的最优性来源。默认上下双环轨迹（+0.35 /
    -0.30 rad 交替），使朝下表面也有观测。

    Args:
        scene: 被扫描的场景。
        n_views: 视角数，默认 24（demo 口径）。
        img_size: 图像边长（正方形），单位 px。
        focal: 焦距 fx = fy，单位 px。
        radius: 轨道半径，单位 m。
        elevation: 单环模式的俯仰角，单位 rad（``elevations=None`` 时生效）。
        elevations: 俯仰角序列（见 :func:`make_orbit_poses`）；None 退化为单环。
        depth_noise_coeff: 噪声系数 c，单位 m（sigma = c·z^2）；0 = 无噪声。
        seed: 深度噪声的随机种子（确定性输出）。

    Returns:
        dict，字段：
        - ``"K"``: (3, 3) 内参，px；
        - ``"poses"``: list[n_views]，每个 (4, 4) 外参 T_wc；
        - ``"depths"``: list[n_views]，每个 (img_size, img_size) 深度图，m。
    """
    K = make_intrinsics(focal, focal, (img_size - 1) / 2.0, (img_size - 1) / 2.0)
    poses = make_orbit_poses(n_views, radius=radius, elevation=elevation,
                             elevations=list(elevations) if elevations else None)
    rng = np.random.default_rng(seed)
    depths: list[np.ndarray] = []
    for T_wc in poses:
        depth = render_depth(scene, K, T_wc, img_size, img_size)
        finite = np.isfinite(depth)
        # 逐像素独立高斯噪声，标准差 sigma = c·z^2（教程 (4.2) 误差模型）。
        sigma = depth_noise_coeff * depth[finite] ** 2
        depth[finite] += rng.normal(0.0, sigma)
        depths.append(depth)
    return {"K": K, "poses": poses, "depths": depths}


def sample_scene_surface(
    scene: Scene,
    n_samples: int,
    band: float = 0.05,
    seed: int = 0,
    chunk: int = 4096,
    max_attempts: int = 200,
) -> np.ndarray:
    """从场景表面解析采样 n 个点（评测用的真值表面点）。

    两步：(1) 在包围盒内拒绝采样、保留 |sdf| <= band 的"表面薄壳"点；
    (2) 用 :meth:`Scene.closest_point` 把它们投影到表面上。基元表面在
    拼缝处（两个 |sdf| 并列）投影有歧义，此类样本被丢弃。

    Args:
        scene: 目标场景（要求基元互不重叠以保证投影精确）。
        n_samples: 需要的样本数。
        band: 薄壳半宽，单位 m（越大接受率越高、投影越准）。
        seed: 随机种子（确定性）。
        chunk: 每批候选点数。
        max_attempts: 最多尝试批数（超出则返回已采到的样本）。

    Returns:
        points: (n', 3) 表面采样点，单位 m，n' <= n_samples（拼缝样本被剔除）。
    """
    rng = np.random.default_rng(seed)
    span = scene.bbox_hi - scene.bbox_lo
    out: list[np.ndarray] = []
    total = 0
    for _ in range(max_attempts):
        if total >= n_samples:
            break
        cand = scene.bbox_lo + rng.random((chunk, 3)) * span   # 均匀候选点
        sdfs = np.stack([prim.sdf(cand) for prim in scene.primitives], axis=0)  # (P, chunk)
        shell = np.min(np.abs(sdfs), axis=0) <= band
        cand = cand[shell]
        if len(cand) == 0:
            continue
        sdfs = sdfs[:, shell]
        # 拼缝歧义剔除：最近与次近 |sdf| 几乎并列（差 < 1e-6）的点不可靠。
        # 单基元场景无拼缝，全部保留（argsort 只有 1 行，跳过该检查）。
        if sdfs.shape[0] >= 2:
            order = np.argsort(np.abs(sdfs), axis=0)[:2]
            gap = np.abs(sdfs[order[1], np.arange(len(cand))]
                         - sdfs[order[0], np.arange(len(cand))])
            cand = cand[gap > 1e-6]
        if len(cand):
            out.append(scene.closest_point(cand))
            total += len(out[-1])
    if not out:
        raise RuntimeError("采样失败：band 过小或场景表面不在包围盒内")
    return np.concatenate(out, axis=0)[:n_samples]


# ---------------------------------------------------------- 程序化纹理 ----------
# 本节为 plane_sweep（教程第 03 章平面扫描）提供数据侧支持：多视图立体重建
# 需要"同一物理表面点在不同视图里颜色一致"的图像，故纹理一律定义为
# 世界坐标的函数（外参/内参不参与纹理——先渲染深度、再反投影、后查色）。

def checker_texture(
    points: np.ndarray,
    cell: float = 0.15,
    low: float = 0.2,
    high: float = 0.9,
) -> np.ndarray:
    """三维棋盘格纹理：世界坐标按 cell 立方体网格的奇偶着两色（MVS 强纹理基准）。

    颜色 = f(世界坐标)：同一表面点从任何视角查色结果相同——这正是平面扫描
    （教程 3.1/3.2 节）能靠光度一致性（NCC，(3.4)）恢复深度的前提；格子的
    高频边界为 NCC 提供"窗口形状"，是弱纹理对照实验的强纹理一侧。

    Args:
        points: (M, 3) 世界坐标查询点，单位 m。
        cell: 棋盘格边长，单位 m，取值 > 0。
        low, high: 两色灰度，无量纲，取值 [0, 1]（灰度图复制到 RGB 三通道）。

    Returns:
        colors: (M, 3) RGB 颜色，float，范围 [low, high]。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    # 三轴格子下标求和的奇偶 = 3D 棋盘着色（与 2D 棋盘同一构造）。
    parity = np.sum(np.floor(pts / float(cell)), axis=-1) % 2.0
    gray = low + (high - low) * parity
    return np.repeat(gray[:, None], 3, axis=-1)


def _lattice_hash(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray,
                  seed: int) -> np.ndarray:
    """整数格点 -> [0, 1) 的确定性散列（值噪声的格点随机值）。

    SplitMix64 风格的混合（乘大奇数 + 右移异或）：numpy uint64 乘法按
    2^64 取模回绕，逐位确定——同输入跨平台/跨版本输出一致。
    """
    h = (ix.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
         ^ iy.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
         ^ iz.astype(np.uint64) * np.uint64(0x94D049BB133111EB)
         ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF))
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) / float(2 ** 53)  # [0, 1)


def value_noise_texture(
    points: np.ndarray,
    cell: float = 0.35,
    seed: int = 0,
    low: float = 0.15,
    high: float = 0.85,
) -> np.ndarray:
    """三维值噪声（value noise）纹理：格点随机值的三线性平滑插值（MVS 备选纹理）。

    与棋盘格（锐利边界）对照：值噪声是连续灰度、无规则取向，更接近真实
    表面外观；同样定义为世界坐标的函数，跨视角查色一致。

    Args:
        points: (M, 3) 世界坐标查询点，单位 m。
        cell: 噪声格点间距，单位 m，取值 > 0。
        seed: 格点散列种子（确定性输出）。
        low, high: 灰度映射下/上限，无量纲，[0, 1]。

    Returns:
        colors: (M, 3) RGB 颜色，float，范围 [low, high]。
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    g = pts / float(cell)                            # 连续格坐标
    i0 = np.floor(g).astype(np.int64)                # 8 个包围格点
    f = g - i0
    # 五次平滑（C2 连续）：s(t) = t^3 (t (6t - 15) + 10)。
    s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    val = np.zeros(len(pts))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                h = _lattice_hash(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
                wgt = ((s[:, 0] if dx else 1.0 - s[:, 0])
                       * (s[:, 1] if dy else 1.0 - s[:, 1])
                       * (s[:, 2] if dz else 1.0 - s[:, 2]))
                val += wgt * h
    gray = low + (high - low) * val
    return np.repeat(gray[:, None], 3, axis=-1)


def render_rgb(
    scene: Scene,
    K: np.ndarray,
    T_wc: np.ndarray,
    width: int,
    height: int,
    texture,
    depth: np.ndarray | None = None,
) -> np.ndarray:
    """渲染一张 RGB 图：深度（球追踪）+ 反投影 + 世界坐标纹理查色。

    与 ``render_depth`` 共用同一几何（光线路径完全一致，仅多一步着色）：
    有效像素按深度反投影回世界系（针孔投影的逆，SLAM 教程 (5.10)），颜色
    = texture(世界坐标)——纹理是世界坐标的函数，故多视角渲染天然满足平面
    扫描所需的跨视角光度一致性（教程 3.2 节 (3.3)-(3.5)）。

    Args:
        scene: 被渲染的场景。
        K: (3, 3) 内参矩阵，单位 px。
        T_wc: (4, 4) 相机 -> 世界位姿，单位 m / rad。
        width, height: 图像宽/高，单位 px。
        texture: 可调用对象，签名 ``texture(points (M,3)) -> colors (M,3)``；
            例如 :func:`checker_texture` / :func:`value_noise_texture`。
        depth: 可选，预先算好的深度图 (H, W)（单位 m，无回波 +inf）；
            None 时内部调用 :func:`render_depth`。

    Returns:
        image: (height, width, 3) RGB 图，float，范围 [0, 1]；
               无回波像素为黑色 (0, 0, 0)。
    """
    if depth is None:
        depth = render_depth(scene, K, T_wc, width, height)
    depth = np.asarray(depth, dtype=float)
    K = np.asarray(K, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    height_d, width_d = depth.shape
    us, vs = np.meshgrid(np.arange(width_d, dtype=float),
                         np.arange(height_d, dtype=float))
    image = np.zeros((height_d, width_d, 3))
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return image
    # 反投影（与 tsdf.depth_to_point_cloud 同式，按依赖方向约定就地实现）：
    # 深度 = 相机系 z，未归一化光线 z 分量为 1 -> 命中点 = z·[(u-cx)/fx, (v-cy)/fy, 1]。
    z = depth[valid]
    cam = np.column_stack([(us[valid] - cx) / fx * z,
                           (vs[valid] - cy) / fy * z, z])
    R_wc, t_wc = np.asarray(T_wc, dtype=float)[:3, :3], np.asarray(T_wc, dtype=float)[:3, 3]
    world = cam @ R_wc.T + t_wc
    image[valid] = np.clip(texture(world), 0.0, 1.0)
    return image

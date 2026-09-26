"""半稠密直接法对齐：在 SE(3) 上最小化光度误差。

教程第 06 章（视觉里程计 II：直接法）的教学实现，取 LSD-SLAM 半稠密跟踪的
思想（Engel, Schöps & Cremers, "LSD-SLAM: Large-Scale Direct Monocular SLAM",
ECCV 2014）：不做特征匹配，而是估计使光度残差（photometric residual）

.. math::  e_p(\\xi) = I_2\\big(\\pi(T p)\\big) - I_1\\big(\\pi(p)\\big)   (6.4)

最小的位姿 ``T``，在 ``I1`` 中梯度幅值最大的像素上做 Gauss-Newton 迭代
（半稠密 = semi-dense：弱纹理像素为何被丢弃见
:func:`select_gradient_pixels`）。

函数流水线
----------
``make_textured_plane``（合成纹理平面场景：I1、I2、真值位姿、参考帧深度图）
→ ``select_gradient_pixels``（梯度幅值前 ``top_frac`` 像素筛选）
→ ``photometric_residuals``（光度残差 (6.4)）
→ ``semi_dense_align``（复用 :func:`core.solver.gauss_newton`，在相对
``T0`` 的左扰动坐标上做 G-N 流形迭代，教程 (8.9)）。
``make_plane_scene``（场景合成）与 ``projection_jacobian``（(5.11) 几何
雅可比）被 ``droidlite`` 复用。

雅可比链（教程 (6.5)，三因子）
------------------------------
取左扰动 ``\\tilde T = \\exp(\\delta\\xi^\\wedge) T``（教程 ch.02
(2.25)-(2.26)、ch.08 (8.9)-(8.10)；``\\delta\\xi`` 分量次序
``(\\rho, \\phi)`` =（平移，旋转），与 :func:`core.lie.se3_exp` 一致），
链式法则给出

.. math::  \\frac{\\partial e_p}{\\partial \\delta\\xi}
           = \\underbrace{\\frac{\\partial I_2}{\\partial u}}_{\\text{图像梯度}}
             \\cdot \\underbrace{\\frac{\\partial \\pi}{\\partial q}}_{\\text{投影导数}}
             \\cdot \\underbrace{\\frac{\\partial q}{\\partial \\delta\\xi}}_{\\text{位姿扰动}},

其中 ``q = T p`` 为该点在当前（目标）相机系下的坐标：

1. **图像梯度** ``(I_{2,u}, I_{2,v})``：用 ``np.gradient`` 计算，并在 warp
   后的像素处双线性采样（教程 (6.5) 第一步：图像离散，工程上以有限差分近似）。
2. **投影导数**：对 ``u = f_x X/Z + c_x`` 用商法则求导（ch.05 (5.11) 因子一）。
3. **位姿扰动导数**：``\\partial q/\\partial\\delta\\xi
   = [\\,I \\;|\\; -q^\\wedge\\,]``（ch.02 (2.26) 左扰动导数）。

因子 2、3 相乘得解析的 2x6 几何雅可比 :func:`projection_jacobian`，逐元素
与 PnP 雅可比 (5.11) 相同（旋转在 4-6 列，如 ``[1, 5] = f_y X'/Z'``）；其
元素清单与符号约定（教程 (5.11) 下方的符号约定说明）已在
``tests/test_direct.py`` 用有限差分验证。

**深度因子。** 参考像素 ``p = \\pi^{-1}(u, Z_1(u))`` 由*固定的*实测深度
``Z_1`` 构造，故 ``\\partial p/\\partial\\delta\\xi = 0``，深度不再贡献
额外因子：逐行微分 ``X' = (T p)_x``，``\\partial X'/\\partial\\delta\\xi``
的第一行为 ``[1, 0, 0, 0, Z', -Y']``（来自 ``[I | -q^\\wedge]``），已经
有限差分验证。（当深度本身也参与优化——DSO 的 inverse depth 参数化——会
多出 ``\\partial q/\\partial\\rho`` 因子；该情形在 ``photoba`` 中实现。）

**双线性采样。** 所有图像取值都经 :func:`bilinear_sample`——手写的分数
权重（不用 cv2），因此残差是 warp 的光滑函数，Gauss-Newton 循环才有定义。

合成数据
--------
:func:`make_textured_plane` 按*真值*几何渲染带纹理平面场景（轻微倾斜的
平面）：参考帧在自己的像素网格上采样纹理；目标帧由*正向 warp* 生成——
把扩展参考网格的每个源像素沿真值位姿推过去，纹理值按双线性权重
（按权重归一）splat 到目标图像上。
"""
from dataclasses import dataclass

import numpy as np

from core.camera import PinholeCamera, make_intrinsics
from core.lie import se3_exp, transform_points
from core.solver import gauss_newton

__all__ = [
    "PlaneScene",
    "bilinear_sample",
    "make_plane_scene",
    "make_textured_plane",
    "photometric_jacobian",
    "photometric_residuals",
    "projection_jacobian",
    "select_gradient_pixels",
    "semi_dense_align",
    "warp_and_sample",
]


def bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """在连续像素坐标处双线性采样 ``image``。

    手写权重公式（双线性采样，不用 cv2）：记 ``(x, y) = (u, v)``，基础胞元
    ``({floor x}, {floor y})``、分数部分 ``(ax, ay)``，则取值为

    ``I[y0, x0] (1-ax)(1-ay) + I[y0, x0+1] ax(1-ay)
    + I[y0+1, x0] (1-ax)ay + I[y0+1, x0+1] ax ay``，

    即离散图像的可微（分片双线性）插值——直接法的残差之所以是 warp 的
    光滑函数，正是因为这次取值本身光滑。

    Args:
        image: (H, W) 数组，灰度（无量纲）。
        uv: (N, 2) 连续像素坐标，列 ``(u, v)`` =（列，行），单位 px。

    Returns:
        (N,) 采样值（灰度，无量纲）。坐标被裁剪到图像矩形内。
    """
    img = np.asarray(image, dtype=float)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    # 基础胞元被夹到 [0, W-2] x [0, H-2]；分数权重裁剪到 [0, 1]——整数坐标
    # （含最后一行/列）于是精确取到该纹素，越界查询则饱和在图像边界。
    x0 = np.clip(np.floor(uv[:, 0]).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(uv[:, 1]).astype(int), 0, img.shape[0] - 2)
    ax = np.clip(uv[:, 0] - x0, 0.0, 1.0)
    ay = np.clip(uv[:, 1] - y0, 0.0, 1.0)
    return (
        (1.0 - ax) * (1.0 - ay) * img[y0, x0]
        + ax * (1.0 - ay) * img[y0, x0 + 1]
        + (1.0 - ax) * ay * img[y0 + 1, x0]
        + ax * ay * img[y0 + 1, x0 + 1]
    )


def projection_jacobian(points_cam: np.ndarray, cam: PinholeCamera) -> np.ndarray:
    """针孔投影对左扰动 SE(3) 的解析 2x6 雅可比。

    相机系点 ``q = (X, Y, Z)`` 与左扰动模型
    ``\\tilde q = \\exp(\\delta\\xi^\\wedge) q``、
    ``\\delta\\xi = (\\rho, \\phi)``（平移在前，教程 ch.02 (2.25)-(2.26)）
    下，``\\partial q/\\partial\\delta\\xi = [\\,I \\;|\\; -q^\\wedge\\,]``，
    与商法则投影导数链式相乘，逐行给出（与 PnP 雅可比 教程 (5.11) 相同；
    旋转 4-6 列的符号跟随教程的 ``\\delta\\phi`` 约定，见 (5.11) 下方的
    符号约定说明）::

        du = [ f_x/Z,  0,      -f_x X/Z^2,  -f_x X Y/Z^2,  f_x(1+X^2/Z^2), -f_x Y/Z ]
        dv = [ 0,      f_y/Z,  -f_y Y/Z^2,  -f_y(1+Y^2/Z^2), f_y X Y/Z^2,   f_y X/Z ]

    即 1-3 列是平移（ρ）列 ``∂π/∂q``，4-6 列是旋转（φ）列
    ``∂π/∂q · (-q^∧)``——已在 ``tests/test_direct.py`` 用有限差分验证。

    Args:
        points_cam: (N, 3) 相机系点，深度为正，单位 m。
        cam: 针孔内参（f、c 单位 px）。

    Returns:
        (N, 2, 6) 数组；``out[i]`` 为点 ``i`` 的 ``[[du/dδξ], [dv/dδξ]]``，
        平移列单位 px/m、旋转列单位 px/rad。
    """
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    if np.any(pts[:, 2] <= 0):
        raise ValueError("projection_jacobian requires positive depth (Z > 0)")
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    z2 = z * z
    zeros = np.zeros_like(z)
    du = np.stack(
        [
            cam.fx / z,
            zeros,
            -cam.fx * x / z2,
            -cam.fx * x * y / z2,
            cam.fx * (1.0 + x * x / z2),
            -cam.fx * y / z,
        ],
        axis=-1,
    )
    dv = np.stack(
        [
            zeros,
            cam.fy / z,
            -cam.fy * y / z2,
            -cam.fy * (1.0 + y * y / z2),
            cam.fy * x * y / z2,
            cam.fy * x / z,
        ],
        axis=-1,
    )
    return np.stack([du, dv], axis=1)


@dataclass
class PlaneScene:
    """参考相机看到的带纹理平面（合成场景）。

    Attributes:
        texture: (H, W) 参考帧外观；即 :func:`make_textured_plane` 的
            ``I1``（纹理按参考帧投影坐标绘制，恒等位姿下正好在自己的像素
            网格上采样——教程 (6.4) 中的 ``I1(π(p))``），灰度无量纲。
        cam: 所有渲染帧共用的针孔内参（px）。
        depth: (H, W) 参考帧逐像素平面深度（固定深度图，即跟踪阶段消费的
            ``depth1``），单位 m。
        texture_grid: (GH, GW) 纹理上采样前的粗随机网格（粗随机网格 +
            双线性上采样）。
        z0: 平面平均深度，单位 m。
        plane_slope: 平面倾度 ``(s_u, s_v)``，平面方程
            ``Z = z0 + s_u X + s_v Y``（s 无量纲，作用于以 m 计的 X, Y）。
    """

    texture: np.ndarray
    cam: PinholeCamera
    depth: np.ndarray
    texture_grid: np.ndarray
    z0: float
    plane_slope: tuple[float, float]

    def plane_depth(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """倾斜平面沿每条参考帧光线的深度。

        解 ``Z = z0 + s_u (u-c_x) Z/f_x + s_v (v-c_y) Z/f_y``（依据：平面
        方程代入针孔反投影 (4.6)）得
        ``Z = z0 / (1 - s_u (u-c_x)/f_x - s_v (v-c_y)/f_y)``。

        Args:
            u, v: 像素坐标（px），任意同形数组。

        Returns:
            与输入同形的深度，单位 m。
        """
        cam = self.cam
        sx, sy = self.plane_slope
        return self.z0 / (
            1.0 - sx * (u - cam.cx) / cam.fx - sy * (v - cam.cy) / cam.fy
        )

    def render_target(self, T: np.ndarray, margin: int = 40) -> np.ndarray:
        """按位姿 ``T`` 正向 warp 渲染平面在目标帧的成像。

        正向 warp + 双线性采样：把*扩展*参考网格（向外扩 ``margin`` px，
        保证平移后的视场仍被覆盖）的每个像素沿真值位姿 ``T`` 推过去
        （世界系 = 参考相机系），纹理值按双线性权重 splat 到目标图像的投影
        位置上；累加权重做归一化（weight-normalized forward splatting），
        对光滑 warp 这等价于逆向 warp 的外观，只差 O(pixel^2) 的重采样误差。

        Args:
            T: (4, 4) 位姿，世界（参考帧）-> 目标相机（平移 m，旋转无量纲）。
            margin: 源网格向外扩展的像素数（px）。

        Returns:
            (H, W) 目标图像 ``I2``（灰度，无量纲）。
        """
        cam = self.cam
        h, w = self.texture.shape
        xs = np.arange(-margin, w + margin)
        ys = np.arange(-margin, h + margin)
        us, vs = np.meshgrid(xs, ys)
        src = np.stack([us.ravel(), vs.ravel()], axis=-1).astype(float)
        depth_src = self.plane_depth(src[:, 0], src[:, 1])
        rays = np.stack(
            [
                (src[:, 0] - cam.cx) / cam.fx,
                (src[:, 1] - cam.cy) / cam.fy,
                np.ones(len(src)),
            ],
            axis=-1,
        )
        points = rays * depth_src[:, None]          # 参考相机系（= 世界系）坐标
        target_pts = transform_points(T, points)    # 沿真值位姿变换到目标相机系
        u_t = cam.fx * target_pts[:, 0] / target_pts[:, 2] + cam.cx
        v_t = cam.fy * target_pts[:, 1] / target_pts[:, 2] + cam.cy

        gh, gw = self.texture_grid.shape
        grid_uv = np.stack(
            [src[:, 0] * (gw - 1) / (w - 1), src[:, 1] * (gh - 1) / (h - 1)],
            axis=-1,
        )
        values = bilinear_sample(self.texture_grid, grid_uv)

        acc_w = np.zeros((h, w))
        acc_v = np.zeros((h, w))
        x0 = np.floor(u_t).astype(int)
        y0 = np.floor(v_t).astype(int)
        ax = u_t - x0
        ay = v_t - y0
        inside = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
        x0, y0, ax, ay, values = x0[inside], y0[inside], ax[inside], ay[inside], values[inside]
        for dx, dy, weight in (
            (0, 0, (1.0 - ax) * (1.0 - ay)),
            (1, 0, ax * (1.0 - ay)),
            (0, 1, (1.0 - ax) * ay),
            (1, 1, ax * ay),
        ):
            np.add.at(acc_w, (y0 + dy, x0 + dx), weight)
            np.add.at(acc_v, (y0 + dy, x0 + dx), weight * values)
        return acc_v / np.maximum(acc_w, 1e-12)


def make_plane_scene(
    width: int = 240,
    height: int = 180,
    focal: float = 300.0,
    z0: float = 3.0,
    plane_slope: tuple[float, float] = (0.05, -0.03),
    texture_cells: tuple[int, int] = (9, 12),
    texture_range: tuple[float, float] = (40.0, 200.0),
    seed: int = 7,
) -> PlaneScene:
    """构造纹理平面场景（合成数据：平面场景 + 平滑随机纹理）。

    纹理是粗随机网格（seeded，确定性）双线性上采样到图像尺寸——分片双线性，
    在 ``width / texture_cells[1]`` 像素尺度上光滑，因此图像梯度几乎处处
    存在，直接法的线性化成立（教程 06.1 ⑤：排除结构张量退化的"白墙"情形）。

    Args:
        width, height: 图像尺寸，单位 px。
        focal: 焦距 ``f_x = f_y``（px）；主点在图像中心。
        z0: 平面平均深度，单位 m。
        plane_slope: 平面倾度 ``(s_u, s_v)``，``Z = z0 + s_u X + s_v Y``
            ——轻微深度变化（全图只有百分之几）。
        texture_cells: 粗网格形状 ``(rows, cols)``。
        texture_range: 随机网格的灰度范围（无量纲灰度）。
        seed: 随机数种子（确定性）。

    Returns:
        :class:`PlaneScene`，其中 ``texture`` = ``I1``，并带参考帧深度图。
    """
    rng = np.random.default_rng(seed)
    grid = rng.uniform(texture_range[0], texture_range[1], size=texture_cells)
    cam = make_intrinsics(focal, focal, width / 2.0, height / 2.0)
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    gh, gw = texture_cells
    grid_uv = np.stack(
        [uu * (gw - 1) / (width - 1), vv * (gh - 1) / (height - 1)], axis=-1
    ).reshape(-1, 2)
    texture = bilinear_sample(grid, grid_uv).reshape(height, width)
    slope = (float(plane_slope[0]), float(plane_slope[1]))
    scene = PlaneScene(
        texture=texture,
        cam=cam,
        depth=np.empty((height, width)),   # 深度图在下方立即填充
        texture_grid=grid,
        z0=float(z0),
        plane_slope=slope,
    )
    scene.depth = scene.plane_depth(uu.astype(float), vv.astype(float))
    return scene


def make_textured_plane(
    width: int = 240,
    height: int = 180,
    focal: float = 300.0,
    z0: float = 3.0,
    plane_slope: tuple[float, float] = (0.05, -0.03),
    t_gt: tuple[float, float, float] = (0.06, -0.03, 0.04),
    r_gt: tuple[float, float, float] = (0.006, -0.012, 0.009),
    texture_cells: tuple[int, int] = (9, 12),
    texture_range: tuple[float, float] = (40.0, 200.0),
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PinholeCamera, np.ndarray]:
    """渲染已知小运动下带纹理平面的两个视图。

    第一帧 ``T = I``（参考帧，世界系即参考相机系）；第二帧由真值小运动
    ``T_gt``（``xi = (t, r)``，:func:`core.lie.se3_exp` 参数化）渲染
    （正向 warp + 双线性采样，见 :meth:`PlaneScene.render_target`）。

    Args:
        width, height, focal, z0, plane_slope, texture_cells, texture_range,
            seed: 透传给 :func:`make_plane_scene`。
        t_gt: 真值平移（参考 -> 目标相机），单位 m。
        r_gt: 真值旋转向量，单位 rad。

    Returns:
        ``(I1, I2, T_gt, cam, depth)`` —— 参考图像 (H, W)、目标图像 (H, W)
        （灰度无量纲）、(4, 4) 真值位姿（平移 m）、内参，以及供
        :func:`semi_dense_align` 消费的参考帧深度图 (H, W)（单位 m）。
    """
    scene = make_plane_scene(
        width=width,
        height=height,
        focal=focal,
        z0=z0,
        plane_slope=plane_slope,
        texture_cells=texture_cells,
        texture_range=texture_range,
        seed=seed,
    )
    xi = np.array([*t_gt, *r_gt], dtype=float)
    T_gt = se3_exp(xi)
    I2 = scene.render_target(T_gt)
    return scene.texture, I2, T_gt, scene.cam, scene.depth


def select_gradient_pixels(
    image: np.ndarray, top_frac: float = 0.3, border: int = 2
) -> np.ndarray:
    """半稠密像素筛选：保留梯度幅值排前 ``top_frac`` 的像素。

    依据（教程 06.2 第三步 + 06.1 第五步）：直接法雅可比 (6.5) 的最外层因子
    是图像梯度——弱纹理区（白墙、均匀表面）梯度小，``J_p ≈ 0``，这些像素
    对增量方程零贡献却仍有光度噪声，信息少且不可靠；半稠密直接法因此只取
    梯度幅值排在前 ``top_frac`` 的像素，既省算又不损信息。梯度用
    ``np.gradient``（中心差分）计算，边缘 ``border`` 像素环一并排除（其
    warp 易落出图像、双线性采样不完整）。

    Args:
        image: (H, W) 参考图像（灰度，无量纲）。
        top_frac: 内部像素中按梯度幅值保留的比例（无量纲，0-1）。
        border: 图像边界排除的像素环宽（px）。

    Returns:
        (N, 2) float 像素坐标，列 ``(u, v)``，单位 px。
    """
    img = np.asarray(image, dtype=float)
    grad_v, grad_u = np.gradient(img)
    magnitude = np.hypot(grad_u, grad_v)
    interior = np.zeros_like(magnitude, dtype=bool)
    interior[border:-border, border:-border] = True
    threshold = np.quantile(magnitude[interior], 1.0 - top_frac)
    rows, cols = np.nonzero(interior & (magnitude >= threshold))
    return np.stack([cols, rows], axis=-1).astype(float)


def _valid_warp_mask(
    uv_target: np.ndarray, depth_target: np.ndarray, guard: float, width: int, height: int
) -> np.ndarray:
    """warp 后位置保持在图像内 ``guard`` px、且目标深度 > 0.1 m 的像素掩码。"""
    return (
        (uv_target[:, 0] > guard)
        & (uv_target[:, 0] < width - 1 - guard)
        & (uv_target[:, 1] > guard)
        & (uv_target[:, 1] < height - 1 - guard)
        & (depth_target > 0.1)
    )


def photometric_residuals(
    I1: np.ndarray,
    I2: np.ndarray,
    depth1: np.ndarray,
    cam: PinholeCamera,
    T: np.ndarray,
    pixels: np.ndarray | None = None,
) -> np.ndarray:
    """光度残差 ``e_i = I2(π(T p_i)) − I1(p_i)``（教程 (6.4)）。

    Args:
        I1, I2: (H, W) 参考 / 目标图像（灰度，无量纲）。
        depth1: (H, W) 参考帧深度图（固定不优化——见模块 docstring 的
            "深度因子"讨论），单位 m。
        cam: 针孔内参（px）。
        T: (4, 4) 候选位姿，参考相机 -> 目标相机（平移 m）。
        pixels: 可选 (N, 2) 参考像素 ``(u, v)``（px）；缺省取全部内部像素
            （其 warp 在 ``T`` 下仍在图像内的）。

    Returns:
        (N,) 给定像素处的残差（灰度，无量纲；warp 无效的像素已剔除）。
    """
    img1 = np.asarray(I1, dtype=float)
    height, width = img1.shape
    if pixels is None:
        pixels = select_gradient_pixels(img1, top_frac=1.0, border=3)
    depth_sel = bilinear_sample(depth1, pixels)
    points_ref = cam.unproject(pixels, depth_sel)
    points_tgt = transform_points(T, points_ref)
    uv_target = cam.project(points_tgt)
    keep = _valid_warp_mask(uv_target, points_tgt[:, 2], 1.0, width, height)
    return bilinear_sample(np.asarray(I2, dtype=float), uv_target[keep]) - bilinear_sample(
        img1, pixels[keep]
    )


def warp_and_sample(
    image: np.ndarray,
    points_ref: np.ndarray,
    cam: PinholeCamera,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把固定参考系点经 ``T`` warp 后在目标图像上取值。

    实现残差的前向映射 ``u' = π(T p)``、``I2(u')``，取值用可微的
    :func:`bilinear_sample`。

    Args:
        image: (H, W) 目标图像 ``I2``（灰度，无量纲）。
        points_ref: (N, 3) 参考相机系点（深度固定），单位 m。
        cam: 针孔内参（px）。
        T: (4, 4) 候选位姿，参考 -> 目标相机。

    Returns:
        ``(uv, values)`` —— (N, 2) warp 后像素坐标（px）与 (N,) 采样灰度
        （无量纲）。
    """
    points_tgt = transform_points(T, points_ref)
    uv = cam.project(points_tgt)
    return uv, bilinear_sample(image, uv)


def photometric_jacobian(
    image: np.ndarray,
    points_ref: np.ndarray,
    cam: PinholeCamera,
    T: np.ndarray,
) -> np.ndarray:
    """采样目标灰度对左扰动的解析雅可比。

    组装链式法则 (6.5)：``I2`` 的双线性采样 ``np.gradient`` 图像梯度
    （因子 1）乘以几何雅可比 :func:`projection_jacobian`（因子 2-3，
    教程 (5.11)）：

    ``J_i = I_u(u'_i) · G_i[0] + I_v(u'_i) · G_i[1]`` —— 每点一行 (6,)。

    Args:
        image: (H, W) 目标图像 ``I2``（灰度，无量纲）。
        points_ref: (N, 3) 参考相机系点，单位 m。
        cam: 针孔内参（px）。
        T: (4, 4) 当前位姿估计。

    Returns:
        (N, 6) 雅可比行，关于 ``T`` 处的 ``δξ = (ρ, φ)``；平移列单位
        px/m、旋转列单位 px/rad。
    """
    grad_v, grad_u = np.gradient(np.asarray(image, dtype=float))
    uv, _ = warp_and_sample(image, points_ref, cam, T)
    points_tgt = transform_points(T, points_ref)
    geo = projection_jacobian(points_tgt, cam)
    return (
        bilinear_sample(grad_u, uv)[:, None] * geo[:, 0, :]
        + bilinear_sample(grad_v, uv)[:, None] * geo[:, 1, :]
    )


def semi_dense_align(
    I1: np.ndarray,
    I2: np.ndarray,
    depth1: np.ndarray,
    cam: PinholeCamera,
    T0: np.ndarray,
    n_iters: int = 20,
    top_frac: float = 0.3,
    guard: float = 10.0,
) -> np.ndarray:
    """半稠密直接法对齐：在 SE(3) 上最小化光度误差。

    半稠密直接法（LSD-SLAM 思想，教程第 06 章）：只取 ``I1`` 中梯度幅值前
    ``top_frac`` 的像素（弱纹理区梯度小、不可靠、信息少，见
    :func:`select_gradient_pixels`），对光度残差 (6.4) 做 Gauss-Newton
    迭代。雅可比 = 图像梯度 × 几何投影雅可比 (6.5)/(5.11)；深度固定故深度
    因子为零（模块 docstring"深度因子"）。迭代在"相对 ``T0`` 的左扰动
    坐标" ``x ∈ R^6``（``T(x) = exp(x^∧)·T0``）上进行并复用
    :func:`core.solver.gauss_newton`：向量增量 ``x ← x + δx`` 对应流形更新
    ``T ← exp(δx^∧)·T``（教程 (8.9)），雅可比即当前位姿处的左扰动雅可比
    (8.10)——参数化本身保证了 SE(3) 约束精确成立。

    Args:
        I1, I2: (H, W) 参考 / 目标图像（灰度，无量纲）。
        depth1: (H, W) 参考帧逐像素深度（固定不优化），单位 m。
        cam: 针孔内参（px）。
        T0: (4, 4) 初始位姿猜测（例如上一帧的估计）。
        n_iters: Gauss-Newton 最大迭代次数。
        top_frac: 内部像素中按梯度幅值保留的比例（无量纲，0-1）。
        guard: warp 后位置必须留在图像内这么多的像素边距（px；覆盖预期
            纠正量，保证像素永不采出图像）。

    Returns:
        (4, 4) 优化后的位姿 ``T``（参考相机 -> 目标相机，平移 m）。
    """
    img1 = np.asarray(I1, dtype=float)
    img2 = np.asarray(I2, dtype=float)
    height, width = img1.shape

    pixels = select_gradient_pixels(img1, top_frac=top_frac)
    depth_sel = bilinear_sample(depth1, pixels)
    points_ref = cam.unproject(pixels, depth_sel)          # 固定的 3D 参考点
    i1_vals = bilinear_sample(img1, pixels)                # 固定的参考灰度

    # 只保留在 T0 下（从而也在附近 refinement 位姿下）warp 仍稳居目标图像
    # 内部的像素。
    q0 = transform_points(T0, points_ref)
    uv0 = cam.project(q0)
    keep = _valid_warp_mask(uv0, q0[:, 2], guard, width, height)
    points_ref, i1_vals = points_ref[keep], i1_vals[keep]

    def residual_fn(x: np.ndarray) -> np.ndarray:
        _, values = warp_and_sample(img2, points_ref, cam, se3_exp(x) @ T0)
        return values - i1_vals

    def jacobian_fn(x: np.ndarray) -> np.ndarray:
        return photometric_jacobian(img2, points_ref, cam, se3_exp(x) @ T0)

    solution = gauss_newton(
        residual_fn, jacobian_fn, x0=np.zeros(6), n_iters=n_iters, lm_lambda=0.0
    )
    return se3_exp(solution.x) @ T0

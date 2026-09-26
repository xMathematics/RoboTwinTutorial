"""带仿射曝光的滑窗光度光束平差（photometric bundle adjustment）。

DSO 光度 BA 的教学实现（Engel, Koltun & Cremers, "Direct Sparse Odometry",
arXiv:1607.02565, ECCV 2016 / TPAMI 2018；
papers/slam/classics/arXiv-1607.02565_DSO.pdf），对应教程第 06 章
（§06.3 ⑤，Eq. (6.6)-(6.8)）与第 08 章（Eq. (8.5)、(8.9)-(8.10)）的推导：
联合优化一小窗关键帧位姿、每帧仿射亮度参数与每点逆深度
（inverse depth），拟合*光度*残差（photometric residual）。

函数流水线
----------
``make_keyframe_scene`` / :class:`KeyframeScene`（复用 :mod:`direct` 的
纹理平面，渲染 3 关键帧真值场景）→ :class:`PhotometricBA`（host 像素
半稠密选取 → 残差 (6.6) / 解析雅可比 / LM 迭代：Marquardt 对角阻尼 +
回溯线搜索 + Nielsen 增益比（gain ratio）阻尼更新 + 逆深度物理箱）→
:meth:`PhotometricBA.solve` → :class:`PhotometricBAResult`。
复用：:func:`core.solver.huber_weights`（Huber IRLS 权重）、
:func:`direct.projection_jacobian`（(5.11) 几何雅可比）、
:func:`core.solver.gauss_newton`（第一阶段曝光初始化）。

光度模型（DSO 论文 Eq. (2)-(5)）
--------------------------------
DSO 把图像成像建模为 ``I_i(x) = G(t_i V(x) B_i(x))``（论文 Eq. (2)），
光度校正后为 ``I'_i = t_i B_i``（Eq. (3)）。曝光时间未知时，校正后的灰度
用*仿射亮度传递*（affine brightness transfer）
``e^{-a_i}(I_i - b_i)`` 参数化（论文 Eq. (4)；增益取对数参数化保证恒正，
论文 §2.2）。本教学实现采用教程的等价正向仿射形式

.. math::  I' = e^{a}\\, I + b,   \\qquad
           e_{k j} = \\big(e^{a_j} I_j(u'_{k j}) + b_j\\big)
                   - \\big(e^{a_0} I_0(u_k) + b_0\\big)   (6.6)

（DSO Eq. (4) 在 ``t_i = 1`` 下的参数代换；教程 (6.7) 的不变性证明逐字
平移过来）。完整目标函数是 DSO Eq. (8) / 教程 (6.8) 的滑窗光度 BA，

.. math::  \\min_{\\{T_i\\},\\{\\rho_k\\},\\{a_i,b_i\\}}
           \\sum_{k}\\sum_{j \\in \\text{obs}(k)}
           \\big\\| e_{k j} \\big\\|^2_{\\text{Huber}},

用写开的流形 Gauss-Newton / Levenberg-Marquardt 循环在堆叠稠密正规方程
(8.5)-(8.6) 上求解：Huber IRLS 权重复用 :func:`core.solver.huber_weights`
（鲁棒核，教程 06.3 ③ / ch.08 §8.5），第一阶段曝光初始化复用
:func:`core.solver.gauss_newton`。

规范自由度（gauge freedom）
---------------------------
要使正规方程非奇异，必须钉住三个规范（gauge）：

1. **亮度规范（2 自由度）** —— 由教程 (6.7)，残差在
   ``a_i -> a_i + log γ``、``b_i -> γ b_i + δ`` 下不变；以固定锚点帧的
   曝光 ``(a_0, b_0) = (0, 0)`` 钉住（教程 06.3 ⑤ 注记：DSO 用先验
   Eq. (9) 加第一个关键帧）。
2. **位姿规范（6 自由度）** —— 世界系任意；以固定 ``T_0`` 于初值钉住。
3. **尺度规范（1 自由度）** —— 单目尺度不确定性（教程 (5.9)）：
   ``(t_j, p_k) -> (t_j/λ, p_k/λ)`` 使每个投影 ``π(T_j p_k)``——从而每个
   光度残差——*不变*，且 ``T_0 = I`` 锚点（零平移）不破坏它。以固定
   0 号点的逆深度钉住（*尺度锚点* scale anchor；保持在其量测初值上不动，
   如同它所代表的深度图读数）。

钉住之后其余 ``(a_j, b_j)`` 可辨识，且在与模型一致生成的数据上可逐字
恢复真值。

流形处理（manifold update，教程 08.3）
---------------------------------------
每次 LM 迭代都在*当前*状态处线性化：增量
``δx = (δξ_1, δξ_2, δρ, δa, δb)`` 住在当前位姿的切空间中，左扰动雅可比
(8.10)——(5.11) 几何因子与图像梯度因子的复合——在该处是精确的（第 02 章
(2.20) 的 BCH 左雅可比在线性化点退化为恒等，(2.23)）。候选状态经指数
映射施加位姿 ``T_j ← exp(δξ_j^∧) T_j``（8.9，精确 SE(3)），深度/曝光
加性更新，然后重新定基线性化点——即 08.3 ⑤ 的教科书迭代。

雅可比
------
所有块均**解析**：位姿块复用教程 (5.11) 的共享 2x6
（:func:`direct.projection_jacobian`）；图像梯度因子是**手写双线性采样器
的精确导数**（见 :func:`_sample_with_derivative`——``direct.semi_dense_align``
用的中心差分 ``np.gradient`` 变体在纹理胞元边界处与之相差
O(梯度跳变)，多参数 BA 会把这个缺口放大成一阶偏差，故此处用精确的权重
导数）；曝光列为 ``∂e/∂a_j = e^{a_j} I_j``、``∂e/∂b_j = 1``；逆深度列由
``p = ray/ρ`` 得 ``∂p/∂ρ = −ray/ρ²``。全部在 ``tests/test_photoba.py``
中与有限差分核对到机器精度。

教学简化声明
------------
数值雅可比换解析式、滑窗固定 3 帧、点全部 host 在第 0 帧（DSO 的多 host
host-frame 分配被简化）——为的是把推导聚焦在参数布局与流形 LM 本身。

合成场景
--------
:func:`make_keyframe_scene` 渲染 :mod:`direct` 纹理平面的三个视图。
*相机*对标准（无曝光）渲染 ``P_i`` 施加增益/偏移 ``(g_i, o_i) =
(e^{-a_i}, -e^{-a_i} b_i)``，即 ``I_i = e^{-a_i}(P_i - b_i)``——校正模型
的逆——于是用真值 ``(a_i, b_i)`` 白化恰好返回 ``P_i``，真值曝光参数就是
优化器的最优点（第 0 帧无曝光生成，与其规范锚点角色一致）。
"""
from dataclasses import dataclass

import numpy as np

from core.camera import PinholeCamera
from core.lie import se3_exp, transform_points
from core.solver import gauss_newton, huber_weights
from direct import (
    bilinear_sample,
    make_plane_scene,
    projection_jacobian,
    select_gradient_pixels,
)

__all__ = ["KeyframeScene", "PhotometricBA", "PhotometricBAResult", "make_keyframe_scene"]

# 合成相机的真值曝光：第 0 帧是规范固定锚点（无曝光），第 1/2 帧带非零
# 增益/偏移（真值曝光非零）。
EXPOSURE_A_GT = (0.0, 0.2, -0.15)
EXPOSURE_B_GT = (0.0, 3.0, -2.0)

# 状态布局：[dxi_1 (6) | dxi_2 (6) | rho_1..rho_N (N) | a_1, a_2 | b_1, b_2]
N_POSE_PARAMS = 12


@dataclass
class KeyframeScene:
    """纹理平面的三个关键帧，附带真值。

    Attributes:
        images: ``[I_0, I_1, I_2]`` *观测*图像——相机已对标准渲染施加仿射
            曝光 ``(g_i, o_i)``；各 (H, W)，灰度无量纲。
        canonical: ``[P_0, P_1, P_2]`` 无曝光的标准渲染（供测试对照）。
        depth: (H, W) 第 0 帧深度图（所有点的 host 帧），单位 m。
        cam: 针孔内参（三帧共用，px）。
        poses: 真值位姿 ``[T_0 = I, T_1, T_2]``（世界系 = 第 0 帧相机系；
            各 (4, 4)，平移 m）。
        exposure_a, exposure_b: 真值 ``(a_i, b_i)``，``i = 0, 1, 2``；
            构造上 ``a_0 = b_0 = 0``（亮度规范锚点）。
    """

    images: list[np.ndarray]
    canonical: list[np.ndarray]
    depth: np.ndarray
    cam: PinholeCamera
    poses: list[np.ndarray]
    exposure_a: np.ndarray
    exposure_b: np.ndarray


def make_keyframe_scene(
    width: int = 240,
    height: int = 180,
    focal: float = 300.0,
    seed: int = 11,
) -> KeyframeScene:
    """在纹理平面上渲染 3 关键帧窗口（真值场景）。

    位姿：``T_0 = I`` 加两个已知小运动（几像素视差）。曝光：第 0 帧无
    曝光，第 1/2 帧带非零 ``(a_i, b_i)``——即 BA 必须恢复的仿射亮度
    （教程 (6.6)-(6.7)）。

    Args:
        width, height, focal: 图像几何（透传给 :func:`direct.make_plane_scene`；
            px、m）。
        seed: 纹理种子（确定性）。

    Returns:
        :class:`KeyframeScene`。
    """
    scene = make_plane_scene(width=width, height=height, focal=focal, seed=seed)
    T1 = se3_exp(np.array([0.05, 0.02, 0.03, 0.004, 0.01, -0.006]))
    T2 = se3_exp(np.array([-0.04, 0.05, 0.06, -0.01, 0.005, 0.008]))
    poses = [np.eye(4), T1, T2]
    canonical = [scene.texture] + [scene.render_target(T) for T in poses[1:]]
    a_gt = np.asarray(EXPOSURE_A_GT, dtype=float)
    b_gt = np.asarray(EXPOSURE_B_GT, dtype=float)
    # 相机施加校正模型的逆：I = e^{-a}(P - b)（依据：模块 docstring"合成场景"，
    # 真值 (a, b) 白化后恰好还原 P_i）。
    images = [np.exp(-a_gt[i]) * (canonical[i] - b_gt[i]) for i in range(3)]
    return KeyframeScene(
        images=images,
        canonical=canonical,
        depth=scene.depth,
        cam=scene.cam,
        poses=poses,
        exposure_a=a_gt,
        exposure_b=b_gt,
    )


def _sample_with_derivative(
    image: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """双线性采样，连同采样器本身的精确导数。

    对手写双线性权重求导即得雅可比的图像梯度因子（教程 (6.5) 第一因子）。
    这里取双线性采样器的**精确**导数而非 ``np.gradient`` 中心差分：

    ``∂val/∂u = (1−ay)(I01 − I00) + ay(I11 − I10)``，

    ``∂val/∂v = (1−ax)(I10 − I00) + ax(I11 − I01)``。

    在分片双线性纹理的胞元边界（kink）之外，它与中心差分相等
    （direct.semi_dense_align 用 ``np.gradient`` 演示该工程近似）；在
    kink 处中心差分平均了两个胞元的斜率，而这里给出真实的局部导数——
    多参数 BA 会把这个离散化缺口放大成一阶偏差，故本模块的解析雅可比用
    精确导数，并与有限差分核对到机器精度（见 ``tests/test_photoba.py``）。

    Returns:
        ``(values, d_du, d_dv)`` —— 各 (N,)；导数单位为 灰度/px。
    """
    img = np.asarray(image, dtype=float)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    x0 = np.clip(np.floor(uv[:, 0]).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(uv[:, 1]).astype(int), 0, img.shape[0] - 2)
    ax = np.clip(uv[:, 0] - x0, 0.0, 1.0)
    ay = np.clip(uv[:, 1] - y0, 0.0, 1.0)
    i00, i01 = img[y0, x0], img[y0, x0 + 1]
    i10, i11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    values = (
        (1.0 - ax) * (1.0 - ay) * i00
        + ax * (1.0 - ay) * i01
        + (1.0 - ax) * ay * i10
        + ax * ay * i11
    )
    d_du = (1.0 - ay) * (i01 - i00) + ay * (i11 - i10)
    d_dv = (1.0 - ax) * (i10 - i00) + ax * (i11 - i01)
    return values, d_du, d_dv


@dataclass
class PhotometricBAResult:
    """优化后的滑窗状态打包（见 :meth:`PhotometricBA.solve`）。

    Attributes:
        poses: 优化后的 ``[T_0（规范锚点，固定不动）, T_1, T_2]``，各
            (4, 4)（平移 m）。
        exposure_a: (3,) 仿射增益参数（log 空间，无量纲）。
        exposure_b: (3,) 仿射偏移参数（灰度）。
        inverse_depths: (N,) host 帧逆深度（1/m）。
        points: (N, 3) 世界系（第 0 帧相机系）下的 3D 点，单位 m。
        cost: 最终加权限鲁棒代价 sum(w * e^2)（灰度^2）。
        n_iters: 实际执行的 LM 迭代数。
        converged: 是否满足收敛判据（False 也可能是迭代耗尽）。
    """

    poses: list[np.ndarray]       # 优化后的 [T_0（固定）, T_1, T_2]
    exposure_a: np.ndarray        # (3,) 仿射增益参数（log 空间）
    exposure_b: np.ndarray        # (3,) 仿射偏移参数
    inverse_depths: np.ndarray    # (N,) host 帧逆深度（1/m）
    points: np.ndarray            # (N, 3) 世界系（第 0 帧相机系）3D 点
    cost: float
    n_iters: int
    converged: bool


class PhotometricBA:
    """3 关键帧滑窗光度 BA（DSO 思想，教程 (6.8)）。

    变量：两个非锚点位姿 ``T_1, T_2``（各 6 维）、每点的 host 帧逆深度
    （各 1 维，DSO 的 "Point Dimensionality"）、第 1/2 帧的仿射亮度参数
    ``(a_j, b_j)``（第 0 帧作为规范锚点固定——见模块 docstring）。每个点
    host 在一个第 0 帧像素上（光线 ``K^{-1}[u, v, 1]``，``p = ray/ρ``，
    ``ρ = 1/Z``——DSO 的逆深度点参数化），并在第 1、2 帧中被光度观测
    ——每点 2 个残差，系统超定。

    增量向量 ``δx`` 的布局（全展长度 ``16 + N``，见
    :meth:`parameter_layout`）：
    ``[δξ_1 (6) | δξ_2 (6) | δρ_1..δρ_N (N) | δa_1, δa_2 | δb_1, δb_2]``。

    Args:
        images: 三帧 (H, W) 观测图像 ``[I_0, I_1, I_2]``（灰度无量纲）。
        ref_depth: (H, W) 第 0 帧深度图（初始化逆深度），单位 m。
        cam: 针孔内参（px）。
        poses_init: 三个 (4, 4) 初始位姿；``poses_init[0]`` 是位姿规范
            锚点，永不改变。
        top_frac: 第 0 帧内部像素中按梯度幅值保留的比例（半稠密选取，
            :func:`direct.select_gradient_pixels`）。
        max_points: 所选像素的确定性子采样规模（教学实现中保持稠密正规
            方程规模较小）。
        optimize_exposure: 若为 ``False``，``(a_j, b_j)`` 保持初值 0 且被
            排除出状态——对照模式，用于演示亮度变化此时如何使位姿产生
            偏差（教程 (6.7) 不变性的反面教材）。
        huber_delta: Huber 核宽度（灰度），用于 IRLS 权重。
        lm_damping: Marquardt 阻尼 ``λ`` 初值（相对 ``diag(JᵀWJ)``）；
            在 :meth:`solve` 中按 Nielsen 增益比规则自适应（预测好的步
            放松、被拒步收紧）。
        guard: warp 后像素必须留在每个目标帧图像内这么多的像素边距
            （px；覆盖初始误差引起的纠正量）。
        seed: 子采样种子（确定性）。
    """

    def __init__(
        self,
        images: list[np.ndarray],
        ref_depth: np.ndarray,
        cam: PinholeCamera,
        poses_init: list[np.ndarray],
        top_frac: float = 0.3,
        max_points: int = 500,
        optimize_exposure: bool = True,
        huber_delta: float = 30.0,
        lm_damping: float = 1e-2,
        guard: float = 10.0,
        seed: int = 11,
    ) -> None:
        if len(images) != 3 or len(poses_init) != 3:
            raise ValueError("PhotometricBA expects exactly 3 keyframes")
        self._images = [np.asarray(im, dtype=float) for im in images]
        self._cam = cam
        self._optimize_exposure = bool(optimize_exposure)
        self._huber_delta = float(huber_delta)
        self._lm_damping = float(lm_damping)

        # Host 像素：第 0 帧半稠密梯度筛选 → 在每个目标帧初始 warp 下有效
        # → 确定性子采样。
        height, width = self._images[0].shape
        pixels = select_gradient_pixels(self._images[0], top_frac=top_frac)
        depth_sel = bilinear_sample(ref_depth, pixels)
        points0 = cam.unproject(pixels, depth_sel)   # 世界系 = 第 0 帧相机系
        for j in (1, 2):
            uv_j, depth_j = self._warp(points0, poses_init[j])
            inside = (
                (uv_j[:, 0] > guard) & (uv_j[:, 0] < width - 1 - guard)
                & (uv_j[:, 1] > guard) & (uv_j[:, 1] < height - 1 - guard)
                & (depth_j > 0.1)
            )
            pixels, depth_sel, points0 = pixels[inside], depth_sel[inside], points0[inside]
        if max_points < len(pixels):
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(pixels), max_points, replace=False))
            pixels, depth_sel = pixels[keep], depth_sel[keep]
        self._pixels = pixels

        # 点几何：host 帧光线上的逆深度，p = ray / ρ（依据：DSO 逆深度点
        # 参数化，模块 docstring"雅可比"）。
        self._rays = np.stack(
            [
                (pixels[:, 0] - cam.cx) / cam.fx,
                (pixels[:, 1] - cam.cy) / cam.fy,
                np.ones(len(pixels)),
            ],
            axis=-1,
        )
        self._rho0 = 1.0 / depth_sel
        self._i0_vals = bilinear_sample(self._images[0], pixels)

        # 当前（可变）状态；T[0] 是位姿规范锚点，永不动；0 号点是尺度规范
        # 锚点（教程 (5.9)），其逆深度保持在量测初值上。
        self._scale_anchor = 0
        # 逆深度的物理箱（DSO 会剔除偏离量测初始化过远的点）：:meth:`solve`
        # 对每步增量做裁剪，使候选状态保持
        # ``0.25 rho0_k <= rho_k <= 4 rho0_k``——任何一次野步都不可能把一个
        # 点甩出合理深度带（箱子取得很宽：优化器实际只需要百分之几的修正）。
        self._rho_lo = 0.25 * self._rho0
        self._rho_hi = 4.0 * self._rho0
        self._T = [np.asarray(T, dtype=float) for T in poses_init]
        self._rho = self._rho0.copy()
        self._a = np.zeros(3)
        self._b = np.zeros(3)
        self.result: PhotometricBAResult | None = None

    # ------------------------------------------------------------------ #
    # 公开只读接口                                                        #
    # ------------------------------------------------------------------ #
    @property
    def pixels(self) -> np.ndarray:
        """(N, 2) 点所 host 的第 0 帧像素坐标（px，列 (u, v)）。"""
        return self._pixels

    @property
    def n_points(self) -> int:
        """优化点数（= host 像素数）。"""
        return len(self._pixels)

    def parameter_layout(self) -> str:
        """增量向量 ``δx`` 的人类可读布局。

        ``δx = [δξ_1 (6) | δξ_2 (6) | δρ_1..δρ_N (N) | δa_1, δa_2 | δb_1, δb_2]``
        ——*当前*状态处的增量（切空间，教程 08.3）。曝光块仅在
        ``optimize_exposure`` 时激活；尺度规范锚点 ``δρ_0`` 永不激活
        （模块 docstring，规范自由度）。
        """
        exp_block = " | da_1, da_2 (2) | db_1, db_2 (2)" if self._optimize_exposure else ""
        return (
            "dx = [dxi_1 (6) | dxi_2 (6) | drho_1..drho_N (N)"
            f"{exp_block}], N = {self.n_points} (drho_{self._scale_anchor} fixed: scale gauge)"
        )

    # ------------------------------------------------------------------ #
    # 状态访问                                                            #
    # ------------------------------------------------------------------ #
    def _warp(self, points: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把世界系点按 ``T`` 投影；返回 (uv, 相机系深度)。"""
        q = transform_points(T, points)
        uv = self._cam.project(q)
        return uv, q[:, 2]

    def _points(self, rho: np.ndarray) -> np.ndarray:
        """逆深度 ``rho`` 对应的 (N, 3) 世界系点：``p = ray / ρ``（m）。"""
        return self._rays / rho[:, None]

    def residual_vector(self) -> np.ndarray:
        """当前状态下的堆叠光度残差 (6.6)，(2N,)。

        ``e_{kj} = (e^{a_j} I_j(u'_{kj}) + b_j) - (e^{a_0} I_0(u_k) + b_0)``，
        锚点曝光 ``(a_0, b_0) = (0, 0)``；第 1 帧块在前、第 2 帧块在后。
        """
        points = self._points(self._rho)
        n = self.n_points
        out = np.empty(2 * n)
        for row, frame in enumerate((1, 2)):
            uv, _ = self._warp(points, self._T[frame])
            vals, _, _ = _sample_with_derivative(self._images[frame], uv)
            out[row * n:(row + 1) * n] = (
                np.exp(self._a[frame]) * vals + self._b[frame] - self._i0_vals
            )
        return out

    def candidate_residuals(self, dx: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """候选状态 ``state ⊕ alpha·dx`` 处的残差（不提交）。

        位姿增量经指数映射 ``T_j ← exp(α δξ_j^∧) T_j``（8.9——精确 SE(3)）；
        深度与曝光加性更新。供 :meth:`solve` 的线搜索与测试中的有限差分
        检查使用。

        Args:
            dx: :meth:`parameter_layout` 布局的增量（全展长度 ``16 + N``；
                未激活的块由求解器忽略）。
            alpha: 沿 ``dx`` 的步长（无量纲）。

        Returns:
            (2N,) 候选状态处的残差（灰度，无量纲）。
        """
        n = self.n_points
        points = self._points(self._rho + alpha * dx[N_POSE_PARAMS:N_POSE_PARAMS + n])
        out = np.empty(2 * n)
        for row, frame in enumerate((1, 2)):
            T_cand = se3_exp(alpha * dx[6 * (frame - 1):6 * frame]) @ self._T[frame]
            uv, _ = self._warp(points, T_cand)
            vals, _, _ = _sample_with_derivative(self._images[frame], uv)
            a_j = self._a[frame] + alpha * dx[N_POSE_PARAMS + n + frame - 1]
            b_j = self._b[frame] + alpha * dx[N_POSE_PARAMS + n + 2 + frame - 1]
            out[row * n:(row + 1) * n] = np.exp(a_j) * vals + b_j - self._i0_vals
        return out

    def jacobian_matrix(self) -> np.ndarray:
        """当前状态处的解析雅可比 ``∂e/∂δx``，(2N, 16+N)。

        增量 ``δx`` 定义在当前状态（切空间，教程 08.3），故位姿块就是当前
        位姿的左扰动雅可比 (8.10)——链式法则 (6.5) 再乘仿射模型的权重
        ``e^{a_j}``：

        - 位姿块 ``δξ_j``：``e^{a_j} · g_{kj} · G(q_{kj})``，``G`` 为几何
          雅可比 (5.11)，``g`` 为采样器精确导数；
        - 逆深度 ``δρ_k``：由 ``p = ray/ρ`` 得 ``∂p/∂ρ = −ray/ρ²``，
          故 ``e^{a_j} g · (∂π/∂q R_j (−ray_k/ρ_k²))``；
        - 曝光 ``δa_j``：``e^{a_j} I_j(u')``；偏移 ``δb_j``：``1``。

        其余块全零（每个残差只触及一个位姿、一个点、一对 (a, b)——DSO
        论文 Fig. 5 的因子图稀疏性）。
        """
        n = self.n_points
        dim = N_POSE_PARAMS + n + 4
        points = self._points(self._rho)
        J = np.zeros((2 * n, dim))
        for row, frame in enumerate((1, 2)):
            q = transform_points(self._T[frame], points)
            uv = self._cam.project(q)
            vals, g_u, g_v = _sample_with_derivative(self._images[frame], uv)
            geo = projection_jacobian(q, self._cam)          # (N, 2, 6)，(5.11)
            w_j = np.exp(self._a[frame])
            col0 = 6 * (frame - 1)
            J[row * n:(row + 1) * n, col0:col0 + 6] = w_j * (
                g_u[:, None] * geo[:, 0, :] + g_v[:, None] * geo[:, 1, :]
            )
            # d p / d rho = -ray / rho^2  =>  d q / d rho = -R_j ray / rho^2；
            # 残差 k 只触及 rho_k -> 每行仅一个非零（对角块）。
            ray_t = self._rays @ self._T[frame][:3, :3].T          # R_j ray_k
            dpi_dq = geo[:, :, :3]                                 # (N, 2, 3)
            d_uv_drho = -np.einsum("nij,nj->ni", dpi_dq, ray_t) / self._rho[:, None] ** 2
            rows = np.arange(row * n, (row + 1) * n)
            J[rows, N_POSE_PARAMS + np.arange(n)] = w_j * (
                g_u * d_uv_drho[:, 0] + g_v * d_uv_drho[:, 1]
            )
            J[rows, N_POSE_PARAMS + n + frame - 1] = w_j * vals
            J[rows, N_POSE_PARAMS + n + 2 + frame - 1] = 1.0
        return J

    # ------------------------------------------------------------------ #
    # 求解器                                                              #
    # ------------------------------------------------------------------ #
    def _initialize_exposure(self, n_iters: int = 30) -> tuple[float, float, float, float]:
        """第一阶段：在位姿/深度固定下拟合 ``(a_1, a_2, b_1, b_2)``。

        warp 冻结在初始猜测时，残差
        ``e^{a_j} I_j(u'_j) + b_j − I_0(u_k)`` 是每帧 2 参数的仿射拟合
        ——光滑、量级小、条件数好，正适合 :func:`core.solver.gauss_newton`。
        其结果把亮度失配白化掉，使联合阶段从几何量级的残差起步
        （DSO 同样先自举光度参数）。

        Returns:
            ``(a_1, a_2, b_1, b_2)``：a 无量纲（log 增益），b 单位灰度。
        """
        n = self.n_points
        points0 = self._points(self._rho0)
        cached = []
        for frame in (1, 2):
            uv, _ = self._warp(points0, self._T[frame])
            cached.append(bilinear_sample(self._images[frame], uv))
        anchor = self._i0_vals

        def residual4(params: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [
                    np.exp(params[0]) * cached[0] + params[2] - anchor,
                    np.exp(params[1]) * cached[1] + params[3] - anchor,
                ]
            )

        def jacobian4(params: np.ndarray) -> np.ndarray:
            jac = np.zeros((2 * n, 4))
            jac[:n, 0] = np.exp(params[0]) * cached[0]
            jac[n:, 1] = np.exp(params[1]) * cached[1]
            jac[:n, 2] = 1.0
            jac[n:, 3] = 1.0
            return jac

        solution = gauss_newton(
            residual4,
            jacobian4,
            x0=np.zeros(4),
            n_iters=n_iters,
            lm_lambda=1e-6,
            robust_delta=None,   # 干净合成拟合：纯最小二乘（此处再套 IRLS
                                 # 会漂向远离白化解的 L1 型最优点）
        )
        return tuple(solution.x)  # type: ignore[return-value]

    def solve(self, n_iters: int = 50) -> PhotometricBAResult:
        """在堆叠光度目标 (6.8) 上跑流形 Gauss-Newton/LM。

        写开的循环（教程 (8.5)-(8.6)、(8.9)；DSO 论文 §3）：

        1. 在当前状态堆叠全部残差与解析雅可比；用
           :func:`core.solver.huber_weights` 的 Huber IRLS 权重降权外点
           像素（鲁棒核，教程 06.3 ③ / ch.08 §8.5）；
        2. 带 Marquardt 对角阻尼的稠密正规方程
           ``(JᵀWJ + λ·diag(JᵀWJ)) δx = −JᵀWe``——尺度感知的阻尼避免平面
           场景的弱可观测方向把步长撑爆（朴素 ``λI`` 不具备尺度感知）；
        3. 沿增量做线搜索，位姿经指数映射 (8.9) 移动：接受第一个
           ``α = 1, 1/2, 1/4, ...`` 中使鲁棒代价有限且严格下降的步长
           （保证单调进展与深度恒正）；
        4. 接受后重新定基线性化点（状态重定基），并按 Nielsen 增益比
           （gain ratio）规则更新 ``λ``——``λ ← λ·max(1/3, 1-(2ρ-1)³)``，
           ``ρ`` 为实际/预测代价下降之比（信赖域更新，教程 (8.6) 的依据）：
           被线搜索大幅砍短（线性化差）的步*提高* λ 而非放松，防止弱观测
           的逆深度方向在离最优点还远时迈出野步；被拒则收紧 ``λ × 10``。

        逆深度额外保持在量测初始化附近的物理箱内
        （``0.25 ρ0 ≤ ρ ≤ 4 ρ0``）；增量在*线搜索之前*裁剪，保证被评估的
        候选与最终提交的状态完全一致。

        ``optimize_exposure``（默认开）时，第一阶段先在位姿/深度固定下
        拟合 ``(a_j, b_j)``（见 :meth:`_initialize_exposure`）——亮度失配
        白化掉后，联合阶段从几何量级的残差起步。

        Args:
            n_iters: 联合阶段的 LM 最大迭代次数。

        Returns:
            :class:`PhotometricBAResult`（同时存入 ``self.result``）。
        """
        n = self.n_points
        if self._optimize_exposure:
            a1, a2, b1, b2 = self._initialize_exposure()
            self._a[1], self._a[2], self._b[1], self._b[2] = a1, a2, b1, b2

        e = self.residual_vector()
        w = huber_weights(e, self._huber_delta)
        cost = float(np.sum(w * e * e))
        lam = self._lm_damping
        converged = False
        iters = 0
        for iters in range(1, n_iters + 1):
            J = self.jacobian_matrix()
            gauge_free = [N_POSE_PARAMS + i for i in range(n) if i != self._scale_anchor]
            active = np.concatenate(
                [
                    np.arange(N_POSE_PARAMS),
                    gauge_free,
                    np.arange(N_POSE_PARAMS + n, N_POSE_PARAMS + n + 4)
                    if self._optimize_exposure
                    else np.array([], dtype=int),
                ]
            )
            J_act = J[:, active]
            H = J_act.T @ (w[:, None] * J_act)
            g = J_act.T @ (w * e)
            damping = lam * np.clip(np.diag(H), 1e-12, None)
            try:
                dx_active = -np.linalg.solve(H + np.diag(damping), g)
            except np.linalg.LinAlgError:
                dx_active = -np.linalg.lstsq(H + np.diag(damping), g, rcond=None)[0]
            dx = np.zeros(N_POSE_PARAMS + n + 4)
            dx[active] = dx_active
            # 逆深度的物理箱：裁剪增量，使每个候选状态保持
            # 0.25 rho0 <= rho <= 4 rho0（量测深度带，见 __init__）。
            # 在线搜索*之前*裁剪，保证被评估的候选与最终提交的状态一致。
            dx[N_POSE_PARAMS:N_POSE_PARAMS + n] = np.clip(
                dx[N_POSE_PARAMS:N_POSE_PARAMS + n],
                self._rho_lo - self._rho,
                self._rho_hi - self._rho,
            )

            alpha = 1.0
            accepted = False
            for _ in range(16):                     # 回溯线搜索
                try:
                    e_new = self.candidate_residuals(dx, alpha)
                except ValueError:                  # 有点落到相机背后（深度 <= 0）
                    alpha *= 0.5
                    continue
                if not np.all(np.isfinite(e_new)):
                    alpha *= 0.5
                    continue
                w_new = huber_weights(e_new, self._huber_delta)
                cost_new = float(np.sum(w_new * e_new * e_new))
                if cost_new < cost:
                    improvement = cost - cost_new
                    e, w, cost = e_new, w_new, cost_new
                    self._T[1] = se3_exp(alpha * dx[0:6]) @ self._T[1]
                    self._T[2] = se3_exp(alpha * dx[6:12]) @ self._T[2]
                    self._rho = self._rho + alpha * dx[N_POSE_PARAMS:N_POSE_PARAMS + n]
                    self._a[1:] += alpha * dx[N_POSE_PARAMS + n:N_POSE_PARAMS + n + 2]
                    self._b[1:] += alpha * dx[N_POSE_PARAMS + n + 2:N_POSE_PARAMS + n + 4]
                    # Nielsen 增益比阻尼更新（信赖域更新）：lambda 只按二次
                    # 模型对实际下降预测得有多好来放松。被线搜索大幅砍短
                    # （增益比 << 1，线性化差）的步会*提高* lambda——这道
                    # 闸门防止弱观测的逆深度方向在状态离最优点尚远时迈出
                    # 野步。
                    predicted = -(
                        alpha * float(g @ dx_active)
                        + 0.5 * alpha * alpha * float(dx_active @ (H @ dx_active))
                    )
                    ratio = improvement / predicted if predicted > 1e-12 else 1.0
                    lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                lam = min(lam * 10.0, 1e12)         # 拒绝：收紧阻尼
                if lam >= 1e12:
                    break
                continue
            if improvement < 1e-9 * max(cost, 1.0) or np.linalg.norm(alpha * dx) < 1e-12:
                converged = True
                break

        result = PhotometricBAResult(
            poses=[self._T[0], self._T[1], self._T[2]],
            exposure_a=self._a.copy(),
            exposure_b=self._b.copy(),
            inverse_depths=self._rho.copy(),
            points=self._points(self._rho),
            cost=cost,
            n_iters=iters,
            converged=converged,
        )
        self.result = result
        return result

"""projects/slam/photoba 的测试——直跑：``python tests/test_photoba.py``。

验证第 06/08 章教学实现的滑窗*光度*光束平差（DSO 思想；Engel, Koltun &
Cremers, "Direct Sparse Odometry", arXiv:1607.02565,
papers/slam/classics/）：DSO 成像模型（论文 Eq. (2)-(4)）
``I_i(x) = G(t_i V(x) B_i(x))``（Eq. 2）、光度校正 ``I'_i = t_i B_i``
（Eq. 3）与仿射亮度传递 ``e^{-a_i}(I_i - b_i)``（Eq. 4，其正向形式代换为
``e^{a_i} I_i + b_i``），与位姿、每点逆深度（inverse depth）一起进入
论文 Eq. (8) / 教程 (6.8) 的滑窗目标

    min_{ {T_i}, {rho_k}, {a_i, b_i} }  sum_k sum_j || e_kj ||^2_Huber,

其中光度残差为教程 (6.6)

    e_kj = (e^{a_j} I_j(u'_kj) + b_j) - (e^{a_0} I_0(u_k) + b_0),

其在 ``I -> gamma I + delta`` 下的不变性（教程 (6.7)）正是下方曝光测试
所利用的性质。流形 LM 循环遵循教程 (8.5)-(8.6)、(8.9)-(8.10)
（左扰动雅可比、指数映射更新）。

各阈值与实测值（确定性场景：seed 11 纹理、seed 2026 扰动、numpy 1.26.4；
全套约 5 s）
------------------------------------------------------------------------
test_photometric_ba_recovers_window（阈值 -> 实测）：
    位姿平移 L2               < 1e-2 m    -> 4.1e-3 m   （第 1/2 帧，取最大）
    位姿旋转 so3_log 范数      < 1e-2 rad  -> 1.3e-3 rad
    点位置，MEDIAN             < 1e-2 m    -> 7.3e-3 m
    点位置，p90                < 5e-2 m    -> 2.2e-2 m
    曝光 a（第 1/2 帧）        < 1e-2      -> 3.2e-3
    曝光 b（第 1/2 帧）        < 0.6 灰度  -> 3.8e-1
test_exposure_compensation_is_what_buys_accuracy：
    曝光关闭（(a, b) 冻结在 0）时平移误差 -> 8.9e-2 m，
    即曝光开启时 4.1e-3 m 的 ~22 倍（断言 >5 倍且 >3e-2）；
    点误差中位数 1.6 m，代价 5.8e4 对 6.9——没有白化，优化器连几何量级的
    残差底限都够不到。
test_huber_kernel_robust_to_outliers（200 个 host 点中污染 6 个，
两个目标帧各 +50 灰度；huber_delta=10 对 1e9 = 纯最小二乘）：
    平移 1.05e-2 对 3.28e-2 m（断言 huber < 关核/2，实测约 1/3.1）；
    点误差中位数 2.11e-2 对 1.50e-1 m（断言 < 关核/2，实测约 1/7.1）；
    曝光 b 误差 6.9e-1 对 1.03 灰度（断言严格更优；b 通道由渲染底限主导，
    见下，故差距只有 ~1.5 倍）。

相对朴素"处处 1e-2"指标的两处已注明的放宽，根源都在*渲染器*的重采样
底限（direct.PlaneScene.render_target 的正向 splat 目标渲染与 BA 的逆向
双线性重采样相差 O(0.1) 灰度的位置相关底限，故代价最优点并非严格真值）：

1. ``b`` 恢复不到 1e-2：每帧仿射偏移会吸收底限的*均值*——从真值初始化
   出发求解本身仍留下 b_err = 2.7e-1 灰度（底限所致），故阈值放宽到
   0.6 灰度。增益 ``a`` 与几何的底限远低于 1e-2，保留原指标。
2. 点误差存在弱纹理尾部（warp 落在近平坦双线性纹理胞元内的点几乎不携带
   深度信息——正是教程 06.1 ⑤ 的信息量判据）：测试因此断言中位数
   （< 1e-2，指标原值）与 90 分位，只*打印*最大值（全套最差观测：
   个别点 ~0.2-1 m）。

规范自由度（模块 docstring）：亮度锚点 (a_0, b_0) = (0, 0)、位姿锚点
T_0 = I、尺度锚点 rho_0 固定——数据以相同规范生成，故真值可逐字恢复。
"""
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lie import se3_exp, se3_log, transform_points  # noqa: E402
from direct import bilinear_sample  # noqa: E402
from photoba import PhotometricBA, make_keyframe_scene  # noqa: E402
from photoba.photoba import EXPOSURE_A_GT, EXPOSURE_B_GT  # noqa: E402

# 共享场景：纹理平面上的 3 关键帧窗口。真值曝光 (a, b) 在第 1/2 帧非零
# （第 0 帧是亮度规范锚点）：a = (0, 0.2, -0.15)，b = (0, 3, -2)。
SCENE = make_keyframe_scene()

# 初值扰动（依据：教程 06.2 ③ "光度误差非凸，需较好初值"）：每帧每轴至多
# 1.5 cm 平移 / 7.5 mrad 旋转，外加初始化逆深度的 host 帧深度图上 ±0.5%
# 的乘性噪声。
POSE_SCALE = np.array([0.015, 0.015, 0.015, 0.0075, 0.0075, 0.0075])
DEPTH_NOISE = 0.005
PERTURB_SEED = 2026

POSE_TOL = 1e-2          # 平移 L2 与旋转范数（指标原值）
POINT_MEDIAN_TOL = 1e-2  # 指标原值；弱纹理尾部由 p90 兜底
POINT_P90_TOL = 5e-2
EXPOSURE_A_TOL = 1e-2
EXPOSURE_B_TOL = 0.6     # 受渲染底限限制，见文件头说明第 1 条

MAX_POINTS = 200
N_ITERS = 60
HUBER_N_ITERS = 100   # 外点场景收敛慢：比较的是接近收敛的两个最优点，
                      # 而非中途状态（纯最小二乘的误差随外点持续拉扯单调
                      # 增长，迭代次数不同会让对比失真）


def _perturbed_init(seed: int):
    """扰动后的关键帧位姿 + 含噪深度图（每个 seed 确定性）。"""
    rng = np.random.default_rng(seed)
    poses_init = [SCENE.poses[0]]
    for j in (1, 2):
        poses_init.append(se3_exp(rng.uniform(-1.0, 1.0, 6) * POSE_SCALE) @ SCENE.poses[j])
    depth_noise = np.random.default_rng(seed + 1).uniform(
        -DEPTH_NOISE, DEPTH_NOISE, SCENE.depth.shape
    )
    return poses_init, SCENE.depth * (1.0 + depth_noise)


@lru_cache(maxsize=1)
def _clean_recovery():
    """共享场景的曝光开启 BA，只求解一次。

    Returns:
        ``(ba, result)``；实例在 :meth:`solve` 后状态冻结，跨测试共享该
        二元组是安全的。
    """
    poses_init, depth_init = _perturbed_init(PERTURB_SEED)
    ba = PhotometricBA(
        SCENE.images, depth_init, SCENE.cam, poses_init,
        max_points=MAX_POINTS, seed=11,
    )
    return ba, ba.solve(n_iters=N_ITERS)


def _point_errors(ba: PhotometricBA, result) -> np.ndarray:
    """每个优化点对真值平面点的 L2 误差（m）。"""
    true_pts = SCENE.cam.unproject(ba.pixels, bilinear_sample(SCENE.depth, ba.pixels))
    return np.linalg.norm(result.points - true_pts, axis=1)


def _pose_errors(result) -> tuple[float, float]:
    """第 1/2 帧上（平移 L2, 旋转 so3_log 范数）的最大值。"""
    dt, dr = [], []
    for j in (1, 2):
        xi = se3_log(result.poses[j] @ np.linalg.inv(SCENE.poses[j]))
        dt.append(float(np.linalg.norm(xi[:3])))
        dr.append(float(np.linalg.norm(xi[3:])))
    return max(dt), max(dr)


def test_photometric_ba_recovers_window():
    """基础恢复（测试一）：从扰动的位姿/深度初值出发，滑窗光度 BA
    （教程 (6.8)）联合恢复位姿、点位置与教程 (6.6) 的仿射曝光 (a, b)
    ——数据由同一模型生成，真值就是优化器的最优点。"""
    ba, result = _clean_recovery()
    assert result.converged, f"LM did not converge (iters={result.n_iters})"

    # 规范锚点保持钉住：T_0 固定在初值，(a_0, b_0)=(0,0)。
    assert np.allclose(result.poses[0], SCENE.poses[0])
    assert result.exposure_a[0] == 0.0 and result.exposure_b[0] == 0.0

    dt, dr = _pose_errors(result)
    perr = _point_errors(ba, result)
    a_err = float(np.max(np.abs(result.exposure_a[1:] - EXPOSURE_A_GT[1:])))
    b_err = float(np.max(np.abs(result.exposure_b[1:] - EXPOSURE_B_GT[1:])))
    print(
        f"\n[photoba] recovery: dt={dt:.2e} m, dr={dr:.2e} rad, "
        f"points med={np.median(perr):.2e} p90={np.quantile(perr, 0.9):.2e} "
        f"max={perr.max():.2e} m, a_err={a_err:.2e}, b_err={b_err:.2e} gray"
    )
    assert dt < POSE_TOL, dt
    assert dr < POSE_TOL, dr
    assert float(np.median(perr)) < POINT_MEDIAN_TOL, np.median(perr)
    assert float(np.quantile(perr, 0.9)) < POINT_P90_TOL, np.quantile(perr, 0.9)
    assert a_err < EXPOSURE_A_TOL, a_err
    assert b_err < EXPOSURE_B_TOL, b_err


def test_jacobian_matches_finite_differences():
    """解析雅可比 (2N, 16+N) 是光度残差在当前状态处的精确导数：位姿块
    （经指数映射的 (8.10)）、逆深度列（来自 p = ray/rho）、曝光列
    e^{a_j} I_j / 1。用 :meth:`PhotometricBA.candidate_residuals` 的中心
    差分检验（位姿经 exp(alpha dxi)^ 施加）——在扰动初值状态（a = b = 0）
    与收敛最优点（a ≠ 0）两处都应到机器精度。

    warp 后像素落在纹理胞元边界（整数 u 或 v）的 FD 步长范围内的行被排除：
    分片双线性采样器在那里*不可微*，中心差分会横跨两个胞元（依据：模块
    docstring "在 kink 之外"的注意事项）；其余行采样器光滑，解析导数必须
    精确。"""
    ba, _ = _clean_recovery()

    def worst_rel(ba_obj, poses, rho) -> float:
        """给定状态下逐列最差的相对 FD 失配。

        ``poses`` / ``rho`` 必须与雅可比求值时的状态*完全*一致（它们决定
        warp 位置，从而决定可微行集合）。"""
        e0 = ba_obj.residual_vector()
        jac = ba_obj.jacobian_matrix()

        # 可微行集合：两个目标帧的 warp 像素都至少离胞元边界（整数 u/v
        # ——分片双线性采样器在那里*不可微*，中心差分会横跨两个胞元，
        # 依据：模块 docstring "在 kink 之外"的注意事项）1e-4 px，且深入
        # 图像内部、远离采样器的边界钳位。用模块自己的数学从公开状态重建：
        # p = ray/rho, uv = pi(T_j p)。
        rays = np.stack(
            [
                (ba_obj.pixels[:, 0] - SCENE.cam.cx) / SCENE.cam.fx,
                (ba_obj.pixels[:, 1] - SCENE.cam.cy) / SCENE.cam.fy,
                np.ones(len(ba_obj.pixels)),
            ],
            axis=-1,
        )
        points = rays / rho[:, None]
        h, w = SCENE.images[0].shape
        frac_dist = np.full(len(points), np.inf)
        for j in (1, 2):
            uv = SCENE.cam.project(transform_points(poses[j], points))
            inside = (uv[:, 0] > 1.0) & (uv[:, 0] < w - 2) & (uv[:, 1] > 1.0) & (uv[:, 1] < h - 2)
            dist = np.minimum(np.abs(uv - np.round(uv)).min(axis=1), 1.0)
            frac_dist = np.where(inside, np.minimum(frac_dist, dist), 0.0)
        keep = frac_dist > 1e-4          # 远大于 FD eps=1e-6：不会横跨 kink
        # 残差行按"第 1 帧块 + 第 2 帧块"堆叠：同一点掩码适用于两半。
        keep_full = np.concatenate([keep, keep])

        rng = np.random.default_rng(3)
        n = ba_obj.n_points
        cols = list(range(12)) + [12 + n, 12 + n + 1, 12 + n + 2, 12 + n + 3]
        cols += sorted(rng.choice(np.arange(12, 12 + n), size=15, replace=False).tolist())
        eps = 1e-6
        worst = 0.0
        for i in cols:
            e_i = np.zeros(jac.shape[1])
            e_i[i] = 1.0
            fd = (ba_obj.candidate_residuals(e_i, eps) - ba_obj.candidate_residuals(e_i, -eps)) / (2 * eps)
            err = float(np.max(np.abs(jac[keep_full, i] - fd[keep_full])))
            worst = max(worst, err / max(float(np.max(np.abs(fd[keep_full]))), 1e-12))
        assert np.allclose(e0, ba_obj.residual_vector()), "FD probing mutated the state"
        return worst

    # 初值状态：未求解的实例，其状态恰为扰动构造数据（poses_init 与像素处
    # rho = 1/depth_init）。
    poses_init, depth_init = _perturbed_init(PERTURB_SEED)
    ba_init = PhotometricBA(
        SCENE.images, depth_init, SCENE.cam, poses_init,
        max_points=MAX_POINTS, seed=11,
    )
    rho_init = 1.0 / bilinear_sample(depth_init, ba_init.pixels)
    worst_init = worst_rel(ba_init, poses_init, rho_init)
    ba_opt, res_opt = _re_solved_small_ba()
    worst_opt = worst_rel(ba_opt, res_opt.poses, res_opt.inverse_depths)
    print(f"\n[photoba] jacobian vs FD: worst rel at init {worst_init:.2e}, "
          f"at optimum {worst_opt:.2e}")
    assert worst_init < 1e-6, worst_init
    assert worst_opt < 1e-6, worst_opt


def _re_solved_small_ba():
    """一个小窗口 BA，解到非平凡最优点（供 a ≠ 0 处的 FD 检查，避免重解
    大窗口）。"""
    poses_init, depth_init = _perturbed_init(PERTURB_SEED)
    ba = PhotometricBA(
        SCENE.images, depth_init, SCENE.cam, poses_init,
        top_frac=0.1, max_points=60, seed=11,
    )
    return ba, ba.solve(n_iters=N_ITERS)


def test_exposure_compensation_is_what_buys_accuracy():
    """曝光补偿的价值（测试二）：同一份带曝光数据（第 1/2 帧携带教程
    (6.6) 的真值仿射亮度），但把仿射参数冻结在 0
    （``optimize_exposure=False``，即关掉 (6.7) 不变性所对应的自由度），
    位姿估计退化一个数量级——亮度变化此时成为几何本身无法吸收的系统性
    光度偏差（为什么期望 ~22 倍：曝光误差以乘性增益 + 加性偏移进入每个
    残差，G-N 只能把这点系统性偏差摊到 12 维位姿/深度增量上，无法消零；
    故误差应停留在与曝光幅度同量级的平台上，而非收敛底限）。"""
    _, good = _clean_recovery()
    dt_on, _ = _pose_errors(good)

    poses_init, depth_init = _perturbed_init(PERTURB_SEED)
    ba = PhotometricBA(
        SCENE.images, depth_init, SCENE.cam, poses_init,
        max_points=MAX_POINTS, optimize_exposure=False, seed=11,
    )
    result = ba.solve(n_iters=N_ITERS)
    dt_off, dr_off = _pose_errors(result)
    perr = _point_errors(ba, result)
    print(
        f"\n[photoba] exposure off: dt={dt_off:.2e} m (on: {dt_on:.2e}), "
        f"dr={dr_off:.2e}, points med={np.median(perr):.2e} m, "
        f"cost={result.cost:.3e} (on: {good.cost:.3e}); "
        f"(a, b) frozen at {result.exposure_a[1:].tolist()} / {result.exposure_b[1:].tolist()}"
    )
    # 对照模式必须真的把曝光参数排除出状态。
    assert np.all(result.exposure_a[1:] == 0.0) and np.all(result.exposure_b[1:] == 0.0)
    assert dt_off > 3e-2, dt_off          # 高出指标一个数量级（平台效应）
    assert dt_off > 5.0 * dt_on, (dt_off, dt_on)   # 实测 ~22 倍


def test_huber_kernel_robust_to_outliers():
    """Huber 鲁棒性（测试三）：少数被污染像素（约 3% 的 host 点的 warp
    处，两个目标帧各 +50 灰度）必须使纯最小二乘（huber_delta -> 无穷）
    远比 Huber IRLS（鲁棒核，依据：教程 06.3 ③ / ch.08 §8.5，权重来自
    core.solver.huber_weights）受伤——为什么期望 ~3 倍量级差距：+50 灰度
    的外点在平方代价下权重为 1、主导正规方程，而 Huber 把它降权到
    delta/|r| ~ 10/50 = 0.2，影响近似按 sqrt(权重) 折进几何。"""
    poses_init, depth_init = _perturbed_init(PERTURB_SEED)
    probe = PhotometricBA(
        SCENE.images, depth_init, SCENE.cam, poses_init, max_points=MAX_POINTS, seed=11
    )
    pixels = probe.pixels                      # 第 1/2 帧未动：像素筛选结果相同
    n_out = max(4, len(pixels) // 33)          # 约 3% 的 host 点
    out_idx = np.sort(np.random.default_rng(5).choice(len(pixels), n_out, replace=False))
    images_out = [im.copy() for im in SCENE.images]
    for j in (1, 2):
        pts = SCENE.cam.unproject(pixels, bilinear_sample(SCENE.depth, pixels))
        uv = SCENE.cam.project(transform_points(SCENE.poses[j], pts))
        for i in out_idx:
            x0, y0 = int(np.floor(uv[i, 0])), int(np.floor(uv[i, 1]))
            images_out[j][y0:y0 + 2, x0:x0 + 2] += 50.0

    runs = {}
    for tag, delta in (("huber", 10.0), ("plain", 1e9)):
        ba = PhotometricBA(
            images_out, depth_init, SCENE.cam, poses_init,
            max_points=MAX_POINTS, huber_delta=delta, seed=11,
        )
        assert np.allclose(ba.pixels, pixels)
        runs[tag] = (ba, ba.solve(n_iters=HUBER_N_ITERS))

    (ba_h, res_h), (ba_p, res_p) = runs["huber"], runs["plain"]
    dt_h, _ = _pose_errors(res_h)
    dt_p, _ = _pose_errors(res_p)
    b_h = float(np.max(np.abs(res_h.exposure_b[1:] - EXPOSURE_B_GT[1:])))
    b_p = float(np.max(np.abs(res_p.exposure_b[1:] - EXPOSURE_B_GT[1:])))
    med_h = float(np.median(_point_errors(ba_h, res_h)))
    med_p = float(np.median(_point_errors(ba_p, res_p)))
    print(
        f"\n[photoba] outliers ({n_out} pts, +50 gray): huber10 dt={dt_h:.2e} "
        f"b_err={b_h:.2e} med={med_h:.2e} (conv={res_h.converged}) | "
        f"plain dt={dt_p:.2e} b_err={b_p:.2e} med={med_p:.2e} (conv={res_p.converged})"
    )
    assert res_h.converged, res_h.n_iters
    assert dt_h < 0.5 * dt_p, (dt_h, dt_p)   # 实测约 1/3.1：外点降权后不再主导
    assert med_h < 0.5 * med_p, (med_h, med_p)
    # b 是最低噪的通道（其误差以渲染底限为主，见文件头）：Huber 仍一致更优，
    # 但差距只有 ~1.5 倍，故只断言严格更优而不要求减半。
    assert b_h < b_p, (b_h, b_p)


if __name__ == "__main__":
    # 单点测试：python tests/test_photoba.py <测试名子串>；不带参数 = 全部测试。
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and pat in k]
    if not fns:
        print(f"没有匹配 '{pat}' 的测试；可用测试：")
        for k in sorted(globals()):
            if k.startswith("test_"):
                print(" ", k)
        sys.exit(1)
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)

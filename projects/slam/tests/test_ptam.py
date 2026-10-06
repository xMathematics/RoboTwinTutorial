"""projects/slam/ptam 的测试 —— 可直跑：``python tests/test_ptam.py``。

验证 Klein & Murray, *PTAM: Parallel Tracking and Mapping for Small AR
Workspaces*, ISMAR 2007（papers/slam/classics/PTAM_ISMAR2007_KleinMurray.pdf）
教学代理（2D-3D 简化，见 ptam 模块 docstring）：

- 带纹理平面场景仿真器（确定性 seed、解析投影锚点）；
- §5 / Eq.(8)-(9) motion-only 跟踪（复用 core.solver 的 G-N 与
  epipolar 的重投影雅可比）：已知运动的位姿恢复与内点统计；
- §6.3 / Eq.(10)-(11) 关键帧局部 BA（X/Y/Z 三分法 + 稠密联合 G-N +
  流形重线性化）：雅可比对照有限差分、扰动地图的重投影 RMSE 收敛、
  端到端每个关键帧处 BA 后 RMSE 不高于 BA 前；
- §6.2 关键帧插入判据（帧距 + 平移/视差阈值）：关键帧数随视差增长
  且被帧距上界约束；
- 系统级：36 帧场景的 ATE（metrics.ate_rmse，Sim(3) 对齐吸收单目
  尺度规范自由度）、跟踪内点率 / 重投影 RMSE 健康值；
- 退化输入（无纹理空观测帧 / 静止帧）不崩溃、不插关键帧。

全部场景定种子、可复现；整套测试远快于 10 s。
"""
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lie import se3_exp, se3_log, transform_points  # noqa: E402
from epipolar import triangulate  # noqa: E402
from metrics import ate_rmse  # noqa: E402
from ptam import (  # noqa: E402
    DEFAULT_XI,
    KeyFrame,
    Ptam,
    ba_terms,
    camera_center,
    local_ba,
    mean_parallax,
    project_visible,
    render_planar_frames,
    run_ptam,
    simulate_planar_scene,
    track_frame,
)

#: ATE 预算（m）：实测 3 个种子 0.059 / 0.079 / 0.094（36 帧、~2.6 m 轨迹、
#: 0.5 px 像素噪声、每帧仅十几条 3D-2D 对应的教学规模）；阈值取最差值的
#: ~1.3 倍，真实实现错误（雅可比错 / 规范自由度未钉住）通常 > 0.3 m。
ATE_TOL = 0.12
N_FRAMES = 36


@lru_cache(maxsize=1)
def _main_run():
    """主端到端场景（定种子，lru_cache 使多个测试共享同一次运行）。"""
    scene = simulate_planar_scene(seed=0, n_points=160, x_range=(-2.0, 5.0),
                                  n_frames=N_FRAMES)
    return scene, run_ptam(scene)


# ------------------------------------------------------------- 仿真器 ----
def test_scene_simulator_deterministic_and_projection():
    """平面场景仿真器：定种子确定性、形状/取值域、解析投影锚点。"""
    s1 = simulate_planar_scene(seed=3, n_frames=20, n_points=90)
    s2 = simulate_planar_scene(seed=3, n_frames=20, n_points=90)
    assert np.array_equal(s1.points, s2.points)          # 定种子可复现
    assert np.array_equal(s1.texture, s2.texture)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(s1.observations, s2.observations))

    assert s1.points.shape == (90, 3) and s1.texture.shape == (90,)
    assert s1.texture.min() >= 0.0 and s1.texture.max() <= 1.0   # 灰度纹理
    assert len(s1.observations) == 20
    assert np.allclose(s1.poses_gt[0], np.eye(4))        # 世界系 = 第 0 帧相机系
    # 解析锚点：无噪声渲染下，第 0 帧（T = I）的路标投影 = 逐字针孔公式。
    uv0, ids0, depth0 = project_visible(s1.cam, s1.poses_gt[0], s1.points,
                                        s1.width, s1.height)
    quiet = render_planar_frames(s1.cam, s1.points, [s1.poses_gt[0]],
                                 s1.width, s1.height, pixel_noise_std=0.0)
    assert np.allclose(quiet[0][0], uv0, atol=0.0)       # 无噪 = project_visible
    x, y, z = s1.points[ids0[:, None], np.arange(3)].T
    u_ref = s1.cam.fx * x / z + s1.cam.cx                # 教程 (4.4)-(4.5)
    v_ref = s1.cam.fy * y / z + s1.cam.cy
    assert np.allclose(uv0[:, 0], u_ref, atol=1e-12)
    assert np.allclose(uv0[:, 1], v_ref, atol=1e-12)
    assert np.allclose(depth0, z, atol=1e-12)
    assert uv0[:, 0].min() > 8 and uv0[:, 0].max() < s1.width - 8   # 可见性边距
    # 视差工具：同一批像素自身视差 ~0（浮点下 arccos(1-eps) 的噪声地板）；
    # 平移 0.3 m @ ~4 m 深度 ≈ 4 度量级。
    par_self = mean_parallax(s1.cam, uv0, uv0)
    assert par_self < 1e-4, par_self
    T1 = se3_exp(np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0]))
    uv1, ids1, _ = project_visible(s1.cam, T1, s1.points, s1.width, s1.height)
    common, i0, i1 = np.intersect1d(ids0, ids1, return_indices=True)
    par = mean_parallax(s1.cam, uv0[i0], uv1[i1])
    assert len(common) > 20 and 2.0 < par < 7.0, par


# --------------------------------------------------------- 跟踪线程 ------
def test_track_frame_recovers_known_pose():
    """§5 motion-only 跟踪：带噪像素 + 5 cm 级初值扰动，位姿须恢复到
    噪声地板（0.5 px 噪声、~80 条对应 => 平移 ~mm 级、旋转 mrad 级）。"""
    scene = simulate_planar_scene(seed=11, n_frames=12, n_points=160)
    T_gt = scene.poses_gt[8]
    uv, ids, _ = project_visible(scene.cam, T_gt, scene.points,
                                 scene.width, scene.height)
    rng = np.random.default_rng(4)
    uv_noisy = uv + rng.normal(0.0, 0.5, uv.shape)
    T_init = se3_exp(np.array([0.05, -0.03, 0.02, 0.008, 0.006, -0.01])) @ T_gt
    res = track_frame(scene.cam, scene.points[ids], uv_noisy, T_init, gate_px=4.0)

    e_gt = se3_log(np.linalg.inv(T_gt) @ res.T_cw)
    t_err = float(np.linalg.norm(e_gt[:3]))
    r_err = float(np.linalg.norm(e_gt[3:]))
    print(f"\n[track] t-err {t_err:.5f} m, rot-err {r_err:.6f} rad, "
          f"inlier {res.inlier_ratio:.3f}, reproj {res.reproj_rmse:.3f} px, "
          f"n_obs {res.n_obs}")
    assert res.n_obs >= 40                   # 帧后半场景可见点数实测 47
    assert res.inlier_ratio > 0.95           # 教学数据无外点：内点率近 1
    assert res.reproj_rmse < 1.0             # 0.5 px 噪声的 RMSE 地板 ~0.7 px
    assert t_err < 0.02                      # ~fx=300、深度 4 m 的噪声地板
    assert r_err < 5e-3
    assert res.converged


# ------------------------------------------------------- 局部 BA --------
def test_ba_jacobian_matches_finite_differences():
    """局部 BA 的堆叠雅可比（位姿块 + 点块）对照中心差分（O(eps^2)）。

    用 GT 位姿/路标构造无噪小问题（2 关键帧 + 6 点），逐列核对
    :func:`ba_terms` 的解析雅可比——尤其点块 ``de/dX = -(dpi/dq) @ R_cw``
    与锚点帧无位姿列、行序 (u0, v0, u1, v1, ...) 的布局。

    雅可比约定（与 epipolar.pnp_refine / droidlite 的流形 G-N 一致）：
    位姿块是教程 (8.10) 的左扰动雅可比，定义在**线性化点 x = 0** 处——
    内层 G-N 迭代远离 0 时它是一阶近似（不动点处 x -> 0，精确）；点块
    因路标线性进入残差，对**任意**状态精确。故在 x = 0 处核对全部列、在
    非零 x 处核对点列。"""
    rng = np.random.default_rng(9)
    cam = simulate_planar_scene(seed=5, n_frames=2, n_points=6).cam
    T2 = se3_exp(np.array([0.25, 0.05, -0.1, 0.02, -0.03, 0.015]))
    X = np.column_stack([rng.uniform(-1.5, 1.5, 6),
                         rng.uniform(-1.0, 1.0, 6),
                         rng.uniform(3.0, 6.0, 6)])
    uv1 = cam.project(X)
    uv2 = cam.project(transform_points(T2, X))
    kfs = [
        KeyFrame(frame=0, T_cw=np.eye(4), ids=np.arange(6), uv=uv1),
        KeyFrame(frame=1, T_cw=T2.copy(), ids=np.arange(6), uv=uv2),
    ]
    points = np.vstack([X, np.full((3, 3), np.nan)])   # 末 3 行 = 未建图占位
    res_fn, jac_fn, dim, n_rows = ba_terms(cam, kfs, points, [0, 1])
    assert dim == 6 * 1 + 3 * 6 and n_rows == 2 * 12   # 1 自由位姿 + 6 点

    # ① 线性化点 x = 0：位姿列 + 点列全部精确（教程 (8.10) 的定义点）。
    J0 = jac_fn(np.zeros(dim))
    eps = 1e-6
    err0 = 0.0
    for k in range(dim):
        xp, xm = np.zeros(dim), np.zeros(dim)
        xp[k] += eps
        xm[k] -= eps
        col = (res_fn(xp) - res_fn(xm)) / (2.0 * eps)
        err0 = max(err0, float(np.abs(col - J0[:, k]).max()))
    # ② 非零状态下的点列：路标线性入残差 => 对任意 x 精确。
    x = rng.normal(scale=1e-2, size=dim)
    Jx = jac_fn(x)
    err_p = 0.0
    for k in range(6, dim):
        xp, xm = x.copy(), x.copy()
        xp[k] += eps
        xm[k] -= eps
        col = (res_fn(xp) - res_fn(xm)) / (2.0 * eps)
        err_p = max(err_p, float(np.abs(col - Jx[:, k]).max()))
    print(f"\n[ba-jacobian] dim {dim}, max |J_fd - J| @x=0 {err0:.3e}, "
          f"点列 @x!=0 {err_p:.3e}")
    # 门限理由：雅可比条目 O(fx/Z) ~ 100 px/m；中心差分在 eps=1e-6 下的
    # 舍入地板 ~ 1e-16*|e|/eps ≈ 1e-7（实测 2.3e-8）。实现错误通常是符号
    # /布局错（差一个块整体），误差 > 10。
    assert err0 < 1e-4, err0
    assert err_p < 1e-4, err_p


def test_local_ba_reduces_reprojection_rmse():
    """§6.3 局部 BA：被扰动的位姿/路标初值经 BA 后重投影 RMSE 大幅下降，
    且不低于纯跟踪水平；端到端主场景中每个关键帧处 BA 后 <= BA 前。"""
    # ---- 独立小问题：GT 建图 + 人工扰动初值（演示 BA 的收敛能力）----
    scene = simulate_planar_scene(seed=5, n_frames=4, n_points=120)
    cam = scene.cam
    k_poses = scene.poses_gt[:4]
    rng_uv = np.random.default_rng(7)
    uv_frames, id_frames = [], []
    for T in k_poses:
        uv, ids, _ = project_visible(cam, T, scene.points, scene.width, scene.height)
        uv_frames.append(uv + rng_uv.normal(0.0, 0.5, uv.shape))
        id_frames.append(ids)
    # 用 GT 位姿三角化公共观测作为地图初值（模拟"前端已建图"）。
    points = np.full((len(scene.points), 3), np.nan)
    T_rel = k_poses[1] @ np.linalg.inv(k_poses[0])
    common, ia, ib = np.intersect1d(id_frames[0], id_frames[1], return_indices=True)
    Xw = transform_points(np.linalg.inv(k_poses[0]),
                          triangulate(np.eye(4), T_rel,
                                      uv_frames[0][ia], uv_frames[1][ib], K=cam.matrix()))
    points[common] = Xw
    kfs = [KeyFrame(frame=k, T_cw=T.copy(), ids=idx, uv=uv)
           for k, (T, idx, uv) in enumerate(zip(k_poses, id_frames, uv_frames))]
    # 扰动：3 个自由位姿走 ~2 cm/15 mrad 的 SE(3) 噪声、点走 5 cm 噪声。
    rng = np.random.default_rng(13)
    for kf in kfs[1:]:
        kf.T_cw = se3_exp(rng.normal(0.0, 0.015, 6)) @ kf.T_cw
    mapped = np.nonzero(np.isfinite(points).all(axis=1))[0]
    points[mapped] += rng.normal(0.0, 0.05, (len(mapped), 3))
    res_fn0, _, dim0, _ = ba_terms(cam, kfs, points, [0, 1, 2, 3])
    e0 = res_fn0(np.zeros(dim0)).reshape(-1, 2)
    rmse_perturbed = float(np.sqrt(np.mean(np.sum(e0 * e0, axis=1))))

    ba = local_ba(cam, kfs, points, [0, 1, 2, 3])
    print(f"\n[local-ba] window RMSE {rmse_perturbed:.2f} px (扰动初值) -> "
          f"{ba.rmse_after:.3f} px (BA 后, 报告 {ba.rmse_before:.3f} -> "
          f"{ba.rmse_after:.3f}), n_outer {ba.n_outer}")
    assert rmse_perturbed > 5.0                     # 扰动初值明显偏离噪声地板
    assert ba.rmse_after < 0.35 * rmse_perturbed    # BA 把扰动基本拉回
    assert ba.rmse_after < 1.2                      # 0.5 px 噪声地板量级

    # ---- 端到端：主场景每个关键帧处 BA 后 RMSE 不高于 BA 前 ----
    _, out = _main_run()
    ba_logs = [l for l in out.logs
               if l.keyframe and l.ba_rmse_before == l.ba_rmse_before]
    assert len(ba_logs) >= 5
    before = np.array([l.ba_rmse_before for l in ba_logs])
    after = np.array([l.ba_rmse_after for l in ba_logs])
    print(f"[local-ba] e2e: {len(ba_logs)} 次 BA, RMSE(before) 均值 "
          f"{before.mean():.3f} px -> (after) 均值 {after.mean():.3f} px, "
          f"平均改进 {(1 - after.mean() / before.mean()) * 100:.1f}%")
    assert np.all(after <= before + 1e-9)           # BA 不恶化任何窗口
    assert after.mean() < 0.97 * before.mean()      # 平均改进 >= 3%（实测 ~7%）


# --------------------------------------------------------- 系统端到端 ----
def test_ptam_end_to_end_ate_and_tracking_quality():
    """36 帧主场景：跟踪/建图交替阶段全程跑通，ATE 低于预算、关键帧数
    有界、跟踪内点率与重投影 RMSE 落在噪声地板量级（健康值打印）。"""
    scene, out = _main_run()
    gt = np.array([camera_center(T) for T in scene.poses_gt])
    est = np.array([camera_center(T) for T in out.poses])
    ate = ate_rmse(est[out.tracked], gt[out.tracked])

    n_kf = len(out.keyframe_frames)
    tracked_logs = [l for l in out.logs if l.n_tracked > 0]
    last = tracked_logs[-1]
    print(f"\n[e2e] ATE {ate:.4f} m (tol {ATE_TOL}), keyframes {n_kf} "
          f"({out.keyframe_frames.tolist()}), mapped {out.n_mapped}/{len(scene.points)}")
    print(f"[e2e] tracked {int(out.tracked.sum())}/{N_FRAMES} 帧, "
          f"末帧 n_obs {last.n_tracked}, inlier {last.inlier_ratio:.3f}, "
          f"reproj {last.reproj_rmse:.3f} px")
    # 为什么期望这些值：实测 ATE 0.059 m（单目尺度规范自由度由 Sim(3) 对齐
    # 吸收后，漂移主要来自 motion-only 跟踪在局部 BA 间隔内的累积）；关键帧
    # 12 个（§6.2 帧距 >= 3 的上界 36/3+2）；内点率 1.0、重投影 ~0.9 px
    # （0.5 px 像素噪声的 RMSE 地板）。
    assert ate < ATE_TOL
    assert int(out.tracked.sum()) >= 30             # 初始化后全部帧有效跟踪
    assert 6 <= n_kf <= N_FRAMES // 3 + 2           # 有界：帧距判据给出上界
    assert last.inlier_ratio > 0.9
    assert last.reproj_rmse < 1.5
    assert out.n_mapped > 20                        # 地图确已增长
    # 跟踪失败的帧（若有）不得静默产出错误位姿：无观测/对应不足的帧保持
    # 上一帧位姿（"initialized" 帧是初始化产物，位姿本来就该更新，不在列）。
    for k in range(1, N_FRAMES):
        if out.logs[k].status in ("no-measurements", "awaiting-init",
                                  "too-few-correspondences"):
            assert np.allclose(out.poses[k], out.poses[k - 1], atol=0.0)


# --------------------------------------------------- 关键帧判据 ----------
def test_keyframe_growth_bounded_by_parallax():
    """§6.2 判据的两个方向：视差增长 => 关键帧数增长（但被帧距上界约束）；
    视差不足（慢运动）=> 停在初始化双帧；纯静止 => 只有锚定帧。"""
    n_frames = 30
    slow = run_ptam(simulate_planar_scene(
        seed=1, n_points=140, n_frames=n_frames, x_range=(-2.0, 5.0),
        xi=0.35 * DEFAULT_XI))
    fast = run_ptam(simulate_planar_scene(
        seed=1, n_points=140, n_frames=n_frames, x_range=(-2.0, 5.0),
        xi=1.7 * DEFAULT_XI))
    static = run_ptam(simulate_planar_scene(
        seed=1, n_points=140, n_frames=n_frames, x_range=(-2.0, 5.0),
        poses=[np.eye(4)] * n_frames))
    n_slow, n_fast, n_static = (len(r.keyframe_frames) for r in (slow, fast, static))
    print(f"\n[keyframes] slow(0.35x) {n_slow}, fast(1.7x) {n_fast}, "
          f"static {n_static}")
    # 慢运动：初始化后累计视差增长缓慢（实测 0.35x 下仅再插 2 帧）。
    assert 2 <= n_slow <= 4
    assert n_fast > n_slow                          # 视差增长 => 关键帧增长
    # 帧距 >= key_min_frames=3 给出上界（锚定/初始化帧除外）。
    assert n_fast <= n_frames // 3 + 2
    gaps = np.diff(fast.keyframe_frames)
    assert np.all(gaps >= 3)                        # 每对相邻关键帧满足帧距
    assert n_static == 1                            # 静止：只有第 0 帧锚定关键帧
    assert all(l.status == "awaiting-init" for l in static.logs[1:])


# ------------------------------------------------------- 退化输入 --------
def test_degenerate_textureless_and_static_frames():
    """无纹理（空观测）帧与静止帧：不崩溃、不插关键帧、位姿保持上一帧，
    且恢复运动后跟踪照常（论文 §5.6 "不干净帧不入图"防线的教学版）。"""
    scene = simulate_planar_scene(seed=2, n_points=160, x_range=(-2.0, 5.0),
                                  n_frames=N_FRAMES)
    obs = list(scene.observations)
    empty = (np.zeros((0, 2)), np.zeros(0, dtype=int))

    # 场景 A：全程无纹理 —— 永不初始化、零关键帧、位姿保持单位阵。
    texless_scene = simulate_planar_scene(
        seed=2, n_points=10, n_frames=8, x_range=(-2.0, 5.0))
    obs_none = [empty] * 8
    system = Ptam(texless_scene.cam, width=texless_scene.width,
                  height=texless_scene.height)
    for uv, ids in obs_none:
        log = system.process_frame(uv, ids)
        assert log.status == "no-measurements"
    assert len(system.keyframes) == 0
    assert np.allclose(system.last_pose, np.eye(4))

    # 场景 B：正常序列中途插入 2 个空观测帧 —— 安全跳过，之后照常跟踪。
    obs[10] = empty
    obs[11] = empty
    scene_b = simulate_planar_scene(seed=2, n_points=160, x_range=(-2.0, 5.0),
                                    n_frames=N_FRAMES)
    system_b = Ptam(scene_b.cam, width=scene_b.width, height=scene_b.height)
    poses = []
    logs = []
    for k in range(N_FRAMES):
        logs.append(system_b.process_frame(*obs[k]))
        poses.append(system_b.last_pose.copy())
    assert logs[10].status == "no-measurements" and logs[11].status == "no-measurements"
    assert np.allclose(poses[10], poses[9]) and np.allclose(poses[11], poses[10])
    assert not logs[10].keyframe and not logs[11].keyframe
    assert logs[12].n_tracked >= 6                  # 恢复后照常跟踪
    assert all(l.status != "tracked+keyframe" for l in logs[10:12])

    # 场景 C：静止段（重复同一 GT 位姿 6 帧，紧跟一个关键帧之后）——
    # 相邻静止帧的原始测量视差只剩像素噪声水平（真实系统"帧间视差不足
    # 则不入图"的物理依据），不插关键帧，跟踪重投影贴噪声地板。
    static_poses = scene_b.poses_gt[:16] + [scene_b.poses_gt[15]] * 6 \
        + scene_b.poses_gt[16:22]
    st = simulate_planar_scene(seed=2, n_points=160, x_range=(-2.0, 5.0),
                               poses=static_poses)
    out_c = run_ptam(st)
    static_span = [l for l in out_c.logs if 16 <= l.frame <= 21]
    kf_in_span = [l for l in static_span if l.keyframe]
    assert len(static_span) == 6 and not kf_in_span
    # 原始测量视差（不经位姿）：静止帧对 ~0.15 度（0.5 px 噪声地板），
    # 运动恢复后的帧对 ~1.2 度——判据的物理来源可复现。
    par_static = [
        mean_parallax(scene_b.cam, st.observations[k][0], st.observations[k + 1][0])
        for k in (17, 18, 19)
    ]
    (uv_a, id_a), (uv_b, id_b) = st.observations[22], st.observations[23]
    common_m, ia_m, ib_m = np.intersect1d(id_a, id_b, return_indices=True)
    par_moving = mean_parallax(scene_b.cam, uv_a[ia_m], uv_b[ib_m])
    print(f"\n[degenerate] 静止帧对视差 {[round(p, 3) for p in par_static]} 度, "
          f"运动帧对 {par_moving:.3f} 度, 关键帧 {out_c.keyframe_frames.tolist()}")
    assert all(p < 0.5 for p in par_static)
    assert par_moving > 1.0
    rmse_static = [l.reproj_rmse for l in static_span if l.n_tracked > 0]
    assert rmse_static and max(rmse_static) < 1.0   # 静止帧重投影贴噪声地板


if __name__ == "__main__":
    # 单点测试：python tests/test_ptam.py <测试名子串>；不带参数 = 全部测试。
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

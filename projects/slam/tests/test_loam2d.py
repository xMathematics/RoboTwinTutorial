"""projects/slam/loam2d 的测试 —— 可直跑：``python tests/test_loam2d.py``。

验证 Zhang & Singh, *LOAM: Lidar Odometry and Mapping in Real-time*, RSS 2014
（papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf）教学实现的 2D 约简版
（见 loam2d 模块 docstring）：合成房间 + 极坐标扫描渲染器（射线/线段 +
解析射线/圆求交）、Eq. (1) 平滑度与 §V-A 特征选取、Eq. (2) 点到线残差及其
解析 SE(2) 雅可比（对照有限差分）、Eq. (9)-(13) 帧间 odometry 配准（LM 经
core.solver.gauss_newton）、§IV/Fig. 9 的双速率（dual-rate）
odometry+mapping 流水线（mapping 误差须低于 odometry 漂移），以及无特征
圆形房间的可观测性退化（Eq. (1) 找不到任何边缘；``J^T J`` 的切向旋转方向
奇异 => 条件数爆炸、配准被判 unhealthy —— 对应特征法初值/可观测性讨论）。

全部场景定种子、可复现；整套测试远快于 10 s。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loam2d import (  # noqa: E402
    FeatureSet,
    RegistrationResult,
    Room,
    extract_features,
    make_room,
    point_line_residuals_jacobian,
    register_scan_to_map,
    register_scan_to_scan,
    render_scan,
    run_loam2d,
    scan_points,
    smoothness,
    wrap_angle,
)

#: 任务给定的 scan-odometry 精度预算（平移 / 旋转，单位 m / rad）。
T_TOL = 5e-2
R_TOL = 1e-2
N_BEAMS = 720


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SE(2) 位姿复合 ``a (+) b``（与 loam2d 同一约定）。"""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        [
            a[0] + c * b[0] - s * b[1],
            a[1] + s * b[0] + c * b[1],
            wrap_angle(a[2] + b[2]),
        ]
    )


def _relative(poses_abs: np.ndarray) -> np.ndarray:
    """(T, 3) 绝对位姿 -> 第 0 帧（运行参考系）下的相对位姿。"""
    c, s = np.cos(-poses_abs[0, 2]), np.sin(-poses_abs[0, 2])
    d = poses_abs[:, :2] - poses_abs[0, :2]
    out = np.empty_like(poses_abs)
    out[:, 0] = c * d[:, 0] - s * d[:, 1]
    out[:, 1] = s * d[:, 0] + c * d[:, 1]
    out[:, 2] = wrap_angle(poses_abs[:, 2] - poses_abs[0, 2])
    return out


def _straight_path(n_steps: int, step: tuple[float, float, float]) -> np.ndarray:
    """恒定 SE(2) 增量的轨迹 (T+1, 3)，起点为原点。"""
    poses = [np.zeros(3)]
    for _ in range(n_steps):
        poses.append(_compose(poses[-1], np.asarray(step, dtype=float)))
    return np.array(poses)


def test_make_room_and_render_scan():
    """房间基元 + 确定性极坐标渲染（含解析锚点校验）。"""
    room = make_room()
    assert room.segments.shape == (16, 2, 2)   # 4 面墙 + 3 根方柱 x 4 条边
    assert room.circles.shape == (0, 3)

    r0, ang = render_scan(room, np.zeros(3), range_noise_std=0.01, seed=3)
    r1, _ = render_scan(room, np.zeros(3), range_noise_std=0.01, seed=3)
    assert np.array_equal(r0, r1)              # 定种子生成（确定性可复现）
    assert np.all(np.isfinite(r0))             # 封闭房间：每条束都命中
    assert r0.min() > 0.5 and r0.max() < 30.0
    # 解析锚点：机体角 0 沿 +x -> 命中 x=+10 的墙，量程应为 10
    # （linspace(-pi, pi, 720, endpoint=False) 的第 360 条束）；用无噪声
    # 重渲染核对精确解析值。
    r_quiet, ang_q = render_scan(room, np.zeros(3))
    assert abs(r_quiet[N_BEAMS // 2] - 10.0) < 1e-12
    assert np.allclose(np.diff(ang_q), 2 * np.pi / N_BEAMS, rtol=0, atol=1e-12)

    # 开放场景：一段孤立墙 —— 只有角度 0 的那束命中。
    open_room = Room(
        segments=np.array([[[5.0, -1.0], [5.0, 1.0]]]),
        circles=np.zeros((0, 3)),
    )
    rr, _ = render_scan(open_room, np.zeros(3), n_beams=8)
    assert rr[4] == 5.0                        # linspace(-pi, pi, 8)[4] == 0
    assert np.isinf(rr).sum() == 7             # 其余束全部脱靶 -> inf

    # 解析圆形房间：传感器位于圆心 -> 每条量程恒等于半径。
    circle = make_room(shape="circle", radius=12.0)
    rc, _ = render_scan(circle, np.zeros(3), n_beams=64)
    assert np.allclose(rc, 12.0, atol=1e-9)
    # 原地旋转传感器不得改变任何量程——这正是
    # test_featureless_circular_room_degeneracy 所依赖的退化性质。
    rc_rot, _ = render_scan(circle, np.array([0.0, 0.0, 0.37]), n_beams=64)
    assert np.allclose(rc_rot, 12.0, atol=1e-9)


def test_smoothness_and_feature_selection():
    """Eq. (1) 数值与朴素重实现逐点一致；边缘点恰落在量程突变处、平面点
    落在光滑弧段（§V-A 规则）。"""
    ang_a = np.linspace(0.0, 1.2, 40)
    arc_a = 10.0 * np.stack([np.cos(ang_a), np.sin(ang_a)], axis=1)
    ang_b = np.linspace(1.4, 2.6, 40)
    arc_b = 9.0 * np.stack([np.cos(ang_b), np.sin(ang_b)], axis=1)
    pts = np.vstack([arc_a, arc_b])            # i=40 处半径由 10 跳到 9
    c = smoothness(pts, r=5)

    for i in (0, 7, 39, 40, 60, 79):           # 含扫描端点与突变点
        js = [j for j in range(max(0, i - 5), min(len(pts), i + 6)) if j != i]
        ref = sum(np.linalg.norm(pts[i] - pts[j]) for j in js) / (
            len(js) * np.linalg.norm(pts[i])
        )
        assert abs(c[i] - ref) < 1e-12         # Eq. (1) 逐字复算

    junction = 40
    far = np.delete(np.arange(len(pts)), np.arange(junction - 6, junction + 7))
    thr_edge = 0.5 * (c[far].max() + c[junction - 6: junction + 7].max())
    feats = extract_features(
        pts, r=5, c_edge_min=thr_edge, c_planar_max=c[far].max(),
        n_edge_max=10, n_planar_max=40,
    )
    assert len(feats.edge_idx) >= 1
    assert np.all(np.abs(feats.edge_idx - junction) <= 6)   # 边缘在突变处
    assert np.all(np.abs(feats.planar_idx - junction) > 6)  # 平面在光滑弧上
    assert not np.intersect1d(feats.edge_idx, feats.planar_idx).size
    # NMS：任意两个被选特征彼此间距 > r=5（依据：论文 §V-A "邻域内无已选
    # 点"的非极大值抑制）。
    sel = feats.edge_idx
    assert all(
        abs(sel[i] - sel[j]) > 5
        for i in range(len(sel)) for j in range(i + 1, len(sel))
    )


def test_point_line_jacobian_matches_finite_differences():
    """Eq. (2) 点到线残差的解析 SE(2) 雅可比对照中心差分（精确到
    O(eps^2)，见 loam2d 模块 docstring）。"""
    rng = np.random.default_rng(11)
    pose = np.array([0.3, -0.2, 0.4])
    pts = rng.normal(scale=3.0, size=(12, 2))
    lines = np.stack(
        [rng.normal(scale=3.0, size=(12, 2)), rng.normal(scale=3.0, size=(12, 2))],
        axis=1,
    )
    e0, J = point_line_residuals_jacobian(pose, pts, lines)
    # e == 带符号垂距 == ((q1 - X) x2 (q2 - q1)) / ||q2-q1||
    R = np.array(
        [[np.cos(0.4), np.sin(0.4)], [-np.sin(0.4), np.cos(0.4)]]
    )  # = R(theta).T，与实现中 X = pts @ R.T + t 一致
    X = pts @ R + pose[:2]
    w = lines[:, 1] - lines[:, 0]
    e_ref = (X - lines[:, 0])[:, 0] * w[:, 1] - (X - lines[:, 0])[:, 1] * w[:, 0]
    e_ref /= np.linalg.norm(w, axis=1)
    assert np.allclose(e0, e_ref, atol=1e-12)

    eps = 1e-7
    for k in range(3):
        dp, dm = pose.copy(), pose.copy()
        dp[k] += eps
        dm[k] -= eps
        ep, _ = point_line_residuals_jacobian(dp, pts, lines)
        em, _ = point_line_residuals_jacobian(dm, pts, lines)
        assert np.allclose(J[:, k], (ep - em) / (2.0 * eps), atol=1e-6), k


def test_odometry_registration_accuracy():
    """两帧已知运动：帧间配准须在 (5e-2 m, 1e-2 rad) 内恢复相对位姿
    —— LOAM Eq. (9)-(13)。"""
    room = make_room()
    gt = np.array([0.4, -0.2, 0.08])
    r0, ang = render_scan(room, np.zeros(3), range_noise_std=0.01, seed=5)
    r1, _ = render_scan(room, gt, range_noise_std=0.01, seed=6)
    f0 = extract_features(scan_points(r0, ang))
    f1 = extract_features(scan_points(r1, ang))
    assert isinstance(f0, FeatureSet)
    res = register_scan_to_scan(f1, scan_points(r0, ang), f0, np.zeros(3))
    assert isinstance(res, RegistrationResult) and res.converged
    assert res.n_residuals >= 3

    t_err = float(np.linalg.norm(res.pose[:2] - gt[:2]))
    r_err = float(abs(wrap_angle(res.pose[2] - gt[2])))
    print(f"\n[odometry] t-err {t_err:.4f} m (tol {T_TOL}), "
          f"th-err {r_err:.5f} rad (tol {R_TOL}), cond {res.condition_number:.1f}")
    assert t_err < T_TOL, t_err
    assert r_err < R_TOL, r_err
    # 为什么期望这个值：良态矩形房间应远低于任务预算——实测 t-err ~1.8 mm
    # （range 噪声 std=0.01 m 下单帧配准的噪声地板量级）、cond ~11.5，
    # 远低于 1e4 的健康上限（可观测性良好）。
    assert res.healthy and res.condition_number < 1e4


def test_dual_rate_mapping_reduces_drift():
    """26 帧扫描、每 5 帧 mapping：map 精化链的终端误差必须小于纯
    odometry 链（漂移在每个关键帧被重新锚定，§IV/Fig. 9）。"""
    room = make_room()
    gt = _straight_path(25, (0.35, 0.02, 0.012))          # 起点为原点
    scans = [
        render_scan(room, p, range_noise_std=0.01, seed=100 + t)[0]
        for t, p in enumerate(gt)
    ]
    ang = render_scan(room, np.zeros(3))[1]
    out = run_loam2d(scans, ang, mapping_every=5)

    gt_rel = _relative(gt)
    err_odom = np.linalg.norm(out.poses_odom[:, :2] - gt_rel[:, :2], axis=1)
    err_map = np.linalg.norm(out.poses_map[:, :2] - gt_rel[:, :2], axis=1)
    assert list(out.key_indices) == [5, 10, 15, 20, 25]   # mapping_every=5
    assert all(r.healthy for r in out.odom_results)
    assert all(r.healthy for r in out.map_results)

    print(f"\n[dual-rate] odom  final t-err {err_odom[-1]:.4f} m, "
          f"th-err {abs(wrap_angle(out.poses_odom[-1, 2] - gt_rel[-1, 2])):.5f} rad")
    print(f"[dual-rate] map   final t-err {err_map[-1]:.4f} m, "
          f"th-err {abs(wrap_angle(out.poses_map[-1, 2] - gt_rel[-1, 2])):.5f} rad")
    print("[dual-rate] map t-err at keyframes:",
          np.round(err_map[out.key_indices], 4).tolist())
    # 漂移抑制：终端误差严格更小，且关键帧处有界。
    assert err_map[-1] < err_odom[-1], (err_map[-1], err_odom[-1])
    # 为什么期望这个值：实测 odom 链终端 ~0.017 m、map 链终端 ~0.0026 m
    # （约 6.6x 提升）——每 5 帧一次的 scan-to-map 精化把帧间累积漂移重新
    # 锚定回地图；关键帧处 map 误差 < 2e-2 m（有界：单帧 odometry 噪声地板
    # 量级，不随帧数增长）。
    assert np.all(err_map[out.key_indices] < 2e-2)
    assert err_odom[-1] < T_TOL
    assert len(out.map_points) > 100                       # 地图确已累积


def test_featureless_circular_room_degeneracy():
    """无特征几何破坏 odometry：从圆心扫描圆形房间时，Eq. (1) 找不到任何
    边缘、``J^T J`` 的切向旋转方向奇异（条件数 -> inf）—— 旋转不可观测；
    同一流水线在矩形房间上可恢复同一运动。
    对应特征法初值/可观测性讨论（教程 05 章 / 第 10 章系统观）."""
    ang = np.linspace(-np.pi, np.pi, N_BEAMS, endpoint=False)
    dth = 0.2                                              # 纯原地旋转

    circle = make_room(shape="circle", radius=12.0)
    rc0, _ = render_scan(circle, np.zeros(3), angles=ang)
    rc1, _ = render_scan(circle, np.array([0.0, 0.0, dth]), angles=ang)
    f0c = extract_features(scan_points(rc0, ang))
    f1c = extract_features(scan_points(rc1, ang))
    assert len(f0c.edge_idx) == 0                          # 无任何边缘点
    res_c = register_scan_to_scan(f1c, scan_points(rc0, ang), f0c, np.zeros(3))
    th_err_c = abs(wrap_angle(res_c.pose[2] - dth))

    rect = make_room()
    rr0, _ = render_scan(rect, np.zeros(3), angles=ang)
    rr1, _ = render_scan(rect, np.array([0.0, 0.0, dth]), angles=ang)
    f0r = extract_features(scan_points(rr0, ang))
    f1r = extract_features(scan_points(rr1, ang))
    res_r = register_scan_to_scan(f1r, scan_points(rr0, ang), f0r, np.zeros(3))
    th_err_r = abs(wrap_angle(res_r.pose[2] - dth))

    print(f"\n[degeneracy] circle: cond {res_c.condition_number:.3e}, "
          f"healthy {res_c.healthy}, th-err {th_err_c:.3f} rad (motion {dth})")
    print(f"[degeneracy] rect  : cond {res_r.condition_number:.1f}, "
          f"healthy {res_r.healthy}, th-err {th_err_r:.5f} rad")
    # 退化必须被"检测出来"而非静默失败：圆形房间对原地旋转完全不变（旋转后
    # 点云逐点重合），切向旋转方向上 ``J^T J`` 曲率为 0 => 最小特征值 -> 0、
    # cond = inf、healthy=False；cond=inf 的含义即"该方向完全不可观测"。
    assert not np.isfinite(res_c.condition_number) or res_c.condition_number > 1e8
    assert not res_c.healthy
    assert th_err_c > 0.1                                  # 运动完全未恢复
    # 矩形房间对照组：以良态 Hessian 恢复同一运动（实测 cond ~9.6）。
    assert res_r.healthy and res_r.condition_number < 1e4
    assert th_err_r < R_TOL


def test_register_scan_to_map_refines_keyframe():
    """扫描-地图配准（§VI 特征值直线拟合）把漂移的 odometry 位姿向地图
    精化：从 5 cm 扰动初值出发，终端须远低于单帧 odometry 的噪声地板。"""
    room = make_room()
    gt = np.array([0.0, 0.0, 0.05])
    # 用短轨迹上的 6 帧扫描建一张小地图（地图点放在真值位姿处，即理想化的
    # 累积地图）。
    poses = [gt]
    for _ in range(5):
        poses.append(_compose(poses[-1], np.array([0.15, 0.01, 0.004])))
    ang = render_scan(room, np.zeros(3))[1]

    def to_world(points: np.ndarray, p: np.ndarray) -> np.ndarray:
        """机体系点 -> 世界（第 0 帧）系点：X = R(theta) p + t。"""
        c, s = np.cos(p[2]), np.sin(p[2])
        return points @ np.array([[c, s], [-s, c]]) + p[:2]   # = pts @ R.T

    map_pts = np.vstack(
        [
            to_world(
                extract_features(
                    scan_points(render_scan(room, p)[0], ang)
                ).feature_points,
                p,
            )
            for p in poses
        ]
    )
    rng = np.random.default_rng(3)
    map_pts = map_pts + rng.normal(0.0, 0.01, map_pts.shape)
    r_new, _ = render_scan(room, _compose(poses[-1], np.array([0.15, 0.01, 0.004])),
                           range_noise_std=0.01, seed=77)
    f_new = extract_features(scan_points(r_new, ang))
    gt_new = _compose(poses[-1], np.array([0.15, 0.01, 0.004]))

    init = gt_new + np.array([0.05, -0.04, 0.01])          # 漂移的 odometry 位姿
    res = register_scan_to_map(f_new, map_pts, pose_init=init)
    t_err = float(np.linalg.norm(res.pose[:2] - gt_new[:2]))
    r_err = float(abs(wrap_angle(res.pose[2] - gt_new[2])))
    print(f"\n[scan-to-map] init t-err "
          f"{np.linalg.norm(init[:2] - gt_new[:2]):.3f} m -> final {t_err:.4f} m, "
          f"th-err {r_err:.5f} rad, cond {res.condition_number:.1f}")
    # 为什么期望这个值：实测 0.064 m -> 0.0009 m（th-err 4e-5 rad）——
    # 协方差特征值直线对应（§VI）在良态房间（cond ~11.6）中把 5 cm 扰动
    # 收敛到单帧噪声地板以下，即 mapping 精化确实"重新锚定"了位姿。
    assert res.converged and res.healthy
    assert t_err < 2e-2 and r_err < R_TOL


if __name__ == "__main__":
    # 单点测试：python tests/test_X.py <测试名子串>；不带参数 = 全部测试。
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

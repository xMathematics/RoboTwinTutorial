"""projects/slam/fastslam 的测试 —— 直接运行：``python tests/test_fastslam.py``。

验证第 07 章 §7.3 的 FastSLAM 教学实现（论文：Montemerlo & Thrun 等，
FastSLAM, AAAI 2002）：测量几何及其解析雅可比、首次观测的路标初始化、
系统重采样的均匀性与 N_eff、拒绝外点的马氏门控，以及两种关联模式
（已知关联与门控最近邻）下相对"仅里程计航位推算"基线的端到端精度。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastslam import (  # noqa: E402
    FastSLAM2D,
    range_bearing_jacobian,
    range_bearing_measurement,
    simulate_rectangle,
    wrap_angle,
)

# 共享场景常数：滤波器的噪声参数必须与仿真器的噪声生成一致（理想模型假设）。
# (0.04, 0.04, 0.02) 把位姿采样误差压得足够小，使门控最近邻关联在 N=50 时
# 不会误复制路标（见 GATE_DEFAULT 的双尺度分析）。
ODOM_NOISE_STD = (0.04, 0.04, 0.02)   # 机体系每步：(m, m, rad)
OBS_NOISE_STD = (0.15, 0.05)          # 测量：距离 (m)，方位角 (rad)
N_PARTICLES = 50
N_STEPS = 60
RUN_SEED = 7


def _dead_reckoning(sim):
    """仅里程计基线：用无噪运动模型（论文 Eq. (1) 的均值）积分带噪里程计，
    并在航位推算位姿处平均"逆测量"路标估计来构建地图
    （使用真值关联 —— 对基线已相当宽厚）。"""
    dr = np.empty_like(sim.gt_trajectory)
    dr[0] = sim.gt_trajectory[0]
    for t in range(sim.odometry.shape[0]):
        dx, dy, dth = sim.odometry[t]
        c, s = np.cos(dr[t, 2]), np.sin(dr[t, 2])
        dr[t + 1, 0] = dr[t, 0] + c * dx - s * dy
        dr[t + 1, 1] = dr[t, 1] + s * dx + c * dy
        dr[t + 1, 2] = wrap_angle(dr[t, 2] + dth)

    n_landmarks = sim.gt_landmarks.shape[0]
    acc = np.zeros((n_landmarks, 2))
    cnt = np.zeros(n_landmarks)
    for t in range(sim.odometry.shape[0]):
        pose = dr[t + 1]  # observations[t] 在真值位姿 t+1 处取得
        for k in range(sim.n_obs[t]):
            j = sim.associations[t, k]
            r, phi = sim.observations[t, k]
            ang = pose[2] + phi
            acc[j] += [pose[0] + r * np.cos(ang), pose[1] + r * np.sin(ang)]
            cnt[j] += 1
    return dr, acc / cnt[:, None]


def _run_fastslam(sim, association):
    """以指定关联模式在整段仿真上运行 FastSLAM。

    返回 ``(slam, est_trajectory)``，其中逐步位姿估计 ``(T, 3)``
    （每步之后取，与 ``gt_trajectory[1:]`` 对齐）。
    """
    slam = FastSLAM2D(
        n_particles=N_PARTICLES,
        n_landmarks=sim.gt_landmarks.shape[0],
        motion_noise_std=ODOM_NOISE_STD,
        obs_noise_std=OBS_NOISE_STD,
        association=association,
        seed=RUN_SEED,
    )
    slam.initialize(sim.gt_trajectory[0])
    est_traj = np.empty_like(sim.gt_trajectory[1:])
    for t in range(sim.odometry.shape[0]):
        k = sim.n_obs[t]
        assoc = sim.associations[t, :k] if association == "known" else None
        slam.step(sim.odometry[t], sim.observations[t, :k], assoc)
        est_traj[t] = slam.pose_estimate()
    return slam, est_traj


def _position_rmse(estimated, reference):
    """对 (x, y) 分量逐行计算 RMSE。"""
    err = np.asarray(estimated)[..., :2] - np.asarray(reference)[..., :2]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _greedy_match_map(est_map, true_map):
    """估计路标到真值路标的贪心一对一最近邻匹配。

    最近邻模式下需要它：槽位顺序是逐粒子的，可能与真值路标顺序不同。
    返回与 ``est_map`` 行对齐的真值位置，即给估计打分的参考基准：
    ``_position_rmse(est_map, matched)``。
    """
    est_rows = np.nonzero(~np.isnan(est_map[:, 0]))[0]
    dist = np.linalg.norm(
        est_map[est_rows][:, None, :] - true_map[None, :, :], axis=-1
    )
    matched = np.full_like(est_map, np.nan)
    while np.isfinite(dist).any():
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        matched[est_rows[i]] = true_map[j]
        dist[i, :] = np.inf
        dist[:, j] = np.inf
    return matched


def test_wrap_angle_wraps_into_range():
    assert abs(wrap_angle(0.0)) < 1e-12
    assert abs(wrap_angle(3.5) - (3.5 - 2.0 * np.pi)) < 1e-12
    assert abs(wrap_angle(-3.5) - (-3.5 + 2.0 * np.pi)) < 1e-12
    arr = wrap_angle(np.array([0.0, 3.5, -3.5, 7.0]))
    assert arr.shape == (4,)
    assert np.all(np.abs(arr) <= np.pi + 1e-12)


def test_range_bearing_jacobian_matches_finite_differences():
    pose = np.array([1.0, -0.5, 0.7])
    lm = pose[:2] + np.array([2.5, 1.5])
    H = range_bearing_jacobian(pose, lm)
    eps = 1e-7
    H_num = np.zeros((2, 2))
    for i in range(2):
        lm_p, lm_m = lm.copy(), lm.copy()
        lm_p[i] += eps
        lm_m[i] -= eps
        H_num[:, i] = (
            range_bearing_measurement(pose, lm_p)
            - range_bearing_measurement(pose, lm_m)
        ) / (2.0 * eps)
    assert np.allclose(H, H_num, atol=1e-6)
    # 预测测量应等于直接的距离 / 方位角公式。
    z = range_bearing_measurement(pose, lm)
    d = lm - pose[:2]
    assert np.allclose(
        z, [np.hypot(d[0], d[1]), wrap_angle(np.arctan2(d[1], d[0]) - pose[2])]
    )


def test_first_observation_initializes_landmark():
    """首次观测必须把路标恰好放在测量射线上（逆测量模型），
    协方差为 Sigma = J R J^T。"""
    sim = simulate_rectangle(seed=11)
    slam = FastSLAM2D(
        n_particles=5,
        n_landmarks=sim.gt_landmarks.shape[0],
        obs_noise_std=OBS_NOISE_STD,
        seed=11,
    )
    slam.initialize(sim.gt_trajectory[0])
    z = sim.observations[0, 0]
    j = int(sim.associations[0, 0])
    slam.step(np.zeros(3), z[None], np.array([j]))
    # 把每个粒子初始化后的均值重新投影，应复现该测量。
    for m in range(slam.n_particles):
        mu = slam.landmark_means[m, j]
        assert np.allclose(
            range_bearing_measurement(slam.poses[m], mu), z, atol=1e-9
        )
        cov = slam.landmark_covs[m, j]
        assert np.allclose(cov, cov.T)             # 对称
        assert np.linalg.eigvalsh(cov).min() > 0.0  # 正定
    assert slam.observed[:, j].all()  # 每个粒子都把该槽位标记为已观测


def test_systematic_resampling_produces_uniform_weights():
    slam = FastSLAM2D(n_particles=10, n_landmarks=3, seed=5)
    w = np.array([0.28, 0.22, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])
    slam.log_weights = np.log(w)
    assert abs(slam.weights().sum() - 1.0) < 1e-12
    assert np.isclose(
        slam.effective_sample_size(), 1.0 / np.sum(w * w), rtol=1e-9
    )
    # 用"标记均值"识别每个副本来自哪个原始粒子；
    # 系统重采样使每个复制数与 N * w_i 相差不超过 1。
    slam.landmark_means[:, 0, 0] = np.arange(10.0)
    slam.resample()
    counts = np.bincount(
        np.round(slam.landmark_means[:, 0, 0]).astype(int), minlength=10
    )
    assert np.all(np.abs(counts - 10.0 * w) < 1.0)
    # 重采样后权重均匀性：sum = 1 且没有粒子权重超过 0.5。
    w_after = slam.weights()
    assert abs(w_after.sum() - 1.0) < 1e-12
    assert w_after.max() <= 0.5
    assert slam.n_resamples == 1


def test_mahalanobis_gate_rejects_outlier():
    """远离门限的测量必须初始化新槽位，而不是污染已跟踪的路标
    （论文 Eq. (12) 门控）。"""
    sim = simulate_rectangle(seed=11)
    slam = FastSLAM2D(
        n_particles=10,
        n_landmarks=sim.gt_landmarks.shape[0],
        motion_noise_std=ODOM_NOISE_STD,
        obs_noise_std=OBS_NOISE_STD,
        association="nearest_neighbor",
        seed=11,
    )
    slam.initialize(sim.gt_trajectory[0])
    slam.step(np.zeros(3), sim.observations[0, :1], None)  # 干净的首次定位
    assert np.all(slam.observed.sum(axis=1) == 1)
    means_before = slam.landmark_means[:, 0].copy()
    # 方位角严重错误（距离相近时 +1 rad）：d^2 远超门限。
    z_bad = sim.observations[0, 0].copy()
    z_bad[1] += 1.0
    slam.step(np.zeros(3), z_bad[None], None)
    assert np.all(slam.observed.sum(axis=1) == 2)  # 开了新槽位，而非更新原路标
    assert np.allclose(slam.landmark_means[:, 0], means_before)  # 原路标未被触动


def test_known_correspondence_beats_dead_reckoning():
    sim = simulate_rectangle(
        n_steps=N_STEPS,
        odom_noise_std=ODOM_NOISE_STD,
        obs_noise_std=OBS_NOISE_STD,
        seed=RUN_SEED,
    )
    dr_traj, dr_map = _dead_reckoning(sim)
    slam, est_traj = _run_fastslam(sim, "known")

    pose_rmse = _position_rmse(slam.pose_estimate(), sim.gt_trajectory[-1])
    traj_rmse = _position_rmse(est_traj, sim.gt_trajectory[1:])
    map_est = slam.map_estimate()
    assert not np.isnan(map_est).any()  # 每个路标都已被建图
    map_rmse = _position_rmse(map_est, sim.gt_landmarks)
    dr_pose_rmse = _position_rmse(dr_traj[-1], sim.gt_trajectory[-1])
    dr_traj_rmse = _position_rmse(dr_traj, sim.gt_trajectory)
    dr_map_rmse = _position_rmse(dr_map, sim.gt_landmarks)

    # FastSLAM 必须明显胜过仅里程计的航位推算（RMSE < 基线的 1/3）。
    assert pose_rmse < dr_pose_rmse / 3.0, (pose_rmse, dr_pose_rmse)
    assert traj_rmse < dr_traj_rmse / 3.0, (traj_rmse, dr_traj_rmse)
    assert map_rmse < dr_map_rmse / 3.0, (map_rmse, dr_map_rmse)
    # 权重保持为合法分布；强制重采样后变回均匀。
    assert abs(slam.weights().sum() - 1.0) < 1e-12
    slam.resample()
    w = slam.weights()
    assert abs(w.sum() - 1.0) < 1e-12 and w.max() <= 0.5

    print(f"\n[known] pose RMSE {pose_rmse:.4f} m vs DR {dr_pose_rmse:.4f} m"
          f" (ratio {pose_rmse / dr_pose_rmse:.3f})")
    print(f"[known] traj RMSE {traj_rmse:.4f} m vs DR {dr_traj_rmse:.4f} m"
          f" (ratio {traj_rmse / dr_traj_rmse:.3f})")
    print(f"[known] map  RMSE {map_rmse:.4f} m vs DR {dr_map_rmse:.4f} m"
          f" (ratio {map_rmse / dr_map_rmse:.3f})")
    print(f"[known] resamples triggered: {slam.n_resamples}")


def test_nearest_neighbor_gating_beats_dead_reckoning():
    sim = simulate_rectangle(
        n_steps=N_STEPS,
        odom_noise_std=ODOM_NOISE_STD,
        obs_noise_std=OBS_NOISE_STD,
        seed=RUN_SEED,
    )
    dr_traj, dr_map = _dead_reckoning(sim)
    slam, est_traj = _run_fastslam(sim, "nearest_neighbor")

    pose_rmse = _position_rmse(slam.pose_estimate(), sim.gt_trajectory[-1])
    traj_rmse = _position_rmse(est_traj, sim.gt_trajectory[1:])
    dr_pose_rmse = _position_rmse(dr_traj[-1], sim.gt_trajectory[-1])
    dr_traj_rmse = _position_rmse(dr_traj, sim.gt_trajectory)
    dr_map_rmse = _position_rmse(dr_map, sim.gt_landmarks)

    # 门控机制必须已发现全部 20 个路标（每粒子每路标恰占一个槽位）。
    assert slam.observed.any(axis=0).all()
    assert int(slam.observed.sum(axis=1).min()) == sim.gt_landmarks.shape[0]
    map_matched = _greedy_match_map(slam.map_estimate(), sim.gt_landmarks)
    assert not np.isnan(map_matched).any()
    map_rmse = _position_rmse(slam.map_estimate(), map_matched)

    assert pose_rmse < dr_pose_rmse / 3.0, (pose_rmse, dr_pose_rmse)
    assert traj_rmse < dr_traj_rmse / 3.0, (traj_rmse, dr_traj_rmse)
    assert map_rmse < dr_map_rmse / 3.0, (map_rmse, dr_map_rmse)

    print(f"\n[nn-gate] pose RMSE {pose_rmse:.4f} m vs DR {dr_pose_rmse:.4f} m"
          f" (ratio {pose_rmse / dr_pose_rmse:.3f})")
    print(f"[nn-gate] traj RMSE {traj_rmse:.4f} m vs DR {dr_traj_rmse:.4f} m"
          f" (ratio {traj_rmse / dr_traj_rmse:.3f})")
    print(f"[nn-gate] map  RMSE {map_rmse:.4f} m vs DR {dr_map_rmse:.4f} m"
          f" (ratio {map_rmse / dr_map_rmse:.3f})")
    print(f"[nn-gate] resamples triggered: {slam.n_resamples}")


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

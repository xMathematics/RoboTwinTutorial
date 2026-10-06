"""FastSLAM 冒烟演示——本项目的核心入口脚本（VS Code 调试配置「SLAM: 核心入口」即指向本文件）。

函数流水线（本脚本在系统中的位置）：
    fastslam.simulate_rectangle （仿真器：矩形轨迹 + 20 路标 + 带噪里程计/观测）
        → FastSLAM2D.initialize / step （Rao-Blackwellized 粒子滤波：位姿粒子 × 路标 EKF，
                                        对应 FastSLAM 2.0 论式 (12) 的门控最近邻关联）
        → slam.pose_estimate()[:2]  （加权平均位姿）
        → 逐帧与真值求欧氏误差 → RMSE（与 projects/slam/metrics.py 的 ATE 口径一致的朴素实现）

输入/输出：无文件 I/O。输入为仿真器内存数据；输出为终端打印的 RMSE 与参考对照。
运行：`python demo.py`（任意 numpy>=1.26 环境；确定性种子，输出可复现）。
"""

import numpy as np

from fastslam import FastSLAM2D, simulate_rectangle


def main() -> None:
    """跑通一次 FastSLAM 2D 并打印 RMSE，作为调试断点入口。

    健康值：RMSE ≈ 0.221 m（50 粒子、seed=7）；纯航位推算对照 ≈ 1.57 m。
    异常信号：RMSE > 0.5 m 通常意味着门控过紧 / 粒子数不足 / 噪声参数与仿真器不一致
    （排查思路见 DEBUG.md §3 的 F-1～F-8 变量表）。
    """
    # ① 仿真数据：矩形轨迹 + 20 个路标，里程计与观测均含噪（deterministic seed）
    result = simulate_rectangle(seed=7)
    n_valid_landmarks = result.gt_landmarks.shape[0]

    # ② 建滤波器：50 个位姿粒子，每粒子维护全部路标的 EKF（论文式 (5)-(7)）
    slam = FastSLAM2D(n_particles=50, n_landmarks=n_valid_landmarks, seed=7)
    slam.initialize(result.gt_trajectory[0])  # 已知起点（x, y, θ）

    # ③ 逐步滤波：先运动预测后量测更新（马氏门控 + 最近邻关联 + 系统重采样）
    traj = []
    for t in range(len(result.odometry)):
        k = result.n_obs[t]  # 该帧有效观测数（其余为 NaN 填充，不入滤波）
        slam.step(result.odometry[t], result.observations[t, :k], result.associations[t, :k])
        traj.append(slam.pose_estimate()[:2])  # 加权平均位姿 (x, y)
    traj = np.array(traj)

    # ④ 评估：对齐真值（已知起点，直接逐帧比较）求 RMSE
    gt = result.gt_trajectory[1:, :2]  # 第 0 帧用于初始化，从第 1 帧起比较
    rmse = float(np.sqrt(np.mean(np.sum((traj - gt) ** 2, axis=1))))

    print(f"FastSLAM 2D 冒烟演示（{len(result.odometry)} 步，{n_valid_landmarks} 路标，50 粒子）")
    print(f"轨迹 RMSE: {rmse:.3f} m   （参考值 0.221 m；纯航位推算 ~1.57 m）")
    print(f"重采样次数: {slam.n_resamples}（参考值 ~35；调试观察点见 DEBUG.md §3 F-6）")


if __name__ == "__main__":
    main()

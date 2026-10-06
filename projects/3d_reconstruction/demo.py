"""3D 重建冒烟演示——本项目的核心入口脚本（VS Code 调试配置"3D 重建: 核心入口"即指向本文件）。

函数流水线（本脚本在系统中的位置）：
    scene.make_demo_scene （球 + 盒合成场景，SDF 外正内负，教程 (1.13)）
        -> scene.render_dataset （24 视角球追踪深度图，sigma(z) = c·z^2 噪声，教程 (4.2)）
        -> tsdf.TSDFVolume.integrate （逐帧融合：(4.1) 离散观测 -> (4.3) 截断 -> (4.8) 递归加权平均）
        -> marching.marching_tetrahedra （MT 提取：棱上插值 (5.1) + 梯度法向 (5.2)）
        -> metrics.sample_mesh_surface / chamfer_distance / accuracy_completion / f_score
                                        （表面采样 (M.4) + 三指标 (M.1)-(M.3)，与解析真值表面点比较）

输入/输出：无文件 I/O。输入为仿真器内存数据；输出为终端打印的指标表格。
运行：`python demo.py`（任意 numpy>=1.26 环境；确定性种子，输出可复现）。
"""

import time

import numpy as np

from marching import marching_tetrahedra
from metrics import (
    accuracy_completion,
    chamfer_distance,
    f_score,
    sample_mesh_surface,
)
from scene import make_demo_scene, render_dataset, sample_scene_surface
from tsdf import TSDFVolume, depth_to_point_cloud

# 演示配置（与 TUTORIAL.md 最小示例、METRICS.md 健康值保持一致）。
N_VIEWS = 24           # 视角数
IMG_SIZE = 56          # 深度图边长（px）
FOCAL = 50.0           # 焦距（px）
NOISE_COEFF = 0.002    # 深度噪声 sigma = c·z^2 的系数 c（m）
RESOLUTION = 64        # TSDF 每轴体素数
F_SCORE_TAU = 0.02     # F-score 容差（m）≈ 1 个体素
N_EVAL_POINTS = 8192   # 评测点数（重建表面 / 真值表面各一份）


def main() -> None:
    """跑通"场景 -> 多视角深度 -> TSDF 融合 -> MT 网格 -> 评测"全管线。

    健康值：TSDF+MT 的 Chamfer ≈ 1.3e-4 m^2（RMS ≈ 1.1 cm）、F@20mm ≈ 0.96；
    单视角基线因背面缺失被 completion 主导，Chamfer ≈ 2.7e-2 m^2、F@20mm ≈ 0.31
    （实测值见打印与 METRICS.md；异常排查见 DEBUG.md §3 的变量表）。
    """
    # ① 合成场景与解析真值表面点（评测基准；band 内薄壳采样 + 解析投影）。
    scene = make_demo_scene()
    gt_points = sample_scene_surface(scene, N_EVAL_POINTS, band=0.05, seed=0)

    # ② 24 视角 RGB-D 数据集（深度即数据；sigma(z) = 0.002·z^2 结构光误差模型）。
    t0 = time.perf_counter()
    data = render_dataset(scene, n_views=N_VIEWS, img_size=IMG_SIZE, focal=FOCAL,
                          depth_noise_coeff=NOISE_COEFF, seed=0)
    t_render = time.perf_counter() - t0

    # ③ TSDF 融合：世界包围盒 [-0.8, 0.8]^3、64^3 体素（voxel = 25 mm、mu = 75 mm）。
    volume = TSDFVolume(np.full(3, -0.8), np.full(3, 0.8), resolution=RESOLUTION)
    t0 = time.perf_counter()
    for T_wc, depth in zip(data["poses"], data["depths"]):
        volume.integrate(depth, data["K"], T_wc)
    t_fuse = time.perf_counter() - t0

    # ④ Marching Tetrahedra 提取网格（场样本在体素中心，原点平移半个体素；
    #    extraction_field 把未观测体素置 NaN，权重掩码提取避免幽灵内壁）。
    t0 = time.perf_counter()
    half = 0.5 * volume.voxel_size
    mesh = marching_tetrahedra(
        volume.extraction_field(), volume.origin + half, volume.voxel_size
    )
    t_extract = time.perf_counter() - t0

    # ⑤ 重建网格表面采样 + 指标（面积加权采样 (M.4) -> (M.1)-(M.3)）。
    rec_points = sample_mesh_surface(mesh.vertices, mesh.triangles,
                                     N_EVAL_POINTS, seed=0)
    cd = chamfer_distance(rec_points, gt_points)
    acc, comp = accuracy_completion(rec_points, gt_points)
    fs = f_score(rec_points, gt_points, F_SCORE_TAU)

    # ⑥ 对照基线：单视角深度点云（视角 0）——展示多视角融合的价值。
    base_points = depth_to_point_cloud(data["depths"][0], data["K"], data["poses"][0])
    cd_b = chamfer_distance(base_points, gt_points)
    acc_b, comp_b = accuracy_completion(base_points, gt_points)
    fs_b = f_score(base_points, gt_points, F_SCORE_TAU)

    print("=" * 74)
    print(f"3D 重建冒烟演示：球 r=0.35 @ 原点 + 盒半边 0.15 @ (0.60, 0, 0)")
    print(f"{N_VIEWS} 视角 {IMG_SIZE}x{IMG_SIZE} 深度（sigma = {NOISE_COEFF}·z^2）-> "
          f"TSDF {RESOLUTION}^3（voxel {volume.voxel_size * 1000:.0f} mm, "
          f"mu {volume.mu * 1000:.0f} mm）-> MT 网格")
    print(f"网格：{len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 三角形"
          f"（渲染 {t_render:.1f}s + 融合 {t_fuse:.1f}s + 提取 {t_extract:.1f}s）")
    print("-" * 74)
    print(f"{'方法':<28}{'Chamfer (m^2)':>14}{'Acc (m^2)':>12}"
          f"{'Comp (m^2)':>12}{'F@20mm':>9}")
    row = f"{'单视角深度点云（视角 0）':<26}{cd_b:>14.6f}{acc_b:>12.6f}{comp_b:>12.6f}{fs_b:>9.3f}"
    print(row)
    row = (f"{'TSDF 融合 + MT 网格':<26}{cd:>14.6f}{acc:>12.6f}"
           f"{comp:>12.6f}{fs:>9.3f}")
    print(row)
    print("=" * 74)


if __name__ == "__main__":
    main()

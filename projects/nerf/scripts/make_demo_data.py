"""生成一个迷你演示场景（彩色球体），输出为 nerf_synthetic 格式。

用途：无需下载庞大的官方 Blender 数据集即可冒烟测试训练。做法是对一个解析
球面做光线求交，多视角渲染出几何一致的图像对（同一球体、不同相机位姿）。

函数流水线：独立脚本，仅依赖 nerf.rays.get_rays_np——
    make_poses（圆轨迹位姿，look_at 构造 c2w）→ get_rays_np 逐像素光线
    → trace_sphere（光线-球求交 + Lambert 着色）→ 存 PNG + transforms_*.json。
输出目录可直接被 data_utils.load_blender_data 读取。

用法（Usage）：
    python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32
"""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np

# 把 nerf 包的父目录加进 sys.path，保证从任意工作目录都能导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nerf.rays import get_rays_np  # noqa: E402


def trace_sphere(rays_o, rays_d, center, radius):
    """光线-球体求交 → 每像素 RGB（Lambert 漫反射着色，未命中处为白色背景）。

    数学依据：解 |o + t·d - center|² = radius²（二次方程），判别式
    disc = b² - c；命中时取较近的根 t = -b - √disc，法线 = 交点相对球心的
    单位向量，亮度 = max(法线·光照方向, 0)。
    """
    oc = rays_o - center
    b = np.sum(oc * rays_d, axis=-1)          # 二次方程一次项系数的一半
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = b * b - c                           # 判别式：>0 表示光线与球相交
    t = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0.0)), -1.0)  # 取近端交点；未命中置 -1
    hit = t > 0
    normal = (rays_o + t[..., None] * rays_d - center)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([0.4, 0.7, -0.6])         # 固定方向光（归一化后用于 Lambert 着色）
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)[..., None]
    base = np.array([0.9, 0.15, 0.15]) * (0.3 + 0.7 * shade)  # 红色底色 × 环境光+漫反射
    rgb = np.where(hit[..., None], base, np.array([1.0, 1.0, 1.0]))
    return rgb


def look_at(pos, center=(0, 0, 0), up=(0, 1, 0)):
    """构造 c2w 矩阵（Blender 约定：相机朝 -z 看；与 run_nerf.look_at 同一套约定）。"""
    pos = np.asarray(pos, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z = pos - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


def make_poses(n, radius=4.0, elev=0.3):
    """生成 n 个绕原点（半径 radius、仰角 elev 弧度）的环形相机位姿，全部朝向原点。

    Returns: [n, 4, 4] 的 c2w 矩阵堆。
    """
    poses = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        pos = np.array(
            [radius * np.cos(theta), radius * np.sin(elev), radius * np.sin(theta)]
        )
        poses.append(look_at(pos, center=(0, 0, 0)))
    return np.stack(poses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/demo_scene")
    p.add_argument("--n_train", type=int, default=8)
    p.add_argument("--n_test", type=int, default=8)
    p.add_argument("--res", type=int, default=32, help="image resolution HxW")
    p.add_argument("--radius", type=float, default=4.0)
    args = p.parse_args()

    H = W = args.res
    angle_x = 0.691111  # 视场角，取值对齐 nerf_synthetic 官方场景
    focal = 0.5 * W / np.tan(0.5 * angle_x)   # 针孔模型：视场角 → 焦距（像素单位）
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.8  # 演示球体的半径（场景单位；相机轨迹半径由 --radius 控制，注意区分）

    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test"), exist_ok=True)

    # 三个 split 的相机轨迹：test 用更低仰角，模拟训练/测试视角不同
    splits = {
        "train": make_poses(args.n_train, args.radius),
        "val": make_poses(max(2, args.n_train // 4), args.radius),
        "test": make_poses(args.n_test, args.radius, elev=0.15),
    }
    for split, poses in splits.items():
        frames = []
        for i, c2w in enumerate(poses):
            ro, rd = get_rays_np(H, W, focal, c2w)
            img = trace_sphere(ro, rd, center, radius)
            imageio.v2.imwrite(
                os.path.join(args.out, split, f"r_{i:03d}.png"),
                (np.clip(img, 0, 1) * 255).astype(np.uint8),
            )
            # 记录与 nerf_synthetic 相同格式的 frame 元数据（路径 + 4x4 c2w）
            frames.append(
                {"file_path": f"./{split}/r_{i:03d}", "transform_matrix": c2w.tolist()}
            )
        with open(os.path.join(args.out, f"transforms_{split}.json"), "w") as f:
            json.dump({"camera_angle_x": angle_x, "frames": frames}, f, indent=2)

    print(f"[demo] wrote {args.out}  ({H}x{W}, {args.n_train} train / "
          f"{args.n_test} test images, focal={focal:.2f})")


if __name__ == "__main__":
    main()

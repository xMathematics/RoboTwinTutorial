"""Blender 合成数据集（nerf_synthetic）的数据加载。

函数流水线：本模块处于管线最前端的数据侧，被 run_nerf.py（train/test 模式）调用；
依赖 nerf.rays.get_rays_np 把每个相机位姿展开成逐像素光线。
输入：场景目录（下述布局）；输出：光线原点/方向网格、图像、位姿与相机内参。

单个场景（例如 ``lego``）的期望目录结构::

    lego/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    └── train/ val/ test/
        ├── r_0.png
        ├── r_1.png
        └── ...

每个 ``transforms_*.json`` 包含 ``camera_angle_x``（水平视场角，弧度）与 ``frames``
列表；每个 frame 有 ``file_path``（如 ``./train/r_0``）和 ``transform_matrix``
（4x4 相机到世界 c2w 矩阵，Blender 约定：相机朝 -z 看）。
"""
import json
import os

import imageio
import numpy as np

from .rays import get_rays_np


def _read_image(fname: str) -> np.ndarray:
    """读取一张 PNG，归一化到 [0, 1]，并把 alpha 通道合成到白色背景上。

    nerf_synthetic 的图像带透明通道；NeRF 训练约定背景为白色，故按
    rgb*a + (1-a) 做 alpha 混合（否则半透明像素会带黑色边）。
    """
    img = imageio.v2.imread(fname).astype(np.float32) / 255.0
    if img.shape[-1] == 4:  # RGBA → 白底 RGB（alpha 合成）
        rgb, a = img[..., :3], img[..., 3:]
        img = rgb * a + (1.0 - a)
    return img


def load_blender_data(
    basedir: str, split: str = "train", half_res: bool = True, testskip: int = 8
):
    """加载一个 nerf_synthetic 场景的某个 split。

    Args:
        basedir: 场景目录路径。
        split: 'train' | 'val' | 'test'，决定读取哪个 transforms_*.json。
        half_res: 分辨率减半（加快迭代；论文用全分辨率），焦距同步减半。
        testskip: split='test' 时每隔 testskip 帧取一帧（加速评测）。

    Returns:
        rays_o [N, H, W, 3]：光线原点（即相机中心，世界坐标）。
        rays_d [N, H, W, 3]：光线方向（未归一化，模长按针孔模型随像素位置变化）。
        imgs [N, H, W, 3]：真值图像，像素值 ∈ [0, 1]。
        poses [N, 4, 4]：相机到世界 c2w 矩阵。
        H, W: 图像高/宽（像素）；focal: 焦距（像素单位）。
    """
    meta = json.load(open(os.path.join(basedir, f"transforms_{split}.json")))
    frames = meta["frames"]
    if split == "test" and testskip > 0:
        frames = frames[::testskip]

    imgs, poses = [], []
    for frame in frames:
        fname = os.path.join(basedir, frame["file_path"] + ".png")
        imgs.append(_read_image(fname))
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
    imgs = np.stack(imgs)     # [N, H, W, 3]，像素值 ∈ [0, 1]
    poses = np.stack(poses)   # [N, 4, 4] c2w

    H, W = imgs.shape[1], imgs.shape[2]
    # 针孔模型：由水平视场角 camera_angle_x 反推焦距 focal（像素单位）
    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])

    if half_res:
        H, W = H // 2, W // 2
        focal = focal / 2.0  # 分辨率减半，焦距等比例减半
        imgs = imgs[:, ::2, ::2, :]  # 简单隔点降采样（教学用途足够）

    rays_o, rays_d = [], []
    for p in poses:
        ro, rd = get_rays_np(H, W, focal, p)
        rays_o.append(ro)
        rays_d.append(rd)
    rays_o = np.stack(rays_o)  # [N, H, W, 3]
    rays_d = np.stack(rays_d)  # [N, H, W, 3]
    return rays_o, rays_d, imgs, poses, H, W, focal

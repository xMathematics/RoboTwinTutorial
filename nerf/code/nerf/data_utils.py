"""Data loading for the Blender synthetic dataset (nerf_synthetic).

Expected layout for one scene, e.g. ``lego``::

    lego/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    └── train/ val/ test/
        ├── r_0.png
        ├── r_1.png
        └── ...

Each ``transforms_*.json`` has ``camera_angle_x`` and a list of ``frames`` with
``file_path`` (e.g. ``./train/r_0``) and ``transform_matrix`` (4x4 c2w).
"""
import json
import os

import imageio
import numpy as np

from .rays import get_rays_np


def _read_image(fname: str) -> np.ndarray:
    """Read a PNG, normalize to [0, 1], and composite the alpha over white."""
    img = imageio.v2.imread(fname).astype(np.float32) / 255.0
    if img.shape[-1] == 4:  # RGBA -> RGB over white background
        rgb, a = img[..., :3], img[..., 3:]
        img = rgb * a + (1.0 - a)
    return img


def load_blender_data(
    basedir: str, split: str = "train", half_res: bool = True, testskip: int = 8
):
    """Load a nerf_synthetic scene split.

    Args:
        basedir: path to the scene directory.
        split: 'train' | 'val' | 'test'.
        half_res: halve resolution (fast iteration; paper uses full res).
        testskip: use every testskip-th frame for 'test' (speeds up evaluation).

    Returns:
        rays_o [N, H, W, 3], rays_d [N, H, W, 3], imgs [N, H, W, 3],
        poses [N, 4, 4], H, W, focal.
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
    imgs = np.stack(imgs)     # [N, H, W, 3]
    poses = np.stack(poses)   # [N, 4, 4]

    H, W = imgs.shape[1], imgs.shape[2]
    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])

    if half_res:
        H, W = H // 2, W // 2
        focal = focal / 2.0
        imgs = imgs[:, ::2, ::2, :]  # simple downsample (fine for teaching)

    rays_o, rays_d = [], []
    for p in poses:
        ro, rd = get_rays_np(H, W, focal, p)
        rays_o.append(ro)
        rays_d.append(rd)
    rays_o = np.stack(rays_o)  # [N, H, W, 3]
    rays_d = np.stack(rays_d)
    return rays_o, rays_d, imgs, poses, H, W, focal

"""Generate a tiny demo scene (a colored sphere) in nerf_synthetic format.

Lets you smoke-test training without downloading the (large) official Blender
dataset. Rays are traced against an analytic sphere to produce consistent
multi-view images.

Usage:
    python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32
"""
import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np

# make the `nerf` package importable regardless of the working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nerf.rays import get_rays_np  # noqa: E402


def trace_sphere(rays_o, rays_d, center, radius):
    """Ray-sphere intersection -> per-pixel RGB (Lambertian shading, white bg)."""
    oc = rays_o - center
    b = np.sum(oc * rays_d, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = b * b - c
    t = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0.0)), -1.0)  # near hit
    hit = t > 0
    normal = (rays_o + t[..., None] * rays_d - center)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([0.4, 0.7, -0.6])
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)[..., None]
    base = np.array([0.9, 0.15, 0.15]) * (0.3 + 0.7 * shade)
    rgb = np.where(hit[..., None], base, np.array([1.0, 1.0, 1.0]))
    return rgb


def look_at(pos, center=(0, 0, 0), up=(0, 1, 0)):
    """c2w matrix (Blender convention: camera looks along -z)."""
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
    angle_x = 0.691111  # nerf_synthetic-style field of view
    focal = 0.5 * W / np.tan(0.5 * angle_x)
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.8

    os.makedirs(os.path.join(args.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "test"), exist_ok=True)

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
            frames.append(
                {"file_path": f"./{split}/r_{i:03d}", "transform_matrix": c2w.tolist()}
            )
        with open(os.path.join(args.out, f"transforms_{split}.json"), "w") as f:
            json.dump({"camera_angle_x": angle_x, "frames": frames}, f, indent=2)

    print(f"[demo] wrote {args.out}  ({H}x{W}, {args.n_train} train / "
          f"{args.n_test} test images, focal={focal:.2f})")


if __name__ == "__main__":
    main()

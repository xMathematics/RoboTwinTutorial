"""Command-line entry point: train / test / render.

Usage:
    python run_nerf.py --config data/lego --mode train --exp lego_full
    python run_nerf.py --config data/lego --mode test  --exp lego_full --ckpt logs/lego_full/latest.pt
    python run_nerf.py --config data/lego --mode render --exp lego_full --ckpt logs/lego_full/latest.pt --frames 120
"""
import argparse
import json
import os

import imageio
import numpy as np
import torch

from nerf.config import Config
from nerf.data_utils import load_blender_data
from nerf.rays import get_rays
from nerf.trainer import build_models, train_nerf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal PyTorch NeRF (teaching version)")
    p.add_argument("--config", required=True, help="path to a nerf_synthetic scene dir")
    p.add_argument("--mode", choices=["train", "test", "render"], default="train")
    p.add_argument("--exp", default="exp", help="experiment name")
    p.add_argument("--steps", type=int, default=None, help="override training steps")
    p.add_argument("--batch_size", type=int, default=None, help="override batch size")
    p.add_argument("--tiny", action="store_true", help="fast smoke-run settings")
    p.add_argument("--ckpt", default=None, help="checkpoint path (test/render)")
    p.add_argument("--frames", type=int, default=120, help="render-path frames")
    p.add_argument("--testskip", type=int, default=8, help="test frame stride")
    p.add_argument("--no_half_res", action="store_true", help="keep full resolution")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    cfg = Config(datadir=args.config, exp_name=args.exp)
    if args.tiny:
        cfg.half_res = True
        cfg.batch_size = 256
        cfg.steps = 1000
        cfg.n_coarse = 32
        cfg.n_fine = 64
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_half_res:
        cfg.half_res = False
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    return cfg


@torch.no_grad()
def render_image(model_coarse, model_fine, rays_o, rays_d, cfg, chunk=8192, perturb=False):
    """Render a full [H, W, 3] image from a [H, W, 3] ray grid, chunked over rays."""
    from nerf.render import render_rays

    H, W = rays_o.shape[:2]
    ro = rays_o.reshape(-1, 3)
    rd = rays_d.reshape(-1, 3)
    rgb = torch.zeros_like(ro)
    for i in range(0, ro.shape[0], chunk):
        out = render_rays(
            model_coarse, model_fine, ro[i : i + chunk], rd[i : i + chunk],
            cfg.near, cfg.far, cfg.n_coarse, cfg.n_fine,
            cfg.l_xyz, cfg.l_dir, cfg.use_viewdirs, perturb=perturb,
        )
        rgb[i : i + chunk] = out["rgb_fine"]
    return rgb.reshape(H, W, 3)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    return float(10.0 * np.log10(1.0 / (mse + 1e-10)))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Windowed SSIM (simple Gaussian-window implementation)."""
    from scipy.ndimage import gaussian_filter

    k1, k2, L = 0.01, 0.03, 1.0
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    mu1 = gaussian_filter(img1, 1.5)
    mu2 = gaussian_filter(img2, 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1_sq = gaussian_filter(img1 * img1, 1.5) - mu1_sq
    s2_sq = gaussian_filter(img2 * img2, 1.5) - mu2_sq
    s12 = gaussian_filter(img1 * img2, 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    )
    return float(np.mean(ssim_map))


def load_ckpt(cfg: Config, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model_coarse, model_fine = build_models(cfg, cfg.device)
    model_coarse.load_state_dict(ckpt["coarse"])
    model_fine.load_state_dict(ckpt["fine"])
    model_coarse.eval()
    model_fine.eval()
    return model_coarse, model_fine


def look_at(pos: np.ndarray, center=(0, 0, 0), up=(0, 1, 0)) -> np.ndarray:
    """Build a c2w matrix (Blender convention: camera looks along -z)."""
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


def render_path_poses(n_frames: int, radius: float = 4.0) -> list[np.ndarray]:
    """Circular camera path in the x-z plane looking at the origin."""
    poses = []
    for i in range(n_frames):
        theta = 2.0 * np.pi * i / n_frames
        pos = np.array([radius * np.cos(theta), 0.0, radius * np.sin(theta)])
        poses.append(look_at(pos, center=(0, 0, 0)))
    return poses


def save_image(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.v2.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    print(f"[config] {cfg}")

    log_dir = os.path.join(cfg.log_dir, cfg.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- train ----------
    if args.mode == "train":
        rays_o, rays_d, imgs, _, H, W, focal = load_blender_data(
            cfg.datadir, "train", cfg.half_res, testskip=1
        )
        print(f"[data] {imgs.shape[0]} train images @ {H}x{W}, focal={focal:.1f}")
        model_coarse, model_fine = train_nerf(rays_o, rays_d, imgs, cfg, log_dir)
        ckpt = os.path.join(log_dir, "latest.pt")
        torch.save(
            {"coarse": model_coarse.state_dict(), "fine": model_fine.state_dict()}, ckpt
        )
        print(f"[train] saved -> {ckpt}")

    # ---------- test / render ----------
    else:
        if args.ckpt is None:
            args.ckpt = os.path.join(log_dir, "latest.pt")
        model_coarse, model_fine = load_ckpt(cfg, args.ckpt)
        print(f"[ckpt] loaded {args.ckpt}")

        if args.mode == "test":
            rays_o, rays_d, imgs, _, H, W, focal = load_blender_data(
                cfg.datadir, "test", cfg.half_res, testskip=args.testskip
            )
            print(f"[test] {imgs.shape[0]} test images @ {H}x{W}")
            out_dir = os.path.join(log_dir, "test")
            os.makedirs(out_dir, exist_ok=True)
            psnrs, ssims = [], []
            for i in range(imgs.shape[0]):
                ro = torch.from_numpy(rays_o[i]).to(cfg.device)
                rd = torch.from_numpy(rays_d[i]).to(cfg.device)
                pred = render_image(model_coarse, model_fine, ro, rd, cfg)
                pred_np = pred.cpu().numpy()
                gt_np = imgs[i]
                save_image(pred_np, os.path.join(out_dir, f"pred_{i:03d}.png"))
                psnrs.append(psnr(pred_np, gt_np))
                ssims.append(ssim(pred_np, gt_np))
            metrics = {
                "psnr_mean": float(np.mean(psnrs)),
                "ssim_mean": float(np.mean(ssims)),
                "psnr_list": psnrs,
                "ssim_list": ssims,
            }
            json.dump(metrics, open(os.path.join(log_dir, "metrics.json"), "w"), indent=2)
            print(f"[test] PSNR {np.mean(psnrs):.2f} | SSIM {np.mean(ssims):.4f}")

        elif args.mode == "render":
            # render a camera trajectory into video frames
            _, _, _, _, H, W, focal = load_blender_data(
                cfg.datadir, "train", cfg.half_res, testskip=1
            )
            poses = render_path_poses(args.frames)
            out_dir = os.path.join(log_dir, "video")
            os.makedirs(out_dir, exist_ok=True)
            print(f"[render] {args.frames} frames @ {H}x{W}")
            for i, c2w in enumerate(poses):
                c2w_t = torch.from_numpy(c2w).to(cfg.device)
                ro, rd = get_rays(H, W, focal, c2w_t)
                pred = render_image(model_coarse, model_fine, ro, rd, cfg)
                save_image(pred.cpu().numpy(), os.path.join(out_dir, f"frame_{i:04d}.png"))
            print(f"[render] done -> {out_dir}")


if __name__ == "__main__":
    main()

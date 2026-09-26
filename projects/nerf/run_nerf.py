"""命令行入口：训练 / 测试 / 渲染三合一。

函数流水线：本模块是整个项目的入口，串起所有模块——
    parse_args → make_config（构造 Config，无 cuda 时回退 cpu）
    train 模式：  load_blender_data（data_utils）→ train_nerf（trainer）→ 保存 checkpoint
    test 模式：   load_ckpt → 逐张 render_image（内部走 render.render_rays）
                  → metrics.psnr / metrics.ssim 逐图评分（统一测评层，见 nerf/metrics.py）
    render 模式： load_blender_data（只为取 H/W/focal）→ render_path_poses 生成圆轨迹位姿
                  → rays.get_rays → render_image → 逐帧存 PNG

用法（Usage）：
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
from nerf.metrics import psnr, ssim
from nerf.rays import get_rays
from nerf.trainer import build_models, train_nerf


def parse_args() -> argparse.Namespace:
    """解析命令行参数（help 文本为运行时字符串，保持英文以便 --help 输出稳定）。"""
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
    """由命令行参数构造 Config：默认值 + tiny 快速冒烟档 + 显式覆写项。"""
    cfg = Config(datadir=args.config, exp_name=args.exp)
    if args.tiny:
        # tiny 档：小 batch、少步数、少采样点，用于分钟级冒烟验证
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
    """把 [H, W, 3] 的光线网格渲染成一整张 [H, W, 3] 图像，按 chunk 分块防止显存爆掉。"""
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
        rgb[i : i + chunk] = out["rgb_fine"]  # 取 fine 网络的渲染结果作为最终颜色
    return rgb.reshape(H, W, 3)


def load_ckpt(cfg: Config, ckpt_path: str):
    """加载 checkpoint 并按 cfg 重建同构网络，切到 eval 模式。

    注：本实现的 MLP 无 dropout/BN，eval() 行为上无实际作用；评测时的确定性
    由 render_rays(perturb=False) 保证。
    """
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model_coarse, model_fine = build_models(cfg, cfg.device)
    model_coarse.load_state_dict(ckpt["coarse"])
    model_fine.load_state_dict(ckpt["fine"])
    model_coarse.eval()
    model_fine.eval()
    return model_coarse, model_fine


def look_at(pos: np.ndarray, center=(0, 0, 0), up=(0, 1, 0)) -> np.ndarray:
    """构造 c2w 矩阵（Blender 约定：相机朝 -z 看，y 轴尽量对齐 up）。

    几何含义：z 轴 = 相机指向"看点"的反方向，x = up × z，y = z × x，
    平移列放相机位置 pos（世界坐标）。
    """
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
    """生成 x-z 平面内半径 radius 的圆形相机轨迹，全部朝向原点（共 n_frames 个 c2w）。"""
    poses = []
    for i in range(n_frames):
        theta = 2.0 * np.pi * i / n_frames
        pos = np.array([radius * np.cos(theta), 0.0, radius * np.sin(theta)])
        poses.append(look_at(pos, center=(0, 0, 0)))
    return poses


def save_image(img: np.ndarray, path: str) -> None:
    """把 [H, W, 3]（∈ [0,1]）裁剪到合法区间后量化为 uint8 存成 PNG。"""
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

    # ---------- 训练 ----------
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

    # ---------- 测试 / 渲染 ----------
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
            # 沿一条相机轨迹逐帧渲染成视频帧
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

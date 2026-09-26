"""训练循环（论文 Sec. 5.3, Eq. 7）。

函数流水线：本模块是训练侧的调度中心，被 run_nerf.py（train 模式）调用；
依赖 config.Config（超参数）、model.NeRF（经 build_models 构建 coarse/fine 两个
网络）、render.render_rays（每步前向）。输入：load_blender_data 预计算好的光线
与图像（NumPy 数组）；输出：训练好的 (model_coarse, model_fine) 与 checkpoint
文件 log_dir/latest.pt。
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import Config
from .model import NeRF
from .render import render_rays


def build_models(cfg: Config, device: str) -> tuple[NeRF, NeRF]:
    """构建 coarse 与 fine 两个 MLP（结构相同、参数相互独立，论文 Sec. 5.2）。

    Args:
        cfg: 超参数（决定编码维度 l_xyz/l_dir 与网络宽度/深度）。
        device: 目标设备（"cpu" 或 "cuda"）。

    Returns:
        (model_coarse, model_fine)：已移到 device 上的两个 NeRF 实例。
    """
    in_dim = 3 * 2 * cfg.l_xyz
    view_dim = 3 * 2 * cfg.l_dir
    kwargs = dict(
        in_dim=in_dim,
        view_dim=view_dim,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        skip_at=cfg.skip_at,
        use_viewdirs=cfg.use_viewdirs,
    )
    model_coarse = NeRF(**kwargs).to(device)
    model_fine = NeRF(**kwargs).to(device)
    return model_coarse, model_fine


def train_nerf(
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    imgs: np.ndarray,
    cfg: Config,
    log_dir: str,
) -> tuple[NeRF, NeRF]:
    """对一个场景优化 NeRF（coarse + fine 联合训练，论文 Eq. 7）。

    Args:
        rays_o, rays_d: [N, H, W, 3] 预计算的光线原点/方向网格。
        imgs: [N, H, W, 3] 真值图像，像素 ∈ [0, 1]。
        cfg: 超参数（步数、batch、学习率等）。
        log_dir: checkpoint 保存目录（每 5000 步与结束时各存一次 latest.pt）。

    Returns:
        (model_coarse, model_fine)：训练完成后的两个网络。
    """
    device = cfg.device
    rays_o = torch.from_numpy(rays_o).to(device)
    rays_d = torch.from_numpy(rays_d).to(device)
    imgs = torch.from_numpy(imgs).to(device)

    # 把光线/图像摊平成 [R_total, 3] 的池子，训练时随机抽 batch
    ro = rays_o.reshape(-1, 3)
    rd = rays_d.reshape(-1, 3)
    target = imgs.reshape(-1, 3)
    num_rays_total = ro.shape[0]

    model_coarse, model_fine = build_models(cfg, device)
    params = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(
        params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-7
    )  # Adam 超参取论文默认值

    os.makedirs(log_dir, exist_ok=True)
    step = 0
    pbar = tqdm(range(cfg.steps), desc="train")
    t0 = time.time()
    for step in pbar:
        # 指数学习率衰减：5e-4 → 5e-5（论文 Sec. 5.3）
        lr = cfg.lr * cfg.lr_decay ** (step / cfg.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 随机抽取一个 batch 的光线（论文：batch = 4096 条）
        idx = torch.randint(0, num_rays_total, (cfg.batch_size,), device=device)
        batch_ro = ro[idx]
        batch_rd = rd[idx]
        batch_target = target[idx]

        out = render_rays(
            model_coarse,
            model_fine,
            batch_ro,
            batch_rd,
            cfg.near,
            cfg.far,
            cfg.n_coarse,
            cfg.n_fine,
            cfg.l_xyz,
            cfg.l_dir,
            cfg.use_viewdirs,
            perturb=True,
        )

        # loss = ||C_c - C||^2 + ||C_f - C||^2（论文 Eq. 7：coarse 与 fine 渲染同时回归真值）
        loss = F.mse_loss(out["rgb_coarse"], batch_target) + F.mse_loss(
            out["rgb_fine"], batch_target
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        if step % 5000 == 0:
            torch.save(
                {"coarse": model_coarse.state_dict(), "fine": model_fine.state_dict()},
                os.path.join(log_dir, "latest.pt"),
            )

    torch.save(
        {"coarse": model_coarse.state_dict(), "fine": model_fine.state_dict()},
        os.path.join(log_dir, "latest.pt"),
    )
    print(f"[train] done in {time.time() - t0:.1f}s, final loss {loss.item():.4f}")
    return model_coarse, model_fine

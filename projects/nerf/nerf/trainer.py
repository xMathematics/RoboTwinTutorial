"""Training loop (paper Sec. 5.3, Eq. 7)."""
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
    """Coarse and fine MLPs (independent parameters, same architecture)."""
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
    """Optimize a NeRF for one scene.

    Args:
        rays_o, rays_d: [N, H, W, 3] precomputed rays.
        imgs: [N, H, W, 3] ground-truth images.
        cfg: hyper-parameters.
        log_dir: where to write checkpoints.

    Returns:
        (model_coarse, model_fine).
    """
    device = cfg.device
    rays_o = torch.from_numpy(rays_o).to(device)
    rays_d = torch.from_numpy(rays_d).to(device)
    imgs = torch.from_numpy(imgs).to(device)

    # Flatten rays/images into [R_total, 3] pools.
    ro = rays_o.reshape(-1, 3)
    rd = rays_d.reshape(-1, 3)
    target = imgs.reshape(-1, 3)
    num_rays_total = ro.shape[0]

    model_coarse, model_fine = build_models(cfg, device)
    params = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(
        params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-7
    )  # paper defaults

    os.makedirs(log_dir, exist_ok=True)
    step = 0
    pbar = tqdm(range(cfg.steps), desc="train")
    t0 = time.time()
    for step in pbar:
        # exponential lr decay: 5e-4 -> 5e-5 (paper)
        lr = cfg.lr * cfg.lr_decay ** (step / cfg.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # sample a batch of rays (paper: batch of 4096)
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

        # loss = ||C_c - C||^2 + ||C_f - C||^2   (paper Eq. 7)
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

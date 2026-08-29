"""Camera -> ray generation (Blender synthetic-dataset convention).

The Blender (nerf_synthetic) convention places the camera looking down -z at
focal length f, then rotates it with the camera-to-world matrix c2w:

    dirs = [(i - W/2)/f, -(j - H/2)/f, -1]
    rays_d = R @ dirs
    rays_o = t  (camera center in world coordinates)
"""
import numpy as np
import torch


def get_rays_np(H: int, W: int, focal: float, c2w: np.ndarray):
    """NumPy version. c2w: [4, 4] camera-to-world matrix.

    Returns rays_o [H, W, 3], rays_d [H, W, 3].
    """
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1
    ).astype(np.float32)  # keep float32 (W * 0.5 is a python float -> would upcast)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)   # [H, W, 3]
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """PyTorch version. c2w: [4, 4] tensor on device."""
    i, j = torch.meshgrid(
        torch.arange(W, device=c2w.device, dtype=torch.float32),
        torch.arange(H, device=c2w.device, dtype=torch.float32),
        indexing="xy",
    )
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1
    )
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

"""相机 → 光线生成（Blender 合成数据集约定）。

函数流水线：本模块是"相机位姿 → 光线"的桥梁，被 data_utils.load_blender_data
（NumPy 版，把每个训练位姿展开成逐像素光线）与 run_nerf.py（PyTorch 版，渲染
轨迹时逐帧生成光线）调用；无包内依赖。
输入：图像高宽、焦距（像素单位）、c2w 矩阵；输出：光线原点/方向的两个 [H, W, 3] 网格。

Blender（nerf_synthetic）约定：相机默认朝 -z 看、焦距为 f，再用相机到世界矩阵
c2w 旋转到当前位姿。针孔模型下像素 (i, j) 的方向为

    dirs = [(i - W/2)/f, -(j - H/2)/f, -1]
    rays_d = R @ dirs          （R 为 c2w 左上 3x3 旋转）
    rays_o = t                 （相机中心，即 c2w 平移列）
"""
import numpy as np
import torch


def get_rays_np(H: int, W: int, focal: float, c2w: np.ndarray):
    """NumPy 版光线生成。c2w: [4, 4] 相机到世界矩阵（float32）。

    Args:
        H, W: 图像高/宽（像素）。focal: 焦距（像素单位）。
        c2w: [4, 4] 相机到世界变换（旋转 R = c2w[:3,:3]，平移 t = c2w[:3,-1]）。

    Returns:
        rays_o [H, W, 3]：光线原点网格（全部等于相机中心，世界坐标）。
        rays_d [H, W, 3]：光线方向网格（世界系，未归一化；长度由针孔模型决定）。
    """
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1
    ).astype(np.float32)  # 保持 float32（W * 0.5 是 python float，否则会向上转型成 float64）
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)   # [H, W, 3]：相机系方向旋转到世界系
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)     # 原点 = 相机中心，广播到 [H, W, 3]
    return rays_o, rays_d


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """PyTorch 版光线生成，逻辑与 get_rays_np 完全一致。c2w: [4, 4] 设备上的张量。

    Args:
        H, W: 图像高/宽（像素）。focal: 焦距（像素单位）。
        c2w: [4, 4] 相机到世界张量（放在目标设备上）。

    Returns:
        rays_o [H, W, 3]，rays_d [H, W, 3]（含义同 get_rays_np）。
    """
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

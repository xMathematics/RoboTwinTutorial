"""Differentiable volume rendering (paper Sec. 4, Eq. 3; Max 1995).

    C(r) = sum_i T_i (1 - exp(-sigma_i * delta_i)) c_i,
    T_i  = exp(- sum_{j<i} sigma_j delta_j)
"""
import torch
import torch.nn as nn

from .encoding import positional_encoding
from .sampling import hierarchical_sample, ray_deltas, stratified_sample


def volume_render(
    rgb: torch.Tensor, sigma: torch.Tensor, deltas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-composite sample colors/densities into a per-ray color.

    Args:
        rgb:    [..., N, 3] predicted colors.
        sigma:  [..., N, 1] predicted densities (already non-negative).
        deltas: [..., N]    distance between adjacent samples.

    Returns:
        color:   [..., 3] rendered ray color.
        weights: [..., N]  compositing weights w_i = T_i * alpha_i (reused for
                           hierarchical sampling).
    """
    alpha = 1.0 - torch.exp(-sigma[..., 0] * deltas)             # [..., N]
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)   # T_i = prod_{j<=i}
    # Shift so that T_1 = 1 (transmittance *before* sample i).
    transmittance = torch.cat(
        [torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1
    )
    weights = transmittance * alpha                              # w_i = T_i alpha_i
    color = (weights[..., None] * rgb).sum(dim=-2)               # sum_i w_i c_i
    return color, weights


def render_rays(
    model_coarse: nn.Module,
    model_fine: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    l_xyz: int,
    l_dir: int,
    use_viewdirs: bool = True,
    perturb: bool = True,
) -> dict[str, torch.Tensor]:
    """Render a batch of rays through the coarse and fine networks (paper Sec. 5.2).

    Coarse and fine are two *independent* MLPs with the same architecture. The
    coarse pass produces the importance distribution for hierarchical sampling;
    the fine pass renders all (N_c + N_f) points.

    Args:
        model_coarse, model_fine: NeRF instances.
        rays_o: [R, 3] ray origins.
        rays_d: [R, 3] ray directions (unit).
        near, far: scene bounds.
        n_coarse, n_fine: N_c and N_f.
        l_xyz, l_dir: positional encoding frequencies.
        use_viewdirs: pass direction into the color branch.
        perturb: randomize sampling (True for training, False for eval).

    Returns:
        dict with 'rgb_coarse' [R, 3], 'rgb_fine' [R, 3], 'weights' [R, N_c].
    """
    device = rays_o.device
    num_rays = rays_o.shape[0]

    # ---- coarse pass: stratified sampling + coarse network ----
    t_coarse = stratified_sample(near, far, n_coarse, num_rays, device)  # [R, Nc]
    pts_coarse = rays_o[:, None, :] + t_coarse[..., None] * rays_d[:, None, :]
    dirs = rays_d[:, None, :].expand(-1, n_coarse, -1)

    rgb_c, sigma_c = model_coarse(
        positional_encoding(pts_coarse, l_xyz),
        positional_encoding(dirs, l_dir) if use_viewdirs else None,
    )
    rgb_coarse, weights = volume_render(rgb_c, sigma_c, ray_deltas(t_coarse))

    # ---- fine pass: hierarchical sampling + fine network on all points ----
    t_fine = hierarchical_sample(t_coarse, weights, n_fine, device, perturb)
    t_all, _ = torch.cat([t_coarse, t_fine], dim=-1).sort(dim=-1)   # [R, Nc+Nf]
    pts_all = rays_o[:, None, :] + t_all[..., None] * rays_d[:, None, :]
    dirs_all = rays_d[:, None, :].expand(-1, n_coarse + n_fine, -1)

    rgb_f, sigma_f = model_fine(
        positional_encoding(pts_all, l_xyz),
        positional_encoding(dirs_all, l_dir) if use_viewdirs else None,
    )
    rgb_fine, _ = volume_render(rgb_f, sigma_f, ray_deltas(t_all))

    return {"rgb_coarse": rgb_coarse, "rgb_fine": rgb_fine, "weights": weights}

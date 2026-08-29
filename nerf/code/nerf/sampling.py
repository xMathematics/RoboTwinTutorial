"""Ray sampling: stratified (paper Eq. 2) and hierarchical (paper Sec. 5.2)."""
import torch


def stratified_sample(
    near: float, far: float, n_samples: int, num_rays: int, device: str = "cpu"
) -> torch.Tensor:
    """Partition [near, far] into n_samples bins and draw one random point per bin.

    Args:
        near, far: ray near/far bounds (scalars, shared by all rays).
        n_samples: number of bins / samples per ray.
        num_rays: batch size.

    Returns:
        t of shape [num_rays, n_samples], each t_i in bin i.
    """
    bins = torch.linspace(near, far, n_samples + 1, device=device)
    bin_size = (far - near) / n_samples
    u = torch.rand(num_rays, n_samples, device=device)
    t = bins[:-1][None, :] + u * bin_size       # [R, N]
    return t


def hierarchical_sample(
    t_coarse: torch.Tensor,
    weights: torch.Tensor,
    n_fine: int,
    device: str = "cpu",
    perturb: bool = True,
) -> torch.Tensor:
    """Sample n_fine points per ray from the coarse weight distribution.

    Uses inverse transform sampling on the piecewise-constant PDF induced by the
    coarse network's alpha-compositing weights (paper Eq. 5 -> Sec. 5.2).

    Args:
        t_coarse: [R, N_c] coarse sample locations (sorted ascending).
        weights:  [R, N_c] coarse compositing weights w_i = T_i * alpha_i.
        n_fine:   number of additional fine samples.

    Returns:
        t_fine of shape [R, n_fine], biased toward high-weight (visible) regions.
    """
    w = weights + 1e-5                                  # numerical stability
    pdf = w / w.sum(dim=-1, keepdim=True)               # normalized PDF (hat w)
    cdf = torch.cumsum(pdf, dim=-1)                     # CDF
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    cdf[..., -1] = 1.0

    u = torch.rand(t_coarse.shape[0], n_fine, device=device)
    idx = torch.searchsorted(cdf, u)                    # [R, N_f]
    idx = idx.clamp(1, t_coarse.shape[-1] - 1)          # keep indices valid

    lo = torch.gather(t_coarse, -1, idx - 1)
    hi = torch.gather(t_coarse, -1, idx)
    if perturb:
        t_fine = lo + torch.rand_like(lo) * (hi - lo)   # random point in the bin
    else:
        t_fine = 0.5 * (lo + hi)
    return t_fine


def ray_deltas(t: torch.Tensor) -> torch.Tensor:
    """Distance between adjacent samples; last interval extended to 'infinity'."""
    deltas = t[..., 1:] - t[..., :-1]
    last = torch.full_like(t[..., :1], 1e10)
    return torch.cat([deltas, last], dim=-1)

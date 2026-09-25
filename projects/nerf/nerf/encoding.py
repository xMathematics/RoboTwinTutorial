"""Positional encoding (paper Eq. 4).

    gamma(p) = [sin(2^0 pi p), cos(2^0 pi p), ...,
                sin(2^{L-1} pi p), cos(2^{L-1} pi p)]

Applied independently to each component of a coordinate. This is a *fixed*,
non-learnable map; it lifts continuous coordinates into a higher-dimensional
space so the MLP can fit high-frequency content.
"""
import math

import torch


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Encode a coordinate tensor with sinusoidal frequencies.

    Args:
        x: tensor of shape [..., D] (e.g. D=3 for (x, y, z)).
        num_freqs: the maximum frequency index L.

    Returns:
        tensor of shape [..., D * 2 * num_freqs].
    """
    # freqs = [2^0 * pi, 2^1 * pi, ..., 2^{L-1} * pi]
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    # args[..., d, k] = p_d * 2^k * pi
    args = x[..., None] * freqs                      # [..., D, L]
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [..., D, 2L]
    return enc.flatten(-2)                            # [..., 2 D L]

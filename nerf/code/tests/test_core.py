"""Smoke tests for the core NeRF components.

Run with pytest, or directly:
    python tests/test_core.py

These verify the math (positional encoding, volume rendering, sampling) and that
a forward+backward pass works end to end on the CPU.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nerf.encoding import positional_encoding  # noqa: E402
from nerf.model import NeRF  # noqa: E402
from nerf.render import render_rays, volume_render  # noqa: E402
from nerf.sampling import hierarchical_sample, ray_deltas, stratified_sample  # noqa: E402


def test_positional_encoding_shape():
    x = torch.rand(5, 3)
    enc = positional_encoding(x, num_freqs=10)
    assert enc.shape == (5, 60)          # 3 * 2 * 10
    enc_d = positional_encoding(x, num_freqs=4)
    assert enc_d.shape == (5, 24)        # 3 * 2 * 4


def test_positional_encoding_zero():
    # sin(0) = 0, cos(0) = 1 for every frequency.
    # Layout per dimension d: [sin_0..sin_{L-1}, cos_0..cos_{L-1}].
    x = torch.zeros(2, 3)
    enc = positional_encoding(x, num_freqs=2)      # [2, 12]
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0]).repeat(2, 3)
    assert torch.allclose(enc, expected, atol=1e-6)


def test_stratified_sample_in_bins():
    t = stratified_sample(2.0, 6.0, 64, 4)
    assert t.shape == (4, 64)
    assert (t >= 2.0).all() and (t <= 6.0).all()
    # sample i should lie in bin i
    bin_size = (6.0 - 2.0) / 64
    lo = 2.0 + torch.arange(64) * bin_size
    assert ((t - lo[None, :]) >= 0).all() and ((t - lo[None, :]) < bin_size).all()


def test_volume_render_empty_space_is_black():
    rgb = torch.ones(4, 64, 3)
    sigma = torch.zeros(4, 64, 1)                 # nothing absorbs light
    deltas = torch.ones(4, 64)
    color, weights = volume_render(rgb, sigma, deltas)
    assert torch.allclose(color, torch.zeros(4, 3), atol=1e-6)
    assert torch.allclose(weights, torch.zeros(4, 64), atol=1e-6)


def test_volume_render_single_opaque_point():
    # one sample with huge density -> the ray color equals that sample's color
    rgb = torch.zeros(1, 3, 3)
    rgb[0, 1] = torch.tensor([0.9, 0.1, 0.2])
    sigma = torch.zeros(1, 3, 1)
    sigma[0, 1] = 1e4
    deltas = torch.full((1, 3), 1.0)
    color, _ = volume_render(rgb, sigma, deltas)
    assert torch.allclose(color[0], rgb[0, 1], atol=1e-3)


def test_hierarchical_sample_count_and_range():
    t = torch.linspace(2.0, 6.0, 64).unsqueeze(0).expand(3, -1)
    w = torch.rand(3, 64)
    t_fine = hierarchical_sample(t, w, n_fine=128, device="cpu")
    assert t_fine.shape == (3, 128)
    assert (t_fine >= 2.0 - 1e-4).all() and (t_fine <= 6.0 + 1e-4).all()


def test_model_forward_shapes():
    model = NeRF(in_dim=60, view_dim=24)
    x = torch.rand(8, 60)
    d = torch.rand(8, 24)
    rgb, sigma = model(x, d)
    assert rgb.shape == (8, 3)
    assert sigma.shape == (8, 1)
    assert (sigma >= 0).all() and (rgb >= 0).all() and (rgb <= 1).all()


def test_end_to_end_forward_backward():
    torch.manual_seed(0)
    model_coarse = NeRF(in_dim=60, view_dim=24)
    model_fine = NeRF(in_dim=60, view_dim=24)
    ro = torch.rand(16, 3) * 2 - 1
    rd = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    target = torch.rand(16, 3)
    out = render_rays(
        model_coarse, model_fine, ro, rd, near=2.0, far=6.0,
        n_coarse=8, n_fine=16, l_xyz=10, l_dir=4, use_viewdirs=True, perturb=False,
    )
    assert out["rgb_coarse"].shape == (16, 3)
    assert out["rgb_fine"].shape == (16, 3)
    loss = torch.nn.functional.mse_loss(out["rgb_coarse"], target) + torch.nn.functional.mse_loss(
        out["rgb_fine"], target
    )
    loss.backward()
    grads = [p.grad for p in model_coarse.parameters() if p.grad is not None]
    assert len(grads) > 0 and all((g.abs() > 0).any() for g in grads)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)

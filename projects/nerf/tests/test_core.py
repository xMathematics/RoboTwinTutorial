"""NeRF 核心组件的冒烟测试（smoke tests）。

可用 pytest 运行，也可直接执行：
    python tests/test_core.py

覆盖范围：验证数学正确性（位置编码、体积渲染、采样），以及一次
前向 + 反向传播能在 CPU 上端到端跑通。
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
    """位置编码输出维度 = D * 2 * L（每个分量 L 对 sin/cos）。"""
    x = torch.rand(5, 3)
    enc = positional_encoding(x, num_freqs=10)
    assert enc.shape == (5, 60)          # 期望 3*2*10=60：每分量 10 对 sin/cos
    enc_d = positional_encoding(x, num_freqs=4)
    assert enc_d.shape == (5, 24)        # 期望 3*2*4=24（对应视角方向分支的 24 维）


def test_positional_encoding_zero():
    # 零点处对所有频率都有 sin(0)=0、cos(0)=1，可直接手算验证。
    # 每个维度 d 的排布: [sin_0..sin_{L-1}, cos_0..cos_{L-1}]。
    x = torch.zeros(2, 3)
    enc = positional_encoding(x, num_freqs=2)      # [2, 12]：3 维 × 2 对 × 2L
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0]).repeat(2, 3)
    assert torch.allclose(enc, expected, atol=1e-6)


def test_stratified_sample_in_bins():
    """分层采样：第 i 个采样点必须落在第 i 个 bin 内（分层采样的定义，论文 Eq. 2）。"""
    t = stratified_sample(2.0, 6.0, 64, 4)
    assert t.shape == (4, 64)
    assert (t >= 2.0).all() and (t <= 6.0).all()   # 全部落在 [near, far] 内
    # 第 i 列应满足 lo_i <= t_i < lo_i + bin_size（bin 左闭右开）
    bin_size = (6.0 - 2.0) / 64
    lo = 2.0 + torch.arange(64) * bin_size
    assert ((t - lo[None, :]) >= 0).all() and ((t - lo[None, :]) < bin_size).all()


def test_volume_render_empty_space_is_black():
    """真空场景（σ 全 0）应渲染为纯黑：alpha = 1-exp(0) = 0 → 所有权重为 0。"""
    rgb = torch.ones(4, 64, 3)
    sigma = torch.zeros(4, 64, 1)                 # 无介质吸收/散射光
    deltas = torch.ones(4, 64)
    color, weights = volume_render(rgb, sigma, deltas)
    assert torch.allclose(color, torch.zeros(4, 3), atol=1e-6)
    assert torch.allclose(weights, torch.zeros(4, 64), atol=1e-6)


def test_volume_render_single_opaque_point():
    # 单个采样点密度极大 → 该点 alpha≈1、其余点 alpha=0，
    # 因此光线颜色应恰好等于该采样点的颜色（不透明度饱和的极端情形）。
    rgb = torch.zeros(1, 3, 3)
    rgb[0, 1] = torch.tensor([0.9, 0.1, 0.2])
    sigma = torch.zeros(1, 3, 1)
    sigma[0, 1] = 1e4                             # σ·δ = 1e4 → alpha = 1-exp(-1e4) ≈ 1
    deltas = torch.full((1, 3), 1.0)
    color, _ = volume_render(rgb, sigma, deltas)
    assert torch.allclose(color[0], rgb[0, 1], atol=1e-3)


def test_hierarchical_sample_count_and_range():
    """层次采样：输出点数正确，且落在 coarse 采样覆盖的深度范围内。"""
    t = torch.linspace(2.0, 6.0, 64).unsqueeze(0).expand(3, -1)
    w = torch.rand(3, 64)
    t_fine = hierarchical_sample(t, w, n_fine=128, device="cpu")
    assert t_fine.shape == (3, 128)
    # 采样点是相邻 coarse 深度构成 bin 的凸组合，故不会越出 [2, 6]（留 1e-4 数值容差）
    assert (t_fine >= 2.0 - 1e-4).all() and (t_fine <= 6.0 + 1e-4).all()


def test_model_forward_shapes():
    """网络输出形状与取值范围：rgb ∈ [0,1]（sigmoid），σ ≥ 0（ReLU）。"""
    model = NeRF(in_dim=60, view_dim=24)
    x = torch.rand(8, 60)
    d = torch.rand(8, 24)
    rgb, sigma = model(x, d)
    assert rgb.shape == (8, 3)
    assert sigma.shape == (8, 1)
    assert (sigma >= 0).all() and (rgb >= 0).all() and (rgb <= 1).all()


def test_end_to_end_forward_backward():
    """端到端：渲染 → Eq. 7 损失 → 反向传播，所有 coarse 参数都应拿到非零梯度。"""
    torch.manual_seed(0)
    model_coarse = NeRF(in_dim=60, view_dim=24)
    model_fine = NeRF(in_dim=60, view_dim=24)
    ro = torch.rand(16, 3) * 2 - 1
    rd = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)  # 方向归一化为单位向量
    target = torch.rand(16, 3)
    out = render_rays(
        model_coarse, model_fine, ro, rd, near=2.0, far=6.0,
        n_coarse=8, n_fine=16, l_xyz=10, l_dir=4, use_viewdirs=True, perturb=False,
    )
    assert out["rgb_coarse"].shape == (16, 3)
    assert out["rgb_fine"].shape == (16, 3)
    # 论文 Eq. 7：coarse 与 fine 渲染同时对真值回归
    loss = torch.nn.functional.mse_loss(out["rgb_coarse"], target) + torch.nn.functional.mse_loss(
        out["rgb_fine"], target
    )
    loss.backward()
    grads = [p.grad for p in model_coarse.parameters() if p.grad is not None]
    assert len(grads) > 0 and all((g.abs() > 0).any() for g in grads)


if __name__ == "__main__":
    import traceback

    # 直接执行时：收集所有 test_ 开头的函数逐个运行，汇总通过率（不依赖 pytest）
    # 单点测试：python tests/test_core.py <测试名子串>；不带参数 = 全部测试。
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and pattern in k]
    if not fns:
        print(f"没有匹配 '{pattern}' 的测试；可用测试：")
        for k in sorted(globals()):
            if k.startswith("test_"):
                print("  ", k)
        sys.exit(1)
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

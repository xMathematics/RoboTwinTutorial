"""位置编码（positional encoding，论文 Eq. 4）。

    gamma(p) = [sin(2^0 pi p), cos(2^0 pi p), ...,
                sin(2^{L-1} pi p), cos(2^{L-1} pi p)]

对坐标的每个分量独立施加该映射。这是一个*固定*、不可学习的映射：把连续坐标
升维到更高维空间，使 MLP 能拟合高频内容（低维输入直接进网络会"糊"掉高频细节，
依据：论文 Sec. 5.3 / 教程 04 章的频谱分析）。

函数流水线：被 render.render_rays 在每次前向中调用，分别对采样点坐标（l_xyz）
与视角方向（l_dir）编码后喂给 model.NeRF；输入任意形状 [..., D] 的连续坐标
张量，输出 [..., D*2*L] 的编码张量；无包内依赖。
"""
import math

import torch


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """用正弦频率族对坐标张量做编码。

    Args:
        x: 形状 [..., D] 的坐标张量（如 D=3 表示 (x, y, z)；取值通常在 [-1, 1]
            或 [-π, π] 量级，由各频率 sin/cos 归一）。
        num_freqs: 最大频率下标 L（每个分量产生 L 个 sin + L 个 cos）。

    Returns:
        形状 [..., D * 2 * num_freqs] 的编码张量；每个维度的排布为
        [sin_0..sin_{L-1}, cos_0..cos_{L-1}]（见 torch.cat 沿最后一维的拼接顺序）。
    """
    # 频率序列 freqs = [2^0 π, 2^1 π, ..., 2^{L-1} π]（几何级数，论文 Eq. 4）
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    # args[..., d, k] = p_d * 2^k * π，即第 d 个分量在第 k 个频率下的自变量
    args = x[..., None] * freqs                      # 形状 [..., D, L]
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # 形状 [..., D, 2L]
    return enc.flatten(-2)                            # 形状 [..., 2 D L]

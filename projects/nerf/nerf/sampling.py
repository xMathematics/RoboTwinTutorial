"""光线采样：分层采样（stratified，论文 Eq. 2）与层次采样（hierarchical，论文 Sec. 5.2）。

函数流水线：被 render.render_rays 调用——stratified_sample 生成 coarse 趟的采样
深度，hierarchical_sample 依据 coarse 权重生成 fine 趟的附加深度，ray_deltas 把
采样深度转成体积渲染需要的段长 δ_i；无包内依赖。
输入：近/远边界、coarse 深度与权重；输出：采样深度 t 与段长 deltas 张量。
"""
import torch


def stratified_sample(
    near: float, far: float, n_samples: int, num_rays: int, device: str = "cpu"
) -> torch.Tensor:
    """把 [near, far] 均分为 n_samples 个 bin，每个 bin 内均匀随机取一个点。

    这是论文 Eq. 2 的分层采样（stratified sampling）：相比确定性等距采样，随机化
    能让相邻光线的采样位置错开，网络输出因此可以被视为对连续积分的期望
    （依据：教程 03 章 §3.4"方式 B"）。

    Args:
        near, far: 光线近/远边界（标量，所有光线共用；与场景坐标同单位）。
        n_samples: bin 数 = 每条光线采样点数 N。
        num_rays: 光线条数 R（batch 大小）。

    Returns:
        t：形状 [num_rays, n_samples]，第 i 列的 t_i 落在第 i 个 bin 内，
        即 t_i ∈ [near + i·Δ, near + (i+1)·Δ)，Δ = (far - near)/N。
    """
    bins = torch.linspace(near, far, n_samples + 1, device=device)
    bin_size = (far - near) / n_samples
    u = torch.rand(num_rays, n_samples, device=device)
    t = bins[:-1][None, :] + u * bin_size       # [R, N]：bin 左端点 + bin 内随机偏移（Eq. 2）
    return t


def hierarchical_sample(
    t_coarse: torch.Tensor,
    weights: torch.Tensor,
    n_fine: int,
    device: str = "cpu",
    perturb: bool = True,
) -> torch.Tensor:
    """从 coarse 权重分布中为每条光线再采 n_fine 个点（层次采样，论文 Eq. 5 → Sec. 5.2）。

    做法：把 coarse 网络的合成权重 ŵ（经 alpha 合成得到，见 render.volume_render）
    视为沿光线的分段常数 PDF，构造 CDF 后做逆变换采样（inverse transform
    sampling）——u ~ U[0,1) 查表落回深度轴，样本自然偏向权重高（可见）的区域。

    Args:
        t_coarse: [R, N_c] coarse 采样深度（已按升序排列）。
        weights:  [R, N_c] coarse 合成权重 w_i = T_i·α_i（非负，沿 N_c 求和 ≤ 1）。
        n_fine:   每条光线附加的 fine 采样点数 N_f。

    Returns:
        t_fine：形状 [R, n_fine]，取值落在 [t_coarse 相邻样本] 构成的 bin 内
        （整体范围约等于 [t_coarse.min(), t_coarse.max()]），偏向高权重区域。
    """
    w = weights + 1e-5                                  # 数值稳定：防止全零权重导致除零
    pdf = w / w.sum(dim=-1, keepdim=True)               # 归一化 PDF（论文 Eq. 5 的 ŵ_i）
    cdf = torch.cumsum(pdf, dim=-1)                     # 累积分布 CDF
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    cdf[..., -1] = 1.0

    u = torch.rand(t_coarse.shape[0], n_fine, device=device)
    idx = torch.searchsorted(cdf, u)                    # [R, N_f]：逆变换采样，u 查 CDF 得 bin 索引
    idx = idx.clamp(1, t_coarse.shape[-1] - 1)          # 索引夹到 [1, N_c-1]，保证 lo/hi 有效

    lo = torch.gather(t_coarse, -1, idx - 1)
    hi = torch.gather(t_coarse, -1, idx)
    if perturb:
        t_fine = lo + torch.rand_like(lo) * (hi - lo)   # bin 内均匀随机取一点
    else:
        t_fine = 0.5 * (lo + hi)                        # 评测：取 bin 中点（确定性）
    return t_fine


def ray_deltas(t: torch.Tensor) -> torch.Tensor:
    """计算相邻采样点的间距 δ_i；末段补大数 1e10 近似"无穷远"。

    δ_i = t_{i+1} - t_i 是体积渲染 Eq. 3 中的段长（σ 乘以它才是不透明度；
    依据：教程 03 章 §3.5，段内密度取常数的指数衰减近似）。末段之后没有
    更多采样点，用大数代表剩余光线长度，使其 α≈1、可吸收残余权重。

    Args:
        t: [..., N] 升序采样深度。

    Returns:
        deltas: [..., N]，前 N-1 项为相邻差，最后一项为 1e10。
    """
    deltas = t[..., 1:] - t[..., :-1]
    last = torch.full_like(t[..., :1], 1e10)
    return torch.cat([deltas, last], dim=-1)

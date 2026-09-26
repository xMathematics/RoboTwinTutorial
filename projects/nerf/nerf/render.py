"""可微体积渲染（论文 Sec. 4, Eq. 3；Max 1995）。

    C(r) = sum_i T_i (1 - exp(-sigma_i * delta_i)) c_i,
    T_i  = exp(- sum_{j<i} sigma_j delta_j)

即连续体积渲染积分的离散化（alpha 合成）形式：C(r) = Σ_i T_i·α_i·c_i，
其中 α_i = 1 - exp(-σ_i·δ_i) 是第 i 个采样点的不透明度，
T_i = exp(-Σ_{j<i} σ_j·δ_j) 是到达第 i 点之前的透射率
（推导见教程 03 章 §3.5"离散化渲染公式"）。

函数流水线：本模块被 trainer.train_nerf（训练前向）与 run_nerf.render_image
（整图渲染）调用；依赖 sampling（三种采样）、encoding（位置编码）、model.NeRF
（coarse/fine 网络）。输入：光线批 + 场景边界；输出：每条光线的颜色与 coarse
合成权重（后者复用为层次采样的分布，论文 Eq. 5）。
"""
import torch
import torch.nn as nn

from .encoding import positional_encoding
from .sampling import hierarchical_sample, ray_deltas, stratified_sample


def volume_render(
    rgb: torch.Tensor, sigma: torch.Tensor, deltas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """把沿光线采样点的颜色/密度做 alpha 合成，得到每条光线的颜色。

    Args:
        rgb:    [..., N, 3] 网络预测的采样点颜色，∈ [0, 1]（sigmoid 输出）。
        sigma:  [..., N, 1] 网络预测的密度 σ，已保证非负（ReLU 输出）；
                量纲为 1/距离，与 deltas 相乘后无量纲。
        deltas: [..., N] 相邻采样点间距 δ_i（末段补大数近似无穷，见 ray_deltas）。

    Returns:
        color:   [..., 3] 该光线的渲染颜色 C(r) ∈ [0, 1]（权重的凸组合）。
        weights: [..., N] 合成权重 w_i = T_i·α_i（论文 Eq. 5；非负、沿 N 求和 ≤ 1，
                 后续被 hierarchical_sample 当作 PDF 使用）。
    """
    # α_i = 1 - exp(-σ_i·δ_i)：第 i 段的不透明度（论文 Eq. 3 中段内密度取常数的指数衰减）
    alpha = 1.0 - torch.exp(-sigma[..., 0] * deltas)             # [..., N]
    # 累乘（含自身）：位置 i 得 prod_{j<=i} (1-α_j) = T_{i+1}，右移一位后才是 T_i；
    # +1e-10 是数值保护（σ·δ 很大时 exp(-σδ) 下溢为 0，避免透射率精确为 0）
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)   # 含自身的累乘，见上
    # 右移一位，使 T_1 = 1：取的是采样点 i *之前* 的透射率（cumprod 给出含自身的累乘）
    transmittance = torch.cat(
        [torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1
    )
    weights = transmittance * alpha                              # w_i = T_i·α_i（论文 Eq. 5）
    color = (weights[..., None] * rgb).sum(dim=-2)               # C(r) = Σ_i w_i·c_i（Eq. 3 离散形式）
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
    """对一批光线先后经过 coarse、fine 两个网络渲染（论文 Sec. 5.2 层次采样）。

    coarse 与 fine 是两个*独立*参数、同构的 MLP：coarse 趟负责用分层采样粗略
    探路，其合成权重给出"哪里值得细看"的分布；fine 趟按该分布补采样，并在全部
    N_c + N_f 个点上重新渲染出最终颜色（依据：教程 05 章层次采样）。

    Args:
        model_coarse, model_fine: 两个 NeRF 实例。
        rays_o: [R, 3] 光线原点（世界坐标）。
        rays_d: [R, 3] 光线方向（单位向量）。
        near, far: 近/远裁剪面距离（标量，所有光线共用）。
        n_coarse, n_fine: N_c 与 N_f（每条光线的 coarse 点数与 fine 附加点数）。
        l_xyz, l_dir: 位置/方向的位置编码频率数 L。
        use_viewdirs: 是否把方向编码送入颜色分支。
        perturb: 是否随机化采样（训练 True；评测 False 保证确定性）。

    Returns:
        dict：'rgb_coarse' [R, 3]、'rgb_fine' [R, 3]（渲染颜色 ∈ [0,1]）、
        'weights' [R, N_c]（coarse 合成权重，供训练损失 Eq. 7 与层次采样复用）。
    """
    device = rays_o.device
    num_rays = rays_o.shape[0]

    # ---- coarse 趟：分层采样 + coarse 网络（论文 Eq. 2 → Eq. 3）----
    t_coarse = stratified_sample(near, far, n_coarse, num_rays, device)  # [R, Nc]
    pts_coarse = rays_o[:, None, :] + t_coarse[..., None] * rays_d[:, None, :]
    dirs = rays_d[:, None, :].expand(-1, n_coarse, -1)

    rgb_c, sigma_c = model_coarse(
        positional_encoding(pts_coarse, l_xyz),
        positional_encoding(dirs, l_dir) if use_viewdirs else None,
    )
    rgb_coarse, weights = volume_render(rgb_c, sigma_c, ray_deltas(t_coarse))

    # ---- fine 趟：按 coarse 权重层次采样，在全部 N_c+N_f 个点上过 fine 网络 ----
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

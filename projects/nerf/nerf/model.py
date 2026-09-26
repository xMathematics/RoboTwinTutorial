"""NeRF 的 MLP 网络（论文 Sec. 3 与 Fig. 7）。

结构（论文 Fig. 7 右图，启用视角分支时）:
    gamma(x) (60 维) → [FC-ReLU ×8, 256 通道，第 5 层处 skip]
        ├──> sigma        (ReLU 后的密度，只依赖位置)
        └──> feature(256) → 拼接 gamma(d) (24 维) → [FC-ReLU 128] → RGB (sigmoid)

函数流水线：本模块被 render.render_rays 调用（coarse、fine 两个同构独立实例），
也可被 trainer.build_models 批量构建；输入是位置编码后的采样点与（可选）方向，
输出每点的颜色 rgb ∈ [0,1]^3 与密度 σ ≥ 0。仅依赖 torch，无包内依赖。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    """单个辐射场 MLP；coarse 与 fine 网络共用本类（各自独立参数）。

    Attributes:
        use_viewdirs: 是否启用视角相关颜色分支（True 时颜色额外依赖方向编码）。
        skip_at: skip connection 位置——第 skip_at+1 层输入处重新注入输入 gamma(x)
            （论文 Fig. 7：第 5 层），帮助梯度直达浅层、保留原始位置信息。
        blocks: 主干 8 个 nn.Linear（各 256 通道，配 ReLU）。
        skip_connection: skip 层的 nn.Linear，输入维度 in_dim + hidden。
        sigma_head: 密度头 nn.Linear(hidden, 1)，输出经 ReLU 保证 σ ≥ 0。
        feature_head: 特征头 nn.Linear(hidden, hidden)，输出 256 维几何特征。
        dir_fc / rgb_head: 视角分支（feature ⊕ gamma(d) → 128 → RGB 3 维）；
            不用视角分支时 rgb_head 直接从 256 维特征出 3 维颜色。
    """

    def __init__(
        self,
        in_dim: int = 60,        # 3 * 2 * l_xyz（位置编码维度）
        view_dim: int = 24,      # 3 * 2 * l_dir（方向编码维度）
        hidden: int = 256,
        num_layers: int = 8,
        skip_at: int = 4,        # 第 5 层输入处拼接 gamma(x)
        use_viewdirs: bool = True,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.skip_at = skip_at

        # 主干：8 个全连接 ReLU 层，每层 256 通道
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden
            self.blocks.append(nn.Linear(in_ch, hidden))

        # skip connection：第 5 层消费 [hidden, in_dim] 的拼接输入
        self.skip_connection = nn.Linear(in_dim + hidden, hidden)

        # 主干之后分成两个输出头
        self.sigma_head = nn.Linear(hidden, 1)       # 密度（之后套 ReLU）
        self.feature_head = nn.Linear(hidden, hidden)  # 供颜色分支用的特征

        # 视角相关颜色分支
        if use_viewdirs:
            self.dir_fc = nn.Linear(hidden + view_dim, 128)
            self.rgb_head = nn.Linear(128, 3)
        else:
            self.rgb_head = nn.Linear(hidden, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 均匀初始化所有 Linear 权重、偏置清零（教学实现：训练更稳定，非论文要求）。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor | None = None):
        """前向：位置（+方向）编码 → (rgb, sigma)。

        Args:
            x: 形状 [..., in_dim] 的位置编码特征（gamma(x)，60 维）。
            view_dirs: 形状 [..., view_dim] 的方向编码特征（gamma(d)，24 维）；
                use_viewdirs=True 时必须提供。

        Returns:
            rgb: 形状 [..., 3]，颜色，经 sigmoid ∈ [0, 1]。
            sigma: 形状 [..., 1]，密度，经 ReLU ≥ 0（量纲：1/距离，即单位长度上的
                吸收率；配合 deltas 使用，见 render.volume_render）。
        """
        h = F.relu(self.blocks[0](x))
        for i, layer in enumerate(self.blocks[1:], start=1):
            if i == self.skip_at:                    # 第 5 层：重新注入输入（论文 Fig. 7 的 skip）
                h = torch.cat([h, x], dim=-1)
                h = self.skip_connection(h)
            else:
                h = layer(h)
            h = F.relu(h)

        sigma = F.relu(self.sigma_head(h))           # 密度 ≥ 0（论文：ReLU 保证非负）

        if self.use_viewdirs:
            if view_dirs is None:
                raise ValueError("view_dirs is required when use_viewdirs=True")
            feature = self.feature_head(h)           # 论文：256 维特征（只依赖位置）
            h = F.relu(self.dir_fc(torch.cat([feature, view_dirs], dim=-1)))

        rgb = torch.sigmoid(self.rgb_head(h))        # 颜色 ∈ [0, 1]
        return rgb, sigma

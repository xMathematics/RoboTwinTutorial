"""NeRF MLP (paper Sec. 3 and Fig. 7).

Structure:
    gamma(x) (60-d) -> [FC-ReLU x8, 256 ch, skip at 5th layer]
        ├──> sigma        (ReLU, non-negative density; position-only)
        └──> feature(256) -> concat gamma(d) (24-d) -> [FC-ReLU 128] -> RGB (sigmoid)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    """A single radiance-field MLP. Coarse and fine share this class."""

    def __init__(
        self,
        in_dim: int = 60,        # 3 * 2 * l_xyz
        view_dim: int = 24,      # 3 * 2 * l_dir
        hidden: int = 256,
        num_layers: int = 8,
        skip_at: int = 4,        # concatenate gamma(x) before the 5th layer
        use_viewdirs: bool = True,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.skip_at = skip_at

        # 8 fully-connected ReLU layers, 256 channels each
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden
            self.blocks.append(nn.Linear(in_ch, hidden))

        # skip connection: the 5th layer consumes [hidden, in_dim]
        self.skip_connection = nn.Linear(in_dim + hidden, hidden)

        # outputs split after the trunk
        self.sigma_head = nn.Linear(hidden, 1)       # density  (ReLU applied later)
        self.feature_head = nn.Linear(hidden, hidden)  # feature for color branch

        # view-dependent color branch
        if use_viewdirs:
            self.dir_fc = nn.Linear(hidden + view_dim, 128)
            self.rgb_head = nn.Linear(128, 3)
        else:
            self.rgb_head = nn.Linear(hidden, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor | None = None):
        """x: [..., in_dim], view_dirs: [..., view_dim] -> (rgb, sigma)."""
        h = F.relu(self.blocks[0](x))
        for i, layer in enumerate(self.blocks[1:], start=1):
            if i == self.skip_at:                    # 5th layer: re-inject input
                h = torch.cat([h, x], dim=-1)
                h = self.skip_connection(h)
            else:
                h = layer(h)
            h = F.relu(h)

        sigma = F.relu(self.sigma_head(h))           # density >= 0 (paper: ReLU)

        if self.use_viewdirs:
            if view_dirs is None:
                raise ValueError("view_dirs is required when use_viewdirs=True")
            feature = self.feature_head(h)           # paper: 256-d feature
            h = F.relu(self.dir_fc(torch.cat([feature, view_dirs], dim=-1)))

        rgb = torch.sigmoid(self.rgb_head(h))        # color in [0, 1]
        return rgb, sigma

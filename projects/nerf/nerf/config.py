"""Hyper-parameters (paper Sec. 5.3 Implementation Details)."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- data ----
    datadir: str = "data/lego"          # path to a nerf_synthetic scene dir
    half_res: bool = True               # render/train at half resolution (faster)

    # ---- model (paper Sec. 3 / Fig. 7) ----
    l_xyz: int = 10                     # positional encoding L for xyz (-> 60 dim)
    l_dir: int = 4                      # positional encoding L for view dir (-> 24 dim)
    use_viewdirs: bool = True           # view-dependent color branch
    hidden: int = 256                   # hidden width
    num_layers: int = 8                 # number of fully-connected layers (main trunk)
    skip_at: int = 4                    # 5th layer gets the skip connection

    # ---- sampling (paper Sec. 4 / 5.2) ----
    n_coarse: int = 64                  # N_c coarse samples per ray
    n_fine: int = 128                   # N_f additional fine samples per ray

    # ---- training (paper Sec. 5.3) ----
    batch_size: int = 1024              # rays per step (paper: 4096; 1024 fits smaller GPUs)
    lr: float = 5e-4                    # Adam learning rate
    lr_decay: float = 0.1               # exponential decay factor over training
    steps: int = 200000                 # optimization iterations
    random_seed: int = 0

    # ---- rendering bounds (blender scenes live in a cube of side 2 at origin) ----
    near: float = 2.0
    far: float = 6.0

    # ---- misc ----
    device: str = "cuda"                # falls back to cpu if unavailable
    exp_name: str = "exp"
    log_dir: str = "logs"

"""超参数定义（对应论文 Sec. 5.3 Implementation Details）。

函数流水线：本模块只定义 dataclass `Config`，无包内依赖；被 run_nerf.py（入口处
构造并按命令行参数覆写）与 nerf/trainer.py（读取模型/采样/训练超参）导入。
"""
from dataclasses import dataclass


@dataclass
class Config:
    """NeRF 训练/渲染的全部超参数，默认值对齐论文 Sec. 5.3。

    Attributes:
        datadir: nerf_synthetic 场景目录路径（含 transforms_*.json 与图像子目录）。
        half_res: 是否按半分辨率加载/训练（True 加速迭代；论文用全分辨率）。
        l_xyz: 位置 (x, y, z) 位置编码的频率数 L，编码后每点 3*2*L = 60 维。
        l_dir: 视角方向编码的频率数 L，编码后每方向 3*2*L = 24 维。
        use_viewdirs: 是否启用视角相关颜色分支（论文 Fig. 7 右图结构）。
        hidden: MLP 隐层通道数（论文 256）。
        num_layers: 主干全连接层数（论文 8）。
        skip_at: 在第 skip_at+1（即第 5）层输入处拼接输入 gamma(x) 做 skip connection。
        n_coarse: 每条光线 coarse 网络采样点数 N_c（论文 Eq. 3 的 N）。
        n_fine: 每条光线 fine 网络附加采样点数 N_f（论文 Sec. 5.2）。
        batch_size: 每步采样的光线条数（论文 4096；1024 适配小显存 GPU）。
        lr: Adam 初始学习率。
        lr_decay: 训练全程指数学习率衰减因子（终止学习率 = lr * lr_decay）。
        steps: 总优化步数。
        random_seed: torch / numpy 随机种子。
        near / far: 光线采样近/远边界（Blender 场景位于原点处边长 2 的立方体内，
            相机距离约 4，故默认 [2, 6]；单位与场景世界坐标一致）。
        device: 计算设备（"cuda" 不可用时由 run_nerf.make_config 回退为 "cpu"）。
        exp_name: 实验名（checkpoint 写入 log_dir/exp_name/ 下）。
        log_dir: checkpoint 输出根目录。
    """

    # ---- 数据 ----
    datadir: str = "data/lego"          # nerf_synthetic 场景目录路径
    half_res: bool = True               # 半分辨率加载/训练（更快；论文用全分辨率）

    # ---- 模型（论文 Sec. 3 / Fig. 7）----
    l_xyz: int = 10                     # 位置编码频率数 L（xyz → 60 维）
    l_dir: int = 4                      # 视角方向编码频率数 L（→ 24 维）
    use_viewdirs: bool = True           # 视角相关颜色分支
    hidden: int = 256                   # 隐层宽度
    num_layers: int = 8                 # 主干全连接层数
    skip_at: int = 4                    # 第 5 层输入处接 skip connection

    # ---- 采样（论文 Sec. 4 / 5.2）----
    n_coarse: int = 64                  # 每条光线 coarse 采样点数 N_c
    n_fine: int = 128                   # 每条光线 fine 附加采样点数 N_f

    # ---- 训练（论文 Sec. 5.3）----
    batch_size: int = 1024              # 每步光线数（论文 4096；1024 适配小显存 GPU）
    lr: float = 5e-4                    # Adam 学习率
    lr_decay: float = 0.1               # 训练全程的指数衰减因子
    steps: int = 200000                 # 优化迭代步数
    random_seed: int = 0

    # ---- 渲染边界（Blender 场景位于原点处边长为 2 的立方体内）----
    near: float = 2.0
    far: float = 6.0

    # ---- 其他 ----
    device: str = "cuda"                # 不可用时回退 cpu（见 run_nerf.make_config）
    exp_name: str = "exp"
    log_dir: str = "logs"

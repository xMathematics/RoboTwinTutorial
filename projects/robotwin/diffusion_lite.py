"""扩散策略的最小教学版（diffusion_lite）——2D 轨迹去噪生成的 DDPM。

函数流水线
----------
Diffusion Policy（Chi et al., RSS 2023, arXiv:2303.04137；本地精读
[tutorials/robotwin/精读/DiffusionPolicy_RSS2023.md](../../tutorials/robotwin/精读/DiffusionPolicy_RSS2023.md)；
教程第 08 章 §8.2）把策略表示为**条件去噪扩散概率模型**（DDPM）：在动作
序列上从高斯噪声迭代去噪，随机 Langevin 动力学的采样机制天然表达多峰
动作分布（精读 §4.4）。本模块是该思想在纯 NumPy 下的最小可跑版本——
生成对象 = 本项目 push/reach 任务末端轨迹的 K=8 路径点（16 维向量），
条件 = (起点, 终点) 4 维。

调用链：:func:`collect_demonstrations`（**复用** ``tasks`` 的回合接口与
``policies.CalibratedPolicy``：在名义世界里滚动策略、从 obs["ee"] 记录
末端轨迹、降采样到 K=8 路径点并归一化到工作区单位）→
:func:`linear_beta_schedule`（线性 β 调度，T=50 教学规模）→
:func:`q_sample`（前向加噪闭式 (D0′)，Ho et al. 2020）→
:class:`NoiseMLP` + :func:`eps_prediction_loss` + :func:`ddpm_train`
（小型 MLP 噪声网络的**手写前向/反向** + MSE ε-预测损失 + Adam，不依赖
torch）→ :func:`ddpm_sample`（DDPM 反向链：从纯噪声迭代去噪）→
:func:`plan`（(起点, 终点) 条件的米制封装）→ :func:`smoothness`（二阶
差分范数，轨迹平滑性度量）。被 ``tests/test_diffusion_lite.py`` 驱动。
依赖方向：``diffusion_lite → tasks, policies, dr``（仅演示数据采集），
库内无反向依赖；不 import ``control_planning/ppo_lite``（手写反传自含）。

与原文的差异（教学化简，皆有意识为之）
------------------------------------
1. **生成对象**：原文是机器人动作序列（$A_t \\in \\mathbb{R}^{T_p \\times d_a}$，
   视觉观测条件化，ResNet-18 编码图像，精读 §4.6）；这里是无视觉的
   **2D 末端路径点序列**，条件退化为 (起点, 终点)——保留"序列生成 +
   条件化"的结构，去掉视觉编码器与滚动时域执行（原文 §2.3 的 receding
   horizon：本模块一次生成整条路径，开环使用）。
2. **去噪骨干**：原文用 1D 时序 CNN / Transformer（精读 §4.5）；这里是
   3 层 ReLU MLP + 正弦时间嵌入 + **零初始化残差捷径**（输入的带噪轨迹
   块直连输出，$U$ 从 0 学起）的手写前向/反向——梯度链逐行可读，是
   "扩散策略在学什么"的最小透明标本。捷径的依据：大 $t$ 步的最优解
   $\\approx$ 恒等复制 $\\epsilon\\hat{} \\approx x^t$（$x^t \\approx \\epsilon$），
   给它一条捷径可显著加速收敛（教学实测：同等轮数下端点误差减半）。
3. **噪声调度与采样**：原文按其 (α, γ, σ) 移参记号实现（论文式 (1)(3)，
   精读 §4.1–4.2），训练 100 步、推理用 DDIM 加速到 10 步（精读 §4.6）；
   这里按 Ho et al. 2020 的标准 DDPM 闭式 (D0′) 实现，线性 β 调度
   T=50、ancestral sampling 反向 50 步，步长方差取后验方差
   $\\sigma_t^2 = \\tilde\\beta_t$（Ho et al. 2020 §3.2 的另一标准选择，
   比 $\\sigma_t^2 = \\beta_t$ 更"确定"，末端步 $\\tilde\\beta_0 = 0$——
   教学实测采样端点误差约减半）；不引入 DDIM，训练步数减半（教学规模
   优先：秒级训练）。
4. **无 CVAE/无分类器引导**：原文对比的 ACT 式 CVAE 与引导采样均不在
   范围内——本模块只复刻"DDPM 训练目标 (3)/(5) + 反向采样 (1)/(4)"
   的主干（条件化按精读式 (4)(5) 的口径：条件只进噪声网络，不进扩散
   变量）。

确定性约定：全部随机性走 ``np.random.default_rng(seed)``（演示采集 /
训练 / 采样各自独立 seed），同 seed 逐位复现；数值全程 float64。
"""
from __future__ import annotations

import numpy as np

from dr import WorldParams
from policies import CalibratedPolicy
from tasks import LEFT, RIGHT, make_task

__all__ = [
    "K_WAYPOINTS",
    "TRAJ_DIM",
    "COND_DIM",
    "T_STEPS",
    "BETA_START",
    "BETA_END",
    "WORKSPACE_CENTER",
    "WORKSPACE_SCALE",
    "TIME_EMB_DIM",
    "HIDDEN",
    "BETAS",
    "ALPHAS_BAR",
    "BETA_TILDE",
    "linear_beta_schedule",
    "cumulative_alpha_bar",
    "time_embedding",
    "q_sample",
    "collect_demonstrations",
    "to_workspace",
    "from_workspace",
    "NoiseMLP",
    "eps_prediction_loss",
    "ddpm_train",
    "ddpm_sample",
    "plan",
    "smoothness",
]

# ---------------------------------------------------------------- 常数 --
#: 每条轨迹的路径点数 K（原文 $T_p$ 动作视野的教学对应；8 点 × 2 维 = 16）。
K_WAYPOINTS: int = 8
#: 扩散变量的维数 = K × 2（16 维向量，原文 $T_p \\times d_a$ 的展平对应）。
TRAJ_DIM: int = K_WAYPOINTS * 2
#: 条件维数 = (起点, 终点) × 2（原文 $O_t$ 观测嵌入的退化对应）。
COND_DIM: int = 4
#: DDPM 总步数 T（教学规模：原文训练 100 步 + DDIM 推理 10 步，见模块
#: docstring 差异 3；减半是为把训练测试压进 8 s 预算）。
T_STEPS: int = 50
#: 线性 β 调度两端（β_1, β_T）。经典 DDPM（T=1000）用 [1e-4, 0.02]；
#: T=50 时按同比例放大使 $\\bar\\alpha_T$ 仍 ≈ 0（先验 ≈ 纯噪声），
#: 实测 $\\bar\\alpha_T \\approx 2.2\\times10^{-3}$（tests 有守恒断言）。
BETA_START: float = 0.005
BETA_END: float = 0.24
#: 时间嵌入维数（正弦/余弦各半；原文 Transformer 变体的轮次嵌入对应物）。
TIME_EMB_DIM: int = 16
#: MLP 隐层宽度（原文 CNN/Transformer 骨干的教学最小对应；教学实测
#: hidden=96 + 3500 轮全批训练在 ~5 s 内把采样端点误差压到 < 0.05 m）。
HIDDEN: int = 96
#: 工作区归一化：路径点 (x, y)（m）→ ((x−0.45)/0.4, y/0.4)，使数据集中在
#: O(1) 区间（DDPM 对数据尺度敏感：米制下数据 std ~0.15 会被单位方差
#: 噪声淹没）。0.45 m ≈ 桌面可达区中心、0.4 m ≈ 其半径（tasks.SCENE_RANGES
#: 与 HOME_Q 末端的包络）。
WORKSPACE_CENTER: tuple[float, float] = (0.45, 0.0)
WORKSPACE_SCALE: float = 0.4


# ------------------------------------------------------------ 噪声调度 --
def linear_beta_schedule(
    t_steps: int = T_STEPS, beta_start: float = BETA_START, beta_end: float = BETA_END
) -> np.ndarray:
    """线性 β 调度：$\\beta_t$ 在 [beta_start, beta_end] 内等间隔（(D0) 的 β 序列）。

    Args:
        t_steps: 总步数 T。
        beta_start / beta_end: 调度两端（无量纲，每步噪声方差系数）。

    Returns:
        (T,) β 序列，float64，严格递增、全部 ∈ (0, 1)。
    """
    return np.linspace(beta_start, beta_end, int(t_steps))


def cumulative_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """$\\bar\\alpha_t = \\prod_{j \\le t}(1-\\beta_j)$（(D0′) 的累积量）。

    Args:
        betas: (T,) β 序列。

    Returns:
        (T,) $\\bar\\alpha$ 序列，单调递减；$\\bar\\alpha_T \\approx 0$
        时前向终态 ≈ 标准高斯（采样先验的依据）。
    """
    return np.cumprod(1.0 - np.asarray(betas, dtype=float))


#: 模块默认调度（教学口径：调度是"协议常数"的一部分，与 benchmark 种子
#: 规则同理写死；测试对 BETAS/ALPHAS_BAR 的性质有守恒断言）。
BETAS: np.ndarray = linear_beta_schedule()
ALPHAS_BAR: np.ndarray = cumulative_alpha_bar(BETAS)
#: 后验方差 $\\tilde\\beta_t = \\beta_t (1-\\bar\\alpha_{t-1})/(1-\\bar\\alpha_t)$
#: （Ho et al. 2020 §3.2；约定 $\\bar\\alpha_{-1} = 1$ ⇒ $\\tilde\\beta_0 = 0$，
#: 最后一步去噪完全确定）。ddpm_sample 的步长方差用它。
ALPHAS_BAR_PREV: np.ndarray = np.concatenate([[1.0], ALPHAS_BAR[:-1]])
BETA_TILDE: np.ndarray = BETAS * (1.0 - ALPHAS_BAR_PREV) / (1.0 - ALPHAS_BAR)


def time_embedding(t_idx: np.ndarray, t_steps: int = T_STEPS, dim: int = TIME_EMB_DIM) -> np.ndarray:
    """去噪轮次 k 的正弦时间嵌入（Transformer 轮次 token 的 MLP 对应物）。

    频率取 $f_i = 2\\pi (i{+}1)/T$：每支频率在 $[0, T)$ 上恰走整数个周期，
    T=50 的 50 个轮次嵌入互相拉开（经典 1/10000^i 频率在 T=50 下几乎
    重合，不适合小 T）。

    Args:
        t_idx: (B,) 去噪轮次（整数 0..T−1）。
        t_steps / dim: 总步数与嵌入维数（dim 须为偶数）。

    Returns:
        (B, dim) 嵌入 = [sin(k·f_1..k·f_{dim/2}), cos(·)]，float64。
    """
    t_idx = np.asarray(t_idx, dtype=float).reshape(-1)
    freqs = 2.0 * np.pi * np.arange(1, dim // 2 + 1) / float(t_steps)  # (dim/2,)
    angles = t_idx[:, None] * freqs[None, :]  # (B, dim/2)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def q_sample(x0: np.ndarray, t_idx: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """前向加噪闭式 (D0′)：$x^t = \\sqrt{\\bar\\alpha_t}\\,x^0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$。

    出处：Ho et al. 2020 的边缘闭式（Gaussian 可加性，精读 §4.1 (D0′)，
    依据：沿 (D0) 逐步代入、方差线性累加）。训练按它一步抽任意轮次的
    带噪样本（原文 §2.2 "随机抽轮次覆盖所有噪声水平"）。

    Args:
        x0: (B, D) 干净数据（这里 = 16 维路径点向量）。
        t_idx: (B,) 轮次。
        eps: (B, D) 标准高斯噪声。

    Returns:
        (B, D) 带噪样本 $x^t$。
    """
    x0 = np.asarray(x0, dtype=float)
    eps = np.asarray(eps, dtype=float)
    abar = ALPHAS_BAR[np.asarray(t_idx, dtype=int)]  # (B,)
    root_a = np.sqrt(abar)[:, None]
    root_1ma = np.sqrt(1.0 - abar)[:, None]
    return root_a * x0 + root_1ma * eps


# ------------------------------------------------------------ 演示数据 --
def _resample_curve(path: np.ndarray, k: int = K_WAYPOINTS) -> np.ndarray:
    """按**弧长**把折线轨迹 (T, 2) 线性插值重采样为 k 个等弧距点 (k, 2)。

    依据：均匀时间索引取整会在"先走完一段再停住"的轨迹上产生重复路径点
    （回合末段策略已到位、末端静止——教学实测 reach 轨迹尾部连成一段），
    弧长参数化保证 k 个点沿路径均匀铺开、互不重复（数据的信息量更大，
    条件均值更平滑）。

    Args:
        path: (T, 2) 轨迹（单位 m，T ≥ 2）。
        k: 目标点数。

    Returns:
        (k, 2) 重采样轨迹；路径总长 < 1e-9 时退化为重复首点。
    """
    path = np.asarray(path, dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)  # (T-1,) 每段长度
    arc = np.concatenate([[0.0], np.cumsum(seg)])  # (T,) 累积弧长
    s = np.linspace(0.0, arc[-1] if arc[-1] > 1e-9 else 1.0, k)  # 等弧距网格
    return np.stack([np.interp(s, arc, path[:, 0]), np.interp(s, arc, path[:, 1])], axis=1)


def collect_demonstrations(
    n_reach: int = 16, n_push: int = 16, seed: int = 0, split: str = "seen"
) -> tuple[np.ndarray, np.ndarray]:
    """从本项目 push/reach 任务采演示轨迹，降采样为 (条件, 路径点) 数据集。

    数据来源**复用回合接口**（不读环境内部状态）：名义世界
    （``WorldParams.nominal()``，无观测噪声）+ ``CalibratedPolicy``
    （标定世界里即完美，轨迹无失败段），每回合从 ``obs["ee"]`` 记录末端
    位置序列：reach 记双臂两条轨迹（各臂独立成样本）、push 记对象侧
    活跃臂一条。每条轨迹按弧长重采样为 K=8 路径点；条件 = (首点,
    末点)。归一化到工作区单位（:func:`to_workspace`）后返回。

    Args:
        n_reach / n_push: reach / push 的回合数（reach 每回合产 2 条样本）。
        seed: 场景种子基值（第 i 回合 reach 用 ``seed*1000+i``、push 用
            ``seed*1000+1000+i``——两任务种子段互不重叠，确定性）。
        split: 目标位姿分层（``"seen"`` / ``"unseen"``，透传 ``reset``；
            教学默认 seen——训练分布即标定分布，与 §4.2.1 ① 的口径一致）。

    Returns:
        ``(conds, trajs)``：conds (N, 4)、trajs (N, 16)，均为归一化单位
        （乘 :data:`WORKSPACE_SCALE` 加 :data:`WORKSPACE_CENTER` 还原米）；
        N = 2·n_reach + n_push。trajs 的第 k 个路径点 = trajs[2k:2k+2]。
    """
    policy = CalibratedPolicy()
    conds: list[np.ndarray] = []
    trajs: list[np.ndarray] = []

    def _record(task: str, episode_seed: int, sides_fn) -> None:
        """跑一个回合，把 ``sides_fn(env)`` 指定臂的末端轨迹各记为一条样本。

        ``sides_fn`` 在 reset 后调用：reach 固定双臂各一条；push 按对象
        侧（``env.side``，tasks 的左右分工约定）选活跃臂一条。
        """
        env = make_task(task, WorldParams.nominal())
        obs = env.reset(seed=episode_seed, split=split)
        policy.reset(task)
        sides = tuple(sides_fn(env))
        history: dict[int, list[np.ndarray]] = {s: [obs["ee"][s].copy()] for s in sides}
        done, step = False, 0
        while not done:  # 逐回合滚动到 done（策略恒成功，无失败段污染）
            action = policy.act(obs, step)
            obs, _reward, done, _info = env.step(action)
            for s in sides:
                history[s].append(obs["ee"][s].copy())
            step += 1
        for s in sides:
            path = np.asarray(history[s], dtype=float)  # (T+1, 2)，单位 m
            traj_m = _resample_curve(path, K_WAYPOINTS)  # (K, 2) 弧长均匀重采样
            trajs.append(to_workspace(traj_m).reshape(-1))  # (16,) 归一化展平
            cond_m = np.concatenate([traj_m[0], traj_m[-1]])  # (4,) 起点+终点
            conds.append(to_workspace(cond_m.reshape(2, 2)).reshape(-1))

    for i in range(int(n_reach)):
        _record("reach", seed * 1000 + i, lambda _env: (LEFT, RIGHT))  # 双臂各一条
    for i in range(int(n_push)):
        _record("push", seed * 1000 + 1000 + i,
                lambda env: (LEFT if env.side > 0 else RIGHT,))
    return np.asarray(conds, dtype=float), np.asarray(trajs, dtype=float)


def to_workspace(pts: np.ndarray) -> np.ndarray:
    """米制点集 (..., 2) → 工作区归一化单位（中心平移 + 尺度缩放）。"""
    pts = np.asarray(pts, dtype=float)
    center = np.asarray(WORKSPACE_CENTER, dtype=float)
    return (pts - center) / WORKSPACE_SCALE


def from_workspace(pts: np.ndarray) -> np.ndarray:
    """工作区归一化点集 (..., 2) → 米制（:func:`to_workspace` 的逆）。"""
    pts = np.asarray(pts, dtype=float)
    center = np.asarray(WORKSPACE_CENTER, dtype=float)
    return pts * WORKSPACE_SCALE + center


# -------------------------------------------------------- 噪声网络 MLP --
class NoiseMLP:
    """ε-预测网络：3 层 ReLU MLP + 残差捷径，**手写前向/反向**（不依赖 torch）。

    输入 = [条件 (4,) | 时间嵌入 (16,) | 带噪轨迹 (16,)] = 36 维，对应
    原文 $\\epsilon_\\theta(O_t, A^k_t, k)$（精读式 (4) 的参数化：条件
    只进网络、不进扩散变量）。**残差捷径**：输入末尾 :attr:`skip_dim`
    维（即带噪轨迹块 $x^t$）经零初始化矩阵 $U$ 直连输出——大 $t$ 步的
    最优解近似恒等（$\\epsilon \\hat{} \\approx x^t$），捷径让该解容易
    表达（依据：$x^T \\approx \\epsilon$ 时最优预测器趋于复制输入；
    零初始化保证训练起点等价于无捷径的普通 MLP）。手写反传的梯度链
    （batch 记 B）：

    - ``dOut = 2(pred − eps)/(B·16)``（MSE 对输出的梯度，见
      :func:`eps_prediction_loss`）；
    - 每层 ``dW = dz^T @ h_prev``、``db = dz.sum(0)``、
      ``dh_prev = dz @ W``、过 ReLU 处 ``dz *= (z > 0)``；
    - 捷径 ``dU = dOut^T @ x_skip``（U 是叶参数，梯度不回传到输入）。

    Attributes:
        params: dict[str, np.ndarray]，键 W1/b1/W2/b2/W3/b3（+U 当
            ``skip_dim > 0``）；W1 (H, D_in)、W2 (H, H)、W3 (D_out, H)、
            U (D_out, skip_dim)。
    """

    def __init__(self, in_dim: int, hidden: int = HIDDEN, out_dim: int = TRAJ_DIM,
                 rng: np.random.Generator | None = None,
                 skip_from: int = -1, skip_dim: int = 0) -> None:
        """He 初始化（依据：ReLU 网络的方差保持惯例），偏置与捷径 U 置零。

        Args:
            in_dim: 输入维（COND_DIM + TIME_EMB_DIM + TRAJ_DIM = 36）。
            hidden: 隐层宽度。
            out_dim: 输出维（= TRAJ_DIM，预测同形噪声 ε）。
            rng: 初始化用的随机数发生器（None = default_rng(0)，确定性）。
            skip_from: 残差捷径在输入中的起始列（< 0 = 关闭捷径）；本项目
                用输入布局 [cond | temb | x_t]，故 skip_from = 20。
            skip_dim: 捷径块宽度（0 = 关闭）。
        """
        rng = np.random.default_rng(0) if rng is None else rng
        self.in_dim, self.hidden, self.out_dim = int(in_dim), int(hidden), int(out_dim)
        self.skip_from, self.skip_dim = int(skip_from), int(skip_dim)
        # 权重形状按"输出 × 输入"存（前向 x @ W.T），He 尺度 sqrt(2/fan_in)。
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(hidden, in_dim)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, hidden)),
            "b2": np.zeros(hidden),
            "W3": rng.normal(0.0, np.sqrt(2.0 / hidden), size=(out_dim, hidden)),
            "b3": np.zeros(out_dim),
        }
        if self.skip_dim > 0:
            self.params["U"] = np.zeros((out_dim, self.skip_dim))  # 零初始化：起点 = 无捷径

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向：缓存隐层激活供 :meth:`backward` 反传，返回网络输出。

        Args:
            x: (B, in_dim) 输入（训练时 = concat(条件, 时间嵌入, x^t)）。

        Returns:
            (B, out_dim) 预测噪声 $\\epsilon_\\theta$。
        """
        x = np.asarray(x, dtype=float)
        p = self.params
        z1 = x @ p["W1"].T + p["b1"]  # (B, H)
        h1 = np.maximum(z1, 0.0)  # ReLU
        z2 = h1 @ p["W2"].T + p["b2"]  # (B, H)
        h2 = np.maximum(z2, 0.0)
        out = h2 @ p["W3"].T + p["b3"]  # (B, out_dim)
        if self.skip_dim > 0:
            out = out + x[:, self.skip_from:self.skip_from + self.skip_dim] @ p["U"].T
        # 缓存本次前向的激活（backward 只在紧随的 forward 后调用有效）。
        self._cache = (x, z1, h1, z2, h2)
        return out

    def backward(self, grad_out: np.ndarray) -> dict[str, np.ndarray]:
        """反向：对紧随的 :meth:`forward` 计算损失对全部参数的梯度。

        Args:
            grad_out: (B, out_dim) 损失对网络输出的梯度
                （MSE 下 = 2(pred−eps)/(B·out_dim)）。

        Returns:
            dict[str, np.ndarray]，与 :attr:`params` 同键同形。
        """
        x, z1, h1, z2, h2 = self._cache
        p = self.params
        grad_out = np.asarray(grad_out, dtype=float)
        grads: dict[str, np.ndarray] = {}
        grads["W3"] = grad_out.T @ h2  # (out, H)
        grads["b3"] = grad_out.sum(axis=0)  # (out,)
        if self.skip_dim > 0:
            # 捷径：dL/dU = dOut^T @ x_skip（x 视为叶输入，不回传）。
            grads["U"] = grad_out.T @ x[:, self.skip_from:self.skip_from + self.skip_dim]
        dh2 = grad_out @ p["W3"]  # (B, H)
        dz2 = dh2 * (z2 > 0.0)  # 过 ReLU 的链式（掩码）
        grads["W2"] = dz2.T @ h1
        grads["b2"] = dz2.sum(axis=0)
        dh1 = dz2 @ p["W2"]
        dz1 = dh1 * (z1 > 0.0)
        grads["W1"] = dz1.T @ x
        grads["b1"] = dz1.sum(axis=0)
        return grads


def eps_prediction_loss(
    model: NoiseMLP,
    cond: np.ndarray,
    x_t: np.ndarray,
    t_idx: np.ndarray,
    eps: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    """MSE ε-预测损失 + 全部参数梯度（原文式 (5) 的 DDPM 标准闭式版）。

    $$\\mathscr{L} = \\mathrm{MSE}\\big(\\epsilon,\\ \\epsilon_\\theta(\\mathrm{cond},\\ x^t,\\ t)\\big)$$

    与精读式 (5) 的对应：条件（此处 = (起点, 终点)，原文 = $O_t$）与轮次
    只进噪声网络；带噪样本由 :func:`q_sample` 按 (D0′) 生成（论文式 (5)
    记作 $A^0 + \\epsilon^k$，是同一目标在其 (α,γ,σ) 记号下的省缩放写法）。

    Args:
        model: :class:`NoiseMLP` 实例。
        cond: (B, 4) 条件（起点+终点，归一化单位）。
        x_t: (B, 16) 带噪轨迹。
        t_idx: (B,) 轮次。
        eps: (B, 16) 真值噪声。

    Returns:
        ``(loss, grads)``：标量损失（全体元素均值）与参数梯度 dict。
    """
    a0 = np.concatenate([cond, time_embedding(t_idx), x_t], axis=1)  # (B, 36)
    pred = model.forward(a0)
    diff = pred - eps
    loss = float(np.mean(diff * diff))
    grad_out = 2.0 * diff / diff.size  # MSE 均值对 pred 的梯度
    return loss, model.backward(grad_out)


class _Adam:
    """手写 Adam（Kingma & Ba 2015）：m/v 指数滑动 + 偏置修正。仅训练用。"""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """就地更新全部参数（梯度已由 :func:`eps_prediction_loss` 算好）。"""
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t  # 偏置修正系数
        bc2 = 1.0 - self.beta2 ** self.t
        for key, g in grads.items():
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * g * g
            m_hat = self.m[key] / bc1
            v_hat = self.v[key] / bc2
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def ddpm_train(
    conds: np.ndarray,
    trajs: np.ndarray,
    epochs: int = 3500,
    batch_size: int = 0,
    lr: float = 6e-3,
    lr_end: float = 2e-4,
    seed: int = 0,
) -> tuple[NoiseMLP, list[float]]:
    """训练 DDPM 噪声网络（确定性：同 seed 同损失曲线，逐位复现）。

    每轮把数据洗牌后按 batch 采样：随机轮次 $t \\sim U\\{0..T{-}1\\}$、
    $\\epsilon \\sim \\mathcal{N}(0, I)$、:func:`q_sample` 生成 $x^t$、
    :func:`eps_prediction_loss` 算损失与梯度、Adam 更新——原文 §2.2
    训练循环的逐行对应。学习率从 ``lr`` 余弦退火到 ``lr_end``（依据：
    余弦退火是小型扩散模型收敛的标准配置）。

    Args:
        conds: (N, 4) 条件（:func:`collect_demonstrations` 的输出）。
        trajs: (N, 16) 干净路径点向量 $x^0$。
        epochs: 轮数（教学默认 3500 全批 ≈ 5 s，端点误差 < 0.05 m，实测）。
        batch_size: 每批样本数；``<= 0`` = 全数据一批（N 小时的推荐配置——
            随机性来自每步重采的 $t$ 与 $\\epsilon$，不必再 minibatch）。
        lr / lr_end: Adam 学习率退火两端。
        seed: 全部随机性（初始化/洗牌/轮次/噪声）的单一种子。

    Returns:
        ``(model, loss_history)``：训练好的 :class:`NoiseMLP` 与逐轮
        平均损失（len = epochs，下降趋势有测试守恒）。
    """
    rng = np.random.default_rng(seed)
    conds = np.asarray(conds, dtype=float)
    trajs = np.asarray(trajs, dtype=float)
    n = conds.shape[0]
    bs = n if int(batch_size) <= 0 else min(int(batch_size), n)
    model = NoiseMLP(COND_DIM + TIME_EMB_DIM + TRAJ_DIM, HIDDEN, TRAJ_DIM, rng,
                     skip_from=COND_DIM + TIME_EMB_DIM, skip_dim=TRAJ_DIM)
    opt = _Adam(model.params, lr=lr)
    loss_history: list[float] = []
    for epoch in range(int(epochs)):
        # 余弦退火（半余弦包络：lr(end) = lr_end，中点 ≈ (lr+lr_end)/2）。
        opt.lr = lr_end + (lr - lr_end) * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs))
        order = rng.permutation(n)
        epoch_losses: list[float] = []
        for start in range(0, n - bs + 1, bs):
            sel = order[start:start + bs]
            t_idx = rng.integers(0, T_STEPS, size=bs)
            eps = rng.standard_normal((bs, TRAJ_DIM))
            x_t = q_sample(trajs[sel], t_idx, eps)
            loss, grads = eps_prediction_loss(model, conds[sel], x_t, t_idx, eps)
            opt.step(model.params, grads)
            epoch_losses.append(loss)
        loss_history.append(float(np.mean(epoch_losses)))
    return model, loss_history


def ddpm_sample(
    model: NoiseMLP, conds: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """DDPM 反向采样：从纯噪声迭代去噪生成轨迹（精读式 (4) 的标准闭式版）。

    反向链（Ho et al. 2020 后验均值 + 后验方差
    $\\sigma_t^2 = \\tilde\\beta_t = \\beta_t (1-\\bar\\alpha_{t-1})/(1-\\bar\\alpha_t)$，
    教学实测比 $\\sigma_t = \\sqrt{\\beta_t}$ 的端点误差约减半、轨迹更平滑）：

    $$x^{t-1} = \\frac{1}{\\sqrt{\\alpha_t}}\\Big(x^t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,
    \\epsilon_\\theta(\\mathrm{cond}, x^t, t)\\Big) + \\sqrt{\\tilde\\beta_t}\\,z,\\quad z \\sim \\mathcal{N}(0, I)$$

    $t=0$ 步 $\\tilde\\beta_0 = 0$，完全确定（不再注噪）。同 seed 同输出
    （确定性）；不同 seed 的随机初值 + 每步注噪落入不同收敛盆——多峰
    表达力的来源（精读 §4.4）。

    Args:
        model: 训练好的 :class:`NoiseMLP`。
        conds: (N, 4) 条件（归一化单位）。
        rng: 采样噪声的发生器（初值 + 每步注噪，顺序固定保证可复现）。

    Returns:
        (N, 16) 生成的干净轨迹向量 $\\hat{x}^0$（归一化单位）。
    """
    conds = np.asarray(conds, dtype=float)
    n = conds.shape[0]
    x = rng.standard_normal((n, TRAJ_DIM))  # 先验：x^T ~ N(0, I)（ᾱ_T ≈ 0.0012）
    alphas = 1.0 - BETAS
    for k in range(T_STEPS - 1, -1, -1):
        t_idx = np.full(n, k, dtype=int)
        pred = model.forward(np.concatenate([conds, time_embedding(t_idx), x], axis=1))
        coef = BETAS[k] / np.sqrt(1.0 - ALPHAS_BAR[k])
        x = (x - coef * pred) / np.sqrt(alphas[k])
        if k > 0:
            x = x + np.sqrt(BETA_TILDE[k]) * rng.standard_normal((n, TRAJ_DIM))
    return x


def plan(
    model: NoiseMLP, start: np.ndarray, end: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """给定 (起点, 终点)（米制）生成一条 K=8 路径点的 2D 轨迹（米制返回）。

    :func:`ddpm_sample` 的米制封装：条件归一化 → 采样 → 反归一化。
    原文里这一步 = "给定观测生成动作序列"；这里观测退化为起终点。

    Args:
        model: 训练好的噪声网络。
        start / end: (2,) 起点 / 终点，单位 m（桌面系）。
        rng: 采样发生器。

    Returns:
        (K, 2) 路径点序列，单位 m（行 = 依次路径点，首末 ≈ start/end）。
    """
    cond = to_workspace(np.stack([np.asarray(start, dtype=float),
                                  np.asarray(end, dtype=float)], axis=0)).reshape(1, -1)
    traj_n = ddpm_sample(model, cond, rng)[0].reshape(K_WAYPOINTS, 2)
    return from_workspace(traj_n)


def smoothness(traj: np.ndarray) -> float:
    """轨迹平滑性：二阶差分范数之和（单位 m；越小越平滑）。

    $S = \\sum_{k=2}^{K-1} \\lVert x_{k} - 2x_{k-1} + x_{k-2} \\rVert$——
    直线/缓弧 ≈ 0，折线为转角量，i.i.d. 噪声 ≈ $\\sqrt{6}·\\sigma·\\sqrt{K-2}$
    量级（测试以此为纯噪声基线）。

    Args:
        traj: (K, 2) 路径点（单位 m）。

    Returns:
        标量 ≥ 0。
    """
    traj = np.asarray(traj, dtype=float)
    return float(np.linalg.norm(np.diff(traj, n=2, axis=0), axis=1).sum())

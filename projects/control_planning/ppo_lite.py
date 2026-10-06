"""微型 PPO：线性高斯策略 + GAE + 截断代理目标（教学实现）。

函数流水线
----------
Schulman, Wolski, Dhariwal, Radford, Klimov, *Proximal Policy Optimization
Algorithms*, arXiv 2017（本地 papers/control_planning/frontier/
arXiv-1707.06347_PPO.pdf；截断目标为论文 (7) 式）与 GAE（Schulman et al.,
arXiv:1506.02438, 2016）的教学实现，推导见教程第 08 章 08.1（策略梯度
定理 (8.5)/(8.7)）与 08.2（重要性采样 (8.9) → 代理目标 (8.10) → 截断
(8.12) → GAE (8.13)–(8.15)）。任务环境为 2D 点质量到达（教程 08 章
MDP 设定 (8.1) 的最小实例）：状态 [p, v, g] 6 维、动作 = 加速度 2 维、
20 步短回合。

调用链：:class:`PointMassReachingEnv`（MDP：转移 = 双积分器、回报 =
-位置误差² - 控制功效）→ :class:`PolicyParams`（线性 + softmax 高斯策略
的"高斯版"：μ = Ws + b、可学习 log σ；线性值函数走二次特征）
→ :func:`collect_batch`（采样）→ :func:`compute_gae`（(8.13)–(8.15)）
→ :func:`ppo_update`（截断目标 (8.12) 的**手写前向/反向**：高斯 log 概率
的解析梯度 + Adam，不依赖 torch）→ :func:`train`。被
``tests/test_ppo_lite.py`` 与 ``demo.py`` 驱动。

手写反向的梯度链（对每个样本 t，μ_t = W s_t + b、σ = exp(log_std)）：

- 比率 ``r_t = exp(logπ_θ - logπ_θold)``（(8.9) 的重要性权重）；
- 截断掩码（(8.12) 的分段形状）：``r_t`` 越出 [1-ε, 1+ε] 且优势同号时
  该项梯度为零——多轮 epoch 自动刹车（教程 08.2 ⑤ 的表格）；
- ``∂logπ/∂μ_j = (a_j - μ_j)/σ_j²``、``∂logπ/∂log σ_j =
  (a_j - μ_j)²/σ_j² - 1``，链式乘 ``c_t · r_t``（c_t = 优势或 0）。

约定
----
- 确定性：所有随机性走 ``np.random.default_rng(seed)``；策略初值取
  固定小尺度的确定性生成——同 seed 训练曲线逐位可复现。
- 数值：float64 全程；优势做批次内标准化（减均值除标准差，PPO 惯例）。

示例
----
>>> from ppo_lite import train
>>> params, history = train(n_iters=40, seed=0)
>>> history[-1] > history[0] * 1.5  # doctest: +SKIP
True
"""
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = [
    "PointMassReachingEnv",
    "PolicyParams",
    "compute_gae",
    "collect_batch",
    "gauss_log_prob",
    "policy_gradient_estimate",
    "policy_mean",
    "ppo_update",
    "rollout_episode",
    "train",
    "value_features",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


# ------------------------------------------------------------------ 环境 --
@dataclass
class PointMassReachingEnv:
    """2D 点质量到达环境（教程 08 章 MDP (8.1) 的最小实例）。

    状态 ``s = [px, py, vx, vy, ex, ey]``（m, m/s, m；``e = g - p`` 为
    目标相对误差——用相对量而非绝对目标坐标做观测，可避免固定目标下
    "目标列与偏置共线"的参数退化，线性策略的最优解恰为 PD 形态
    ``μ = K_p e - K_d v``）、动作 = 加速度（m/s²，截断到 u_limit）、
    转移 = 精确欧拉双积分器、回报 ``r = -w_pos‖p - g‖² - w_u‖u‖²``、
    固定 20 步回合。初始状态恒为零（确定性环境：随机性全部来自策略
    采样，便于回归测试）。

    Attributes:
        goal: (2,) 目标位置，单位 m。
        dt: 控制周期，单位 s。
        n_steps: 回合长度 T。
        w_pos / w_u: 位置误差 / 控制功效权重。
        u_limit: 动作截断幅值，单位 m/s²。
    """

    goal: np.ndarray = field(default_factory=lambda: np.array([1.2, 0.6]))
    dt: float = 0.1
    n_steps: int = 20
    w_pos: float = 1.0
    w_u: float = 0.01
    u_limit: float = 8.0

    def _obs(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """打包观测 (6,) = [px, py, vx, vy, gx-px, gy-py]。"""
        e = self.goal - pos
        return np.array([pos[0], pos[1], vel[0], vel[1], e[0], e[1]])

    def reset(self) -> np.ndarray:
        """重置回合：返回初始观测 (6,)（原点静止出发）。"""
        return self._obs(np.zeros(2), np.zeros(2))

    def step(self, obs: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """执行一步：转移 + 回报 + 终止标志。

        Args:
            obs: (6,) 当前观测（末两维为相对误差 e = g - p）。
            u: (2,) 加速度指令，单位 m/s²（自动截断到 ±u_limit）。

        Returns:
            (next_obs (6,), reward, done)。
        """
        u = np.clip(np.asarray(u, dtype=float).reshape(2), -self.u_limit, self.u_limit)
        pos = obs[:2] + obs[2:4] * self.dt
        vel = obs[2:4] + u * self.dt
        err = self.goal - pos  # (2,) 当前位置误差（回报用绝对误差）
        reward = -self.w_pos * float(err @ err) - self.w_u * float(u @ u)
        done = False  # 由外层按步数截断（有限时域 MDP）
        return self._obs(pos, vel), reward, done


# ------------------------------------------------------------------ 策略 --
@dataclass
class PolicyParams:
    """线性高斯策略 + 线性值函数的参数（教程 08 章的 actor-critic 结构）。

    Attributes:
        W: (2, 6) 策略均值线性阵（μ = W s + b）。
        b: (2,) 策略均值偏置。
        log_std: (2,) 高斯对数标准差（可学习参数；σ = exp(log_std)）。
        v_w: (8,) 值函数线性系数（作用于 :func:`value_features`）。
        v_c: 值函数偏置（标量）。
    """

    W: np.ndarray
    b: np.ndarray
    log_std: np.ndarray
    v_w: np.ndarray
    v_c: float

    @staticmethod
    def init(seed: int = 0, scale: float = 0.1) -> "PolicyParams":
        """确定性初始化：小尺度随机 W（打破对称）+ 零偏置 + σ₀ = 0.5。

        Args:
            seed: 随机种子。
            scale: W 的初值标准差。

        Returns:
            :class:`PolicyParams`。
        """
        rng = np.random.default_rng(seed)
        return PolicyParams(
            W=rng.normal(0.0, scale, size=(2, 6)),
            b=np.zeros(2),
            log_std=np.full(2, np.log(0.5)),  # σ₀ = 0.5：探索与可控性的折中
            v_w=np.zeros(8),
            v_c=0.0,
        )


def policy_mean(params: PolicyParams, obs: np.ndarray) -> np.ndarray:
    """策略均值前向：``μ = W s + b``（输入 (N, 6) → 输出 (N, 2)）。"""
    return np.asarray(obs, dtype=float) @ params.W.T + params.b[None, :]


def value_features(obs: np.ndarray) -> np.ndarray:
    """值函数特征（8 维：线性项 + 误差二次项）。

    V(s) 的真值关于位置/速度近似二次（到达任务的最优代价是二次型），
    纯线性特征拟合不动；补上误差 ``e = g - p`` 的平方与 e·v 交叉项后
    线性值函数可覆盖主要形状——这是"特征工程版"的 critic，保持全线性
    以便手写梯度。

    Args:
        obs: (N, 6) 观测批（末两维为相对误差 e = g - p）。

    Returns:
        (N, 8) 特征批 ``[e_x, e_y, v_x, v_y, e_x², e_y², e_x v_x, e_y v_y]``。
    """
    obs = np.asarray(obs, dtype=float)
    e = obs[:, 4:6]  # (N, 2) 相对位置误差 g - p
    v = obs[:, 2:4]  # (N, 2) 速度
    return np.column_stack(
        [e, v, e * e, e * v]
    )


def value_fn(params: PolicyParams, obs: np.ndarray) -> np.ndarray:
    """值函数前向：``V = v_w·φ(s) + v_c``（输入 (N, 6) → 输出 (N,)）。"""
    return value_features(obs) @ params.v_w + params.v_c


def gauss_log_prob(
    params: PolicyParams, obs: np.ndarray, act: np.ndarray
) -> np.ndarray:
    """高斯策略的对数概率 ``log π(a|s)``（批输入 → (N,)）。

    log π = Σ_j [-(a_j-μ_j)²/(2σ_j²) - log σ_j - ½log 2π]（对角协方差）。
    """
    obs = np.asarray(obs, dtype=float)
    act = np.asarray(act, dtype=float)
    mu = policy_mean(params, obs)
    sigma = np.exp(params.log_std)[None, :]
    z = (act - mu) / sigma
    return np.sum(-0.5 * z * z - params.log_std[None, :] - 0.5 * _LOG_2PI, axis=1)


def rollout_episode(
    env: PointMassReachingEnv, params: PolicyParams, rng: np.random.Generator
) -> dict:
    """按当前策略采样一条轨迹（MDP 交互，教程 (8.2) 的轨迹分布采样）。

    Args:
        env: 到达环境。
        params: 当前策略参数（即 (8.10) 的 θ_old）。
        rng: 动作采样的随机源。

    Returns:
        dict：``obs (T, 6)``、``act (T, 2)``、``logp (T,)``、``reward (T,)``、
        ``value (T,)``、``last_obs (6,)``（自举用终态）。
    """
    obs = env.reset()
    obs_list, act_list, logp_list, rew_list, val_list = [], [], [], [], []
    for _ in range(env.n_steps):
        mu = policy_mean(params, obs[None, :])[0]
        sigma = np.exp(params.log_std)
        act = mu + sigma * rng.standard_normal(2)  # 重参数化采样
        obs_list.append(obs)
        act_list.append(act)
        logp_list.append(gauss_log_prob(params, obs[None, :], act[None, :])[0])
        val_list.append(value_fn(params, obs[None, :])[0])
        obs, reward, _ = env.step(obs, act)
        rew_list.append(reward)
    return {
        "obs": np.asarray(obs_list),
        "act": np.asarray(act_list),
        "logp": np.asarray(logp_list),
        "reward": np.asarray(rew_list),
        "value": np.asarray(val_list),
        "last_obs": obs,
    }


def collect_batch(
    env: PointMassReachingEnv, params: PolicyParams, n_episodes: int, seed: int
) -> tuple[list[dict], np.random.Generator]:
    """采样一批轨迹（on-policy：数据按 θ_old 生成，(8.10) 的前提）。

    Args:
        env: 到达环境。
        params: 采样策略参数。
        n_episodes: 回合数。
        seed: 随机种子。

    Returns:
        (episodes, rng)：轨迹 dict 列表与继续可用的随机源。
    """
    rng = np.random.default_rng(seed)
    return [rollout_episode(env, params, rng) for _ in range(n_episodes)], rng


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计（GAE，教程 (8.13)–(8.15)）：TD 残差的 (γλ)ˡ 指数加权。

    实现 (8.15) 的截断自举形式：从段尾反向递推
    ``Â_t = δ_t + γλ Â_{t+1}``（δ 为 (8.13) 的 TD 残差），等价于把
    k 步估计按 (1-λ)λ^{k-1} 加权求和——λ 在"蒙特卡洛无偏"与"单步低方差"
    之间折中（tests/test_ppo_lite.py 与手工 k 步加权递推交叉验证）。

    Args:
        rewards: (T,) 回报序列。
        values: (T,) 各时刻状态价值 V(s_t)。
        last_value: 自举终值 V(s_T)（(8.15) 段尾自举项）。
        gamma: 折扣因子 γ ∈ [0, 1)。
        lam: GAE 的 λ ∈ [0, 1]。

    Returns:
        (advantages (T,), returns (T,))：优势估计与价值回归目标
        ``returns = Â + V``。
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    t_len = rewards.shape[0]
    advantages = np.zeros(t_len)
    adv_next = 0.0  # Â_T（段外为 0：终态后无数据）
    for t in range(t_len - 1, -1, -1):
        v_next = last_value if t == t_len - 1 else values[t + 1]
        # TD 残差 (8.13)：δ_t = r_t + γV(s_{t+1}) - V(s_t)。
        delta = rewards[t] + gamma * v_next - values[t]
        # 反向递推 (8.15) 的截断实现：Â_t = δ_t + γλ Â_{t+1}。
        adv_next = delta + gamma * lam * adv_next
        advantages[t] = adv_next
    return advantages, advantages + values


def policy_gradient_estimate(
    params: PolicyParams,
    episodes: list[dict],
    gamma: float = 1.0,
    weights: list[np.ndarray] | None = None,
) -> PolicyParams:
    """策略梯度估计（教程 (8.7)：score × reward-to-go 的期望）。

    用于与中心差分交叉验证 (8.5) 的正确性（tests/test_ppo_lite.py）：
    ``∇J = E[Σ_t ∇logπ(u_t|s_t)·G_t]``，G_t 为 reward-to-go（(8.7) 的
    因果裁剪）。返回结构与 :class:`PolicyParams` 相同的"梯度"对象
    （v_w/v_c 置零——值函数不进策略梯度，(8.7) 只对 actor 求导）。

    Args:
        params: 采样时策略参数（log π 在 θ_old 处求导）。
        episodes: :func:`rollout_episode` 的输出列表。
        gamma: reward-to-go 的折扣（教程 (8.1) 的无折扣口径取 1.0）。
        weights: 可选的逐样本权重（与 ``episodes`` 对齐，每条 (T,)）——
            传 GAE/优势即 (8.8) 的基线版（减去 V(s_t) 不改变期望、大幅
            降方差）；None 用纯 reward-to-go G_t。

    Returns:
        梯度（同参数布局；用作解析梯度）。
    """
    grad_W = np.zeros_like(params.W)
    grad_b = np.zeros_like(params.b)
    grad_log_std = np.zeros_like(params.log_std)
    # (8.7) 的期望是对**轨迹**取的：估计量 = (1/n_episodes)·Σ_τ Σ_t [...]，
    # 归一化按回合数而非总样本数（后者是 PPO 代理目标的逐样本均值口径，
    # 两者差一个 T 因子——这是历史 bug，tests/test_ppo_lite.py 的中心差分
    # 交叉验证正是为钉住这一点）。
    n_traj = len(episodes)
    sigma = np.exp(params.log_std)
    for idx, ep in enumerate(episodes):
        obs, act, rew = ep["obs"], ep["act"], ep["reward"]
        t_len = rew.shape[0]
        if weights is not None:
            w_seq = np.asarray(weights[idx], dtype=float)
        elif gamma == 1.0:
            # reward-to-go（(8.7) 的 G_t）：后缀和。
            w_seq = np.cumsum(rew[::-1])[::-1]
        else:
            w_seq = np.zeros(t_len)
            acc = 0.0
            for t in range(t_len - 1, -1, -1):
                acc = rew[t] + gamma * acc
                w_seq[t] = acc
        mu = policy_mean(params, obs)  # (T, 2)
        z = (act - mu) / sigma[None, :]
        for t in range(t_len):
            # ∇_μ logπ = z/σ；∇_logσ logπ = z² - 1（对角高斯的解析导数）。
            w = w_seq[t]  # 标量权重（期望意义下的 (8.7)/(8.8)）
            grad_W += w * np.outer(z[t] / sigma, obs[t])
            grad_b += w * z[t] / sigma
            grad_log_std += w * (z[t] * z[t] - 1.0)
    grad_W /= n_traj
    grad_b /= n_traj
    grad_log_std /= n_traj
    return PolicyParams(grad_W, grad_b, grad_log_std, np.zeros(8), 0.0)


# ------------------------------------------------------------ PPO 更新 --
def _adam_step(
    param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, lr: float, t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单参数数组的 Adam 步（Kingma & Ba 2015；确定性，无随机性）。"""
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * grad * grad
    m_hat = m_new / (1.0 - beta1**t)
    v_hat = v_new / (1.0 - beta2**t)
    return param - lr * m_hat / (np.sqrt(v_hat) + adam_eps), m_new, v_new


def ppo_update(
    params: PolicyParams,
    episodes: list[dict],
    lr: float = 0.05,
    clip: float = 0.2,
    epochs: int = 10,
    vf_coef: float = 0.5,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr_log_std: float | None = None,
    target_kl: float | None = 0.05,
) -> tuple[PolicyParams, dict]:
    """PPO 多轮更新（教程 (8.12) 截断代理目标的批量上升，手写反向）。

    流程：算 GAE (8.15) → 优势标准化 → 对同一批数据做 ``epochs`` 轮
    全批量上升。每轮记录诊断：``approx_kl``（新旧行为的 KL 代理，健康值
    应 ≪ 1；DEBUG.md 变量表 P-5）、``clip_frac``（比率越出 [1-ε,1+ε]
    的样本占比，应随更新轮次上升但不过半）。

    Args:
        params: 采样策略参数（θ_old）。
        episodes: :func:`collect_batch` 的输出。
        lr: Adam 学习率。
        clip: 截断半径 ε（论文示例 0.2 量级）。
        epochs: 同批数据的更新轮数（(8.12) 的"自动刹车"允许 > 1）。
        vf_coef: 价值损失系数。
        gamma / lam: 透传给 :func:`compute_gae`。
        lr_log_std: 对数标准差参数的独立学习率（None = 用 ``lr``；调小可
            抑制 σ 的噪声漂移，见 DEBUG.md 变量表 P-4）。
        target_kl: 早停阈值——某轮更新后 approx_kl 超过它就停止本轮剩余
            epoch（PPO 论文 §4 的自适应刹车口径；None 关闭）。

    Returns:
        (new_params, info)：更新后的参数与诊断 dict（``approx_kl``、
        ``clip_frac``、``surrogate``、``value_loss``、逐轮列表）。
    """
    # ---- GAE + 目标整理（批内拼接，优势标准化：PPO 惯例的数值卫生）。
    all_obs, all_act, all_logp = [], [], []
    all_adv, all_ret = [], []
    for ep in episodes:
        adv, ret = compute_gae(
            ep["reward"], ep["value"], value_fn(params, ep["last_obs"][None, :])[0],
            gamma, lam,
        )
        all_obs.append(ep["obs"])
        all_act.append(ep["act"])
        all_logp.append(ep["logp"])
        all_adv.append(adv)
        all_ret.append(ret)
    obs = np.concatenate(all_obs)  # (S, 6)
    act = np.concatenate(all_act)  # (S, 2)
    logp_old = np.concatenate(all_logp)  # (S,)
    adv = np.concatenate(all_adv)
    ret = np.concatenate(all_ret)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    new_params = replace(params)
    # Adam 状态：每个参数数组一份（一阶/二阶矩）。
    state = {
        "W": [np.zeros_like(new_params.W), np.zeros_like(new_params.W)],
        "b": [np.zeros_like(new_params.b), np.zeros_like(new_params.b)],
        "log_std": [np.zeros_like(new_params.log_std), np.zeros_like(new_params.log_std)],
        "v_w": [np.zeros_like(new_params.v_w), np.zeros_like(new_params.v_w)],
        "v_c": [np.zeros(1), np.zeros(1)],
    }
    step_count = 0
    kl_hist, cf_hist, sur_hist, vf_hist = [], [], [], []
    for _ in range(epochs):
        step_count += 1
        mu = policy_mean(new_params, obs)  # (S, 2)
        sigma = np.exp(new_params.log_std)[None, :]
        z = (act - mu) / sigma
        logp = np.sum(-0.5 * z * z - new_params.log_std[None, :] - 0.5 * _LOG_2PI, axis=1)
        ratio = np.exp(logp - logp_old)  # (S,) 重要性权重 (8.9)
        # 截断代理 (8.12)：L = min(r·Â, clip(r, 1±ε)·Â)。
        sur = np.minimum(ratio * adv, np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv)
        # 截断掩码（(8.12) 分段形状：越界且优势同号 → 梯度为零）。
        masked = (
            (adv >= 0.0) & (ratio > 1.0 + clip)
        ) | ((adv < 0.0) & (ratio < 1.0 - clip))
        coef = np.where(masked, 0.0, adv)  # c_t：零或优势
        # ---- 手写反向：∇sur = Σ c_t·r_t·∇logπ（链式：∂r/∂θ = r·∇logπ）。
        w_s = coef * ratio  # (S,) 每样本权重
        grad_mu = w_s[:, None] * z / sigma  # (S, 2) = ∂L/∂μ
        grad_W = grad_mu.T @ obs  # (2, 6)：Σ_t ∂L/∂μ_t · s_tᵀ
        grad_b = grad_mu.sum(axis=0)  # (2,)
        grad_log_std = np.sum(w_s[:, None] * (z * z - 1.0), axis=0)  # (2,)
        # 价值损失 0.5·vf_coef·(V - ret)² 的梯度（线性值函数）。
        phi = value_features(obs)  # (S, 8)
        v_pred = phi @ new_params.v_w + new_params.v_c
        v_res = v_pred - ret
        grad_v_w = vf_coef * (v_res[:, None] * phi).sum(axis=0)  # (8,)
        grad_v_c = vf_coef * float(v_res.sum())
        # ---- Adam 更新：_adam_step 做梯度*下降*（param ← param - lr·adam），
        # 故 actor（最大化代理目标）传负梯度、critic（最小化价值损失）传正梯度。
        new_W, m0, v0 = _adam_step(new_params.W, -grad_W, *state["W"], lr, step_count)
        state["W"] = [m0, v0]
        new_b, m1, v1 = _adam_step(new_params.b, -grad_b, *state["b"], lr, step_count)
        state["b"] = [m1, v1]
        new_ls, m2, v2 = _adam_step(
            new_params.log_std, -grad_log_std, *state["log_std"],
            lr if lr_log_std is None else lr_log_std, step_count,
        )
        state["log_std"] = [m2, v2]
        new_vw, m3, v3 = _adam_step(new_params.v_w, grad_v_w, *state["v_w"], lr, step_count)
        state["v_w"] = [m3, v3]
        new_vc, m4, v4 = _adam_step(
            np.array([new_params.v_c]), grad_v_c, *state["v_c"], lr, step_count
        )
        state["v_c"] = [m4, v4]
        new_params = PolicyParams(new_W, new_b, new_ls, new_vw, float(new_vc[0]))
        # ---- 诊断（教程 08.2 的"刹车"仪表）。
        mu2 = policy_mean(new_params, obs)
        sig2 = np.exp(new_params.log_std)[None, :]
        z2 = (act - mu2) / sig2
        logp2 = np.sum(
            -0.5 * z2 * z2 - new_params.log_std[None, :] - 0.5 * _LOG_2PI, axis=1
        )
        kl_hist.append(float(np.mean(logp_old - logp2)))  # approx_kl 代理
        # clip_frac：更新后比率越出 [1-ε, 1+ε] 的样本占比（越界即失去梯度，
        # (8.12) 的"自动刹车"仪表）。
        ratio2 = np.exp(logp2 - logp_old)
        cf_hist.append(float(np.mean((ratio2 < 1.0 - clip) | (ratio2 > 1.0 + clip))))
        sur_hist.append(float(np.mean(sur)))
        vf_hist.append(float(np.mean(0.5 * v_res * v_res)))
        if target_kl is not None and kl_hist[-1] > target_kl:
            break  # 依据：PPO 论文 §4——KL 超限即停，防止单批数据走太远
    info = {
        "approx_kl": kl_hist,
        "clip_frac": cf_hist,
        "surrogate": sur_hist,
        "value_loss": vf_hist,
    }
    return new_params, info


def train(
    n_iters: int = 120,
    n_episodes: int = 32,
    seed: int = 0,
    env: PointMassReachingEnv | None = None,
    lr: float = 0.05,
    epochs: int = 10,
) -> tuple[PolicyParams, list[float]]:
    """PPO 外层循环（采样 → GAE → 多轮更新，教程 08.2 ② 的两步框架）。

    超参为实测调定（确定性 seed=0 下的依据）：回报均值从 ≈ -39.4 升到
    ≈ -9.5（|回报| 缩到 ~1/4，接近手工 PD 律 -8.9 的量级）；``lr_log_std``
    取 0——本任务中 log_std 的梯度被优势噪声支配、σ 会单调漂移，故把
    探索预算 σ 固定（诊断见 DEBUG.md 变量表 P-4）。

    Args:
        n_iters: 外层迭代数（学习式控制的"训练轮次"；测试预算内取小值）。
        n_episodes: 每轮回合数。
        seed: 总种子（第 k 轮采样用 seed + k，确定性复现）。
        env: 环境；None 用默认到达任务。
        lr / epochs: 透传给 :func:`ppo_update`。

    Returns:
        (final_params, history)：最终权重与逐轮平均回报（未折扣、
        原始尺度——训练健康度的主仪表）。
    """
    env = env if env is not None else PointMassReachingEnv()
    params = PolicyParams.init(seed=seed)
    history: list[float] = []
    for it in range(n_iters):
        episodes, _ = collect_batch(env, params, n_episodes, seed + it)
        history.append(float(np.mean([ep["reward"].sum() for ep in episodes])))
        params, _ = ppo_update(params, episodes, lr=lr, epochs=epochs, lr_log_std=0.0)
    return params, history

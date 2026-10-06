"""projects/control_planning/ppo_lite.py 的测试 —— 直接运行：``python tests/test_ppo_lite.py``。

验证教程第 08 章的微型 PPO 教学实现（Schulman et al., 2017；GAE 2016）：
GAE 与手工 k 步加权递推的一致性 (8.15) 及 λ 端点退化、高斯 log 概率与独立
公式的一致性、策略梯度估计 (8.7) 与中心差分的方向一致性、截断目标 (8.12) 的
"自动刹车"仪表、固定 seed 训练后回报显著上升。全部确定性 seed。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ppo_lite  # noqa: E402
from ppo_lite import (  # noqa: E402
    PolicyParams,
    PointMassReachingEnv,
    collect_batch,
    compute_gae,
    gauss_log_prob,
    policy_gradient_estimate,
    policy_mean,
    ppo_update,
    train,
    value_fn,
)


def test_gae_matches_manual_kstep_weighting():
    """GAE (8.15) 与手工 k 步估计的 (1-λ)λ^{k-1} 加权递推一致；λ 端点退化正确。"""
    rng = np.random.default_rng(4)
    for trial in range(5):
        t_len = int(rng.integers(3, 8))
        rewards = rng.normal(size=t_len)
        values = rng.normal(size=t_len)
        last_value = float(rng.normal())
        gamma, lam = 0.9, 0.7
        adv, ret = compute_gae(rewards, values, last_value, gamma, lam)
        # 手工版：k 步估计 A^(k) = Σ_{l<k} γ^l r + γ^k V(s_{t+k}) - V(s_t)，
        # 按 (8.15) 以 (1-λ)λ^{k-1} 加权求和（k = 1..T-t），末项权重 λ^{T-t-1}。
        for t in range(t_len):
            acc = 0.0
            w_sum = 0.0
            for k in range(1, t_len - t + 1):
                boot = values[t + k] if t + k < t_len else last_value
                a_k = sum(gamma ** l * rewards[t + l] for l in range(k)) \
                    + gamma ** k * boot - values[t]
                w = (1.0 - lam) * lam ** (k - 1)
                if k == t_len - t:
                    w = lam ** (k - 1)  # 截断段末项吸收剩余权重
                acc += w * a_k
                w_sum += w
            assert abs(adv[t] - acc) < 1e-9, (trial, t, adv[t], acc)
            assert abs(w_sum - 1.0) < 1e-9  # 权重归一（λ<1 的等比和）
            assert abs(ret[t] - (adv[t] + values[t])) < 1e-12
    # 端点：λ=0 → 逐 TD 残差 (8.13)；λ=1（γ=1）→ 蒙特卡洛端 G_t - V(s_t)。
    rewards = np.array([1.0, -2.0, 3.0])
    values = np.array([0.5, -0.5, 1.0])
    adv0, _ = compute_gae(rewards, values, 2.0, gamma=1.0, lam=0.0)
    assert np.allclose(adv0, rewards + np.append(values[1:], 2.0) - values)
    adv1, _ = compute_gae(rewards, values, 2.0, gamma=1.0, lam=1.0)
    g_to = np.cumsum(rewards[::-1])[::-1] + 2.0
    assert np.allclose(adv1, g_to - values)


def test_log_prob_matches_manual_formula():
    """高斯 log 概率与独立公式一致；同 seed 采样逐位可复现。"""
    params = PolicyParams.init(seed=1)
    env = PointMassReachingEnv()
    obs = env.reset()
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)
    # 采样可复现（直接重演 rollout_episode 的采样式）。
    mu = policy_mean(params, obs[None, :])[0]
    act_a = mu + np.exp(params.log_std) * rng_a.standard_normal(2)
    ep = ppo_lite.rollout_episode(env, params, rng_b)
    assert np.allclose(ep["act"][0], act_a)
    # 独立公式：log π = Σ_j [-(a-μ)²/(2σ²) - log σ - ½log 2π]。
    a = ep["act"][0] + 0.3
    lp = gauss_log_prob(params, obs[None, :], a[None, :])[0]
    sigma = np.exp(params.log_std)
    manual = float(np.sum(
        -0.5 * ((a - mu) / sigma) ** 2 - params.log_std - 0.5 * np.log(2 * np.pi)
    ))
    assert abs(lp - manual) < 1e-12


def test_policy_gradient_matches_central_difference():
    """策略梯度估计 (8.7) 与 J(θ) 的中心差分在随机方向上一致。

    实测口径（记入阈值的依据）：用训练收敛的策略（seed=0）缩放 0.5 倍——
    保证 |a| 远离截断 u_limit（截断使 J(θ) 出现拐点，差分失效，见 DEBUG.md
    §4 案例 4）——以 8000 条共享随机数的轨迹做公共随机数差分，两个随机
    方向上 实际/预期 ∈ {0.68, 0.84}（比值含采样噪声，8k 条下 ~15%）。
    断言取 [0.5, 1.5] 且方向符号一致——足以钉住"归一化按回合数"这类
    整数量级的实现错误（历史 bug 恰差 T = n_steps 倍）。
    """
    trained, _ = train(seed=0)
    env = PointMassReachingEnv(n_steps=10)
    params = PolicyParams(
        trained.W * 0.5, trained.b * 0.5, trained.log_std, trained.v_w, trained.v_c
    )
    n_ep = 8000
    batch, _ = collect_batch(env, params, n_ep, seed=777)
    grad = policy_gradient_estimate(params, batch, gamma=1.0)

    def flatten(p: PolicyParams) -> np.ndarray:
        return np.concatenate([p.W.ravel(), p.b, p.log_std])

    def unflatten(v: np.ndarray) -> PolicyParams:
        return PolicyParams(
            v[:12].reshape(2, 6), v[12:14], v[14:16], np.zeros(8), 0.0
        )

    def j_of(p: PolicyParams) -> float:
        eps_, _ = collect_batch(env, p, n_ep, seed=777)  # 公共随机数
        return float(np.mean([e["reward"].sum() for e in eps_]))

    g_flat = flatten(grad)
    rng = np.random.default_rng(5)
    h = 1e-5
    for k in range(2):
        d = rng.normal(size=16)
        d /= np.linalg.norm(d)
        j_plus = j_of(unflatten(flatten(params) + h * d))
        j_minus = j_of(unflatten(flatten(params) - h * d))
        fd = (j_plus - j_minus) / (2.0 * h)
        analytic = float(g_flat @ d)
        assert np.sign(analytic) == np.sign(fd), (k, analytic, fd)
        assert 0.5 < analytic / fd < 1.5, (k, analytic, fd)
        print(f"\n[pg dir {k}] fd {fd:.5f} vs analytic {analytic:.5f} "
              f"(ratio {analytic / fd:.3f})")


def test_training_improves_return():
    """固定 seed 训练后平均回报显著上升（阈值以实测定，依据见注释）。"""
    _, hist = train(seed=0)
    # 实测依据（seed=0，train 默认 120 轮 × 32 回合）：回报从 -39.75 升到
    # -9.53（幅度比 4.2x，接近手工 PD 律的 -8.9 量级）；seed 1/2 为 3.4x/3.0x。
    # 断言阈值 1.5x（回报为负，用幅度比 hist[0]/hist[-1]）留 2 倍裕量。
    ratio = hist[0] / hist[-1]  # 两者皆为负 → 比值 > 1 即改善
    assert ratio > 1.5, (hist[0], hist[-1])
    # 确定性策略（取均值动作）确实到达目标附近。
    env = PointMassReachingEnv()
    params, _ = train(seed=0)
    obs = env.reset()
    for _ in range(env.n_steps):
        obs, _, _ = env.step(obs, policy_mean(params, obs[None, :])[0])
    err = float(np.linalg.norm(obs[:2] - env.goal))
    assert err < 0.2, err
    print(f"\n[train] return {hist[0]:.2f} -> {hist[-1]:.2f} ({ratio:.1f}x), "
          f"greedy final err {err:.3f} m")


def test_ppo_update_deterministic_and_brakes():
    """同输入逐位可复现；target_kl 早停刹车；大步长下截断仪表（8.12）启动。"""
    env = PointMassReachingEnv()
    params = PolicyParams.init(seed=0)
    episodes, _ = collect_batch(env, params, 16, seed=0)
    out1, info1 = ppo_update(params, episodes, epochs=20, lr_log_std=0.0, target_kl=0.05)
    out2, info2 = ppo_update(params, episodes, epochs=20, lr_log_std=0.0, target_kl=0.05)
    assert np.array_equal(out1.W, out2.W) and np.array_equal(out1.b, out2.b)
    assert np.array_equal(out1.log_std, out2.log_std)
    # 刹车：KL 超限后剩余 epoch 不再执行（实测首轮 kl=0.072 > 0.05 即停，1 < 20；
    # 记录的末轮 kl 正是触发早停的那一轮，故 > target_kl 是"已刹车"的证据）。
    assert len(info1["approx_kl"]) < 20
    assert info1["approx_kl"][-1] > 0.05
    # 大步长：更新后比率越出 [1-ε, 1+ε] 的样本占比过半（实测 0.93）。
    _, info_big = ppo_update(params, episodes, epochs=1, lr=0.3, lr_log_std=0.0, target_kl=None)
    assert info_big["clip_frac"][-1] > 0.5
    print(f"\n[ppo] epochs executed {len(info1['approx_kl'])} (target_kl=0.05), "
          f"clip_frac @lr=0.3 {info_big['clip_frac'][-1]:.2f}")


def test_env_shapes_reward_and_clamp():
    """环境契约：观测/回报形状与量纲、动作截断、reset 确定性。"""
    env = PointMassReachingEnv()
    obs = env.reset()
    assert obs.shape == (6,) and np.allclose(obs, [0, 0, 0, 0, env.goal[0], env.goal[1]])
    obs2, r, done = env.step(obs, np.array([100.0, -100.0]))  # 超限动作被截断
    assert obs2.shape == (6,) and not done
    # 截断后的有效动作 = u_limit：速度增量恰为 u_limit·dt。
    assert np.allclose(obs2[2:4], env.u_limit * env.dt * np.array([1.0, -1.0]))
    # 回报上界 = 0（目标处零控制）；量纲 = -w_pos‖e‖² - w_u‖u‖²。
    e = env.goal - obs2[:2]
    expect = -env.w_pos * float(e @ e) - env.w_u * 2.0 * env.u_limit**2
    assert abs(r - expect) < 1e-12
    # 相对误差通道 e = g - p 与绝对位置自洽。
    assert np.allclose(obs2[4:6], env.goal - obs2[:2])
    assert np.allclose(env.reset(), obs)  # reset 确定性


if __name__ == "__main__":
    # 单点测试：python tests/test_X.py <测试名子串>；不带参数 = 全部测试。
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and pat in k]
    if not fns:
        print(f"没有匹配 '{pat}' 的测试；可用测试：")
        for k in sorted(globals()):
            if k.startswith("test_"):
                print(" ", k)
        sys.exit(1)
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
    sys.exit(1 if failed else 0)

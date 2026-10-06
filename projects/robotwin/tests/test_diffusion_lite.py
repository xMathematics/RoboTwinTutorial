"""projects/robotwin/diffusion_lite 的测试 —— 直接运行：``python tests/test_diffusion_lite.py``。

验证扩散策略教学代理（DDPM 2D 轨迹去噪）的行为主张（对应精读
DiffusionPolicy §4 的公式与教程第 08 章 §8.2）：
- β 调度与 $\\bar\\alpha$ 的性质（调度是采样先验的依据）；
- 前向加噪闭式 (D0′) 的边缘分布统计（均值/方差与解析值一致）；
- 演示数据集的形状/条件一致性/确定性（复用回合接口的采集）；
- 手写 MLP 反传与有限差分交叉验证（梯度链正确性）；
- 训练损失单调下降（固定 seed）且限时 < 8 s（小网络 + 小轮数实测）；
- 采样轨迹端点贴合条件 (起点, 终点)、平滑性优于纯噪声基线、同 seed 确定复现。

门限均为 seed=0 实测值再留 ≥1.7 倍裕量（详见各行注释；训练耗时随机器
浮动，门限 8 s 对实测 ~4.7 s（系统 python3）/ ~1.2 s（llm_env，BLAS 多线程）
留余量；两解释器的损失曲线逐位一致）。
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diffusion_lite as dl  # noqa: E402


def test_beta_schedule_and_alpha_bar_properties():
    """线性 β 调度性质：β ∈ (0,1) 递增；ᾱ 单调递减、ᾱ_1≈1、ᾱ_T≈0
    （ᾱ_T ≈ 0.0012 实测——采样先验 N(0,I) ≈ q(x_T|x_0) 的依据）。"""
    betas, abar = dl.BETAS, dl.ALPHAS_BAR
    assert betas.shape == (dl.T_STEPS,) and abar.shape == (dl.T_STEPS,)
    assert np.all(betas > 0.0) and np.all(betas < 1.0), "β 必须是合法方差系数"
    assert np.all(np.diff(betas) > 0.0), "线性调度应严格递增"
    assert np.all(np.diff(abar) < 0.0), "ᾱ 应单调递减（噪声越来越大）"
    assert 0.99 < abar[0] < 1.0, f"ᾱ_1 应接近 1（实测 {abar[0]:.4f}）"
    assert abar[-1] < 0.01, f"ᾱ_T 应 ≈ 0（实测 {abar[-1]:.5f}）→ 先验 ≈ 纯噪声"
    # 后验方差 β̃：非负、β̃_0 = 0（最后一步去噪确定）、和 ≤ Σβ。
    assert dl.BETA_TILDE[0] == 0.0
    assert np.all(dl.BETA_TILDE >= 0.0)
    print(f"\n[beta] β∈[{betas[0]}, {betas[-1]}], ᾱ_1={abar[0]:.4f}, ᾱ_T={abar[-1]:.5f}")


def test_q_sample_closed_form_marginal():
    """前向加噪闭式 (D0′)：q(x^t|x^0) 的均值 ≈ √ᾱ·x0、方差 ≈ 1−ᾱ
    （大样本频率学检验，Ho et al. 2020 边缘闭式的实现正确性）。"""
    rng = np.random.default_rng(11)
    x0 = np.abs(rng.normal(0.0, 0.5, size=(4, dl.TRAJ_DIM)))  # 任意固定数据
    n_draw = 6000
    for k in (1, 10, 35, 49):
        abar = dl.ALPHAS_BAR[k]
        means = np.zeros((4, dl.TRAJ_DIM))
        second = np.zeros_like(means)
        for _ in range(n_draw):
            eps = rng.standard_normal((4, dl.TRAJ_DIM))
            x_t = dl.q_sample(x0, np.full(4, k), eps)
            means += x_t
            second += x_t * x_t
        means /= n_draw
        var = second / n_draw - means * means
        # 均值门限 = 5 个标准误（√((1−ᾱ)/n_draw)，下限 0.02 防零除）；
        # 方差门限 = 样本方差的 CLT 波动 5σ（√(2/n_draw)·(1−ᾱ)，Gaussian 惯例）。
        tol_mean = max(5.0 * np.sqrt((1.0 - abar) / n_draw), 0.02)
        tol_var = 5.0 * np.sqrt(2.0 / n_draw) * (1.0 - abar) + 1e-3
        assert np.max(np.abs(means - np.sqrt(abar) * x0)) < tol_mean, (k, "mean")
        assert np.max(np.abs(var - (1.0 - abar))) < tol_var, (k, "var")
    print("\n[q_sample] t=1/10/35/49 的均值与方差均闭合到解析值 (D0′)")


def test_dataset_shapes_endpoints_and_determinism():
    """演示数据集：形状 (48,4)/(48,16)；条件 = 首末路径点（逐位相等，
    因为二者由同一条轨迹算出）；同 seed 两次采集逐位一致。"""
    conds, trajs = dl.collect_demonstrations()
    assert conds.shape == (48, dl.COND_DIM) and trajs.shape == (48, dl.TRAJ_DIM)
    for i in range(conds.shape[0]):
        traj = trajs[i].reshape(dl.K_WAYPOINTS, 2)
        assert np.array_equal(trajs[i][:2], conds[i][:2]), (i, "start")
        assert np.array_equal(trajs[i][-2:], conds[i][2:]), (i, "end")
        # 弧长重采样应消灭重复路径点（回到"均匀铺开"的设计意图）。
        min_gap = float(np.min(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
        assert min_gap > 1e-6, (i, min_gap)
    conds2, trajs2 = dl.collect_demonstrations()
    assert np.array_equal(conds, conds2) and np.array_equal(trajs, trajs2)
    # 归一化往返一致（to_workspace 与 from_workspace 互逆）。
    back = dl.from_workspace(trajs[0].reshape(dl.K_WAYPOINTS, 2))
    assert np.allclose(dl.to_workspace(back), trajs[0].reshape(dl.K_WAYPOINTS, 2), atol=1e-15)
    print(f"\n[dataset] N={conds.shape[0]}（reach 双臂 32 + push 16），条件=首末点，确定复现")


def test_noise_mlp_gradients_match_finite_difference():
    """手写反传 vs 中心差分（与 arm.py 的解析/数值雅可比互证同款思路）：
    对 eps_prediction_loss 的参数梯度做逐元素交叉验证。"""
    rng = np.random.default_rng(21)
    model = dl.NoiseMLP(dl.COND_DIM + dl.TIME_EMB_DIM + dl.TRAJ_DIM, hidden=8,
                        out_dim=dl.TRAJ_DIM, rng=rng,
                        skip_from=dl.COND_DIM + dl.TIME_EMB_DIM, skip_dim=dl.TRAJ_DIM)
    b = 5
    cond = rng.normal(0.0, 0.5, size=(b, dl.COND_DIM))
    t_idx = rng.integers(0, dl.T_STEPS, size=b)
    eps = rng.standard_normal((b, dl.TRAJ_DIM))
    x_t = dl.q_sample(rng.normal(0.0, 0.5, size=(b, dl.TRAJ_DIM)), t_idx, eps)

    # 用一个小包装：把参数写入模型后重新前向算损失。
    def loss_with(params: dict) -> float:
        backup = {k: v.copy() for k, v in model.params.items()}
        for k, v in params.items():
            model.params[k][...] = v
        a0 = np.concatenate([cond, dl.time_embedding(t_idx), x_t], axis=1)
        pred = model.forward(a0)
        diff = pred - eps
        out = float(np.mean(diff * diff))
        for k, v in backup.items():
            model.params[k][...] = v
        return out

    _loss, grads = dl.eps_prediction_loss(model, cond, x_t, t_idx, eps)
    eps_fd = 1e-6
    checked = 0
    for name in ("W3", "b3", "U", "b2"):
        arr = model.params[name]
        flat = arr.reshape(-1)
        gflat = grads[name].reshape(-1)
        for j in rng.choice(flat.size, size=min(60, flat.size), replace=False):
            orig = flat[j]
            flat[j] = orig + eps_fd
            lp = loss_with(model.params)
            flat[j] = orig - eps_fd
            lm = loss_with(model.params)
            flat[j] = orig
            fd = (lp - lm) / (2.0 * eps_fd)
            assert abs(fd - gflat[j]) < 1e-6 + 1e-5 * abs(fd), (name, j, fd, gflat[j])
            checked += 1
    print(f"\n[grad] 手写反传与中心差分一致（{checked} 个参数分量，容差 1e-6）")


_TRAINED_CACHE: tuple = None  # (model, losses, conds, trajs)——多个测试共享一次训练


def _trained_model():
    """惰性训练一次默认配置（3500 轮全批 ≈ 5 s），供采样类测试共享。"""
    global _TRAINED_CACHE
    if _TRAINED_CACHE is None:
        conds, trajs = dl.collect_demonstrations()
        model, losses = dl.ddpm_train(conds, trajs, seed=0)
        _TRAINED_CACHE = (model, losses, conds, trajs)
    return _TRAINED_CACHE


def test_training_loss_monotone_decrease():
    """训练损失下降（固定 seed，几十~几千轮小数据）+ 限时 < 8 s。

    独立训练一次以测量耗时（共享缓存会使本测试测到 0）；实测系统
    python3 ~4.7 s / llm_env ~1.2 s（BLAS 多线程），门限 8 s 留裕量；
    两解释器损失曲线逐位一致（3.170 → 0.0222，实测）。"""
    conds, trajs = dl.collect_demonstrations()
    t0 = time.time()
    model, losses = dl.ddpm_train(conds, trajs, seed=0)
    elapsed = time.time() - t0
    assert len(losses) == 3500
    assert elapsed < 8.0, f"训练超时：{elapsed:.2f} s（预算 8 s）"
    n = len(losses)
    first, mid, last = (float(np.mean(losses[:n // 3])),
                        float(np.mean(losses[n // 3:2 * n // 3])),
                        float(np.mean(losses[2 * n // 3:])))
    assert last < mid < first, (first, mid, last)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
    print(f"\n[train] {elapsed:.2f} s，损失 {losses[0]:.3f} → {losses[-1]:.4f} "
          f"（三段均值 {first:.3f} → {mid:.3f} → {last:.3f}，单调下降）")


def test_sampled_endpoints_match_condition():
    """采样轨迹端点贴合条件：对数据集全部 48 个条件各采一条，首末点与
    (起点, 终点) 的最大距离 < 0.08 m（实测 max 0.044 m，留 ~1.8 倍裕量；
    数据里端点由条件决定，故这是"模型学会条件化"的直接检验）。"""
    model, _losses, conds, _trajs = _trained_model()
    rng = np.random.default_rng(123)
    worst = 0.0
    for i in range(conds.shape[0]):
        start = dl.from_workspace(conds[i][:2].reshape(1, 2))[0]
        end = dl.from_workspace(conds[i][2:].reshape(1, 2))[0]
        traj = dl.plan(model, start, end, rng)  # (8, 2)，单位 m
        err = max(float(np.linalg.norm(traj[0] - start)),
                  float(np.linalg.norm(traj[-1] - end)))
        worst = max(worst, err)
        assert err < 0.08, (i, err)
    print(f"\n[endpoint] 48 条采样轨迹首末点最大偏差 = {worst:.4f} m（门限 0.08）")


def test_sampled_trajectory_smoothness_beats_noise():
    """平滑性（二阶差分范数）优于纯噪声基线：基线 = 先验尺度的 i.i.d.
    高斯路径点（即反向链的起点 N(0, I) 映回米制，smoothness ≈ √6·0.4·√6
    ≈ 3.9 m 实测）；采样轨迹应显著更平滑（实测比值 ≤ 0.14，门限 0.5）。"""
    model, _losses, conds, _trajs = _trained_model()
    rng = np.random.default_rng(321)
    prior_noise = dl.from_workspace(rng.standard_normal((dl.K_WAYPOINTS, 2)))
    baseline = dl.smoothness(prior_noise)
    worst = 0.0
    for i in range(0, conds.shape[0], 4):  # 每 4 个条件抽 1 条（12 条）
        start = dl.from_workspace(conds[i][:2].reshape(1, 2))[0]
        end = dl.from_workspace(conds[i][2:].reshape(1, 2))[0]
        traj = dl.plan(model, start, end, rng)
        ratio = dl.smoothness(traj) / baseline
        worst = max(worst, ratio)
        assert ratio < 0.5, (i, ratio)
    print(f"\n[smooth] 采样/纯噪声 平滑度比最大 = {worst:.3f}（基线 {baseline:.2f} m，门限 0.5）")


def test_sampling_determinism_same_seed():
    """采样确定性：同 seed 同条件逐位一致；不同 seed 输出不同
    （多峰采样的随机性来自初值与逐步注噪，精读 §4.4）。"""
    model, _losses, conds, _trajs = _trained_model()
    cond = conds[0]
    start = dl.from_workspace(cond[:2].reshape(1, 2))[0]
    end = dl.from_workspace(cond[2:].reshape(1, 2))[0]
    r1 = np.random.default_rng(7)
    r2 = np.random.default_rng(7)
    r3 = np.random.default_rng(8)  # 异 seed：检验采样随机性仍在
    t_a = dl.plan(model, start, end, r1)
    t_b = dl.plan(model, start, end, r2)
    t_c = dl.plan(model, start, end, r3)
    assert np.array_equal(t_a, t_b), "同 seed 必须逐位一致"
    assert not np.array_equal(t_a, t_c), "不同 seed 应给出不同样本（随机性仍在）"
    print("\n[determinism] 同 seed 逐位一致；异 seed 输出不同")


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

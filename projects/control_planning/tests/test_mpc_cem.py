"""projects/control_planning/mpc_cem.py 的测试 —— 直接运行：``python tests/test_mpc_cem.py``。

验证教程第 05 章的 CEM-MPC 教学实现（PETS 的已知动力学简化档，Chua et al.,
NeurIPS 2018）：候选代价对"朝目标序列"的偏好（(5.1) 的代价结构）、CEM 精英
重分布的收敛性 (5.13)、滚动时域到达 (5.2) 与执行噪声下的全程无碰撞、
温启动接口。全部确定性 seed（CEM 采样与执行噪声共用一条随机流）。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util as _ilu

# 本项目 metrics.py 按路径加载（唯一模块键）——仓库内 4 个项目各有 metrics.py
# （CONSTRAINTS §4.7），合并收集（仓库根 pytest / VS Code Testing）时
# sys.modules["metrics"] 会被先导入者占据，故不能裸 `import metrics`。
_mpath = Path(__file__).resolve().parents[1] / "metrics.py"
_mspec = _ilu.spec_from_file_location(f"_metrics_{_mpath.parent.name}", _mpath)
metrics = _ilu.module_from_spec(_mspec)
_mspec.loader.exec_module(metrics)
from ilqr import double_integrator_step  # noqa: E402
from mpc_cem import (  # noqa: E402
    candidate_cost,
    cem_mpc_rollout,
    cem_plan,
    make_reach_scene,
)

SCENE = make_reach_scene()
X0 = np.array([0.5, 0.5, 0.0, 0.0])
DT = 0.15


def test_candidate_cost_prefers_goal_directed_sequence():
    """候选评分：同样的控制功效下，朝目标的序列代价必须低于背向的序列。"""
    goal = SCENE["goal"]
    horizon = 12
    toward = np.zeros((1, horizon, 2))
    toward[0, :, 0] = 1.0  # 沿 +x 加速（目标在 (9,6)，起点近原点）
    away = np.zeros((1, horizon, 2))
    away[0, :, 0] = -1.0  # 背向目标
    costs = candidate_cost(
        np.vstack([toward, away]), X0, goal, SCENE["obstacles"], DT
    )
    assert costs[0] < costs[1]
    # 障碍惩罚只在穿透安全余量时激活：远离障碍的候选不受罚。
    free_cost = candidate_cost(toward, X0, goal, [], DT)[0]
    assert free_cost > 0.0


def test_cem_elite_refinement_converges():
    """CEM 主循环 (5.13)：精英均值代价逐轮改善、搜索分布方差收缩。"""
    plan = cem_plan(X0, SCENE["goal"], SCENE["obstacles"], DT, seed=0)
    assert plan.elite_cost_mean[-1] < plan.elite_cost_mean[0]
    assert plan.sigma_trace[-1] < plan.sigma_trace[0]  # 方差迹收缩（健康：单调不增趋势）
    assert plan.u_seq.shape == (14, 2)  # 默认 horizon = 14
    assert np.all(np.abs(plan.u_seq) <= 6.0 + 1e-9)  # 动作箱 u ∈ U（(5.1)）
    print(f"\n[cem] elite cost {plan.elite_cost_mean[0]:.1f} -> {plan.elite_cost_mean[-1]:.1f}, "
          f"sigma trace {plan.sigma_trace[0]:.3f} -> {plan.sigma_trace[-1]:.3f}")


def test_rolling_horizon_reaches_goal():
    """滚动时域 (5.2)：执行噪声下终端误差 < 到达容差 0.35 m。"""
    res = cem_mpc_rollout(X0, SCENE["goal"], SCENE["obstacles"], seed=3)
    assert res.final_pos_err < 0.35
    assert res.traj.shape == (81, 4) and res.u_applied.shape == (80, 2)
    assert len(res.plans) == 80
    print(f"\n[mpc] final err {res.final_pos_err:.3f} m")


def test_rolling_horizon_stays_collision_free_with_noise():
    """执行噪声（σ = 0.05）下全程最小净距 > 0（无碰撞；实测健康值 ~0.58 m）。"""
    for seed in (3, 7):
        res = cem_mpc_rollout(X0, SCENE["goal"], SCENE["obstacles"], seed=seed)
        assert res.min_dist > 0.0, (seed, res.min_dist)
        print(f"\n[mpc seed {seed}] min net distance {res.min_dist:.3f} m")


def test_warm_start_and_mean_init_interface():
    """温启动接口（教程 05.1 ③）：mean_init 决定初值形状并被 1 轮 CEM 消化。"""
    warm = np.full((14, 2), 0.5)
    plan = cem_plan(
        X0, SCENE["goal"], SCENE["obstacles"], DT,
        n_iters=1, mean_init=warm, seed=0,
    )
    assert plan.mean.shape == (14, 2)
    assert np.all(np.isfinite(plan.u_seq))
    assert np.isfinite(plan.best_cost)
    # 初值周围的单轮精化不会离谱：均值仍在动作箱内。
    assert np.all(np.abs(plan.mean) <= 6.0 + 1e-9)


def test_success_rate_over_seeds():
    """metrics.success_rate 口径（M.1）：多 seed 滚动执行至少 3/4 成功（实测 4/4）。"""
    trajs = []
    for seed in (3, 7, 11, 21):
        res = cem_mpc_rollout(X0, SCENE["goal"], SCENE["obstacles"], seed=seed)
        trajs.append(res.traj)
        assert res.final_pos_err < 0.35 and res.min_dist > 0.0
    sr = metrics.success_rate(trajs, SCENE["goal"], SCENE["obstacles"], tol=0.35)
    assert sr >= 0.75, sr
    print(f"\n[mpc] success_rate over 4 seeds = {sr:.2f}")


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

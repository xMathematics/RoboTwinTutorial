"""projects/robotwin/benchmark 的测试 —— 直接运行：``python tests/test_benchmark.py``。

验证评测协议（教程 §5.6）：同配置两次 run 逐位一致、回合数与 seen/unseen
对半分配正确、种子派生规则与 DR 参数流确定可复算、calibrated 策略在
none 档的泛化差距恰为 0（标定世界无 seen/unseen 之分）。
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark as bm  # noqa: E402
import dr  # noqa: E402
import tasks as tk  # noqa: E402
from policies import CalibratedPolicy  # noqa: E402

EPISODES = 12  # 每格回合数（seen 6 + unseen 6）——测试用小网格


def test_run_deterministic_bitwise():
    """同配置两次 run → 记录逐条逐字段一致（协议确定性的总检验）。"""
    run_a = bm.run(CalibratedPolicy(), episodes_per_task=EPISODES, seed0=0)
    run_b = bm.run(CalibratedPolicy(), episodes_per_task=EPISODES, seed0=0)
    assert len(run_a) == len(run_b)
    for ra, rb in zip(run_a, run_b):
        assert ra == rb  # dataclass 的字段级相等


def test_episode_count_and_split():
    """回合总数 = 档位 × 任务 × 每格回合数；每格 seen/unseen 对半
    （奇数时 unseen 多 1）；遍历顺序 = (regime, task, split, episode)。"""
    records = bm.run(CalibratedPolicy(), episodes_per_task=EPISODES, seed0=1)
    n_regimes, n_tasks = len(dr.REGIMES), len(tk.TASKS)
    assert len(records) == n_regimes * n_tasks * EPISODES
    per_cell = {}
    for rec in records:
        per_cell.setdefault((rec.regime, rec.task, rec.split), []).append(rec)
    assert len(per_cell) == n_regimes * n_tasks * 2
    for (regime, task, split), recs in per_cell.items():
        assert len(recs) == EPISODES // 2, (regime, task, split)
        assert all(r.split == split and r.regime == regime and r.task == task
                   for r in recs)


def test_seed_derivation_rule_reproducible():
    """种子派生规则可复算：记录里的 seed 与文档写死的公式逐条一致；
    DR 参数流（独立 Generator）同样确定。"""
    seed0, episodes = 7, 2
    records = bm.run(CalibratedPolicy(), episodes_per_task=episodes, seed0=seed0)
    idx = 0
    for regime_idx, regime in enumerate(dr.REGIMES):
        for task_idx, task in enumerate(tk.TASKS):
            for split_idx, split in enumerate(("seen", "unseen")):
                for episode in range(episodes // 2):
                    rec = records[idx]
                    idx += 1
                    expect = (7919 * episode
                              + 104729 * (task_idx + 1)
                              + 1299709 * (regime_idx + 1)
                              + 15485863 * (split_idx + 1)
                              + seed0)
                    assert rec.seed == expect, (regime, task, split, episode)
                    # DR 参数流：同公式重建 rng 必须采出同一组参数。
                    rng_a = bm._params_rng(seed0, regime_idx, task_idx, split_idx, episode)
                    rng_b = bm._params_rng(seed0, regime_idx, task_idx, split_idx, episode)
                    pa = dr.sample(regime, rng_a)
                    pb = dr.sample(regime, rng_b)
                    assert pa == pb
                    # 世代独立：不同 episode 的参数流不同（strong 档必异）。
                    if regime == "strong":
                        other = dr.sample(regime, bm._params_rng(
                            seed0, regime_idx, task_idx, split_idx, episode + 1))
                        assert pa != other


def test_generalization_gap_zero_for_calibrated_at_none():
    """calibrated 在 none 档（标定世界）seen/unseen 成功率都应为满 ——
    泛化差距恰为 0：模型即世界时,"没见过的目标位置"不存在。"""
    records = bm.run(
        CalibratedPolicy(), tasks=("reach",), regimes=("none",),
        episodes_per_task=8, seed0=0,
    )
    seen = [r.success for r in records if r.split == "seen"]
    unseen = [r.success for r in records if r.split == "unseen"]
    assert all(seen) and all(unseen), ([r.seed for r in records if not r.success],)
    gap = bm.mt.generalization_gap(
        [bm.mt.success_rate(seen)], [bm.mt.success_rate(unseen)]
    )
    assert abs(gap) < 1e-12


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

"""评测协议（benchmark）——迷你基准的"跑分器"，对应教程第 05 章 §5.6。

函数流水线
----------
``run(policy, ...)`` 是协议的唯一入口：按"档位 × 任务 × 分层 × 回合"四层
循环，每回合 ① 经 ``dr.sample`` 抽世界参数（DR 施加）→ ② ``tasks.make_task``
建环境 → ③ ``reset(seed, split)`` 采场景 → ④ 驱动策略至 ``done`` → ⑤ 记录
``info["success"]``。聚合交给 ``metrics.py``（成功率/宏平均+最差/泛化差距），
打印交给 ``demo.py``。依赖方向：``demo / tests → 本模块 → dr, tasks,
policies, metrics``。

确定性（seed 派生规则，写死于 :func:`episode_seed` 与 :func:`_params_rng`）
--------------------------------------------------------------------------
- **环境/场景种子**：``episode_seed = 7919·episode + 104729·(task_idx+1)
  + 1299709·(regime_idx+1) + 15485863·(split_idx+1) + seed0``（素数混散，
  保证四层索引互不碰撞）。
- **DR 参数流**：``np.random.default_rng([seed0, regime_idx, task_idx,
  split_idx, episode])``（与场景流独立，互不消耗对方的随机数）。
- 同一 ``(seed0, 配置)`` 两次 ``run`` 的逐回合结果**逐位一致**（有测试守恒）。

泛化分层（seen / unseen，教程 §5.6 ④ / §6.5 的配置轴）
------------------------------------------------------
``split="seen"`` 的目标位姿采在标定工作空间内部（``tasks.SCENE_RANGES``
的 seen 区），``split="unseen"`` 采在外围一圈（unseen 区）。两组都跑，
``metrics.generalization_gap`` 给出差距。每任务 ``episodes_per_task`` 个
回合按 **seen 一半 + unseen 一半** 分配（奇数时 unseen 多 1 个）。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import importlib.util

import numpy as np

import dr
import tasks as tk

# 本项目 metrics.py 按路径加载（唯一模块键 _metrics_robotwin）——
# 仓库内 4 个项目各有 metrics.py（CONSTRAINTS §4.7），合并收集
# （仓库根 pytest / VS Code Testing）时 sys.modules["metrics"] 会被
# 先导入者占据，裸 `import metrics` 会拿到别人的模块。
_mspec = importlib.util.spec_from_file_location(
    "_metrics_robotwin", Path(__file__).resolve().parent / "metrics.py")
mt = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(mt)

__all__ = ["EpisodeRecord", "run", "episode_seed", "aggregate", "SUMMARY_KEYS"]


@dataclass(frozen=True)
class EpisodeRecord:
    """一个回合的评测记录（协议的最小数据单元）。

    Attributes:
        task: 任务名（``"reach" / "push" / "pick_place"``）。
        regime: DR 档位名（``"none" / "mild" / "strong"``）。
        split: 泛化分层（``"seen" / "unseen"``）。
        episode: 该 (regime, task, split) 格内的回合序号（0 起）。
        seed: 环境种子（:func:`episode_seed` 的派生值，可单独复跑）。
        success: 该回合是否成功（评测协议唯一的结果变量）。
    """

    task: str
    regime: str
    split: str
    episode: int
    seed: int
    success: bool


def episode_seed(
    seed0: int,
    regime_idx: int,
    task_idx: int,
    split_idx: int,
    episode: int,
) -> int:
    """回合环境种子的确定性派生规则（素数混散，规则写死即协议的一部分）。

    Args:
        seed0: 协议级基础种子。
        regime_idx / task_idx / split_idx: 档位 / 任务 / 分层在**本次调用**
            的遍历序列（传入的 ``regimes`` / ``tasks`` / 固定
            ``("seen", "unseen")``）中的下标——默认调用下恰为
            ``dr.REGIMES`` / ``tk.TASKS`` 的下标；子集调用时按子集内
            重新编号（种子随编号改变，但仍完全确定）。
        episode: 格内回合序号。

    Returns:
        int 种子，非负；不同索引组合不碰撞（素数间隔远大于格内回合数）。
    """
    return (
        7919 * episode
        + 104729 * (task_idx + 1)
        + 1299709 * (regime_idx + 1)
        + 15485863 * (split_idx + 1)
        + seed0
    )


def _params_rng(
    seed0: int,
    regime_idx: int,
    task_idx: int,
    split_idx: int,
    episode: int,
) -> np.random.Generator:
    """DR 参数流的独立确定性发生器（与场景种子流互不干扰）。"""
    return np.random.default_rng(
        [seed0, regime_idx, task_idx, split_idx, episode]
    )


def run(
    policy,
    tasks: Sequence[str] = tk.TASKS,
    regimes: Sequence[str] = dr.REGIMES,
    episodes_per_task: int = 40,
    seed0: int = 0,
    max_steps: int = tk.MAX_STEPS,
    verbose: bool = False,
) -> list[EpisodeRecord]:
    """执行评测协议：驱动策略跑完 (regime × task × split × episode) 全网格。

    对应教程 §5.6 的评测流程（"每任务 N 次 rollout、报告成功率"）+ Easy/
    Hard 两档（此处细化为 none/mild/strong 三档剂量）。策略接口：``reset(task)``
    + ``act(obs, step) -> action``（见 ``policies.ScriptedPolicy``）。

    Args:
        policy: 被评测策略实例（每回合前调用 ``policy.reset(task)``）。
        tasks: 任务名序列，默认全部三种。
        regimes: DR 档位名序列，默认全部三档。
        episodes_per_task: 每（任务, 档位）的回合总数，seen/unseen 对半分。
        seed0: 协议级基础种子。
        max_steps: 回合超时步数（协议常数，默认 :data:`tasks.MAX_STEPS`）。
        verbose: True 时打印逐回合进度（调试用）。

    Returns:
        list[:class:`EpisodeRecord`]：按 (regime, task, split, episode)
        遍历顺序排列的全部回合记录（顺序本身也是确定性的一部分）。
    """
    splits = ("seen", "unseen")
    records: list[EpisodeRecord] = []
    for regime_idx, regime in enumerate(regimes):
        for task_idx, task in enumerate(tasks):
            n_seen = episodes_per_task // 2
            counts = {"seen": n_seen, "unseen": episodes_per_task - n_seen}
            for split_idx, split in enumerate(splits):
                for episode in range(counts[split]):
                    seed = episode_seed(
                        seed0, regime_idx, task_idx, split_idx, episode
                    )
                    # ① DR 施加：每回合独立采一组世界参数（参数流与场景流独立）。
                    params = dr.sample(
                        regime, _params_rng(seed0, regime_idx, task_idx, split_idx, episode)
                    )
                    # ② 建环境 + ③ 场景采样。
                    env = tk.make_task(task, params, max_steps=max_steps)
                    obs = env.reset(seed=seed, split=split)
                    policy.reset(task)
                    # ④ 回合循环：策略驱动至成功或超时。
                    done = False
                    step = 0
                    info: Mapping = {"success": False}
                    while not done:
                        action = policy.act(obs, step)
                        obs, _reward, done, info = env.step(action)
                        step += 1
                    if verbose:
                        print(
                            f"  [{regime}/{task}/{split}#{episode}] seed={seed}"
                            f" success={info['success']}"
                        )
                    # ⑤ 记录（协议只关心成功率这一个结果变量）。
                    records.append(
                        EpisodeRecord(
                            task=task,
                            regime=regime,
                            split=split,
                            episode=episode,
                            seed=seed,
                            success=bool(info["success"]),
                        )
                    )
    return records


#: :func:`aggregate` 输出 dict 的键（seen/unseen/all 三种口径）。
SUMMARY_KEYS: tuple[str, ...] = ("seen", "unseen", "all")


def aggregate(records: Sequence[EpisodeRecord]) -> dict:
    """把逐回合记录聚合成嵌套 dict：``[regime][task][key] -> 成功率``。

    纯聚合不做任何指标数学——公式全部在 :mod:`metrics`（``success_rate``）。

    Args:
        records: :func:`run` 的返回值。

    Returns:
        ``dict[regime][task][key] -> float``：key ∈ ``{"seen", "unseen",
        "all"}``，分别为 seen/unseen 分层成功率与两层合并成功率。
    """
    bucket: dict[str, dict[str, dict[str, list[float]]]] = {}
    for rec in records:
        cell = bucket.setdefault(rec.regime, {}).setdefault(
            rec.task, {"seen": [], "unseen": [], "all": []}
        )
        value = 1.0 if rec.success else 0.0
        cell[rec.split].append(value)
        cell["all"].append(value)
    return {
        regime: {
            task: {key: mt.success_rate(values) for key, values in cell.items()}
            for task, cell in cells.items()
        }
        for regime, cells in bucket.items()
    }

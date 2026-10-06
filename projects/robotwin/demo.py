"""RoboTwin 迷你基准冒烟演示——本项目的核心入口脚本（VS Code 调试配置
「RoboTwin: 核心入口」即指向本文件）。

函数流水线（本脚本在系统中的位置）：
    dr.sample                    （DR 采样：none/mild/strong 三档剂量 × 5 个随机化维）
        → tasks.make_task        （3 任务回合环境：reach / push / pick_place）
        → benchmark.run          （评测协议：每格 40 回合，seen/unseen 对半，确定性种子）
        → policies.CalibratedPolicy / RobustPolicy （脚本对照策略：名义模型 vs 每回合自标定）
        → metrics.success_rate / macro_mean_and_worst / generalization_gap （指标聚合）

输入/输出：无文件 I/O。输入为内存仿真；输出为终端打印的
"策略 × DR 档"成功率表（每格 = 宏平均（最差任务）），及一句话结论。
运行：`python demo.py`（任意 numpy>=1.26 环境；确定性种子，输出可复现，
秒级完成）。

它复现教程的哪条曲线：教程 §4.2.3 / §6.4 讲"DR 剂量 vs 成功率"——只见过
标定/干净条件的策略，随随机化剂量加大而失败率上升；对条件变化稳健的
策略保持。本演示用 calibrated（名义模型）vs robust（每回合自标定）两行
数字把这条论断在一分钟内跑出来。这不是 RoboTwin 本体的复现（那需要
CoppeliaSim/MLLM/GPU），而是其"DR + 评测协议"方法论的 NumPy 教学化。
"""

import benchmark
import metrics as mt
import tasks as tk
from dr import REGIMES
from policies import CalibratedPolicy, RobustPolicy

#: 每格回合数（seen 20 + unseen 20）——教学默认值，秒级完成。
EPISODES_PER_TASK: int = 40
#: 参加演示的策略（名字 → 工厂；工厂保证两个策略互不共享状态）。
POLICIES: dict[str, type] = {"calibrated": CalibratedPolicy, "robust": RobustPolicy}


def main() -> None:
    """跑 3 任务 × 3 DR 档 × 40 回合 × 2 策略，打印成功率表与结论。

    健康值（确定性输出，llm_env/系统 python3 双环境一致）：
    calibrated 列 none→mild→strong 单调下降（strong 明显塌）；robust 三档
    基本持平（≥0.9）。异常信号：robust 的 strong 档也大幅塌陷 → 观测噪声
    或测量流程坏了（排查见 DEBUG.md §3 R 组变量表）。
    """
    title = (
        f"RoboTwin 迷你基准：{len(tk.TASKS)} 任务 × {len(REGIMES)} DR 档 × "
        f"{EPISODES_PER_TASK} 回合/格 × {len(POLICIES)} 策略（纯 NumPy 教学实现）"
    )
    print("=" * 78)
    print(title)
    print("每格数字 = 宏平均（最差任务）成功率；GAP = seen − unseen 泛化差距")
    print("=" * 78)

    for policy_name, policy_cls in POLICIES.items():
        records = benchmark.run(
            policy_cls(),
            tasks=tk.TASKS,
            regimes=REGIMES,
            episodes_per_task=EPISODES_PER_TASK,
            seed0=0,
        )
        table = benchmark.aggregate(records)
        print(f"\n[策略 {policy_name}]  逐任务成功率（行=任务，列=DR 档）")
        # 标签列宽按显示宽度 12 对齐（1 个 CJK 字符 ≈ 2 个等宽列，手工补空格）。
        header = "  " + "任务" + " " * 8 + "".join(
            f" {regime:>13s}" for regime in REGIMES
        )
        print(header)
        for task in tk.TASKS:
            row = "  {:<12s}".format(task)
            for regime in REGIMES:
                cell = table[regime][task]["all"]
                row += f" {cell:>13.2f}"
            print(row)
        # 协议口径：宏平均 + 最差任务 + 泛化差距（metrics.py 三个纯函数）。
        macro_row = "  " + "宏平均" + " " * 6
        worst_row = "  " + "最差任务" + " " * 4
        gap_row = "  " + "GAP" + " " * 9
        for regime in REGIMES:
            per_task = {t: table[regime][t]["all"] for t in tk.TASKS}
            macro, worst, worst_name = mt.macro_mean_and_worst(per_task)
            gap = mt.generalization_gap(
                [table[regime][t]["seen"] for t in tk.TASKS],
                [table[regime][t]["unseen"] for t in tk.TASKS],
            )
            macro_row += f" {macro:>13.2f}"
            worst_row += f" {worst:>6.2f}({worst_name[:4] if worst_name else '—'})"
            gap_row += f" {gap:>+13.2f}"
        print(macro_row)
        print(worst_row)
        print(gap_row)

    print("\n" + "=" * 78)
    print(
        "结论：DR 剂量 ↑ → calibrated（信任名义模型）成功率单调下降，"
        "robust（每回合自标定）基本持平——"
        "复现教程 §4.2.3/§6.4 的“DR 剂量 vs 成功率”论断。"
    )


if __name__ == "__main__":
    main()

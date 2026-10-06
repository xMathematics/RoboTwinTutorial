"""控制与规划冒烟演示——本项目的核心入口脚本（VS Code 调试配置「control_planning: 核心入口」即指向本文件）。

函数流水线（本脚本在系统中的位置）：
    rrt.plan_rrt / plan_rrt_star （采样式规划：走廊障碍图上的 RRT 与 RRT*，
                                  教程 03 章 (3.3)–(3.7)）
        → ilqr.ilqr_solve （同一走廊的轨迹平滑：反向递推 + 前向回滚，教程 04 章 (4.5)–(4.14)）
        → mpc_cem.cem_mpc_rollout （滚动时域 + CEM：测量→重解→执行首步，教程 05 章 (5.1)/(5.2)/(5.13)）
        → cbf.simulate_filtered （安全滤波对照：无滤波撞入 / 有滤波 h ≥ -tol，教程 07 章 (7.9)）
        → metrics.success_rate / path_length / trajectory_cost （统一指标，见 METRICS.md）

输入/输出：无文件 I/O。输入为内存中的合成场景（确定性 seed）；输出为
终端打印的对齐表格（各段健康值见 DEBUG.md §3 变量表）。
运行：`python demo.py`（任意 numpy>=1.26 环境；确定性输出可复现）。
"""

import numpy as np

import metrics as mtr
from cbf import simulate_filtered
from ilqr import ReachCost, ilqr_solve
from mpc_cem import cem_mpc_rollout, make_reach_scene
from rrt import plan_rrt, plan_rrt_star

#: 基准场景常数（与各测试共享的口径）：10x10 场地、中部带缺口圆墙。
BOUNDS = (0.0, 10.0, 0.0, 10.0)
START = np.array([0.5, 0.5])
GOAL = np.array([9.0, 6.0])
GAP = np.array([5.0, 6.0])  # 墙上缺口中心（y=6 附近无圆）——走廊路标点
N_SEEDS = 5  # ① 中成功率统计的规划次数
#: 双积分器初态（静止在 START）：demo 四段共用。
START_X0 = np.array([START[0], START[1], 0.0, 0.0])


def _section_rrt(obstacles: list) -> None:
    """① 采样式规划：同 seed 下 RRT vs RRT* 的成功率 / 路径长度 / 代价。"""
    paths_rrt: list[np.ndarray] = []
    paths_star: list[np.ndarray] = []
    costs_rrt: list[float] = []
    costs_star: list[float] = []
    for seed in range(N_SEEDS):
        res = plan_rrt(START, GOAL, obstacles, BOUNDS, seed=seed, max_iters=6000)
        res_s = plan_rrt_star(START, GOAL, obstacles, BOUNDS, seed=seed, max_iters=6000)
        paths_rrt.append(res.path if res.path is not None else np.full((2, 2), np.nan))
        paths_star.append(
            res_s.path if res_s.path is not None else np.full((2, 2), np.nan)
        )
        costs_rrt.append(res.cost if res.success else float("nan"))
        costs_star.append(res_s.cost if res_s.success else float("nan"))
        if seed == 0:
            len0 = mtr.path_length(res_s.path) if res_s.success else float("nan")
    sr_rrt = mtr.success_rate(paths_rrt, GOAL, obstacles, tol=0.35)
    sr_star = mtr.success_rate(paths_star, GOAL, obstacles, tol=0.35)
    print("【① 采样式规划】走廊障碍图（5 seed，教程 03 章）")
    print(f"  成功率 success_rate : RRT {sr_rrt:.2f}   RRT* {sr_star:.2f}   （指标 M.1）")
    print(f"  路径长度 path_len   : RRT* seed0 {len0:.3f} m          （指标 M.2）")
    for s in range(N_SEEDS):
        print(
            f"    seed {s}: RRT cost {costs_rrt[s]:7.3f} | RRT* cost {costs_star[s]:7.3f}"
            f"   （代价 (3.5)；RRT* ≤ RRT 见 test_rrt）"
        )


def _section_ilqr(obstacles: list) -> None:
    """② 轨迹优化：iLQR 以缺口为路标点，把直线初值平滑成穿越走廊的轨迹。"""
    dt = 0.1
    n = 60
    cost = ReachCost(
        w_u=0.05,
        goal=GOAL,
        w_goal=200.0,
        w_vf=5.0,
        waypoint=(n // 2, GAP),
        w_wp=200.0,
    )
    u_init = np.zeros((n, 2))
    res = ilqr_solve(START_X0, u_init, cost, dt=dt, max_iters=80)
    j0, j1 = res.costs[0], res.costs[-1]
    cost_init = mtr.trajectory_cost(res.x_traj[: n + 1], u_init, dt=dt)
    cost_opt = mtr.trajectory_cost(res.x_traj[: n + 1], res.u_traj, dt=dt)
    print("【② iLQR 平滑】同一走廊：直线初值 → 经缺口 (5,6) 到目标（教程 04 章）")
    print(
        f"  总代价 J: {j0:9.3f} → {j1:9.3f}   迭代 {res.n_iters} 轮，"
        f"收敛 = {res.converged}   （单调下降见 test_ilqr）"
    )
    print(
        f"  指标 trajectory_cost: 初值(零控制) {cost_init:8.3f} → 优化 {cost_opt:8.3f}   （指标 M.3）"
    )
    print(
        f"    （零控制初值不产生位移故功效为 0；优化轨迹付出控制功效"
        f"换取总代价 J 下降 {j0 / j1:.0f} 倍）"
    )
    print(
        f"  终端位置误差: {np.linalg.norm(res.x_traj[-1, :2] - GOAL):.4f} m"
        f"   终端速度: {np.linalg.norm(res.x_traj[-1, 2:]):.4f} m/s"
    )


def _section_mpc(scene: dict) -> None:
    """③ 滚动时域控制：CEM-MPC 在执行噪声下滚动到达并保持净距。"""
    res = cem_mpc_rollout(
        START_X0, scene["goal"], scene["obstacles"], n_steps=80, dt=0.15, seed=3
    )
    sr = mtr.success_rate([res.traj], scene["goal"], scene["obstacles"], tol=0.35)
    print("【③ CEM-MPC】滚动时域到达（执行噪声 σ=0.05，教程 05 章）")
    print(
        f"  终端误差 {res.final_pos_err:.3f} m（< 0.35 到达）   全程最小净距 "
        f"{res.min_dist:.3f} m（> 0 无碰撞）"
    )
    print(
        f"  success_rate = {sr:.2f}   （指标 M.1；滚动重解吸收噪声 = (5.2) 的反馈机制）"
    )


def _section_cbf() -> None:
    """④ 安全滤波：同一直冲指令，无滤波撞入 / 有滤波保持 h ≥ -tol。"""
    obstacle = [(np.array([2.4, 0.0]), 0.5)]
    x0 = np.array([0.0, 0.0, 1.5, 0.0])
    u_des = np.array([1.2, 0.0])
    raw = simulate_filtered(x0, u_des, obstacle, dt=1e-3, n_steps=6000, filtered=False)
    safe = simulate_filtered(x0, u_des, obstacle, dt=1e-3, n_steps=6000, filtered=True)
    print("【④ CBF 安全滤波】直冲障碍的指令 u_des=[1.2, 0]（教程 07 章）")
    print(
        f"  无滤波: 最小净距 {raw.min_dist:7.3f} m（< 0 = 撞入）   最小 h "
        f"{raw.min_h:7.3f} m"
    )
    print(
        f"  有滤波: 最小净距 {safe.min_dist:7.3f} m（≥ 0 = 不穿透）   最小 h "
        f"{safe.min_h:.2e} m   投影残余 {float(np.max(safe.viol_resid)):.1e}"
    )


def main() -> None:
    """跑通四段冒烟演示，作为调试断点入口（断点建议见 DEBUG.md §2）。"""
    scene = make_reach_scene()
    obstacles = scene["obstacles"]
    print("=" * 68)
    print("control_planning 冒烟演示（确定性 seed，秒级；全部输出可复现）")
    print("=" * 68)
    _section_rrt(obstacles)
    print("-" * 68)
    _section_ilqr(obstacles)
    print("-" * 68)
    _section_mpc(scene)
    print("-" * 68)
    _section_cbf()
    print("=" * 68)
    print("指标口径：success_rate (M.1) / path_length (M.2) / trajectory_cost (M.3)")
    print("健康值与异常信号见 DEBUG.md §3；指标定义见 METRICS.md。")


if __name__ == "__main__":
    main()

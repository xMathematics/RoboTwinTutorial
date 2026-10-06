"""projects/control_planning/rrt.py 的测试 —— 直接运行：``python tests/test_rrt.py``。

验证教程第 03 章的采样式规划教学实现（LaValle TR 98-11；Karaman & Frazzoli,
IJRR 2011）：steer 的限步长语义 (3.3)、圆域碰撞检测的相交判定、空图直线
可达性、goal bias 的首解加速 (3.11)、RRT* 选父/重布线后代价不劣于同 seed
RRT（(3.6) 的单调性）与树的一致性（代价递归 (3.5) + 全树无碰撞边）。
全部场景确定性 seed，答案可精确复现。
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
from rrt import (  # noqa: E402
    gamma_constant,
    make_corridor_obstacles,
    neighbor_radius,
    plan_rrt,
    plan_rrt_star,
    segment_free,
    steer,
)

START = np.array([0.5, 0.5])
GOAL = np.array([9.0, 6.0])
BOUNDS = (0.0, 10.0, 0.0, 10.0)


def test_steer_limits_step_and_direction():
    """steer (3.3)：远点走满 ε、近点直达、方向沿单位向量。"""
    near = np.array([1.0, 1.0])
    far = steer(near, np.array([5.0, 1.0]), eta=0.5)
    assert np.allclose(far, [1.5, 1.0])  # 远点：走满 ε，方向不变
    close = steer(near, np.array([1.1, 1.0]), eta=0.5)
    assert np.allclose(close, [1.1, 1.0])  # 近点：一步直达（min(ε, dist)）
    diag = steer(near, np.array([2.0, 2.0]), eta=np.sqrt(2.0))
    assert np.allclose(diag, [2.0, 2.0])
    assert abs(np.linalg.norm(steer(near, np.array([9.0, 9.0]), eta=0.5) - near) - 0.5) < 1e-12


def test_segment_free_and_neighbor_radius_constants():
    """碰撞检测的圆相交判定；邻域半径 (3.7) 的收缩与上截断；γ 满足 Theorem 38 下界。"""
    circle = [(np.array([0.5, 0.0]), 0.4)]
    assert not segment_free(np.array([-1.0, 0.0]), np.array([2.0, 0.0]), circle)  # 穿心
    assert segment_free(np.array([-1.0, 0.9]), np.array([2.0, 0.9]), circle)      # 从上方绕过
    assert segment_free(np.array([-1.0, 0.0]), np.array([0.05, 0.0]), circle)     # 止步于圆外
    # 半径随 n 收缩、被 η 截断（(3.7)：min{γ(log n/n)^{1/d}, η}）。
    # 本场景 γ ≈ 17.9：n ≤ ~8000 时半径触及 η = 0.5 的上截断，故取大 n 对照。
    gamma = gamma_constant(BOUNDS, make_corridor_obstacles())
    r_capped = neighbor_radius(100, gamma, eta=0.5)
    r_small = neighbor_radius(20000, gamma, eta=0.5)
    assert neighbor_radius(1, gamma, eta=0.5) == 0.5  # n < 2：无邻域可言，取 η
    assert r_capped == 0.5 and r_small < 0.5
    # Theorem 38 下界：γ > (2(1+1/d))^{1/d} (μ(C_free)/ζ_d)^{1/d}（教程 (3.7) 行内注）。
    mu_free = 100.0 - 5.0 * np.pi  # bounds 面积扣除 5 个 r=1 圆
    lb = (2.0 * (1.0 + 0.5)) ** 0.5 * (mu_free / np.pi) ** 0.5
    assert gamma > lb


def test_empty_map_plans_straight_reachable_path():
    """空图：无障碍时必可达，路径连续、每段 ≤ η，总长有直线距离下界。"""
    res = plan_rrt(START, GOAL, [], BOUNDS, seed=0, max_iters=4000, goal_bias=0.3)
    assert res.success
    assert res.path is not None and res.path.shape[1] == 2
    assert np.allclose(res.path[0], START)
    # 终点落入目标邻域（半径 0.3）。
    assert np.linalg.norm(res.path[-1] - GOAL) <= 0.3 + 1e-9
    seg = np.linalg.norm(np.diff(res.path, axis=0), axis=1)
    assert float(seg.max()) <= 0.5 + 1e-9  # steer 限步长（(3.3)）
    straight = float(np.linalg.norm(GOAL - START))
    assert metrics.path_length(res.path) >= straight - 1e-9  # 三角不等式
    assert metrics.path_length(res.path) < 2.0 * straight  # 空图不绕远
    # 代价 (3.5) 与路径长度一致（欧氏边代价口径）。
    assert abs(res.cost - metrics.path_length(res.path)) < 1e-9


def test_goal_bias_shortens_first_solution():
    """goal bias (3.11) 加速首解：p_g = 0.3 的首解轮数应显著少于纯均匀采样。

    实测依据（seed=0，空图，max_iters=8000）：bias 0.0 → 170 轮、
    0.3 → 49 轮、0.5 → 33 轮——单调改善，断言取跨档比较（非边界值）。
    """
    iters = {}
    for bias in (0.0, 0.3):
        res = plan_rrt(START, GOAL, [], BOUNDS, seed=0, max_iters=8000, goal_bias=bias)
        assert res.success
        iters[bias] = res.n_iters
    assert iters[0.3] < iters[0.0] / 2.0, iters


def test_corridor_map_success_within_limited_iters():
    """走廊障碍图（带缺口圆墙）：RRT 与 RRT* 都在限迭代内成功且无碰撞。"""
    obstacles = make_corridor_obstacles()
    for seed in range(3):
        res = plan_rrt(START, GOAL, obstacles, BOUNDS, seed=seed, max_iters=3000)
        assert res.success, seed
        for p in res.path:  # 路径点全部避开障碍圆（点机器人口径）
            for c, r in obstacles:
                assert np.linalg.norm(p - c) >= r - 1e-9
        res_s = plan_rrt_star(START, GOAL, obstacles, BOUNDS, seed=seed, max_iters=3000)
        assert res_s.success, seed
        assert res_s.cost < float(np.linalg.norm(GOAL - START)) + 5.0  # 不明显绕远


def test_rrt_star_cost_not_worse_than_rrt_same_seed():
    """RRT* 的重布线只改善不恶化（(3.6)）：同 seed 下代价 ≤ RRT 的代价。

    两模式每轮消耗同一随机数序列（一次 goal-bias 判定 + 一次均匀采样），
    生成相同的节点序列，故代价可直接逐 seed 对照。
    """
    obstacles = make_corridor_obstacles()
    for seed in range(3):
        res = plan_rrt(START, GOAL, obstacles, BOUNDS, seed=seed, max_iters=3000)
        res_s = plan_rrt_star(START, GOAL, obstacles, BOUNDS, seed=seed, max_iters=3000)
        assert res.success and res_s.success
        # RRT 在首个目标节点即返回（提前 break），两棵树共享前缀节点序列。
        assert np.allclose(res.nodes, res_s.nodes[: len(res.nodes)])
        # 依据：(3.6) 的条件更新——同一次插入中 RRT* 的选父含最近邻候选，
        # 其代价必不高于 RRT 的固定最近邻接法，此后重布线只减不增。
        assert res_s.cost <= res.cost + 1e-9, (seed, res.cost, res_s.cost)
        # 实测参考（3000 迭代）：RRT ≈ 12.5–14.0，RRT* ≈ 10.4–11.0。
        print(f"\n[rrt* seed {seed}] RRT cost {res.cost:.3f} vs RRT* cost {res_s.cost:.3f}")


def test_rrt_star_tree_consistency_after_rewire():
    """重布线后的树自洽：每条树边无碰撞、代价满足 (3.5) 递归、路径长 = 终点代价。"""
    obstacles = make_corridor_obstacles()
    res = plan_rrt_star(START, GOAL, obstacles, BOUNDS, seed=1, max_iters=1500)
    assert res.success
    nodes, parent, costs = res.nodes, res.parent, res.node_costs
    for i in range(1, len(nodes)):
        j = int(parent[i])
        assert 0 <= j < len(nodes) and j != i  # 父存在且唯一（重布线后父可后于子）
        assert segment_free(nodes[j], nodes[i], obstacles)  # 全树无碰撞边（重布线后）
        # 代价递归 (3.5)：c(v) = c(parent) + ‖q_v - q_parent‖。
        assert abs(costs[i] - (costs[j] + np.linalg.norm(nodes[i] - nodes[j]))) < 1e-9
    assert costs[0] == 0.0
    # 路径提取：相邻行必为父子关系，折线长 = 目标节点代价。
    for k in range(len(res.path) - 1):
        tail = np.nonzero(
            (np.linalg.norm(nodes - res.path[k + 1], axis=1) < 1e-12)
        )[0]
        head = np.nonzero(
            (np.linalg.norm(nodes - res.path[k], axis=1) < 1e-12)
        )[0]
        assert int(parent[int(tail[0])]) == int(head[0])
    assert abs(metrics.path_length(res.path) - res.cost) < 1e-9


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

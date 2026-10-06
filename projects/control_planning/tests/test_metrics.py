"""projects/control_planning/metrics.py 的测试 —— 直接运行：``python tests/test_metrics.py``。

验证统一测评模块的 3 个指标（success_rate / path_length / trajectory_cost，
CONSTRAINTS §4.7 为 control_planning 点名的基线三指标）。每个指标至少一个
测试与**朴素实现**（显式循环）或手工构造用例交叉验证；全部场景确定性，
无随机通过/失败。指标教程见 ../METRICS.md。
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

path_length = metrics.path_length
success_rate = metrics.success_rate
trajectory_cost = metrics.trajectory_cost

OBST = (np.array([0.0, 0.0]), 0.5)  # 原点处半径 0.5 的圆障碍
GOAL = np.array([3.0, 0.0])
TOL = 0.1


def _naive_path_length(traj: np.ndarray) -> float:
    """朴素路径长度：显式循环逐段累加，用于交叉验证向量化实现。"""
    total = 0.0
    for i in range(len(traj) - 1):
        total += float(np.linalg.norm(traj[i + 1] - traj[i]))
    return total


def _naive_trajectory_cost(traj: np.ndarray, u: np.ndarray, dt: float, w: float) -> float:
    """朴素轨迹代价：逐点循环累加功效与平滑项，交叉验证 (M.3)。"""
    effort = 0.0
    for k in range(len(u)):
        effort += float(np.sum(u[k] * u[k])) * dt
    smooth = 0.0
    for k in range(len(u) - 1):
        d = u[k + 1] - u[k]
        smooth += float(np.sum(d * d)) * dt
    return effort + w * smooth


def test_success_rate_manual_hit_collide_miss():
    """手工构造三类轨迹（命中 / 碰撞 / 未达各 1）→ 成功率恰为 1/3（M.1）。"""
    t = np.linspace(0.0, 1.0, 50)[:, None]
    hit = (1.0 - t) * np.array([0.9, 0.9]) + t * GOAL        # 斜线直达目标（避开障碍）
    collide = np.hstack([t * GOAL[0], 0.5 - t])              # 穿过原点障碍（y: 0.5→-0.5）
    miss = np.hstack([t * GOAL[0], 1.0 + 0.0 * t])           # 终点 (3,1)：距目标 1.0 > tol
    sr = success_rate([hit, collide, miss], GOAL, [OBST], TOL)
    assert sr == 1.0 / 3.0, sr
    print(f"\n[rate] hit/collide/miss -> {sr:.4f}")


def test_success_rate_requires_both_conditions():
    """到达但碰撞、或无碰撞但未达，都必须判负；两条同时满足才判正。"""
    t = np.linspace(0.0, 1.0, 50)[:, None]
    hit = (1.0 - t) * np.array([0.9, 0.9]) + t * GOAL
    collide = np.hstack([t * GOAL[0], 0.5 - t])
    miss = np.hstack([t * GOAL[0], 1.0 + 0.0 * t])
    assert success_rate([collide], GOAL, [OBST], TOL) == 0.0  # 到达但穿墙
    assert success_rate([miss], GOAL, [], TOL) == 0.0         # 无障碍但未达
    assert success_rate([hit], GOAL, [OBST], TOL) == 1.0
    assert success_rate([], GOAL, [OBST], TOL) == 0.0         # 空输入 → 0
    # (N, 4) 双积分器状态自动取位置分量。
    state_traj = np.hstack([hit, np.zeros((50, 2))])
    assert success_rate([state_traj], GOAL, [OBST], TOL) == 1.0
    # 相切不碰撞（边界距离恰为半径）。
    tangent = np.column_stack([t[:, 0], 0.5 + 0.0 * t[:, 0]])  # y ≡ 0.5 = 半径
    assert success_rate([tangent], GOAL, [OBST], TOL) == 0.0   # 但终点未达
    tangent_hit = np.vstack([tangent, GOAL[None, :]])
    assert success_rate([tangent_hit], GOAL, [OBST], TOL) == 1.0


def test_path_length_matches_naive_loop():
    """path_length (M.2) 与朴素逐段累加一致（随机轨迹，1e-12）。"""
    rng = np.random.default_rng(7)
    for _ in range(5):
        traj = np.cumsum(rng.normal(scale=0.3, size=(30, 2)), axis=0)
        assert abs(path_length(traj) - _naive_path_length(traj)) < 1e-12
    state_traj = np.cumsum(rng.normal(scale=0.2, size=(20, 4)), axis=0)
    assert abs(path_length(state_traj) - _naive_path_length(state_traj[:, :2])) < 1e-12


def test_path_length_known_values():
    """解析已知值：单位间距直线 = N-1；折返路径 = 2 倍；静止 = 0。"""
    line = np.column_stack([np.arange(11.0), np.zeros(11)])
    assert abs(path_length(line) - 10.0) < 1e-12
    fold = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    assert abs(path_length(fold) - 4.0) < 1e-12
    still = np.zeros((5, 2))
    assert path_length(still) == 0.0


def test_trajectory_cost_matches_naive_loop():
    """trajectory_cost (M.3) 与朴素逐点累加一致（随机控制序列，1e-12）。"""
    rng = np.random.default_rng(11)
    traj = np.cumsum(rng.normal(scale=0.1, size=(25, 2)), axis=0)
    u = rng.normal(scale=0.5, size=(24, 2))
    dt, w = 0.1, 0.3
    got = trajectory_cost(traj, u, dt=dt, w_smooth=w)
    want = _naive_trajectory_cost(traj, u, dt, w)
    assert abs(got - want) < 1e-12, (got, want)
    print(f"\n[cost] {got:.6f} vs naive {want:.6f}")


def test_trajectory_cost_smoothness_weight():
    """平滑项的语义：常值控制平滑项为 0；交替控制的 ‖Δu‖² 被显式计价。"""
    traj = np.arange(10.0)[:, None] * np.array([1.0, 0.0])
    u_const = np.ones((9, 2))
    # 常值控制：Δu = 0 → 代价 = 纯功效 Σ‖u‖² dt = 9 × 2 × 0.05。
    assert abs(trajectory_cost(traj, u_const, dt=0.05, w_smooth=0.7) - 0.9) < 1e-12
    # 交替控制（±1）：纯功效 18；8 次跳变 Δu = (±2, ±2)，‖Δu‖² = 8 → 平滑 64。
    u_alt = np.ones((9, 2)) * (2.0 * (np.arange(9) % 2)[:, None] - 1.0)
    c0 = trajectory_cost(traj, u_alt, dt=1.0, w_smooth=0.0)
    c1 = trajectory_cost(traj, u_alt, dt=1.0, w_smooth=1.0)
    assert abs(c0 - 18.0) < 1e-12
    assert abs(c1 - (18.0 + 64.0)) < 1e-12


def test_input_validation_and_shapes():
    """非法输入显式报错（早失败优于静默）；(N, 4) 输入取位置分量。"""
    traj = np.arange(20.0).reshape(10, 2)
    u = np.ones((9, 2))
    for bad_call, exc in (
        (lambda: trajectory_cost(traj, u[:-1]), ValueError),   # 控制长度不符
        (lambda: trajectory_cost(traj, u, dt=0.0), ValueError),  # dt 非正
        (lambda: path_length(np.ones((1, 2))), ValueError),    # 单点轨迹
        (lambda: path_length(np.ones((3, 2))[:1]), ValueError),
    ):
        try:
            bad_call()
            raised = False
        except exc:
            raised = True
        assert raised, bad_call
    # 正常路径：成功率为 0（终点 (5,5) 距目标 (0,0) 远超 tol）。
    assert success_rate([np.ones((5, 2))], np.zeros(2), [], tol=0.1) == 0.0
    # (N, 4) 状态轨迹：位置分量参与判定，与 (N, 2) 一致。
    state = np.hstack([traj, np.zeros((10, 2))])
    assert abs(path_length(state) - path_length(traj)) < 1e-12


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

"""projects/control_planning/mpnet_lite.py 的测试 —— 直接运行：``python tests/test_mpnet.py``。

验证教程第 09 章 §09.1 的 MPNet 教学代理（Qureshi et al., ICRA 2019；
蒸馏损失 (9.2)、采样偏置 (9.1)）：粗 SDF 编码的符号与截断口径、专家数据
管线的等距重采样、手写反向与中心差分的一致性、固定 seed 训练后蒸馏损失
显著下降、网络建议点的自由空间比率显著高于均匀采样基线、带偏置 RRT 在
固定环境集上的平均求解迭代数少于均匀 RRT（同一段代码 p_bias=0 作对照臂）、
全程逐位确定性。全部场景确定性 seed，答案可精确复现。
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
from rrt import plan_rrt  # noqa: E402
from mpnet_lite import (  # noqa: E402
    MLPParams,
    build_dataset,
    make_inputs,
    make_random_env,
    mlp_forward,
    mse_loss,
    mse_loss_grad,
    plan_rrt_biased,
    points_free,
    propose_points,
    resample_path_by_arclength,
    sdf_grid,
    train_mpnet,
)

BOUNDS = (0.0, 10.0, 0.0, 10.0)
# 固定测试环境集：与训练集（build_dataset 的随机环境流）不同 seed——顺带
# 检验"泛化到未见环境"这一 MPNet 的核心主张（精读 §5 的 unseen-X_obs 口径）。
TEST_ENVS = [make_random_env(seed=s) for s in (101, 102, 103, 104)]

# 训练成本高（~4 s），进程内缓存一份供多个测试共用（pytest 与直跑同一进程，
# 训练恰好只发生一次；确定性训练保证缓存与即时训练逐位一致）。
_TRAINED_CACHE: dict = {}


def _trained() -> tuple[MLPParams, list[float]]:
    """返回（缓存的）seed=0 默认配置训练结果：偏置网络 + 损失曲线。"""
    if "net" not in _TRAINED_CACHE:
        _TRAINED_CACHE["net"] = train_mpnet(seed=0)
    return _TRAINED_CACHE["net"]


def test_sdf_grid_encoding_basics():
    """粗 SDF 编码：形状/值域、符号口径（格心在障碍内为负）、远场截断、确定性。"""
    enc = sdf_grid([], BOUNDS, k=4)
    assert enc.shape == (16,) and np.all(enc == 1.0)  # 空障碍：处处"远场自由"
    # 单圆障碍放在格心 (1.25, 1.25)（k=4 → 格心间距 2.5）：该格净距 = -r < 0，
    # 对角邻格净距 = 2.5√2 - r > 0，远角净距 > 归一化尺度 → 截断到 +1。
    obst = [(np.array([1.25, 1.25]), 1.0)]
    enc1 = sdf_grid(obst, BOUNDS, k=4)
    scale = 0.25 * float(np.hypot(10.0, 10.0))
    assert enc1[0] < 0.0  # 障碍内：符号为负（SDF 口径）
    diag = float(np.hypot(3.75 - 1.25, 3.75 - 1.25))
    assert abs(enc1[5] - (diag - 1.0) / scale) < 1e-12  # 对角邻格：净距线性保留
    assert enc1[-1] == 1.0  # 远角：净距超过 0.25×对角线 → 截断饱和
    assert np.all(enc1 >= -1.0) and np.all(enc1 <= 1.0)
    assert np.array_equal(enc1, sdf_grid(obst, BOUNDS, k=4))  # 确定性
    # 逐点自由判定与 rrt 同口径（点到圆心距离 < r 判碰、相切不碰）。
    pts = np.array([[1.25, 1.25], [2.25, 1.25], [2.26, 1.25]])
    free = points_free(pts, obst)
    assert free.tolist() == [False, True, True]
    try:
        sdf_grid([], BOUNDS, k=1)
        raise AssertionError("k < 2 应抛 ValueError")
    except ValueError:
        pass


def test_resample_and_dataset_pipeline():
    """弧长重采样的等距口径；(9.2) 数据管线的形状与目标模长分布。"""
    line = np.array([[0.0, 0.0], [3.0, 0.0]])
    rs = resample_path_by_arclength(line, 1.0)
    assert np.allclose(rs, [[0, 0], [1, 0], [2, 0], [3, 0]])
    # 折线：拐点处的弧长参数化（0.75 处仍在第一段）。
    elbow = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    rs2 = resample_path_by_arclength(elbow, 0.75)
    assert np.allclose(rs2[1], [0.75, 0.0]) and np.allclose(rs2[2], [1.0, 0.5])
    assert np.allclose(rs2[-1], [1.0, 1.0])  # 末点强制对齐 path[-1]
    # 数据管线：小规模端到端（3 个随机环境）。
    x, y, info = build_dataset(seed=5, n_envs=3)
    assert x.shape[0] == y.shape[0] == info["n_samples"] and info["n_solved"] >= 1
    assert x.shape[1] == 4 + 16 * 16  # 布局 = (当前点/s, 目标点/s, 16×16 SDF)
    norms = np.linalg.norm(y, axis=1)
    # 目标偏移 = 相邻重采样点的弦长：不超过 stride（弦 ≤ 弧，三角不等式），
    # 直线段占主导（实测均值 ≈ 0.95·stride）——(9.2) 的回归目标量纲有界，
    # 网络只需学方向（推理侧偏移再被 clamp 截断）。
    assert float(norms.max()) <= 1.1 + 1e-9
    assert float(norms.mean()) >= 0.8, norms.mean()
    # 输入布局装配：位置分量按对角线归一化，编码段原样拼接。
    enc = sdf_grid([], BOUNDS, k=16)  # 空障碍编码（全 1，(256,)）
    row = make_inputs(np.array([[2.0, 4.0]]), np.array([8.0, 6.0]), enc, BOUNDS)
    s = float(np.hypot(10.0, 10.0))
    assert row.shape == (1, 4 + 256)
    assert np.allclose(row[0, :2], [2.0 / s, 4.0 / s])
    assert np.allclose(row[0, 2:4], [8.0 / s, 6.0 / s])
    assert np.allclose(row[0, 4:], enc)


def test_mlp_gradient_matches_central_difference():
    """手写反向与损失曲面的中心差分逐坐标一致（tanh MLP 全参数布局）。"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 7))
    y = rng.normal(size=(16, 2))
    params = MLPParams.init(d_in=7, hidden=8, seed=1)
    loss, grad = mse_loss_grad(params, x, y)
    assert abs(loss - mse_loss(params, x, y)) < 1e-12
    # 抽查各参数数组的若干坐标（覆盖 W1/b1/W2/b2 四个布局段）。
    coords = (
        [(params.W1, grad.W1, 3), (params.b1, grad.b1, 2),
         (params.W2, grad.W2, 3), (params.b2, grad.b2, 2)]
    )
    h = 1e-5
    for arr, g_arr, n_pick in coords:
        flat_idx = rng.choice(arr.size, size=n_pick, replace=False)
        for fi in flat_idx:
            i = np.unravel_index(int(fi), arr.shape)
            plus = _perturb(params, arr, i, +h)
            minus = _perturb(params, arr, i, -h)
            fd = (mse_loss(plus, x, y) - mse_loss(minus, x, y)) / (2 * h)
            an = float(g_arr[i])
            # 依据：tanh/线性复合光滑，中心差分误差 O(h²)+舍入 O(1e-11)/2h。
            assert abs(fd - an) < 1e-6 * max(1.0, abs(an)), (i, fd, an)


def _perturb(params: MLPParams, arr: np.ndarray, idx: tuple, delta: float) -> MLPParams:
    """返回把 params 的某一坐标平移 delta 后的副本（中心差分辅助）。"""
    W1, b1, W2, b2 = params.W1, params.b1, params.W2, params.b2
    if arr is params.W1:
        W1 = W1.copy(); W1[idx] += delta
    elif arr is params.b1:
        b1 = b1.copy(); b1[idx] += delta
    elif arr is params.W2:
        W2 = W2.copy(); W2[idx] += delta
    else:
        b2 = b2.copy(); b2[idx] += delta
    return MLPParams(W1, b1, W2, b2)


def test_training_reduces_distillation_loss():
    """固定 seed 训练后 (9.2) 蒸馏损失显著下降（阈值以实测定，依据见注释）。"""
    params, hist = _trained()
    # 实测依据（seed=0，默认 40 环境 × 600 轮，k=16 编码 + 32 隐单元）：
    # 损失 2.6111 -> 0.1082（幅度比 0.041——初值是"瞎指路"的偏移方差，
    # 训练后接近步幅内的方向误差水平）。阈值 0.25 留 6 倍裕量。
    assert len(hist) == 601  # 第 0 项为初始化损失，其后逐轮，末项训练后损失
    assert hist[-1] < 0.25 * hist[0], (hist[0], hist[-1])
    # 输出偏移量纲合理：模长集中在步幅量级（训练目标 = 弧长重采样的 1.1 m 弦）。
    goal, enc = np.array([9.0, 9.0]), sdf_grid([], BOUNDS, k=16)
    out = mlp_forward(params, make_inputs(np.array([[1.0, 1.0]]), goal, enc, BOUNDS))
    assert 0.2 < float(np.linalg.norm(out[0])) < 3.0
    print(f"\n[train] loss {hist[0]:.4f} -> {hist[-1]:.4f} "
          f"(ratio {hist[-1] / hist[0]:.3f}), sample |offset| {np.linalg.norm(out[0]):.3f} m")


def test_proposals_free_rate_beats_uniform():
    """建议点自由空间比率显著高于均匀采样基线（固定测试环境集，泛化到未见环境）。

    实测依据（seed=0 网络，环境 seed=101..104，条件点 = 专家路径 0.6 m
    弧长重采样、共 77 点；均匀基线 = 每环境 4000 点）：自由比率
    net 0.938（逐环境 1.00 / 0.86 / 0.89 / 1.00）vs uniform 0.868——
    平均高出 +0.070。阈值取 gap > 0.04（约 1.8 倍裕量）；net 绝对比率
    另断言 > 0.90（网络的"避障方向"主命题）。
    """
    params, _ = _trained()
    rng = np.random.default_rng(0)
    rate_net, rate_uni = [], []
    n_cond = 0
    for i, (start, goal, obst) in enumerate(TEST_ENVS):
        res = plan_rrt(start, goal, obst, BOUNDS, seed=200 + i,
                       max_iters=2000, goal_bias=0.1)
        assert res.success, i
        pts = resample_path_by_arclength(res.path, 0.6)[:-1]  # 条件点 = 路径中间点
        n_cond += len(pts)
        enc = sdf_grid(obst, BOUNDS, k=16)
        props = propose_points(params, pts, goal, enc, BOUNDS)
        rate_net.append(float(np.mean(points_free(props, obst))))
        uni = np.column_stack([rng.uniform(0, 10, 4000), rng.uniform(0, 10, 4000)])
        rate_uni.append(float(np.mean(points_free(uni, obst))))
    gap = float(np.mean(rate_net) - np.mean(rate_uni))
    assert gap > 0.04, (rate_net, rate_uni)
    assert float(np.mean(rate_net)) > 0.90, rate_net
    print(f"\n[free-rate] net {np.mean(rate_net):.3f} {['%.2f' % v for v in rate_net]}"
          f" vs uniform {np.mean(rate_uni):.3f} (gap {gap:+.3f}, {n_cond} proposals)")


def test_biased_rrt_needs_fewer_iters():
    """带偏置 RRT 的平均求解迭代数少于均匀 RRT（同段代码 p_bias=0 对照）。

    实测依据（seed=0 网络，环境 seed=101..104 × 规划 seed 0/1，p_bias=0.5，
    max_iters=4000）：平均 86.0 vs 109.4 轮（比值 0.786），且逐 (环境, seed)
    配对 8/8 偏置臂不劣于均匀臂；固定环境集 4/4 泛化成功。阈值取
    ratio < 0.85（留 ~8% 裕量）。两臂共用同一主循环、同一随机数消费顺序
    （② 的偏置判定在 p_bias=0 时恒假、③ 的抖动 draw 不执行），唯一差异
    即采样分布——对照口径干净。
    """
    params, _ = _trained()
    iters_uni, iters_bias = [], []
    paths_by_env = []  # (start, goal, obst, [两条偏置臂路径])——metrics 逐环境判定
    for start, goal, obst in TEST_ENVS:
        env_paths = []
        for sd in (0, 1):
            uni = plan_rrt_biased(start, goal, obst, BOUNDS, params,
                                  p_bias=0.0, seed=sd, max_iters=4000)
            bias = plan_rrt_biased(start, goal, obst, BOUNDS, params,
                                   p_bias=0.5, seed=sd, max_iters=4000)
            assert uni.success and bias.success, sd
            iters_uni.append(uni.n_iters)
            iters_bias.append(bias.n_iters)
            env_paths.append(bias.path)
            # steer 限步长（(3.3)）：路径相邻点间距 ≤ η（两臂同口径）。
            assert float(np.linalg.norm(np.diff(bias.path, axis=0), axis=1).max()) <= 0.5 + 1e-9
        paths_by_env.append((start, goal, obst, env_paths))
    ratio = float(np.mean(iters_bias)) / float(np.mean(iters_uni))
    assert ratio < 0.85, (np.mean(iters_uni), np.mean(iters_bias))
    # metrics 复用（按路径加载，见文件头）：偏置臂路径全部"到达且无碰撞"
    # (M.1)，路径长度不低于直线距离（三角不等式，(M.2) 口径）。
    for start, goal, obst, env_paths in paths_by_env:
        assert metrics.success_rate(env_paths, goal, obst, tol=0.3 + 1e-9) == 1.0
        straight = float(np.linalg.norm(goal - start))
        for path in env_paths:
            assert metrics.path_length(path) >= straight - 1e-9
    print(f"\n[biased-rrt] mean iters uniform {np.mean(iters_uni):.1f} vs biased "
          f"{np.mean(iters_bias):.1f} (ratio {ratio:.3f}); detail bias {iters_bias}")


def test_determinism_and_validation():
    """同 seed 逐位可复现（训练/规划）；p_bias 与 params 的契约校验。"""
    p1, h1 = train_mpnet(seed=7, n_envs=2, epochs=25)
    p2, h2 = train_mpnet(seed=7, n_envs=2, epochs=25)
    assert np.array_equal(p1.W1, p2.W1) and np.array_equal(p1.b1, p2.b1)
    assert np.array_equal(p1.W2, p2.W2) and np.array_equal(p1.b2, p2.b2)
    assert h1 == h2
    start, goal, obst = TEST_ENVS[0]
    r1 = plan_rrt_biased(start, goal, obst, BOUNDS, p1, p_bias=0.5, seed=3)
    r2 = plan_rrt_biased(start, goal, obst, BOUNDS, p1, p_bias=0.5, seed=3)
    assert np.array_equal(r1.nodes, r2.nodes) and r1.n_iters == r2.n_iters
    # 不同 seed 产生不同的树（采样流不同的 sanity）。
    r3 = plan_rrt_biased(start, goal, obst, BOUNDS, p1, p_bias=0.5, seed=4)
    assert not np.array_equal(r1.nodes, r3.nodes)
    # 契约：p_bias 越界 / 偏置开启但未传网络 → ValueError；p_bias=0 可无网络。
    for bad in (-0.1, 1.1):
        try:
            plan_rrt_biased(start, goal, obst, BOUNDS, p1, p_bias=bad, seed=0)
            raise AssertionError("p_bias 越界应抛 ValueError")
        except ValueError:
            pass
    try:
        plan_rrt_biased(start, goal, obst, BOUNDS, None, p_bias=0.5, seed=0)
        raise AssertionError("p_bias>0 且 params=None 应抛 ValueError")
    except ValueError:
        pass
    ok = plan_rrt_biased(start, goal, obst, BOUNDS, None, p_bias=0.0, seed=3)
    assert ok.success


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

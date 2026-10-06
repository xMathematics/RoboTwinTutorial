"""projects/3d_reconstruction/metrics 的测试 —— 直接运行：``python tests/test_metrics.py``。

验证统一测评模块（公式 (M.1)-(M.4)，指标教程见 ../METRICS.md）：
Chamfer / accuracy / completion 与 O(n^2) 朴素实现交叉验证（1e-10 级）、
F-score 在 est=gt 时 = 1、偏移超 tau 时 = 0、合成场景的 P/R 手算验证、
表面采样点落在三角形上（重心坐标非负）与面积加权比例。
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

accuracy_completion = metrics.accuracy_completion
chamfer_distance = metrics.chamfer_distance
f_score = metrics.f_score
sample_mesh_surface = metrics.sample_mesh_surface

rng = np.random.default_rng(7)


# ---------------------------------------------------------------- 朴素实现 --
def _naive_min_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """朴素最近点平方距离：逐点显式双重循环。"""
    out = np.empty(len(a))
    for i in range(len(a)):
        best = np.inf
        for j in range(len(b)):
            d = float(np.sum((a[i] - b[j]) ** 2))
            if d < best:
                best = d
        out[i] = best
    return out


def _naive_chamfer(p: np.ndarray, q: np.ndarray) -> float:
    """朴素对称 Chamfer（(M.1) 的双重循环版）。"""
    return 0.5 * (float(_naive_min_sq(p, q).mean())
                  + float(_naive_min_sq(q, p).mean()))


def _barycentric_in_triangle(p: np.ndarray, tri: np.ndarray) -> bool:
    """判断点是否落在三角形上：先验共面，再解重心坐标非负。"""
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n)
    if abs(np.dot(p - tri[0], n)) > 1e-9:
        return False                                   # 不在三角形所在平面
    # 舍弃法向主导分量，在平面内解 [e1 e2] @ (u, v) = p - v0。
    drop = int(np.argmax(np.abs(n)))
    keep = [i for i in range(3) if i != drop]
    u, v = np.linalg.solve(np.column_stack([e1[keep], e2[keep]]),
                           (p - tri[0])[keep])
    return u >= -1e-9 and v >= -1e-9 and u + v <= 1.0 + 1e-9


# ------------------------------------------------------------- chamfer ----
def test_chamfer_matches_naive_o2():
    """向量化 Chamfer vs O(n^2) 双重循环朴素实现：一致到 1e-10。"""
    p = rng.normal(size=(40, 3))
    q = rng.normal(size=(35, 3)) + np.array([0.3, -0.2, 0.1])
    got = chamfer_distance(p, q)
    naive = _naive_chamfer(p, q)
    assert abs(got - naive) < 1e-10, (got, naive)
    print(f"\n[metrics] chamfer vectorized {got:.6f} vs naive {naive:.6f}")


def test_chamfer_zero_and_monotone_growth():
    """重合点集 -> 0；整体扰动越大 -> Chamfer 单调增大（平方距离口径，m^2）。"""
    # 稀疏网格点（间距 1.0 >> 平移 0.2）：最近点恒为自身，解析值 = t^2。
    g = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0), np.arange(4.0),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    assert chamfer_distance(g, g) < 1e-24
    values = [chamfer_distance(g + t * np.array([1.0, 0.0, 0.0]), g)
              for t in (0.01, 0.05, 0.2)]
    assert values[0] < values[1] < values[2], values
    assert abs(values[2] - 0.2 ** 2) < 1e-12, values[2]


def test_chamfer_requires_nonempty():
    """空点集必须显性报错（早失败优于静默 inf/nan）。"""
    p = rng.normal(size=(5, 3))
    for a, b in ((p, np.zeros((0, 3))), (np.zeros((0, 3)), p)):
        try:
            chamfer_distance(a, b)
        except ValueError:
            continue
        raise AssertionError("空点集未触发 ValueError")


# ------------------------------------------------- accuracy / completion --
def test_accuracy_completion_naive_cross_check():
    """向量化 accuracy/completion vs 朴素双重循环：一致到 1e-10。"""
    est = rng.normal(size=(30, 3))
    gt = rng.normal(size=(45, 3)) * 0.7
    acc, comp = accuracy_completion(est, gt)
    assert abs(acc - float(_naive_min_sq(est, gt).mean())) < 1e-10
    assert abs(comp - float(_naive_min_sq(gt, est).mean())) < 1e-10
    print(f"\n[metrics] acc {acc:.6f} / comp {comp:.6f} (naive cross-checked)")


def test_accuracy_completion_directional_semantics():
    """方向语义：多余点只推高 accuracy，缺失点只推高 completion（(M.2)）。"""
    gt = rng.normal(size=(80, 3))
    # 情形 1：est = gt 的一半（缺一半）-> completion 显著 > 0，accuracy ≈ 0。
    acc, comp = accuracy_completion(gt[::2], gt)
    assert acc < 1e-24 and comp > 0.01, (acc, comp)
    # 情形 2：est = gt + 一半点在远处垃圾 -> accuracy 显著 > 0，completion ≈ 0。
    junk = gt[:40] + np.array([50.0, 0.0, 0.0])
    est2 = np.vstack([gt, junk])
    acc2, comp2 = accuracy_completion(est2, gt)
    assert acc2 > 100.0 and comp2 < 1e-24, (acc2, comp2)


# ---------------------------------------------------------------- f-score -
def test_f_score_perfect_when_identical():
    """est = gt -> P = R = 1 -> F = 1（任意 tau）。"""
    p = rng.normal(size=(50, 3))
    for tau in (0.01, 0.1, 1.0):
        assert abs(f_score(p, p, tau) - 1.0) < 1e-12


def test_f_score_zero_when_shift_beyond_tau():
    """整体平移 > tau -> P = R = 0 -> F = 0（约定 P+R=0 时 F=0）。"""
    p = rng.normal(size=(50, 3))
    q = p + np.array([1.0, 0.0, 0.0])              # 平移 1 m >> tau
    assert f_score(p, q, tau=0.05) == 0.0


def test_f_score_synthetic_precision_recall():
    """可手算的合成场景：P、R 分量与 F 逐一验证（(M.3)）。

    gt = 100 个点；est = 80 个 gt 点（在 tau 内）+ 20 个远离点：
    precision = 80/100，recall = 80/100，F = 0.8。
    """
    gt = np.column_stack([np.arange(100.0), np.zeros(100), np.zeros(100)])
    est_in = gt[:80]
    est_out = gt[:20] + np.array([1000.0, 0.0, 0.0])  # 远离所有 gt 点
    est = np.vstack([est_in, est_out])
    f = f_score(est, gt, tau=0.5)
    precision, recall = 0.8, 0.8
    expect = 2 * precision * recall / (precision + recall)
    assert abs(f - expect) < 1e-12, (f, expect)
    # 只有 est 一侧全对、gt 另 20 点无对应：F 由 recall 拖低。
    f2 = f_score(gt[:80], gt, tau=0.5)
    assert abs(f2 - 2 * (1.0 * 0.8) / 1.8) < 1e-12, f2
    # est 点带 0.1 m 偏移时，tau 收紧到 1e-6 会切断全部对应 -> F 显著下降。
    est_off = est + np.array([0.1, 0.0, 0.0])
    assert f_score(est_off, gt, tau=0.5) > 0.5
    assert f_score(est_off, gt, tau=1e-6) == 0.0


# ------------------------------------------------------- mesh sampling ----
def test_sample_mesh_points_on_triangles():
    """采样点确实落在三角形上：共面且重心坐标全部非负（含边界容差）。"""
    v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    f = np.array([[0, 1, 2], [0, 1, 3]])
    pts = sample_mesh_surface(v, f, 300, seed=0)
    assert pts.shape == (300, 3)
    for p in pts:
        assert any(_barycentric_in_triangle(p, v[tri]) for tri in f), \
            f"采样点 {p} 不在任何三角形上"


def test_sample_mesh_area_weighting_and_determinism():
    """面积加权：两三角形面积比 -> 采样比例 ≈ 面积份额；同 seed 结果逐位一致。"""
    # 两块平行三角形（z=0 与 z=5 平面），按 z 坐标即可无歧义区分归属。
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # 小：面积 0.5
                  [0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]])  # 大：面积 2.0
    f = np.array([[0, 1, 2], [3, 4, 5]])
    a_small = 0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0]))
    a_big = 0.5 * np.linalg.norm(np.cross(v[4] - v[3], v[5] - v[3]))
    pts = sample_mesh_surface(v, f, 4000, seed=0)
    on_small = pts[:, 2] < 2.5                         # 平面 z=0 vs z=5，可分
    ratio = float(on_small.mean())
    expect = a_small / (a_small + a_big)
    assert abs(ratio - expect) < 0.05, (ratio, expect)
    assert np.array_equal(pts, sample_mesh_surface(v, f, 4000, seed=0))
    pts_other = sample_mesh_surface(v, f, 4000, seed=1)
    assert not np.array_equal(pts, pts_other)


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

"""projects/slam/bowloop 的测试——直跑: ``python tests/test_bowloop.py``.

全合成的第 09 章基准: TF-IDF/L1 评分端点（式 9.1-9.3）、只对共享单词帧可见
的倒排索引查询、20 帧的回环检测（帧 0 与重访段 17-19 为同一场景，即帧 0 与
帧 19 同场景），以及一张 2 维位姿图——须用单条高精度回环边把 ~30 度累积航
向漂移压到 1/5 以下（式 9.7-9.8）。
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bowloop import (  # noqa: E402
    InvertedIndex,
    PoseGraph2D,
    Vocabulary,
    detect_loop,
    score,
    se2_relative_jacobian,
    se2_relative_residual,
    wrap_angle,
)

DESC_DIM = 32
N_DESC = 40
DEG = np.pi / 180.0


def _scene_frames(
    scene_bases: np.ndarray, scene_of_frame: list[int], rng: np.random.Generator
) -> list[np.ndarray]:
    """每帧 = 场景偏置向量 + 每描述子独立高斯噪声（伪描述子）."""
    frames = []
    for s in scene_of_frame:
        noise = rng.normal(scale=0.35, size=(N_DESC, DESC_DIM))
        frames.append(scene_bases[s][None, :] + noise)
    return frames


def _bow_pipeline():
    """训练词表并把 20 帧编码成 BoW 向量: 帧 0..16 = 场景 0..16, 17..19 重访场景 0."""
    scene_bases = np.random.default_rng(42).normal(size=(17, DESC_DIM))
    # 帧 17-19 为场景 0 的重访段: 回到旧地必然连续若干帧都看到它——这正是
    # 时间一致性检验(连续 3 帧)的物理基础; 帧 0 与帧 19 同场景。
    scene_of_frame = list(range(17)) + [0, 0, 0]

    train_rows = []
    rng_train = np.random.default_rng(100)
    for s in range(len(scene_bases)):
        for _ in range(3):  # 每场景 3 帧训练描述子
            train_rows.append(scene_bases[s][None, :] + rng_train.normal(
                scale=0.35, size=(N_DESC, DESC_DIM)
            ))
    train_pool = np.concatenate(train_rows, axis=0)

    vocab = Vocabulary(n_words=20, n_iters=15, seed=0).fit(train_pool)
    test_frames = _scene_frames(
        scene_bases, scene_of_frame, np.random.default_rng(200)
    )
    vectors = [vocab.transform(frame) for frame in test_frames]
    return vocab, vectors


def test_score_endpoints():
    """式 (9.3) 端点: 相同向量 s=1, 支撑集不相交 s=1/2（公式值）."""
    v = np.array([0.0, 0.5, 0.5])
    assert abs(score(v, v) - 1.0) < 1e-12   # 自比: 分子 ||v-v||_1 = 0 -> s = 1
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(score(a, b) - 0.5) < 1e-12   # 不相交: |a-b|_1 = |a|_1 + |b|_1 -> s = 1/2
    # 部分重叠介于两端点之间且对 L1 距离单调
    c = np.array([0.8, 0.2])
    assert 0.5 < score(a, c) < 1.0
    assert score(a, c) > score(a, b)


def test_vocabulary_idf_downweights_common_words():
    """式 (9.1): 训练库中越常见的单词 IDF 越低（抑制常见词）."""
    rng = np.random.default_rng(1)
    common = rng.normal(size=DESC_DIM)[None, :] + rng.normal(
        scale=0.05, size=(300, DESC_DIM)
    )
    rare = rng.normal(size=(6, DESC_DIM)) * 0.05 + rng.normal(
        size=DESC_DIM
    )[None, :]
    pool = np.concatenate([common, rare], axis=0)
    vocab = Vocabulary(n_words=8, n_iters=15, seed=0).fit(pool)
    d2 = (
        np.sum(pool**2, axis=1)[:, None]
        + np.sum(vocab.words**2, axis=1)[None, :]
        - 2.0 * pool @ vocab.words.T
    )
    counts = np.bincount(np.argmin(d2, axis=1), minlength=8)
    most_common = int(np.argmax(counts))
    least_common = int(np.argmin(counts))
    assert counts[most_common] > counts[least_common] * 10   # 造出的"常见/罕见"差距须足够悬殊
    assert vocab.idf[most_common] < vocab.idf[least_common]  # IDF 单调: 越常见权重越低 (9.1)
    print(f"\n[vocab] idf: most common word {vocab.idf[most_common]:.3f}, "
          f"least common word {vocab.idf[least_common]:.3f}")


def test_transform_is_l1_normalized_tfidf():
    """式 (9.2): BoW 向量非负、L1 和为 1；单一单词的帧质量集中于该单词."""
    rng = np.random.default_rng(2)
    pool = rng.normal(size=(200, DESC_DIM))
    vocab = Vocabulary(n_words=10, n_iters=15, seed=0).fit(pool)
    v = vocab.transform(pool[:20])
    assert v.shape == (10,)
    assert np.all(v >= 0.0)                    # tf、idf 均非负 -> BoW 向量非负
    assert abs(v.sum() - 1.0) < 1e-12          # L1 归一化 (9.2): 和为 1
    single = vocab.transform(pool[:1])          # 单描述子 -> 单单词
    assert abs(single.sum() - 1.0) < 1e-12
    assert abs(score(v, v) - 1.0) < 1e-12
    assert vocab.transform(np.zeros((0, DESC_DIM))).sum() == 0.0  # 空帧 -> 零向量约定


def test_inverted_index_query():
    """倒排索引: 只返回共享单词的候选, 按 (9.3) 相似度降序, top_k 截断."""
    index = InvertedIndex()
    index.add(0, np.array([1.0, 0.0, 0.0]))
    index.add(1, np.array([0.9, 0.1, 0.0]))
    index.add(2, np.array([0.0, 1.0, 0.0]))    # 与查询向量无公共单词
    result = index.query(np.array([1.0, 0.0, 0.0]))
    assert [fid for fid, _ in result] == [0, 1]  # 帧 2 不共享单词, 不为候选
    assert abs(result[0][1] - 1.0) < 1e-12
    assert abs(result[1][1] - 0.95) < 1e-12      # 1 - 0.5*0.2/2 (式 9.3)
    assert index.query(np.array([1.0, 0.0, 0.0]), top_k=1) == [result[0]]


def test_detect_loop_finds_revisit_pair():
    """帧 0 与帧 19 同场景（重访段 17-19 连续检出）-> 确认回环 (19, 0)."""
    _, vectors = _bow_pipeline()
    same_scene = score(vectors[19], vectors[0])
    cross_scene = max(
        score(vectors[t], vectors[0]) for t in range(1, 10)
    )
    assert same_scene > 0.8, same_scene
    assert cross_scene < 0.7, cross_scene        # 不相交单词集 -> (9.3) 下限

    loops = detect_loop(vectors, min_gap=5, score_thresh=0.7)
    assert (19, 0) in loops, loops
    assert len(loops) == 1, loops               # 无其他场景被误检（其余帧互不同场景且跨场景分 < 阈值）
    for cur, match in loops:
        assert cur - match >= 5                 # min_gap 全程被尊重
    print(f"\n[detect_loop] same-scene score = {same_scene:.3f}, "
          f"max cross-scene = {cross_scene:.3f}, loops = {loops}")


def test_detect_loop_min_gap_and_consistency():
    """相邻重复帧被 min_gap 排除; 时间一致性不足 3 帧不确认."""
    v = np.array([1.0, 0.0])
    w = np.array([0.0, 1.0])
    vectors = [v, w, v, v]                      # 只有帧 2, 3 是帧 0 的"重访"
    assert detect_loop(vectors, min_gap=2, score_thresh=0.9) == []
    # 默认 temporal_consistency=3: 帧 2、3 连续检出仅 2 帧, 不足 3 帧不确认
    loops = detect_loop(
        vectors, min_gap=2, score_thresh=0.9, temporal_consistency=2
    )
    assert loops == [(3, 0)]                    # 连续 2 帧检出即确认


def test_se2_jacobian_matches_finite_differences():
    """SE(2) 残差雅可比与有限差分一致（符号/解析式校验）."""
    rng = np.random.default_rng(3)
    pi_ = rng.normal(scale=0.5, size=3)
    pj = rng.normal(scale=0.5, size=3)
    z = rng.normal(scale=0.3, size=3)
    J = se2_relative_jacobian(pi_, pj)

    def residual(state):
        return se2_relative_residual(state[:3], state[3:], z)

    state = np.concatenate([pi_, pj])
    eps = 1e-7
    J_num = np.zeros((3, 6))
    for k in range(6):
        sp, sm = state.copy(), state.copy()
        sp[k] += eps
        sm[k] -= eps
        J_num[:, k] = (residual(sp) - residual(sm)) / (2.0 * eps)
    err = float(np.abs(J - J_num).max())
    assert err < 1e-6, err
    print(f"\n[se2 jacobian] max |analytic - finite difference| = {err:.2e}")


def test_pose_graph_exact_data_is_fixed_point():
    """无噪声无漂移时残差为零 (9.8 自洽), 优化不应移动任何节点."""
    gt = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, np.pi / 2],
            [0.0, 2.0, np.pi],
        ]
    )
    graph = PoseGraph2D()
    for k, pose in enumerate(gt):
        graph.add_node(k, pose)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edges:
        d = gt[j, :2] - gt[i, :2]
        c, s = np.cos(gt[i, 2]), np.sin(gt[i, 2])
        z_xy = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])  # 世界系差 -> 体坐标系 (9.7)
        z_th = wrap_angle(gt[j, 2] - gt[i, 2])
        graph.add_edge(i, j, [*z_xy, z_th])
    cost = graph.optimize()
    assert cost < 1e-18, cost   # 真值处残差为 0，GN 第一步增量即为 0（不动点）
    poses = graph.poses()
    for k in range(4):
        assert np.allclose(poses[k], gt[k], atol=1e-9), poses[k]  # 无漂移不应移动任何节点


def _rectangle_gt(n_nodes: int = 20, w: float = 6.0, h: float = 4.0) -> np.ndarray:
    """沿矩形边界等弧长布点的真值轨迹（末点回到起点，朝向为边界切向）."""
    perimeter = 2.0 * (w + h)
    corners = np.array(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h], [0.0, 0.0]]
    )
    seg = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    arc = np.linspace(0.0, perimeter, n_nodes)
    idx = np.clip(np.searchsorted(cum, arc, side="right") - 1, 0, 3)
    along = arc - cum[idx]
    direction = (corners[idx + 1] - corners[idx]) / seg[idx][:, None]
    xy = corners[idx] + along[:, None] * direction
    theta = np.arctan2(direction[:, 1], direction[:, 0])
    return np.concatenate([xy, theta[:, None]], axis=1)


def test_pose_graph_closes_rectangle_loop():
    """矩形轨迹 + ~30 度累积角度漂移 + 一条准确回环边 -> 末端漂移 < 1/5."""
    gt = _rectangle_gt()
    n = len(gt)
    rng = np.random.default_rng(7)
    sigma_xy, sigma_th = 0.03, 0.01
    drift_total = 30.0 * DEG
    bias = drift_total / (n - 1)                # 每步系统性角度漂移
    info_odom = np.diag([1.0 / sigma_xy**2, 1.0 / sigma_xy**2, 1.0 / sigma_th**2])

    graph = PoseGraph2D()
    graph.add_node(0, gt[0])                    # 锚定节点 = 真值（规范固定）
    init = np.empty_like(gt)
    init[0] = gt[0]
    for t in range(n - 1):
        d = gt[t + 1, :2] - gt[t, :2]
        c, s = np.cos(gt[t, 2]), np.sin(gt[t, 2])
        z_xy = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        z_xy = z_xy + rng.normal(scale=sigma_xy, size=2)
        z_th = wrap_angle(gt[t + 1, 2] - gt[t, 2]) + bias + rng.normal(
            scale=sigma_th
        )
        graph.add_edge(t, t + 1, [*z_xy, z_th], info_odom)
        # 里程计递推 -> 漂移初值
        cc, ss = np.cos(init[t, 2]), np.sin(init[t, 2])
        init[t + 1, :2] = init[t, :2] + np.array(
            [cc * z_xy[0] - ss * z_xy[1], ss * z_xy[0] + cc * z_xy[1]]
        )
        init[t + 1, 2] = wrap_angle(init[t, 2] + z_th)
    for k in range(1, n):
        graph.add_node(k, init[k])

    # 一条准确回环边 (19 -> 0): 相对位姿取真值, 信息 100 倍于里程计
    d = gt[0, :2] - gt[n - 1, :2]
    c, s = np.cos(gt[n - 1, 2]), np.sin(gt[n - 1, 2])
    z_loop = [
        c * d[0] + s * d[1],
        -s * d[0] + c * d[1],
        wrap_angle(gt[0, 2] - gt[n - 1, 2]),
    ]
    graph.add_edge(n - 1, 0, z_loop, info_odom * 100.0)

    before_pos = float(np.linalg.norm(init[n - 1, :2] - gt[n - 1, :2]))
    before_ang = float(abs(wrap_angle(init[n - 1, 2] - gt[n - 1, 2])))
    cost = graph.optimize()
    poses = graph.poses()
    after_pos = float(np.linalg.norm(poses[n - 1][:2] - gt[n - 1, :2]))
    after_ang = float(abs(wrap_angle(poses[n - 1][2] - gt[n - 1, 2])))

    assert before_ang > 25.0 * DEG, before_ang   # 漂移确已积累 ~30 度（实验前提自检）
    assert after_pos < before_pos / 5.0, (after_pos, before_pos)   # 100 倍信息回环边应把漂移压到 1/5 以下
    assert after_ang < before_ang / 5.0, (after_ang, before_ang)
    print(f"[pose graph] end-node drift before: {before_pos:.3f} m / "
          f"{np.degrees(before_ang):.1f} deg -> after: {after_pos:.3f} m / "
          f"{np.degrees(after_ang):.1f} deg (chi2 {cost:.2f})")


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

"""词答回环检测与位姿图优化（教程第 09 章：回环检测）.

函数流水线: 本模块对应 ORB-SLAM 回环线程的检索与验证核心（前端重定位共用
同一词袋机制）。离线支线: ``Vocabulary.fit``（k-means 聚出视觉单词 + IDF
统计）-> ``Vocabulary.transform``（每帧描述子 -> TF-IDF BoW 向量）->
``InvertedIndex.add``（word -> 帧的倒排索引入库）；在线支线:
``detect_loop``（倒排索引检索 + L1 评分 (9.3) + 时间一致性 -> 确认的回环
对）-> ``PoseGraph2D``（``add_node``/``add_edge`` 建图 -> ``optimize``
稠密 Gauss-Newton 求解 (9.8) -> ``poses()``）。输入为各帧描述子 (N, D)
与位姿图节点/边，输出为确认的回环对列表与优化后节点位姿。仅依赖 numpy
（教学实现不依赖 core.*）。

式号 (9.x) 指向 ``tutorials/slam/09_回环检测.md``：

- (9.1) TF-IDF 权重 ``w_i = tf_i * log(N / n_i)``
- (9.2) 词袋向量（L1 归一化）
- (9.3) L1 相似度评分（DBoW2）
- (9.7) 相对位姿观测模型 ``z_ij = log(T_ij)^vee``
- (9.8) 位姿图目标函数（马氏范数）

DBoW2 式场景识别流水线的教学实现（Gálvez-López
& Tardós, "Bags of Binary Words for Fast Place Recognition in Image
Sequences", IEEE T-RO 2012），即 ORB-SLAM 回环线程所用机制
(Mur-Artal, Martinez Montiel & Tardos, IEEE T-RO 2015,
papers/slam/classics/arXiv-1502.00956_ORB-SLAM.pdf, §VII Loop Closing)，加上
一个小规模 2 维位姿图优化器（g2o, Kümmerle et al., ICRA 2011；Grisetti et
al., "A tutorial on graph-based SLAM", ICRA 2010 的教学版）.

记号约定
-----------
- 描述子为 (N, D) 实向量（教学实现用随机向量模拟 ORB 描述子的"可聚类
  性"；真实系统中 D = 32/64 字节二进制串，距离换成汉明距离，机制不变）。
- 2 维位姿 ``s = (x, y, theta)``；边测量 ``z = (dx, dy, dtheta)`` 为 *体坐
  标系* 下的相对位姿（帧 i 观测帧 j）：``p_j = p_i + R(theta_i) [dx, dy]``，
  ``theta_j = theta_i + dtheta``（(9.7) 的 SE(2) 特例）。
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PoseGraph2D",
    "PoseGraphEdge",
    "Vocabulary",
    "InvertedIndex",
    "detect_loop",
    "score",
    "se2_relative_jacobian",
    "se2_relative_residual",
    "wrap_angle",
]


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """把角度（标量或数组）wrap 到 ``(-pi, pi]``（角度残差跨分支切割时防 2pi 跳变）.

    依据: ``arctan2(sin(a), cos(a))`` 是 ``a`` 模 ``2pi`` 后落在 ``(-pi, pi]``
    的等价角——残差 (9.7) 必须落在向量空间，角度须先绕回单值分支。
    """
    a = np.asarray(angle, dtype=float)
    wrapped = np.arctan2(np.sin(a), np.cos(a))
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def score(v1: np.ndarray, v2: np.ndarray) -> float:
    """L1 归一化相似度评分 (式 9.3, DBoW2).

    ``s = 1 - (1/2) ||v_a - v_b||_1 / (||v_a||_1 + ||v_b||_1)``。端点性质由
    范数分解直接给出: 同图自比 ``v_a = v_b`` 时分子为 0，``s = 1``；支撑集
    完全不相交时逐分量 ``|a_i - b_i| = |a_i| + |b_i|``，代入得 ``s = 1/2``
    ——故对非负 BoW 向量 ``s`` 单调落在 ``[1/2, 1]``，区分度集中在共享单词
    的质量占比上（对 L1 距离单调递减）。

    Args:
        v1, v2: 等长向量（通常为 L1 归一化的 BoW 向量）。

    Returns:
        相似度，范围 ``[1/2, 1]``（同支撑集单调性见上；两输入全零时按约定
        返回 0.0，见实现中的分母保护）。
    """
    a = np.asarray(v1, dtype=float).reshape(-1)
    b = np.asarray(v2, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("v1 and v2 must have the same shape")
    denom = float(np.abs(a).sum() + np.abs(b).sum())
    if denom <= 0.0:
        return 0.0
    return float(1.0 - 0.5 * np.abs(a - b).sum() / denom)


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D) 与 (M, D) 的成对平方欧氏距离 (N, M)，展开范数恒等式降内存."""
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


def _kmeans(data: np.ndarray, k: int, n_iters: int, rng: np.random.Generator) -> np.ndarray:
    """NumPy 实现的 k-means（k-means++ 初始化 + Lloyd 迭代）.

    初始化: k-means++（Arthur & Vassilvitskii, 2007）——按 D^2 概率逐个选种
    子，避免全随机初始化的坏簇；迭代: Lloyd，空簇用随机训练样本重置（seeded，
    保持确定性）。词表训练是离线一次性操作（第 09 章 9.1 第 1 步），此为
    教学规模的实现。

    Args:
        data: (N, D) 训练描述子。
        k: 簇（视觉单词）数。
        n_iters: Lloyd 迭代次数。
        rng: numpy 随机数生成器（确定性）。

    Returns:
        (k, D) 簇中心（视觉单词）。
    """
    n = len(data)
    centers = np.empty((k, data.shape[1]))
    centers[0] = data[int(rng.integers(n))]
    d2 = _sq_dist(data, centers[0:1])[:, 0]
    for j in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=d2 / total))
        centers[j] = data[idx]
        d2 = np.minimum(d2, _sq_dist(data, centers[j : j + 1])[:, 0])

    for _ in range(n_iters):
        labels = np.argmin(_sq_dist(data, centers), axis=1)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = data[mask].mean(axis=0)
            else:
                centers[j] = data[int(rng.integers(n))]  # 空簇重置
    return centers


class Vocabulary:
    """视觉词表: k-means 聚类出 K 个视觉单词 + TF-IDF 权重 (式 9.1, 9.2).

    离线 ``fit`` 把训练描述子聚成 K 个单词并统计 IDF；在线
    :meth:`transform` 把一帧描述子编码成 BoW 向量。

    Attributes:
        n_words: 视觉单词数 K。
        n_iters: k-means Lloyd 迭代次数。
        _rng: ``numpy.random.Generator``——k-means 初始化/空簇重置的确定性
            随机源。
        _words: 词表矩阵 (K, D)，各行为视觉单词（簇中心）；未训练时为
            ``None``（经只读属性 :attr:`words` 访问）。
        _idf: 逆文档频率 (K,)，``log(N / n_i)``；未训练时为 ``None``（经
            :attr:`idf` 访问）。
    """

    def __init__(self, n_words: int = 64, n_iters: int = 15, seed: int = 0) -> None:
        """Args:
            n_words: 视觉单词数 K（聚类簇数）。
            n_iters: k-means Lloyd 迭代次数（10-20 足够收敛到稳定划分）。
            seed: k-means 初始化与空簇重置的随机种子（确定性）。
        """
        if n_words < 1:
            raise ValueError("n_words must be >= 1")
        self.n_words = int(n_words)
        self.n_iters = int(n_iters)
        self._rng = np.random.default_rng(seed)
        self._words: np.ndarray | None = None
        self._idf: np.ndarray | None = None

    @property
    def words(self) -> np.ndarray:
        """(K, D) 视觉单词（簇中心）。"""
        if self._words is None:
            raise RuntimeError("call fit() before reading words")
        return self._words

    @property
    def idf(self) -> np.ndarray:
        """(K,) 逆文档频率 ``log(N / n_i)``（式 9.1）。"""
        if self._idf is None:
            raise RuntimeError("call fit() before reading idf")
        return self._idf

    def fit(self, descriptors: np.ndarray) -> "Vocabulary":
        """聚类训练描述子并统计 IDF (式 9.1).

        IDF 依据: 信息量 ``I(w) = -log p(w)``（香农自信息——概率越小、信息
        量越大），以出现频率估计 ``p(w_i) = n_i / N`` 得 ``log(N / n_i)``。
        常见词 ``n_i -> N`` 时权重趋于 0（弱纹理区域的平凡角点不再贡献虚假
        相似度），独有词权重最大——IDF 抑制常见词的依据。教学实现以"训练
        描述子"为文档粒度统计 ``n_i``（DBoW2 以训练图像为文档；对"常见词
        降权"的机制相同，见模块 docstring 的约定）。

        Args:
            descriptors: (N, D) 训练描述子池（N >= K）。

        Returns:
            ``self``（链式调用）。
        """
        data = np.asarray(descriptors, dtype=float)
        if data.ndim != 2 or len(data) < self.n_words:
            raise ValueError(
                f"fit expects (N, D) descriptors with N >= {self.n_words}"
            )
        self._words = _kmeans(data, self.n_words, self.n_iters, self._rng)
        labels = np.argmin(_sq_dist(data, self._words), axis=1)
        counts = np.bincount(labels, minlength=self.n_words).astype(float)
        # n_i 截为 >= 1（平滑）: 训练库中零命中的词若无截断会给 log(N/0)=inf，
        # 截为 1 得最大权重 log(N)，与"越罕见越重要"方向一致且数值有限。
        self._idf = np.log(len(data) / np.maximum(counts, 1.0))
        return self

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """一帧描述子 -> TF-IDF 加权、L1 归一化的 BoW 向量 (式 9.2).

        ``tf_i`` = 该帧落入单词 i 的描述子数（图内常见度），乘以 IDF（库内
        罕见度）后按 L1 范数归一化——归一化消除图像特征数不同的影响，使不同
        规模的帧可比较（第 09 章 9.1 第 3 步）。

        Args:
            descriptors: (N, D) 该帧描述子。

        Returns:
            (K,) BoW 向量（非负、和为 1；空帧返回零向量）。
        """
        words = self.words  # 触发未训练检查
        descs = np.asarray(descriptors, dtype=float)
        if descs.size == 0:
            return np.zeros(self.n_words)
        descs = descs.reshape(-1, words.shape[1])
        labels = np.argmin(_sq_dist(descs, words), axis=1)
        tf = np.bincount(labels, minlength=self.n_words).astype(float)
        v = tf * self._idf
        l1 = float(np.abs(v).sum())
        return v / l1 if l1 > 0.0 else v


class InvertedIndex:
    """倒排索引 (第 09 章 9.1 第 5 步): word -> 挂链的帧列表.

    查询只沿当前帧 BoW 向量的非零单词所挂链表取候选、逐候选评分——单次查
    询代价与库规模解耦（近似 O(1)），这是词袋相对两两比对的复杂度优势。
    """

    def __init__(self) -> None:
        self._postings: dict[int, list[int]] = {}
        self._vectors: dict[int, np.ndarray] = {}

    def add(self, frame_id: int, vector: np.ndarray) -> None:
        """把一帧的 BoW 向量入库：非零单词逐一挂链.

        Args:
            frame_id: 帧编号（须唯一）。
            vector: (K,) BoW 向量。

        Raises:
            ValueError: ``frame_id`` 重复入库。
        """
        if frame_id in self._vectors:
            raise ValueError(f"frame {frame_id} already indexed")
        v = np.asarray(vector, dtype=float).reshape(-1)
        self._vectors[frame_id] = v
        for word in np.nonzero(v)[0]:
            self._postings.setdefault(int(word), []).append(frame_id)

    def query(
        self, vector: np.ndarray, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """按 (9.3) 相似度降序返回与 ``vector`` 共享单词的候选帧.

        Args:
            vector: (K,) 查询 BoW 向量。
            top_k: 最多返回的候选数；``None`` 返回全部候选。

        Returns:
            ``(frame_id, score)`` 列表，按相似度降序。
        """
        v = np.asarray(vector, dtype=float).reshape(-1)
        candidates: set[int] = set()
        for word in np.nonzero(v)[0]:
            candidates.update(self._postings.get(int(word), ()))
        scored = [(fid, score(v, self._vectors[fid])) for fid in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored if top_k is None else scored[:top_k]


def detect_loop(
    vectors: Sequence[np.ndarray],
    min_gap: int,
    score_thresh: float,
    temporal_consistency: int = 3,
) -> list[tuple[int, int]]:
    """词袋检索 + 时间一致性的回环检测 (第 09 章 9.1/9.2).

    逐帧在倒排索引中取 (9.3) 相似度最高、且间隔 ``t - fid >= min_gap``、
    分数超 ``score_thresh`` 的历史帧作为候选（``min_gap`` 排除相邻帧的平凡
    "回环"——相邻帧本来就长得像）。**时间一致性**（感知混淆兜底依据）: 候
    选须连续 ``temporal_consistency`` 帧都被检出才确认——感知混淆
    （perceptual aliasing，第 09 章 9.2 ①）使走廊、重复立面等"长得像"的不
    同地点也会偶尔得高分，但这类误检是瞬时、随机的，难以连续多帧命中同一
    历史区域；而真实回到旧地时必然连续若干帧都看到它。假阳性回环会把错误
    约束灌进位姿图、摧毁地图，假阴性只是少修正一次——风险不对称，用少量
    延迟换精度（ORB-SLAM §VII 要求候选在连续 3 个关键帧上一致检出；
    ORB-SLAM3 §VI "validation continues until three keyframes fire in a
    row"）。教学实现只做阈值连击；ORB-SLAM 还要求候选在共视图中相互一致，
    见 9.2 (b)。

    Args:
        vectors: 按时间序排列的各帧 BoW 向量序列。
        min_gap: 候选回环两帧的最小帧间隔。
        score_thresh: 候选相似度阈值 (9.3)。
        temporal_consistency: 连续检出帧数阈值（ORB-SLAM 取 3）。

    Returns:
        确认的回环 ``(current_id, matched_id)`` 列表，按时间序。
    """
    index = InvertedIndex()
    streak = 0
    loops: list[tuple[int, int]] = []
    for t, vec in enumerate(vectors):
        best_fid = -1
        for fid, s in index.query(vec):  # 已按相似度降序
            if t - fid >= min_gap and s >= score_thresh:
                best_fid = fid  # 第一个满足间隔与阈值的即最优
                break
        streak = streak + 1 if best_fid >= 0 else 0
        if best_fid >= 0 and streak >= temporal_consistency:
            loops.append((t, best_fid))
        index.add(t, vec)
    return loops


def se2_relative_residual(
    pose_i: np.ndarray, pose_j: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """SE(2) 相对位姿残差 (式 9.7 的 SE(2) 特例), shape (3,).

    ``e = [R(theta_i)^T (t_j - t_i) - z_xy; wrap(theta_j - theta_i - z_theta)]``
    ——测量在帧 i 的体坐标系下表达，角度残差 wrap 到 ``(-pi, pi]`` 防 2pi
    跳变（(9.7) 要求残差落在向量空间，角度必须先绕回单值分支）。
    """
    pi_ = np.asarray(pose_i, dtype=float).reshape(3)
    pj = np.asarray(pose_j, dtype=float).reshape(3)
    z = np.asarray(z, dtype=float).reshape(3)
    c, s = np.cos(pi_[2]), np.sin(pi_[2])
    d = pj[:2] - pi_[:2]
    e_t = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) - z[:2]
    e_theta = wrap_angle(pj[2] - pi_[2] - z[2])
    return np.concatenate([e_t, [e_theta]])


def se2_relative_jacobian(pose_i: np.ndarray, pose_j: np.ndarray) -> np.ndarray:
    """SE(2) 相对位姿残差的解析雅可比 ``de/d(pose_i, pose_j)``，shape (3, 6).

    由 ``R^T(theta) = [[cos, sin], [-sin, cos]]`` 逐项求导:
    ``de_t/d t_i = -R^T``、``de_t/d t_j = R^T``、
    ``de_t/d theta_i = (dR^T/dtheta) (t_j - t_i) = [[-sin, cos], [-cos, -sin]] (t_j - t_i)``、
    ``de_theta/d theta_i = -1``、``de_theta/d theta_j = +1``。
    """
    pi_ = np.asarray(pose_i, dtype=float).reshape(3)
    pj = np.asarray(pose_j, dtype=float).reshape(3)
    c, s = np.cos(pi_[2]), np.sin(pi_[2])
    d = pj[:2] - pi_[:2]
    J = np.zeros((3, 6))
    J[:2, 0:2] = np.array([[-c, -s], [s, -c]])        # -R^T
    J[:2, 2] = np.array([-s * d[0] + c * d[1], -c * d[0] - s * d[1]])
    J[:2, 3:5] = np.array([[c, s], [-s, c]])          # R^T
    J[2, 2] = -1.0
    J[2, 5] = 1.0
    return J


@dataclass(frozen=True)
class PoseGraphEdge:
    """位姿图的一条边: 相对位姿观测 + 信息矩阵 (式 9.7, 9.8).

    Attributes:
        i, j: 两端节点编号（测量方向 i -> j）。
        z: (3,) 体坐标系相对位姿测量 ``(dx, dy, dtheta)``。
        info: (3, 3) 信息矩阵 ``Lambda_ij``（噪声协方差之逆，马氏范数 (9.8)
            的加权）。
    """

    i: int
    j: int
    z: np.ndarray
    info: np.ndarray


class PoseGraph2D:
    """SE(2) 位姿图优化 (第 09 章 9.3; 式 9.8 的 SE(2) 特例, 小规模稠密 GN).

    节点 = 关键帧 SE(2) 位姿，边 = 里程计/回环的相对位姿观测 + 信息矩阵。
    优化在流形上迭代 (第 08 章 (8.3)/(8.7) 的 SE(2) 版): 每轮在当前估计处
    线性化 (9.7) 残差、组装正规方程、整体更新后把角度 wrap 回 ``(-pi, pi]``。

    **Gauge freedom（固定第一个节点的依据）**: (9.8) 的残差只含*相对*位姿，
    对全局刚体变换 ``(x, y, theta) -> (R (x, y) + t, theta + phi)`` 严格不
    变——目标函数沿全局 SE(2) 的 3 个生成元方向平坦，Hessian 必有 3 维零空
    间、正规方程奇异。必须以先验锚定（gauge fixing）: 固定第一个节点等价于
    给它的 3 个自由度加单位信息矩阵先验；g2o/GTSAM 与 Grisetti et al.
    ("A tutorial on graph-based SLAM", ICRA 2010) 同此处理。

    Attributes:
        _init_poses: ``dict[int, ndarray]``——节点初值表，键为节点编号，
            值为 (3,) 位姿 ``(x, y, theta)``（theta 单位 rad，任意实数）。
        _edges: ``list[PoseGraphEdge]``——相对位姿边（里程计/回环观测 +
            信息矩阵）。
        _optimized: ``dict[int, ndarray] | None``——``optimize`` 成功后的
            节点表（结构同 ``_init_poses``）；未优化时为 ``None``。
    """

    def __init__(self) -> None:
        self._init_poses: dict[int, np.ndarray] = {}
        self._edges: list[PoseGraphEdge] = []
        self._optimized: dict[int, np.ndarray] | None = None

    def add_node(self, node_id: int, pose: Sequence[float]) -> None:
        """加入位姿节点（初值，通常为里程计递推的漂移位姿）.

        Args:
            node_id: 节点编号（整数，须唯一）。
            pose: ``(x, y, theta)`` 初值。

        Raises:
            ValueError: 节点编号重复。
        """
        if node_id in self._init_poses:
            raise ValueError(f"node {node_id} already added")
        self._init_poses[node_id] = np.asarray(pose, dtype=float).reshape(3)

    def add_edge(
        self,
        i: int,
        j: int,
        measurement: Sequence[float],
        information: np.ndarray | None = None,
    ) -> None:
        """加入一条相对位姿边.

        Args:
            i, j: 测量方向 i -> j 的两端节点编号。
            measurement: (3,) 体坐标系相对位姿 ``(dx, dy, dtheta)`` (9.7)。
            information: (3, 3) 信息矩阵；``None`` 取单位阵。回环边通常给
                远高于里程计边的信息（验证通过的回环更可信, 第 09 章 9.2）。
        """
        z = np.asarray(measurement, dtype=float).reshape(3)
        info = (
            np.eye(3)
            if information is None
            else np.asarray(information, dtype=float).reshape(3, 3)
        )
        self._edges.append(PoseGraphEdge(i=int(i), j=int(j), z=z, info=info))

    def optimize(self, n_iters: int = 30, tol: float = 1e-10) -> float:
        """稠密 Gauss-Newton 迭代求解位姿图 (式 9.8), 返回最终卡方代价.

        每轮: 组装 ``H = sum J^T Lambda J``、``b = sum J^T Lambda e``，锚定
        第一个节点（gauge fixing，见类 docstring）后解 ``H delta = -b``，
        按流形更新 ``x <- x + delta``（角度 wrap）。收敛判据: 增量无穷范数
        < ``tol``。

        Args:
            n_iters: 最大迭代轮数。
            tol: 增量收敛阈值。

        Returns:
            优化后的总马氏代价 ``sum e^T Lambda e``。

        Raises:
            ValueError: 图为空。
        """
        if not self._init_poses:
            raise ValueError("pose graph has no nodes")
        ids = sorted(self._init_poses)
        index = {nid: k for k, nid in enumerate(ids)}
        n = len(ids)
        x = np.stack([self._init_poses[nid] for nid in ids]).reshape(-1)
        anchor = 3 * index[ids[0]]

        for _ in range(n_iters):
            poses = x.reshape(n, 3)
            H = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)
            for edge in self._edges:
                pi_ = poses[index[edge.i]]
                pj = poses[index[edge.j]]
                e = se2_relative_residual(pi_, pj, edge.z)
                J = se2_relative_jacobian(pi_, pj)  # (3, 6) = [J_i | J_j]
                bi, bj = 3 * index[edge.i], 3 * index[edge.j]
                H[bi : bi + 3, bi : bi + 3] += J[:, 0:3].T @ edge.info @ J[:, 0:3]
                H[bi : bi + 3, bj : bj + 3] += J[:, 0:3].T @ edge.info @ J[:, 3:6]
                H[bj : bj + 3, bi : bi + 3] += J[:, 3:6].T @ edge.info @ J[:, 0:3]
                H[bj : bj + 3, bj : bj + 3] += J[:, 3:6].T @ edge.info @ J[:, 3:6]
                b[bi : bi + 3] += J[:, 0:3].T @ edge.info @ e
                b[bj : bj + 3] += J[:, 3:6].T @ edge.info @ e
            H[anchor : anchor + 3, :] = 0.0  # gauge fixing: 锚定第一个节点
            H[:, anchor : anchor + 3] = 0.0
            H[anchor : anchor + 3, anchor : anchor + 3] = np.eye(3)
            b[anchor : anchor + 3] = 0.0
            delta = -np.linalg.solve(H, b)
            x = x + delta
            x[2::3] = wrap_angle(x[2::3])
            if float(np.max(np.abs(delta))) < tol:
                break

        poses = x.reshape(n, 3)
        self._optimized = {nid: poses[index[nid]].copy() for nid in ids}
        cost = 0.0
        for edge in self._edges:
            e = se2_relative_residual(
                poses[index[edge.i]], poses[index[edge.j]], edge.z
            )
            cost += float(e @ edge.info @ e)
        return cost

    def poses(self) -> dict[int, np.ndarray]:
        """优化后的节点位姿 ``{node_id: (x, y, theta)}``.

        Raises:
            RuntimeError: 尚未调用 :meth:`optimize`。
        """
        if self._optimized is None:
            raise RuntimeError("call optimize() before reading poses")
        return self._optimized

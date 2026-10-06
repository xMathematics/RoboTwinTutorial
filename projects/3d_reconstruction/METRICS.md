# 3D 重建统一测评指标教程（METRICS）

> 零基础基准撰写：不假设读者有机器人学/图形学背景，术语首现给中文通俗解释。
> 代码：[metrics.py](metrics.py)（全部指标的唯一定义处）｜ 测试：[tests/test_metrics.py](tests/test_metrics.py)
> 章节互引：[tutorials/3d_reconstruction 总览](../../tutorials/3d_reconstruction/README.md) ｜
> [第 01 章](../../tutorials/3d_reconstruction/01_3D重建问题与表示全景.md)（(1.2) 零水平集、(1.13) SDF、(1.14) TSDF 截断）｜
> [第 04 章 TSDF](../../tutorials/3d_reconstruction/04_RGB-D融合与TSDF.md)（(4.2) 噪声模型、(4.8) 融合）｜
> [第 05 章 表面提取](../../tutorials/3d_reconstruction/05_从体素到网格表面提取.md)（(5.1)/(5.2)）｜
> [第 06 章 SDF](../../tutorials/3d_reconstruction/06_学习式形状表示SDF与占据.md)（(6.1) Eikonal）

## 0. 为什么需要统一的指标

**① 问题场景**：本项目的重建管线有两条产出路径——单视角深度点云（`tsdf.depth_to_point_cloud`，
基线）与 TSDF 融合 + MT 网格（主管线）。"重建得好不好"必须用同一把尺子量才有意义：
如果每个模块各自手写一个距离函数、各自决定"用不用平方距离、阈值取多少"，同一数字
在不同模块含义不同，无法横向比较。

**② 解决方法**：所有指标只在一个地方定义——`projects/3d_reconstruction/metrics.py`
的 4 个纯函数；`demo.py` 与全部指标测试从这里取公式与实现，`tests/test_metrics.py`
中每个指标都与**朴素双重循环实现**交叉验证到 1e-10（与 projects/slam 的
`tests/test_metrics.py` 同一套路）。

**③ 为什么这四个指标够用**：一个重建表面的"好"可以拆成三个互相独立的侧面——
**整体偏差**（Chamfer）、**多余 vs 缺失**（accuracy / completion 的方向分解）、
**容差内的比例**（F-score@τ）。主管线（TSDF+MT）验证融合与提取（教程第 04/05 章）、
单视角基线验证多视角融合的价值，它们的主张恰好落在这几个侧面上。

**通俗概念铺垫**（后文反复出现）：

- **表面点集**：网格是连续表面，逐点指标先把两边都离散成点云——重建网格用
  面积加权采样（(M.4)），真值用解析 SDF 投影（`scene.sample_scene_surface`，
  精确到 1e-16 m）。教程语境：表面 = 标量场的零水平集（(1.2)），采样即零水平集
  上的均匀测度。
- **平方距离口径**：Chamfer / accuracy / completion 用**平方**距离的均值（DTU
  MATLAB 实现口径，单位 m²）；F-score 用**非平方**距离与阈值 τ 比较（单位 m）。
  平方口径对大误差更敏感（一个离谱点平方后淹没全部好点）——这正是需要 F-score
  补充的原因。

### 指标速查表

| 指标 | 一句话作用 | 代码函数（`metrics.py`） | 单位 | 公式 |
|------|-----------|--------------------------|------|------|
| 表面采样 | 评测的公共前置：网格 → 均匀点云 | `sample_mesh_surface(v, f, n, seed)` | m（点坐标） | (M.4) |
| 对称 Chamfer 距离 | 整体偏差（精度 + 完整度的合成） | `chamfer_distance(P, Q)` | m² | (M.1) |
| accuracy / completion | 多余程度 / 缺失程度（方向分解） | `accuracy_completion(est, gt)` | m² 各 | (M.2) |
| F-score@τ | 容差内的比例（对离群稳健） | `f_score(est, gt, tau)` | 无量纲 [0,1] | (M.3) |

统一运行方式（详见各节"如何运行"）：

```bash
cd projects/3d_reconstruction
python3 tests/test_metrics.py            # 全部指标测试（10 个，~0.1 s）
python3 tests/test_metrics.py chamfer    # 单点：只跑名字含 chamfer 的测试
pytest tests/test_metrics.py -v          # pytest 等价跑法
```

---

## 1. 表面采样 —— `sample_mesh_surface`

### 1.1 作用

不是"分数"，而是 (M.1)-(M.3) 的**公共前置**：重建产物是三角网格（教程 (5.1)/(5.2)
的输出），真值是连续表面；逐点指标要求两边都是点云。采样必须**均匀于面积**——
否则三角形密集处（MT 在高曲率区三角形更多）会被过度计权，指标测的是"网格化密度"
而非几何误差。真值侧的对应用 `scene.sample_scene_surface`（包围盒拒绝采样 +
解析最近点投影），它依赖每个基元的 `closest_point`（第 06 章 SDF 的解析载体，
(6.1) Eikonal 保证投影方向即梯度方向）。

### 1.2 如何计算

两步蒙特卡洛表面积分：

$$P(\text{选面 } t) = \frac{A_t}{\sum_{t'} A_{t'}},\qquad
p = (1-\sqrt{u})\,v_0 + \sqrt{u}\,(1-v)\,v_1 + \sqrt{u}\,v\,v_2 \tag{M.4}$$

其中 $A_t = \tfrac12\|(v_1-v_0)\times(v_2-v_0)\|$，$u, v \sim U(0,1)$ 独立。
带 $\sqrt{u}$ 的折叠把"重心坐标直角三角"摊成均匀分布（不带根号会向 v₀ 聚心）。
代码位置：`metrics.py::sample_mesh_surface`（第 1 步 `rng.choice(..., p=area2/total)`，
第 2 步重心坐标组合）。

教程互引：面积均匀采样对应"零水平集上的均匀测度"（(1.2)/(1.3) 的水平集几何）；
三角形顶点/法向来自 (5.1) 棱上插值与 (5.2) 梯度法向。

文献出处：蒙特卡洛表面积分的标准做法（教科书方法，如 Pharr et al., *Physically
Based Rendering* 第 3 版 §13.6 的三角形采样）。

### 1.3 健康值范围

| 场景（测试） | 健康表现 | 异常信号 |
|------|----------|----------|
| 面积比 0.5 : 2.45 的两块平行三角形 | 采样比例 = 面积份额 ± 0.05（实测 4000 点） | 比例偏向小三角形 → 折叠/选面加权丢失 |
| 同 seed 两次采样 | 逐位一致（确定性） | 不一致 → 用了全局随机态 |
| 全部采样点 | 落在三角形上（共面 + 重心坐标非负，1e-9） | 出现平面外点 → 重心坐标公式错 |
| 退化网格（总面积 0） | 显性 `ValueError` | 静默返回 nan → 早失败被跳过 |

### 1.4 如何运行

```bash
cd projects/3d_reconstruction
python3 tests/test_metrics.py sample
# 预期输出（节选）：
# PASS test_sample_mesh_area_weighting_and_determinism
# PASS test_sample_mesh_points_on_triangles
# 2/2 tests passed.
```

---

## 2. 对称 Chamfer 距离 —— `chamfer_distance`

### 2.1 作用

衡量重建表面与真值表面的**整体偏差**——demo 与测试中"重建到底准不准"的主数字。
它同时惩罚精度（重建点离真值多远）与完整度（真值点离重建多远）：只有 P→Q 方向会
奖励"重建点都贴着真值"却漏掉大半个物体（稀疏作弊）；只有 Q→P 方向会奖励"覆盖全"
却容忍大量离谱外点。教程语境：量化第 04 章融合 (4.8) + 第 05 章提取 (5.1)/(5.2)
管线的综合质量；学习式重建（第 06/07 章）的论文表格里它是最常见的一列。

### 2.2 如何计算

$$d_{CD}(P, Q) = \frac{1}{2}\left(\frac{1}{|P|}\sum_{p\in P}\min_{q\in Q}\|p-q\|^2
+ \frac{1}{|Q|}\sum_{q\in Q}\min_{p\in P}\|q-p\|^2\right) \tag{M.1}$$

代码位置：`metrics.py::chamfer_distance`（内部 `_pairwise_min_dist_sq` 分块计算
最近点平方距离，控制内存）。向量化实现与 O(n²) 双重循环朴素实现在
`tests/test_metrics.py::test_chamfer_matches_naive_o2` 交叉验证到 1e-10。

文献出处：Chamfer 距离作为形状匹配度量的起源——H. G. Barrow, J. M. Tenenbaum,
R. C. Bolles, H. C. Wolf, *"Parametric Correspondence and Chamfer Matching"*,
IJCAI 1977；平方均值口径沿用形状生成评测惯例（Tatarchenko et al., CVPR 2019）。

### 2.3 健康值范围（本项目实测，demo 确定性输出）

| 方法（demo.py，8192 评测点） | 实测 Chamfer | RMS 等效距离 | 说明 |
|------|------|------|------|
| TSDF 融合 + MT 网格 | **1.27–1.29e-4 m²** | ≈ 1.1 cm | 深度噪声（σ≈5 mm，(4.2)）+ 体素离散（25 mm）+ 像素对齐量化的合成 |
| 单视角深度点云（基线） | 2.71e-2 m² | ≈ 16.5 cm | 被 coverage 缺口主导——多视角融合价值的直接证据 |
| 单位球 MT 提取（无融合，`test_sphere_radius_deviation`） | mean\|r−1\| = 0.00045 | ≈ 0.5 mm | 提取器本身的离散化底限（voxel=0.075 m） |

异常信号：**Chamfer > 1e-3** ⇒ 幽灵内壁（未用 `extraction_field()` 权重掩码，
[DEBUG.md](DEBUG.md) §4 案例 1）；**> 1e-2** ⇒ 位姿/内参错或深度方向约定反。

### 2.4 如何运行

```bash
cd projects/3d_reconstruction
python3 demo.py
# 预期输出（表格末行；系统 python3 / llm_env 在打印末位内一致）：
# TSDF 融合 + MT 网格      0.000129   0.000129   0.000129   0.960
python3 tests/test_metrics.py chamfer
# 预期：3/3 tests passed.（含朴素交叉验证 1e-10 与解析值 t² 校验）
```

---

## 3. accuracy / completion —— `accuracy_completion`

### 3.1 作用

把 Chamfer 的两个单向项拆开说话：**accuracy 只看重建的多余程度**——重建了真值上
不存在的几何（噪声、飞点、幽灵面）会推高它，漏掉半个物体不影响；**completion 只看
缺失程度**——漏测（遮挡、视角不足）会推高它，多造垃圾不影响。单视角基线正是靠
completion 主导失败（背面全缺），TSDF 融合靠多视角把 completion 压下来——这两个
数字讲的是两个不同的故事，单一 Chamfer 会把它们混在一起。教程语境：DTU 多视图
立体重建评测（教程第 03 章 MVS 的标准协议）的标准指标。

### 3.2 如何计算

$$\mathrm{acc} = \frac{1}{|E|}\sum_{e\in E}\min_{g\in G}\|e-g\|^2,\qquad
\mathrm{comp} = \frac{1}{|G|}\sum_{g\in G}\min_{e\in E}\|g-e\|^2 \tag{M.2}$$

（E = 重建点集，G = 真值点集；平方距离，单位 m²。）代码位置：
`metrics.py::accuracy_completion`（与 (M.1) 共用 `_pairwise_min_dist_sq`，
朴素交叉验证到 1e-10）。

**方向怎么记**：acc 的求和跑在 **E**（估计）上——"每个估计点都有真值背书吗"；
comp 的求和跑在 **G**（真值）上——"每块真值都被重建到了吗"。

文献出处：DTU 评测协议——R. Jensen, A. Dahl, G. Vogiatzis, E. Tola,
H. Aanæs, *"Large Scale Multi-view Stereopsis Evaluation"*, CVPR 2014（§3 的
accuracy / completion 定义，平方距离口径沿用其官方 MATLAB 实现）。

### 3.3 健康值范围（本项目实测）

| 方法（demo.py） | 实测 accuracy | 实测 completion | 结论 |
|------|------|------|------|
| TSDF 融合 + MT 网格 | 1.27–1.29e-4 m² | 1.29e-4 m² | 两者同量级 = 无系统性偏倚 |
| 单视角深度点云 | **1.22e-4 m²** | **5.41e-2 m²**（RMS ≈ 23 cm） | acc 好而 comp 崩 = "看得见的都准，看不见的全缺" |

（单视角基线的 completion 是融合版的 ~420 倍——缺的就是单视角看不到的背面。
acc ≈ 1.2e-4 与融合版同量级说明深度测量本身是准的，缺的是覆盖。demo 双环境的
指标在打印末位（~2e-6 m²）内一致：BLAS 求和顺序的微小差异经最近邻并列选择放大，
属预期，同一环境内重跑则逐位可复现。）

异常信号：**acc ≫ comp** ⇒ 重建出了真值上没有的几何（查幽灵面/法向朝内）；
**comp ≫ acc** ⇒ 覆盖缺口（查视角数 / 轨道仰角环 / 遮挡）；**两者都大** ⇒ 几何
系统性错误（位姿、内参、截断带宽 μ）。

### 3.4 如何运行

```bash
cd projects/3d_reconstruction
python3 tests/test_metrics.py accuracy
# 预期输出（节选）：
# [metrics] acc 0.666122 / comp 0.320127 (naive cross-checked)
# PASS test_accuracy_completion_naive_cross_check
# PASS test_accuracy_completion_directional_semantics
# 2/2 tests passed.
python3 demo.py    # 表格中 Acc / Comp 两列即本指标的端到端实测
```

---

## 4. F-score@τ —— `f_score`

### 4.1 作用

以"距离多近算对"的工程判定回答**容差内比例**：重建点要么能用（≤ τ）要么不能用
（> τ）。(M.2) 的平方距离对离群点极敏感（一个 10 cm 飞点的平方 = 1e-2 m²，等于
六千个 1 mm 好点的贡献），F-score 把误差二值化后对离群稳健、也最贴近应用判断。
它是 DTU 与 Tanks and Temples 基准的主报告指标——教程第 07/08 章神经隐式与 3DGS
的重建质量表用的就是它，本项目的实现让教学管线能直接对齐那一套口径。

### 4.2 如何计算

$$P = \frac{\#\{e \in E : \min_g \|e-g\| \le \tau\}}{|E|},\quad
R = \frac{\#\{g \in G : \min_e \|g-e\| \le \tau\}}{|G|},\quad
F_\tau = \frac{2PR}{P+R}\ (\ P+R=0 \Rightarrow F_\tau = 0\ ) \tag{M.3}$$

代码位置：`metrics.py::f_score`（内部用平方距离 ≤ τ² 比较，免开方）。P 是查准率
（precision：重建点里多少比例算对），R 是查全率（recall：真值点里多少比例被覆盖）。

**τ 怎么选**：τ 是与传感器精度、应用需求绑定的工程量（DTU 惯用 5/10/20 mm；T&T
场景尺度用厘米级）。**必须与评测点云密度匹配**：τ < 平均采样间距时，完美重建也会
因"最近的对方点不在 τ 内"被扣分（demo 用 8192 点、间距 ~1.8 cm，故 τ=20 mm；
2000 点时间距 ~3.7 cm，τ 应放到 50 mm——[DEBUG.md](DEBUG.md) §4 案例 4）。

教程互引：τ 的物理意义对接 (4.2) 的深度噪声模型——τ 应显著大于 σ(z)（demo：
σ≈5 mm ≪ τ=20 mm，故融合结果的失败由几何而非测量噪声主导）；被比较的表面来自
(5.1)/(5.2) 的提取精度。

文献出处：DTU F-score 与 Tanks and Temples 基准——A. Knapitsch, T. Aanæs,
R. Jensen, R. Koch, *"A Benchmark and Comparison of Point Cloud Registration
Algorithms"*, IJCV 2021（Tanks and Temples 版本：ACM TOG 2017 同作者团队；
DTU F@5/10/20 mm 惯例见 DTU 评测网站及 Jensen et al. 2014 后续）。

### 4.3 健康值范围（本项目实测）

| 方法（demo.py，τ=20 mm） | 实测 F@20mm | 分量 | 结论 |
|------|------|------|------|
| TSDF 融合 + MT 网格 | **0.960** | P ≈ R ≈ 0.96 | 健康：误差分布集中在 τ 内 |
| 单视角深度点云 | **0.315** | P≈1 / R≈0.31 | "看得见的都算对、看不见的全缺"——融合价值的第二证据 |
| TUTORIAL §3.5 片段（2000 点，τ=50 mm） | 0.996 | — | 密度匹配后的上限参考 |

解析自检（`tests/test_metrics.py`）：est=gt → F=1；整体平移 > τ → F=0；
可手算的 80/100 合成场景 → P=R=0.8、F=0.8（1e-12）。

异常信号：**P 高 R 低** ⇒ 覆盖缺口（视角不足/遮挡，对照 demo 单视角基线）；
**R 高 P 低** ⇒ 重建含大量离表面垃圾（幽灵内壁、飞点）；**F 随评测点数下降** ⇒
密度-τ 失配（案例 4），不是几何退化。

### 4.4 如何运行

```bash
cd projects/3d_reconstruction
python3 tests/test_metrics.py f_score
# 预期输出（节选）：
# PASS test_f_score_perfect_when_identical
# PASS test_f_score_zero_when_shift_beyond_tau
# PASS test_f_score_synthetic_precision_recall
# 3/3 tests passed.
python3 demo.py    # 表格中 F@20mm 一列即端到端实测
```

---

## 5. 选型速查：什么时候看哪个

| 你想回答的问题 | 用哪个 | 理由 |
|----------------|--------|------|
| 管线改动后"整体变好还是变坏" | **Chamfer** (M.1) | 单数字、对称、对实现改动最敏感 |
| 变好/变坏的原因是覆盖还是噪声 | **Acc / Comp** (M.2) | 两个单向项把 Chamfer 拆成"多余 vs 缺失" |
| 对标论文表格 / 报告给第三方 | **F-score@τ** (M.3) | DTU / T&T 标准口径，对离群稳健 |
| 只关心某个工程容差（如抓取需要 ±1 cm） | **F-score@τ**，τ=工程容差 | 直接回答"多大比例的点可用" |
| 网格还没采点、要先跑通评测 | **sample_mesh_surface** (M.4) | 一切逐点指标的前置 |

一句话记忆：**Chamfer 回答"整体错了多少"，Acc/Comp 回答"错在多余还是缺失"，
F-score 回答"容差内有多少能用"**——三者共用 (M.4) 的采样前置。所有公式的唯一定义在
[metrics.py](metrics.py)，回归测试在 [tests/test_metrics.py](tests/test_metrics.py)
（10 个测试，全部确定性，双环境全绿）。

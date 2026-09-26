# 第 02 章｜多视图几何与 SfM

第 01 章把重建形式化为三级管线（(1.4)），第 ① 级"几何估计"要回答：手头只有一组无序照片，相机位姿与三维点从哪里来。本章给出经典答案——运动恢复结构（Structure-from-Motion, SfM）。对极约束、八点法与 BA 的**完整推导已在 SLAM 教程给出**（[第 05 章](../slam/05_视觉里程计-i特征点法.md)、[第 08 章](../slam/08_后端-ii图优化与-ba.md)），本章按跨主题规范**回引而不重推**（编号已逐条核实）；自推导的主菜是三角化（DLT 与中点法，02.1）与鲁棒核的 IRLS 推导（02.3），中间完整走一遍增量式 SfM 的流程与选型（02.2）。本章产出的位姿 $\{T_{cw}^{(k)}\}$ 与稀疏点云是第 03 章 MVS 稠密重建的直接输入。

## 02.1 两视图几何回顾与三角化

**① 问题场景** 手头：上百张无序照片 $\{I_k\}$（内参 $K$ 可预先标定——机器人相机是标配；未标定时走基础矩阵 $F$ 路线，(5.1) 已给出两种形式）。缺的：每台相机的位姿 $T_{cw}^{(k)}$ 与场景三维点 $\{\mathbf{X}_j\}$——一个都不知道。难点是又一座"鸡生蛋"：三角化点需要先知道相机位姿（投影方程里 $T$ 是已知量），而求解/标定相机又需要先有一批三维点（PnP 的输入是 3D–2D 对应）。想要的：一个把两个未知量"同时"解出来的可执行框架。

**② 解决方法** 增量式 SfM 的思路：不一次解全部未知，而是先攻**最小子问题**——选一对视角合适、匹配充分的照片，用对极几何在两视图上同时破局：对极约束 (5.1) 只需匹配点与 $K$、不需要任何三维点，先解出这两帧的相对运动（八点法 (5.2) → $E$ 分解 → 手性检验，完整推导见 SLAM 05.1–05.3，回引不重推）；位姿一旦落定，"蛋"的问题消失——逐点用**三角化**（triangulation）恢复三维点，得到种子地图；其余照片再逐张 PnP 注册（02.2）。本节推导主菜就是最后一环：**位姿已知后，如何从两视图恢复三维点**。

**③ 选型理由** 三角化为什么给两条解法： (a) DLT（线性齐次 + SVD）与八点法同一条依据链——无初值依赖、一次 SVD、可批量，是"从零开始"阶段的标配；代价是最小化**代数**误差，对噪声方向敏感（其解稍后被 BA 联合精化，SLAM 05.3 ③ 同款分工论证）。 (b) 中点法（midpoint method）把目标直接写成**几何**距离——两条光线的公垂线段取中点，误差量纲是米、几何直观、调试友好；代价是每次只处理一个点对的两视图，多视图需推广。选型：批量建点与初始化用 DLT（精度由 BA 兜底），几何诊断与教学理解用中点法。

**④ 理论依据** 三角化的系统表述见 Hartley & Zisserman, *Multiple View Geometry in Computer Vision*（三角化章）；齐次最小二乘 + 最小奇异向量的依据与 [SLAM 教程第 05 章](../slam/05_视觉里程计-i特征点法.md) 八点法 (5.2)→(5.3) 完全同源；中点法 = 两条三维直线公垂线的标准几何构造；对极约束 (5.1)、手性深度公式 (5.7)、单目尺度不确定 (5.9) 均回引 SLAM 第 05 章（编号已核实）。

**⑤ 完整推导**（本章主菜）

**设定。** 已知两帧位姿 $T_{cw}^{(1)}, T_{cw}^{(2)} \in SE(3)$（世界→相机）、内参 $K$，以及一对匹配的像素齐次坐标 $\tilde{\mathbf{u}}_1, \tilde{\mathbf{u}}_2$（末位补 1；来源是特征匹配，SLAM 第 04 章）。求：该对应的世界系三维点 $\mathbf{X} \in \mathbb{R}^3$（齐次 $\tilde{\mathbf{X}} = (\mathbf{X}^\top, 1)^\top$）。

**第一步（两个投影方程）。** 同一点在两帧下的投影（依据：SLAM 第 04 章针孔模型的齐次形式）：
$$
s_1\,\tilde{\mathbf{u}}_1 = K\,T_{cw}^{(1)}\,\tilde{\mathbf{X}}, \qquad s_2\,\tilde{\mathbf{u}}_2 = K\,T_{cw}^{(2)}\,\tilde{\mathbf{X}}, \qquad s_1, s_2 \in \mathbb{R}_{>0}
\tag{2.1}
$$
未知量核算：$\tilde{\mathbf{X}}$ 齐次（尺度等价）故有效未知 3 个，加 $s_1, s_2$ 共 5 个；每个 3 维向量方程中恰有一行只用来定义 $s_k$，故每帧提供 2 个独立约束，两帧合计 4 个约束解 3 个未知量——超定，噪声下无精确解，退而求最小二乘。

**第二步（消去深度标量）。** (2.1) 第一式两边同时左乘叉积矩阵 $\tilde{\mathbf{u}}_1^\wedge$（依据：向量与自身叉积为零——叉积的反对称性，$s_1\,\tilde{\mathbf{u}}_1^\wedge \tilde{\mathbf{u}}_1 = \mathbf{0}$，深度 $s_1$ 被消去）：
$$
\tilde{\mathbf{u}}_1^\wedge\, K\,T_{cw}^{(1)}\,\tilde{\mathbf{X}} = \mathbf{0}, \qquad \tilde{\mathbf{u}}_2^\wedge\, K\,T_{cw}^{(2)}\,\tilde{\mathbf{X}} = \mathbf{0}
\tag{2.2}
$$
未知量只剩 $\tilde{\mathbf{X}}$，且方程对它**线性齐次**。

**第三步（堆叠成齐次方程组，写出 $A$ 的行结构）。** 每个 $\tilde{\mathbf{u}}^\wedge K T$ 是 $3 \times 4$ 矩阵，两帧堆叠：
$$
A\,\tilde{\mathbf{X}} = \mathbf{0}, \qquad
A = \begin{bmatrix} \tilde{\mathbf{u}}_1^\wedge\, K\, T_{cw}^{(1)} \\[0.5ex] \tilde{\mathbf{u}}_2^\wedge\, K\, T_{cw}^{(2)} \end{bmatrix} \in \mathbb{R}^{6 \times 4}
\tag{2.3}
$$
行结构：上三行来自帧 1、下三行来自帧 2；每块秩为 2（依据：叉积矩阵 $\tilde{\mathbf{u}}^\wedge$ 的零空间是 $\mathrm{span}(\tilde{\mathbf{u}})$，秩为 2；右乘满秩 $K T$ 不改变秩）。一般位形下 $\mathrm{rank}(A) = 3$：$\tilde{\mathbf{X}} \neq \mathbf{0}$ 在零空间中故秩 $\le 3$；而两帧光心不同、两条光线不共面时，两块行空间的交一般恰好补成一维零空间 $\mathrm{span}(\tilde{\mathbf{X}})$（退化位形——两光心重合、或被测点落在基线连线上——时降秩，三角化失效，定性）。

**第四步（齐次最小二乘：SVD 取最小奇异向量）。** 噪声使 (2.3) 无精确非零解；由尺度等价取规范 $\|\tilde{\mathbf{X}}\| = 1$，最小化 $\|A\tilde{\mathbf{X}}\|^2$。拉格朗日函数 $\mathcal{L} = \tilde{\mathbf{X}}^\top A^\top A\, \tilde{\mathbf{X}} - \lambda(\tilde{\mathbf{X}}^\top\tilde{\mathbf{X}} - 1)$，对 $\tilde{\mathbf{X}}$ 求导置零（依据：$\partial(\tilde{\mathbf{X}}^\top M \tilde{\mathbf{X}})/\partial\tilde{\mathbf{X}} = 2M\tilde{\mathbf{X}}$，$M$ 对称；约束优化乘子法）得 $A^\top A\,\tilde{\mathbf{X}} = \lambda\,\tilde{\mathbf{X}}$；两边左乘 $\tilde{\mathbf{X}}^\top$ 知目标值 $= \lambda$，故最优解是 $A^\top A$ **最小**特征值的单位特征向量，即 $A$ 的**最小奇异向量**（依据：$A^\top A$ 的特征值 = $A$ 奇异值的平方——与八点法 (5.2)→(5.3) 同一条依据链）：
$$
\tilde{\mathbf{X}}^* = \mathbf{v}_4 \quad \big(A = U\Sigma V^\top \text{ 的最小奇异向量}\big), \qquad \hat{\mathbf{X}} = \tilde{\mathbf{X}}^*_{1:3}\, /\, \tilde{\mathbf{X}}^*_4
\tag{2.4}
$$
反齐次化要求 $\tilde{\mathbf{X}}^*_4 \neq 0$（点在有限远处）；再做手性检验：该点在两帧相机系的深度 $Z_1, Z_2 > 0$ 才接受（依据：(5.7) 的手性分析——$E$ 分解的四个候选全满足对极约束，"深度全正"是唯一物理判据）。备注：与 SLAM (5.8) 的 DLT 形式一致——那里在帧 1 相机系下用归一化坐标写 $A$，这里在世界系下用像素齐次坐标写，相差一个已知变换，解等价（依据：坐标共轭关系）。

**第五步（中点法：公垂线段推导）。** 换一条纯几何路线。相机光心的世界坐标：世界原点在相机 $k$ 系下为 $\mathbf{t}_{cw}^{(k)}$（依据：$T_{cw}(0,0,0,1)^\top$ 的平移分量）；反解令 $R_{cw}\mathbf{O}_k + \mathbf{t}_{cw} = \mathbf{0}$ 的点：
$$
\mathbf{O}_k = -\,R_{cw}^{(k)\top}\,\mathbf{t}_{cw}^{(k)}, \qquad k = 1, 2
\tag{2.5}
$$
（依据：$R_{cw}$ 正交，$R_{cw}^{-1} = R_{cw}^\top$；代入验证 $R_{cw}\mathbf{O}_k + \mathbf{t}_{cw} = \mathbf{0}$。）光线方向：取 $\mathbf{X} = \mathbf{O}_k + \alpha\,\mathbf{d}_k$ 代入 (2.1)，平移项归零（依据：上式），得 $R_{cw}\mathbf{d}_k \sim K^{-1}\tilde{\mathbf{u}}_k$——沿光线平移不改变投影方向（依据：线性性），故
$$
\mathbf{d}_k = R_{cw}^{(k)\top} K^{-1} \tilde{\mathbf{u}}_k, \qquad \|\mathbf{d}_k\| = 1\ \ (\text{归一化不改射线})
\tag{2.6}
$$
两条光线 $L_1: \mathbf{O}_1 + t\,\mathbf{d}_1$ 与 $L_2: \mathbf{O}_2 + s\,\mathbf{d}_2$ 因观测噪声一般**不相交**；求最近点对——最小化 $\phi(t, s) = \|t\,\mathbf{d}_1 - s\,\mathbf{d}_2 - \mathbf{b}\|^2$，其中 $\mathbf{b} = \mathbf{O}_2 - \mathbf{O}_1$（依据：$P_1 - P_2 = t\mathbf{d}_1 - s\mathbf{d}_2 - (\mathbf{O}_2 - \mathbf{O}_1)$）。求偏导置零（依据：二次函数极值的一阶条件）：
$$
\frac{\partial \phi}{\partial t} = 2\,(P_1 - P_2)\cdot\mathbf{d}_1 = 0, \qquad \frac{\partial \phi}{\partial s} = -2\,(P_1 - P_2)\cdot\mathbf{d}_2 = 0
\tag{2.7}
$$
几何读出：距离向量 $P_1 - P_2$ 同时垂直于两个方向——**公垂线段**（依据：一阶条件即正交条件）。代入 (2.6) 的单位方向、记 $c = \mathbf{d}_1^\top \mathbf{d}_2$，(2.7) 的分量形式为 $t - s\,c = \mathbf{b}^\top\mathbf{d}_1$、$s - t\,c = -\mathbf{b}^\top\mathbf{d}_2$（依据：逐项展开；$\partial\phi/\partial t = 2(t - sc - \mathbf{b}^\top\mathbf{d}_1)$），写成 $2\times 2$ 线性方程组并求逆（依据：克拉默法则）：
$$
\begin{bmatrix} 1 & -c \\ -c & 1 \end{bmatrix} \begin{bmatrix} t \\ s \end{bmatrix} = \begin{bmatrix} \mathbf{b}^\top\mathbf{d}_1 \\ -\mathbf{b}^\top\mathbf{d}_2 \end{bmatrix}
\;\Longrightarrow\;
t^* = \frac{\mathbf{b}^\top\mathbf{d}_1 - c\,\mathbf{b}^\top\mathbf{d}_2}{1 - c^2}, \quad s^* = \frac{c\,\mathbf{b}^\top\mathbf{d}_1 - \mathbf{b}^\top\mathbf{d}_2}{1 - c^2}
\tag{2.8}
$$
（可逆条件 $|c| < 1$：依据柯西–施瓦茨不等式，等号当且仅当两方向共线——平行光线退化，三角化失败，定性。）取中点：
$$
\hat{\mathbf{X}} = \tfrac{1}{2}\big(P_1 + P_2\big), \qquad P_1 = \mathbf{O}_1 + t^*\,\mathbf{d}_1, \quad P_2 = \mathbf{O}_2 + s^*\,\mathbf{d}_2
\tag{2.9}
$$
（依据：公垂线段两端点各在一条光线上，取中点即"真值到两光线距离相等"的对称估计——噪声对称时无偏（定性）；中点法名由此。）与 DLT 的关系：(2.3)–(2.4) 最小化**代数**误差（齐次坐标残差），(2.7)–(2.9) 最小化**几何**误差（公垂线长度）——与 ③ 的选型一致。

**第六步（齐次性 ⟹ 单目尺度不确定）。** 三角化 (2.3) 是齐次方程组：$A(\lambda\tilde{\mathbf{X}}) = \lambda A\tilde{\mathbf{X}} = \mathbf{0}$，单点只定到公共尺度。更强的不确定性在系统层面：把世界整体做相似变换 $g: \mathbf{X} \mapsto \lambda R_g\mathbf{X} + \mathbf{t}_g$，同时把每个相机的外参改写为复合映射"$g^{-1}$ 再进相机 $k$"，则每个点进相机后的坐标逐点不变（依据：$g^{-1} \circ g = \mathrm{id}$；复合映射线性块的整体缩放被 (2.1) 中的 $s_k$ 吸收——齐次投影对尺度不敏感，与 (5.9) 的两视图论证同源）：
$$
\big(\{\mathbf{X}_j\},\, \{T_{cw}^{(k)}\}\big) \;\longrightarrow\; \big(\{g(\mathbf{X}_j)\},\, \{T_{cw}^{(k)} \circ g^{-1}\}\big) \quad \text{产生完全相同的全部像素观测}
\tag{2.10}
$$
即单目重建只定到一个 7 自由度相似等价类（尺度 1 + 旋转 3 + 平移 3）——规范自由度（gauge freedom）；(5.9) 的 $(\mathbf{t}, \mathbf{X}) \to (\lambda\mathbf{t}, \lambda\mathbf{X})$ 是其两视图特例。出路：RGB-D/双目深度直接给出绝对尺度（第 04 章线）、IMU（SLAM 第 10 章）、或以初始化尺度为单位自洽（SLAM 05.3 ⑤ 同款结论）。

## 02.2 增量式 SfM 全流程

**① 问题场景** 手头：02.1 的两视图种子（一对初始位姿 + 初始点云）、其余 $K-2$ 张照片，以及"谁跟谁共视"的匹配线索。缺的：把其余照片逐一接入，同时让全部位姿与点云保持**全局一致**——逐帧独立求解会让误差各漂各的、从不互相咬合。想要的：一个可生长、每步可验证、误差可回头修正的重建循环。

**② 解决方法** 增量式 SfM（incremental SfM）的五步循环（集大成者为 COLMAP，④）：
1. **scene graph（场景图）构建**：特征提取与匹配（回引 SLAM 第 04 章）后，对每个候选图像对做**几何验证**（geometric verification）——估计 $F$ 或 $H$ 并统计内点，通过验证的对才连边：
$$
G = (\mathcal{V}, \mathcal{E}), \qquad \mathcal{V} = \{I_k\}, \qquad (I_i, I_j) \in \mathcal{E} \iff (I_i, I_j) \text{ 几何验证通过}
\tag{2.11}
$$
验证把"外观相似"升级为"几何一致"，并按单应内点占比区分一般场景 / 平面 / 全景（panoramic，纯旋转）等退化类型（COLMAP §4.1），为下一步初始化选型服务。
2. **两视图初始化**：从 $G$ 中选视角对足够分散、共视充分的对（COLMAP 的论证：从图像图的"稠密位置"初始化冗余度高、重建更稳；初始化选错，重建可能无法恢复）；用 SLAM 05.1–05.3 的对极管线（(5.1)–(5.9)）解相对位姿，再用 02.1 的三角化建种子点云。纯旋转对不可初始化——(5.1) 退化为恒等约束（SLAM 05.1 ⑤ 已推）。
3. **PnP 注册新视图**：地图在手，新照片与地图做 2D–3D 匹配解 PnP（Perspective-n-Point）得位姿——完整推导在 [SLAM 教程 05.4](../slam/05_视觉里程计-i特征点法.md)：重投影误差 (5.10)、雅可比 (5.11)、Gauss-Newton (5.12)，RANSAC 兜底误匹配；注册质量用内点率检验，不达标即拒注册——"可验证性"的来源。
4. **三角化扩充**：新注册视图带来新的观察对，用 02.1 的 DLT (2.3)–(2.4) 为更多对应三角化出新点（COLMAP 强调这同时提高已有点的冗余度）。
5. **局部/全局 BA 交替 + 外点滤除**：每注册若干视图做一次**局部 BA**（只优化与当前视图共视的位姿与点，快）；周期性与收尾做**全局 BA**（目标函数见 ⑤）。外点滤除：重投影误差过大的观测剔除、三角化视角过小的点剔除，随后 re-triangulation——COLMAP 的迭代 BA 策略，用于缓解漂移（其论文 §4 第四项贡献）。循环直到没有可注册的图像。COLMAP 同时警示：不持续精化，"SfM 通常很快漂移到不可恢复的状态"（§2.2）。

**③ 选型理由**（重点）增量式 vs 全局式 vs 层级式：
- **增量式（incremental）**：如上逐张生长。优点：每步有可检验的中间产物（PnP 内点率、重投影残差），错误能被发现、可回滚——**鲁棒**；单张失败不拖垮全局。代价：BA 被反复执行、计算冗余大；误差随注册顺序向后累积（漂移，drift）；对初始化对敏感。
- **全局式（global）**：先从 scene graph 全部边的两视图关系解全局旋转平均（rotation averaging）与平移，再一次性三角化 + 一次 BA。优点：BA 次数少、效率高；误差分布均匀、无顺序漂移。代价：全部两视图关系（含误匹配）一起进入"平均"步，一处系统性误匹配污染全局解；没有可验证的中间产物，出错难定位。
- **层级式（hierarchical）**：按相似度聚类 → 子集各自重建 → 自底向上合并。优点：并行友好、子问题小；代价：合并阶段要处理子图间的尺度与漂移不一致，策略复杂。
- **为何工程默认与本书主线取增量式**（COLMAP 论文的论证）：无序照片重建的核心瓶颈是鲁棒性与完备性（其摘要点名 "robustness, accuracy, completeness, and scalability remain the key problems"）——增量式"边生长边验证"的结构正打在这里；其代价（慢、漂移）由局部/全局 BA 交替、下一最优视图选择（next-best view selection）与迭代 re-triangulation 缓解（§4 四项贡献）。机器人场景的照片有采集顺序、位姿还可来自 SLAM/里程计先验，此时全局式或直接配准（第 04 章线）可能更合适——选型跟着数据形态走。

**④ 理论依据** Schönberger & Fischer, *"Structure-from-Motion Revisited"*, CVPR 2016（COLMAP 的 SfM 半边；[本地 PDF](../../papers/3d_reconstruction/classics/COLMAP-SfM_CVPR2016_SchoenbergerFischer.pdf)，其管线、初始化论证与四项贡献已通读核实）；对极几何与 PnP：SLAM 教程第 05 章；BA 与图优化：SLAM 教程第 08 章（Triggs et al. 1999 综述）。

**⑤ 完整推导**（BA 目标函数在本章的形式，及其与 SLAM 第 08 章的对应）

增量式 SfM 每轮精化的目标是：
$$
\min_{\{T_{cw}^{(k)}\},\, \{\mathbf{X}_j\}} \;\; \sum_{(k,j) \in \mathcal{O}} \rho\!\left( \big\|\, \mathbf{u}_{kj} - \pi\big(K\,T_{cw}^{(k)}\,\tilde{\mathbf{X}}_j\big) \,\big\|_{\Lambda_{kj}}^2 \right)
\tag{2.12}
$$
$\mathcal{O}$ = scene graph 中已验证的（视图, 点）观测对；$\Lambda_{kj} = \Sigma_{kj}^{-1}$ 为信息加权；$\rho$ = 鲁棒核（02.3 推导；先取 $\rho \equiv x$ 即无核情形）。**这就是 SLAM 第 08 章的 BA，不是"类似"**——逐项对照其 (8.1)、(8.2)：
- **变量一致**：$\{T_{cw}^{(k)}\}$（位姿节点）↔ $\{\mathbf{T}_i\}$；$\{\mathbf{X}_j\}$（路标节点）↔ $\{\mathbf{p}_j\}$——同一位姿集 + 路标集，仅记号不同；
- **观测一致**：$\mathbf{u}_{kj}$ ↔ $\mathbf{z}_{ij}$，同为像素观测；$\pi$ 同为针孔投影（SLAM 第 04 章）；
- **目标一致**：(2.12) 在 $\rho \equiv x$ 时与 (8.2) 的信息加权平方和逐字相同；
- **残差方向**差一个整体符号（$\mathbf{u} - \pi$ vs (8.1) 的 $\mathbf{z} - \pi(\mathbf{T}\mathbf{p})$）——雅可比整体变号、最优增量不变（SLAM ch05 (5.10) 的旁注同款说明）。
求解同一套：残差一阶展开 (8.3) → **正规方程 (8.5)** $(J^\top\Sigma^{-1}J)\,\Delta = -J^\top\Sigma^{-1}\mathbf{e}$（LM 阻尼 (8.6)）；雅可比**回引不重推**——位姿块 = 左扰动链式法则 (8.10)，其单点版本 = PnP 雅可比 (5.11)，路标块 $\partial\pi/\partial\mathbf{X}$ 由投影的商法则直接写出；稀疏结构与 Schur 补 (8.11)–(8.15) 原样适用——COLMAP 的全局 BA 正是"先解约简相机系统、再回代更新点"的 Schur 补路径（其论文 §2.2）。带核时唯一改动是信息矩阵换为加权版本 $\Sigma^{-1} \to W\Sigma^{-1}$（02.3 ⑤）。一句话收束：**SfM 的 BA 与 SLAM 后端的 BA 是同一个最小二乘**——变量一致、观测一致；SLAM 逐帧在线地解它（滑窗），SfM 离线批量地解它（全局），区别只在数据组织，不在数学。

## 02.3 鲁棒核：外点降权与 IRLS

**① 问题场景** 目标函数 (2.12) 的平方增长对外点是灾难。量级感（定性）：内点残差 ~1 px，平方后贡献 ~1；一条误匹配残差达数百 px，平方后贡献 ~$10^4$–$10^5$——它的梯度足以把整批内点的均衡解拖偏。SfM 的外点来源比 SLAM 更棘手：无序照片集没有时间连续性先验、重复纹理与动态物体制造系统性误匹配、scene graph 验证只能滤掉大部分而非全部。SLAM 08.2 ③ 已预警"误匹配多时小残差假设退化"——在 SfM 里这个预警是常态。想要的：让大残差观测"少说话"、又不至于让整组解崩掉的损失函数。

**② 解决方法** M-估计（M-estimator）：把平方损失换成增长**次线性**的鲁棒核（robust kernel / loss）$\rho(e)$，目标 $\min_\theta \sum_i \rho(e_i(\theta))$；求解用迭代重加权最小二乘（iteratively reweighted least squares, IRLS）：反复"按当前残差定权重 → 解加权正规方程"。常用核：Huber 与 Tukey（⑤ 给出定义与权重）。

**③ 选型理由** 四个候选对比： 无核——最快，但 ① 的量级分析说明 SfM 不可用； 截断损失（超阈值记 0）——拒绝最狠，但零梯度区间让优化器"看不见"外点、非凸更严重； Huber——小残差二次、大残差线性：**凸**、收敛稳，外点降权但不清零（极端外点仍留有杠杆）； Tukey（biweight）——大残差权重趋于 0：**彻底拒绝**爆炸外点，代价是非凸、依赖好初值。工程组合：增量框架正好逐轮提供初值——早期用 Huber 稳收敛、后期用 Tukey 清残渣（定性惯例）；COLMAP 的 BA 目标 (2.12) 即带 $\rho_j$（其论文 Eq.(1)，已核实）。为何 SfM **必须**用核：外点比例高是常态而非例外，无核 BA 的解由外点主导——这是正确性问题，不是精度微调。

**④ 理论依据** Huber, *"Robust Estimation of a Location Parameter"*, Ann. Math. Statist. 1964（M-估计奠基）；稳健统计与 IRLS 的标准表述见 Huber & Ronchetti, *Robust Statistics*, 2nd ed.；COLMAP 论文 Eq.(1) 的损失函数 $\rho_j$（本地 PDF 已核实）。

**⑤ 完整推导**（IRLS：鲁棒 M-估计的一阶条件 = 加权最小二乘）

设标量残差 $e_i(\theta)$，目标 $E(\theta) = \sum_i \rho\big(e_i(\theta)\big)$。一阶条件（依据：链式法则 + 极小点梯度为零）：
$$
\mathbf{0} = \nabla_\theta E = \sum_i \rho'(e_i)\,\nabla_\theta e_i
$$
做恒等变形——在 $e_i \neq 0$ 处定义权重 $w_i = \rho'(e_i)/e_i$，即 $\rho'(e_i) = w_i\, e_i$（$e_i = 0$ 处按连续性取极限 $w_i = \rho''(0)$，依据：$\rho$ 二阶连续）。代回一阶条件：
$$
\mathbf{0} = \sum_i w_i\, e_i\, \nabla_\theta e_i \;=\; \tfrac{1}{2}\,\nabla_\theta \Big( \sum_i w_i\, e_i^2 \Big)
\tag{2.13}
$$
右端正是**加权最小二乘** $\min_\theta \sum_i w_i e_i^2$ 的一阶条件（依据：复合函数求导，$\nabla_\theta \sum_i w_i e_i^2 = 2\sum_i w_i e_i \nabla_\theta e_i$）。结论：固定 $\{w_i\}$ 时，鲁棒目标的最优解与加权最小二乘的解**相同**——"解鲁棒问题"于是变成迭代：用当前残差算 $w_i$ → 解加权正规方程 → 更新残差，即 IRLS。

**接到 BA 上。** 把 $w_i$ 并入信息矩阵（依据：对角加权与信息加权的复合，块对角结构不变）：SLAM 正规方程 (8.5) 中的 $\Sigma^{-1}$ 替换为 $W\Sigma^{-1}$（$W = \mathrm{blkdiag}(w_i)$）：
$$
\big(J^\top\, W\, \Sigma^{-1} J\big)\,\Delta = -\,J^\top\, W\, \Sigma^{-1}\,\mathbf{e}
\tag{2.14}
$$
工程实现（g2o/Ceres 的 robust kernel）就是这一处对角缩放，其余流水线零改动；每轮重线性化时用**新**残差重算 $W$。

**两个核的定义与权重。** Huber（阈值 $\delta$）：
$$
\rho_H(e) = \begin{cases} e^2/2, & |e| \le \delta \\ \delta|e| - \delta^2/2, & |e| > \delta \end{cases}
\;\;\Longrightarrow\;\;
w_H(e) = \frac{\rho_H'(e)}{e} = \min\!\Big(1,\ \frac{\delta}{|e|}\Big)
\tag{2.15}
$$
（依据：分段求导——内段 $\rho' = e$ 给 $w = 1$；外段 $\rho' = \delta\,\mathrm{sign}(e)$ 给 $w = \delta/|e|$。残差越大权重越小，但只按 $1/|e|$ 衰减——**降权而不清零**。）Tukey（biweight）：
$$
\rho_T(e) = \begin{cases} \dfrac{\delta^2}{6}\Big[1 - \big(1 - e^2/\delta^2\big)^3\Big], & |e| \le \delta \\[1ex] \delta^2/6, & |e| > \delta \end{cases}
\;\;\Longrightarrow\;\;
w_T(e) = \begin{cases} \big(1 - e^2/\delta^2\big)^2, & |e| \le \delta \\ 0, & |e| > \delta \end{cases}
\tag{2.16}
$$
（依据：链式法则求导，内段 $\rho_T'(e) = e\,(1 - e^2/\delta^2)^2$——权重随 $|e| \to \delta$ 平滑归零；外段 $\rho_T$ 为常数、导数为零，权重恰为 0——**彻底拒绝**。$\rho_T$ 单调有界但非凸，③ 的代价来由。）

**"降权外点"的读法与光束聚合。** BA 之名由来：每条观测是从光心穿过像素打到路标的一根光束（bundle of light rays），(2.12) 让所有光束在各自路标处聚合一致。权重 $w_i$ 决定每根光束在聚合中的话语权：内点 $w \approx 1$，外点 $w \to 0$——外点没有消失，只是失去了选票。回到 ① 的量级：Huber 把 $10^4$ 量级的外点贡献压到线性段（$O(\delta \cdot 10^2)$ 量级），Tukey 直接归零——方程组的解重新由内点多数决。（延伸一句：COLMAP 在进 BA 之前还有 scene graph 的多模型几何验证——$F/H/E$ 判定 + 内点统计 + 全景/平面/水印对标记——把大部分外点物理剔除出 (2.11) 的图；鲁棒核负责兜住残余，两层防御互为分工。）

## 本章要点

- 两视图三角化 DLT (2.1)–(2.4)：投影方程 (2.1) → 叉积消深度 (2.2) → 6×4 齐次方程组 (2.3) → SVD 最小奇异向量 (2.4)；依据与八点法 (5.2)→(5.3) 同源（齐次最小二乘）。
- 中点法 (2.5)–(2.9)：光心 (2.5)、光线方向 (2.6)、公垂线一阶条件 (2.7)、最近点对解析解 (2.8)、中点 (2.9)；最小化几何误差，与 DLT 的代数误差互补。
- 单目尺度不确定 (2.10)：整体相似变换下观测逐点不变——重建只定到 7 自由度相似等价类；(5.9) 是其两视图特例。
- scene graph (2.11)：几何验证通过的对才连边；全景/平面等退化类型在验证阶段标记。
- 增量式 SfM = scene graph → 两视图初始化 → PnP 注册（回引 (5.10)–(5.12)）→ 三角化 → 局部/全局 BA 交替 + 外点滤除，循环生长；取增量式的根由是鲁棒与可验证（COLMAP 论证）。
- BA 目标 (2.12) 与 SLAM (8.1)/(8.2) 逐项同一：变量一致、观测一致、目标一致；求解同用正规方程 (8.5) 与 LM (8.6)，雅可比回引 (8.10)/(5.11)，Schur 补 (8.15) 原样适用——SfM 的 BA 与 SLAM 后端 BA 是同一个最小二乘。
- 鲁棒核 IRLS (2.13)–(2.16)：$\rho'(e) = w(e)\,e$ 把 M-估计一阶条件等价为加权最小二乘 (2.14)；Huber (2.15) 降权不清零，Tukey (2.16) 平滑归零。

## 配套阅读

- 主读论文：[COLMAP-SfM（Schönberger & Fischer, CVPR 2016）](../../papers/3d_reconstruction/classics/COLMAP-SfM_CVPR2016_SchoenbergerFischer.pdf)——scene graph 增强、下一最优视图选择、鲁棒三角化、迭代 BA/外点滤除四项贡献的原文论证。
- 相邻章节：[第 01 章｜3D 重建问题与表示全景](./01_3D重建问题与表示全景.md)（问题形式化与三级管线 (1.4)）｜第 03 章（多视图立体重建 MVS：本章位姿与稀疏点云的稠密化，架构见 [README](./README.md)）。
- 跨主题回引：[SLAM 教程第 05 章](../slam/05_视觉里程计-i特征点法.md)——对极约束 (5.1)、八点法 (5.2)/(5.3)、$E$ 分解与手性检验 (5.5)–(5.7)、DLT 三角化 (5.8)、尺度不确定 (5.9)、PnP (5.10)–(5.12)；[SLAM 教程第 08 章](../slam/08_后端-ii图优化与-ba.md)——BA 残差 (8.1)、目标函数 (8.2)、正规方程 (8.5)、LM (8.6)、左扰动雅可比 (8.10)、Schur 补 (8.11)–(8.15)。
- 教材：Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.（三角化、自标定章）；Triggs et al., *"Bundle Adjustment — A Modern Synthesis"*, ICCV 1999；Huber & Ronchetti, *Robust Statistics*, 2nd ed.。

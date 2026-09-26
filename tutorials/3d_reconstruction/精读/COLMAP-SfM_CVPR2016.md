# 论文精读｜COLMAP-SfM：Structure-from-Motion Revisited（CVPR 2016）

> **PDF**：[papers/3d_reconstruction/classics/COLMAP-SfM_CVPR2016_SchoenbergerFischer.pdf](../../../papers/3d_reconstruction/classics/COLMAP-SfM_CVPR2016_SchoenbergerFischer.pdf) ｜ **教程**：[第 02 章｜多视图几何与 SfM](../02_多视图几何与SfM.md) ｜ **代码**：[projects/slam/epipolar/](../../../projects/slam/epipolar/)（八点法 / PnP / 三角化，与本文前端几何同源）

## 1. 论文信息与一句话贡献

- **题目**：Structure-from-Motion Revisited
- **作者**：Johannes L. Schönberger（UNC Chapel Hill / ETH Zürich）、Jan-Michael Frahm（UNC Chapel Hill）
- **发表**：CVPR 2016（本仓库收录 PDF 共 10 页，全部式号已逐条对照原文）；开源实现即 COLMAP
- **一句话贡献**：把增量式 SfM（incremental Structure-from-Motion）的每个组件逐一重造——多模型几何验证的场景图（scene graph）增强、不确定度驱动的下一最优视图选择、RANSAC 化的鲁棒多视图三角化、迭代式 BA / 重三角化 / 外点滤除、面向稠密照片集的冗余视图分组 BA——在 17 个数据集（144,953 张无序网络照片）上同时刷新鲁棒性、完备性（completeness）与精度，且效率不降。

## 2. 问题与动机

**要解决什么**：无序照片集合的通用 SfM 重建。当时增量式系统（Bundler、VisualSFM）虽是最主流策略，但摘要点名四大未解问题：**鲁棒性（robustness）、精度（accuracy）、完备性（completeness）、可扩展性（scalability）**。

**两个失败根源**（Sec. 3，本文全部贡献都对着它们设计）：
1. **对应搜索不完整**：场景图缺边、缺冗余——匹配只看外观相似，几何验证粗糙，图既不连通也不可靠；
2. **重建阶段自我强化失败**：图像注册与三角化是**共生关系**（symbiotic）——图像只能注册到已有结构上，结构只能从已注册图像三角化（Sec. 3 引 [64]）。一步坏决策（注册错一张图、三角化坏一个点）会级联放大成整段漂移甚至模型断裂。

**设计哲学**：既然"每一步都依赖前一步的质量"，那就把每一步都做成**不确定度感知 + 可验证 + 可迭代修复**的——这正是把 SfM 从"管线"变成"系统"的关键一跃。教程第 02 章选型论证（增量式 vs 全局式 vs 层级式）正是建立在这篇论文的动机与实验之上。

## 3. 方法总览

```
无序图像集 I = {I_i}
   │  特征提取/匹配（SIFT；回引 SLAM 第 04 章）
   ▼
【对应搜索 Correspondence Search】
   匹配 → 几何验证（§4.1）：F 内点数阈值 + H/E 内点占比 + 中位三角化角 α_m
        + 水印/时间戳(WTF) 相似变换检验 → 带标注的场景图（边 = 验证通过的对，
        标签 ∈ {general, panoramic, planar} + 最大支持模型内点）
   ▼
【增量重建 Incremental Reconstruction（§4.2–4.4）】
   初始化：选非全景、最好已标定的种子对（两视图对极管线，SLAM 05.1–05.3）
   循环：
     ① 下一最优视图选择（§4.2）：多分辨率格子评分，可见点数 × 分布均匀性
     ② 图像注册：2D-3D 对应解 PnP（SLAM (5.10)–(5.12)，RANSAC 兜底）
     ③ 三角化（§4.3）：特征轨道上的 RANSAC 采样三角化（Eq.(2)–(5)），递归多轨道恢复
     ④ 局部 BA（最连通图像集）→ 模型增长一定百分比后全局 BA（Eq.(1)）
     ⑤ 过滤 + 重三角化 RT + 再 BA，迭代直至收敛（§4.4）
   ▼
【稠密集优化（§4.5）】冗余视图挖掘：高重叠图像分组，组内坍缩为单相机（Eq.(6)–(7)）
```

## 4. 关键公式推导（式号按论文 PDF 原文；论文未编号的判据式注明出处小节）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $\mathcal{I} = \{I_i \mid i=1..N_I\}$ | 输入图像集（Sec. 2.1） |
| $\mathbf{x}_j \in \mathbb{R}^2$，$\mathbf{f}_j$ | 局部特征位置与外观描述子（Sec. 2.1） |
| $\mathbf{P}_n = [\mathbf{R}^T \; -\mathbf{R}^T\mathbf{t}] \in SE(3)$ | 第 $n$ 个相机位姿（世界→相机），$\mathbf{R}\in SO(3)$，$\mathbf{t}\in\mathbb{R}^3$（Sec. 4.3） |
| $\bar{\mathbf{x}}_n \in \mathbb{R}^2$ | 归一化图像观测（Sec. 4.3） |
| $\mathcal{T} = \{\mathcal{T}_n \mid n = 1..N_T\}$ | 特征轨道（feature track），内点比例 $\epsilon$ 先验未知（Sec. 4.3） |
| $\mathbf{X}_{ab} \sim \tau(\cdot)$ | 由两视图 $a\neq b$ 三角化的点；$\tau$ 取 DLT 法（Sec. 4.3，Eq.(2)） |
| $N_F, N_H, N_E, N_S$ | 基础矩阵 / 单应 / 本征矩阵 / 边界相似变换的内点数（Sec. 4.1） |
| $\epsilon_{HF}, \epsilon_{EF}, \epsilon_{SF}, \epsilon_{SE}$ | 上述内点比的判定阈值（Sec. 4.1） |
| $\alpha$，$\alpha_m$ | 三角化角；轨道内点三角化角的**中位数**（Sec. 4.1 / 4.3） |
| $\pi(\cdot)$，$\rho_j$ | 投影函数与鲁棒损失（Sec. 2.2，Eq.(1)） |
| $\mathbf{v}_i \in \{0,1\}^{N_X}$ | 图像 $i$ 的二值可见性向量（Sec. 4.5，Eq.(6)） |
| $\mathbf{G}_r \in SE(3)$，$\mathbf{P}_c$ | 组 $r$ 的外参（共享、被优化）与组内图像位姿（固定）（Sec. 4.5，Eq.(7)） |

### 4.1 场景图增强：多模型几何验证的逐条判据（Sec. 4.1，判据按原文顺序）

论文不满足于"估计一个 F 就连边"，而是给每条边打上**模型类型标签**。逐条判据（每条都是论文原文的判定条件，形式化写出）：

1. **几何验证底线（匹配数阈值）**：估计基础矩阵 $\mathrm{F}$，若内点数 $N_F$ 达到阈值（"at least $N_F$ inliers are found"），该图像对才算几何验证通过——否则不进场景图。
2. **一般场景判据（单应内点占比）**：对同一对再统计单应内点数 $N_H$；若
   $$\frac{N_H}{N_F} < \epsilon_{HF}$$
   则判定"运动相机 + 一般场景"。依据（论文原文）：该比值近似 GRIC 式模型选择——单应能解释（$N_H/N_F$ 大）意味着平面场景或纯旋转；解释不了才是一般三维运动。
3. **标定正确性判据（本征矩阵内点占比）**：图像已标定时再估本征矩阵 $\mathrm{E}$ 的内点数 $N_E$；若
   $$\frac{N_E}{N_F} > \epsilon_{EF}$$
   则认为标定正确。依据：$\mathrm{E}$ 受 $K$ 约束、自由度比 $\mathrm{F}$ 少，标定正确时 $\mathrm{E}$ 的解释力不弱于 $\mathrm{F}$。
4. **退化位形判据（中位三角化角）**：在"标定正确且 $N_H/N_F < \epsilon_{HF}$"下，分解 $\mathrm{E}$、对内点对应三角化，取三角化角**中位数** $\alpha_m$ 区分纯旋转（panoramic，全景）与平面场景（planar）。依据：纯旋转下两视线相交于任意远处/无基线，三角化角趋于 0（4.3 的角度-深度不确定性分析给出定量解释）；平面场景则单应本就是精确模型、角度分布正常。全景对**不用于初始化、也永不参与三角化**——双保险防退化。
5. **水印/时间戳判据（WTF）**：在图像**边界区域**估计相似变换，内点数 $N_S$；若
   $$\frac{N_S}{N_F} > \epsilon_{SF} \;\vee\; \frac{N_S}{N_E} > \epsilon_{SE}$$
   判为 WTF（watermarks/timestamps/frames）对，**不插入**场景图。依据：水印是图像内容上的公共"贴片"，相似变换（旋转+各向同性缩放+平移）即可对齐——与场景几何无关。
6. **边标签**：通过验证的对按最大支持模型打标签（$N_H$、$N_E$ 或 $N_F$ 中最大者对应 general / panoramic / planar 类型），供初始化选型（非全景、优先已标定）与三角化（跳过全景对）使用。

### 4.2 下一最优视图选择：可见点数 × 分布均匀性的金字塔评分（Sec. 4.2，论文以文字定义，此处形式化）

**候选集**：尚未注册、且至少可见 $N_t > 0$ 个已三角化点的图像（论文原文定义）。直觉：完全看不到已有结构的图像注册必失败——先保证"能注"，再从中挑"值得注"。

**评分机制**（论文原文的完整形式化）。把候选图像 $i$ 划分为 $L$ 层分辨率金字塔：第 $l$ 层是 $K_l \times K_l$ 网格，$K_l = 2^l$；每个格子两态 empty / full；重建中当某个已三角化点**首次**落入空格时，该格转 full 并给图像加分：

$$
S_i \;=\; \sum_{l=1}^{L} w_l\, n_l(i), \qquad n_l(i) = \#\{\text{第 } l \text{ 层已转 full 的格子}\}, \qquad w_l = K_l^2
\tag{4.1}
$$

（(4.1) 是对论文文字描述的忠实改写：论文原文为 "the score is accumulated over all levels with a resolution-dependent weight $w_l = K_l^2$"；每格只加一次分是原文 "cells only contribute to the overall score once"。）行为读法：

- **点数**：可见点越多，转 full 的格子越多，$S_i$ 越大——对应"最多三角化点"经典策略 [52]；
- **均匀性**：同量点若聚在一角，只填满少数格子，$S_i$ 低——对应 Lepetit et al. [34] 的实验结论"PnP 精度取决于观测数量与其在图像中的分布"，均匀分布还能稳定自标定内参 [41]；
- **权重 $w_l = K_l^2$ 的直觉**（作者未推证，属解读）：细层格子面积按 $K_l^2$ 缩小，按格子总数加权使各层贡献量级可比——粗层管"覆盖"、细层管"均匀"。

**为什么不用协方差传播**（论文对 Haner et al. [24] 的取代论证）：每一步都要对**每个候选**计算并分析协方差，代价不可行；多分辨率格子评分是对"不确定度驱动"的廉价近似——可见点数目（重方差的量）与分布（方向条件数）正是 PnP 协方差两大决定因素的代理变量。论文 Fig. 3 给出量化示例：同为 80/200 个点，聚簇分布得分 66/146，均匀分布得分 80/200。Sec. 5 的 Quad 实验证明该评分带来更准的注册顺序（注册图像交并比收敛更快、相机位置误差更低）。

### 4.3 鲁棒三角化：RANSAC 采样三角化与三角化角检验（Sec. 4.3，Eq.(2)–(5)）

**问题形态**：特征轨道由两视图对应拼接而成，常因误匹配把多个独立点的轨道错并成一条——四条等长轨道错并一条，外点率即 75%。Bundler 的穷举两两三角化既慢又无法拆分错并轨道。论文把"从一条脏轨道中恢复所有独立点"形式化为 RANSAC：

**模型假设（Eq.(2)）**：轨道 $\mathcal{T} = \{\mathcal{T}_n\}$ 中任取两个测量 $\mathcal{T}_a, \mathcal{T}_b$（含归一化观测 $\bar{\mathbf{x}}$ 与位姿 $\mathbf{P}$），用任意三角化法 $\tau$（本文取 DLT，依据：与 [SLAM 教程 (5.8)](../../slam/05_视觉里程计-i特征点法.md)、[教程第 02 章 (2.3)–(2.4)](../02_多视图几何与SfM.md) 同一条齐次最小二乘链）恢复模型点：

$$
\mathbf{X}_{ab} \sim \tau\big(\bar{\mathbf{x}}_a,\ \bar{\mathbf{x}}_b,\ \mathbf{P}_a,\ \mathbf{P}_b\big), \qquad a \neq b
\tag{2}
$$

**良置条件一：三角化角充分（Eq.(3)）**。模型点须与两相机光心构成足够大的对顶角：

$$
\cos\alpha \;=\; \frac{\mathbf{t}_a - \mathbf{X}_{ab}}{\|\mathbf{t}_a - \mathbf{X}_{ab}\|_2} \cdot \frac{\mathbf{t}_b - \mathbf{X}_{ab}}{\|\mathbf{t}_b - \mathbf{X}_{ab}\|_2}
\tag{3}
$$

（式 (3) 照录原文；其中 $\mathbf{t}_a$、$\mathbf{t}_b$ 在此处指两相机的**投影中心**位置——按论文位姿约定 $\mathbf{P} = [\mathbf{R}^T \; -\mathbf{R}^T\mathbf{t}]$，投影中心为 $-\mathbf{R}^T\mathbf{t}$，原文以 $\mathbf{t}$ 简记，属记号沿用。）两个单位向量点乘即夹角余弦（依据：$\cos\alpha = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$ 的定义）。**为什么小三角角不可靠**——论文只给判据不推理由，此处补一条延伸推导（标准结果，非论文内容）：

设两光心基线长 $B$，点 $\mathbf{X}$ 对两相机的张角（即 (3) 的 $\alpha$），相机 1 处观测角 $\theta_1$、相机 2 处 $\theta_2$，三角形内角和给出 $\alpha = \pi - \theta_1 - \theta_2$。由正弦定理（依据：平面三角形正弦定理），点到相机 1 的距离
$$
d_1 \;=\; B\,\frac{\sin\theta_2}{\sin\alpha}.
$$
给相机 2 的方位角观测加噪声 $\delta$：则 $\theta_2 \to \theta_2 + \delta$，且由内角和不变得 $\alpha \to \alpha - \delta$。对 $d_1(\theta_2, \alpha) = B\sin\theta_2/\sin\alpha$ 取全微分（依据：商法则 + 链式法则；$\partial_{\theta_2}\sin\theta_2 = \cos\theta_2$，$\partial_\alpha (\sin\alpha)^{-1} = -\cos\alpha/\sin^2\alpha$，$\delta\alpha = -\delta$）：
$$
\frac{\delta d_1}{d_1} \;=\; \big(\cot\theta_2 + \cot\alpha\big)\,\delta.
$$
小三角化角情形（$\alpha \to 0$）下 $\cot\alpha \approx 1/\alpha$ 占主导（依据：$\cot\alpha$ 在 $\alpha\to 0$ 处的一阶泰勒展开），且由正弦定理 $\sin\alpha = B\sin\theta_2/d_1 \approx \alpha$，代入得
$$
\sigma_{d_1} \;\approx\; \frac{d_1}{\alpha}\,\sigma_\theta \;\approx\; \frac{d_1^2}{B\,\sin\theta_2}\,\sigma_\theta,
$$
即**深度误差与三角化角近似成反比、与距离平方成正比**（$\sigma_\theta = \sigma_x/f$ 时正对观测 $\theta_2 = 90^\circ$ 退化为经典的 $\sigma_d \approx \frac{d^2}{fB}\sigma_x$；每步依据：小角近似 $\sin\alpha\approx\alpha$、像移–角噪声换算 $x = f\tan\theta \approx f\theta$）。结论：沿视线方向的信息量 $\propto \sin\alpha$——基线接近零或点在远方时，深浅任意挪动都几乎不改变像素观测，三角化结果被噪声任意摆布。这就是 (3) 设阈值、以及 4.1 中全景对（纯旋转、基线为零）禁入三角化的定量根据。

**良置条件二：正深度（Eq.(4)，手性约束）**。模型点在两视图下的深度须为正：

$$
d \;=\; \begin{bmatrix} p_{31} & p_{32} & p_{33} & p_{34} \end{bmatrix} \begin{bmatrix} \mathbf{X}_{ab}^{\pi} & 1 \end{bmatrix}^{\!\top} > 0
\tag{4}
$$

（$p_{mn}$ 为 $\mathbf{P}$ 的第 $m$ 行第 $n$ 列元素；$\mathbf{X}^{\pi}_\cdot = (\mathbf{X}^{\top},1)^{\top}$ 为齐次坐标。依据：投影 $[\mathbf{P}\tilde{\mathbf{X}}]_3$ 即相机系 $z$ 坐标——第三行与齐次点的内积，与 [SLAM 教程 (5.7)](../../slam/05_视觉里程计-i特征点法.md) 的手性分析同一判据。）

**内点确认（Eq.(5)）**：测量 $\mathcal{T}_n$ 服从模型当且仅当深度 $d_n > 0$ 且重投影误差

$$
r_n \;=\; \left\|\, \bar{\mathbf{x}}_n - \begin{bmatrix} x'_n/z'_n \\ y'_n/z'_n \end{bmatrix} \,\right\|_2, \qquad \begin{bmatrix} x' \\ y' \\ z' \end{bmatrix} = \mathbf{P}_n \begin{bmatrix} \mathbf{X}_{ab} \\ 1 \end{bmatrix}
\tag{5}
$$

小于阈值 $t$（依据：齐次投影除以第三分量得像素坐标，再取欧氏残差——与 [SLAM 教程 (5.10)](../../slam/05_视觉里程计-i特征点法.md) 的重投影定义同形，仅符号方向相反）。

**采样策略与递归**（论文文字，要点照录）：置信度 $\eta$ 下至少采到一个全内点最小集需 $\hat{K}$ 次迭代（自适应停止：先验内点率取小初值 $\epsilon_0$，找到更大共识集即重算 $\hat{K}$；采样器只生成**互异**最小集，避免小轨道上重复采样）；共识集从剩余测量中移除后**递归**执行，恢复错并的多点轨道；递归停止条件为最新共识集小于 3。Sec. 5 参数：Dubrovnik 上 $\alpha = 2^\circ$、$t = 8$ px、$\epsilon_0 = 0.03$（穷举对照限 10K 次迭代，即 $\epsilon_{crit}\approx 0.02$、$\eta = 0.999$）。

### 4.4 光束法平差：目标函数、调度与 Schur 补（Sec. 2.2 / 4.4，Eq.(1)）

**目标函数（Eq.(1)，照录原文）**：

$$
E \;=\; \sum_j \rho_j\Big( \big\|\, \pi\big(\mathbf{P}_c,\ \mathbf{X}_k\big) - \mathbf{x}_j \,\big\|_2^2 \Big)
\tag{1}
$$

$\mathbf{P}_c$ 为相机参数、$\mathbf{X}_k$ 为点参数，$\pi$ 把场景点投到像面，$\rho_j$ 为降权外点的损失函数。这与 [教程第 02 章 (2.12)](../02_多视图几何与SfM.md) 逐项同一（变量、观测、目标一致；残差方向差一整体符号），其"同一最小二乘"的逐项对照见该章 ⑤；带核求解等价于信息矩阵换 $W\Sigma^{-1}$（教程 (2.13)–(2.14) 的 IRLS 推导）。**本文的具体选型**：局部 BA 用 **Cauchy 核** $\rho_j$（对照教程 (2.15) Huber / (2.16) Tukey：Cauchy 介于两者之间——大残差按 $1/e^2$ 量级降权但不清零）；数百相机内用稀疏直接解法、更大规模用 PCG；求解器为 Ceres；未标定图像用单径向畸变参数的简化相机模型 + 纯自标定。

**调度（局部/全局 BA 的"滑动窗口"含义）**：图像注册与三角化是独立过程但产物强耦合（位姿误差传播到三角化、反之亦然），故 BA 必须反复执行。本文调度：**每次注册后对"最连通图像集"做局部 BA**——即只优化与当前视图共视的位姿/点子集，形式上就是一个以共视图为邻接定义的滑动窗口；模型增长一定百分比才做一次全局 BA，把全局 BA 摊销成**均摊线性**的总耗时（论文原文 "amortized linear run-time"）。

**稀疏解法（Schur 补）**。LM 法解 (1) 的正规方程时，Hessian 按 (相机块, 点块) 分块（依据：每条残差只含一个相机与一个点，雅可比的两列块结构确定 Hessian 的四块结构）：

$$
\begin{bmatrix} H_{cc} & H_{cp} \\ H_{pc} & H_{pp} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{c} \\ \Delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}
\;\Longrightarrow\;
\underbrace{\big(H_{cc} - H_{cp}H_{pp}^{-1}H_{pc}\big)}_{S\ (\text{约简相机系统})}\,\Delta\mathbf{c} = \mathbf{b}_c - H_{cp}H_{pp}^{-1}\mathbf{b}_p,
$$

第一步由第二行解出 $\Delta\mathbf{p} = H_{pp}^{-1}(\mathbf{b}_p - H_{pc}\Delta\mathbf{c})$（依据：分块方程逐行展开；$H_{pp}$ 块对角可逆——每条残差只涉一个点，非对角块逐项为零），代入第一行合并 $\Delta\mathbf{c}$ 项即得约简系统；再回代恢复 $\Delta\mathbf{p}$（每步推导的完整展开见 [SLAM 教程 (8.11)–(8.15)](../../slam/08_后端-ii图优化与-ba.md)，此处不重推只引结论）。论文引用的正是这一 Schur 补技巧 [8]：**先解约简相机系统、再回代更新点**——相机数远小于点数，故先解小的那一侧。复杂度对照（论文 Sec. 2.2）：精确法存储 $O(N_P^2)$、时间 $O(N_P^3)$；PCG 等间接法时间与空间均 $O(N_P)$——几百相机内直接法、更大规模间接法，与本文调度一致。

### 4.5 冗余视图挖掘：分组 BA（Sec. 4.5，Eq.(6)–(7)）

**动机**：网络照片集的可见性高度不均匀（热门景点被海量近重复视角拍摄），而增量 SfM 每步只局部改动模型——大量"未受影响"的图像反复进入 BA 是纯浪费。对策：把未受影响的高重叠图像**分组、组内坍缩为单个相机**共同参数化。

**受影响判定**（论文原文判据）：图像属"受影响"若 (i) 在最近一次模型扩展中被加入，或 (ii) 其观测中重投影误差大于 $r$ 像素的比例超过 $\epsilon_r$（为重三角化的相机留精化通道）。其余图像进入分组。

**重叠度度量（Eq.(6)）**。给场景中 $N_X$ 个点定义图像 $i$ 的二值可见性向量 $\mathbf{v}_i \in \{0,1\}^{N_X}$（第 $n$ 位取 1 当点 $\mathbf{X}_n$ 在图像 $i$ 可见），图像 $a, b$ 的**交互度**：

$$
V_{ab} \;=\; \frac{\|\mathbf{v}_a \wedge \mathbf{v}_b\|}{\|\mathbf{v}_a \vee \mathbf{v}_b\|}
\tag{6}
$$

（$\wedge/\vee$ 为按位与/或，$\|\cdot\|$ 取元素和。依据：分子 = 共视点数，分母 = 至少在一个视图中可见的点数——(6) 正是 Jaccard 系数，$\in [0,1]$，共视越纯越大；与 Ni et al. [43] 用图割度量共视相比，(6) 只需位运算，构造代价可忽略。）

**贪心分组**（论文原文流程）：按 $\|\mathbf{v}_i\|$ 降序排序得 $\bar{\mathcal{I}}$；取出队首图像 $I_a$，在其 $K_r$ 个空间最近邻（共视方向差 $\pm\beta$ 度内——依据：大位移图像共视少的先验）中找最大化 $V_{ab}$ 的 $I_b$；若 $V_{ab} > V$ 且 $|G_r| < S$ 则 $I_b$ 入组 $G_r$，否则新开一组；重复直至 $\bar{\mathcal{I}}$ 空。

**分组代价函数（Eq.(7)，照录原文）**：

$$
E_g \;=\; \sum_j \rho_j\Big( \big\|\, \pi_g\big(\mathbf{G}_r,\ \mathbf{P}_c,\ \mathbf{X}_k\big) - \mathbf{x}_j \,\big\|_2^2 \Big), \qquad \mathbf{P}_r^c = \mathbf{P}_c\,\mathbf{G}_r
\tag{7}
$$

组共享外参 $\mathbf{G}_r \in SE(3)$ 为**优化变量**，组内图像位姿 $\mathbf{P}_c$（组局部系）保持固定，图像的世界位姿由二者复合 $\mathbf{P}_r^c = \mathbf{P}_c\mathbf{G}_r$ 拼接（旋转部分用四元数以高效复合；依据：论文原文 "defined as the concatenation of the group and image pose"）。效果：一组 $S$ 个相机从 $S$ 个独立位姿变量坍缩为 1 个共享 $\mathbf{G}_r$——4.4 约简相机系统 $S$ 的维数从 $6\times(\text{相机数})$ 降到 $6\times(\text{未分组相机数} + \text{组数})$，直接法 $O(N_P^3)$ 的立方律使这一缩减被立方级放大（依据：4.4 复杂度对照；论文补充 "a reduction in the number of cameras affects the cubic computational complexity of direct methods more than the linear complexity of indirect methods"）。总代价 $\bar{E}$ = 分组贡献 + 未分组图像的标准 BA（Eq.(1)）贡献。与 Ni et al. 的两点差异（论文自述）：不做昂贵的图割划分；组小而重叠高，故**省去分隔变量交替优化**。

### 4.6 迭代精化与退化检测（Sec. 4.4 Filtering / Re-Triangulation / Iterative Refinement）

- **过滤**：BA 后剔除重投影误差过大的观测；每点对**所有视线对**强制最小三角化角（依据同 4.3 的角度–不确定性推导；策略引 [53] Photo Tourism）。全局 BA 后再查**退化相机**（全景/人为增强图像）：特征是只剩外点观测或内参收敛到荒谬极小——因此焦距与畸变不做先验区间约束、任其在 BA 中自由优化（坏相机自己会露馅），主点因标定病态 [15] 固定于图像中心；视场角异常或畸变系数过大的相机在全局 BA 后被滤除。
- **重三角化 RT**：漂移使先前失败三角化的点在位姿改善后可救活。本文在 VisualSfM 式 pre-BA RT 之外增设 **post-BA RT**——BA 提升位姿与点之后，把误差仍在过滤阈值内的轨道续上、并尝试合并轨道，为下一轮 BA 提供更多冗余。
- **迭代精化**：BA 严重受外点影响，而 pre-BA RT 后轨道中恰有大量外点；一轮"BA → RT → 过滤"清不干净。故迭代执行直至"被过滤观测数与 post-BA RT 新增点数"收敛，通常第二轮后即大幅改善（论文 Sec. 5 消融：迭代精化显著提升完备性）。

## 5. 实验与结果解读

- **总体对比**（17 个数据集、144,953 张图；对手：增量式 Bundler / VisualSFM，全局式 DISCO / Theia）：本文在**完备性**（注册图像数、稠密点数、平均轨道长度）上全面领先，数据集越大领先越明显——更长轨道带来更高 BA 冗余，形成正反馈。Fig. 1 示例：Rome 约 21K 张注册（总量 75K）。
- **精度**：带真值的 Quad 数据集上相机位置误差最小（本文 0.85 m；对照 Bundler 1.01 m、VisualSFM 0.89 m、DISCO 1.16 m）——归功于更准的注册顺序（4.2 评分）与更长的轨道。
- **效率**：与 VisualSFM 相当（略慢），比 Bundler 快 50 倍以上，略慢于最快的 Theia；对应搜索不计入计时。
- **三角化消融**（Dubrovnik，2.9M 轨道 / 47M 验证匹配）：RANSAC 版相对穷举版快 10–40 倍，恢复的**长轨道显著更多**（长轨道 = 抗外点 + BA 冗余），代价是轨道略短；调 $\eta$ 可权衡速度与精度。
- **冗余视图挖掘消融**：强制场景覆盖比 $V=0.6/0.3/0.1$ 时，约简相机系统求解提速 5%/14%/32%，平均重投影误差 0.26 px → 0.27/0.28/0.29 px——精度基本无损；取 $V=0.4$ 整条管线提速 36% 而重建等价；$V<0.3$ 后质量开始退化。

**定性总结**：鲁棒性与完备性的提升不是靠堆算力，而是靠"把几何正确性判据内嵌进每一步决策"——这一系统方法论比任何单项数字更重要。

## 6. 局限与后续影响

- **局限**：增量式的注册顺序依赖仍存在（坏初始化仍可能不可恢复，论文以"精心选对"缓解而非根除）；BA 反复执行的计算冗余是结构性代价（由局部/全局调度与分组 BA 缓解）；对超大稠密集，组参数化的静态分组不适应模型持续演化。
- **后续影响**：COLMAP 成为学术与工业界的**事实标准**重建管线；其场景图多模型验证、下一最优视图评分、RANSAC 三角化被后续系统（OpenMVG、Theia 后续版本、GLOMAP 等）吸收或对标；分组 BA 思想影响了大规模 BA 的分层/子图参数化研究；MVS 半边（姊妹篇，见 [COLMAP-MVS 精读](./COLMAP-MVS_ECCV2016.md)）直接以本文 SfM 输出为输入，构成"稀疏→稠密"完整开源链路。

## 7. 与本项目对照

**epipolar 模块覆盖了本文前端几何的同源内核**（[projects/slam/epipolar/epipolar.py](../../../projects/slam/epipolar/epipolar.py)）：

| 本文环节 | 论文依据 | epipolar 模块对应 | 与 SLAM 教程对应 |
|---|---|---|---|
| 两视图初始化 | Sec. 2.2 Initialization | `essential_from_rt` / `fundamental_from_E` / `decompose_E` | (5.1)、(5.5)–(5.7) |
| 对应验证 | Sec. 4.1（$F$/八点法路线） | `eight_point`（Hartley 归一化 + 齐次最小二乘） | (5.2)–(5.4) |
| 图像注册 | Sec. 2.2 Image Registration（PnP） | `pnp_dlt`（线性初值）+ `pnp_refine`（非线性精化） | (5.10)–(5.12) |
| 三角化 | Sec. 4.3 Eq.(2)/(4)（DLT + 手性） | `triangulate`（批量 SVD 最小奇异向量 + $w>0$ 约定） | 教程 (2.3)–(2.4)、(5.8) |
| BA 雅可比 | Sec. 4.4 Eq.(1) | `reprojection_jacobian` | (8.10) |

- **BA 的完整推导**在 [SLAM 教程第 08 章](../../slam/08_后端-ii图优化与-ba.md)：残差 (8.1)、目标 (8.2)、正规方程 (8.5)、LM 阻尼 (8.6)、左扰动雅可比 (8.10)、Schur 补 (8.11)–(8.15)——本文 4.4 只是"引用级"复述，不另立门户；鲁棒核的 IRLS 推导在 [教程第 02 章 (2.13)–(2.16)](../02_多视图几何与SfM.md)。
- **未建 projects/3d_reconstruction 代码的原因**：本文的增量价值在**系统工程层**（场景图管理、调度、递归 RANSAC、分组 BA），其几何内核已被 epipolar 覆盖、其优化内核已被 SLAM 教程覆盖；离线 SfM 管线是重工程投入（匹配检索、增量图维护、Ceres 级优化器），教学收益密度低——故按"教程回引 + 精读对照"处理，完整实现列为规划项。机器人场景中位姿常可由 SLAM/里程计提供（第 04 章线），进一步降低了复刻 SfM 前端的优先级。

## 配套阅读

- 教程主线：[第 02 章｜多视图几何与 SfM](../02_多视图几何与SfM.md)（三角化 (2.1)–(2.10)、场景图 (2.11)、BA (2.12)、鲁棒核 (2.13)–(2.16)）｜[第 03 章｜MVS](../03_多视图立体重建MVS.md)（本文输出的稠密化）。
- 跨主题回引：[SLAM 教程第 05 章](../../slam/05_视觉里程计-i特征点法.md)（对极 (5.1)、八点法 (5.2)–(5.4)、E 分解 (5.5)–(5.7)、DLT (5.8)、PnP (5.10)–(5.12)）｜[SLAM 教程第 08 章](../../slam/08_后端-ii图优化与-ba.md)（BA 全链路 (8.1)–(8.15)）。
- 代码：[projects/slam/epipolar/](../../../projects/slam/epipolar/)（本文前端几何的教学实现）；BA 参考 [projects/slam/core/](../../../projects/slam/core/)。
- 姊妹篇：[COLMAP-MVS 精读](./COLMAP-MVS_ECCV2016.md)（同一作者的稠密重建半边）。
- 教材：Hartley & Zisserman, *Multiple View Geometry*（三角化与光束平差章）；Triggs et al., *"Bundle Adjustment — A Modern Synthesis"*, ICCV 1999。

# 论文精读｜ORB-SLAM3（IEEE T-RO 2021）

> **PDF**：[arXiv-2007.11898_ORB-SLAM3.pdf](../../../papers/slam/classics/arXiv-2007.11898_ORB-SLAM3.pdf) ｜ **教程**：第 10 章（主线）＋第 09/08 章 ｜ **代码**：[projects/slam/epipolar/](../../../projects/slam/epipolar/) + [projects/slam/bowloop/](../../../projects/slam/bowloop/) + [projects/slam/preint/](../../../projects/slam/preint/)（ORB-SLAM3 是系统级工作，本项目无独立对应模块，见 §7）

**式号约定**：下文"式 (1)–(10)"均指本 PDF（arXiv v2，即 T-RO 录用版）原文编号，已逐式与 PDF 页面核对（式 (1)(2) 在第 6–7 页 §V-A，式 (3)(4) 在第 7 页，式 (5)–(10) 在第 7–8 页 §V-B）。"(10.x)"指教程第 10 章，"(8.x)/(9.x)"指教程第 08/09 章。

## 1. 论文信息与一句话贡献

- **标题**：ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM（Campos, Elvira, Gómez Rodríguez, Montiel & Tardós，*IEEE Transactions on Robotics*，2021；DOI: 10.1109/TRO.2021.3075648）
- **一句话贡献**：把 ORB-SLAM 系升级为"视觉 / 视觉-惯性 / 多地图"三合一系统——以 MAP（maximum a posteriori，最大后验）估计统一 IMU 初始化与视觉-惯性 BA，以 Atlas（地图集）多地图表示 + 高召回地点识别支撑无缝地图合并，论文自述在既有方法上精度提升"two to ten times"（摘要），EuRoC 无人机序列平均精度 3.5 cm、TUM-VI 室内手持序列 9 mm。
- **谱系定位**：架构直接继承 [ORB-SLAM（T-RO 2015）](./ORB-SLAM_TRO2015.md)的三线程设计；视觉-惯性部分承自 ORB-SLAM-VI（Mur-Artal & Tardós, "Visual-inertial monocular SLAM with map reuse", RA-L 2017，本文引 [4]——首个可复用地图的单目 VI SLAM），并把其过慢的初始化换成本文三阶段 MAP 法（快速初始化的先行版出自 Campos et al., "Inertial-only optimization for visual-inertial initialization", ICRA 2020，引 [6]）；论文 §I 自我定位"This is essentially a system paper"——贡献在系统组织与若干关键算法件，而非单一公式。

## 2. 问题与动机

论文 §I 用"数据关联"的三层划分立论——SLAM 与纯 VO（visual odometry，视觉里程计）的分界在于用不用后两类：

1. **短期关联（short-term）**：只匹配最近几帧的地图元素，出视野即忘——多数 VO 的全部；漂移因此只被抑制、不被清除。
2. **中期关联（mid-term）**：匹配"靠近相机、累计漂移还小"的地图元素并纳入 BA——ORB-SLAM3 强调这是其精度优势的直接来源（§I）。
3. **长期关联（long-term）**：地点识别（place recognition）跨越累计漂移认出旧地——回环（loop closing）或重定位（relocalization）。

由此引出三个待解问题：其一，跟踪一旦丢失，单地图系统要么重启（丢掉全部历史），要么无力恢复——需要**多地图**机制：丢失时新建地图，重逢时合并；其二，ORB-SLAM-VI 的 IMU 初始化太慢（论文 §II-B：约 15 s 才得到首个尺度估计，"harmed robustness and accuracy"），VI-DSO 一类从零跑 VI-BA 的方法要 20–30 s 才收敛到 1% 尺度误差（§V-B 引述）——需要**快且把传感器不确定性算进去**的初始化；其三，经典 DBoW2 查询靠"时间一致性 + 几何校验"把工作点移到 100% 精度、30–40% 召回（§VI-A，引 [9][75]）——召回低意味着该合并的地图合不上、该闭的环闭不上，需要**高召回**的地点识别。

## 3. 方法总览

系统为三线程 + Atlas（论文图 1）：

- **Tracking（跟踪线程）**：逐帧解算相对活跃地图的位姿（含 VI 情形下的速度与 IMU 零偏），决定关键帧；跟踪丢失时先尝试短期恢复（投影地图点做大窗口匹配），5 s 仍失败且 15 s 内 IMU 初始化已成熟则转长期处理。
- **Local Mapping（局部建图线程）**：向活跃地图增删关键帧与地图点，做局部 VI-BA；VI 情形下用 MAP 技术初始化并精调 IMU 参数。
- **Loop and Map Merging（回环与地图合并线程）**：在关键帧率上对活跃地图 vs 全 Atlas 做地点识别；公共区域属于活跃地图→回环校正，属于其他地图→两图合并，之后全图 BA 在独立线程异步执行。

**Atlas 机制**（§III）：Atlas = 一组互不连通的地图；正在被跟踪与优化的是**活跃地图（active map）**，其余为**非活跃地图（non-active maps）**；全 Atlas 共用一个 DBoW2 关键帧数据库支撑重定位、回环与合并。地图创建判据（论文原话）："Otherwise, after a certain time, the active map is stored as non-active, and a new active map is initialized from scratch"——即短期恢复失败后，旧活跃地图整体转非活跃、从零建新图；**VI 特例**：IMU 初始化后 15 s 内就丢失的地图直接丢弃（"If the system gets lost within 15 seconds after IMU initialization, the map is discarded"，§V-D）——初始化未成熟的地图不可信，留着只会注入错误信息。

**合并与缝合的分工**（§VI-B）：地图合并分两步走——先在**缝合窗口（welding window）**内做重活：即 $K_a$（活跃地图关键帧）与 $K_m$（匹配地图关键帧）各自的共视邻域，以及它们观测的全部地图点；再把校正经本质图传播到合并后大图的其余部分。术语上：**merging（合并）**指把 $M_a$ 变换到 $M_m$ 参考系、融为一张新活跃地图并去重的全过程；**welding（缝合）**特指其中"在新图拼接面上做局部 BA + 位姿图传播"的收口步骤。这样切分的理由：合并是高风险操作（改错全图皆错），在封闭的小窗口内先收敛、再传播，风险与计算都被限定。

**相机模型抽象**（§IV）：投影、反投影、雅可比被抽成独立模块，针孔之外支持 Kannala-Brandt 鱼眼模型；鱼眼不做整幅矫正（畸变与分辨率代价），立体 rig 按两个单目处理（恒定 SE(3) 外参 + 可选共同观测区）。

**论文 §I 末自列的六项创新**（本文公式的落点）：① 单目 / 双目视觉-惯性 SLAM 完全基于 MAP 估计，连 IMU 初始化阶段也是（→ §4.3 式 (5)–(10)）；② 改进召回的地点识别：先几何一致性、再三共视关键帧 local consistency（→ §4.5）；③ ORB-SLAM Atlas：第一个能处理视觉与视觉-惯性的完整多地图 SLAM（→ §3）；④ 抽象相机表示，代码与模型解耦（→ §3 末）；⑤ 对单目 / 双目 / 单目-惯性 / 双目-惯性在公开数据集上的系统评测（→ §5）；⑥ 结论性对比：立体-惯性最准、单目-惯性在相机不理想时以 IMU 频率供位姿（§VIII）。

## 4. 关键公式推导

### 4.0 符号表（论文 §V-A 记号）

| 符号 | 含义 |
|---|---|
| $\mathbf{T}_i = [\mathbf{R}_i, \mathbf{p}_i] \in SE(3)$ | 第 $i$ 个关键帧的 body（IMU）位姿，世界系←body 系 |
| $\mathbf{v}_i \in \mathbb{R}^3$ | body 系速度（世界系表达） |
| $\mathbf{b}_i^g, \mathbf{b}_i^a \in \mathbb{R}^3$ | 陀螺仪 / 加速度计零偏（bias） |
| $\Delta\mathbf{R}_{i,i+1}, \Delta\mathbf{v}_{i,i+1}, \Delta\mathbf{p}_{i,i+1}$ | 帧间 IMU 预积分测量（旋转 / 速度 / 位置） |
| $\Sigma_{\mathcal{I}_{i,i+1}}$ | 预积分测量总协方差 |
| $\mathbf{g} = \mathbf{R}_{wg}\,\mathbf{g}_I,\ \mathbf{g}_I = (0,0,G)^\top$ | 世界系重力矢量（指向下），$\mathbf{R}_{wg}$ 把惯性系重力对齐到世界系 |
| $\mathbf{T}_{CB} \in SE(3)$ | body-IMU → 相机外参（标定已知） |
| $\mathbf{x}_j \in \mathbb{R}^3,\ \mathbf{u}_{ij},\ \Sigma_{ij}$ | 3D 点 $j$、其在帧 $i$ 的像素观测、观测协方差 |
| $\Pi:\mathbb{R}^3 \to \mathbb{R}^n,\ \oplus$ | 抽象相机投影函数；SE(3) 对 $\mathbb{R}^3$ 的作用（变换操作） |
| $\mathcal{K}^j$ | 观测到点 $j$ 的关键帧集合 |
| $\rho_{Hub}$ | Huber 鲁棒核 |
| $\mathcal{Y}_k = \{s, \mathbf{R}_{wg}, \mathbf{b}, \mathbf{v}_{0:k}\}$ | 纯惯性初始化的估计变量（式 (5)） |
| $\Sigma_b$ | 零偏先验协方差（"偏置应为小量"的先验知识） |
| $\bar{\mathbf{p}}_{0:k}$ | 视觉解的上到尺度（up-to-scale）平移量（bar 记未定尺度） |
| $\mathbf{T}_{nm}$ | 匹配地图 $M_m$ → 活跃地图 $M_a$ 的对齐变换（Sim(3) 或 SE(3)） |
| $K_a,\ K_m,\ M_a,\ M_m$ | 活跃 / 匹配关键帧，活跃 / 匹配地图 |

记法提醒：论文马氏范数 $\|\mathbf{e}\|^2_\Sigma$ 下标写**协方差**（$\mathbf{e}^\top\Sigma^{-1}\mathbf{e}$）；教程 (8.2) 与综述式 (3)(4) 下标写**信息矩阵**（$\Omega = \Sigma^{-1}$）。两者只差记号，推导时须盯住定义。

### 4.1 视觉-惯性状态与 IMU 预积分残差（式 (1)(2)）

**状态定义（式 (1)）**：

$$\mathcal{S}_i \doteq \{\mathbf{T}_i,\ \mathbf{v}_i,\ \mathbf{b}_i^g,\ \mathbf{b}_i^a\} \tag{1}$$

依据（论文原话）：纯视觉 SLAM 只估相机位姿，VI 须补估 body 速度与陀螺/加计零偏，且它们"assumed to evolve according to a Brownian motion"（按布朗运动演化）——这正是预积分框架把零偏当分段常数 + 随机游走处理的状态侧对应（教程 10.2 第 7 步）。

**预积分量从哪来**：论文按 [60][61]（Forster 框架）在帧 $i, i+1$ 之间预积分得到 $\Delta\mathbf{R}_{i,i+1}, \Delta\mathbf{v}_{i,i+1}, \Delta\mathbf{p}_{i,i+1}$ 与协方差 $\Sigma_{\mathcal{I}_{i,i+1}}$。其完整推导（世界系积分的初值锚定问题 → 相对量重定义 → 测量模型）= 教程第 10 章式 (10.3)→(10.4)→(10.5)，本项目实现见 §7 的 `preint` 模块。此处只强调一条对读式 (2) 必要的性质：**三个预积分量只依赖 IMU 原始测量与偏置估计，与初值 $\mathbf{R}_i, \mathbf{v}_i, \mathbf{p}_i$ 无关**（教程 (10.4) 代入验证的结论）；重力不出现在预积分递推里，只在测量模型中重新出现（教程 (10.5)；`preint` 模块 docstring "为什么这里没有重力"一节）。

**IMU 残差（式 (2)）**：$\mathbf{r}_{\mathcal{I}_{i,i+1}} = [\mathbf{r}_{\Delta\mathbf{R}_{i,i+1}},\ \mathbf{r}_{\Delta\mathbf{v}_{i,i+1}},\ \mathbf{r}_{\Delta\mathbf{p}_{i,i+1}}]$，逐段推导（每段依据 = 教程 (10.5) 测量模型移项 + 流形残差必须落在向量空间）：

旋转段：

$$\mathbf{r}_{\Delta\mathbf{R}_{i,i+1}} = \mathrm{Log}\big(\Delta\mathbf{R}_{i,i+1}^{\top}\ \mathbf{R}_i^{\top}\ \mathbf{R}_{i+1}\big)$$

推导：由测量模型（教程 (10.5) 第一式，零噪声、偏置修正项并入 $\Delta\mathbf{R}$ 记号）$\Delta\mathbf{R}_{i,i+1} \approx \mathbf{R}_i^\top \mathbf{R}_{i+1}$，即自洽时 $\Delta\mathbf{R}_{i,i+1}^\top\,\mathbf{R}_i^\top\mathbf{R}_{i+1} = I$（依据：矩阵乘法结合律与 $\Delta\mathbf{R}^\top\Delta\mathbf{R} = I$ 的 SO(3) 正交性）。但两个旋转矩阵之差不能直接相减定义残差——差矩阵不在向量空间、无法配高斯噪声；对数映射 $\mathrm{Log}: SO(3) \to \mathbb{R}^3$ 把群元素映到李代数向量（论文第 7 页明说 "where Log : SO(3) → R³ maps from the Lie group to the vector space"；教程 (9.7) 推导位姿图残差时是同一论证），且 $\mathrm{Log}(I) = \mathbf{0}$——自洽时残差恰为零，模型闭合。复合顺序 $\Delta\mathbf{R}^\top(\mathbf{R}_i^\top\mathbf{R}_{i+1})$ 把"测得的相对旋转"与"状态给出的相对旋转"放进同一参考系再相乘，与 Forster 框架的误差表达约定一致（教程 (10.10) 的旋转段同构）。

速度段：

$$\mathbf{r}_{\Delta\mathbf{v}_{i,i+1}} = \mathbf{R}_i^{\top}\big(\mathbf{v}_{i+1} - \mathbf{v}_i - \mathbf{g}\,\Delta t_{i,i+1}\big) - \Delta\mathbf{v}_{i,i+1}$$

推导：测量模型（教程 (10.5) 第二式）$\Delta\tilde{\mathbf{v}}_{ij} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) + \delta\mathbf{v}_{ij}$，把噪声项移除、取期望后移项（依据：残差 = 状态预测 − 测量期望，即 $\mathbf{z} = h(\mathbf{x}) + \mathbf{v}$ 的标准残差化，第 03 章观测方程同构）。左乘 $\mathbf{R}_i^\top$ 的依据：预积分量定义在"帧 $i$ body 系"（教程 (10.4) 的定义式就是 $\mathbf{R}_i^\top(\cdot)$），残差两端必须在同一坐标系表达才能相减。重力 $\mathbf{g} = \mathbf{R}_{wg}\mathbf{g}_I$ 出现在**测量模型**侧而非预积分递推侧——这正是 $\mathbf{R}_{wg}$ 进入 IMU 残差、从而可被优化的入口。

位置段：

$$\mathbf{r}_{\Delta\mathbf{p}_{i,i+1}} = \mathbf{R}_i^{\top}\Big(\mathbf{p}_{i+1} - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{i,i+1} - \tfrac{1}{2}\,\mathbf{g}\,\Delta t_{i,i+1}^2\Big) - \Delta\mathbf{p}_{i,i+1}$$

推导：与速度段完全同理，把教程 (10.5) 第三式移项；$\tfrac{1}{2}\mathbf{g}\Delta t^2$ 是重力对位置的二次积分贡献（依据：匀加速运动学 $\Delta p = v\Delta t + \tfrac{1}{2}a\Delta t^2$ 在世界系成立，再整体左乘 $\mathbf{R}_i^\top$ 换到帧 $i$ body 系）。

**零偏随机游走残差**：注意式 (2) 只有三段——陀螺/加计零偏随机游走（random walk）残差在论文中**没有编号公式**，而是以因子图形式出现：图 2 图例列有 "Random Walk" 残差（紫色方块），图 3 标题写明 "bias random walk (purple squares)"。其形式沿用 Forster 框架的标准写法：$\mathbf{r}_{\Delta\mathbf{b}} = \mathbf{b}_{i+1} - \mathbf{b}_i$，代价 $\|\mathbf{r}_{\Delta\mathbf{b}}\|^2_{\Sigma_{\Delta\mathbf{b}}}$，其中 $\Sigma_{\Delta\mathbf{b}}$ 由随机游走密度（本教程 `preint.ImuParams` 的 `sigma_bg`/`sigma_ba`）乘时长累积（依据：式 (1) 的布朗运动假设在区间上离散化，方差随时间线性增长）。**提醒**：此式为本教程按 Forster 框架补写，论文原文未给出，读源码时在 `ORB_SLAM3` 的 IMU 因子里能找到它。

### 4.2 重投影残差与联合 MAP（式 (3)(4)）

**重投影残差（式 (3)）**：

$$\mathbf{r}_{ij} = \mathbf{u}_{ij} - \Pi\big(\mathbf{T}_{CB}\ \mathbf{T}_i^{-1} \oplus \mathbf{x}_j\big)$$

逐步推导：世界系点 $\mathbf{x}_j$ 先被 $\mathbf{T}_i^{-1}$ 变换到第 $i$ 帧 body 系（依据：$\mathbf{T}_i$ 定义为世界←body，逆变换即 body←世界），再被外参 $\mathbf{T}_{CB}$ 变到相机系（依据：$\mathbf{T}_{CB}$ 的定义"from body-IMU to camera"），最后经 $\Pi$ 投到像素——$\oplus$ 正是 SE(3) 对点的变换操作。与教程对照：教程 (8.1) 的 $\mathbf{e}_{ij} = \mathbf{z}_{ij} - \pi(\mathbf{T}_i\mathbf{p}_j)$ 是本式去掉 IMU body 系中转、把 $\Pi$ 具体化为针孔模型的特例（残差同向：观测 − 预测）；第 05 章 (5.10) 的 PnP 残差则取预测 − 观测（该章已注明与 (8.1) 差一个整体符号，雅可比随之整体变号、GN 增量不变）。

**联合优化（式 (4)）**：

$$\min_{\bar{\mathcal{S}}_k,\ \mathcal{X}}\ \Bigg(\sum_{i=1}^{k}\big\|\mathbf{r}_{\mathcal{I}_{i-1,i}}\big\|^2_{\Sigma_{\mathcal{I}_{i-1,i}}}\ +\ \sum_{j=0}^{l-1}\ \sum_{i\in\mathcal{K}^j}\ \rho_{Hub}\Big(\big\|\mathbf{r}_{ij}\big\|^2_{\Sigma_{ij}}\Big)\Bigg) \tag{4}$$

解读与依据：这是教程 (8.2)（$\min_\mathbf{x} \sum_k \|\mathbf{e}_k\|^2_{\Lambda_k}$）的 VI 实例——IMU 残差与重投影残差同入一个最小二乘，权重都是各自协方差之逆。逐项 unpack：$\bar{\mathcal{S}}_k = \{\mathcal{S}_0 \dots \mathcal{S}_k\}$ 是 $k+1$ 个关键帧的 VI 状态集（式 (1)），$\mathcal{X} = \{\mathbf{x}_0 \dots \mathbf{x}_{l-1}\}$ 是 $l$ 个 3D 点，$\mathcal{K}^j$ 是观测到点 $j$ 的关键帧集合——内层对 $\mathcal{K}^j$ 求和正是"一个点被多帧看到"的多观测约束叠加，其稀疏结构即教程 (8.12) 的块对角来源（每条残差只连 1 位姿 + 1 路标）。Huber 核只加在视觉项，论文给的依据（原话）："for reprojection error we use a robust Huber kernel $\rho_{Hub}$ to reduce the influence of spurious matchings, while for inertial residuals it is not needed, since miss-associations do not exist"——视觉匹配有外点（需鲁棒核截断大残差的影响），IMU 预积分按时间戳对齐、不存在错配（残差本身已含噪声模型，不需要再截断）。鲁棒核的机制与教程第 08 章 08.2③ 的定性说明一致（Huber 在小残差区取二次、大残差区取线性，把外点对正规方程 (8.5) 的污染限幅）。另注意式 (4) 没有先验项：作为"种子问题"它只有 IMU + 视觉两类观测因子；跟踪阶段把它裁成"只优化最近两帧状态、地图点固定"的简化版，建图阶段取共视关键帧滑窗、窗口外位姿固定（§V-C）——这与教程 (10.9) 边缘化先验因子的取舍不同（VINS-Mono 滑窗保留先验因子，ORB-SLAM3 靠共视窗口 + 异步全局 BA 兜住历史信息），两种工程路线在教程 10.3 已对比。式 (4) 是"种子完备"的完整问题：论文同时警告它 "requires good initial seeds to converge"——这正是 4.3 初始化存在的理由。

### 4.3 IMU 初始化：三阶段 MAP（式 (5)–(10)）

初始化把"传感器不确定性算进估计"分三步走（§V-B）：

**第 1 步：视觉 MAP。** 跑纯单目 2 秒（4 Hz 插关键帧），视觉 BA 得到上到尺度的轨迹 $\mathbf{T}_{0:k}\cdot\bar{\mathbf{p}}_{0:k}$（bar 记"up-to-scale"量；依据：单目尺度不可观，教程 (5.9)、(1.7)）。

**第 2 步：纯惯性 MAP。** 固定视觉解，只优化惯性变量（式 (5)）：

$$\mathcal{Y}_k \doteq \{s,\ \mathbf{R}_{wg},\ \mathbf{b},\ \mathbf{v}_{0:k}\} \tag{5}$$

$s \in \mathbb{R}^+$ 是视觉解的尺度因子，$\mathbf{R}_{wg} \in SO(3)$ 把惯性系重力对齐到世界系（$\mathbf{g} = \mathbf{R}_{wg}\mathbf{g}_I$），$\mathbf{b} = (\mathbf{b}^s, \mathbf{b}^p) \in \mathbb{R}^6$ 是初始化期间视为常值的零偏，$\mathbf{v}_{0:k}$ 是各关键帧 body 速度。后验（式 (6)）：

$$p\big(\mathcal{Y}_k \mid \mathcal{I}_{0:k}\big) \propto p\big(\mathcal{I}_{0:k} \mid \mathcal{Y}_k\big)\, p\big(\mathcal{Y}_k\big) \tag{6}$$

依据：贝叶斯定理 + 分母与待估量无关（与教程 (1.3)→(1.4) 同一条链）。因式分解（式 (7)）：

$$\mathcal{Y}_k^* = \arg\max_{\mathcal{Y}_k}\ p(\mathcal{Y}_k)\prod_{i=1}^{k} p\big(\mathcal{I}_{i-1,i} \mid s,\ \mathbf{R}_{wg},\ \mathbf{b},\ \mathbf{v}_{i-1},\ \mathbf{v}_i\big) \tag{7}$$

依据：$\mathcal{I}_{0:k} \doteq \{\mathcal{I}_{0,1}, \dots, \mathcal{I}_{k-1,k}\}$，给定 $\mathcal{Y}_k$ 后每个区间的似然只经该区间的预积分量（仅依赖 $\mathbf{b}$ 与区间内 IMU 测量）与区间两端速度进入，而各区间预积分噪声是相互独立的白噪声（依据：预积分把区间内测量压缩为充分统计量 + 独立高斯噪声，教程 (10.5)–(10.7) 的噪声模型）——条件独立因此成立，连乘可分解。注意 $s, \mathbf{R}_{wg}$ 不单独出现：它们通过把视觉解缩放 / 旋转到真尺度与重力对齐来影响速度与位置状态，从而进入每段似然。

取负对数得式 (8)——**MAP 形式的完整负对数**，逐步推导：

1. 取 $-\log$（依据：$\log$ 单调，$\arg\max \to \arg\min$，教程 (1.4)→(1.6) 同一变形）。
2. 先验项：$p(\mathcal{Y}_k)$ 只在 $\mathbf{b}$ 上非平凡，取零均值高斯——$-\log p(\mathbf{b}) = \tfrac{1}{2}\|\mathbf{b}\|^2_{\Sigma_b} + \text{const}$（依据：高斯负对数 = 马氏范数 + 与变量无关的常数；常数丢弃，教程 (8.2) 的推导链）。论文原话说明其作用："adding a prior residual that forces IMU biases to be close to zero. Covariance matrix $\Sigma_b$ represents prior knowledge about the range of values IMU biases may take"——没有它，偏置在不可观方向上的解会漂。
3. 似然项：每段 $-\log p(\mathcal{I}_{i-1,i}\mid\cdot) = \tfrac{1}{2}\|\mathbf{r}_{\mathcal{I}_{i-1,i}}\|^2_{\Sigma_{\mathcal{I}_{i-1,i}}} + \text{const}$（依据：预积分噪声 $\boldsymbol{\eta}^{\Delta}_{ij} \sim \mathcal{N}(\mathbf{0}, \Sigma_{\mathcal{I}_{i-1,i}})$（教程 (10.7) 前提），而残差 (2) 正是噪声经一阶线性化后的像——马氏范数即其负对数）。
4. 合并、弃常数（式 (8)）：

$$\mathcal{Y}_k^* = \arg\min_{\mathcal{Y}_k}\ \Big(\|\mathbf{b}\|^2_{\Sigma_b}\ +\ \sum_{i=1}^{k}\big\|\mathbf{r}_{\mathcal{I}_{i-1,i}}\big\|^2_{\Sigma_{\mathcal{I}_{i-1,i}}}\Big) \tag{8}$$

这就是"先验 × IMU 残差"的纯惯性 MAP：视觉解不上场（被当作常数，论文指出这与式 (4) 的关键差别即在此），先验把零偏钉在零附近，IMU 残差负责定尺度、重力方向与速度。

**流形上的 retraction（式 (9)(10)）**。优化变量含 $\mathbf{R}_{wg} \in SO(3)$，增量必须经指数映射回到流形（依据：教程 (8.9) 的流形更新论证）。绕重力方向的旋转不改变重力在世界系的表示（依据：绕轴 $\mathbf{n}$ 的旋转保持 $\mathbf{n}$ 本身不动，Rodrigues 公式直接结论；论文原话 "Since rotation around gravity direction does not suppose a change in gravity"）——该方向对 IMU 残差不可观，参数化时固定为零分量，只留两个角度（式 (9)）：

$$\mathbf{R}_{wg}^{new} = \mathbf{R}_{wg}^{old}\ \mathrm{Exp}\big((\delta\alpha_g,\ \delta\beta_g,\ 0)\big) \tag{9}$$

尺度取乘性更新保证正性（式 (10)）：

$$s^{new} = s^{old}\exp(\delta s) \tag{10}$$

依据：$s \in \mathbb{R}^+$ 是约束集，指数映射把无约束的 $\delta s \in \mathbb{R}$ 精确映回正实数——与 (9) 同为 retraction 思想（增量定义在切空间、更新精确保持在流形上）。

**第 3 步：视觉-惯性 MAP。** 拿第 2 步的好初值做联合优化（式 (4) 的结构 + 同款零偏先验 + 共享偏置），随后立即做一次 5 秒的 VI BA，15 秒内收敛到 1% 尺度误差——此时地图称为**成熟（mature）**：尺度、IMU 参数与重力方向均已可靠。论文对比：ORB-SLAM-VI 需 15 s 才有首个尺度估计，VI-DSO 需 20–30 s 收敛到 1%——MAP 式初始化"在 IMU 初始化阶段就全依赖 MAP 估计"（摘要）是快而准的原因。

### 4.4 单目配置的似然写法（按论文实际内容，含与预期的一处出入）

论文对单目的处理**没有**独立的似然公式：式 (3) 中抽象的 $\Pi$ 与协方差 $\Sigma_{ij}$ 对单目 / 双目 / RGB-D / 鱼眼统一生效，单目的特殊性不在似然、而在尺度——地图上到尺度，故地点识别对齐用 $\mathbf{T}_{nm} \in \mathrm{Sim}(3)$（地图不成熟时）或 SE(3)（成熟 VI 图），评估时以 Sim(3) 对齐算尺度误差（§VII-A：单目按 7 自由度对齐、误差经 $\mathrm{Sim}(3)$ 的 $s$ 报 $|1-s|$）。

**核对声明（重要）**：任务预设的"ML 单目（Map Light）：本质图上 MLE 拟合多尺度关键点 $\mathbf{z}_{ij}$ 重投影似然"一节，经全文检索（likelihood / MLE / octave / multi-scale / Map Light / $z_{ij}$）**未见于本 PDF**——论文唯一的"Maximum Likelihood"字样是重定位求解器 ML-PnP（§IV-A，引 [73][74]），唯一的似然字样在式 (6)(7) 的 IMU 初始化。为不杜撰式号，本节只写论文实际内容（如上），多尺度关键点的 MLE 拟合不作为本文公式引用。

### 4.5 回环与地图合并：Sim(3) 对齐与本质图位姿图（§VI）

**地点识别六步**（§VI-A，机制而非公式）：

1. **DBoW2 候选**：对活跃关键帧 $K_a$ 查全 Atlas 数据库，取最相似三帧、排除共视邻居；
2. **组装局部窗口（local window）**：$K_a$ 及其最佳共视邻居 + 它们观测的全部地图点；
3. **3D 对齐变换**：RANSAC 在局部窗口与匹配地图的 3D-3D 对应上求 $\mathbf{T}_{nm}$——单目或地图不成熟时 $\mathbf{T}_{nm} \in \mathrm{Sim}(3)$，否则 $\in SE(3)$；两种情形都用 Horn 算法（引 [77]，绝对定向闭式解：教程 09.2 ⑤ 的去质心 / 范数比尺度 / 互协方差 SVD 三步）以三点最小集出假设、按变换后地图点的重投影误差投票；
4. **引导匹配精化**：用 $\mathbf{T}_{nm}$ 把两图地图点互投、双向（bidirectional）匹配扩充，再以双向重投影误差为目标非线性优化精化（内点过阈值则再迭代一轮、缩窗重搜）；
5. **三共视关键帧验证**：不再等 DBoW2 连续三次命中（那会延迟或漏检），而是在活跃地图中找与 $K_a$ 共视、且各含两条以上窗口匹配的关键帧来验证 $\mathbf{T}_{nm}$——验证所需信息多半已在图里，召回由此提高；
6. **VI 重力方向验证**：地图成熟时 $\mathbf{T}_{nm} \in SE(3)$，检查其 roll/pitch 是否低于阈值——重力方向已由 IMU 钉死，两图对齐后的残余倾斜即假阳性证据。

**合并四步**（§VI-B，对应图 3 的焊接 BA 因子图）：

1. **缝合窗口组装**：$K_a$、$K_m$ 各自共视邻域 + 观测的地图点，并入前经 $\mathbf{T}_{nm}$ 变换；VI 成熟图直接用 SE(3)，否则 Sim(3)；
2. **地图合并**：$M_a, M_m$ 融为新活跃地图，按匹配去重、更新共视与本质图（新增边即合并过程的"中期关联"成果）；
3. **缝合 BA（welding BA）**：窗口内局部 BA——缝合窗口外的关键帧若观测了窗口地图点，则以其局部地图点代入、位姿固定（图 3a/3b：蓝色重投影误差项 + 黄色 IMU 预积分项 + 紫色零偏随机游走项；$K_m$ 前一关键帧位姿固定以定 gauge，$K_a$ 最优关键帧仅位姿可优化）；
4. **本质图优化**：对整张合并图做位姿图优化（缝合窗口关键帧固定），把校正传播全图。

**回环（§VI-D）**是合并的特例——两关键帧同属活跃地图：拼装焊接窗口、去重并建新边后，位姿图优化传播校正，最后做全局 BA（VI 情形仅当关键帧数低于阈值才做，防计算爆炸）。

**与 ORB-SLAM 一代式 (8) 的关系**：一代（[精读](./ORB-SLAM_TRO2015.md) §4.6）把本质图位姿图的边误差写成 $\mathbf{e}_{i,j} = \log_{\mathrm{Sim3}}\big(S_{ij}\,S_{jw}\,S_{iw}^{-1}\big)$，误差向量 $\in \mathbb{R}^7$（3 平移 + 3 旋转 + 1 尺度，Sim(3) 定义即教程 (9.4)）。ORB-SLAM3 论文**未再给出位姿图公式**，机制上沿用同一条 Sim(3)/SE(3) 位姿图路线（单目 7 维、VI 成熟图 6 维），真正的演进在**边约束的来源与质量**：一代回环边靠 Sim(3) 非线性精化（式 (10)(11)），三代靠"Horn 闭式初值 → 引导匹配 BA 精化 → 三关键帧验证 → VI 重力验证"四级流水线，且同样的对齐结果直接驱动跨地图合并——位姿图仍是那副骨架，换的是灌进去的约束。

## 5. 实验与结果解读

- **数据集与配置**（§VII）：EuRoC（11 序列 × 4 配置：单目 / 双目 / 单目-惯性 / 双目-惯性）与 TUM-VI（28 序列、6 环境、鱼眼立体-惯性）。**演进线索**：[ORB-SLAM 一代（T-RO 2015）](./ORB-SLAM_TRO2015.md)评测用的是 NewCollege / TUM RGB-D / KITTI——当时 EuRoC（2016 年发布）尚不存在；三代以 EuRoC + TUM-VI 为主战场并首次系统评测 VI 配置，数据集更替本身记录了领域重心从纯视觉 SLAM 向视觉-惯性与 AR/VR 场景的迁移。
- **评测协议**（§VII-A）：指标为 RMS ATE（absolute trajectory error，绝对轨迹误差）；纯单目用 SE(3) 对齐前先按 Sim(3) 求尺度、其余配置用 SE(3) 对齐；尺度误差按 $|1 - s|$ 报告（$s$ 来自 Sim(3) 对齐）；每配置跑 10 次取**中位数**——作者的理由值得记住：鲁棒系统应表现为"10 次执行的颜色热图（图 4）整体一致"，只报均值会掩盖不稳定系统的高方差（图 4 用每格一色显示 10 次执行的 ATE 分布，直观看鲁棒性）。硬件：Intel Core i7-7700 CPU、32 GB 内存，仅用 CPU。
- **总体结论**（摘要与 §VII-A）：比既有方法精度高"two to ten times"；单目-惯性比 MCSKF / OKVIS / ROVIO 精确 5–10 倍（§VII-A 原话）；双目-惯性比 Kimera 与 VINS-Fusion 精确 3–4 倍；比 VINS-Mono 单会话精确 2.6 倍、多会话 3.2 倍（§III-C 原话，归因于中期关联与合并用局部 BA）。
- **逐配置读法**（表 II）：纯视觉列的对照组是 ORB-SLAM2 与 DSO 系——ORB-SLAM3 胜在更早闭环 + 中期关联；VI 列的对照是 VINS-Mono / VI-DSO / BASALT——差距主要出现在无回环序列（中期关联直接起作用）。单目-惯性与纯单目同表对照可见：IMU 带来的是量级改善（尺度直接钉死，免 Sim(3) 对齐），这正是 §4.3 初始化的实际回报。少数未完成全部序列的系统在表中带星号标注——比较时须注意"能否全程跑完"本身就是鲁棒性指标。
- **绝对精度**：EuRoC 无人机序列平均 3.5 cm（摘要）；TUM-VI room 序列（AR/VR 代表场景）9 mm（摘要）；表 IV 显示 room 序列四种配置 ATE 均在毫米到厘米级（单目最差，因 7 自由度对齐 + 无真实尺度）。室内走廊 / room 类序列误差多在 1 cm 以下（§VII-B 定性）；室外长序列（最长约 900 m）多数误差约 1 m 级，个别序列因尺度 / 加计零偏积累出现 10–70 m 级离群（论文定性描述）；dark-tunnel 类 slides 序列中，无回环也可凭 IMU 全程处理。
- **初始化效率**（§V-B）：2 s 达 5% 尺度误差、15 s 收敛到 1%——对应式 (5)–(10) 的三阶段 MAP。
- **运行时间**（§VII-D）：实时 30–40 帧/秒、每秒 3–6 个关键帧；惯性部分在跟踪中开销可忽略；新地点识别每关键帧仅约 10 ms；合并亚秒量级、回环零点几秒到数秒（表 VII，随地图规模浮动），且都在独立线程执行、不影响实时性。（注：论文表 VI 的逐操作毫秒分解针对 EuRoC V202；本文不逐项复录，需要时查原表。）
- **多会话**（§VII-C）：按环境顺序处理多个 session，首个序列建图、后续序列先新建、再与旧图合并并复用——表 V 显示 VI 配置多数房间级误差收敛到厘米级；多会话增益在单目 / 双目上尤其显著（V103、V203 这类单会话困难序列靠复用旧图变稳）。

## 6. 局限与后续影响

- **论文自承的失败案例**（§VIII 原话）："The main failure case of ORB-SLAM3 is low-texture environments"——直接法（DSO/LSD-SLAM 系）在低纹理更鲁棒，但它们只做短期 / 中期关联（引 [27][31]）；作者展望适合四类数据关联的光度法。
- **结构性局限**：纯旋转探索无法积累深度（§VIII 原话 "pure rotations during exploration would not allow to estimate depth"）；地图点典型距离 < 5 m（§VII-B），远点信息弱；特征描述子跟踪在剧烈运动下弱于 Lucas-Kanade 类（论文与 slides 序列上 VINS-Mono / BASALT 的对比即此）。传感器选型结论（§VIII）也值得一记：立体-惯性最准、IMU 频率供位姿适合 AR/VR；纯旋转探索 / 慢运动场合 IMU 难初始化，可改立体；单目想免初始化可借 CNN 深度估计获得可靠单目尺度（作者的展望）。
- **地图表示层面的留白**：本文的地图仍是稀疏地图点 + 共视/本质图——综述（[Cadena et al. 2016](./SLAM-Survey_TRO2016.md)）列出的语义化、长期记忆管理等问题本文并未触碰；稠密 / 神经隐式方向的延伸见教程第 10 章 10.4 的前沿七篇。
- **后续影响**：三阶段 MAP 式 IMU 初始化、Atlas 多地图 + 高召回地点识别、SE(3)/Sim(3) 双模对齐验证成为后续 VI-SLAM 系统的通用组件；"short/mid/long-term 数据关联"的术语经本文系统化后被广泛引用；开源库（UZ-SLAMLab）成为 VI-SLAM 的事实基线——教程第 10 章实战（10.5 节）即直接跑它。

## 7. 与本项目对照

本项目**不建 ORB-SLAM3 的完整模块**：它是三线程并发系统（Tracking / Local Mapping / Loop&Merging），加上 Atlas 状态机（建图、合并、弃图的时机与判据）、四种传感器配置 × 两种相机模型的组合空间，以及对 DBoW2 / g2o 生态的依赖——这些是论文 §I 自认的"system paper"级工程量，超出教学代码的定位。项目采取"机制切片"策略：把构成 ORB-SLAM3 的**理论内核**拆成三个教学模块，函数与论文环节一一对应：

| ORB-SLAM3 环节（论文章节） | 本项目模块 / 函数 | 教程公式 |
|---|---|---|
| Tracking 前端几何：初始化（§III）、PnP 跟踪（§IV-A 重定位几何） | `epipolar.eight_point`（内含 `_normalize_points_hartley`）、`epipolar.decompose_E`、`epipolar.triangulate` | 对极约束 (5.1)、八点法 (5.2)、Hartley 归一化 (5.4)、本质流形 (5.5)、4 候选分解 (5.6)、手性检验 (5.7)、三角化 (5.8) |
| Tracking 位姿估计 / 重定位精化 | `epipolar.pnp_dlt` → `epipolar.pnp_refine`（复用 `core.solver.gauss_newton`，解析雅可比 `epipolar.reprojection_jacobian`） | PnP 残差与雅可比 (5.10)–(5.12)、正规方程 (8.5)、流形更新 (8.9) |
| 地点识别：词袋检索 + 时间一致性（§VI-A 第①②步） | `bowloop.Vocabulary`（`fit` / `transform`）、`bowloop.InvertedIndex`（`add` / `query`）、`bowloop.detect_loop`（`score` 即 L1 评分） | TF-IDF (9.1)、词袋向量 (9.2)、L1 评分 (9.3) |
| 回环 / 合并后的位姿图传播（§VI-B 第④步） | `bowloop.PoseGraph2D`（`add_node` / `add_edge` / `optimize` / `poses`，SE(2) 特例） | 位姿图目标 (9.8)、Sim(3) 定义 (9.4)（7 自由度论证） |
| IMU 预积分与残差（§V-A，式 (2)） | `preint.ImuParams` / `preint.Preintegration`（属性 `delta_R` / `delta_v` / `delta_p` / `cov` / `j_bias`；方法 `correct` / `predict`；`so3_right_jacobian`） | 预积分推导 (10.1)–(10.8)；残差即 (10.10) 同构 |
| IMU 因子入图（式 (2) + 图 2 随机游走项） | `vins.ImuFactor`（包装 `Preintegration`）、`vins.ReprojectionFactor`、`vins.ViBundle` | 紧耦合目标 (10.9)–(10.11) |

两点读法提示：其一，`preint.Preintegration.correct` 对应教程 (10.8)（一阶偏置修正、免重积分），`predict` 是测量模型 (10.5) 的零噪声反解——把式 (2) 的三段残差"状态侧 − 测量侧"对上即可逐行核对；其二，`bowloop.PoseGraph2D` 用 SE(2) 示范 (9.8) 的流形 GN + gauge fixing 结构，单目 ORB-SLAM3 的对应物是 Sim(3) 位姿图（7 维切空间），维度不同、优化结构同源（教程 09.3 第 3 步）。

**fastslam 模块为何不在对照表里**：[FastSLAM（AAAI 2002）](./FastSLAM_AAAI2002.md)（`fastslam.FastSLAM2D`：粒子 + 条件独立路标 EKF）是概率 SLAM 的**另一条求解路线**（滤波 / 采样 vs 批量 MAP）——它对应综述"经典时代"的第二个公式支柱，与 ORB-SLAM3 的因子图 MAP 在目标函数层面就不同（教程 (1.6) 的两种解法之分，见第 01 章 01.1③）。放着它的原因是让两条路线在同一仓库里可对比，而非功能对应。

**走读建议**：`python3 tests/test_preint.py` 驱动 `Preintegration`，对照教程 (10.4)–(10.8) 看右乘更新与协方差递推；`tests/test_epipolar.py`、`tests/test_bowloop.py` 的健康值与走读路径见[ORB-SLAM 精读](./ORB-SLAM_TRO2015.md) §7；系统级体验按教程第 10 章 10.5 编译运行官方 ORB-SLAM3（EuRoC 单目-惯性入口）。

## 配套阅读

- 本文 PDF：[arXiv-2007.11898_ORB-SLAM3.pdf](../../../papers/slam/classics/arXiv-2007.11898_ORB-SLAM3.pdf)（重点 §III 系统与 Atlas、§V 视觉-惯性 SLAM（式 (1)–(10) 全部在此）、§VI 合并与回环、§VII 实验）
- 预积分原始推导：[Forster et al., On-Manifold Preintegration（T-RO 2017）](../../../papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf)——本文式 (2) 的出处（引 [60][61]），教程 10.2 的完整推导对象
- 姊妹精读：[ORB-SLAM（T-RO 2015）](./ORB-SLAM_TRO2015.md)（三线程架构与本质图式 (8) 的出处）｜[ORB-SLAM2（T-RO 2016）](./ORB-SLAM2_TRO2016.md)（双目 / RGB-D 扩展）｜[PTAM（ISMAR 2007）](./PTAM_ISMAR2007.md)（双线程分治的源头）｜[SLAM 综述（T-RO 2016）](./SLAM-Survey_TRO2016.md)（short/mid/long-term 数据关联术语的体系化）
- 教程：[第 10 章 建图与系统实战](../10_建图与系统实战.md)（(10.1)–(10.11) 与 10.1 对应表）｜[第 09 章 回环检测](../09_回环检测.md)（(9.1)–(9.8)）｜[第 08 章 图优化与 BA](../08_后端-ii图优化与-ba.md)（(8.1)–(8.17)）
- 代码：[projects/slam/epipolar/](../../../projects/slam/epipolar/)｜[projects/slam/bowloop/](../../../projects/slam/bowloop/)｜[projects/slam/preint/](../../../projects/slam/preint/)｜[projects/slam/vins/](../../../projects/slam/vins/)（`ImuFactor` / `ReprojectionFactor` 紧耦合因子图）

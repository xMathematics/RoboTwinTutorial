# 论文精读｜SLAM 综述：Past, Present, and Future（IEEE T-RO 2016）

> **PDF**：[arXiv-1606.05830_SLAM-Survey-Cadena.pdf](../../../papers/slam/classics/arXiv-1606.05830_SLAM-Survey-Cadena.pdf) ｜ **教程**：第 01 章（总览） ｜ **代码**：无（综述，无对应实现；机制切片散见各模块，见 §7）

**式号约定**：下文"式 (1)–(4)"均指本 PDF（arXiv v4，T-RO 32(6): 1309–1332 录用版）原文编号，已逐式与 PDF 核对（四式全部在第 4 页 §II "Maximum a posteriori (MAP) estimation and the SLAM back-end"小节）。"(1.x)/(8.x)/(9.x)"指教程第 01/08/09 章。综述无单一公式主线，§4 改为"框架与形式化"。

## 1. 论文信息与一句话贡献

- **标题**：Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age（Cadena, Carlone, Carrillo, Latif, Scaramuzza, Neira, Reid & Leonard，*IEEE Transactions on Robotics* 32(6)，2016；DOI: 10.1109/TRO.2016.2624754）
- **一句话贡献**：以"MAP 后验 + 因子图"为统一骨架把 30 年 SLAM 研究组织成**前端 / 后端**架构，划分**经典时代（1986–2004）→ 算法分析时代（2004–2015）→ 鲁棒感知时代（robust-perception age）**三个阶段，并把长期运行（鲁棒性、可扩展性）、地图表示、理论保证、主动感知与新技术（新传感器、深度学习）列为开放问题清单——至今仍是 SLAM 领域的"坐标系"式文献。
- **写法定位**：这是 position paper + tutorial：不求覆盖每篇论文，而求给出可批判的框架与问题清单（配套材料含参考文献扩展表与数据集表：https://slam-future.github.io/ ）。

## 2. 问题与动机

- **时代划分**（§I）：三个时代一张表（日期与标志为原文口径）：

  | 时代 | 时期 | 标志 |
  |---|---|---|
  | classical age（经典时代） | 1986–2004 | 三大概率表述确立（EKF-SLAM、Rao-Blackwellised 粒子滤波、最大似然估计）；效率与鲁棒数据关联两大基本挑战定型（Durrant-Whyte & Bailey 两篇 tutorial [7][9] 是该期总结） |
  | algorithmic-analysis age（算法分析时代） | 2004–2015 | 转向**可观测性、一致性、收敛性**等根本性质研究；稀疏性使大规模求解器与开源库成熟 |
  | robust-perception age（鲁棒感知时代） | 2016– | 本综述宣告并给出验收清单 |

  本综述宣告第三阶段——**鲁棒感知时代**——的四条关键要求（§I，原文逐条）：
  1. **robust performance**：低失效率、长时运行、自整定（含失效感知与恢复）；
  2. **high-level understanding**：超越几何重建，给出环境的高层（几何 + 语义 + 物理可供性）理解；
  3. **resource awareness**：按可用传感与计算资源调整计算负载与地图表示；
  4. **task-driven perception**：按任务主动选择感知信息、自适应调整地图复杂度。
- **两个争论问题**（§I 提出、§X 回答）："**Do autonomous robots really need SLAM?**"——综述给三层论证：其一，SLAM 研究直接产出了当前的视觉-惯性里程计（VIO），而 VIN 只是关掉回环的"简化版 SLAM"（§I 原话），研究 SLAM 的回报已经兑现；其二，只有回环才能恢复环境的**真实拓扑**——纯里程计把带回路的楼层拉直成"无限长走廊"，回环事件才揭示"起点 = 终点"，从而找到 A–B–C 之间的捷径（图 1），而度量信息还能剔除虚假回环，"SLAM provides a way to predict and validate future measurements"；其三，许多应用（军事侦察、建筑结构检查）内在要求**全局一致**的地图。故答案为是。 "**Is SLAM solved?**"——问题只在给定 robot/environment/performance 组合（机器人运动类型、环境对称性、精度/时延/建图规模要求、计算资源）后才是良定义的，故答案为"组合定了才能谈解决"，四大方向（鲁棒、高层理解、资源感知、任务驱动）仍未闭合。
- **为什么需要统一形式化**：SLAM 在底层（front-end）与计算机视觉、信号处理、图理论、优化、概率估计交叉，没有一个公共骨架就无法比较方法、定位瓶颈——这正是 §II 用 MAP + 因子图统一的原因，也是教程第 01 章把 (1.1)–(1.6) 作为全书起点的依据（教程 §1.1④ 明引本综述）。

## 3. 方法总览（论文的组织结构）

| 章节 | 主题 | 一句话 |
|---|---|---|
| §II | Anatomy of a Modern SLAM System | MAP + 因子图形式化；前端 / 后端划分（图 2） |
| §III | Long-term Autonomy I: Robustness | 数据关联失效、动态世界、鲁棒回环 |
| §IV | Long-term Autonomy II: Scalability | 稀疏化、连续时间、核外并行、多机器人 |
| §V/VI | Representation I/II: Metric / Semantic Map Models | 度量（稀疏 / 稠密 / 对象级）与语义地图表示 |
| §VII | New Theoretical Tools | 可观测性 / 一致性 / 非凸性 / 对偶 / 验证 |
| §VIII | Active SLAM | 探索–利用决策（TOED、POMDP 框架） |
| §IX | New Frontiers: Sensors and Learning | 事件相机等新传感器；深度学习接入点 |
| §X | Conclusion | 回答两个争论问题 |

阅读时可与教程章节对位：§II ↔ 教程第 01/03/08 章；§III ↔ 第 09 章（回环）与第 06 章（鲁棒残差）；§IV ↔ 第 08 章（稀疏性、边缘化）；§V ↔ 第 10 章（建图）；§VII 为教程未展开的理论层。

## 4. 框架与形式化

### 4.1 统一概率形式化：MAP 后验与因子图（式 (1)–(4)）

设待估变量 $\mathcal{X}$（轨迹 + 地图，教程 (1.3) 的联合状态），量测 $\mathcal{Z} = \{z_k\}_{k=1}^m$，每个量测可写成**观测函数**（measurement/observation function）加零均值高斯噪声：

$$z_k = h_k(\mathcal{X}_k) + \epsilon_k$$

其中 $\mathcal{X}_k \subset \mathcal{X}$ 是该量测涉及的变量子集（依据：观测的局部性——每条量测只约束少数状态，即教程 (1.2) 的抽象化；$\epsilon_k \sim \mathcal{N}(0, \Sigma_k)$，信息矩阵记 $\Omega_k \doteq \Sigma_k^{-1}$）。

**第一步（MAP 定义，式 (1)）**：

$$\mathcal{X}^* \doteq \arg\max_{\mathcal{X}}\ p(\mathcal{X} \mid \mathcal{Z}) = \arg\max_{\mathcal{X}}\ p(\mathcal{Z} \mid \mathcal{X})\, p(\mathcal{X}) \tag{1}$$

依据：贝叶斯定理；等号右侧分母 $p(\mathcal{Z})$ 与 $\mathcal{X}$ 无关、在 $\arg\max$ 下可省略（与教程 (1.3)→(1.4) 完全同一变形）。无先验知识时 $p(\mathcal{X})$ 取均匀分布（不可积常数，可弃），MAP 退化为**最大似然估计**（maximum likelihood estimation）——论文特别提醒：Kalman 滤波与 MAP 在线性高斯情形给出同一估计，一般情形不然。

**第二步（量测独立分解，式 (2)）**：

$$\mathcal{X}^* = \arg\max_{\mathcal{X}}\ p(\mathcal{X}) \prod_{k=1}^{m} p(z_k \mid \mathcal{X}_k) \tag{2}$$

依据：量测条件独立（各量测噪声独立产生，即"measurement noises are uncorrelated"假设；先验 $p(\mathcal{X})$ 保持整体）——与教程 (1.5) 的条件独立展开是同一假设在不同记号下的两次出现。

**第三步（高斯似然，式 (3)）**：

$$p(z_k \mid \mathcal{X}_k) \propto \exp\Big(-\tfrac{1}{2}\,\|h_k(\mathcal{X}_k) - z_k\|^2_{\Omega_k}\Big) \tag{3}$$

依据：单点高斯密度取指数形式；$\|\mathbf{e}\|^2_{\Omega_k} \doteq \mathbf{e}^\top \Omega_k\, \mathbf{e}$ 为马氏范数（注意综述的下标是**信息矩阵**，ORB-SLAM3 论文的下标是协方差——记号相反、数值同物）。先验同理写成 $p(\mathcal{X}) \propto \exp(-\tfrac{1}{2}\|\mathbf{u}\|^2_{\Omega_0})$（$\mathbf{u}$ 为先验均值参数化）。

**第四步（负对数 → 最小二乘，式 (4)）**：

$$\mathcal{X}^* = \arg\min_{\mathcal{X}}\ -\log\Big(p(\mathcal{X}) \prod_{k=1}^{m} p(z_k \mid \mathcal{X}_k)\Big) = \arg\min_{\mathcal{X}}\ \sum_{k=1}^{m} \|h_k(\mathcal{X}_k) - z_k\|^2_{\Omega_k} \tag{4}$$

逐步依据：$\log$ 单调 → $\arg\max$ 变 $\arg\min$（教程 (1.4)→(1.6) 同一变形）；$\log$ 把连乘变连加（依据：$\log ab = \log a + \log b$）；高斯项的 $-\log$ = $\tfrac{1}{2}$ 马氏范数 + 常数，常数与 $\tfrac{1}{2}$ 不影响 $\arg\min$、惯例省去。**这就是非线性最小二乘**，每个 $h_k$ 对应因子图（factor graph）中一条因子：变量为节点、因子为边，因子图的两条结构性优点（论文原话）：可直观可视化复杂问题；可建模异构变量 + 任意互联，且连通性决定稀疏性 → 稀疏求解器（g2o / GTSAM / Ceres / iSAM2，论文列举）可在数秒内解数十万变量。

**与 BA 的关系**（论文 §II 明文）：式 (4) 与 Structure from Motion 里的 BA（bundle adjustment）同源——都出自 MAP 表述；SLAM 的两点特殊：因子不限于投影几何（含 IMU、轮速、GPS 等各种传感器模型），且必须**增量（incrementally）**求解（新量测随运动不断到来）。与教程对照：式 (4) 即教程 (1.6) 的因子图记法、教程 (8.2) 的概率语义来源；Gauss-Newton / LM 求解、稀疏 Schur 补全部是教程第 08 章 (8.3)–(8.17) 的内容。逐条对应：

- 式 (1) ↔ 教程 (1.3)–(1.4)：MAP 定义与贝叶斯分解（同一变形，教程多了 $\mathbf{u}$ 输入的条件独立论证）；
- 式 (2) ↔ 教程 (1.5)：条件独立因子分解（教程按运动/观测链展开，综述按量测独立性展开——假设同源）；
- 式 (3) ↔ 教程 (1.6) 的单项：高斯负对数 = 马氏范数（综述下标 $\Omega_k$ 是信息矩阵，教程 (8.2) 的 $\Lambda_k$ 同物）；
- 式 (4) ↔ 教程 (8.2)：最小二乘目标 + 信息加权；综述随后指出的 $l_1$ / 鲁棒核替换即 (8.2) 的鲁棒化方向。

**SLAM 与滤波的关系**（论文 §II）：MAP（smoothing / full SLAM）比 EKF 类滤波更准——滤波在线性化点处一次性线性化、历史不可撤回；滤波与平滑在"线性化点对 EKF 足够准（如视觉-惯性）+ 滑窗滤波 + 一致性处理到位"时性能接近（论文引 MSCKF [175] 等）。这正对应教程第 07 → 08 章的"范式切换"论证。

### 4.2 前端 / 后端模块划分（图 2）

论文把系统切成（教程 §1.2 的术语出处）：

- **front-end（前端）**：面向传感器的模块，把传感器数据抽象成适合 (4) 的表示——特征提取 + **数据关联（data association）**：把每条量测 $z_k$ 关联到未知变量子集 $\mathcal{X}_k$。前端又含两层：**短期关联**（相邻量测间跟踪特征——论文举的最小例子是"连续两帧中的两个像素观测对应同一个 3D 点"）与**长期关联**（loop closure，把新量测关联到更老的路标）。前端常被实现为把"像素观测反投影到 3D 点"的模块（论文 §II 的特征法前端实例：路标初始化靠多视图三角化）。
- **back-end（后端）**：对前端输出的抽象数据做 MAP 推断，对传感器类型无感；可向前端反馈（图 2 标题注明：back-end can provide feedback to the front-end for loop closure detection and validation）——ORB-SLAM3 的三关键帧验证正是这种反馈回路的实例。
- 图 2 还画出 SLAM 估计的输出侧——**建图**在本文框架里是表示问题（§V/VI）而非独立估计问题。

**一个诊断性引理**（§II 原话）："Visual-Inertial Navigation (VIN) is SLAM: VIN can be considered a **reduced SLAM** system, in which the loop closure (or place recognition) module is disabled"——VIO 是关掉回环模块的 SLAM。这一定位把 [ORB-SLAM3](./ORB-SLAM3_TRO2021.md) 强调的"short/mid/long-term 三层数据关联"直接放进框架：三层 = 前端数据关联的三种时间跨度，后端统一由式 (4) 吸收。前端因此是**传感器依赖的**（论文原话：front-end is sensor dependent, since the notion of feature changes depending on the input stream）——"特征"对相机是描述子、对激光是几何基元、对 IMU 是预积分区间；后端对传感器无感正是这种抽象的回报，也是教程 §1.2"关注点分离"论证的原始出处。

### 4.3 鲁棒感知路线：机制对比

综述 §III 的诊断：算法脆弱性主要是**数据关联失效**——感知混淆（perceptual aliasing，不同地点外观相似 → 错误的正匹配）、假阴性（错拒真量测，损失精度）、未建模动态 + **静态世界假设**（static world assumption：只在单次建图运行内、短时动态（人、门、物体移动）可忽略时成立）。

**综述自身的两个落点**（必须如实标注：综述没有给出鲁棒估计的系统分类，其原文仅两处触及）：

1. **参数化鲁棒估计**：式 (4) 的推导处（§II 原话）"it is also common to substitute the squared $l_2$-norm in (4) with robust loss functions (e.g., Huber or Tukey loss) [112]"——即在负对数似然层面把二次代价换成有界影响函数。机制：Huber 核在残差小于阈值时取二次（保高斯统计效率）、大于阈值时取线性（限幅，外点不再主导正规方程）。教程对应：第 08 章 08.2③ 定性指出"误匹配多时小残差假设退化，实际系统用鲁棒核函数兜底"（该章未给编号公式，故此处回引 08.2③ 而非 (8.x)）；编号实例见[直接法一章](../06_视觉里程计-ii直接法.md)的 (6.8)——DSO 光度代价对每点残差取 $\|\cdot\|^2_{\text{Huber}}$，与 ORB-SLAM3 式 (4) 对重投影残差套 $\rho_{Hub}$ 是同一机制在光度 / 几何两种残差上的应用。

   机制最小推导（自含数学，非综述公式）：把 $\rho(e)$ 代入式 (4) 后对变量求导，由链式法则 $\partial\,\rho(e)/\partial\mathbf{x} = \rho'(e) \cdot \partial e/\partial\mathbf{x}$；与二次情形 $\partial\, e^2/\partial\mathbf{x} = 2e \cdot \partial e/\partial\mathbf{x}$ 对比，等价于把每个残差的权重从常数改为 $w(e) = \rho'(e)\,/\,(2e)$（依据：把 $\rho'(e)$ 因式分解为 $2w(e)\,e$，可除性由核函数在原点的对称性 $\rho'(e) = -\rho'(-e)$ 保证，如 Huber 的 $w(e) = \min(1,\ \delta/\|e\|)$）——大残差 $w \to 0$ 降权、小残差 $w \to 1$ 不变。据此每轮用当前残差算权、再解一次加权正规方程 (8.5)，即**迭代重加权最小二乘**（iteratively reweighted least squares, IRLS）。这一步变换同时说明：鲁棒核不改变求解器结构，只改变喂进 (8.5) 的权重。
2. **最大共识（maximum consensus）**：§III 回环验证处（原话）"In vision-based applications, RANSAC is commonly used for geometric verification and outlier rejection"——随机抽最小样本集、以"能被同一几何模型解释的内点数"为共识度量。教程对应：第 09 章 (9.5)–(9.6)——对极约束回引第 05 章 (5.1)，迭代次数公式 $m \ge \log(1-\eta)/\log(1-\rho^8)$ 逐步推导。机制特点：非参数化、对任意外点分布有效，但只输出二值的内点/外点划分，不产生加权信息矩阵。

**文献中完整的四类鲁棒估计**（下表为教程式综合——源文献主要是后续的对比研究而非本综述，机制按"在式 (4) 求解流程的哪一步拦截外点"划分，便于与教程公式对接）：

| 路线 | 拦截位置 | 机制一句话 | 代价 | 教程对应 |
|---|---|---|---|---|
| M-estimation（Huber/Tukey/Cauchy） | 残差→代价之间 | 大残差降权，迭代重加权（IRLS）归入 GN | 阈值需调；强外点群仍偏置 | 08.2③（定性）、(6.8)（实例） |
| 级联 / 开关变量（switchable constraints、级联滤波） | 因子图边本身 | 给每条回环边挂开关变量 $s_{ij} \in [0,1]$ 与先验，优化中自动关掉坏边 | 变量增多；过松会连好边一起关 | 第 09 章 09.2 验证漏斗的"优化版"思想 |
| 最大共识（RANSAC / 内点计数） | 线性化之前 | 用最小集假设 + 内点投票，先筛后估 | 迭代数随内点率指数增 (9.6)；只给二值判决 | (9.5)–(9.6) |
| 后端韧性校验（综述 §III 引 [43][130][191][238]） | 优化中/前 | 看回环约束在优化中诱导的残差来判有效性；或先验地核查回环是否被里程计支持 [215] | 需保真残差；阈值敏感 | 教程 09.2"风险不对称"论证的工程化 |

综述对这条线的立场（§III "Open Problems"）：迭代非凸优化的两大后果——外点判决依赖初值质量、单个外点混入即劣化估计——使 fail-safe / failure-aware 系统（失效可恢复）仍未达成；**前端–后端更紧的整合**被点名为最有希望的路径。ORB-SLAM3 的三关键帧验证 + VI 重力验证（其 §VI）正是"前端多级验证 + 后端重操作兜底"整合路线的代表作。

**四条路线在真实系统中的落点**（本系列精读的映射，供对照）：ORB-SLAM / ORB-SLAM3 = 最大共识（RANSAC 几何校验）+ 前端多级验证漏斗；DSO / LSD-SLAM = M-estimation（光度 Huber）；VINS-Mono = 视觉重投影 Huber + 词袋回环词匹配校验；开关变量一系的直接实现多见于位姿图 SLAM（g2o 生态）而非视觉系统——视觉系统更依赖前端验证，因为视觉残差的外点在匹配阶段就有结构信息（对极几何、重投影误差）可用，而位姿图边的外点只能靠优化中的残差暴露。

### 4.4 长期运行：可扩展性与地图重用

- **可扩展性**（§IV）的动机先记住：直接线性求解器的内存随变量数平方增长、迭代法线性增长，且重访同一区域会使图不断变稠密——因子图规模必须有界。综述的对策分四条：
  - **节点 / 边稀疏化（sparsification）**：按信息量删减——Ila et al. [115] 用信息论准则只保留非冗余节点与高信息量测；Johansson et al. [120] 干脆不往图里加新节点、把新约束并入现有节点（变量数随探索空间而非运行时长增长）；Kretzschmar et al. [141] 提出选"哪些节点该被边缘化"的信息论判据；Carlevaris-Bianco & Eustice [28] 与 Mazuran et al. [170] 分别提出 GLC（Generic Linear Constraint）因子与 NGS（Nonlinear Graph Sparsification）——共同思路是对被边缘化节点的马尔可夫覆盖（Markov blanket）做稀疏近似。
  - **与教程的接口**：稀疏化的代价即教程 (8.17) 的 fill-in（Schur 补 = 边缘化会引入位姿间耦合块）——综述把"何时删、删什么"归为开放问题，教程 08.4 末尾"滑窗与位姿图是受控使用边缘化"正是同一权衡。
  - **连续时间轨迹**：B 样条基函数参数化轨迹、节点间的任意时刻位姿可插值求出（综述例举 Furgale et al. [88] 的时序基函数批量估计；卷帘快门相机、事件相机是其自然应用场景）。
  - **核外 / 并行与多机器人**：子图划分 + 全局精调的 submapping（可上溯到 Atlas）；多机器人分**中心化**（子图传到中心站推断）与**去中心化**（无中心融合、经公共坐标系达成共识）两族，Gauss-Seidel 类方法的通信量线性于 separator 数——DDF-SAM 的 separators 高斯消元与通信成本的权衡是综述给出的核心对比。
- **地图重用与生命周期**（§IV "Open Problems" + §V/VI）：记忆管理三问（综述原话框架）——**learning, forgetting, remembering**：地图多久更新？何时过期、可否遗忘？忘掉的还能否找回？综述指出"存储表示"（原始点云 vs 压缩重建）与"定位 vs 建图"两种模式的切换都缺理论。ORB-SLAM3 的 Atlas（多地图 + 合并）可视为对"何时遗忘"的一种工程回答：不可信的图丢弃、可合并的图缝合——但何时主动泛化记忆，仍开放。
- **动态世界的两条保守路线**（§III）：当前系统面对动态要么"维护同一地点的多张（随时间的）地图"，要么"用随时间变化的参数参数化单一表示"（综述原话框架，引 [60][140]）——都还未触及"检测、丢弃或跟踪变化"的主动处理；这解释了为何绝大多数 SLAM 论文（含本精读系列全部篇目）仍在静态世界假设内工作。
- **语义与表示**（§V/VI，定性）：稀疏路标 → 稠密（TSDF / 点云）→ 对象级与语义地图的表示谱系；综述的判断是语义信息应进入式 (4) 的联合推断而非后处理分类——这预示了后来语义 SLAM 与场景图的走向。
- **语义建图的三条路线**（§VI，综述明确列出）： SLAM helps Semantics——先用几何 SLAM 建图、再分割 / 分类（早期路线，代表为 Mozos et al. [176]：2D 激光建图 + 关联马尔可夫网络离线融合语义地点；及 Castle et al. [44]、Lai et al. [148]；缺陷是分类错误无法反馈修正几何）； Semantics helps SLAM——用对象知识改进数据关联与重定位（如用已知对象辅助特征匹配）； Joint SLAM and Semantics——几何与语义在同一优化里联合估计（综述点名为最有前景但最缺统一公式的一支；2016 年的联合系统仍太慢、类别太少，无法在线运行）。注意综述同时警告"语义 ≠ 拓扑"：拓扑 SLAM 用地点识别建图、丢度量信息；语义 SLAM 关心的是给地点 / 对象打语义标签——两者是正交的简化方向。
- **语义方向的开题清单**（§VI "Open Problems"）：一致的语义–度量融合（多个时刻的语义证据如何与度量信息一致合并）；语义建图远不止分类（可供性 affordance、实体间关系）；未知 / 未知类（ignorance, awareness, and adaptation——机器人应能发现新类别并按交互更新表示）。这份清单在 2016 年几乎全空，如今对应着开放词汇语义 SLAM / 场景图谱系——是综述"前瞻命中率"最高的部分之一。

### 4.5 理论保证概览（定性）

§VII 的结论按"MAP 估计量的性质 → 算法性质 → 解的全局性"三层：

1. **估计量性质**：无先验时，式 (4) 的 MAP 估计量是**一致的（consistent）、渐近高斯、渐近有效（efficient）**，且对欧氏空间变换不变（论文引 [171]，Théorems 11-12）；一旦加入先验，部分性质即失（不变性不再）。
2. **算法性质**：式 (4) 是非凸问题，迭代求解只保证局部收敛；论文用 sphere-a / torus 两个仿真（图 6）展示"局部极小 = 全图被拉歪"的失效样态——回到教程 08.2③ 对 GN/LM 初值依赖的定性警告，只是这里给出了可视化证据。对策之一是以**旋转平均（rotation averaging）**等子问题求好初值再作 bootstrap。
3. **全局性与验证**：在强对偶条件下位姿图可经半定规划（SDP / 凸松弛）全局求解（论文引 [31][36] 等）；拉格朗日对偶还给出**验证技术（verification）**——判定一个给定 SLAM 解是否最优，可迁移到多机器人、SfM 等场景。开放问题：结论能否从位姿图推广到任意因子图、任意噪声模型。
4. **主动 SLAM 的决策地位**（§VIII，与"理论保证"互补的另一半）：SLAM 至此被当作被动估计问题，而机器人本可以**主动运动以改善建图与定位**——从 Bajcsy 的主动感知与 Thrun 的机器人探索范式出发，决策框架含探索–利用分解（Thrun [230]）、最优实验设计 TOED（A-opt / D-opt / E-opt 效用判据）、POMDP 形式化（引 [123]，计算上不可解、需近似）与信息增益驱动的 action selection；标准流程三步：选候选视点 → 评效用 → 执行并决定继续或终止。综述的判断：效用函数的可比性、未来状态的快速预测、"何时收手"的停止判据都是开放问题——把估计（式 (4)）与控制接起来的这一环，至今仍是 SLAM 与规划的分界面。

### 4.6 传感器与代表系统时间线

综述自身的引用网给出一条谱系（方括号为综述参考文献编号，已核对；单目尺度 / 回环关键节点为本教程串联）：

- **经典滤波时代**：EKF-SLAM（Smith, Self & Cheeseman 1986）→ FastSLAM 粒子滤波（Montemerlo et al., AAAI 2002；综述 §I 把 Rao-Blackwellised 粒子滤波列为经典时代三大概率表述之一，未附单篇编号）——粒子 + 条件独立路标，教程第 07 章与[精读](./FastSLAM_AAAI2002.md)覆盖。
- **关键帧图优化时代**：**PTAM**（Klein & Murray, ISMAR 2007；综述引 [135]）确立跟踪 / 建图分线程 + 关键帧 BA（[精读](./PTAM_ISMAR2007.md)）→ **ORB-SLAM**（Mur-Artal et al., T-RO 2015；综述引 [179]）把词袋回环、共视图、本质图位姿图纳入统一系统（[精读](./ORB-SLAM_TRO2015.md)）→ **ORB-SLAM2/3** 扩展双目 / RGB-D / VI / 多地图（[ORB-SLAM2 精读](./ORB-SLAM2_TRO2016.md)、[ORB-SLAM3 精读](./ORB-SLAM3_TRO2021.md)）。
- **直接法支线**：DTAM（Newcombe et al., ICCV 2011；综述引 [184]）、**LSD-SLAM**（Engel et al., ECCV 2014；综述引 [72]）到半直接 / 稠密直接（SVO / DSO 系）——光度残差替代描述子匹配，教程第 06 章按 (6.7)(6.8) 推导。综述 §V 的对照结论：特征法成熟、可回环；直接法吃透像素信息、低纹理占优，但稠密结构 / 运动只能分两步估。
- **VIO 时代**：MSCKF（Mourikis & Roumeliotis；综述引 [175]）为代表的多状态 EKF，到 Forster 等的**流形预积分**（综述引 [82]）把 IMU 装进因子图——即 ORB-SLAM3 式 (2) 的来源，教程第 10 章 (10.1)–(10.8) 的完整推导。
- **求解器生态**（§II）：稀疏性 + 因子图直接催生开源库——g2o、GTSAM、Ceres、iSAM2、SLAM++（论文列举），"数十万变量的非线性最小二乘数秒内可解"；这正是教程第 08 章 (8.11)–(8.16)（$H$ 稀疏结构 + Schur 补）在工程侧的兑现，也是 ORB-SLAM 系全部优化（局部 BA、位姿图、全局 BA）的底座。
- **新传感器前瞻**（§IX-A）：RGB-D / ToF、光场、**事件相机**（1 ms 时延、140 dB 动态范围、低带宽——综述列其五优点与范式转移难题）等；综述写作时（2016）对深度学习的判断（§IX-B）谨慎而准确：单目深度回归 / 位姿回归可学，但"是否替代几何"存疑，SLAM 需要的是几何与学习的接入框架。

## 5. 实验与结果解读（综述无实验——论证结构替代）

综述不含系统实验，其"论证"以三种替代物呈现，读法如下：

- **表 I（surveying the surveys）**：按年列 2006–2016 的八篇综述 / tutorial，用"覆盖缺口"论证自身定位（此前的综述要么只覆盖经典时代，要么只覆盖单一子方向）。
- **图 1（走廊反例）**：纯里程计建的地图拓扑等价于"无限长走廊"，回环事件才能揭示"起点 = 终点"——一个几何 + 拓扑的最小反例，回答"为什么需要 SLAM"。
- **图 6（sphere-a / torus）**：两个仿真最小实例演示非凸代价的局部极小如何把轨迹拉歪——给 §VII 的理论讨论提供可视化"实验"。读法提示：机器人沿球面 / 环面运动，上图（蓝）为收敛到全局最优的正确轨迹估计，下图（红）为收敛到局部极小的错误估计——正是前端初值（里程计）质量决定后端成败的几何展示，也呼应教程 08.2③ 对 GN/LM 初值依赖的警告。
- **可复现材料**：扩展参考文献（bibTEX）、数据集表与开放问题清单在线维护（slam-future.github.io）——综述把"持续更新"本身做成贡献，这是它与单篇方法论文的根本差别。
- **表 I 的用法**：表 I（Surveying the surveys）按年列 2006–2016 八篇综述 / tutorial（概率方法、滤波、SLAM 回环、鲁棒性 / 一致性、视觉里程计、多机器人、位置识别、理论），每行一个子领域——它既是"此前综述覆盖了什么"的证据，也是读者按子领域找第二篇文献的索引。

## 6. 局限与后续影响

- **时代局限**：写作于 2016，深度学习部分止步于趋势判断（其后 NeRF / 3DGS、可微 BA、基础模型未及覆盖）；学习型回环（NetVLAD 等）与语义 SLAM 在文中仅作前瞻；事件相机部分，硬件成熟度远超当年预期。
- **框架局限**：MAP + 因子图骨架对批量 / 滑窗优化是精确的，但对滤波派（MSCKF 系）只是"可映射"而非"同构"——滤波社区对一致性 / 可观测性的独立贡献（如 OC-EKF）在文中以引用带过；连续时间、事件驱动等非标准表述也只是支线。
- **后续影响**：robust-perception age 的四要求成为后续系统（含 ORB-SLAM3 的自述定位）引用的验收清单；front-end/back-end 术语、short/mid/long-term 数据关联的分层、"SLAM 何时是良定义问题"的追问进入了教科书（含本教程第 01 章——该章 §1.2 的四模块划分与"为什么是估计视角"均明引本综述）；slam-future 社区清单演化为此后综述（视觉位置识别 Lowry et al. [160]、SLAM 近期发展 Dissanayake et al. [64] 等）的接续坐标。
- **对多机器人与资源受限平台的留白**（§III–IV 的 Open Problems 汇总）：多机器人下的离群鲁棒、资源受限平台上的"旋钮"式精度–算力权衡（综述点名手机 / 微型飞行器 / 微型昆虫机器人场景）、带宽受限的分布式一致性——这些在 2016 年是清单，如今多数已有专门子领域，但综述给出的问题表述方式仍被沿用。

## 7. 与本项目对照

- **两种组织方式的互补**：教程按**概念**组织（旋转 → 概率 → 相机 → 前端 → 滤波 → 图优化 → 回环 → 系统），一篇章一个公式主线（(k.x) 编号即为此设计）；本精读系列按**论文**组织（每篇一个系统 / 方法），同一概念在不同论文中的具体化差异（如"鲁棒核"在 DSO 是光度残差 Huber、在 ORB-SLAM3 是重投影 Huber）只有对着一篇篇读才显形。综述恰好是两种组织方式的**映射表**：§4.1 的式 (1)–(4) 就是教程 (1.3)–(1.6) + (8.2) 的公共原型，§4.6 的时间线就是精读系列的篇目顺序。
- **本文档的使用方式**：它是精读系列里唯一的"横向"文档——先读它可以拿到问题地图（什么问题重要、为什么），但公式细节须回到纵向篇目；读完系列再读它，则每条 Open Problems 都能对号入座到某个系统的具体短板。两种用法都比"只读综述"有效。
- **建议阅读顺序**（配合已有精读文档）：① 本综述 §II（框架）对照[教程第 01 章](../01_SLAM问题定义与全景.md)建立骨架；② [PTAM 精读](./PTAM_ISMAR2007.md) → [ORB-SLAM 精读](./ORB-SLAM_TRO2015.md) → [ORB-SLAM2 精读](./ORB-SLAM2_TRO2016.md) → [ORB-SLAM3 精读](./ORB-SLAM3_TRO2021.md)走"关键帧 + 图优化 + 回环 + VI"主线；③ [FastSLAM 精读](./FastSLAM_AAAI2002.md)作对照路线（滤波 / 采样 vs MAP）；④ 回到综述 §III/§IV/§VII 读开放问题——此时每条问题都能对应到已读系统的具体短板。
- **代码层注记**：综述无实现，但 §4.3 表中每条鲁棒路线都能在仓库找到至少一个教学触点——RANSAC 在 `bowloop.detect_loop` 的几何校验环节（配合 `epipolar.eight_point` / `epipolar.decompose_E`，教程 (9.5)–(9.6)）；M-estimation 见 DSO 一章对应的光度 Huber（(6.8)）；开关变量与后端校验未实现（教学范围取舍，定性）。前端 / 后端分离在本仓库的体现是 `core`（求解器 / 李群 / 相机）与各前端模块的分层。

## 配套阅读

- 本文 PDF：[arXiv-1606.05830_SLAM-Survey-Cadena.pdf](../../../papers/slam/classics/arXiv-1606.05830_SLAM-Survey-Cadena.pdf)（重点 §II 形式化（式 (1)–(4)）、§III 鲁棒性、§IV 可扩展性、§VII 理论、§X 结论）
- 教程：[第 01 章 SLAM 问题定义与全景](../01_SLAM问题定义与全景.md)（(1.1)–(1.7)，本综述式 (1)–(4) 的教程版推导）｜[第 08 章 图优化与 BA](../08_后端-ii图优化与-ba.md)（(8.1)–(8.17)）｜[第 09 章 回环检测](../09_回环检测.md)（(9.1)–(9.8)，RANSAC 迭代公式）｜[第 06 章 直接法](../06_视觉里程计-ii直接法.md)（(6.8) Huber 实例）
- 姊妹精读（时间线互引）：[PTAM（ISMAR 2007）](./PTAM_ISMAR2007.md)｜[FastSLAM（AAAI 2002）](./FastSLAM_AAAI2002.md)｜[ORB-SLAM（T-RO 2015）](./ORB-SLAM_TRO2015.md)｜[ORB-SLAM2（T-RO 2016）](./ORB-SLAM2_TRO2016.md)｜[ORB-SLAM3（T-RO 2021）](./ORB-SLAM3_TRO2021.md)
- 代码：[projects/slam/](../../../projects/slam/)（机制切片总览见各模块 docstring 与 `METRICS.md`）

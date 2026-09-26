# 论文精读｜LOAM：实时激光里程计与建图（RSS 2014）

> **PDF**：[papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf](../../../papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf) ｜ **教程**：[第 10 章](../10_建图与系统实战.md)（系统架构扩展）｜ **代码**：[projects/slam/loam2d/](../../../projects/slam/loam2d/)（2D 教学版）

## 1. 论文信息与一句话贡献

- **题目**：LOAM: Lidar Odometry and Mapping in Real-time
- **作者**：Ji Zhang、Sanjiv Singh（卡内基梅隆大学 CMU，Robotics Institute）
- **发表**：RSS 2014（Robotics: Science and Systems；获该届会议最佳论文奖——属公开信息，PDF 内未载）。本仓库 PDF 共 7 页，**式号 (1)–(12) 以此 PDF 为准**（论文到式 (12) 为止，无式 (13)）。
- **一句话贡献**：把"激光里程计 + 建图"这一难问题**拆成两个并行算法**——高频（约 10Hz）低精度的激光里程计（lidar odometry）负责估计扫描内运动、低频（约 1Hz）高精度的激光建图（lidar mapping）负责精细配准与出图——并以**运动补偿**（motion compensation）纠正一次扫描内的运动畸变，在不依赖高精度测距/惯导、无回环的条件下实时输出 6-DOF 里程计与三维地图（KITTI odometry 榜单当时排名第一，平均平移误差为行驶距离的 0.88%）。

## 2. 问题与动机

**要解决什么**：用一台 2 轴旋转激光雷达（本文硬件：Hokyo UTM-30LX + 电机，视场 180°、分辨率 0.25°、扫描频率 40 lines/s、电机转速 180°/s）在 6-DOF 运动中实时完成 ego-motion 估计与建图，且不需要 IMU 或 GPS/INS 辅助。

**三重困难**（§I）：

1. **扫描内运动畸变**（motion distortion）：一次 sweep 持续约 **1s**（电机 180°/s 扫过 180°；扫描 40 lines/s ⟹ 每 sweep 约 40 条 scan 线），期间雷达持续运动，同一帧点云中的点在不同时刻取得——不做补偿直接配准会产生系统性错位。2 轴雷达尤其严重：一个轴转得快，另一轴慢得多，畸变不可忽略。
2. **频率与精度不可兼得**：里程计要快（否则丢帧漂移），建图要准（否则地图糊）。单一算法同时优化"大量变量 + 高频运行"正是传统 SLAM（§I 引 [8]）的计算瓶颈。
3. **无先验传感器**：常用方案靠 IMU/视觉/轮式里程计先去畸变再配准；本文要求纯激光。

**分治思想**（§I、§III）：既然"同时优化大量变量"是难点，就把它拆成两个较易的问题并行解：里程计以高频率、低保真度估计雷达速度（对应查找追求快）；建图以低一个数量级的频率做精细匹配与配准（对应查找追求准）。建图给定充足时间收敛，里程计给定实时性——并行结构（Fig. 3）保证两者都不阻塞对方。

## 3. 方法总览

```
激光雷达 ──▶【点云配准 Point Cloud Registration】一次 sweep 的点云 P_k（畸变未纠）
                     │
     ┌───────────────┴────────────────┐
     ▼（10Hz，§V）                     ▼（1Hz，§VI）
【激光里程计 Lidar Odometry】      【激光建图 Lidar Mapping】
  ① 特征提取（式(1) 曲率）：         ① 同式(1) 提特征，数量 ×10
     边缘点（c 大）+ 平面点（c 小）   ② 特征对累积地图 Q_k 配准：
  ② 对应查找（§V-B）：                 邻域点协方差特征值分解
     边缘点→边缘线，平面点→平面片      定出边缘线/平面片（§VI）
  ③ 运动补偿（式(4)-(8)）：         ③ 距离用与式(2)/(3) 相同公式，
     一次 sweep 内线性插值位姿          LM + 鲁棒拟合精化 T^W_{k+1}
     把每点归算到统一时刻            ④ 5cm 体素栅格降采样入库
  ④ LM 求解（式(9)-(12)）→ T^L_{k+1}   （§VI）
     并把 P_k 重投影到 t_{k+1} 得 P̄_k        │
     └──────────┬─────────────────────────────┘
                ▼（10Hz，Transform Integration）
    位姿合成：mapping 位姿 T^W ⊕ odometry 增量 T^L → 10Hz 地图系位姿输出
```

两个算法共享同一套特征提取与残差公式（式 (1)–(3)、(9)–(12)），差异只在：对应查找的目标（上一帧点云 vs 累积地图）、特征数量（1× vs 10×）、运行频率（10Hz vs 1Hz）、是否需要运动补偿（里程计需要，建图处理的是已去畸变点云、全点同时间戳）。

**读法建议**：先读 §III（双系统问题定义）与 §IV-A（雷达硬件——式 (1) 的邻域定义完全由扫描模式决定），再按 §4.1 → §4.5 → §4.6 的顺序过公式，最后对照 Fig. 3 把 §VI 的位姿合成补进全图；式 (5)–(8) 的方向（重投影点 → 原始时刻）最容易读反，见 §4.5 的链条注记。

## 4. 关键公式推导（式号按论文 PDF 原文 (1)–(12)）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $\mathcal{P}_k$ | 第 $k$ 次 sweep 感知的点云；$\mathbf{X}_{(k,i)}^L$：点 $i$ 在雷达系 $\{L\}$ 中的坐标（§III） |
| $\{L\}$, $\{W\}$ | 雷达坐标系（x 左、y 上、z 前）与世界系（初始时与 $\{L\}$ 重合）（§III） |
| $S$ | 点 $i$ 在同一 scan 中的连续邻域：两侧各半、0.25° 间隔（§V-A） |
| $c$ | 平滑度/曲率（式 (1)） |
| $\mathcal{E}_{k+1}, \mathcal{H}_{k+1}$ | 从 $\mathcal{P}_{k+1}$ 提取的边缘点集与平面点集（§V-B） |
| $\tilde{\mathcal{E}}_{k+1}, \tilde{\mathcal{H}}_{k+1}$ | 重投影到 sweep 起点 $t_{k+1}$ 的对应点集（§V-B） |
| $\bar{\mathcal{P}}_k$ | $\mathcal{P}_k$ 重投影到 $t_{k+1}$ 后的点云（Fig. 6） |
| $d_\mathcal{E}, d_\mathcal{H}$ | 点到线 / 点到面距离（式 (2)/(3)） |
| $T_{k+1}^L \in \mathbb{R}^6$ | sweep $k+1$ 内 $[t_{k+1}, t]$ 的雷达运动：$[t_x, t_y, t_z, \theta_x, \theta_y, \theta_z]^\top$（§V-C） |
| $T_{(k+1,i)}^L$ | $[t_{k+1}, t_i]$ 的运动增量，$s$-比例插值（式 (4)） |
| $T_{k+1}^W$ | sweep $k$ 结束时雷达在地图系中的位姿（§VI） |
| $t_{k+1}, t_i, t$ | sweep 起点、点 $i$ 的时间戳、当前时刻（§V-C） |
| $Q_k, \bar{Q}_{k+1}$ | 地图点云、里程计输出去畸变点云投影到地图系后的点云（§VI） |

### 4.1 特征提取：曲率与选取规则（论文式 (1)，§V-A）

$$c = \frac{1}{|S| \cdot \|\mathbf{X}_{(k,i)}^L\|}\ \Big\| \sum_{j \in S,\, j \neq i} \big( \mathbf{X}_{(k,i)}^L - \mathbf{X}_{(k,j)}^L \big) \Big\|. \tag{1}$$

**逐项解释**：

- 内层求和：点 $i$ 指向邻域内每个点 $j$ 的位移矢量之和。先看量纲——它是"长度"，故 $c$ 的量纲是**无量纲数**（两次除法都是除以长度）。
- 除以 $|S|$（邻域点数）：把"位移和"变成"平均意义"，使 $c$ 不随邻域取点半数变化——否则半径固定的邻域在点密集处自动采到更多点，$c$ 被点数抬高。
- 除以 $\|\mathbf{X}_{(k,i)}^L\|$（点 $i$ 的距离/range）：消除**深度偏置**。雷达按角度均匀采样（0.25° 间隔），近处的两个相邻采样点物理间距约为 $r\Delta\theta$（$r$ 为 range）——**近处点间距天然小、远处天然大**（2D 版同理：`loam2d.smoothness` 的 docstring 明说"远处角采样间距更大"）。平坦表面上位移和近似为采样间隔的累积，与 $r$ 成正比；不除以 range，远处的平面点会被系统性高估 $c$、被误判为边缘。除以 $\|\mathbf{X}_{(k,i)}^L\|$ 后，$c$ 只反映"局部表面的凹凸/断裂程度"，与远近无关。
- 外层范数 $\|\cdot\|$：位移和是矢量（两侧位移方向相反时相消），取模后 $c$ 度量邻域的**整体离散程度**。

**$c$ 的取值语义**：理想平面上各点 $c$ 小且均匀 → 平面点（planar point）取 $c$ 最小者；跨越深度不连续（物体轮廓、遮挡沿）处邻域位移突变 → $c$ 大 → 边缘点（edge point）取 $c$ 最大者。

**选取规则**（§V-A，对应 `loam2d._select_by_smoothness`）：(a) 每 scan 分成 4 个相同子区域，每个子区域最多提供 **2 个边缘点、4 个平面点**——把特征均匀分布到环境中，避免特征挤在近处的一小块表面；(b) 一点被选中后其邻域内其他点不再参选（"None of its surrounding point is already selected"，即非极大值抑制）；(c) $c$ 必须超过（边缘）或低于（平面）阈值。此外剔除两类不可靠点（Fig. 4）：与激光束近乎平行的表面点（入射角过小，range 噪声被放大）；遮挡边界上的"假边缘"点（换个视角就消失了）。

### 4.2 对应查找（§V-B）

sweep $k+1$ 期间点云 $\mathcal{P}_{k+1}$ 逐步增长，每轮迭代把 $\mathcal{E}_{k+1}, \mathcal{H}_{k+1}$ 用当前估计变换重投影到 sweep 起点，得 $\tilde{\mathcal{E}}_{k+1}, \tilde{\mathcal{H}}_{k+1}$；对应关系到上一帧的 $\bar{\mathcal{P}}_k$（KD-tree 索引）中找：

- **边缘线**：$\tilde{\mathcal{E}}_{k+1}$ 中的点 $i$，最近邻 $j \in \bar{\mathcal{P}}_k$，再在 **$j$ 所在 scan 的相邻两条 scan** 中找最近邻 $l$——$(j, l)$ 即对应边缘线。要求 $j, l$ 来自不同 scan：同一条边缘线在一个 scan 内至多穿过一个点（唯一例外是边缘线恰在 scan 平面内退化的情形，此时特征本不该被提取）。用式 (1) 复查 $j, l$ 的平滑度，确认它们确实是边缘点。
- **平面片**：$\tilde{\mathcal{H}}_{k+1}$ 中的点 $i$，最近邻 $j \in \bar{\mathcal{P}}_k$，再在 $j$ 的同 scan 与相邻 scan 各找一点 $l, m$——三点**不共线**（分属不同 scan 保证了这一点），$(j, l, m)$ 即对应平面片；同样以式 (1) 复查三者均为平面点。

### 4.3 点到线残差（论文式 (2)）

$$d_\mathcal{E} = \frac{\big| (\tilde{\mathbf{X}}_{(k+1,i)}^L - \bar{\mathbf{X}}_{(k,j)}^L) \times (\tilde{\mathbf{X}}_{(k+1,i)}^L - \bar{\mathbf{X}}_{(k,l)}^L) \big|}{\big| \bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,l)}^L \big|}. \tag{2}$$

**为什么这是垂距（无跳步推导）**：记 $\mathbf{u} = \bar{\mathbf{X}}_{(k,l)}^L - \bar{\mathbf{X}}_{(k,j)}^L$（直线的方向矢量，$\|\mathbf{u}\| \neq 0$ 因 $j, l$ 不同点）、$\mathbf{w} = \tilde{\mathbf{X}}_{(k+1,i)}^L - \bar{\mathbf{X}}_{(k,j)}^L$（从线上一点指向特征点）。

1. 依据叉积模的几何意义（两矢量张成的平行四边形面积）：$\|\mathbf{w} \times \mathbf{u}\| = \|\mathbf{w}\| \, \|\mathbf{u}\| \sin\theta$，$\theta$ 为 $\mathbf{w}$ 与 $\mathbf{u}$ 的夹角。
2. 依据点到直线距离的投影定义：$\mathbf{w}$ 垂直于直线方向的分量长度即垂距，$d_\perp = \|\mathbf{w}\| \sin\theta$（把 $\mathbf{w}$ 分解为沿 $\mathbf{u}$ 与垂直于 $\mathbf{u}$ 的两个分量，垂直分量长度即 $\|\mathbf{w}\|\sin\theta$）。
3. 代入：$\dfrac{\|\mathbf{w} \times \mathbf{u}\|}{\|\mathbf{u}\|} = \dfrac{\|\mathbf{w}\|\,\|\mathbf{u}\|\sin\theta}{\|\mathbf{u}\|} = \|\mathbf{w}\|\sin\theta = d_\perp$。∎

即式 (2) 的 3D 叉积形式把"点到线距离"写成免反三角函数的紧凑形式。注意残差的"观测"侧是 $\tilde{\mathbf{X}}$——待估变换通过运动补偿决定它（§4.5）。

### 4.4 点到面残差（论文式 (3)）

$$d_\mathcal{H} = \frac{\Big| (\tilde{\mathbf{X}}_{(k+1,i)}^L - \bar{\mathbf{X}}_{(k,j)}^L) \cdot \big( (\bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,l)}^L) \times (\bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,m)}^L) \big) \Big|}{\Big| (\bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,l)}^L) \times (\bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,m)}^L) \Big|}. \tag{3}$$

（印刷注：PDF 中分子按两行叠排——第一行 $(\tilde{\mathbf{X}} - \bar{\mathbf{X}}_{(k,j)})$、第二行叉积——点乘号隐含在换行处，语义为标量三重积，本行按高清渲染核对整理。）

**推导**：记 $\mathbf{n} = (\bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,l)}^L) \times (\bar{\mathbf{X}}_{(k,j)}^L - \bar{\mathbf{X}}_{(k,m)}^L)$。

1. 依据叉积定义，$\mathbf{n}$ 垂直于 $\bar{\mathbf{X}}_{(k,j)}, \bar{\mathbf{X}}_{(k,l)}, \bar{\mathbf{X}}_{(k,m)}$ 三点张成的平面（三点不共线由 §4.2 的跨 scan 选点保证，故 $\mathbf{n} \neq \mathbf{0}$），$\|\mathbf{n}\|$ 等于该平面内以两条棱为边的平行四边形面积。
2. 依据标量三重积的几何意义（平行六面体体积）：$\big| (\tilde{\mathbf{X}} - \bar{\mathbf{X}}_{(k,j)}) \cdot \mathbf{n} \big| = \|\mathbf{n}\| \cdot d_\perp$——把平行六面体立在底面（平行四边形，面积 $\|\mathbf{n}\|$）上，其高恰是 $\tilde{\mathbf{X}}$ 到平面的垂距 $d_\perp$（体积 = 底面积 × 高）。
3. 两式相除：$d_\mathcal{H} = \dfrac{\|\mathbf{n}\| \cdot d_\perp}{\|\mathbf{n}\|} = d_\perp$。∎

即分子（体积）除以分母（底面积）得点到三点所定平面的垂距。

### 4.5 运动补偿：一次扫描内的位姿插值（论文式 (4)–(8)）

**问题**：$\mathcal{P}_{k+1}$ 中的点在不同时刻 $t_i$ 取得，而对应目标 $\bar{\mathcal{P}}_k$ 统一在 $t_{k+1}$ 时刻——不补偿则残差 (2)/(3) 混入了扫描内的真实运动。

**恒速模型**（§V-C："The lidar motion is modeled with constant angular and linear velocities during a sweep"）：一次 sweep 内线/角速度近似恒常 ⟹ 运动增量与经过时间成正比 ⟹ 可对 6-DOF 增量向量**线性插值**。记 sweep $k+1$ 区间 $[t_{k+1}, t]$ 的总增量为 $T_{k+1}^L$，点 $i$（时刻 $t_i$）的增量 $T_{(k+1,i)}^L$ 满足式 (4)：

$$T_{(k+1,i)}^L = \frac{t_i - t_{k+1}}{t - t_{k+1}}\, T_{k+1}^L. \tag{4}$$

记比例 $s = \frac{t_i - t_{k+1}}{t - t_{k+1}} \in [0, 1]$：$s = 0$ 对应 sweep 起点（无增量），$s = 1$ 对应当前时刻 $t$（全额增量）。（注：论文对平移与三个转角同时做线性插值——旋转并未走流形指数插值，这是"恒速模型"假设下的工程选择；1s 的 sweep 内机动剧烈时偏差加大，§VII-B 的 IMU 辅助实验正是针对这一点。）

**变换链（式 (5)–(8)）**：$\tilde{\mathcal{E}}_{k+1}, \tilde{\mathcal{H}}_{k+1}$ 中的点已被重投影到 sweep 起点 $t_{k+1}$；式 (5) 用 $s$-时刻的增量把它们推回各自的原始时刻 $t_i$：

$$\mathbf{X}_{(k+1,i)}^L = \mathbf{R}\, \tilde{\mathbf{X}}_{(k+1,i)}^L + T_{(k+1,i)}^L(1:3), \tag{5}$$

其中 $T_{(k+1,i)}^L(1:3)$ 是该向量的平移分量（$(a:b)$ 记号 = 向量的第 $a$ 至 $b$ 个分量），$\mathbf{R}$ 由其旋转分量经 Rodrigues 公式（式 (6)–(8)）构造：

$$\mathbf{R} = e^{\hat{\omega}\theta} = \mathbf{I} + \hat{\omega} \sin\theta + \hat{\omega}^2 (1 - \cos\theta), \tag{6}$$

$$\theta = \|T_{(k+1,i)}^L(4:6)\|, \tag{7}$$

$$\omega = T_{(k+1,i)}^L(4:6)\, /\, \|T_{(k+1,i)}^L(4:6)\|, \tag{8}$$

$\hat{\omega}$ 为 $\omega$ 的反对称矩阵。（6）即轴角到旋转矩阵的 Rodrigues 公式：转角 $\theta$（7）、单位转轴 $\omega$（8）均从增量向量的旋转分量读出。**补偿的完整链条**：待估量 $T_{k+1}^L$ ──(4) 按 $s$ 缩放──▶ $T_{(k+1,i)}^L$ ──(5)–(8)──▶ 点在各时刻与 sweep 起点之间的往返变换 ──▶ $\tilde{\mathbf{X}}$ 与 $\bar{\mathcal{P}}_k$ 在同一时刻 $t_{k+1}$ 下比较、残差 (2)/(3) 才几何上有效。每轮迭代用当前估计重投影（$\tilde{\mathcal{E}}, \tilde{\mathcal{H}}$ 随迭代更新），对应关系在 $\bar{\mathcal{P}}_k$ 中重新查找。

### 4.6 位姿求解：残差堆叠与 LM（论文式 (9)–(12)）

把 (2) 与 (4)–(8) 复合（$\tilde{\mathbf{X}}$ 依赖 $T_{k+1}^L$），得每条边缘残差（式 (9)）；同理对平面点（式 (10)）：

$$f_\mathcal{E}(\mathbf{X}_{(k+1,i)}^L, T_{k+1}^L) = d_\mathcal{E}, \quad i \in \mathcal{E}_{k+1}, \tag{9}$$

$$f_\mathcal{H}(\mathbf{X}_{(k+1,i)}^L, T_{k+1}^L) = d_\mathcal{H}, \quad i \in \mathcal{H}_{k+1}. \tag{10}$$

对所有特征点堆叠（式 (11)），以 Levenberg-Marquardt 迭代极小化（依据：教程第 08 章 (8.5) 正规方程与 (8.6) LM 阻尼的同一框架）：

$$f(T_{k+1}^L) = \mathbf{d}, \tag{11}$$

$$T_{k+1}^L \leftarrow T_{k+1}^L - (\mathbf{J}^\top \mathbf{J} + \lambda\, \mathrm{diag}(\mathbf{J}^\top \mathbf{J}))^{-1} \mathbf{J}^\top \mathbf{d}, \tag{12}$$

$\mathbf{J} = \partial f / \partial T_{k+1}^L$，$\lambda$ 由 LM 方法自适应确定。

**雅可比的链式结构**（论文只写 $\mathbf{J} = \partial f/\partial T_{k+1}^L$，未展开显式表达式；以下按 (2)/(4)–(8) 的复合关系给出结构解读）：每条残差是三层复合 $d = d\big(\tilde{\mathbf{X}}\big(T_{(k+1,i)}^L\big)\big(T_{k+1}^L\big)\big)$，故

$$\frac{\partial d}{\partial T_{k+1}^L} = \underbrace{\frac{\partial d}{\partial \tilde{\mathbf{X}}}}_{\text{几何项（垂距对点的导数）}} \cdot \underbrace{\frac{\partial \tilde{\mathbf{X}}}{\partial T_{(k+1,i)}^L}}_{\text{刚体变换导数（经 (5)–(8) 的 Rodrigues 链）}} \cdot \underbrace{\frac{\partial T_{(k+1,i)}^L}{\partial T_{k+1}^L}}_{=\, s\,\mathbf{I}_6\ \text{（由 (4)）}}.$$

最后一项正是**运动补偿引入的额外因子**：若无补偿（$s \equiv 1$ 的静态配准），增量变换就是被估变换本身、该项为 $\mathbf{I}$；补偿后每个点按自己的时间戳比例 $s_i$ 缩放被估量对残差的影响——同一帧内的点"看到"的变换各不相同，这正是 LOAM 能在残差层面纠正扫描内畸变的机制。对平移分量与旋转分量（$\theta_x, \theta_y, \theta_z$）分别求导即得 $\partial d / \partial t$ 与 $\partial d/\partial \theta$ 两种列块。

**鲁棒权重**（Algorithm 1 第 15 行；§VI 同样"以鲁棒拟合 [27] 经 LM 重解"）：每行残差按 bisquare 权重加权——距对应越远权重越小，超过阈值的对应判为外点、权重置零。（偏离说明：本文的权重是**基于残差距离的 bisquare 鲁棒权重**，并非某些二手材料所说的"按不确定度 $s$ 加权"。）

### 4.7 低频建图与位姿合成（§VI，复用式 (2)/(3)、(9)–(12)）

建图每 sweep 运行一次（约 1Hz），对**里程计已去畸变**的点云 $\bar{\mathcal{P}}_{k+1}$（投影到地图系记 $\bar{Q}_{k+1}$）优化雷达在地图系的位姿 $T_{k+1}^W$：

- 特征提取同 §V-A 但**数量 ×10**；地图 $Q_k$ 按 **10m 立方体**分块存储、只取与 $\bar{Q}_{k+1}$ 相交的块建 KD-tree。
- 对应以邻域点协方差特征值分解判定几何：在特征点周围某区域取地图点集 $S'$（按特征类型过滤——边缘点只留边缘线上的点、平面点只留平面片上的点），计算 $S'$ 的协方差 $\mathbf{M}$，特征值/特征向量记 $V, E$——若一个特征值显著大于另两个（$S'$ 线状），$V$ 最大者的特征向量即边缘线方向，直线过 $S'$ 的几何中心；若两个大一个小（$S'$ 面状），最小特征值的特征向量即平面法向。
- 距离"using the same formulations as (2) and (3)"、方程同 (9)/(10)——区别仅在 $\bar{Q}_{k+1}$ 中所有点共享同一时间戳，**无需运动补偿**（畸变已在里程计端纠正）。随后仍以鲁棒拟合 + LM（式 (12) 的机制）求解，$\bar{Q}_{k+1}$ 注册入图，地图以 **5cm 体素栅格滤波**降采样。
- **位姿合成**（Fig. 9）：建图输出 $T_{k+1}^W$（1Hz）覆盖一整个 sweep，里程计输出 sweep 内增量 $T_{k+1}^L$（10Hz），二者复合成 10Hz 的地图系位姿输出——即教程第 10 章双速率系统观（10.1 节 Tracking/Local Mapping 分线程）在激光端的最早范本。

## 5. 实验与结果解读

- **硬件与实时性**（§VII）：笔记本 2.5GHz 四核 + 6GiB 内存、ROS；里程计与建图各占一个核。激光 10Hz 采样。
- **室内外测试**（§VII-A）：走廊/大堂（室内）与植被道路/果园（室外），速度 0.5m/s。局部地图精度用"同环境二次采集点云 + 点到面 ICP 匹配误差"评估（Fig. 11 给出四个场景的误差分布密度）：室内误差分布更窄、更靠近零（人造环境特征匹配更准，自然环境的特征匹配不那么精确）；累计漂移（Table I）：走廊 58m **0.9%**、46m **1.1%**；果园 52m **2.3%**、67m **2.8%**——室内约 1%、室外约 2.5% 的相对精度。
- **IMU 辅助消融**（§VII-B，Table II）：陀螺 + 加计经 Kalman 滤波预处理后四组测试全部最优（如走廊 32m：仅 IMU 16.7% / 纯 LOAM 2.1% / LOAM+IMU **0.9%**）；仅用 IMU 朝向反而最差（陀螺积分 5 分钟漂 25°）——结论：**IMU 负责吸收非线性运动，LOAM 优化负责线性运动**。
- **KITTI odometry 基准**（§VII-C）：39.2km 城市/乡村/高速序列，Velodyne 10Hz，上传榜单自动评测——当时**在全部传感模态中排名第一**（含双目视觉里程计 [32][33]），平均平移误差为行驶距离的 **0.88%**（按 100m–800m 轨迹段统计）。论文未在正文展开逐序列数字（引导读者看榜单网站），故本节对具体名次变化不做引申。
- **解读**：两条主线的证据链完整——双算法分治（10Hz + 1Hz，两核）支撑实时性；特征+补偿+鲁棒 LM 支撑精度（0.88% 无 IMU、无回环）。

## 6. 局限与后续影响

**局限**：

1. **无回环**（结论自述"the current method does not recognize loop closure"，列为 future work）：纯里程计 + 建图，漂移只能靠建图精化压低、无法归零。
2. **几何退化环境**：特征提取与配额机制依赖环境中存在边缘/平面结构——隧道（长直墙 + 圆截面：yaw 与侧向平移难约束）、开阔场地（远景稀疏点）等退化场景下 $T_{k+1}^L$ 的某些方向失去约束，LM 会放大噪声。论文未直接处理该问题（后续由第一作者在退化检测/受限求解方向的工作系统化）；本仓库 `loam2d` 用 $J^\top J$ 条件数把它做成了显式诊断量（§7）。
3. **扫描内恒速假设**：sweep 内恒线/角速度模型在急变速（载体颠簸、手持晃动）下失真——Table II 显示加 IMU 预处理可显著缓解。
4. **旋转线性插值**：式 (4) 对转角做线性插值而非流形精确插值，只是恒速模型下的工程近似（小增量时无碍，机动剧烈时与上一条一并恶化）。
5. **特征提取依赖雷达扫描模式**：式 (1) 的邻域 $S$ 按 0.25° 角分辨率、CW/CCW 扫描顺序定义，换雷达需重调。

**后续影响**：LOAM 的"边缘 + 平面特征 / 双速率 / 运动补偿"三件套成为激光 SLAM 的事实基线——LeGO-LOAM（地面优化 + 两种特征）、LIO-SAM、以及各类激光-惯性系统直接继承其残差设计；作者团队后续把该框架与 IMU/视觉组合成 V-LOAM 一系（期刊版见 ROS 包 `loam_back_and_forth` / `loam_continuous` 的公开脉络）。它也是"系统架构论文"的教科书案例（教程第 10 章 10.1 的分线程思想在激光端的源头）。

## 7. 与本项目对照

代码：[projects/slam/loam2d/loam2d.py](../../../projects/slam/loam2d/loam2d.py)（docstring 自带逐条式号对照表）。教学版把传感器与运动都约简到 SE(2) 平面（2D 激光 + `(x, y, theta)` 位姿），逐条对应与**刻意差距**如下：

| 论文机制 | loam2d 实现 | 简化说明 |
|---|---|---|
| 式 (1) 曲率 | `smoothness(points, r=5)`：邻域距离和 ÷（点数 × range） | **逐字保留**双重归一化（含深度偏置的解释）；2D 中 $\|\mathbf{X}_i\|$ 即 range |
| §V-A 选取规则 | `_select_by_smoothness`：阈值 + $r$ 邻域非极大值抑制 + 配额 | **子区域配额 → 全局配额**（论文 4 子区域 × 2 边缘/4 平面；教学版 `n_edge_max=20, n_planar_max=60`）；近平行表面/遮挡边界剔除在 2D 中省略 |
| 式 (2) 点到线 | `point_line_residuals_jacobian`：**3D 叉积 → 2D 标量叉积**，$e = (q_1 - X) \times_2 (q_2 - q_1)/\|q_2 - q_1\|$（带符号垂距）；解析雅可比 $\partial e/\partial t = \mathbf{n}$、$\partial e/\partial\theta = \mathbf{n}\cdot(\mathbf{J}(X - t))$ | 两点对应简化为"最近 2 个同型特征点"（论文要求 $j, l$ 分属不同 scan，教学版无 scan 结构可用）；有限差分校验到机器精度（`tests/test_loam2d.py::test_point_line_jacobian_matches_finite_differences`） |
| 式 (3) 点到面 | 2D 中平面退化为直线，残差与式 (2) 同型、仅特征类型不同 | 建图端以 2×2 协方差特征值分解做直线对应（`register_scan_to_map`，$\lambda_1/\lambda_2 \ge 30$ 判线状、直线过质心、方向取主特征向量）——即 §VI 特征值判据的 2D 版 |
| 式 (4)–(8) 运动补偿 | **无对应物** | 教学场景每帧在单一位姿下整帧渲染，无扫描内畸变可补偿——这正是把式 (4)–(8) 从实现中"减掉"的前提 |
| 式 (9)–(12) LM | `_solve_point_to_line` → `core.solver.gauss_newton`（`lm_lambda` 阻尼、Huber IRLS 代 bisquare、**λI 型阻尼略去论文的 $\lambda\,\mathrm{diag}(\mathbf{J}^\top\mathbf{J})$ 列缩放**） | 对应查找每轮迭代重找（ICP 惯例）+ `max_corr_dist` 门限剔除外点 |
| §VI 双速率 | `run_loam2d(mapping_every=5)`：每帧 odometry 配准、每 5 帧 mapping 对累积地图精化；位姿合成 = 上一帧 map 位姿 ⊕ odometry 增量、关键帧处被精化位姿重置 | 论文 10Hz/1Hz → 教学版 1×/5×；10m 立方体 + KD-tree → `corr_radius` 暴力最近邻；5cm 体素 → `_voxel_downsample`（0.1m，逐体素取质心） |
| 可观测性诊断 | `RegistrationResult.condition_number`（终端线性化点 $J^\top J$ 条件数）+ `healthy` 判定 | 论文未做，教学版补上——对应 §6 的几何退化局限 |

**健康值**（实测，`python3 tests/test_loam2d.py`，7/7 通过；口径见 [METRICS.md](../../../projects/slam/METRICS.md) §2/§3）：

- 双速率对照（`[dual-rate]`）：odometry 末帧平移误差 **17.2 mm**、0.00119 rad（≈0.068°）；mapping 精化后 **2.6 mm**、0.00085 rad（≈0.049°）——**建图回灌把漂移压到约 1/6.6**，即论文"低频建图保精度"主张的活体样本。
- 无特征圆形房间（`[degeneracy]`）：`cond = inf`、`healthy = False`（旋转方向完全不可观测），对照矩形房间 `cond ≈ 9.6` 正常恢复——§6 退化局限的可计算化。
- 单帧配准精度（`[odometry]`）：0.0018 m、`cond ≈ 11.5`；扫描-地图精化（`[scan-to-map]`）：初值 0.064 m → 终值 **0.0009 m**、`cond ≈ 11.6`。

## 配套阅读

- 本文 PDF：[LOAM（Zhang & Singh, RSS 2014）](../../../papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf)（本精读所有式号以此 PDF 为准：式 (1)–(12)，无式 (13)）
- 姊妹精读：[IMU 预积分（T-RO 2017）](./IMU-Preintegration_TRO2017.md)（同属"高频传感器 + 低频关键帧"的测量压缩思想，对象换成 IMU）｜[LSD-SLAM（ECCV 2014）](./LSD-SLAM_ECCV2014.md)（同年、视觉端的"双模块 + 位姿图"路线）｜[FastSLAM（AAAI 2002）](./FastSLAM_AAAI2002.md)｜[ORB-SLAM（T-RO 2015）](./ORB-SLAM_TRO2015.md)
- 教程：[第 10 章](../10_建图与系统实战.md)（10.1 分线程架构与模块-章节-论文对应表；10.2–10.3 为视觉-惯性侧的"压缩测量"对偶）｜[第 08 章](../08_后端-ii图优化与-ba.md)（(8.5) 正规方程、(8.6) LM——式 (12) 的求解器）
- 代码：[projects/slam/loam2d/](../../../projects/slam/loam2d/)（模块 docstring 的式号对照表）｜[projects/slam/METRICS.md](../../../projects/slam/METRICS.md)（ATE/旋转误差口径与健康值表）｜[projects/slam/DEBUG.md](../../../projects/slam/DEBUG.md)

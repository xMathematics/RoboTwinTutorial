# 第 10 讲｜紧耦合 LIO：IEKF 路线与预积分路线

> 对应：书第 8 章（p.307–328）｜大纲：[00_自动驾驶SLAM技术教程大纲.md](../00_自动驾驶SLAM技术教程大纲.md) §讲 10
> 本章推导产出：**D10-1**（IEKF 迭代更新 = MAP 代价上的 GN）、**D10-2**
> （紧耦合点到面残差对状态/外参的雅可比）、**D10-3**（高维观测的等效
> 一次更新与卡方门限）。公式 (10.1)–(10.9)。
> 代码对象：`src/ch8/lio-iekf/iekf.hpp`（`sad::IESKF`）、`lio_iekf.{h,cc}`
> （`sad::LioIEKF`）、`src/ch8/lio-preinteg/lio_preinteg.{h,cc}`
> （`sad::LioPreinteg`）、测试 `test_lio_iekf / test_lio_preinteg`。
> 紧耦合思想的 VIO 版本见精读 [../精读/VINS-Mono_RAM2018.md](../精读/VINS-Mono_RAM2018.md)。

## 10.1 本章定位

第 9 讲的松耦合有一个开环：点云 → 配准在 ESKF **之外**完成，用的位姿
预测一旦不准，去畸变与对应全错，而滤波器对此无能为力（它只看到一个
已经固化的 6 维结果）。紧耦合把这条链收进估计器内部：**原始点即观测，
去畸变、找平面、线性化全部挂在当前误差状态上，随迭代在线重算**。
本章给出两条实现路线——IEKF 滤波（FAST-LIO 一系）与预积分图优化
（LIO-SAM 一系），它们分别是第 4、5 讲两路线在 LIO 上的兑现；
当代工程形态（ikd-tree 增量地图等）作为扩展阅读。

## 10.2 学习目标

1. 用一句话说清松/紧耦合的分界，并列举紧耦合的三个信息论/工程优点；
2. **独立复现 D10-1**：从 MAP 代价推出 IEKF 迭代式，并证明"线性观测时
   一次迭代即 EKF"；
3. **独立复现 D10-2**：点到面残差对位姿与外参的全套雅可比（(2.8) 第三次
   兑现）；
4. 复现 D10-3：N 个点残差如何合成一次 18 维更新，卡方门限从哪来；
5. 跑通 `test_lio_iekf`，做"迭代次数 1/2/5"消融，解释 1 次迭代退化为
   什么。

## 10.3 问题场景（五步法①）

**手头**：第 9 讲的松耦合 LIO。**症状**：①剧烈机动或 IMU 短暂失效时，
预测位姿差 → 去畸变错 → NDT 配准差 → 观测更差（**开环放大**）；
②配准把数万点压成 6 维，点级信息（哪些方向有结构、置信多少）在
滤波器之外被丢弃；③雷达-IMU 外参误差只能离线标定。
**想要**：观测管线（去畸变/找对应/线性化）随状态迭代在线修正、点级
信息直达滤波器、外参可选入状态的框架。

## 10.4 核心内容

### 10.4.1 紧耦合的原理与优点（书 §8.1）

观测函数直接作用在原始点上：

$$\mathbf{z}_i = h_i(\mathbf{x}) + \mathbf{v}_i, \qquad
h_i(\mathbf{x}) = \mathbf{n}_i^\top\big(T_{wi}(\mathbf{x})\,T_{il}(\mathbf{x})\,\mathbf{p}_i - \mathbf{q}_i\big) \tag{10.1}$$

（$\mathbf{p}_i$：帧内 $t_k$ 时刻的雷达点；$T_{il}$：雷达-IMU 外参——**可选状态**；
$T_{wi}$：该点的 IMU 位姿，由帧中间状态 + IMU 递推组合（(9.1) 的闭环版）；
$(\mathbf{q}_i, \mathbf{n}_i)$：局部地图中的对应平面。）
③选型理由（与松耦合对照）：①**无信息瓶颈**——每点独立贡献残差，
退化场景（走廊）里"哪些方向没信息"由 $H^\top R^{-1}H$ 直接可见，而非被
配准器的"收敛成功"掩盖；②**开环变闭环**——去畸变用的位姿就是被估计的
状态，迭代中自动修正；③**外参可估**——外参进状态即在线标定。
代价：每迭代一次要重新投影/查邻域/线性化全部点，计算量显著上升
（工程答案：增量式空间索引，见扩展阅读）。

### 10.4.2 D10-1：IEKF——MAP 代价上的高斯牛顿（书 §8.2.1–8.2.2）

**① 问题场景**：观测更新用普通 EKF 一步线性化：$h_i$ 在**先验**处线性化。
但 (10.1) 强非线性（法向、投影、去畸变都随状态变），先验处的切平面
远远不够。
**② 解决方法**：把更新写成 MAP 代价上的 **GN 迭代**——这就是 IEKF。
**④ 依据**：MAP 目标 (2.9) + 第 2 讲的扰动/雅可比工具。

**⑤ 推导**。设先验 $\hat{\mathbf{x}}$、协方差 $P$；当前迭代点
$\mathbf{x} = \hat{\mathbf{x}} \oplus \delta\hat{\mathbf{x}}$（误差状态注入，(4.12) 同式）。
MAP 代价写成对新增量 $\delta\mathbf{x}$ 的函数（先验项把已走的 $\delta\hat{\mathbf{x}}$
计回）：

$$J(\delta\mathbf{x}) = \|\delta\hat{\mathbf{x}} + \delta\mathbf{x}\|_P^2
+ \sum_i \| \mathbf{e}_i(\mathbf{x} \oplus \delta\mathbf{x}) \|_{R_i}^2 \tag{10.2}$$

线性化（$\mathbf{e}_i(\mathbf{x}\oplus\delta\mathbf{x}) \approx \mathbf{e}_i + H_i\,\delta\mathbf{x}$，
$H_i = \partial\mathbf{e}_i/\partial\delta\mathbf{x}$ 即 D10-2）并置零梯度：

$$\big(P^{-1} + \textstyle\sum_i H_i^\top R_i^{-1} H_i\big)\delta\mathbf{x}
= -\Big(P^{-1}\delta\hat{\mathbf{x}} + \textstyle\sum_i H_i^\top R_i^{-1}\mathbf{e}_i\Big) \tag{10.3}$$

解出 $\delta\mathbf{x}^*$，**注入并重复**：$\delta\hat{\mathbf{x}} \leftarrow \delta\hat{\mathbf{x}} +
\delta\mathbf{x}^*$，$\mathbf{x} \leftarrow \hat{\mathbf{x}} \oplus \delta\hat{\mathbf{x}}$，用**新状态重算**
$\mathbf{e}_i, H_i$（点重新去畸变、重新投影、重新找平面）直到
$\|\delta\mathbf{x}^*\|$ 小于门限（书实现迭代 2–4 次）。收敛后协方差按
(10.3) 的信息阵取逆（Joseph 形式同第 4 讲）。

**两个必答问题**：①线性 $h$ 时一次迭代给出什么？——(10.3) 一次求解
恰好是标准 KF 更新（$\delta\hat{\mathbf{x}} = 0$、$H$ 常值代入，自检题 2）——
IEKF 是 EKF 的严格推广；②迭代到底在修什么？——**去畸变与对应**：
每次 $\delta\hat{\mathbf{x}}$ 变化，每个点的 $T_{wi}T_{il}\mathbf{p}_i$ 重算，
$\mathbf{q}_i$ 可能换邻居——松耦合里这两步是"出厂设置"，这里是被
估计器实时校正的对象。`iekf.hpp` 的 `Predict` 复用 (4.2)(4.9)，
`Update` 即 (10.2)–(10.3) 的循环。

### 10.4.3 D10-2：点到面残差的雅可比（书 §8.2.2–8.2.3）

**⑤ 推导**。点从雷达到世界的链：$\mathbf{p}_b = R_{il}\mathbf{p}_i + \mathbf{t}_{il}$（雷达→IMU），
$\mathbf{p}_w = R\,\mathbf{p}_b + \mathbf{t}$（IMU→世界）。残差与 D8-2 同形但作用在
**误差状态**上（第 4 讲 (4.3) 的右扰动约定）：

$$\mathbf{e}_i = \mathbf{n}_i^\top(\mathbf{p}_w - \mathbf{q}_i) \tag{10.4}$$

逐项扰动（$R \leftarrow R\,\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)$，$\mathbf{t} \leftarrow \mathbf{t} + \delta\mathbf{t}$，
外参 $R_{il} \leftarrow R_{il}\mathrm{Exp}(\delta\boldsymbol\theta_{il}^\wedge)$，
$\mathbf{t}_{il} \leftarrow \mathbf{t}_{il} + \delta\mathbf{t}_{il}$；全部用 (2.8) 的展开
$\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)\mathbf{x} \approx \mathbf{x} + \delta\boldsymbol\theta\times\mathbf{x}$）：

$$\delta\mathbf{e}_i = \mathbf{n}_i^\top\Big(\delta\boldsymbol\theta\times(R\mathbf{p}_b) + \delta\mathbf{t}
+ R\big(\delta\boldsymbol\theta_{il}\times(R_{il}\mathbf{p}_i) + \delta\mathbf{t}_{il}\big)\Big) \tag{10.5}$$

整理成行向量（混合积轮换，同 (8.5) 末步）：

$$\mathbf{J}_i = \Big[\ \big(\mathbf{n}_i\times R\mathbf{p}_b\big)^\top\ \ \ \mathbf{n}_i^\top\ \ \
\big(R\,\mathbf{n}_i\times R\,R_{il}\mathbf{p}_i\big)^\top\ \ \ (R\,\mathbf{n}_i)^\top\ \Big] \tag{10.6}$$

（四块分别对 $\delta\boldsymbol\theta, \delta\mathbf{t}, \delta\boldsymbol\theta_{il}, \delta\mathbf{t}_{il}$；
外参块可整体关掉——固定外参时置零即可，不改变前三块。）
**帧内多时刻**：点在 $t_k$ 采集，其位姿
$T_{wi}(t_k) = T_{wi}(t_m)\cdot T_{i}(t_m\!:\!t_k)$——右因子由 IMU 递推给出
（当步进常数，不进雅可比或按一阶处理），左因子即被估计的帧中间状态
——这就是"去畸变在滤波器内"的代数落点（对照 (9.1) 的开环版）。

### 10.4.4 D10-3：高维观测的等效更新与外点门限（书 §8.2.3）

**① 问题场景**：一帧上万个 (10.4) 残差，难道做一万次 KF 更新？
**② 解决方法**：块对角噪声下**等效一次更新**。设 $\mathbf{e} =
[\mathbf{e}_1^\top,\dots]^\top \in \mathbb{R}^{N}$、$R = \mathrm{diag}(\sigma_k^2)$、
$H = [\mathbf{J}_1; \dots]$，则 KF 更新所需的三个量全部由**求和**给出：

$$S = H P H^\top + R, \qquad
K = P H^\top S^{-1}, \qquad
P^+ = (I - K H)P \tag{10.7}$$

其计算只经 $H^\top R^{-1} H$ 与 $H^\top R^{-1}\mathbf{e}$ 两个 18 维聚合
（依据：分块矩阵代数，$S^{-1}$ 用 Woodbury 恒等式展开后 $N$ 维求逆
消失）——**一帧 = 一次更新**，(10.3) 的 $\sum H^\top R^{-1}H$ 即此。
**外点门限**：单点的马氏距离

$$\mathrm{NIS}_i = \mathbf{e}_i^\top\big(\mathbf{J}_i P \mathbf{J}_i^\top + R_i\big)^{-1}\mathbf{e}_i
\;<\; \chi^2_{1-\alpha}(m) \tag{10.8}$$

（$m$ 为残差维数，面残差 $m = 1$；$\alpha$ 取 0.05/0.01——按自由度查卡方表。
）超限点直接丢弃（动态物体、匹配失败）。**这是紧耦合独有的能力**：
松耦合的配准器也会加权，但点级外点在滤波器内不可见。

### 10.4.5 基于预积分的 LIO（书 §8.4，`sad::LioPreinteg`）

把第 5 讲的图优化搬过来：顶点 = 关键帧的 $T, \mathbf{v}, \mathbf{b}$；边 = 预积分
因子 (5.12)–(5.14) + **激光相对位姿因子**（关键帧点云对局部地图 NDT 配准，
残差即 (9.4)）+ 可选 GNSS 因子（加一条边的事——第 5 讲"可扩展性"的兑现）。
去畸变用预积分的预测轨迹（(5.1)–(5.3) 逐步递推），配准在图**外**做一轮、
图内随 GN 迭代修正相对位姿边——介于松与紧之间：**IMU 侧全紧
（预积分因子按原始 IMU 建模），激光侧经配准降维（相对位姿因子）**。
对照 IEKF-LIO：滤波路线每帧 $O(N)$ 点级迭代、内存恒定；图优化路线
天然保留历史（关键帧图可直接升级为带回环的建图管线，第 11 讲的
后端正是它），但单帧计算更重。书中两实现同数据对跑（§8.5 实验），
精度同量级——选型由"要不要历史"决定。

## 10.5 实践任务

- **操作步骤，不含理论**（豁免声明）：跑 `test_lio_iekf`、`test_lio_preinteg`，
  与第 9 讲 `test_loosely_lio` 同序列对比轨迹误差。
- 消融实验（理论回引）：①迭代次数 1/2/5：1 次迭代 = 先验线性化的
  EKF 式更新，剧烈机动段误差明显增大——验证 D10-1"迭代在修什么"；
  ②打开/关闭外参估计（(10.6) 后两块）：给外参注入 2° 初始误差，
  观察在线标定的收敛曲线；③卡方门限 $\alpha$ 从 0.01 放宽到 0.5：
  动态物体（行人）穿行段的轨迹鲁棒性变化——门限的两难
  （拒真 vs 纳伪）在这里肉眼可见。

## 10.6 易错点与高频疑问

1. **迭代中不复算 $\mathbf{e}_i, H_i$**：只迭代 (10.3) 而不重新投影/找平面
   ——这是"GN 只走一步"，紧耦合的收益（闭环去畸变）一分不剩；
2. **(10.5) 外参块符号**：$R(\delta\boldsymbol\theta_{il}\times\cdot)$ 外面还有一层
   $R$——写成 $R_{il}$ 层次是高频错误，差分校验是最好的保险；
3. **把 $\chi^2$ 门限当"精度旋钮"**：它只应挡外点；残差系统性大应当去查
   外参/时间同步，而不是放宽门限（纳伪会让动态点污染状态）；
4. **协方差忘更新/复用错**：收敛后 $P^+$ 由 (10.7) 给出——直接沿用
   $P$ 会让后续 IMU 传播过信/过疑；
5. **局部地图查询放在迭代外**：平面 $(\mathbf{q}_i, \mathbf{n}_i)$ 随状态更新
   可能换——每迭代重查（或至少重投影后校验距离），否则线性化点
   与残差不一致；
6. **问"FAST-LIO 还多了什么"**：增量式 ikd-tree（免重建地图索引）、
   按距离的反向传播误差、更精细的运动补偿——架构与本讲同构，
   工程增量见其论文（配套阅读）。

## 10.7 自检题

1. 由 (10.2)(10.3) 完整重推 IEKF 的一步更新；说明 $J$ 对 $\delta\mathbf{x}$
   的梯度公式中每一项的出处。
2. 证明：$h$ 线性、单次迭代、$\delta\hat{\mathbf{x}}_0 = 0$ 时 (10.3) 退化为
   标准 KF 更新（第 4 讲 (4.11)）。
3. 完整重推 (10.6) 的外参两块（提示：(2.8) 用两次）。
4. 推导 (10.7) 的 Woodbury 化简：从 18 维信息阵写出等效一次更新的
   $K$ 与 $P^+$。
5. 解释 NIS 门限 (10.8) 中为什么用 $\mathbf{J}_i P \mathbf{J}_i^\top + R_i$
   而不是只用 $R_i$（残差的不确定度里还有什么？）。
6. 列表对比松耦合（第 9 讲）/ IEKF-LIO / 预积分 LIO 三者：观测层次、
   去畸变位置、历史保留、单帧计算量、典型代表系统。

## 10.8 本章要点

- 紧耦合 = 原始点直接进观测 (10.1)：开环的去畸变/对应全部闭环进迭代
  （D10-1 的"迭代在修什么"）；
- IEKF = MAP 代价上的 GN (10.2)–(10.3)：线性观测退化为 EKF，
  非线性观测是本讲的主力引擎；
- 雅可比 (10.6) 是 (2.8) 的第三次兑现（位姿块 + 外参块）；
- 高维观测一次更新 (10.7) + 卡方门限 (10.8)：紧耦合的算力与鲁棒性
  同时成立；
- 预积分 LIO（§10.4.5）把 IMU 侧做紧、激光侧经配准降维——要不要
  历史信息是它与 IEKF 路线的选型分水岭。

## 10.9 配套阅读

- 上一讲：[09_3D里程计与松耦合LIO.md](09_3D里程计与松耦合LIO.md)；下一讲：
  [11_离线地图构建.md](11_离线地图构建.md)（关键帧图升级为建图后端）；
- 精读：[../精读/VINS-Mono_RAM2018.md](../精读/VINS-Mono_RAM2018.md)
  （紧耦合 + 边缘化的 VIO 兄弟版）、[../精读/IMU-Preintegration_TRO2017.md](../精读/IMU-Preintegration_TRO2017.md)；
- 扩展（当代工程形态）：FAST-LIO / FAST-LIO2（IEKF + ikd-tree）、
  LIO-SAM（预积分 + 因子图 + 回环）、Point-LIO——架构均为本讲两条
  路线的工业级实现；
- 代码：`src/ch8/`（`iekf.hpp`、`lio_iekf`、`lio_preinteg`）；
  numpy 对照：`projects/slam/vins/`、`projects/slam/preint/`。

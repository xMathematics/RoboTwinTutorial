# 第 5 讲｜预积分学与基于图优化的 GINS

> 对应：书第 4 章（p.99–134）｜大纲：[00_自动驾驶SLAM技术教程大纲.md](../00_自动驾驶SLAM技术教程大纲.md) §讲 5
> 本章推导产出：**D5-1**（预积分定义）、**D5-2**（噪声传播与协方差递推）、
> **D5-3**（零偏一阶修正）、**D5-4**（残差与 g2o 因子组织）。公式 (5.1)–(5.15)。
> 代码对象：`src/ch4/imu_preintegration.{h,cc}`（`sad::IMUPreintegration`）、
> `src/ch4/g2o_types.{h,cc}`（`EdgeInertial` 等）、`src/ch4/gins_pre_integ`（GINS）、
> `src/ch4/test_preintegration.cc`（与朴素积分对拍）。
> 预积分的因子化视角另有精读详解：[../精读/IMU-Preintegration_TRO2017.md](../精读/IMU-Preintegration_TRO2017.md)。

## 5.1 本章定位

第 4 讲的 ESKF 是**滤波路线**：按时间顺序走，观测来了更新一次。本讲是同一
任务的**优化路线**：把一段 IMU 压成一个"测量"（预积分），与 RTK 一起塞进
图优化。选择优化路线的真正动机要到第 10 讲才完全兑现——图里可以继续加
激光因子、回环因子；本讲用 RTK-only 的 GINS 把机制吃透，并用同一数据集
与 ESKF 对跑，回答"两条路线差多少"。

## 5.2 学习目标

1. 说清预积分解决的核心矛盾：**状态迭代重线性化 vs IMU 积分依赖线性化点**；
2. **独立复现 D5-1**：从状态转移方程的变量替换推出 (5.1)–(5.4)；
3. 复现 D5-2：$\Delta R$ 噪声递推与 $A/B$ 阵协方差递推；
4. 复现 D5-3：零偏变化的一阶修正及其雅可比累积；
5. 说清 (5.12)–(5.14) 残差中每一项的归属（哪些来自测量、哪些来自状态），
   并用数值差分验收雅可比。

## 5.3 问题场景（五步法①）

**手头**：一段 $[t_i, t_j]$ 内的 IMU 流（200 Hz），图优化顶点是 $t_i, t_j$ 两时刻
的状态 $x_i, x_j$。**想要**：一条把 IMU 信息接进图的边。**矛盾**：IMU 积分
(4.2) 的每一步都乘了 $R(t)$——积分结果是 $x_i$ 的函数；图优化每轮 GN/LM 都
会改动 $x_i$，按朴素做法就要**每轮重新积分整段 IMU**，一次求解迭代数百轮
就是数百次积分。**预积分的回答**：把积分改写成"只依赖测量与 $t_i$ 时刻零偏"
的相对量，状态依赖以**一阶修正**的形式外挂。

## 5.4 核心内容

### 5.4.1 D5-1：预积分的定义（五步法）

**② 解决方法**：对 (4.2) 在 $[t_i, t_j]$ 上积分，把所有含 $R_i, \mathbf{v}_i, \mathbf{g}$
的因子用变量替换搬到积分号外。**④ 依据**：(4.2)、右乘小旋转的可交换近似
（每步 $\Delta t$ 内角增量小，D2-1 的一阶条件）。

**⑤ 推导**。姿态：把 $\dot R = R(\tilde{\boldsymbol\omega} - \mathbf{b}_g)^\wedge$ 逐步右乘积分：

$$\Delta\tilde R_{ij} \equiv \prod_{k=i}^{j-1} \mathrm{Exp}\big((\tilde{\boldsymbol\omega}_k - \mathbf{b}_g)\Delta t\big)
\;\Longrightarrow\; R_j = R_i\,\Delta R_{ij} \tag{5.1}$$

速度：对 $\dot{\mathbf{v}} = R(\tilde{\mathbf{a}} - \mathbf{b}_a) + \mathbf{g}$ 积分，右端 $R(t)$ 无法
解析提出 → 用 (5.1) 的逆替换 $R(t) = R_i\Delta R_i(t)$，$\Delta R_i(t)$ 只依赖测量
与 $t_i$ 零偏：

$$\Delta\tilde{\mathbf{v}}_{ij} \equiv \sum_{k=i}^{j-1} \Delta\tilde R_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t
\;\Longrightarrow\; \mathbf{v}_j = \mathbf{v}_i + \mathbf{g}\Delta t_{ij} + R_i\,\Delta\tilde{\mathbf{v}}_{ij} \tag{5.2}$$

位置：对 $\dot{\mathbf{p}} = \mathbf{v}$ 积分并代入 (5.2) 的分段形式：

$$\Delta\tilde{\mathbf{p}}_{ij} \equiv \sum_{k=i}^{j-1}\Big[\Delta\tilde{\mathbf{v}}_{ik}\Delta t + \tfrac12\Delta\tilde R_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}_a)\Delta t^2\Big]
\;\Longrightarrow\; \mathbf{p}_j = \mathbf{p}_i + \mathbf{v}_i\Delta t_{ij} + \tfrac12\mathbf{g}\Delta t_{ij}^2 + R_i\,\Delta\tilde{\mathbf{p}}_{ij} \tag{5.3}$$

（三式右侧的"⟹"即 IMU 测量对状态的**约束形式**——它与 (4.2) 完全等价，
但左侧三个 $\Delta\tilde{\cdot}$ 量**只含测量与 $\mathbf{b}_i$**：这就是预积分。）
跨步近似声明：严格推导中相邻两步的 $\Delta R$ 乘积含 $[\cdot,\cdot]$ 交换项，
一阶近似下丢弃——依据与 D2-1 相同，200 Hz 下每步角增量 $10^{-3}$ rad 级，成立。

### 5.4.2 D5-2：噪声传播与协方差（书 §4.1.3）

**① 问题场景**：(5.1)–(5.3) 用的是带噪测量；边的**信息矩阵**需要
$\mathrm{Cov}(\eta)$，$\eta$ 为三个预积分量的联合噪声。
**② 解决方法**：对每步递推做一阶误差展开。
**④ 依据**：(3.1)(3.2) 的噪声定义 + D2-1（$\mathrm{Exp}(\boldsymbol\phi + \delta) =
\mathrm{Exp}(\boldsymbol\phi)\mathrm{Exp}(J_r(\boldsymbol\phi)\delta)$）。

**姿态项完整推导**。单步 $\mathrm{Exp}((\boldsymbol\omega - \mathbf{n}_g)\Delta t)$，由 BCH 的
$J_r$ 形式，第 $k$ 步噪声贡献 $\delta_k = J_r^k\,\mathbf{n}_{g,k}\Delta t$；各步扰动右乘
传播到乘积（扰动全部推到最右侧，依据：伴随共轭恒等式，十四讲 D11-2 第三步
的同一技巧）：

$$\eta^{\Delta R}_{k+1} = \eta^{\Delta R}_{k} + J_r^k\,\Delta t\,\mathbf{n}_{g,k} \tag{5.4}$$

（一阶近似 $J_r^k \approx I$ 的完整版差异为二阶小量；代码保留 $J_r^k$。）
速度与位置项：对 (5.2)(5.3) 的求和项做同样展开（每项三个噪声源：
$\mathbf{n}_a$ 直接进入、$\mathbf{n}_g$ 经 $\Delta R_{ik}$ 以 (5.4) 的形式进入、
$\Delta\tilde{\mathbf{v}}_{ik}$ 自身的噪声经 $\Delta t$ 传播），合并成矩阵递推：

$$\eta_{k+1} = A_k\,\eta_k + B_k\,\eta_{d,k}, \qquad
\Sigma_{k+1} = A_k\Sigma_k A_k^\top + B_k\,\Sigma_{d}\,B_k^\top \tag{5.5}$$

$$A_k = \begin{bmatrix}
I & 0 & 0\\
-\Delta R_{ik}(\tilde{\mathbf{a}}_k-\mathbf{b}_a)^\wedge\Delta t & I & 0\\
-\tfrac12\Delta R_{ik}(\tilde{\mathbf{a}}_k-\mathbf{b}_a)^\wedge\Delta t^2 & I\Delta t & I
\end{bmatrix} \tag{5.6}$$

$B_k$ 同型（列对应陀螺/加计/零偏游走噪声，含经 $J_r^k$ 的交叉项，完整形式
见 [Forster TRO2017](../精读/IMU-Preintegration_TRO2017.md) (5.x) 与
`imu_preintegration.cc` 的 `BuildNoise`——推导方法与 (5.4) 相同，
逐项为 (5.2)(5.3) 各求和项的一阶展开，不再复写）。

### 5.4.3 D5-3：零偏一阶修正（书 §4.1.4）

**① 问题场景**：预积分量"只依赖 $t_i$ 零偏"不等于"不依赖零偏"——优化中
$\mathbf{b}_i$ 也是顶点，每轮都在变。
**② 解决方法**：对零偏取一阶 Taylor 展开，把依赖外挂为雅可比。
**④ 依据**：与 D5-2 同一套一阶工具，把"噪声扰动"换成"零偏扰动"。

**⑤ 推导（姿态项）**。$\mathbf{b}_g \to \mathbf{b}_g + \delta\mathbf{b}_g$ 使每步指数变
$\mathrm{Exp}((\tilde{\boldsymbol\omega}_k - \mathbf{b}_g - \delta\mathbf{b}_g)\Delta t)
= \mathrm{Exp}\big((\tilde{\boldsymbol\omega}_k - \mathbf{b}_g)\Delta t\big)\,
\mathrm{Exp}\big(-J_r^k\,\delta\mathbf{b}_g\,\Delta t\big)$（依据：D2-1 右侧版）。
逐个把扰动右乘推过乘积（同 D5-2 的伴随技巧），扰动累积为：

$$\Delta\tilde R_{ij}(\mathbf{b}_g + \delta\mathbf{b}_g)
= \Delta\tilde R_{ij}(\mathbf{b}_g)\,\mathrm{Exp}\big((J^{\Delta R}_{b_g}\,\delta\mathbf{b}_g)^\wedge\big), \qquad
J^{\Delta R}_{b_g} = -\sum_{k=i}^{j-1}\Delta\tilde R_{i,k+1}\,J_r^k\,\Delta t \tag{5.7}$$

（求和式即"每个被扰动的因子 × 它右侧的部分乘积"的数学化：第 $k$ 步的
扰动被推到乘积最右后，前面 $k$ 步的部分乘积 $\Delta\tilde R_{i,k+1}$ 预乘它——
这就是积分式雅可比必须**边积分边累积**的原因。速度/位置项同法，得
$J^{\Delta v}_{b_a}, J^{\Delta v}_{b_g}, J^{\Delta p}_{b_a}, J^{\Delta p}_{b_g}$
（代码 `IMUPreintegration::Propagate` 中与积分同步更新）。一阶修正的
适用条件：$\|\delta\mathbf{b}\|$ 小；零偏大幅变化（如初始化期）需触发**重积分**——
代码以 $\delta\mathbf{b}$ 超阈值作为重积分条件，这是 (5.7) 的工程边界。）

### 5.4.4 D5-4：残差与 g2o 因子（书 §4.1.5、§4.2.2–4.2.4）

把 (5.1)–(5.3) 的约束形式两端相减、并用 (5.7) 修正零偏，得三块残差
（$\delta\mathbf{b} = \mathbf{b}_j - \mathbf{b}_i$，顶点为 $T_i,\ x^{vb}_i,\ T_j,\ x^{vb}_j$）：

$$\mathbf{r}_{R} = \ln\Big(\big(\Delta\tilde R_{ij}\,\mathrm{Exp}\big((J^{\Delta R}_{b_g}\delta\mathbf{b}_g)^\wedge\big)\big)^{\!\top} R_i^\top R_j\Big)^{\!\vee} \tag{5.12}$$

$$\mathbf{r}_{v} = R_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij})
- \big(\Delta\tilde{\mathbf{v}}_{ij} + J^{\Delta v}_{b_g}\delta\mathbf{b}_g + J^{\Delta v}_{b_a}\delta\mathbf{b}_a\big) \tag{5.13}$$

$$\mathbf{r}_{p} = R_i^\top\big(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t_{ij} - \tfrac12\mathbf{g}\Delta t_{ij}^2\big)
- \big(\Delta\tilde{\mathbf{p}}_{ij} + J^{\Delta p}_{b_g}\delta\mathbf{b}_g + J^{\Delta p}_{b_a}\delta\mathbf{b}_a\big) \tag{5.14}$$

*读法*（每一项的归属，写作必给）：右括号 = **测量**（预积分 + 零偏修正），
左括号 = **状态预测**（由顶点算出）——残差为零 ⟺ 状态满足测量约束 (5.1)–(5.3)。

**雅可比工作表**（右扰动 $R \leftarrow R\,\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)$）。
以 $\partial\mathbf{r}_R/\partial\delta\boldsymbol\theta_j$ 为例完整推导：扰动只进 (5.12) 的
$R_j$，$R_i^\top R_j \to R_i^\top R_j\mathrm{Exp}(\delta\boldsymbol\theta_j^\wedge)$，由 D2-1 的
右侧小量合并公式 $\ln(\mathrm{Exp}(\mathbf{r}^\wedge)\mathrm{Exp}(\delta^\wedge))
= \mathbf{r} + J_r(\mathbf{r})^{-1}\delta$：

$$\frac{\partial\mathbf{r}_R}{\partial\delta\boldsymbol\theta_j} = J_r(\mathbf{r}_R)^{-1} \tag{5.15}$$

$\partial\mathbf{r}_R/\partial\delta\boldsymbol\theta_i$：扰动经伴随推到最左
（十四讲 D11-2 第三步同技巧），结果 $-J_r(\mathbf{r}_R)^{-1}R_j^\top R_i\,\cdot$ 型伴随项；
$\mathbf{r}_v, \mathbf{r}_p$ 对 $\delta\boldsymbol\theta_i$ 各出现 $R_i^\top(\cdot)^\wedge$ 型块（由 (2.8)）。
**完整 $9\times 9$ 表不再抄写**——落地判据与第 14 讲十四讲相同：对
`EdgeInertial::computeError` 做数值差分（`test_preintegration.cc` 提供了
对拍骨架），最大相对误差 < 1e-6 才算实现正确。

**GINS 组装（书 §4.2.4，`sad::GinsPreInteg`）**：每个 RTK 时刻 $k$ 建局部图：
上一时刻先验边 + 预积分边 (5.12)–(5.14) + RTK 位置边（信息阵取 RTK 协方差），
GN/LM 解出后固定旧顶点、滚动向前——"边缘化"以最朴素的固定lag形式出现，
为第 10 讲预积分 LIO 的同款结构热身。

### 5.4.5 与 ESKF 的对照（书两章的对跑结论）

| 维度 | ESKF（第 4 讲） | 预积分 + 图优化（本讲） |
|------|----------------|------------------------|
| 线性化点 | 每次观测更新后前进 | 每轮迭代整体重线性化 |
| 历史信息 | 压缩进 $P$（有损） | 保留因子（可加回环/激光） |
| 计算量 | 极轻（18 维） | 随窗口增长，GN/LM 迭代 |
| 扩展性 | 加观测类型要改 $H$ | 加因子类型即可（第 10 讲） |
| RTK-only 精度 | 相当（RTK 主导） | 相当 |

（依据：同一数据集两实现的轨迹误差对比，书 §4.3 及配套实验。）

## 5.5 实践任务

- **操作步骤，不含理论**（豁免声明）：跑 `ch4/run_gins_pre_integ`，
  与第 4 讲 ESKF 轨迹叠加比较。
- 理论回引：①`test_preintegration.cc`：预积分量 vs 朴素积分对拍（验证 D5-1）；
  ②给 $\mathbf{b}_g$ 施加 1e-3 扰动，验证 (5.7) 的修正把偏差压回一阶量级
  （再加大扰动看失效——D5-3 的适用边界）；③对 (5.15) 数值差分。

## 5.6 易错点与高频疑问

1. **把预积分当绝对量**：$\Delta\tilde{\cdot}$ 是 $i \to j$ 的**相对**测量，约束形式
   (5.1)–(5.3) 中 $R_i, \mathbf{v}_i, \mathbf{p}_i$ 一个都不能少；
2. **零偏修正漏乘雅可比**：直接把 (5.12) 的 $\mathrm{Exp}(\cdot)$ 项删掉，
   优化中零偏一动就系统性偏差（这是 VINS 类系统"越优化越歪"的常见根因）；
3. **$\Sigma$ 与信息矩阵颠倒**：边的 `information() = Σ^{-1}`，$\Sigma$ 来自
   (5.5)——它随**时间长度**增长，所以长时间无观测的两帧间 IMU 边权重自动变低，
   这是物理正确的"降权"；
4. **重力进了 (5.13)(5.14) 但预积分量里没有 $\mathbf{g}$**：重力被变量替换搬到
   状态侧——这就是第 4 讲"$\mathbf{g}$ 必须进状态"的优化路线版本；
5. **窗口滚动时忘固定旧顶点**：GINS 若重解整个历史会随时间变慢且发散
   （可观测性不足），固定 lag/边缘化是必须的。

## 5.7 自检题

1. 从 (4.2) 出发完整重推 (5.1)–(5.3)，明确写出每步变量替换与丢弃的二阶项。
2. 推导 (5.4)：说明 $J_r^k$ 从哪里来（提示：D2-1 的右侧版）。
3. 由 (5.7) 的推导模式写出 $J^{\Delta v}_{b_a}$ 的求和式（对 (5.2) 逐项扰动）。
4. 推导 (5.15)，再推 $\partial\mathbf{r}_v/\partial\delta\boldsymbol\theta_i$（提示：(2.8)）。
5. 解释为什么预积分协方差 $\Sigma_{ij}$ 随段长增长，以及这对"RTK 中断后的
   第一条预积分边"意味着什么。
6. 把本讲 GINS 与第 4 讲 ESKF 在同一中断实验（屏蔽 RTK 30 s）下对比：
   哪条路线恢复更快？用"线性化点"与"信息保留"解释差异。

## 5.8 本章要点

- 预积分 = 把"状态依赖的积分"改写成"测量 + $t_i$ 零偏依赖的相对量" (5.1)–(5.3)，
  状态依赖以零偏雅可比 (5.7) 一阶外挂——图优化的 IMU 接口问题就此解决；
- 噪声经 $A/B$ 递推 (5.5) 形成边的信息矩阵，时间越长权重越低；
- 残差 (5.12)–(5.14) = 状态预测 −（测量+修正）；雅可比验收只有一条：
  数值差分；
- ESKF 与预积分是同一 MAP 的两条路线，本讲 RTK-only 下精度相当；
  可扩展性差异在第 10 讲兑现。

## 5.9 配套阅读

- 上一讲：[04_ESKF组合导航.md](04_ESKF组合导航.md)（滤波路线）；下一讲：
  [06_基础点云处理.md](06_基础点云处理.md)（进入激光部分）；
- 精读详解：[../精读/IMU-Preintegration_TRO2017.md](../精读/IMU-Preintegration_TRO2017.md)
  （Forster 因子化视角；本讲 $A/B$ 阵与雅可比表的原始出处）、
  [../精读/VINS-Mono_RAM2018.md](../精读/VINS-Mono_RAM2018.md)（预积分在 VIO 中的用法）；
- 十四讲对照：[../十四讲/04_李群与李代数.md](../十四讲/04_李群与李代数.md)（一阶工具箱）；
- 代码：`src/ch4/`（`imu_preintegration`、`g2o_types`、`gins_pre_integ`）、
  numpy 对照 `projects/slam/preint/`。

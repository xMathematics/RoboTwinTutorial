# 第 4 讲｜ESKF 组合导航

> 对应：书 §3.4–3.6（p.72–98）｜大纲：[00_自动驾驶SLAM技术教程大纲.md](../00_自动驾驶SLAM技术教程大纲.md) §讲 4
> 本章推导产出：**D4-1**（误差状态运动学，本系列最核心推导）、**D4-2**（连续 F 阵
> 离散化与协方差传播）、**D4-3**（观测更新、误差注入与协方差投影）。公式 (4.1)–(4.13)。
> 代码对象：`src/ch3/eskf.hpp`（`sad::ESKF`）、`src/ch3/static_imu_init`、
> `src/ch3/run_eskf_gins.cc`。
> §3.5 的工程实现节为操作+走读，按规范对纯步骤部分豁免推导，但每步回引 D4-1–D4-3。

## 4.1 本章定位

第 3 讲证明了两件事：IMU 递推秒级内可信（(3.6)），RTK 厘米级但低频会消失。
本讲把两者装进同一个估计器——**误差状态卡尔曼滤波（ESKF）**，得到第一个
可运行的多传感器融合系统 GINS（GNSS+INS）。它是第 9 讲松耦合 LIO 的直接
底座（把 RTK 观测换成配准位姿即可），也是理解第 10 讲 IEKF 的台阶。
"误差状态"四字的所有含义都在 D4-1 里。

## 4.2 学习目标

1. 说清**为什么必须用误差状态**而不能直接对名义状态做 EKF（三个理由，自检题 1）；
2. **独立复现 D4-1 全链**：从 $R_t = R\,\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)$ 的
   定义出发，推出 (4.4)(4.5)；
3. 复现 D4-2：写出 18 维 $F$ 阵并按 $\Phi \approx I + F\Delta t$ 离散；
4. 复现 D4-3：更新、误差注入 (4.10)、协方差投影 (4.11)；
5. 跑通 `run_eskf_gins`，并解释初始化为何要静止检测。

## 4.3 问题场景（五步法①）

**手头**：200 Hz 的 IMU 流（带噪声与零偏，(3.1)(3.2)）+ 1 Hz 的 RTK 位置流
（厘米级但会丢）。**想要**：全速率（200 Hz）的连续状态 $(\mathbf{p}, \mathbf{v}, R,
\mathbf{b}_a, \mathbf{b}_g, \mathbf{g})$，在 RTK 中断期间靠 IMU 平滑续命、在 RTK
恢复时被迅速拉回。**缺**：一个把"高频递推"当骨架、"低频观测"当修正、
且正确处理旋转量约束（$R \in SO(3)$ 不是欧氏空间）的递推估计器。

## 4.4 核心内容

### 4.4.1 名义状态与运动方程

状态定义（`common/nav_state.h` 的 `sad::NavState`）：

$$\mathbf{x} = \big[\,\mathbf{p},\ \mathbf{v},\ R_{wi},\ \mathbf{b}_a,\ \mathbf{b}_g,\ \mathbf{g}\,\big] \tag{4.1}$$

名义状态用**带噪声的测量**直接递推（这是"名义"的含义——它不含噪声信息）：

$$\dot{\mathbf{p}} = \mathbf{v}, \qquad
\dot{\mathbf{v}} = R(\tilde{\mathbf{a}} - \mathbf{b}_a) + \mathbf{g}, \qquad
\dot R = R(\tilde{\boldsymbol\omega} - \mathbf{b}_g)^\wedge, \qquad
\dot{\mathbf{b}}_{a,g} = \mathbf{0},\ \dot{\mathbf{g}} = \mathbf{0} \tag{4.2}$$

（依据：(2.6) 代入 (3.1)(3.2)；零偏与重力在名义递推中取当前估计常值。）
真实状态满足同一方程、但用真值 $\boldsymbol\omega = \tilde{\boldsymbol\omega} - \mathbf{b}_g -
\delta\mathbf{b}_g - \mathbf{n}_g$（$\mathbf{a}$ 同理）。

### 4.4.2 D4-1：误差状态运动学（五步法，全书最常用推导）

**① 问题场景**：直接 EKF 需要对 9 维旋转矩阵求偏导——过参数化、有正交约束、
且在 $\|\cdot\| = \pi$ 处奇异。**② 解决方法**：另立一个**最小参数化**的误差状态，
只用 3 维表达姿态误差，在恒等元附近线性化——处处无奇异。
**③ 选型理由**：对比直接 EKF：误差量小（近恒等元）→ 线性化更准；$SO(3)$ 的
约束被 $\mathrm{Exp}$ 映射天然满足；协方差最小（18 维而非 21 维）。
代价：多一套"名义 + 误差"双状态簿记（④依据：Solà《Quaternion kinematics…》§5，
书 §3.4.1–3.4.2 沿用）。

**误差状态定义**：

$$\delta\mathbf{p} = \mathbf{p}_t - \mathbf{p},\quad \delta\mathbf{v} = \mathbf{v}_t - \mathbf{v},\quad
R_t = R\,\mathrm{Exp}(\delta\boldsymbol\theta^\wedge),\quad
\delta\mathbf{b}_{a,g} = \mathbf{b}_{t} - \mathbf{b},\quad \delta\mathbf{g} = \mathbf{g}_t - \mathbf{g} \tag{4.3}$$

**⑤ 推导（姿态误差，核心步）**。对 $R_t = R\,\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)$ 求导
（左端用真实运动 (4.2) 的真值版，右端乘积求导 + 一阶展开
$\tfrac{\mathrm{d}}{\mathrm{d}t}\mathrm{Exp}(\delta\boldsymbol\theta^\wedge) \approx (\delta\dot{\boldsymbol\theta})^\wedge$，
依据：小量指数映射的一阶性质，第 2 讲 D2-1 同源）：

$$R_t(\tilde{\boldsymbol\omega} - \mathbf{b}_g - \delta\mathbf{b}_g - \mathbf{n}_g)^\wedge
= R(\tilde{\boldsymbol\omega} - \mathbf{b}_g)^\wedge\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)
+ R(\delta\dot{\boldsymbol\theta})^\wedge$$

左端一阶展开 $\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)\mathbf{x} \approx \mathbf{x} + \delta\boldsymbol\theta\times\mathbf{x}$，
右端第一项展开 $\mathbf{A}^\wedge\mathrm{Exp}(\delta\boldsymbol\theta^\wedge) \approx \mathbf{A}^\wedge + \mathbf{A}^\wedge\delta\boldsymbol\theta^\wedge$
（$\mathbf{A} \equiv \tilde{\boldsymbol\omega} - \mathbf{b}_g$，依据同上）。消去公共项
$R\mathbf{A}^\wedge$、左乘 $R^\top$、取 $\vee$：

$$\mathbf{A}\times\delta\boldsymbol\theta - \delta\mathbf{b}_g - \mathbf{n}_g
= \delta\boldsymbol\theta\times\mathbf{A} + \delta\dot{\boldsymbol\theta}$$

由 $\delta\boldsymbol\theta\times\mathbf{A} = -\mathbf{A}\times\delta\boldsymbol\theta$（叉积反对称）：

$$\boxed{\ \delta\dot{\boldsymbol\theta} = -\mathbf{A}^\wedge\,\delta\boldsymbol\theta - \delta\mathbf{b}_g - \mathbf{n}_g\ } \tag{4.4}$$

**速度误差**：对 $\dot{\mathbf{v}}_t = R_t(\tilde{\mathbf{a}} - \mathbf{b}_a - \delta\mathbf{b}_a - \mathbf{n}_a) + \mathbf{g} + \delta\mathbf{g}$
做同样一阶展开（用第 2 讲 (2.8) 的展开技巧 $R\,\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)\mathbf{x}
= R\mathbf{x} - R\mathbf{x}^\wedge\delta\boldsymbol\theta$）：

$$\boxed{\ \delta\dot{\mathbf{v}} = -R(\tilde{\mathbf{a}} - \mathbf{b}_a)^\wedge\,\delta\boldsymbol\theta
- R\,\delta\mathbf{b}_a + \delta\mathbf{g} - R\,\mathbf{n}_a\ } \tag{4.5}$$

**位置与零偏**（两式相减即得，无交叉项）：

$$\delta\dot{\mathbf{p}} = \delta\mathbf{v}, \qquad
\delta\dot{\mathbf{b}}_a = \mathbf{w}_a,\quad \delta\dot{\mathbf{b}}_g = \mathbf{w}_g,\quad
\delta\dot{\mathbf{g}} = \mathbf{0} \tag{4.6}$$

合并 (4.4)–(4.6) 成连续线性系统 $\delta\dot{\mathbf{x}} = F\,\delta\mathbf{x} + \mathbf{n}$，
$\delta\mathbf{x} \in \mathbb{R}^{18}$（顺序 $[\delta\mathbf{p}, \delta\mathbf{v}, \delta\boldsymbol\theta,
\delta\mathbf{b}_a, \delta\mathbf{b}_g, \delta\mathbf{g}]$）。$F$ 只有四个非零非对角块：

$$F = \begin{bmatrix}
 & I & & & &\\
 & & -R(\tilde{\mathbf{a}}-\mathbf{b}_a)^\wedge & -R & & I\\
 & & -\mathbf{A}^\wedge & & -I &\\
 & & & & &\\
 & & & & &\\
\end{bmatrix} \tag{4.7}$$

（每块的出处即 (4.4)–(4.6) 对应项；注意 $-\mathbf{A}^\wedge$ 的负号在 (4.4) 已推导，
抄书时最容易丢。）

### 4.4.3 D4-2：离散化与协方差传播

**① 问题场景**：$F$ 是连续时间阵且随名义状态时变，KDA 精确离散代价高。
**② 解决方法**：一阶 Taylor 离散，200 Hz 下误差可忽略。
**④ 依据**：状态转移阵定义 $\Phi = \dot\Phi F$；一阶解 $\Phi \approx I + F\Delta t$。

$$\Phi_k \approx I + F\Delta t, \qquad
Q_{d,k} \approx \Phi_k\big(\mathrm{diag}(\sigma^2_{n}\Delta t\ \text{与}\ (3.4)\ \text{的游走项})\big)\Phi_k^\top \tag{4.8}$$

噪声离散方差来自 (3.3)(3.4)——白噪声项乘 $\Delta t$（连续方差率 × 时长）、
游走项同样按 (3.4)（③理由：一阶近似下 $Q$ 的主项是对角块；$FQF^\top$ 交叉项
为二阶小量，丢弃）。传播：

$$\delta\mathbf{x}_k \sim \mathcal{N}(\mathbf{0}, P_k)\ \Longrightarrow\
P_{k+1} = \Phi_k P_k \Phi_k^\top + Q_{d,k} \tag{4.9}$$

### 4.4.4 D4-3：观测更新、误差注入与协方差投影

**观测模型**。RTK 位置（天线在车体系的杠杆臂 $\mathbf{l}_a$，依据：(2.1) 刚体变换）：

$$\mathbf{z}_{gps} = \mathbf{p} + R\,\mathbf{l}_a + \mathbf{v}_{gps}, \qquad
H = \frac{\partial h}{\partial\delta\mathbf{x}} = \big[\,I\ \ \mathbf{0}\ \ {-R\mathbf{l}_a^\wedge}\ \ \mathbf{0}\ \ \mathbf{0}\ \ \mathbf{0}\,\big] \tag{4.10}$$

（$-R\mathbf{l}_a^\wedge$ 由 (2.8) 直接代用——这就是第 2 讲说"母公式"的意思。）
轮速计速度观测（§3.5.6）：车体系前向速度 $\Rightarrow h = \mathbf{e}_x^\top R^\top\mathbf{v}$，
$H$ 的速度/姿态块由 (2.8) 得 $-\mathbf{e}_x^\top R^\top$ 与 $-\mathbf{e}_x^\top R^\top\mathbf{v}^\wedge$ 类似项。

**KF 更新（作用在误差状态上）**：

$$K = P H^\top (H P H^\top + R_{obs})^{-1}, \qquad
\delta\hat{\mathbf{x}} = K(\mathbf{z} - h), \qquad
P \leftarrow (I - K H)P(I - K H)^\top + K R_{obs} K^\top \tag{4.11}$$

（末式为 Joseph 形式，③理由：数值稳定、保证对称正定；数学上与标准
$(I-KH)P$ 等价。）

**误差注入（观测更新后的收尾，ESKF 特有）**：把误差并回名义状态、误差清零：

$$\mathbf{p} \leftarrow \mathbf{p} + \delta\hat{\mathbf{p}},\ \ \mathbf{v} \leftarrow \mathbf{v} + \delta\hat{\mathbf{v}},\ \
R \leftarrow R\,\mathrm{Exp}(\delta\hat{\boldsymbol\theta}^\wedge),\ \ \
\mathbf{b}_{a,g} \leftarrow \mathbf{b}_{a,g} + \delta\hat{\mathbf{b}}_{a,g},\ \ \
\mathbf{g} \leftarrow \mathbf{g} + \delta\hat{\mathbf{g}} \tag{4.12}$$

**协方差投影（可选精化，书 §3.4.5）**：注入后误差被重定义为零附近，
新误差与旧误差经 $J_r(\delta\hat{\boldsymbol\theta})$ 相连（依据：D2-1，
$\ln\big(\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)\mathrm{Exp}(\delta\hat{\boldsymbol\theta}^\wedge)\big)
\approx \delta\boldsymbol\theta + J_r(\delta\hat{\boldsymbol\theta})^{-1}\delta\hat{\boldsymbol\theta}$ 的差）：

$$P \leftarrow G P G^\top, \qquad G = \mathrm{diag}\big(I,\ I,\ J_r(\delta\hat{\boldsymbol\theta})^{-1},\ I,\ I,\ I\big) \tag{4.13}$$

（$\delta\hat{\boldsymbol\theta}$ 小时 $J_r \approx I$，工程上常省略——但推导必须知道它在哪。）

### 4.4.5 实现走读与初始化（书 §3.5，操作节豁免）

`src/ch3/eskf.hpp`（`sad::ESKF`）三方法与推导的对应：`Predict` ↔ (4.2)+(4.9)；
`ObserveGps` ↔ (4.10)(4.11)；`ProjectCov/UpdateAndReset` ↔ (4.12)(4.13)。
初始化（`static_imu_init`，静止检测：加计滑动窗口方差低于阈值 → 静止）：
$\mathbf{b}_g \leftarrow$ 陀螺均值（依据：(3.1)，静止时 $\boldsymbol\omega = 0$）；
$R_{wi} \leftarrow$ 重力对准（加计均值方向 ≈ $-R^\top\mathbf{g}$，第 3 讲自检题 4）；
$\mathbf{p} \leftarrow$ 首帧 RTK 投影到 UTM/ENU。**为什么必须静止初始化**：(4.5) 显示
$\delta\mathbf{v}$ 与 $\delta\boldsymbol\theta$ 耦合——姿态初值错了，速度协方差会被
$R(\cdot)^\wedge$ 项持续吹大，RTK 要花远更久才把系统拉回（自检题 5）。

## 4.5 实践任务

- **操作步骤，不含理论**（豁免声明）：跑 `ch3/run_eskf_gins`（NCLT/ULHK 数据），
  与纯 RTK、纯 IMU 两条基线对比轨迹。
- 消融实验（理论回引）：①把 (4.7) 中 $-R(\tilde{\mathbf{a}}-\mathbf{b}_a)^\wedge$ 块清零
  → 观察 RTK 中断段姿态-速度耦合消失、恢复后收敛变慢；②人工屏蔽 RTK 60 s，
  验证漂移量级与 (3.6) 的估计一致；③对 `eskf.hpp` 的雅可比做数值差分校验
  （判据：最大相对误差 < 1e-6）。

## 4.6 易错点与高频疑问

1. **(4.4) 的 $-\mathbf{A}^\wedge$ 符号**：来自 $\delta\boldsymbol\theta\times\mathbf{A} = -\mathbf{A}\times\delta\boldsymbol\theta$，
   推导中省略该步必翻车；
2. **$\mathbf{A}$ 用名义值还是真值**：线性化点用**名义值**（当前估计）——真值不可得；
3. **注入后忘清零/忘投影**：$\delta\hat{\mathbf{x}}$ 注入后必须重置，否则下次更新的
   $K$ 作用在已被吸收的误差上（表现为状态震荡）；
4. **杠杆臂忽略**：天线离 IMU 半米时 (4.10) 的 $R\mathbf{l}_a$ 项在转弯时产生分米级
   误差——厘米级预算下不可忽略；
5. **重力当常值 ($0,0,-9.8)$ 放死**：它进状态（(4.1)）正是因为初始化的重力对准
   有误差、需要 RTK 在线修正——第 3 讲埋的伏笔在此收口；
6. **问"为什么不直接 UKF/粒子"**：18 维状态 UKF 采样代价高、粒子维数灾难；
   误差状态 EKF 在近线性区足够准且最轻——第 5 讲的图优化是另一极，中间形态是
   第 10 讲的 IEKF。

## 4.7 自检题

1. 列出"直接 EKF on 名义状态"的三个缺陷，并各用 D4-1 中的一步说明
   误差状态如何消除它。
2. 完整重推 (4.4)：从 $R_t = R\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)$ 求导到消去公共项，
   每一步注明所用的一阶性质。
3. 推导 (4.5) 中 $-R(\tilde{\mathbf{a}}-\mathbf{b}_a)^\wedge$ 块（提示：第 2 讲 (2.8)）。
4. 把 (4.10) 换成"RTK 直接给航向角 $\psi$"的观测（双天线），写 $H$ 的姿态块
   （提示：航向是 $R$ 的非线性函数，需在名义姿态处线性化）。
5. 数值实验：`eskf.hpp` 中把 (4.13) 的投影打开/关闭，比较 RTK 频率降至 0.1 Hz
   时的收敛段轨迹差异。
6. 说明 ESKF 的"预测 200 Hz、更新 1 Hz"在计算量上的不对称性，
   并解释为什么这正好匹配车规嵌入式预算。

## 4.8 本章要点

- 双状态架构：名义状态 (4.2) 负责走，误差状态 (4.4)–(4.6) 负责不确定度；
  误差最小参数化消除 $SO(3)$ 约束与奇异；
- D4-1 的四块 $F$（(4.7)）是**一张纸写完的 ESKF**——能默写即掌握本讲；
- 更新三步曲 (4.11)–(4.13)：KF 更新误差 → 注入名义 → （可选）投影协方差；
- 初始化（静止检测 + 重力对准 + 首帧 RTK）不是工程杂务，而是 (4.5) 耦合结构
  的直接后果。

## 4.9 配套阅读

- 上一讲：[03_IMU与惯性导航.md](03_IMU与惯性导航.md)；下一讲：
  [05_预积分学与图优化GINS.md](05_预积分学与图优化GINS.md)（同一任务的优化路线解法）；
- 十四讲对照：[../十四讲/10_后端1滤波与BA.md](../十四讲/10_后端1滤波与BA.md)（KF 推导）、
  [../十四讲/04_李群与李代数.md](../十四讲/04_李群与李代数.md)（D2-1 母推导）；
- 原始文献：Solà, *Quaternion kinematics for the error-state Kalman filter*（误差状态
  运动学的完整参考）；Joan Solà 课程讲义《Simultaneous localization and mapping
  with the extended Kalman filter》；
- 代码：`src/ch3/eskf.hpp`、`static_imu_init`、`run_eskf_gins.cc`；
  numpy 对照：`projects/slam/vins/`（其中 ESKF 风格的 IMU 预积分骨架）。

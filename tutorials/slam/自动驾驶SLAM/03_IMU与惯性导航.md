# 第 3 讲｜IMU 与惯性导航：运动学、噪声、航迹推算

> 对应：书第 3 章 §3.1–3.3（p.49–71）｜大纲：[00_自动驾驶SLAM技术教程大纲.md](../00_自动驾驶SLAM技术教程大纲.md) §讲 3
> 本章推导产出：**D3-1**（纯惯性递推的发散链条）、**D3-2**（连续→离散噪声参数换算）。
> 公式 (3.1)–(3.12)。§3.3 的 RTK 安装与数据显示为操作步骤，不含理论，予豁免。

## 3.1 本章定位

第 2 讲把运动学 (2.6) 铸好了；本讲回答三个落地问题：① IMU 测的到底是什么量、
噪声长什么样（§3.2）；② 只用 IMU 递推能撑多久——答案"秒级"，且能算出精确
发散速率（D3-1，这是全书存在的理由）；③ 谁来提供不发散的绝对位置——
GNSS/RTK（§3.4）。ESKF 把两者融合，是第 4 讲的全部内容。

## 3.2 学习目标

1. 从比力定义出发默写 IMU 测量方程 (3.1)(3.2)，说清为什么加计**测不出**匀速运动；
2. **独立复现 D3-1**：由零偏→姿态漂移→重力误投→位置三次方发散的完整链条；
3. 复现 D3-2：把数据手册的连续谱密度换算成离散协方差（两个平方根因子方向相反）；
4. 说清 RTK 为什么能到厘米级、什么场景失效；完成经纬高 → ECEF → ENU 的手算。

## 3.3 问题场景（五步法①）

**手头**：一颗 6 轴 IMU（200 Hz，MEMS 或光纤），输出角速度与比力；数据手册给了
噪声谱密度与零偏稳定性。**想要**：任意两时刻之间的相对位姿递推（航迹推算）。
**缺**：测量方程里的噪声如何进入状态、递推多久可信——这决定后续每次
"观测间隔内 IMU 是否兜得住"的设计判断。

## 3.4 核心内容

### 3.4.1 IMU 测量方程与噪声模型（书 §3.1.1–3.1.2）

**陀螺（角速度）**：

$$\tilde{\boldsymbol\omega} = \boldsymbol\omega + \mathbf{b}_g + \mathbf{n}_g, \qquad
\dot{\mathbf{b}}_g = \mathbf{w}_g \tag{3.1}$$

**加计（比力，specific force）**：

$$\tilde{\mathbf{a}} = R_{wi}^\top(\dot{\mathbf{v}}_w - \mathbf{g}) + \mathbf{b}_a + \mathbf{n}_a, \qquad
\dot{\mathbf{b}}_a = \mathbf{w}_a \tag{3.2}$$

（依据：比力的物理定义——加计测量**支撑力产生的加速度**，即载体坐标加速度
去掉重力项再投影到体系。静止时 $\dot{\mathbf{v}} = \mathbf{0}$、$g$ 指向下，故加计读数
为 $R^\top(-\mathbf{g}) \approx [0,0,+9.8]$——"测上"是重力反号，这是静止初始化
重力对准的原理。匀速直线运动时读数为零，所以**惯性系统测不出匀速运动**，
位置必须靠积分。）

噪声模型（③选型理由：白噪声 + 随机游走是既能刻画器件物理、又可解的最低复杂度
模型；④依据：IEEE 传感器标准建模约定，Allan 方差可辨识两个参数）：

- $\mathbf{n}_g, \mathbf{n}_a$：**角度/速度白噪声**，每时刻独立，
  $\mathrm{Cov}(\mathbf{n}_g) = \sigma_g^2 I$；
- $\mathbf{b}_g, \mathbf{b}_a$：**零偏**，随机游走 (3.1)(3.2) 第二式，驱动噪声
  $\mathbf{w}$ 白。零偏不可直接观测、又随时间漂移——它必须进状态向量
  （第 4 讲把 $\mathbf{b}$ 列入 $\mathrm{NavState}$）。

### 3.4.2 D3-2：连续谱密度 → 离散协方差（书 §3.1.3）

**① 问题场景**：数据手册给的是**连续**谱密度（单位 rad/s/√Hz、mg/√Hz 级），
而第 4 讲的 ESKF 需要每个采样周期的**离散**噪声协方差 $Q_k$。
**② 解决方法**：按谱密度在采样带宽上积分。
**④ 依据**：白噪声 PSD 为常数 $S(f) = \sigma_c^2$，离散采样等效带宽 $B = \tfrac{1}{2\Delta t}$。

**⑤ 推导**。白噪声项：离散样本方差 = 带内积分

$$\mathrm{Var}(n_k) = \int_{-B}^{B}\sigma_c^2\,\mathrm{d}f = \frac{\sigma_c^2}{\Delta t}
\;\Rightarrow\; \boxed{\sigma_d = \frac{\sigma_c}{\sqrt{\Delta t}}} \tag{3.3}$$

零偏游走项：$\mathbf{b}_k = \mathbf{b}_{k-1} + \mathbf{w}_k$，一个采样内游走增量为
谱密度对**时长**的积分

$$\mathrm{Var}(w_k) = \sigma_{b,c}^2\,\Delta t
\;\Rightarrow\; \boxed{\sigma_{b,d} = \sigma_{b,c}\sqrt{\Delta t}} \tag{3.4}$$

（依据：随机游走是白噪声的时间积分，积分把 PSD 的量纲从 √Hz 分之积变成
√s 分之积。）两个平方根因子**方向相反**——采样越快，白噪声样本方差越大、
游走增量越小。记错方向（不除 $\sqrt{\Delta t}$）是滤波调参最常见的 bug：
200 Hz 数据配错因子会让 $Q$ 差 200 倍。

**现实中的 IMU（书 §3.1.4）**：MEMS（消费级，$\sigma_{b,g} \sim 0.01\!-\!0.1\,^\circ/\mathrm{s}$）到
光纤（车规/行业级，$10^{-3}\,^\circ/\mathrm{s}$ 级）差 2–3 个量级；两个参数用 **Allan 方差
曲线**读出：双对数曲线 $\tau^{-1/2}$ 段交于 1 s 的值 ≈ 角度随机游走，$\tau^{1/2}$ 段
交于 1 s 的值 ≈ 零偏不稳定性。书中数据集的 IMU 参数已在 `config/*.yaml`
里按 (3.3)(3.4) 换算好。

### 3.4.3 D3-1：纯惯性递推能撑多久（书 §3.2）

**① 问题场景**：观测（RTK）中断了，只剩 IMU——用 (2.6) 从当前状态往前递推，
多久之后位置误差会超出厘米级预算？
**② 解决方法**：追踪一个初始零偏误差 $\delta\mathbf{b}_g$（常数）在递推链上的放大路径。
**④ 依据**：(3.1)(3.2) + (2.6)，一阶误差分析。

**⑤ 推导**。第一步：零偏直接进入姿态积分（依据：(2.3) 右乘 + (3.1)），
$t$ 秒后姿态误差 $\delta\boldsymbol\theta(t) = \mathbf{b}_g\, t$；
第二步：姿态误差把重力投影错了（依据：(3.2)，重力项 $R^\top\mathbf{g}$ 对
$\delta\boldsymbol\theta$ 的敏感度由 (2.8) 型雅可比给出）：

$$\delta\mathbf{a} \approx \mathbf{g}\times\delta\boldsymbol\theta \;\Rightarrow\;
\|\delta\mathbf{a}\| \approx g\,\|\mathbf{b}_g\|\,t \tag{3.5}$$

第三步：加速度误差二次积分成位置误差：

$$\delta\mathbf{v}(t) \approx \tfrac12 g\|\mathbf{b}_g\|t^2, \qquad
\delta\mathbf{p}(t) \approx \tfrac16\, g\,\|\mathbf{b}_g\|\,t^3 \tag{3.6}$$

**数值落地**：$\|\mathbf{b}_g\| = 0.01\,^\circ/\mathrm{s} \approx 1.7\times10^{-4}\,\mathrm{rad/s}$
（较好的 MEMS），$t = 60\,\mathrm{s}$：$\delta p \approx \tfrac16(9.8)(1.7\times10^{-4})(60)^3
\approx 60\,\mathrm{m}$——一分钟漂 60 米。若只撑 $1\,\mathrm{s}$：
$\delta p \approx 0.3\,\mathrm{mm}$。

**结论（全书的立足点）**：IMU 递推在**秒级间隔内**近乎完美、在**分钟级**完全
不可用。所以融合系统的设计目标是：让任何观测间隔都落在秒级内（第 4 讲的
ESKF 以 200 Hz IMU 为骨架），并靠高频观测把零偏**估计出来**——零偏一旦
被估准，(3.6) 的源头被封住，短时中断的鲁棒性由第 10 讲的紧耦合 LIO 继续提升。

**航迹推算实现（`src/ch3/imu_integration.h`，`sad::IMUIntegration::Integrate`）**：
按 (2.6) 做欧拉/矩形离散递推——

$$R_{k+1} = R_k\exp\big((\tilde{\boldsymbol\omega}_k - \mathbf{b}_g)\Delta t\big),\quad
\mathbf{v}_{k+1} = \mathbf{v}_k + \big(R_k(\tilde{\mathbf{a}}_k - \mathbf{b}_a) + \mathbf{g}\big)\Delta t,\quad
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k\Delta t + \tfrac12(\cdots)\Delta t^2 \tag{3.7}$$

（200 Hz 下一阶格式已够；中值积分等高阶格式属优化项。）

### 3.4.4 卫星导航与 RTK（书 §3.3）

**GNSS 家族（§3.3.1）**：GPS（美）、北斗 BDS、GLONASS（俄）、Galileo（欧）；
多系统同收可提升可见星数与几何构型（DOP）。伪距单点定位精度米级——
远不够 L4。

**RTK（§3.3.2，实时动态载波相位差分）**：基准站（位置已知）与移动站同步观测
同一组卫星，载波相位差分解算**整周模糊度**后可达厘米级。失效场景：隧道/室内
（无星）、高架与城市峡谷（多路径 + NLOS）、电离层突发。网络 RTK（CORS）免去
自建基站，但仍需要网络——所以 RTK 的角色是"**高频可用但会消失的绝对观测**"，
消失时由 IMU（秒级）与激光地图匹配（第 8–12 讲）接力。

**世界坐标系（§3.3.3）**：GNSS 原始输出是 WGS-84 经纬高 $(\lambda, \varphi, h)$。
工程上两步转换：

经纬高 → ECEF（地心地固）：

$$\begin{bmatrix}x\\y\\z\end{bmatrix}_{e} =
\begin{bmatrix}(N+h)\cos\varphi\cos\lambda\\ (N+h)\cos\varphi\sin\lambda\\ \big(N(1-e^2)+h\big)\sin\varphi\end{bmatrix}, \qquad
N = \frac{a}{\sqrt{1-e^2\sin^2\varphi}} \tag{3.8}$$

（$a$ 为长半轴、$e$ 为第一偏心率，依据：WGS-84 椭球定义。）

ECEF → ENU（站心东北天，原点取某参考点）：先平移再转正交阵（由当地经纬度
的 $\sin/\cos$ 构成，标准结果，手推见自检题 3）：

$$\mathbf{p}_n = R_{ne}\,(\mathbf{p}_e - \mathbf{p}_{e,0}) \tag{3.9}$$

代码用 **UTM**（横轴墨卡托，6° 分带平面投影，`src/ch3/utm_convert.h`）替代 ENU
做全局坐标——平面坐标便于直接与 2D 地图/轨迹比较。RTK 航向：双天线主副站
连线方向给 $\mathrm{heading}$（`gnss.h` 中 `heading_valid` 标志其可用性）。

## 3.5 实践任务

- **操作步骤，不含理论**（豁免声明）：运行 `ch3/run_imu_integration`：
  IMU 纯递推 vs RTK 轨迹，肉眼确认 (3.6) 量级的发散；运行 `process_gnss`
  显示 RTK 轨迹（`utm_convert` 投影）。
- 理论回引：修改 `imu_integration` 的零偏参数为手册值 ×10，验证发散速率符合
  (3.6) 的三次方标度（自检题 1）。

## 3.6 易错点与高频疑问

1. **零偏当常值**：(3.1) 第二式说它是随机游走——递推 10 分钟用出厂零偏必然漂飞；
   零偏只能靠状态估计在线收敛（第 4 讲）；
2. **(3.3)(3.4) 因子方向记反**：白噪声除 $\sqrt{\Delta t}$、游走乘 $\sqrt{\Delta t}$；
   校验口诀——"采样翻 4 倍，$Q$ 里的白噪声方差大 4 倍、游走方差小 4 倍"；
3. **ENU vs NED**：航空惯导惯例是 NED（北东地），本书与数据集用 ENU（东北天）；
   混用表现为"轨迹镜像 + 高度反号"；
4. **加计读数当坐标加速度**：(3.2) 里有 $-\mathbf{g}$——静止时读数 $[0,0,g]$ 而非零；
5. **RTK heading 无条件使用**：单天线 RTK 无航向；双天线 heading 在静止/低速时
   也不稳（`heading_valid` 为假时必须降级，第 4 讲初始化里体现）。

## 3.7 自检题

1. 由 (3.6) 推导"$t$ 秒内位置误差 < $d$ 米"允许的最大陀螺零偏表达式，
   并算 $d = 0.1\,\mathrm{m}, t = 10\,\mathrm{s}$ 的值（这就是紧耦合 LIO 对陀螺的硬件要求量级）。
2. 手册给 $\sigma_{a,c} = 60\,\mu\mathrm{g}/\sqrt{\mathrm{Hz}}$、$\sigma_{b_a,c} = 40\,\mu\mathrm{g}/\sqrt{\mathrm{h}}$，
   采样 200 Hz：按 (3.3)(3.4) 算离散值（注意 $\mu\mathrm{g}$ 与 $\mathrm{m/s^2}$、
   $\sqrt{\mathrm{h}}$ 与 $\sqrt{\mathrm{s}}$ 的单位换算）。
3. 推导 (3.9) 中 $R_{ne}$ 的具体矩阵（用 $\sin\varphi, \cos\varphi, \sin\lambda, \cos\lambda$），
   并验证它是正交阵。
4. 静止 20 s 的加计数据均值 $[0.02, -0.01, 9.79]\,\mathrm{m/s^2}$、陀螺均值
   $[0.002, 0.003, -0.001]\,\mathrm{rad/s}$：给出初始 $R_{wi}$（重力对准）与
   $\mathbf{b}_g$ 估计——这正是 `static_imu_init` 的核心逻辑，写出伪代码。
5. 解释为什么零偏必须进状态向量而不能"测准一次存起来"（结合 (3.1) 第二式
   与温度/振动场景）。

## 3.8 本章要点

- IMU 测量方程 (3.1)(3.2)：比力含重力、零偏随时间游走——两个"坏消息"都写进模型；
- 噪声换算 (3.3)(3.4)：除/乘 $\sqrt{\Delta t}$，方向相反，$Q$ 配错的头号来源；
- D3-1：零偏 → 姿态 → 重力误投 → $\delta p \sim \tfrac16 g b t^3$——IMU 的价值与
  天花板都在这一条链上：秒级完美、分钟级报废；
- RTK = 会消失的厘米级绝对观测；ENU/UTM 坐标转换是所有数据集接口的公共底座。

## 3.9 配套阅读

- 上一讲：[02_基础数学回顾.md](02_基础数学回顾.md)（(2.6) 是本讲递推的骨架）；
  下一讲：[04_ESKF组合导航.md](04_ESKF组合导航.md)（把本讲递推与 RTK 观测融成 ESKF）；
- 十四讲对照：讲 3 §3.4（四元数积分）；主题教程：
  [../07_后端-i滤波与增量估计.md](../07_后端-i滤波与增量估计.md)（KF 基础，为第 4 讲铺路）；
- 扩展：Solà《Quaternion kinematics for the error-state Kalman filter》§5（误差状态
  运动学的完整版，第 4 讲 D4-1 的原始文献）；
- 代码：`src/ch3/`（`imu_integration`、`static_imu_init`、`utm_convert`）。

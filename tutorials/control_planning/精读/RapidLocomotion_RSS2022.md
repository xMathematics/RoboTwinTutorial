# 论文精读｜Rapid Locomotion via Reinforcement Learning（RSS 2022）

> **PDF**：[papers/control_planning/frontier/arXiv-2205.02824_RapidLocomotion.pdf](../../../papers/control_planning/frontier/arXiv-2205.02824_RapidLocomotion.pdf)（arXiv:2205.02824v1，共 12 页，式号以此为准） ｜ **教程**：[第 09 章](../09_前沿学习式规划与腿式控制.md) §09.2（特权教师–学生蒸馏） ｜ **代码**：官方 MIT 仓库 ImprobableAI/rapid-locomotion（基于 legged_gym / Isaac Gym；论文 §II 明言仿真代码改编自开源仓库 [35]，即 Rudin et al. 2020 的 legged_gym；项目页 agility.csail.mit.edu）

## 1. 论文信息与一句话贡献

- **题目**：Rapid Locomotion via Reinforcement Learning
- **作者**：Gabriel B. Margolis、Ge Yang、Kartik Paigwar、Tao Chen、Pulkit Agrawal（MIT Improbable AI Lab / CSAIL；作者分属计算机科学与人工智能实验室 CSAIL、LIDS 与 MIT-IBM Watson AI Lab / NSF AI Institute，见论文第 1 页脚注）
- **发表**：Robotics: Science and Systems（RSS）2022；arXiv:2205.02824v1（2022-05-05，本地 PDF 共 12 页）
- **作者名单核对（与任务规格的偏离）**：规格所记 "Margolis, Yang, Talwala, Agrawal" 与 PDF 不符——论文作者为上列**五人，无 Talwala**。本精读一律以 PDF 原文为准。
- **一句话贡献**：一个端到端神经网络控制器（输入本体感知 proprioception 与速度指令，输出关节位置指令），靠两个组件把 MIT Mini Cheetah 在自然地形上跑到 **3.9 m/s 持续速度、5.7 rad/s 偏航转速**并零微调直接上真机：(i) **速度指令的自适应课程**（adaptive curriculum：reward 门控的 Box/Grid 两种命令分布扩张规则，论文式 (6)–(11)）；(ii) **特权教师–学生蒸馏**（教师吃仿真真值域参数 $\mathbf{d}_t$ 用 PPO 训练；学生用 $h=15$ 步本体观测历史回归教师的隐编码 $\hat{\mathbf{z}}_t \approx \mathbf{z}_t$，即**在线系统辨识** online system identification，论文式 (1)–(5)）。

## 2. 问题与动机

**要解决什么**（§I）：快速跑过**自然地形**（草地、冰面、碎石）。地形参数突变（摩擦、软硬、坡度）对控制器性能的冲击随速度增大；执行器力矩限制、大接触力、飞相（flight phase）身体姿态等因素在高速下才开始主导动力学（§I 引文）。模型基路线（简化模型 + 人类设计）在实时计算约束下表达能力受限；RL 提供了"从高奖励轨迹直接学策略"的替代，且不依赖人类工程化约简模型。

**RL 的多任务失败问题**（§I）：速度条件化策略要对线性/角速度指令的**大范围组合**都可用；一次性在所有任务上均匀采样训练会失败——高速奔跑从零学起本质上困难（离心力等物理约束使得"高速 + 高速转弯"的组合大多不可实现），多数指令采到就拿不到足够奖励，学习信号稀薄。这直接催生了 §III-D 的课程设计。

**信息不对称**（§III-B）：仿真里有一切真值——训练时随机化的仿真参数 $\mathbf{d}_t$（机体质量、质心、电机强度、地面摩擦、地面恢复系数，范围见 Table I）；而真机上 $\mathbf{d}_t$ **无法用机载传感器直接测量**（论文原话）。纯域随机化 domain randomization, DR 得到的 $\pi_{\mathrm{DR}}(\mathbf{x}_t)$ 能跨过 sim-to-real 缺口，但行为保守（论文引 [38, 42]）：**没有机制适配当前参数**——"从同一起点，在冰上理应与在草地上跑法不同，$\pi_{\mathrm{DR}}$ 对此无能为力"（§III-B 原文意译）。这即教程 §09.2 ① 的"仿真有真值、真机没有"。

**部署约束**（§II）：控制器须以 **50 Hz** 跑在机载 NVIDIA Jetson TX2 NX 上；传感只有关节编码器 + IMU——无外部相机/LiDAR/动捕（"最小传感套件"，§I–II）。这排除了"把真值喂给策略"的一切部署方案，也决定了学生输入只能来自本体观测历史。

## 3. 方法总览

**平台与仿真**（§II）：MIT Mini Cheetah（9 kg，12 个准直驱 quasi-direct drive 电机，最大输出力矩 17 N·m）。仿真用 IsaacGym [26]（代码改编自 [35] legged_gym），**4000 个并行智能体**采 **400M 仿真步**（约合 92 个真机日的经验；单张 RTX 3090 三个小时内可算完）。注：正文写 4000 个并行智能体，附录 Table V 记"# environments per worker = 4096、# workers = 1"——论文两处口径略有出入，按原文分别如实记录。

**策略与执行器接口**（§III-A，Fig. 2）：策略 $\pi_\theta(\cdot)$ 输入 $\mathbf{x}_t$（观测 + 指令），输出关节位置指令 $\mathbf{a}_t \in \mathbb{R}^{12}$，经 PD 控制器（**位置增益 $k_p = 20$、微分增益 $k_d = 0.5$，"为促进平滑运动而选低增益，实验期间未调"**——原文）转为力矩。教程第 10 章 10.2 把这三个数（$k_p, k_d$, 50 Hz）称为 sim-to-real 的**隐藏超参**：仿真与真机锁定同值，执行器被当作接口契约。

**两阶段框架**（§III-B/C）：

1. **特权教师** $\pi_T(\mathbf{x}_t, \mathbf{d}_t) = \pi_{\theta_b}(\mathbf{x}_t, g_{\theta_d}(\mathbf{d}_t))$：编码器 $g_{\theta_d}$ 把真值域参数压成隐向量 $\mathbf{z}_t$（式 (1)），策略体 $\pi_{\theta_b}$ 由 $(\mathbf{x}_t, \mathbf{z}_t)$ 出动作（式 (2)），$\theta_b, \theta_d$ 用 **PPO**（[36]）联合最大化折扣回报（式 (3)）。教师在仿真内训练，可任意使用仿真器内部信息。
2. **学生** $\pi_S(\mathbf{x}_t, \mathbf{x}_{[t-h:t-1]}) = \pi_{\theta_b}(\mathbf{x}_t, h_{\theta_a}(\mathbf{x}_{[t-h:t-1]}))$：把编码器换成**在线辨识模块**（adaptation module，术语沿 [23, 24]）$h_{\theta_a}$，只吃 $h=15$ 步输入历史（每步 42 维），输出 $\hat{\mathbf{z}}_t$ 对齐 $\mathbf{z}_t$（式 (4)），用**on-policy 数据上的监督回归**训练（式 (5)）；策略体参数与教师**共享**（Table II 图注：reuses $\theta_b$ from the teacher）。部署时只跑学生。

**课程策略**（§III-D）：速度指令从概率分布 $p^k_{v_x,\omega_z}(\cdot,\cdot)$ 采样（第 $k$ 回合），侧向指令 $v_y^{\mathrm{cmd}}$ 单独从固定小均匀分布采样（纵向 + 偏航已足够实现全向移动，原文）。两种自动课程：**Box Adaptive**（独立边缘分布，式 (8a-b)）与 **Grid Adaptive**（联合分布，式 (9)），更新规则式 (10)/(11)，由跟踪奖励是否超过阈值 $\gamma$ 门控。

**训练量与网络规模**（Table II/V）：编码器 [256, 128] → 8；适应模块 [256, 32] → 8；策略体 [512, 256, 128] → 12（Fig. 2 称部署的控制器为"5 层网络"）；ELU 激活；PPO 超参见 Table V（§4 已核对）。

**数据流一图**（仿真训练，据 §III-B/C 整理）：

```text
仿真（IsaacGym，4000 智能体并行，d_t 每回合从 Table I 范围采样）
  真值 d_t ──g_θd──► z_t ─┐
  观测 x_t ───────────────┤► π_θb ──► a_t ──► PD(kp=20,kd=0.5) ──► 机器人
  历史 x_[t-h:t-1] ─h_θa─► ẑ_t（对齐 z_t：损失 (5)，on-policy 监督）
部署（真机，50 Hz Jetson TX2 NX）：只跑 π_S = π_θb(x_t, h_θa(x_[t-h:t-1]))，无 d_t、不再训练
```

训练顺序的诚实表述：RL 目标 (3) 与监督损失 (5) 由论文实现为**同时**优化（4.2(d)），"两阶段"指的是优化方式的二分——教师走 RL、学生走监督蒸馏——而非严格的时间先后（对照：Lee et al. [24] 是先训教师、再采集数据、后训学生的串行流程，本文明确列为其差异之一）。

## 4. 关键公式推导

### 4.0 符号表

| 符号 | 含义 | 维度 / 取值（论文出处） |
|------|------|------------------------|
| $\mathbf{d}_t$ | 域参数真值（特权信息）：地面摩擦、地面恢复系数 restitution、负载质量、机体质心、电机强度 | $\mathbb{R}^{12}$（Table II；范围 Table I：摩擦 0.05–4.00、恢复系数 0.00–1.00、负载质量 −1.0–3.0 kg、质心 ±0.10 m、电机强度 90–110%。论文未逐项给出各参数在 12 维中的分解） |
| $\mathbf{z}_t$ | 域参数的隐编码 | $\mathbb{R}^8$（Table II） |
| $\mathbf{o}_t$ | 本体观测 $[\mathbf{q}_t,\ \dot{\mathbf{q}}_t,\ \mathbf{g}^{\omega}_t,\ \mathbf{a}_{t-1}]$：关节角(12)、关节速度(12)、机体系重力方向(3, IMU)、上一步动作(12) | $\mathbb{R}^{39}$（§III-A） |
| $\mathbf{x}_t$ | $\mathbf{o}_t \oplus \mathbf{v}^{\mathrm{cmd}}_t$（指令条件化）；历史记 $\mathbf{x}_{[t-h:t-1]}$ | $\mathbb{R}^{42}$（§III-A、Table II 记 $42 \times 15$） |
| $\mathbf{v}^{\mathrm{cmd}}_t$ | 指令 $(v_x^{\mathrm{cmd}}, v_y^{\mathrm{cmd}}, \omega_z^{\mathrm{cmd}})$：纵/侧向线速度 + 偏航角速度 | $\mathbb{R}^3$（§III 开头） |
| $\mathbf{a}_t$ | 关节位置指令，经 PD（$k_p=20,\ k_d=0.5$）成力矩 | $\mathbb{R}^{12}$（§III-A） |
| $g_{\theta_d},\ h_{\theta_a},\ \pi_{\theta_b}$ | 编码器 / 适应模块 / 策略体（MLP，ELU） | Table II（8 / 8 / 12 输出） |
| $h$ | 历史窗口长度 | 15（§III-C 2)） |
| $r_t$ | 奖励（Table VI，训练目标改编自 [35]） | 标量 |
| $\gamma$ | 折扣因子 = 0.99（Table V）。**注意符号撞名**：§III-D 课程的成功阈值也记 $\gamma \in (0,1)$——本节按上下文区分 | — |
| $p^k_{v_x,\omega_z}$ | 第 $k$ 回合命令采样分布（离散网格 0.5 m/s × 0.5 rad/s，中心 $(0, 0)$） | §III-D |

### 4.1 特权教师与 RL 目标（论文式 (1)–(3)）

$$
\mathbf{z}_t = g_{\theta_d}(\mathbf{d}_t), \tag{1}
$$

$$
\mathbf{a}_t = \pi_{\theta_b}(\mathbf{x}_t, \mathbf{z}_t), \tag{2}
$$

$$
\max_{\theta_b, \theta_d}\ \mathbb{E}_{\pi_{\theta_b,\theta_d}}\Big[\sum_{t=0}^{\infty} \gamma^t r_t\Big]. \tag{3}
$$

**逐式说明**（(1)(2) 是结构定义，(3) 是优化目标，推导在于指出特权性改变的是什么）：

- 式 (1) 右端**只有 $\mathbf{d}_t$**——编码器把 12 维域参数压缩为 8 维隐向量；$\mathbf{z}_t$ 因此是域参数的（学习出的）表示，不含当前观测（依据：式 (1) 变量依赖关系，原文"compresses $\mathbf{d}_t$ into an intermediate latent vector"）。
- 式 (2) 把 $\mathbf{z}_t$ 与本体观测 $\mathbf{x}_t$ 拼进同一策略体——特权信息**只通过输入通道**进入决策，不改变优化问题本身（依据：式 (2) 结构）。
- 式 (3) 是标准折扣回报最大化目标；求解器为 PPO [36]（依据：原文"We optimize the teacher's parameters $\theta_d, \theta_b$ together using PPO"）。PPO 的截断代理目标与超参见教程第 08 章 §08.2（式 (8.12)）与本文 Table V 核对：折扣 0.99、GAE 参数 0.95、每回合 rollout 21 步、每 rollout 训 5 个 epoch、每 epoch 4 个 minibatch、熵bonus 0.01、价值损失系数 1.0、clip 0.2、奖励归一化开、学习率 1e-3、Adam、共 400M 步。
- 与 $\pi_{\mathrm{DR}}$ 的对照（论文 §III-B 的论证，逐步）：(i) DR 是"对所有随机化参数学单一行为"；(ii) $\pi_{\mathrm{DR}}$ 的优化问题即 (3) 去掉 $\mathbf{z}_t$ 输入通道的版本——同一目标、更少信息；(iii) 解的最优性随之退化：策略只能对参数分布取"保守平均"，因为没有一个可随参数变化的自由度（依据：信息集缩小 ⇒ 可实现策略类缩小 ⇒ 最优值不增；论文给出冰面/草地实例与保守性引文 [38, 42]）。

### 4.2 学生与隐空间蒸馏（论文式 (4)–(5)，核心）

$$
\hat{\mathbf{z}}_t = h_{\theta_a}(\mathbf{x}_{[t-h:t-1]}), \tag{4}
$$

$$
\mathcal{L}_{\theta_a} = \Big(h_{\theta_a}(\mathbf{x}_{[t-h:t-1]}) - g_{\theta_d}(\mathbf{d}_t)\Big)^2 = (\hat{\mathbf{z}}_t - \mathbf{z}_t)^2. \tag{5}
$$

**(a) 论文的概念目标是动作模仿。** §III-B：学生"trained to mimic the teacher's action via **behavior cloning** [34]"。若在动作分布空间做模仿，标准目标是最小化 forward KL（教程 (9.6)）；按 KL 定义展开（依据：$\mathrm{KL}(p\|q) = \mathbb{E}_{a\sim p}[-\log q(a)] - H(p)$），它等于教师采样下的 NLL 减去与 $\theta_a$ 无关的教师熵 $H(\pi_T)$（教程 (9.7)）；教师为确定性逐点输出时进一步退化为动作空间 MSE（依据：教程 §09.2 ⑤ 第二步的高斯展开）。**论文没有走这条路线**——式 (5) 把回归目标从动作 $\mathbf{a}_t$ 换成隐编码 $\mathbf{z}_t$，是"隐空间蒸馏"。

**(b) 为什么隐空间回归足以对齐动作（逐步论证）：**

1. 教师动作经由复合映射依赖 $\mathbf{d}_t$：把 (1) 代入 (2)（依据：逐步代入）得 $\mathbf{a}_t = \pi_{\theta_b}(\mathbf{x}_t,\ g_{\theta_d}(\mathbf{d}_t))$。
2. 学生复用同一策略体：$\mathbf{a}^{S}_t = \pi_{\theta_b}(\mathbf{x}_t, \hat{\mathbf{z}}_t)$（依据：Table II 图注 "reuses $\theta_b$ from the teacher" 与 §III-C 2)）。
3. 于是 $\hat{\mathbf{z}}_t = \mathbf{z}_t \Rightarrow \mathbf{a}^{S}_t = \mathbf{a}_t$——同一函数作用在同一输入上必给同一输出（依据：函数取值的确定性）。
4. 因此对齐隐编码是**对齐动作的充分条件**。必要性论文不作声明：需 $g_{\theta_d}$ 的压缩保留策略体实际用到的信息；论文给出的经验判据是原话——"当此损失低时，隐表示 $\mathbf{z}_t$ 在师生间共享，学生可**不经再训练**复用教师的策略体 $\mathbf{a}_t = \pi_{\theta_b}(\hat{\mathbf{z}}_t, \mathbf{x}_t)$ 选动作"。
5. (5) 的极小点：右端是 8 维欧氏范数平方 $\|\hat{\mathbf{z}}_t - \mathbf{z}_t\|_2^2 = \sum_{i=1}^{8}(\hat z_{t,i} - z_{t,i})^2$（依据：向量的平方范数按坐标展开），对 $\hat z_{t,i}$ 求偏导得 $2(\hat z_{t,i} - z_{t,i})$（依据：逐项求导），凸二次的驻点即极小点 $\hat{\mathbf{z}}_t = \mathbf{z}_t$。

**(c) 为什么这就是"在线系统辨识"（信息流论证，逐步）：**

1. 由 (1)，$\mathbf{z}_t$ 只依赖 $\mathbf{d}_t$——教师隐编码是域参数的压缩表示（依据：4.1 第一条）。
2. 由 (4) + (5) 的极小点，学生须实现 $\mathbf{x}_{[t-h:t-1]} \mapsto g_{\theta_d}(\mathbf{d}_t)$：从**纯历史**恢复域参数的编码信息（依据：第 1 条 + 损失极小条件）。
3. $\mathbf{d}_t$ 不在 $\mathbf{x}_t$ 里，只能通过动力学间接留痕：相同的上一步动作 $\mathbf{a}_{t-1}$ 在不同摩擦/质量/电机强度下产生不同的 $\mathbf{q}_t, \dot{\mathbf{q}}_t$ 与姿态响应——历史窗口里的"预期加速度没出来、身体姿态意外偏斜"等**动力学残差**正是参数的可观测印记（依据：动力学方程对参数的依赖；论文表述为"准确匹配教师动作迫使学生从 $h$ 步状态历史隐式推断 $\mathbf{d}_t$"）。
4. 可辨识性上界：对观测历史毫无影响的参数分量原则上不可恢复——可辨识性依赖参数对历史的实际作用（教程 §09.2 ⑤ 的同款限定，定性表述）。

**(d) 训练流程与 [23, 24]（RMA、Lee et al.）的两点差异**（论文原文列出）：(1) 历史更短（$h = 15$），"小到足以让适应模块与策略体**实时同步**运行"；(2) **适应模块与教师同时训练**，数据是 on-policy 的——教师滚动采样的每一步天然提供配对监督 $(\mathbf{x}_{[t-h:t-1]}, \mathbf{d}_t)$，无需单独的数据采集阶段。论文称高速奔跑能力对这两个设计选择不敏感（"not sensitive to these design choices"），选择它们只为训练与部署更简单。

### 4.3 速度指令的自适应课程（论文式 (6)–(11)）

**框架**（§III-D）：纵/偏航指令在第 $k$ 回合从 $p^k_{v_x,\omega_z}(\cdot,\cdot)$ 采样；两种策略都初始化为 $[-1.0, 1.0] \times [-1.0, 1.0]$ 上的均匀分布（依据：原文初始化规则），并控制扩张速度——定义常数成功阈值 $\gamma \in (0, 1)$（依据：§III-D，与 Table V 的折扣因子撞名）。

$$
p^{k+1}_{v_x,\omega_z}(\cdot,\cdot) \leftarrow p^{k}_{v_x,\omega_z}(\cdot,\cdot), \tag{6}
$$

$$
p^{k+1}_{v_x,\omega_z}(\cdot,\cdot) \leftarrow f\big(p^{k}_{v_x,\omega_z}(\cdot,\cdot),\ k\big), \tag{7}
$$

$$
p^{k+1}_{v_x}(\cdot) \leftarrow f_v\big(p^{k}_{v_x}(\cdot), r_{v_x}\big), \qquad p^{k+1}_{\omega_z}(\cdot) \leftarrow f_\omega\big(p^{k}_{\omega_z}(\cdot), r_{\omega_z}\big), \tag{8a-b}
$$

$$
p^{k+1}_{v_x,\omega_z}(\cdot,\cdot) \leftarrow f\big(p^{k}_{v_x,\omega_z}(\cdot,\cdot),\ r_{v_x},\ r_{\omega_z}\big). \tag{9}
$$

- (6) 是**无课程**的恒等更新（依据：直接代入，分布永不变化）；(7) 是**固定时间表**的手动课程（依赖回合数 $k$）——论文指其两个缺陷：需手工调参；环境或算法一变、学习速度随之改变，就得重调时间表。
- (8a-b) **Box Adaptive**：维持独立边缘 $p_{v_x,\omega_z}(\cdot,\cdot) = p_{v_x}(\cdot)\,p_{\omega_z}(\cdot)$（依据：原文乘积形式），两分量各自按跟踪奖励更新——密度在 $v_x$–$\omega_z$ 平面上呈矩形"盒"状（命名依据：原文）。
- (9) **Grid Adaptive**：直接维护**联合**分布并按两个奖励更新。动机（逐步）：若 $v_x, \omega_z$ 独立采样，则"双高"命令被采到的概率 = $P(v_x\text{ 高}) \cdot P(\omega_z\text{ 高})$，与"单高"命令同阶（依据：独立性下联合密度 = 边缘密度之积）；而高速奔跑叠加高速转弯因离心力远比单项困难（原文），大多数"双高"采样拿不到奖励 ⇒ 学习信号稀薄 ⇒ 独立课程可能学不到这些行为。Grid 以联合分布显式建模这种**交互难度**。

**扩张规则**（奖励门控；$r_{v_x^{\mathrm{cmd}}}, r_{\omega_z^{\mathrm{cmd}}}$ 即 Table VI 的两个速度跟踪项）：

$$
p^{k+1}_{v_x}(v_x^{n}) \leftarrow \begin{cases} p^{k}_{v_x}(v_x^{n}), & r_{v_x^{\mathrm{cmd}}} < \gamma, \\ 1, & \text{否则}, \end{cases} \tag{10a}
$$

（(10b) 对 $\omega_z$ 同构。）读法（依据：规则分段定义 + 原文解释）：把命令域离散成 0.5 步长的格；回合结束时，若该命令处的跟踪奖励达标（成功），则**把该格密度置 1**——等效于向该命令的 ±0.5 邻域（$v_x^n \in \{v_x^{\mathrm{cmd}} \pm 0.5\}$）注入概率密度，且只在"该命令尚未被加入过"时生效——分布**只增不减、单调扩张**；失败则密度保持不变。联合版本：

$$
p^{k+1}_{v_x,\omega_z}(v_x^{n}, \omega_z^{n}) \leftarrow \begin{cases} p^{k}_{v_x,\omega_z}(v_x^{n}, \omega_z^{n}), & r_{v_x^{\mathrm{cmd}}} < \gamma \ \text{或}\ r_{\omega_z^{\mathrm{cmd}}} < \gamma, \\ 1, & \text{否则}, \end{cases} \tag{11}
$$

其中邻域取 4-连通网格邻居。与 (10) 的关键差异是失败条件由"与"变"**或**"：只有跑与转**都**达标才扩张联合格——扩张被门控在"组合难度确实被攻克"之后（依据：规则 (11) 与 4.3 的离心力讨论）。一步物理论证（把论文的 force-balance 论证补全并显式标注）：匀速转弯半径 $R = v_x/\omega_z$，向心加速度 $a = v_x \omega_z$；轮胎-地面摩擦约束 $a \le \mu g$ 给出 $|\omega_z| \le \mu g / |v_x|$——约束激活时命令面积边界呈 $|\omega_z| \propto 1/|v_x|$，与 Fig. 3b 实测边界的"逆平方"形状一致（论文原文描述该现象并解读为"机器人到达了高速下的物理转弯极限"；数值推演为本精读补全）。

### 4.4 评价度量（论文式 (12)–(13) 与 Froude 数）

把 $v_x^{\mathrm{cmd}}$–$\omega_z^{\mathrm{cmd}}$ 平面离散为 0.5 m/s × 0.5 rad/s 的格（指标 $i, j$），跟踪误差定义为该格内的均方根偏差（RMS）：

$$
\epsilon_{ij}[v_x^{\mathrm{cmd}}] = \mathbb{E}_{v_x^{\mathrm{cmd}} \sim [i-1,i],\ \omega_z^{\mathrm{cmd}} \sim [j-1,j]}\Big[\sqrt{\mathbb{E}_t\big[(v_x^{\mathrm{cmd}} - v_x^t)^2\big]}\Big], \tag{12a}
$$

（(12b) 对 $\omega_z^{\mathrm{cmd}}$ 同构。）读法（依据：嵌套期望结构）：内层 $\mathbb{E}_t$ 对时间平均瞬时平方误差 → 开根号得单次试验的 RMS 跟踪误差 → 外层 $\mathbb{E}$ 对该格内的多次试验平均；实测每格跑 5 次试验（§III-E 原文）。**命令面积**（command area）汇总为

$$
\epsilon_{ij}[v_x^{\mathrm{cmd}}] + \epsilon_{ij}[\omega_z^{\mathrm{cmd}}] < \epsilon_0: \tag{13}
$$

满足该式的格组成的区域面积（单位 m/s·rad/s），$\epsilon_0$ 为误差容限；报告时用 5 个随机种子并给标准差误差棒（依据：§III-E）。跨硬件比较敏捷性用**菲劳德数** Froude number $Fr = v/\sqrt{gl}$（附录 C；$v$ 体速度、$g$ 重力、$l$ 标称腿长——依据：动态相似性假说，速度与腿长平方根成比例的动物运动动力学相似，论文引 Alexander [3]）。

**奖励函数核对**（Table VI，训练目标"改编自 [35]"即 Rudin et al. 的 legged_gym，论文注明只做小改动）：任务项 = xy 速度跟踪 $\exp\{-|\mathbf{v}_{xy} - \mathbf{v}^{\mathrm{cmd}}_{xy}|^2/\sigma_{v_{xy}}\}$（权重 0.02）+ 偏航跟踪 $\exp\{-(\omega_z - \omega_z^{\mathrm{cmd}})^2/\sigma_{\omega_z}\}$（0.01）；稳定性项 = $v_z^2$（−0.04）、roll-pitch 角速度 $|\omega_{xy}|^2$（−0.001）、基座高度 $(h - h_0)^2$（−0.6）、基座姿态 $|\mathbf{g}^{\mathrm{ori}}_{xy}|^2$（−0.002）、自碰撞指示（−0.02）、关节限位违反指示（−0.2）；平滑项 = 关节力矩 $|\boldsymbol{\tau}|^2$（−2e-7）、关节加速度 $|\ddot{\mathbf{q}}|^2$（−5e-9）、动作变化率 $|\mathbf{a}_{t-1} - \mathbf{a}_t|^2$（−2e-4）、足端腾空时间（+0.02）。正文补充动机：机器人高速时会"沉身、前倾向航向"，故专门加了基座高度与姿态惩罚（§III-A）。

## 5. 实验与结果解读

**仿真：课程消融**（§IV-A，Fig. 3）：命令平面 $[-6, 6]\times[-6, 6]$（m/s, rad/s）上的跟踪误差热图 + 命令面积-阈值曲线。三个课程对比：**无课程**完全失败——训练初期随机探索几乎不产生快速身体运动，奖励几乎总是很小、学习信号最小（原文）；**Box 课程**大幅改善（先学会小初始分布再渐进扩容）；**Grid 课程**命令面积在**所有**误差阈值下最大——维持联合分布建模了跑/转交互，而 Box 因边缘独立会把命令空间的极端组合排除在外。边界呈 $|\omega_z| \propto 1/|v_x|$ 的物理极限形（4.3 已解释）。

**敏捷性对比**（Table III）：本文 Mini Cheetah（RL）Froude 数 **5.1**（3.9 m/s、腿长 30 cm），与并行工作 Ji et al. [18] 一起是**首批 Froude ≥ 1 的强化学习腿式控制**；对照：Park et al. Cheetah 2（模型基，7.1 / 6.4 m/s / 59 cm）、Kim et al. Mini Cheetah MPC（4.6 / 3.7 m/s）、Unitree A1 MPC（2.8 / 3.3 m/s）、Kumar et al. A1 RL（0.8 / 1.8 m/s）、Hwangbo et al. ANYmal RL（0.5 / 1.5 m/s）。

**真机：速度与 sim-to-real 缺口**（§IV-B，Table IV）：室内动捕场地、指令斜坡至 6.0 m/s。带在线辨识的学生 $\pi_{\theta_{ST}}$ 仿真 5.46 m/s → 真机 **3.81 ± 0.09 m/s**（3 个种子；最高持续 3.9 m/s），纯 DR 策略 $\pi_{\theta_{DR}}$ 仿真 5.07 → 真机 **2.49 ± 0.07 m/s**（2 个种子）——在线辨识同时提高了绝对速度并**缩小了 sim-to-real 缺口**；3.8 m/s 的平均持续速度超过同机器人模型基 MPC 的此前纪录 3.7 m/s（[21]），与并行工作 [18] 一起显著快于此前 RL 腿式应用。**偏航**：室内转速 5.7 rad/s，为模型基控制器最快纪录 6.28 rad/s 的 90%（模型基纪录用了两个不同控制器分别创造直线 [21] 与偏航 [6] 纪录，本文单一策略完成全部行为）。**户外**（§IV-B，定性 + 少量实测）：10 米草地冲刺 **2.94 s**（平均 3.4 m/s）；冰面上高频打滑仍保持旋转稳定；**涌现行为**（作者声明定性观察）：碎石陡坡上行、单电机机械堵转下维持平衡、高速绊倒后空中翻转并以步态变换恢复四足落地。在最拿手的两个场景部署 [21] 的 MPC 基线，均未能恢复（碎石坡下滑、越障绊倒）——作者据此主张的不是"比 MPC 更鲁棒"，而是"RL 范式提供了获得鲁棒行为的可扩展替代"（原文）。

**消融**（§IV-C）：(i) 特权信息价值（Fig. 4）：命令面积在所有阈值下**教师 > 学生 ≈ 教师 > 纯 DR**；特权教师严格大于仅看机器人状态的 $\pi_{\theta_{DR}}$，而学生几乎追平教师——蒸馏把特权优势转移到了可部署策略上。(ii) 地形粗糙度训练（Fig. 5）：训练时地形噪声取 0 / 0.05 / 0.10 / 0.15 cm，命令面积随粗糙度**单调下降**（在 $\epsilon = 0.3$ 切片处清晰可见）——**盲策略在平地极速与崎岖鲁棒性之间存在权衡**；尽管只在平地上训练，策略已能在多种户外地形部署。

**关键数字汇总**（全部出自上表核对，供回引）：

| 数字 | 数值 | 出处 |
|------|------|------|
| 持续速度（真机室内，最高） | 3.9 m/s | §IV-B / Fig. 1 / Table III |
| 真机平均速度（3 种子均值） | 3.81 ± 0.09 m/s（仿真 5.46） | Table IV |
| 纯 DR 对照 | 2.49 ± 0.07 m/s（仿真 5.07） | Table IV |
| 户外 10 米冲刺 | 2.94 s（平均 3.4 m/s） | §IV-B Outdoor Running |
| 偏航转速 | 5.7 rad/s（模型基纪录 6.28 的 90%） | §IV-B Yaw Control |
| Froude 数 | 5.1（腿长 30 cm） | Table III / §VII |
| 训练规模 | 400M 步，约 92 真机日经验，单卡 RTX 3090 < 3 h | §II |
| 历史窗口 | h = 15 步 × 42 维输入 | §III-C / Table II |
| PD 增益 / 控制频率 | k_p = 20，k_d = 0.5 / 50 Hz | §II / §III-A |

## 6. 局限与后续影响

**论文自述局限**（§VII）：(1) 行为谱系窄——只训练了地面平面内的体速度控制；跳跃、下蹲、编排舞蹈、locomanipulation 均在范围外，需要截然不同的任务规范。(2) **无视觉、不能前瞻规划**——不能高效上楼梯或绕开坑洞（任务规范不含感知）。(3) 高速步态不应解读为"普遍更好"——体速度是欠规范目标，同等速度可能存在多组同等可取的步态；与辅助目标/人类偏好结合是未来方向。(4) **仪器限制**：户外无动捕，且在真机上记录大量高速摔倒/翻转不安全——户外行为只能定性评估，定量分析限于实验室。(5) 跨平台解释力存疑：Froude 数 5.1 为 Mini Cheetah 最高但低于 Cheetah 2 的 7.1，论文对"敏捷差异来自分析控制器还是硬件"只作推测。

**结构性局限（本精读归纳）**：(a) 依赖大规模并行仿真与域随机化管线（400M 步、IsaacGym 4000 智能体）——训练基建本身是门槛；(b) 地形训练只有平地 + 粗糙度噪声，无结构化地形课程（Fig. 5 的权衡即其代价）；(c) 学生泛化受限于 $h=15$ 历史窗口——4.2(c) 的可辨识性上界：参数对历史不可见即无法蒸馏，Table IV 显示缺口仍在（仿真 5.46 → 真机 3.81）。

**后续影响（定性）**：本文与并行工作把"自适应课程 + 特权蒸馏 + 低成本本体感知"确立为高速腿式 RL 的标准配方；其训练框架直接建立在 legged_gym [35] / Isaac Gym [26] 生态上，同组此前工作 [27] 与 RMA [23]、Lee et al. [24] 构成特权学习的谱系（[24] 的教师吃的是地形高度图真值，本文的 $\mathbf{d}_t$ 是纯动力学参数——特权信息从"几何"换到"物理"，教师不再需要任何地形表示）。教程第 09 章 §09.2 即以本文为该路线的代表。

## 7. 与本项目对照

**教程第 09 章 §09.2 的映射（已逐条核对）**：

| 教程（第 09 章） | 论文原文 | 一致性 |
|------------------|----------|--------|
| (9.5) $z_t = g_{\theta_s}(d_t),\ a_t = \pi_{\theta_o}(z_t, x_t)$ | 式 (1)(2) | 实质一致；**记号差异**：论文编码器下标 $\theta_d$、策略体 $\theta_b$，教程合并记 $\theta_s/\theta_o$ |
| 优化目标 = 第 08 章 PPO 的期望折扣回报，论文 Eq.(3) | 式 (3) | 一致（PPO = 教程第 08 章 §08.2，截断代理目标 (8.12)；超参 Table V 已核对） |
| (9.6)–(9.7)：forward KL ⇔ 教师采样下 NLL，高斯+固定协方差退化 MSE | 论文未用 KL/NLL——直接行为克隆式隐空间回归 | **教程是一般框架、论文是特例**：论文的 (5) 对应教程第三步"隐空间蒸馏"，教程已正确标注"论文 Eq.(5)" |
| (9.8) $\mathcal{L}_{\theta_s} = (h_{\theta_s}(o_{t-h:t-1}) - g_{\theta_s}(d_t))^2$ | 式 (5) | **一致**（隐空间 MSE）；记号差异：论文 $\mathcal{L}_{\theta_a}$、历史记 $\mathbf{x}_{[t-h:t-1]}$（含指令 $\oplus v^{\mathrm{cmd}}$），教程记 $o_{t-h:t-1}$（不含指令） |
| 学生 $h=15$ 步本体观测历史、策略体参数共享 | §III-C 2)、Table II 图注 | 一致（逐字核对） |
| ③(a) 纯 DR 保守、无适配机制 | §III-B 引 [38, 42] + 冰面/草地例 | 一致；Table IV 的 2.49 vs 3.81 m/s 是其定量证据 |
| ⑤ 第五步 DR 与蒸馏**嵌套** | 教师训练本身在 $d_t$ 随机化下进行（§III-B） | 一致 |

**一处教程未展开的细节**：教程把流程概括为"两阶段"（先 RL 后蒸馏）；论文原文是适应模块**与教师同时**训练、用 on-policy 数据（§III-C 2) 差异 (2)），并非先训完教师再离线训学生的串行流程——不影响 (9.8) 的数学形式，但影响数据管线实现，部署学生时不再更新任何参数。

**与第 10 章的衔接**：[第 10 章](../10_控制栈实战与robotwin衔接.md) 10.2 已以本文为例说明"隐藏超参"——$k_p = 20$、$k_d = 0.5$、50 Hz 在仿真与真机锁定同值（论文 "did not tune" 原文核对一致）；10.3 的"sim-to-real 三种补法"把本文的蒸馏定位为 DR 的升级（DR 制造参数多样性、蒸馏把多样性变成可辨识信号），与本精读 Table IV 的证据链闭合。

**跨主题（RoboTwin）**：sim-to-real 的两条实现路线——[RoboTwin 教程](../../robotwin/README.md)第 03/04 章与 [RoboTwin 2.0 精读](../../robotwin/精读/RoboTwin2.0_arXiv2025.md)用**五维域随机化自动生成** 10 万+ 轨迹摊平扰动（数据侧）；本文用 DR + 蒸馏把参数不确定性变成**可在线辨识的隐变量**（控制侧）。操作臂的关节位置增量 + 隐式 PD（第 10 章 (10.1)）下参数几乎不可辨识，故 RoboTwin 走纯 DR；腿式力矩级接口暴露参数效应，蒸馏才有增益——教程 §09.2 ③ 的"接口越透明、蒸馏收益越大"论证即由此而来。

**与安全层的互补**：本文的摔倒恢复是**经验**鲁棒性（涌现、无形式保证）；教程第 09 章 09.3"经典为骨、学习为肉、CBF 为保险"主张在其外再套经典安全滤波层——见 [CBF 综述精读](./CBF-Survey_ECCV2019.md)（前向不变性给硬保证，不依赖策略好坏）。本文亦无运动规划层；与 [MPNet 精读](./MPNet_ICRA2019.md)同属第 09 章混合范式：两者都把学习组件放在"保证最便宜"的层（采样分布 / 特权输入），把契约留给经典结构。

## 配套阅读

- 本地 PDF：[Rapid Locomotion（Margolis et al., RSS 2022）](../../../papers/control_planning/frontier/arXiv-2205.02824_RapidLocomotion.pdf)——重点读 §III（式 (1)–(13)、Table I/II）、§IV（Fig. 3–5、Table III/IV）、附录（Table V PPO 超参、Table VI 奖励项、Froude 数）；教师训练算法见 [PPO（Schulman et al., 2017）](../../../papers/control_planning/frontier/arXiv-1707.06347_PPO.pdf)。
- 谱系论文（未收录本地 PDF，文字索引）：RMA（Kumar et al., Sci. Robot. 2021，论文 [23]）与 Lee et al.（Sci. Robot. 2020，论文 [24]）——特权学习与在线辨识的源头；Rudin et al.（CoRL 2020，论文 [35]）——legged_gym 与奖励结构出处；Ji et al.（RA-L 2022，论文 [18]）——并行工作，显式状态估计 vs 本文隐式辨识的对照。
- 教程章节：[第 08 章｜RL](../08_学习式控制强化学习.md)（教师的 PPO 训练）｜ [第 09 章](../09_前沿学习式规划与腿式控制.md)（§09.2 教程推导、§09.3 混合范式）｜ [第 10 章](../10_控制栈实战与robotwin衔接.md)（10.2 隐式 PD 与隐藏超参、10.3 sim-to-real 三种补法）｜ [第 07 章｜CBF](../07_安全控制控制屏障函数.md)（部署安全层）。
- 精读互引：[MPNet（ICRA 2019）](./MPNet_ICRA2019.md)（同章 §09.1，混合范式另一实例）｜ [CBF 综述](./CBF-Survey_ECCV2019.md)（安全层）｜ [RoboTwin 2.0](../../robotwin/精读/RoboTwin2.0_arXiv2025.md)（域随机化的操作任务路线）｜ [精读总索引](../../robotwin/精读/README.md)。

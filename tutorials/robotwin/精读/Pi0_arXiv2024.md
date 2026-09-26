# 论文精读｜π0（arXiv 2024）

> **PDF**：[../../../papers/robotwin/frontier/arXiv-2410.24164_Pi0.pdf](../../../papers/robotwin/frontier/arXiv-2410.24164_Pi0.pdf) ｜ **教程**：[第 08 章 §8.2 策略表](../08_策略训练与部署.md) ｜ **代码**：本 PDF 未附开源声明（权重未随论文发布）；Physical Intelligence 后续以 openpi 库开源 π0 / π0-FAST 实现（论文外信息，供延伸）
>
> 勘误提示（规格与论文不符处，均按 PDF 原文写）：
>
> - 论文正文与附录的公式**均无编号**（`pdftotext -layout` 逐页核实，无 (1)(2) 标记），第 4 节标号 P1–P6 为本精读自编，公式本体按 PDF 逐字照录。
> - 概率路径原文为 $q(A^\tau_t \mid A_t) = \mathcal{N}(\tau A_t,\ (1-\tau)I)$、采样式 $A^\tau_t = \tau A_t + (1-\tau)\epsilon$——**不是**常见 Lipman 记号 $x_t=(1-t)x_0+tx_1+t\sigma\epsilon$（论文无独立 $\sigma$ 项），且两条陈述的方差口径差一个平方，见 §4.2 读者注。
> - 预训练**自采**数据的口径是 **7 种机器人构型 / 68 任务**（非"7 个数据集"），另混入 OXE（22 台机器人）数据；总时长 10,000+ 小时。
> - 本论文**未评测 π0-FAST**（其发布晚于本文）；§5 的对比基线是 OpenVLA / Octo / π0-small / Diffusion Policy / ACT。

## 1. 论文信息与一句话贡献

- **题目**：$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control
- **作者/机构**：Kevin Black、Noah Brown、Danny Driess、Adnan Esmail、Michael Equi、Chelsea Finn、Niccolo Fusai、Lachy Groom、Karol Hausman、Brian Ichter、Szymon Jakubczak、Tim Jones、Liyiming Ke、Sergey Levine、Adrian Li-Bell、Mohith Mothukuri、Suraj Nair、Karl Pertsch、Lucy Xiaoyang Shi、James Tanner、Quan Vuong、Anna Walling、Haohuan Wang、Ury Zhilinsky（Physical Intelligence，San Francisco）
- **发表**：arXiv:2410.24164（2024-10 首发；本地 PDF 为 2026-01-08 的 v4，全文以该 PDF 为准）
- **一句话贡献**：提出 **VLM 主干 + 动作专家（action expert）** 双权重架构的流匹配 VLA——PaliGemma（SigLIP 400M + Gemma 2B ≈ 3B）处理图像与语言，**300M 独立权重**的动作专家经**分块因果注意力**接收机器人状态与含噪动作块，用**条件流匹配**（conditional flow matching）生成 $H=50$ 步连续动作块、以最高 **50 Hz** 开环整块执行；配合"互联网级 VLM 预训练 + OXE 混合 + 10,000+ 小时自采灵巧数据"三源配方与"预训练—后训练"两阶段 recipe，在折衣、清桌、装箱等 5–20 分钟长程灵巧任务上整体超出 OpenVLA / Octo / DP / ACT（§VI）。

## 2. 问题与动机

- **机器人基础模型的三大瓶颈**（§I 原文归纳）：
  1. **规模**：预训练收益只在超大规模下显现；
  2. **架构**：既要复用互联网数据学到的语义，又要表达高频、多模态、连续的机器人动作；
  3. **训练 recipe**：数据策展与两阶段划分，"与架构同等重要"（论文原话，指 NLP/视觉的成功更多来自 recipe）。
- **离散 token VLA 的短板**（§I–II）：
  - 自回归离散化（RT 系）难以承载**高频动作块**与灵巧高频控制（本文目标 "up to 50 Hz"）；
  - 扩散/流匹配恰擅长复杂连续动作分布，但此前未被装进**用预训练 VLM 做主干**的 VLA 框架——这是本文声称的 novelty（"the first flow matching VLA that produces high-frequency action chunks for dexterous control"，§II）。
- **预训练/后训练分离的行为学论证**（§I，本精读认为是全文最清晰的动机段）：
  - 只在高质量数据上训练 → 模型**不会从错误中恢复**（数据里没见过错误样例）；
  - 只在低质量数据上训练 → 高质量行为不熟练、不流畅；
  - 两者结合 = "广度 + 流畅性 + 从错误中恢复的能力"。
- **与模块化/微调派的一句话差异**（§II 相关工作）：不在预训练 VLM 上"缝"新组件（IceAI/VoxPoser 类），而是**直接微调 VLM 本身生成连续动作**；与 RT-2 类工作相比，贡献还包括 recipe 与大规模真机评测体系。

## 3. 方法总览

### 3.1 数据三源（§III、Fig. 3–4）

- **自采灵巧操作数据**：**7 种机器人构型、68 任务、903M 时间步**（106M 单臂 + 797M 双臂）；任务以"名词+动词组合"构成（如 bussing = 放置大量不同杯盘餐具入桶），故行为范围远超 68 这个名义数（§V 原文）；
- **OXE**（Magic Soup 子集，占预训练混合 **9.1%**，来自 22 台机器人；另在结论中提及 DROID、Bridge）；
- **互联网级 VLM 预训练**（继承自 PaliGemma 的 SigLIP + Gemma 2 初始化）。
- 语言侧：给**任务名 + 约 2 秒片段级标注（segment annotations）**，让指令与子行为对齐（§III）。

### 3.2 机器人平台与动作空间（§V-C，Fig. 5）

- 7 类平台：UR5e（7 维：6 臂 + 夹爪，2 相机）、双臂 UR5e（14 维，3 相机）、Franka（8 维，2 相机）、双臂 Trossen（ALOHA 血统 ViperX 双臂，14 维，3 相机）、双臂 ARX/AgileX（14 维，3 相机）、Mobile Trossen/ARX（14+2 底盘 = 16 维）、Mobile Fibocom（14+3 全向底盘 = 17 维）；
- **维度对齐**：构型/动作向量按数据集最大维 **18** 统一、不足零填充；相机不足 3 路的屏蔽缺失槽位——跨本体训练由此成为定长序列问题。

### 3.3 架构（§IV、附录 B，Fig. 3）

- **单一 Transformer、两套权重（experts）**：图像/语言 token 走 PaliGemma 主干（width 2048, depth 18, mlp 16,384, 18 heads, head dim 256，Gemma 2B 配置）；机器人专属 token（$q_t$、含噪动作块）走**动作专家**；
- **动作专家刻意做小**：width 1024、mlp 4,096，约 **300M** 参数——因推理时动作专家要前向多次（每积分步一次），小则快；两套权重**只在自注意力层交互**，各自线性层不必同宽（mixture-of-experts 式，灵感引 Transfusion）；
- 观测进入方式：图像过 ViT 编码器、$q_t$ 经线性投影，均投到语言 token 嵌入空间（§IV）。

### 3.4 训练 recipe（§V）

- **预训练**（主模型 700k 步）：广度优先——跨构型、跨任务混训，产物是"听得懂指令、多任务初步胜任"的 base 模型；
- **后训练**：高质量任务数据微调——最简单任务 **5 小时**数据、最复杂任务 **100+ 小时**数据（§V-A 原文）；
- **高层分解**：清桌等语义复合任务外接 LMM/VLM 高层策略（π0-HL，SayCan 血统），把 "buss the table" 分解为拾取/倾倒/放置子指令（§V-B）。

### 3.5 控制与执行（§IV、附录 D）

- 一次生成整块 $H=50$ 步；**实测时间聚合/集成（temporal aggregation/ensembling）反而有害**，故开环整块执行；
- 执行协议：20 Hz 的 UR5e/Franka 每 **0.8 s** 推理一次、执行 **16 步**；其余 50 Hz 平台每 **0.5 s** 推理一次、执行 **25 步**。

## 4. 关键公式推导（核心章节）

### 符号表

| 符号 | 含义（依据：§IV、附录 B 原文） |
|---|---|
| $A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}]$ | 动作块：$H=50$ 步**未来**动作（附录 B 明确 $H$ 个 token） |
| $o_t = [I^1_t,\dots,I^n_t,\ \ell_t,\ q_t]$ | 观测：1–3 路 RGB、语言指令 token 序列、关节角向量 |
| $\tau \in [0,1]$ | 流匹配时间（上标 $\tau$）；$t$ 为机器人时间步（下标 $t$） |
| $\epsilon \sim \mathcal{N}(0,I)$ | 训练时采样的高斯噪声 |
| $v_\theta(A^\tau_t, o_t)$ | 动作专家输出的速度场（模型） |
| $u(A^\tau_t \mid A_t)$ | 条件流的目标速度场（训练目标） |
| $w,\ d,\ r$ | 动作专家嵌入宽 / 动作维 / 其他维（见 P5） |

### 4.1 条件流匹配损失（论文原式照录，无编号）

$$
L^\tau(\theta) = \mathbb{E}_{p(A_t\mid o_t),\ q(A^\tau_t \mid A_t)} \big\lVert\, v_\theta(A^\tau_t, o_t) - u(A^\tau_t \mid A_t) \,\big\rVert^2
\tag{P1}
$$

- **依据**：§IV 原式逐字照录（论文引 Lipman et al. [28] 与 Rectified Flow [32]）；期望对数据分布 $p(A_t \mid o_t)$ 与路径分布 $q(A^\tau_t \mid A_t)$ 双重采样；
- **训练一步的完整流程**（§IV 原文）：采噪声 $\epsilon \sim \mathcal{N}(0,I)$ → 构造含噪动作 $A^\tau_t$（见 P2）→ 网络回归目标场 $u$（见 P3）；
- **时间步采样**：$\tau$ 非均匀采，来自强调低（更噪）时间步的 Beta 分布——见 4.5 节 P6。

### 4.2 线性高斯路径：目标场 $u = A_t - \epsilon$ 的逐行推导

**论文原文三条陈述**（§IV）：路径 $q(A^\tau_t \mid A_t) = \mathcal{N}\big(\tau A_t,\ (1-\tau)I\big)$；采样式与目标场如下。

1. **路径定义**（依据：§IV 原文采样式）：
$$
A^\tau_t = \tau A_t + (1-\tau)\,\epsilon
\tag{P2}
$$
2. **对 $\tau$ 求导**（依据：对 $\tau$ 求偏导时 $A_t$ 与 $\epsilon$ 均视为常量；第一项 $\frac{\partial}{\partial \tau}[\tau A_t] = A_t$，第二项 $\frac{\partial}{\partial \tau}[(1-\tau)\epsilon] = -\epsilon$）：
$$
\frac{\partial A^\tau_t}{\partial \tau} = A_t - \epsilon
\tag{P3}
$$
3. **条件速度场 = 条件路径对时间的导数**（依据：流匹配定义，Lipman et al. 2023；论文引 [28, 32]）。故 $u(A^\tau_t \mid A_t) = A_t - \epsilon$，与论文原文给出的目标场一致。线性路径下该场**沿路径为常向量**——一条从噪声指向数据的"直线速度"。
4. **端点自查**（依据：把 $\tau = 0, 1$ 代入 P2）：$\tau = 0$ 得 $A^0_t = \epsilon \sim \mathcal{N}(0,I)$（纯噪声）；$\tau = 1$ 得 $A^1_t = A_t$（数据）。与论文"integrating the learned vector field from $\tau = 0$ to $\tau = 1$、starting with random noise $A^0_t \sim \mathcal{N}(0, I)$"的积分方向完全一致。
5. **读者注（方差口径，论文记号层面的不自洽，照录不擅改）**：若严格按 P2 采样，则 $\mathrm{Var}(A^\tau_t) = (1-\tau)^2 I$；而论文的路径陈述写协方差 $(1-\tau)I$——两者相差一个平方。**这不影响 P1 的训练与推理**：P3 的目标场只依赖路径均值的 $\tau$-导数，与协方差标定无关；推理起始点固定取 $\mathcal{N}(0,I)$，也不经过该协方差项。
6. **与"高斯路径"标准式的对应**（供读过 Lipman 原文的读者）：令 $x_0 = \epsilon$（噪声）、$x_1 = A_t$（数据），则 P2 即 $x_\tau = \tau x_1 + (1-\tau) x_0$——论文采用"数据在 $\tau=1$"的方向约定，且**无**独立 $\sigma$ 项。

### 4.3 推理：前向欧拉 ODE 积分 10 步 + KV 缓存

生成动作 = 从 $\tau = 0$ 的噪声出发，积分学得的速度场到 $\tau = 1$（§IV、附录 D）：

$$
A^{\tau+\delta}_t = A^\tau_t + \delta\, v_\theta(A^\tau_t, o_t),\qquad \delta = 0.1
\tag{P4}
$$

- **依据（P4 从何而来）**：§IV 原文 "forward Euler integration rule"，即 Taylor 一阶展开 $A(\tau + \delta) \approx A(\tau) + \delta \frac{dA}{d\tau}$，以 $v_\theta$ 近似 $\frac{dA}{d\tau}$ 后取步长 $\delta$；
- **步数**：论文原文 "10 integration steps (corresponding to $\delta = 0.1$)"，即 $1/\delta = 10$ 步；
- **实现与耗时**（附录 B、D，Table I，RTX 4090）：观测 token 只前向一次，其 keys/values **缓存**供 10 步复用，每步只重算动作专家的 suffix（动作 token）——动作专家做小到 300M 的动机正在于此。单块耗时：图像编码 14 ms + 观测前向 32 ms + 10 次动作专家前向 27 ms = **板载 73 ms**（离板再加 13 ms 网络延迟 = 86 ms）。

### 4.4 双分支注意力：分块因果掩码与动作 token 嵌入

**注意力掩码**（附录 B 原文，"blockwise causal"）：3 个块 $[I^1_t,\dots,I^n_t,\ \ell_t]\ \big|\ [q_t]\ \big|\ [A^\tau_t,\dots,A^\tau_{t+H-1}]$。

- 块内**全双向注意力**；各块**不可看见未来块**（块间因果）；
- 设计依据逐条（附录 B 原文）：① 首块含 PaliGemma 预训练模态，禁止其看见新模态以**减小对预训练分布的偏移**；② $q_t$ 单独成块，因为它在 10 步积分间不变——其 keys/values 可**跨积分步缓存**；③ 动作块位于序列末位，**可见全部输入**；
- **双分支结构小结**：图像/语言路由到 PaliGemma 主干权重，$[q_t, A^\tau_t]$ 路由到动作专家权重；两分支唯一的交互点是自注意力层的 K/V/Q 混合——这就是任务书所说"action expert 独立权重的流匹配分支、attention 与 VLM 拼接"在论文中的精确形态。

**动作 token 嵌入**（附录 B 原式照录）：对每个含噪动作 $a^\tau_{t'}$，

$$
e = W_3 \cdot \mathrm{swish}\!\big(W_2 \cdot \mathrm{concat}(W_1 a^\tau_{t'},\ \phi(\tau))\big)
\tag{P5}
$$

- $\phi:\mathbb{R} \to \mathbb{R}^w$ 为正弦位置编码（流匹配时间步 $\tau$ 的注入点）；$W_1 \in \mathbb{R}^{w \times d}$、$W_2 \in \mathbb{R}^{w \times 2w}$、$W_3 \in \mathbb{R}^{w \times w}$（维度自查：concat 输出 $2w$，恰配 $W_2$ 的列宽）；
- $H$ 个嵌入进动作专家，**只取对应的 $H$ 个输出位**，经线性投影解码为 $v_\theta(A^\tau_t, o_t)$（附录 B）。

### 4.5 训练技巧：时间步的 β 采样

$$
p(\tau) = \mathrm{Beta}\big(\tfrac{s-\tau}{s};\ 1.5,\ 1\big),\qquad s = 0.999
\tag{P6}
$$

- **依据**（附录 B 原文的完整论证链）：原版流匹配均匀采 $\tau \sim U(0,1)$；SD3 主张 lognormal 强调中间步，理由是高时间步（低噪）网络只需学恒等映射、低时间步（高噪）只需学数据均值；
- **本文的不同判断**：动作预测与图像合成不同——给定文本预测均值图像容易，而给定观测预测"均值动作"（学 $\mathbb{E}[A_t \mid o_t]$）是难问题，因为观测 $o_t$ 对动作分布的约束远强于文本对图像分布的约束；所以强调**低时间步（高噪端）**；
- **P6 的读法**：令 $u' = (s-\tau)/s \in [0, 1]$，则 $u' \sim \mathrm{Beta}(1.5, 1)$，密度 $\propto u'^{0.5}$ 随 $u'$ 递增——质量偏向 $u' \to 1$，即 $\tau \to 0$（噪声端）；$\tau > s$ 永不采样，且只要积分步 $\delta > 1 - s = 0.001$ 就不影响推理（$s = 0.999$ 允许至多 1,000 步积分；本文用 10 步，$\delta = 0.1 \gg 0.001$）。

### 4.6 与扩散 Policy 的对比：确定性 ODE vs 随机去噪链

对照 [Diffusion Policy 精读 §4.1–4.2](./DiffusionPolicy_RSS2023.md)（DDPM 式 (3) 及其随机 Langevin 解读）：

| 维度 | 扩散 Policy（DDPM/DDIM） | π0（流匹配） |
|---|---|---|
| 生成过程 | 随机迭代去噪：DDPM 每步注入新噪声（随机链） | **确定性 ODE** 积分：同一起始噪声解唯一（P4） |
| 训练目标 | 预测所加噪声 $\epsilon^k$（DDPM 简单目标） | 回归路径导数 $A_t - \epsilon$（P3） |
| 推理开销 | 训练 100 步、DDIM 解耦后推理 10 步（0.1 s @ RTX 3080） | 10 步积分 + 一次 VLM 前向（73 ms/块 @ RTX 4090） |
| 条件化 | 编码器特征外注入（cross-attention/injection） | VLM 主干 token 级拼接，语义与动作同场注意力 |
| 输出粒度 | 动作块（horizon 8，DP 论文设置） | 动作块 $H=50$，最高 50 Hz 执行 |
| 频控结论 | 0.1 s 级延迟适配中频闭环 | 块执行摊薄延迟，适配 50 Hz 高频灵巧控制 |

## 5. 实验与结果解读

**评分制**（§VI、附录 E）：每任务 **10 次 rollout 的归一化平均分**，满分 1.0，部分完成按比例给分（如 bussing 按正确入桶物体数、折衣按折平袖子等子目标给分）。本节图表均为柱状图、无印刷数字，故一律定性表述。

### 5.1 开箱即用（Fig. 6–7）

- 任务：折 T 恤、bussing easy（7 物）/ bussing hard（12 物、含预训练外物体与互相遮挡）、食品装袋、烤面包出机——均需灵巧操作 + 多阶段行为 + 语义识别的组合；
- 结果：完整 π0（700k 步预训练）**所有任务最佳**；与基线等算力的 160k 步 "compute parity" 版**仍超全部基线**；
- 基线诊断：**OpenVLA 明显落后**——其自回归离散化**不支持动作分块**，与高频灵巧任务失配；仅在 UR5e 数据上微调的 OpenVLA 变体好一些但仍远低于 π0；Octo 支持分块但 93M 容量有限；
- 结论：分块表达力 + VLM 预训练 + 数据规模三因素缺一不可。

### 5.2 语言跟随（Fig. 8–9）

- 设置：bussing / 摆桌 / 装袋 3 任务 × 三档指令——flat（总指令如 "bag the groceries"）、human（专家人给中间步骤）、HL（高层 VLM 给中间步骤）；
- 结果：π0 的语言跟随**显著优于**无 VLM 初始化的 π0-small → **VLM 预训练直接转化为指令跟随能力**；π0-HL 最好；π0-small 因语言能力弱，加高层专家也无收益（论文明确指出该混杂因素）。

### 5.3 微调学新灵巧任务（Fig. 10–11）

- 任务分档：叠碗、毛巾折叠（**易**档，与预训练分布相近）；微波炉加热 Tupperware（微波炉为预训练未见物）、换纸巾、Franka 抽屉放物（**难**档，预训练无相似任务）；
- 协议：微调数据 1 / 5 / 10 小时三档；基线 = DP、ACT（只在各自微调数据上从零训练）+ Octo、OpenVLA（公开 OXE 预训练 checkpoint 再微调）；
- 结论（§VI-C 原文归纳）：① 微调自预训练权重的 π0 总体最佳；**最强的先验恰是那些在多样数据上预训练过的模型**；② 预训练带来的增益在更难任务上更大（论文原话"有时高达 2 倍"）；③ π0 的曲线随数据增加持续上升，DP 全程垫底。

### 5.4 复杂多阶段长程任务（Fig. 12–13）

- 任务：双臂折衣（皱衣任意初始构型，需抓—展—折—叠序列）、移动平台折衣、烘干机卸衣、清桌（未知物体、需 π0-HL 分解）、纸箱组装（折痕对齐、双臂压持箱体、必要时重试）、to-go 盒打包、装蛋（蛋形易滑、精细放置、双臂合盖）、打包餐食；任务历时 **5–20 分钟**；满分制见附录 E（折衣 4 分、清桌 12 分、装箱 5 分、装蛋 7 分等）；
- 结果：π0（全量预训练 + 微调）**全场最佳，所有任务超过满分 50%**，最难任务改进最大；"仅微调不预训练"（out-of-training）与"从零"（scratch）两个消融都大幅更差；
- 意义：直接支撑 §2 的"预训练广度 + 后训练质量"哲学——预训练模型见过错误与恢复行为，才学得会长程多阶段任务。

## 6. 局限与后续影响

**作者自述（§VII Discussion, Limitations, and Future Work）**：

1. 预训练数据**应如何组合仍不清楚**："把所有能拿到的数据全混进去"并非最优，各数据源如何加权是开放问题；
2. 并非所有任务都可靠；难以预测需要多少、什么样的数据才能接近完美执行；
3. 跨任务、跨机器人的**正迁移**程度有待理解；
4. 框架能否推广到自动驾驶、导航、腿式运动等其他域，是未来工作。

**读者补充**：

5. 本论文**未开源权重**（openpi 与 π0-FAST 为后续论文外进展），复现门槛显著高于 OpenVLA；
6. 最长程任务（清桌等）仍需外接高层 VLM（π0-HL），π0 本体只承担 5–20 分钟内的技能段执行；
7. 每块 73 ms 推理 + 开环整块执行，意味着块间纠错延迟为 0.5–0.8 s 量级（附录 D 执行协议的隐含代价）。

**后续影响**：确立"VLM 主干 + 动作专家 + 流匹配动作块 + 两阶段 recipe"作为灵巧操作基础模型的主流配方之一；评分制与"预训练 vs 从零"消融范式被后续灵巧操作论文广泛沿用；π0-FAST（语义 token 加速）为后续工作。

## 7. 与本项目对照

- **RoboTwin 2.0 基准中的 π0**（数字引自 [RoboTwin 2.0 精读 §5.5](./RoboTwin2.0_arXiv2025.md)，对应论文 Table 5）：
  - 协议：50 任务、Aloha-AgileX 单本体、每任务 50 条干净专家演示微调、每任务 100 rollouts、Easy（干净）/ Hard（五维 DR）两档；
  - 数字：**π0 Easy 46.4% / Hard 16.3%**——Hard 档**全场最佳**，Easy 档次于 DP3 的 55.2%；
  - 训练设置（该精读附录 D 口径）：预训练 10 万步（batch 32）+ 微调 3 万步；
  - 关键读数：Easy→Hard 掉 **30.1 个百分点**——π0 的 VLA 先验给出五策略中最好的 DR 鲁棒性**下限**，但域偏移仍是未解挑战（与该精读 §6 结论一致）。
- **三路线对照表**（动作生成范式；离散 token 谱系见 [RT-1 精读 §4.1](./RT1_arXiv2022.md) 与 [RT-2 精读](./RT2_arXiv2023.md)，分块谱系见 [ACT 精读 §4.1](./ACT_ALOHA_RSS2023.md)）：

| 路线 | 代表 | 动作表示 | 推理 | 频控 | 多模态分布 |
|---|---|---|---|---|---|
| 自回归离散 token | RT-1 / RT-2 / OpenVLA | 逐维 256-bin 分类 token | 单次前向（OpenVLA ≈6 Hz @ RTX 4090） | 低频、无分块 | 隐式、逐维自回归 |
| CVAE 分块 | ACT | $k{=}50$ 步连续块 | 单次解码（≈0.01 s）+ 时间集成 | 50 Hz 块执行 | 显式（隐变量采样） |
| 扩散 | DP / RDT / Octo | 连续动作块 | 迭代去噪（DDIM 10 步 ≈0.1 s） | 中频 | 显式（迭代精化） |
| **流匹配** | **π0** | $H{=}50$ 连续块 | 10 步 ODE + VLM 前向（73 ms/块） | **50 Hz 块执行** | 显式（速度场积分） |

- **教程映射**：[第 08 章 §8.2](../08_策略训练与部署.md) 策略表"Pi0 (π0)：流匹配生成动作的 VLA 基础模型"即本文；§8.3 训练细节表与 §8.4"预训练→微调"流程是 π0 在 RoboTwin 上的实操形态；π0 评测的 100-rollout 协议与教程 §8.6 一致；[第 09 章 §9.2](../09_进阶研究方向.md) 将 π0 收录于 VLA 清单（"流匹配 VLA，通用机器人控制"）。π0 后训练仍是模仿学习，不涉及 [控制规划教程第 08 章 RL 式 (8.1)](../../control_planning/08_学习式控制强化学习.md) 的 RL 目标。
- **实践提示**：RoboTwin 场景复刻 π0 的可行路径 = openpi 开源实现 + 小时级后训练数据（论文 §V-A 的 5–100 小时区间）；其 Hard 档落差提示——RoboTwin 生成器的**域随机化数据**正是补 π0"Easy→Hard 跌落"的数据杠杆（对照 [RoboTwin 2.0 精读 §4.3](./RoboTwin2.0_arXiv2025.md) 的 DR 预训练实验）。

## 配套阅读

- 教程内：[第 08 章 8.2–8.6](../08_策略训练与部署.md)、[第 09 章 9.2 阅读清单](../09_进阶研究方向.md)、[控制规划教程第 08 章 RL 式 (8.1)](../../control_planning/08_学习式控制强化学习.md)
- 姊妹精读：[OpenVLA（离散 token 对照面）](./OpenVLA_CoRL2024.md)、[Diffusion Policy（DDPM vs ODE 详推）](./DiffusionPolicy_RSS2023.md)、[ACT（50 Hz 分块谱系源头）](./ACT_ALOHA_RSS2023.md)、[RT-1（3 Hz 离散 token 起点）](./RT1_arXiv2022.md)、[RT-2（自回归 VLA 路线）](./RT2_arXiv2023.md)、[RoboTwin 2.0（π0 双臂评测所在基准）](./RoboTwin2.0_arXiv2025.md)
- 论文原文：Lipman et al. 2023（Flow Matching，arXiv:2210.02747，(P1)–(P3) 框架出处）、Liu et al. 2022（Rectified Flow，arXiv:2209.03003）、PaliGemma（arXiv:2407.07726）、Octo（arXiv:2405.12213）、OpenVLA（arXiv:2406.09246）
- 教程外：Transfusion（Zhou et al. 2024，双目标单 Transformer 设计参照）、SD3（Esser et al. 2024，时间步采样对照）、Gemma 2B（Gemma Team 2023）、SayCan（Ahn et al. 2022，π0-HL 高层分解血统）、ALOHA 2（Zhao et al. 2024，双臂硬件血统）
- 项目页：physicalintelligence.company/blog/pi0（折衣、装箱等任务视频）

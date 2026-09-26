# 论文精读｜RT-1（arXiv 2022）

> **PDF**：[../../../papers/robotwin/classics/arXiv-2212.06817_RT1.pdf](../../../papers/robotwin/classics/arXiv-2212.06817_RT1.pdf) ｜ **教程**：[第 01 章](../01_具身智能入门.md)、[第 08 章](../08_策略训练与部署.md) ｜ **代码**：本体为 Google Everyday Robots 移动操作平台（模型与数据未开源），项目页 robotics-transformer1.github.io 提供视频与说明
>
> 阅读前提：[第 01 章](../01_具身智能入门.md) 1.3 的三大范式（RL / 模仿学习 / VLA）与 [第 08 章](../08_策略训练与部署.md) 8.1 的"策略"定义。

## 1. 论文信息与一句话贡献

- **题目**：RT-1: Robotics Transformer for Real-World Control at Scale
- **作者/机构**：Brohan、Brown、Carbajal 等数十位共同一作，Google Robotics / Everyday Robots / Google Research Brain Team
- **发表**：arXiv:2212.06817（2022 年 11 月首发；本地 PDF 为 2023-08 的 v2 预印本，全文以该 PDF 为准）
- **一句话贡献**：用 13 台真机、17 个月收集的 **13 万+ 条真实演示**（覆盖 **700+ 条任务指令**），训练一个 **35M 参数、3Hz 实时推理** 的 Transformer 策略（FiLM-EfficientNet + TokenLearner + decoder-only Transformer + 离散动作 token），在见过的任务上达到 **97%** 成功率，并系统性论证了数据规模、模型容量与数据多样性对真实机器人泛化的 scaling 关系。

> ⚠️ 勘误提示：网上常见"RT-1 用了 700k+ episodes"的说法与论文原文不符——论文 §1 与 §5.2 均写明数据集是 "over 130k individual demonstrations / episodes"（13 万+），"700+" 指的是**任务指令条数**（over 700 distinct task instructions）。本文一律按原文 130k+ episodes / 700+ instructions 表述。

## 2. 问题与动机

**要解决的问题**（论文 §1）：端到端机器人学习（无论模仿还是强化）传统上都为单个任务收集窄分布的专用数据，模型无法跨任务吸收经验。CV/NLP 已完成"小模型 + 小数据 → 大模型 + 大数据"的范式转移，机器人还没有——因为机器人数据采集极贵，**泛化**因此更加关键。

**三个核心研究问题**（论文 §6 开篇）：

1. RT-1 能否学会大量指令，并泛化到新任务、新物体、新环境？
2. 能否通过吸收异构数据（仿真、其他机器人）进一步变强？
3. 模型与数据设计中的哪些决策真正影响性能与泛化？

**为什么"数据"是主角**：论文的立场是，RT-1 的能力"在很大程度上由数据集与任务集决定"。这正是[第 01 章](../01_具身智能入门.md) 1.4 "数据是当前最大瓶颈"论断的原始出处之一：没有跨任务的大规模真机数据，任何架构都无法展示泛化。

**部署约束**（论文 §5.1，容易被忽视的动机）：人类执行论文中一条指令约需 2–4 秒，机器人要"不显著更慢"，对应**至少 3Hz 控制频率**；扣除相机延迟与通信开销后，模型推理时间预算 **< 100ms**。这个约束直接塑造了架构（§4.5）。

## 3. 方法总览

RT-1 是一个**条件策略** $\pi_\theta(a_t \mid o_{t-4:t},\,\ell)$：输入最近 6 帧图像（300×300）与一条自然语言指令 $\ell$，输出一个动作 $a_t$，以 3Hz 闭环执行，直到输出"terminate"或达到步数上限（论文 §2 末、§5.1）。

数据流（论文 Fig. 3，自底向上）：

1. **指令编码**：Universal Sentence Encoder（USE，Cer et al. 2018）把指令嵌入为 512 维向量；
2. **FiLM 条件化图像编码**：ImageNet 预训练 EfficientNet-B3 逐帧编码，内部 MBConv 块间插入 **FiLM 层**用指令嵌入调制图像特征——语言信息在**早期**就参与视觉特征筛选；输出 9×9×512 特征图，展平为 **81 个视觉-语言 token**（FiLM-EfficientNet-B3 共 16M 参数、26 个 MBConv 块、26 个 FiLM 层）；
3. **TokenLearner 压缩**：逐元素注意力把每帧 81 个 token 软选择为 **8 个**；6 帧拼成 **48 个 token**（加位置编码）；
4. **decoder-only Transformer**：8 层自注意力、19M 参数，输出**离散动作 token**；
5. **动作离散化**：11 维动作逐维离散化为 256 bins（§4.1），训练用分类交叉熵（§4.4）。

与两条路线的关系：它是[第 01 章](../01_具身智能入门.md) 1.3 **路线二（模仿学习/行为克隆）**在多任务大规模数据上的形态，同时是 1.3 **路线三（VLA）**的直系前身——[第 09 章](../09_进阶研究方向.md) 9.2 的 VLA 谱系表把 RT-1 列为第一代"视觉-语言-动作 Transformer"。

## 4. 关键公式推导

> **式号说明（重要）**：RT-1 是系统型论文，**原文没有编号公式**。本章所有编号 (R1)–(R6) 是本文档自加的重新形式化：每个式子先给出论文原文依据（节号/原句），再逐步推导；凡属论文引用的外部标准结果（FiLM、TokenLearner），均注明原始文献。

### 符号表

| 符号 | 含义 | 备注（出处） |
|------|------|------|
| $o_{t-4:t} = (o_{t-5},\dots,o_t)$ | 最近 6 帧图像历史 | 论文 §5.1 "a history of 6 images" |
| $\ell$ | 自然语言指令 | 同上 |
| $e \in \mathbb{R}^{512}$ | USE 指令嵌入 | 论文 Fig. 3 |
| $F \in \mathbb{R}^{9\times 9\times 512}$ | 单帧图像特征图 | §5.1 "spatial feature map of shape 9×9×512" |
| $z_{ij}$ | 第 $i$ 帧的第 $j$ 个视觉-语言 token（共 81） | §5.1 |
| $a_t \in \mathbb{R}^{11}$ | 动作向量 | §5.1 |
| $l_d, u_d$ | 第 $d$ 维动作的取值下/上界 | §5.1 "bounds of each variable" |
| $b_d \in \{0,\dots,255\}$ | 第 $d$ 维离散化后的 bin 编号 | §5.1 "256 bins" |
| $p_\theta(\cdot)$ | 模型输出的离散分布 | §5.1 Loss 段 |

### 4.1 动作离散化：为何与如何

**论文原文**（§5.1 Action tokenization）："each action dimension in RT-1 is discretized into 256 bins… the bins are uniformly distributed within the bounds of each variable"。动作的 11 维构成：**7 维臂**（$x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw}$，夹爪开合）+ **3 维基座**（$x,y,\mathrm{yaw}$）+ **1 维离散模式**（控制臂 / 控制基座 / 终止回合）。

**如何离散化**（把原文规则形式化）：对第 $d$ 维连续目标值 $a_d \in [l_d, u_d]$，均匀映射到 bin 编号：

$$
b_d \;=\; \mathrm{clip}\!\left(\left\lfloor \frac{a_d - l_d}{u_d - l_d} \cdot 256 \right\rfloor,\; 0,\; 255\right)
\tag{R1}
$$

依据：(R1) 是"uniformly distributed within the bounds"的直接数学化——分母 $u_d - l_d$ 归一化到 $[0,1)$，乘 256 落到 bin 宽度 $（u_d-l_d)/256$ 的区间，下取整得编号；clip 处理右端点（$a_d = u_d$ 时上取整会得到 256，需截断回 255）。

**为何要离散化**（三条依据，全部出自论文）：

1. **Token 化是 Transformer 输出动作的前提**。Transformer 的输出头天然是分类分布（softmax），动作 token 化后，策略变成"下一 token 分类"问题，与语言建模同构（论文 §1：与 Gato、BeT 一脉的 "discretized action tokens"）。这是连续控制问题嫁接到序列建模框架的桥梁。
2. **多峰表达力**。消融 §6.5-(3)：把离散动作换成"连续动作 + 多维高斯输出 + MSE 损失"后性能**显著下降**，原文解释："per-dimension discretization allows our model to represent complex multi-modal distributions, while the Gaussian distribution captures only a single mode"。256 bins 的逐维分类分布可以表达任意多峰（bin 直方图），单高斯只有一个峰——这也是 [Diffusion Policy 精读](./DiffusionPolicy_RSS2023.md) §4.4 用连续扩散模型解决的同一个问题，两文在动机层完全一致、在手段上分流。
3. **训练稳定性**。分类交叉熵对 bin 内误差不敏感（预测错相邻 bin 与预测错离谱 bin 的对数损失不同阶），比回归对动作幅度异常值更稳健（论文 §6.5 把该设计归因于此前 Transformer 控制器的成熟实践：Reed et al. 2022; Lee et al. 2022a）。

代价与修正：均匀 256 bins 分辨率有限（每 bin 宽 $(u_d-l_d)/256$），后续 BeT 在 bin 上加连续偏移修正，ACT 表述其差异时提到 "categorical distribution over discrete bins, but with an added continuous offset from the bin center"（ACT 论文 §V-C，RT-1 无此修正，见[ACT 精读](./ACT_ALOHA_RSS2023.md) §5）。

### 4.2 FiLM 条件化：语言如何调制视觉特征

**论文原文**（§5.1）：指令经 USE 得到嵌入，"used as input to identity-initialized FiLM layers added to the pretrained EfficientNet to condition the image encoder"。

FiLM（Feature-wise Linear Modulation，Perez et al. 2018）的标准定义：设条件信息（此处为指令嵌入）$e$，特征图 $F$ 的第 $c$ 个通道为 $F_c$，则 FiLM 用 $e$ 预测逐通道仿射变换：

$$
\mathrm{FiLM}(F \mid e)_c \;=\; \gamma_c(e)\,\odot\, F_c \;+\; \beta_c(e)
\tag{R2}
$$

其中 $\gamma_c, \beta_c \in \mathbb{R}^{H\times W}$ 是由 $e$ 经稠密层产生的逐空间位置调制系数，$\odot$ 为逐元素乘。依据：这是 FiLM 原论文的定义，RT-1 论文 Fig. 3 中每个 MBConv 块后的 "β (1 γ) FiLM" 即此结构。

**恒等初始化的推导**。问题（论文原文指出）：向预训练网络内部插入调制层会破坏中间激活，使 ImageNet 预训练权重白费。论文的解法："initialize the weights of the dense layers ($fc$ and $h_C$) which produce the FiLM affine transformation to zero, allowing the FiLM layer to initially act as an identity and preserve the function of the pretrained weights."

逐步论证"零初始化 ⇒ 初始为恒等"：

1. 要使 (R2) 初始时刻等于 $F_c$，充要条件是初始 $\gamma_c(e) \equiv \mathbf{1}$ 且 $\beta_c(e) \equiv \mathbf{0}$（依据：把 $F_c' = \gamma \odot F_c + \beta$ 与 $F_c$ 比对，对任意 $F_c$ 成立当且仅当 $\gamma=1, \beta=0$）。
2. 等价地，把调制改写成"残差形式" $\gamma_c = \mathbf{1} + \Delta\gamma_c,\ \beta_c = \Delta\beta_c$：则 $F_c' = (1+\Delta\gamma_c)\odot F_c + \Delta\beta_c$，恒等 ⇔ $\Delta\gamma_c = \Delta\beta_c = 0$（依据：代数恒等变形，与上一步等价）。
3. 因此"把产生仿射变换的稠密层零初始化"配合残差式参数化，初始时 $\Delta\gamma = \Delta\beta = 0$，FiLM 层恰为恒等映射；训练中语言信号通过梯度逐步"唤醒"调制（依据：论文原句 "initially act as an identity" 的数学展开；论文另报告该初始化在从头训练时也有收益，但不及 ImageNet 预训练 + 恒等初始化的组合）。

效果：语言早期介入使视觉特征**任务相关**（同一段视频，"pick the apple"与"move the apple"关注的通道不同），这是 RT-1 区别于"先编码视觉、后拼接语言"方案（如 late-fusion）的核心设计，论文 §6.5 消融显示 ImageNet 预训练 + FiLM 早期融合对泛化尤其关键（去掉 ImageNet 预训练使未见任务掉 33%）。

### 4.3 TokenLearner：81 → 8 的有理由压缩

**论文原文**（§5.1 TokenLearner）："TokenLearner is an elementwise attention module that learns to map a large number of tokens into a much smaller number of tokens… soft-select image tokens based on their information."（结构出自 Ryoo et al. 2021）

形式化：单帧 81 个 token 堆叠为 $Z \in \mathbb{R}^{81\times 512}$，TokenLearner 用卷积网络从 $Z$ 生成注意力图 $A \in \mathbb{R}^{81\times 8}$（对 81 个 token 维做 softmax 归一），输出 8 个 token：

$$
z'_j \;=\; \sum_{i=1}^{81} \alpha_{ij}\, z_i, \qquad
\alpha_{\cdot j} = \mathrm{softmax}_i\big(\mathrm{Conv}(Z)\big)_{ij}, \qquad j = 1,\dots,8
\tag{R3}
$$

依据：(R3) 是原文 "elementwise attention / soft-select" 的标准数学化（Ryoo et al. 2021 的自适应池化形式）——不是取平均（$\alpha$ 依赖内容、逐 token 学出），而是**可微的软选择**，梯度可以回传去学"该保留哪些 token 组合"。

**压缩的收益计算**（依据：自注意力复杂度公式 $\mathcal{O}(N^2 d)$，Vaswani et al. 2017）。不去压缩时 6 帧 × 81 = 486 个 token 进入 Transformer；压缩后 6 × 8 = 48 个。注意力矩阵规模比：

$$
\frac{48^2}{486^2} = \frac{2304}{236196} \approx 0.98\%
$$

即注意力部分理论上压缩约 **102 倍**（依据：平方律）。论文实测整模型加速 **2.4 倍**（§5.1 Inference speed：TokenLearner 使推理加速 2.4×；注意这是含视觉主干在内的端到端口径，故小于注意力的理论比值）。没有这一步，35M 模型无法满足 3Hz（§3 的部署约束）。

### 4.4 训练目标：离散动作的极大似然 = 分类交叉熵

**论文原文**（§5.1 Loss）："We use a standard categorical cross-entropy objective and causal masking that was utilized in prior Transformer-based controllers (Reed et al., 2022; Lee et al., 2022a)."

从极大似然推导。数据集 $\mathcal{D} = \{(o^i_{t-4:t}, \ell^i, a^i_t)\}_{i=1}^{N}$，策略是参数化条件分布 $\pi_\theta(a \mid o,\ell)$。极大似然估计（MLE，依据：监督模仿学习即对演示动作做极大似然，对应[第 01 章](../01_具身智能入门.md) 1.3 路线二）：

$$
\max_\theta \; \sum_{i} \log \pi_\theta\big(a^i_t \,\big|\, o^i_{t-4:t}, \ell^i\big)
\tag{R4}
$$

代入离散化（(R1)）：动作是 11 维独立离散变量，$\pi_\theta(a\mid o,\ell) = \prod_{d=1}^{11} p_\theta\big(b_d \mid o,\ell\big)$（依据：论文按维独立预测、每维一个 256 类 softmax，Fig. 3 的 11 个输出头），故

$$
\log \pi_\theta(a \mid o,\ell) \;=\; \sum_{d=1}^{11} \log p_\theta\big(b_d \,\big|\, o,\ell\big)
\;\;\Longrightarrow\;\;
\mathcal{L}(\theta) \;=\; -\sum_{d=1}^{11} \log p_\theta\big(b_d(a_d) \,\big|\, o,\ell\big)
\tag{R5}
$$

(R5) 即**逐维分类交叉熵**（依据：负对数似然的分类标准形；"causal masking" 指 Transformer 内部 token 间的因果掩码，继承自序列建模实践）。注意 RT-1 **不做动作自回归**：11 维动作一次并行输出——论文消融 §6.5-(4) 明确报告，改为像 Gato 那样自回归地条件于已生成的动作 token "did not benefit performance and slowed inference by more than 2x"。这一点与 [Diffusion Policy 精读](./DiffusionPolicy_RSS2023.md) 的序列预测、[ACT 精读](./ACT_ALOHA_RSS2023.md) 的 k 帧分块形成三种不同的"动作时序组织"范式。

### 4.5 推理吞吐设计：3Hz 是怎么凑出来的

预算分解（论文 §5.1 Inference speed）：3Hz ⇒ 每步 333ms；扣除相机与通信延迟 ⇒ 模型 < 100ms。两个提速手段（论文原文给出实测倍数）：

1. **TokenLearner 压缩 token 数**：加速 **2.4×**（机制见 (R3) 的平方律分析）；
2. **跨窗复用 token**：6 帧滑窗相邻步重叠 5 帧，重叠帧的视觉-语言 token **只算一次、后续窗复用**，加速 **1.7×**（依据：§5.1 "compute these tokens only once and reuse them for the following windows that overlap"）。

两者相乘的量级（$2.4 \times 1.7 \approx 4.1$）使 35M 模型落入 100ms 预算。这个"为实时而反推架构"的设计顺序，是 RT-1 区别于同期大模型（Gato 无法实时上真机，论文 §6.1 明确说明因此把 Gato 缩到 37M 同量级比较）的关键工程贡献。

## 5. 实验与结果解读

**数据与评测设置**（论文 §2、§5.2、§6.1）：13 台 Everyday Robots 移动操作臂，17 个月，**130k+ episodes**，700+ 指令（约 700+ 个"技能×物体"组合）；训练环境为 1 个模拟真实厨房的采集教室，评测在 2 个**真实**厨房（Kitchen1/Kitchen2，光照、背景、几何均不同于训练环境）。评测指令 200+ 条（36 挑选、35 击倒、35 立放、48 移动、18 开合抽屉、36 抽屉取放），另有 21 条**未见**指令（技能与物体都在训练集中见过、但组合是新的）。

**主结果（论文 Table 2，成功率 %）**：

| 模型 | 见过任务 | 未见任务 | 干扰物鲁棒 | 背景鲁棒 |
|------|------|------|------|------|
| Gato（37M，同数据重训） | 65 | 52 | 43 | 35 |
| BC-Z | 72 | 19 | 47 | 41 |
| BC-Z XL | 56 | 43 | 23 | 35 |
| **RT-1** | **97** | **76** | **83** | **59** |

要点：见过任务 97%，比 BC-Z 高 25 个百分点、比 Gato 高 32 个百分点；未见指令 76%，比次优高 24 个百分点（依据：Table 2 及 §6.2 原文）。语言组合泛化的来源是语言条件化——但论文强调所有基线同样有语言条件，因此差距要归因于**架构 + 数据共同作用**（§6.2）。

**真实厨房长程场景（论文 Table 3）**：由 SayCan 把"如何把桌上的东西都扔掉"类高层指令拆成约 10 步技能序列，在真实厨房按泛化层级打分：L1（新台面与光照）RT-1 88%、L2（再加未见干扰物）75%、L3（再加全新任务设定/物体位置）50%；Gato 从 L1 的 63% 跌到 L3 的 0%，BC-Z 为 28/66/50。RT-1 是唯一全层级 ≥50% 的模型（依据：Table 3）。

**消融（论文 §6.5 / Table 13，定性总结）**：逐维离散动作 → 连续高斯输出**显著掉点**（多峰论证，见 (R1)）；去 ImageNet 预训练 → 未见任务 −33%；去 TokenLearner/Transformer 结构 → 全面小幅下降；去 6 帧历史 → 主要伤干扰物鲁棒性；动作自回归 → 无收益且推理慢 2 倍以上。结论：**动作表示、预训练初始化、历史输入、模型容量四个决策都重要，但重要程度不同**。

**结论**：RT-1 的成功 = 大规模多样数据 ×（容量足够且实时）的架构；它首次在真实家庭/厨房任务族上把"数据 scaling ⇒ 泛化提升"做成受控对照实验。

## 6. 局限与后续影响

**论文自己承认/数据可见的局限**：

1. **数据封闭**：130k+ 真机数据集未开源，外界无法复现其数据 scaling 曲线——这正是社区推动 Open X-Embodiment 与 RT-2X 数据联盟的直接动因；
2. **基座动作受限**：仅平面 $x,y,\mathrm{yaw}$ 三维，无高度/地形能力，场景限于厨房台面（论文 §5.1 动作定义）；
3. **语言泛化是"重组"而非"理解"**：未见指令的技能与物体都必须在训练集中出现过（§6.1 的 held-out 设计），没有新概念、新语义推理能力；
4. 3Hz、单臂任务的形态限制，无精确双手操作。

**后续影响**（对应[第 09 章](../09_进阶研究方向.md) 9.2 的谱系表）：RT-2（2023）把动作 token 化推到极限——直接让 VLM 输出动作 token，解锁网页知识带来的语义泛化；RT-X/Open X-Embodiment（2023）把 RT-1 架构用于跨机构数据混合；π0/RDT 等现代 VLA 的"动作 token + 大骨干"配方都始于 RT-1 的"策略即序列建模"命题。

## 7. 与本项目对照

- **在 RoboTwin 基准中的生态位**：RoboTwin 官方 5 策略基线（[第 08 章](../08_策略训练与部署.md) 8.2）不含 RT-1——RT-1 面向移动单臂 + 语言指令的桌面抓放族，而 RoboTwin 聚焦**双臂精细操作**且任务指令由模板生成。但 RT-1 的评测方法论（seen/unseen 任务分层、干扰物/背景鲁棒、L1–L3 泛化层级）正是 RoboTwin 2.0 的 Easy/Medium/Hard + 域随机化评测设计的思想源头（对照[第 06 章](../06_实验结果与解读.md) 6.6 的分层成功率表）。
- **范式差异**：RT-1 = 逐维 256-bin 分类分布 + 单步动作输出；ACT（[姊妹精读](./ACT_ALOHA_RSS2023.md)）= 连续动作 + CVAE + k 帧 chunk；Diffusion Policy（[姊妹精读](./DiffusionPolicy_RSS2023.md)）= 连续动作 + 迭代去噪 + 序列预测。三者对"动作分布怎么参数化"给出的三种答案，恰对应 RoboTwin 基准上三种 baseline 的训练行为差异（ACT/DP 在域随机化 Hard 下几乎归零而 VLA 抗跌，[第 06 章](../06_实验结果与解读.md) 6.6.3）。
- **对本项目读者的操作建议**：读 RT-1 重点吸收三件事——离散动作 token 化（理解 RDT/Pi0 的动作 token 基础）、3Hz 实时预算反推架构的工程方法、seen/unseen/鲁棒性分层评测表的设计（可直接套用到你自己的 RoboTwin 实验报告）。

## 配套阅读

- 教程内：[第 01 章 1.3 三大范式](../01_具身智能入门.md)、[第 08 章 8.2 五种策略](../08_策略训练与部署.md)、[第 09 章 9.2 VLA 谱系](../09_进阶研究方向.md)、[姊妹精读：Diffusion Policy](./DiffusionPolicy_RSS2023.md)、[姊妹精读：ACT/ALOHA](./ACT_ALOHA_RSS2023.md)
- 教程外：RT-2（arXiv:2307.15818）、Open X-Embodiment（arXiv:2310.08864）、BC-Z（Jang et al., 2021）、Gato（Reed et al., 2022）
- 项目页：robotics-transformer1.github.io（含视频与模型卡）

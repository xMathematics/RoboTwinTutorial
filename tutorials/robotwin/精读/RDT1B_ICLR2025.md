# 论文精读｜RDT-1B：双臂操作扩散基础模型（ICLR 2025）

> **PDF**：[papers/robotwin/frontier/arXiv-2410.07864_RDT-1B.pdf](../../../papers/robotwin/frontier/arXiv-2410.07864_RDT-1B.pdf)（arXiv:2410.07864v2，"Published as a conference paper at ICLR 2025"；本精读所有式号、表格号与数字均按该 PDF 逐页核对）｜ **教程**：[第 08 章｜策略训练与部署](../08_策略训练与部署.md)（RDT 为五种策略之一）、[第 09 章｜进阶研究方向](../09_进阶研究方向.md) ｜ **代码**：官方全开源（预训练权重 + 微调数据 + 训练/部署脚本，模型仓库 HuggingFace `thu-ml/RoboticsDiffusionTransformer`，入口为论文 project page）

## 1. 论文信息与一句话贡献

- **题目**：RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation
- **作者与单位**：Songming Liu、Lingxuan Wu、Bangguo Li、Hengkai Tan、Huayu Chen、Zhengyi Wang、Ke Xu、Hang Su、Jun Zhu（共同一作前两位，通讯 Jun Zhu）；清华大学计算机系、AI 基础设施中心（Bnrist）、清华-博世机器学习中心、THBI Lab。
- **发表**：ICLR 2025（arXiv v2 为会议版排版）。
- **一句话贡献**：把扩散模型从"单任务小策略"放大成**双臂操作基础模型**——以带三项机器人专属改造的去噪 Transformer（DiT，1.2B 参数）为骨干，配合**物理可解释的统一动作空间**（128 维，按物理含义填充对齐不同本体），在 46 个数据集、100 万+ 轨迹、21TB 的最大多机器人语料上预训练，再用自采 6K+ 轨迹双臂数据集微调；真机 7 项任务上零样本泛化（未见物体/场景）、语言跟随（"倒三分之一水"）、少样本学新技能（1~5 条演示）与精细操作（遥控机器狗）全面超过 ACT/OpenVLA/Octo。

## 2. 问题与动机

- **双臂操作的数据饥渴**：双臂演示需要协调两臂、动作空间翻倍、多模态严重，真机遥操作采集成本高，单机器人任务数据通常远不足训练基础模型所需的万级以上轨迹量（§1、§3：目标双臂机器人可用数据 "<10K 条"）。
- **解法：跨机器人预训练 + 目标机器人微调**（cross-robot pretraining，§3）——借多机器人数据放大数据量三个数量级，把"物理先验"从大规模数据中迁移出来。但这条路有两堵墙（论文 Challenge 1/2）：
  1. **架构挑战（表现力 + 可扩展性）**：双臂抓方块这类任务存在多个可行双手协同模式（Fig. 2b），确定性回归会学成"各模式的平均"而产生完全不可行的分布外动作（引 Pearce et al. 2023）；同时骨架必须能稳定地吃进文本/图像/动作等异构模态并随数据扩大。
  2. **数据挑战（异构性）**：不同本体的物理结构与动作空间差异巨大，负迁移风险高（§1）；此前做法要么只保留动作空间相近的机器人，要么丢弃结构不一致的输入，损失大量信息。

| 挑战 | 论文原文关键词 | RDT 对应组件 |
|---|---|---|
| Challenge 1：表现力（多模态动作分布） | expressiveness | 扩散建模（式 (1)(2)）+ 非线性 MLP 解码器 |
| Challenge 1：可扩展性（异构模态、大规模稳定训练） | scalability | DiT + QKNorm/RMSNorm + ACI 交替条件注入 |
| Challenge 2：跨本体异构数据 | negative transfer | 物理可解释统一动作空间（128 维对位填充） |

- **与既有扩散基础模型的差距**：Octo 是扩散基础模型但最大仅 93M 参数且不做双臂；OpenVLA 7B 但离散化动作在双臂上产生量化误差与不协调行为（§2 Related Work、§5.1 Baselines）。RDT 的定位：**最大的扩散式机器人操作基础模型（1.2B）**，且目标就是目标双臂机器人的能力而非"跨本体通用"本身（§3 明确说 "rather than developing a cross-embodiment model"）。

## 3. 方法总览

论文 Fig. 3 框架的忠实转述（条件与输出的完整清单）：

```text
语言指令 ℓ（T5-XXL 冻结编码 → 2 层 MLP 投影到 token 空间）
图像 X_{t-1:t+1}（T_img=2，外景/右腕/左腕三视角，SigLIP 冻结编码 → MLP 投影 + 多模态位置网格）
低维输入：本体感知 z_t、噪声动作块 ã_{t:t+T_a}、控制频率 c、扩散时间步 k
   → z_t 与 ã 先映射进 128 维统一动作空间，再经共享 MLP 编成 token（各物理量同语义不歧义）
   → c、k 各经一个 MLP 编成单个 token；全部拼接为长度 1 + T_a + 1 + 1 的 in-context 序列
        ▼
L × DiT Block（带交叉注意力）
   · QKNorm + RMSNorm（数值稳定，防 token shift）
   · 交叉注意力逐层交替注入图像/文本 token（Alternating Condition Injection, ACI）
        ▼
归一化 + 非线性 MLP 解码器 → 干净动作块 a_{t:t+T_a}（T_a = 64）
```

- **扩散骨架的理由**（§4.1）：动作 $\boldsymbol{a}_t$ 维度远低于图像，扩散的采样开销"minor"，而表现力与采样质量俱佳——继承 Diffusion Policy 的判断（见[Diffusion Policy 精读](./DiffusionPolicy_RSS2023.md) §4.4 的多峰论证）；但论文指出 CNN 时序卷积骨干（DP 的选择，偏好低频信号）不适配机器人数据的三个特性，故换 **Transformer（DiT）**。
- **机器人数据三特性 → 三项架构改造**（§4.1，论文原文表述）：①非线性动力学 → **MLP 解码器**替换最终线性投影；②高频变化（碰撞/阻尼等物理交互）→ 配合动作块建模；③数值范围不稳定（传感器极端值）→ **QKNorm**（防注意力数值爆炸）+ **RMSNorm**（去掉 LayerNorm 的中心化操作，避免时序预测中的 token shift，大尺度预训练"非常不稳定甚至爆炸"——Fig. 4a 损失曲线对照）。
- **物理可解释的统一动作空间**（§4.2 + App. C）：解决异构数据挑战——所有本体映射到同一 128 维向量，**逐维按物理含义对位填充、缺者补零**，保留物理语义以促进跨数据学习共享物理规律。
- **微调**（§4.2 + App. E）：自采 Mobile ALOHA 双臂多任务数据集（300+ 任务、6K+ 轨迹、300 万+ 帧）微调，弥合 embodiment 差距。

**预训练数据配方**（App. D，Table 5）：46 个数据集、合计 100 万+ 轨迹、21TB——"迄今最大的机器人数据集预训练合集"。初始采样权重按 $\sqrt{N_i}$（$N_i$ 为数据集大小）设置并依多样性与质量调整（依据 App. D：线性权重会让大数据集被过度采样、小数据集采样不足）；预训练中还按中间损失结果动态调权，调高收敛慢的数据集权重。主力数据集（轨迹数 / 采样占比均出自 App. D）：

| 数据集 | 规模（App. D 原文） | 控制频率 | 采样占比（Table 5） |
|---|---|---|---|
| RT-1 | 13 万条轨迹、13 种本体 | 3 Hz | 9.00% |
| DROID | 7.6 万条、564 场景、Franka | 15 Hz | 10.06% |
| RH20T | 11 万条、140 任务 | 10 Hz | 10.99% |
| Mobile ALOHA | 1K+ 条（双臂、14 维关节） | — | 4.98% |
| BridgeData V2 / BC-Z / CALVIN 等 OXE 系与仿真数据 | 若干 | — | 7.44% / 6.91% / 3.32% 等 |

数据清洗（App. D）：剔除重复前缀与失败回合、删除空白图像、剔除错误记录的速度、过滤过短轨迹、过长的降采样。多模态预处理（App. D）：图像统一三视角（外景 + 双腕，单臂机器人以背景色填充缺省视角）并 resize 为 384×384、$T_{img}=2$；本体感知/动作统一单位（m、rad 等）；语言指令做基本清洗并保留变量长度。

> **规格偏离说明**：任务规格猜测预训练含 "AgiBot"——论文通篇未出现该数据集（其发布晚于本文写作），实际主力为 RT-1/DROID/RH20T/Mobile ALOHA 与 OXE 系（RoboSet、BridgeData V2、CALVIN、ManiSkill 等），上表按论文实况给出。

## 4. 关键公式推导（核心章节）

> 式号 (1)(2) 按论文 PDF §4.1 逐字核对；动作块改造按 §4.1 末与 App. A。记号：$\boldsymbol{a}_t$ 为干净动作，$\boldsymbol{a}_t^k$ 为第 $k$ 轮噪声动作，$\ell$ 语言、$\boldsymbol{o}_t$ 观测。

### 4.1 建模对象：条件分布而非确定性映射

策略要学的是 $p(\boldsymbol{a}_t \mid \ell, \boldsymbol{o}_t)$（§4.1）。若建模成确定性映射 $(\ell,\boldsymbol{o}_t)\mapsto\boldsymbol{a}_t$ 并做回归，数据里的多峰动作会被"平均"，产生不可行的分布外动作——这是[Diffusion Policy 精读](./DiffusionPolicy_RSS2023.md) §4.4 的同一论证在双臂上的加强版（双臂的模式数更多，论文 Fig. 2b）。

### 4.2 反向去噪链：论文式 (1) 的完整推导

**背景式（前向闭式，Ho et al. 2020 标准结果，论文未单独编号；同见 Diffusion Policy 精读 (D0′)）**：前向过程每步加噪

$$
q(\boldsymbol{a}_t^k \mid \boldsymbol{a}_t^{k-1}) = \mathcal{N}\big(\sqrt{\alpha^k}\,\boldsymbol{a}_t^{k-1},\ \beta^k\boldsymbol{I}\big),
\qquad
\beta^k := 1-\alpha^k,\quad \bar{\alpha}^{k-1} := \prod_{i=1}^{k-1}\alpha^i
$$

由高斯分布的方差可加性逐级合并（依据：马尔可夫链逐步代入，$\sqrt{\alpha^k}$ 缩放前一级的均值与方差后叠加 $\beta^k\boldsymbol{I}$，归纳即得）

$$
\boldsymbol{a}_t^k = \sqrt{\bar{\alpha}^k}\,\boldsymbol{a}_t + \sqrt{1-\bar{\alpha}^k}\,\boldsymbol{\epsilon},\qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}).
\tag{F}
$$

**第一步（贝叶斯 + 马尔可夫性）**。采样时从 $\boldsymbol{a}_t^K\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ 出发执行 $K$ 步去噪；反向条件分布按贝叶斯公式分解（依据：$q(\boldsymbol{a}_t^{k-1}\mid\boldsymbol{a}_t^k,\boldsymbol{a}_t) = \frac{q(\boldsymbol{a}_t^k\mid\boldsymbol{a}_t^{k-1},\boldsymbol{a}_t)\,q(\boldsymbol{a}_t^{k-1}\mid\boldsymbol{a}_t)}{q(\boldsymbol{a}_t^k\mid\boldsymbol{a}_t)}$，前向是马尔可夫链故第一个因子等于 $q(\boldsymbol{a}_t^k\mid\boldsymbol{a}_t^{k-1})$）：

$$
q(\boldsymbol{a}_t^{k-1}\mid\boldsymbol{a}_t^k,\boldsymbol{a}_t) \propto q(\boldsymbol{a}_t^k\mid\boldsymbol{a}_t^{k-1})\;q(\boldsymbol{a}_t^{k-1}\mid\boldsymbol{a}_t).
$$

**第二步（两个高斯因子都显式写出）**。第一因子即前向定义：$\mathcal{N}(\sqrt{\alpha^k}\boldsymbol{a}_t^{k-1}, \beta^k\boldsymbol{I})$；第二因子是把 (F) 的 $k$ 换成 $k-1$ 解出 $\boldsymbol{a}_t$ 的分布（依据：(F) 对任意轮次成立）：$\mathcal{N}(\sqrt{\bar{\alpha}^{k-1}}\boldsymbol{a}_t,\ (1-\bar{\alpha}^{k-1})\boldsymbol{I})$。

**第三步（取对数、按二次型配方）**。两个高斯乘积仍是高斯（依据：指数上的二次型相加）。展开指数并按 $\boldsymbol{a}_t^{k-1}$ 配方，只保留依赖它的二次项与一次项：

$$
\log q = -\frac12\Big[\underbrace{\Big(\tfrac{\alpha^k}{\beta^k} + \tfrac{1}{1-\bar{\alpha}^{k-1}}\Big)}_{=:P}\|\boldsymbol{a}_t^{k-1}\|^2 - 2\Big(\tfrac{\sqrt{\alpha^k}}{\beta^k}\boldsymbol{a}_t^k + \tfrac{\sqrt{\bar{\alpha}^{k-1}}}{1-\bar{\alpha}^{k-1}}\boldsymbol{a}_t\Big)^{\!\top}\!\boldsymbol{a}_t^{k-1} + C\Big]
$$

**第四步（精度与方差）**。精度 $P$ 通分（依据：$\bar{\alpha}^k=\alpha^k\bar{\alpha}^{k-1}$、$\beta^k=1-\alpha^k$）：

$$
P = \frac{\alpha^k(1-\bar{\alpha}^{k-1}) + \beta^k}{\beta^k(1-\bar{\alpha}^{k-1})}
  = \frac{\alpha^k - \bar{\alpha}^k + 1 - \alpha^k}{\beta^k(1-\bar{\alpha}^{k-1})}
  = \frac{1-\bar{\alpha}^k}{\beta^k(1-\bar{\alpha}^{k-1})}
\;\Longrightarrow\;
(\sigma^k)^2 = \frac{\beta^k(1-\bar{\alpha}^{k-1})}{1-\bar{\alpha}^k}.
$$

**第五步（均值 = 一次项系数 ÷ 精度）**（依据：高斯密度配方后"一次项系数 = 精度 × 均值"）：

$$
\tilde{\boldsymbol{\mu}} = \frac{\beta^k(1-\bar{\alpha}^{k-1})}{1-\bar{\alpha}^k}\Big(\tfrac{\sqrt{\alpha^k}}{\beta^k}\boldsymbol{a}_t^k + \tfrac{\sqrt{\bar{\alpha}^{k-1}}}{1-\bar{\alpha}^{k-1}}\boldsymbol{a}_t\Big)
= \frac{\sqrt{\bar{\alpha}^{k-1}}\,\beta^k\,\boldsymbol{a}_t + \sqrt{\alpha^k}\,(1-\bar{\alpha}^{k-1})\,\boldsymbol{a}_t^k}{1-\bar{\alpha}^k}.
$$

**论文式 (1)**（反向采样，逐项与推导一致）：

$$
\boldsymbol{a}_t^{k-1}
= \frac{\sqrt{\bar{\alpha}^{k-1}}\,\beta^k}{1-\bar{\alpha}^k}\,\boldsymbol{a}_t
+ \frac{\sqrt{\alpha^k}\,(1-\bar{\alpha}^{k-1})}{1-\bar{\alpha}^k}\,\boldsymbol{a}_t^k
+ \sigma^k\boldsymbol{z},
\qquad k = K,\dots,1
\tag{1}
$$

其中 $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ 当 $k>1$，且 $\bar{\alpha}^0=1$、$\boldsymbol{z}=\boldsymbol{0}$（依据：论文式 (1) 下方定义——空积为 1，最后一步无随机性）。**关键衔接**：式中干净动作 $\boldsymbol{a}_t$ 在采样结束前不可得，故用可学习网络从噪声动作估计它：$\boldsymbol{a}_t \leftarrow f_\theta(\ell, \boldsymbol{o}_t, \boldsymbol{a}_t^k, k)$（论文式 (1) 下方原文）。注意 $f_\theta$ 预测的是**干净样本**（x0-prediction），而非 Diffusion Policy 的噪声 $\boldsymbol{\epsilon}$。

### 4.3 训练目标：论文式 (2) 与动作块改造

由 (F)，任意轮次的噪声输入可以一步构造，故训练时随机抽轮次。**论文式 (2)**（去噪 MSE）：

$$
\mathcal{L}(\boldsymbol{\theta}) := \mathrm{MSE}\Big(\boldsymbol{a}_t,\ f_\theta\big(\ell,\ \boldsymbol{o}_t,\ \sqrt{\bar{\alpha}^k}\,\boldsymbol{a}_t + \sqrt{1-\bar{\alpha}^k}\,\boldsymbol{\epsilon},\ k\big)\Big),
\tag{2}
$$

其中 $k\sim\mathrm{Uniform}(\{1,\dots,K\})$、$\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$、三元组 $(\ell,\boldsymbol{o}_t,\boldsymbol{a}_t)$ 从训练集采样（依据：式 (2) 下方原文）。式 (2) 逐项含义：输入 = (F) 构造的带噪动作（论文记 $\tilde{\boldsymbol{a}}_t$，略去 $k$ 上标），目标是干净动作——"估计干净样本"的最小二乘训练，等价地促使 $f_\theta$ 学会各噪声水平上的条件均值。

**动作块（action chunk）改造**（§4.1 末 + App. A）：一次性预测一段序列以鼓励时间一致性、并减少任务中的决策次数从而缓解误差累积（依据：论文引 Chi et al. 2023、Zhao et al. 2023）：

$$
p\big(\boldsymbol{a}_{t:t+T_a}\mid \ell, \boldsymbol{o}_t\big),\qquad
\boldsymbol{a}_{t:t+T_a} := (\boldsymbol{a}_t,\dots,\boldsymbol{a}_{t+T_a-1}),
$$

并把式 (1)(2) 中的 $\boldsymbol{a}_t$ 替换为 $\tilde{\boldsymbol{a}}_{t:t+T_a}$（App. A 原文）。$T_a = 64$（App. D：沿用先前消融研究的取值，兼顾效率与性能）。训练用 DDPM 调度器（glide cosine 变体，1000 步），采样用 DPMSolver++ 压到 **5 步**（App. H）。

### 4.4 多模态条件的编码与注入（App. B，无编号公式，逐条注明出处）

- **低维输入**：$\boldsymbol{z}_t$ 与 $\tilde{\boldsymbol{a}}_{t:t+T_a}$ 先映射进统一动作空间、再经**共享 MLP** 编成 token（同类物理量共享编码避免精度损失）；频率 $c$ 与时间步 $k$ 各经一个 MLP 编成单个 token；拼接成长度 $1 + T_a + 1 + 1$ 的 in-context 序列（App. B）。
- **图像**：SigLIP 冻结 + MLP 投影到 token 空间；位置编码扩展为 $(T_{img}, N_{cam}, N_{patch}, D)$ 的多模态网格，区分视角与时刻（App. B）。
- **语言**：T5-XXL 冻结 + 2 层 MLP 投影；用 language attention mask 屏蔽填充 token（App. B）。
- **模态平衡**：训练时各模态以 10% 概率独立掩码（App. B），防对某单一输入过依赖；交叉注意力**逐层交替**注入图像/文本 token（ACI）——图像 token 数远多于文本，若每层同时注入会淹没文本信息、损害指令跟随（§4.1 + Fig. 4b：Robot Dog 系任务的正确倒水量子项成功率从 62.5% 一度跌至 12.0%）。

### 4.5 物理可解释的统一动作空间（§4.2 + App. C）

设计两步（§4.2 原文归纳）：①动作通常是自己本体感知的子集（$\boldsymbol{a}_t$ 是 $\boldsymbol{z}_{t+1}$ 的目标值），故两者共用一个空间；②设计涵盖所有主要物理量（most of the main physical quantities）的统一空间，**把每个本体动作向量的各元素按物理含义填入对位、其余位置补零**。App. C 的 Table 4 给出 128 维定义：右臂关节位置 $[0,10)$、右夹爪关节位置 $[10,15)$、右臂/夹爪速度 $[15,25)/[25,30)$、末端位置 $[30,33)$、末端 6D 位姿 $[33,39)$、末端速度/角速度 $[39,42)/[42,45)$，$[45,50)$ 保留；左臂对称占用 $[50,100)$；底盘线/角速度 $[100,103)$；其余保留。单臂机器人映射到"右臂"，6 自由度臂填对应 10 个位置的前 6 个（Table 4 说明）。落地细节：①填充位与真实 0 语义混淆（"速度为 0"≠"没这只手"）→ 每维拼接 0/1 可用性指示向量，输入升为 256 维（App. F）；②各数据集单位统一（m、rad 等）而非逐数据集归一化——"1 m" 的物理含义跨本体一致，破坏它就损害迁移（App. D）；③末端旋转用 6D 表示避免万向锁（App. D）；④控制频率 $c$ 作为输入喂给模型，让模型感知本体间的频率差异（App. D）。

> **规格偏离说明**：任务规格所称"物理可解释性三设计原则（统一动作空间/小动作块/高频控制）"在论文中并无此"三原则/三定律"表述——论文实际结构为"机器人数据三特性 → 三项架构改造（MLP 解码器/QKNorm+RMSNorm/动作块）"加"统一动作空间"独立一节；动作块 $T_a=64$ 的设定与控制频率 $c$ 的输入化按上文实况撰写。

### 4.6 训练与推理配置（App. B/F/G/H，无编号公式，数字均按附录核对）

- **模型规模（Table 9）**：28 层、hidden size 2048、32 头、共 1.2B 参数（"1B" 之名取整）。各模态 token 空间：语言 4096、图像 1152、RDT 主干 2048，均以带 GELU 的适配器（adaptor）对齐（App. H）。
- **编码器冻结策略（Table 8）**：语言 T5-XXL 冻结 + 2 层 MLP 适配器；图像 SigLIP 冻结 + 2 层 MLP 适配器；动作侧 3 层 MLP（无冻结）。
- **优化（Table 10）**：AdamW，常数学习率 $1\times10^{-4}$（500 步 warm-up），batch size $32\times48$，bf16 混合精度，权重衰减 $10^{-2}$；48 张 H100 80GB 预训练 100 万步（约一个月），微调 13 万步（约三天），微调从 50 万步检查点启动（App. H）。
- **训练期监测（App. F）**：周期性做一次扩散采样，与训练集真值算 MSE，发现该 MSE 与真机部署表现普遍正相关——可作为训练进度的代理指标；过低的 MSE 反而提示过拟合。
- **微调技巧（App. E/F）**：图像增广（颜色抖动）+ 本体感知加信噪比 40 dB 的高斯噪声防过拟合；GPT-4-Turbo 对每条人工指令扩写 100 条变体 + 1 条简化版，训练时按 1/3 概率三选一；不使用 Classifier-Free Guidance——实测不涨点且带来机械臂行为不稳定（App. F 原文）。
- **推理（App. H）**：训练 1000 步 DDPM 调度 → 采样换 DPMSolver++ 5 步；动作块频率 6 Hz（每秒 6 个块）、动作平均推理频率 38 Hz（RTX 4090 24GB，App. B 训练细节节）；每步把上一块的预测作为后续的先验自然衔接。

## 5. 实验与结果解读

- **协议（§5.1）**：ALOHA 双臂真机，7 项任务对应 5 个泛化维度（Table 1）：Wash Cup（未见物体）、Pour Water（未见场景）、Pour Water-L-1/3 与 R-2/3（指令跟随："左手倒到三分之一满"）、Handover（5-shot）、Fold Shorts（1-shot）、Robot Dog（精细操作：推摇杆让机器狗走直线）。指标为成功率；试验数：Wash Cup 24（3 杯 × 8）、Pour Water 24（3 房间 × 8）、L-1/3 与 R-2/3 各 8、Handover/Fold Shorts/Robot Dog 各 25。基线：ACT（VAE 建模动作分布）、OpenVLA（7B 离散化）、Octo（扩散，93M）。
- **微调数据按任务的演示数**（§5.1 Data，原文数字）：Wash Cup 见过杯子合计 133 条（未见杯 0 条）；Pour Water 见过房间合计 350 条（未见房间 0 条）；L-1/3 与 R-2/3 各水位 18/19/19 条；Handover 仅 5 条；Fold Shorts 仅 1 条；Robot Dog 68 条。
- **主结果（Table 3 各任务 Total 列）**：Wash Cup——RDT 50%，其余全部 0；Pour Water——RDT 100%，ACT 37.5%，OpenVLA/Octo 0；Fold Shorts——RDT 68%，RDT (scratch) 40%，Octo 4%，其余 0；Handover——RDT 40%，scratch 16%，其余 0；Robot Dog——RDT 48%，ACT/scratch 32%，Octo 0。指令跟随子项：Pour Water-L-1/3 的正确倒水量 RDT 100%（scratch 62.5%，OpenVLA/Octo 0）。论文引言口径：在一众挑战任务上较基线取得 **56% 的成功率相对提升**。
- **消融（Table 2；Wash Cup 未见杯 2 / Pour Water 未见房间 3 / Pour Water-L-1/3 正确倒水量）**：RDT (regress，去掉扩散) 12.5/50/12.5；RDT (small，166M) 37.5/62.5/25；RDT (scratch，不预训练) 0/25/62.5；RDT (ours) 50/62.5/100——扩散、大参数、大规模预训练三者缺一都大幅掉点，且不预训练对未见物体/场景几乎归零（预训练知识是泛化的来源）。
- **RoboTwin 2.0 仿真基准互证**（数字取自[RoboTwin 2.0 精读 §5.5](./RoboTwin2.0_arXiv2025.md)，与本论文无重叠任务，读法：RDT 作为"预训练 VLA"在标准双臂基准上的横向位置）：50 任务平均成功率 Easy/Hard——RDT **34.5%/13.7%**，π0 46.4%/16.3%，DP3 55.2%/5.0%，ACT 29.7%/1.7%，DP 28.0%/0.6%（教程[第 08 章 §8.6](../08_策略训练与部署.md) 的 Easy/Hard 协议；第 09 章方向 4 也引了 RDT 34.5%→13.7% 的抗跌证据）。同文实验三：RDT 用域随机化数据预训练再微调，8 任务成功率 18.8%→24.8%（相对 +31.9%），说明 RDT 骨架能有效吸收 DR 数据的分布宽度。
- **真机长程/精细任务定性**（§5.2 + Fig. 5/10）：倒水任务连续多阶段（拿起—倒—放回）可完整执行，未见房间与见过房间表现接近；从没见过 "one-third / two-thirds" 字样仍能按指令精确控制水量与左右手；Robot Dog 需要毫米级摇杆角度控制，扩散 + 强网络的表达力使动作精度达标（ACT 因摇杆与遥控器同为黑色而常认错物体）。

## 6. 局限与后续影响

- **论文自述与数据可读出的局限**：①目标是特定双臂机器人能力而非跨本体通用（§3 定位即如此），统一动作空间主要作为预训练的过渡桥；②Hard/未见条件下仍明显下滑——RoboTwin 2.0 基准上 Easy→Hard 掉 20.8 个百分点，真机上未见物体子项（如 Wash Cup 未见杯的 Place Back Cup 50%）仍低于见过的；③语言泛化有边界：能泛化到未见词（"one-third"）靠的是预训练语料的语言多样性，但论文未测长指令/多语言；④1B 规模的采样开销仍在——动作维度低故开销"minor"（§4.1），但每秒 6 个动作块的推理节奏（38 Hz 动作频率，RTX 4090）依赖 DPMSolver++ 加速。
- **微调数据的规模天花板**（App. E 的三个质量因素反读）：数量上 6K+ 轨迹已是"最大的开源多任务双臂数据集之一"，但相对单臂数据仍差两个数量级；多样性上覆盖 100+ 物体、15+ 场景与多样光照，可任务形态仍以桌面操作为主（含借用开源 Songling 数据集的 3 个任务、140 条 episode——App. E）；指令多样性靠 GPT-4-Turbo 扩写弥补人工标注的匮乏。
- **后续影响**：①确立"扩散 Transformer + 统一动作空间"作为双臂基础模型路线，被 RoboTwin 双臂挑战赛（CVPR 2025 MEIS Workshop）与 RoboTwin 2.0 基准选为默认 VLA 基线之一；②与 π0 的流匹配路线形成扩散/流两大生成式策略范式的正面对照（见 §7）；③其"物理可解释对齐"思想被后续跨本体动作表示研究沿用（教程第 09 章方向 5）。

> **规格偏离说明**：任务规格称微调数据为 "RoboTwin-2K 数据"——论文微调集是作者**自采的 Mobile ALOHA 双臂数据集**（300+ 任务 / 6K+ 轨迹 / 300 万+ 帧），全文未出现 "RoboTwin-2K" 名称；RoboTwin 与 RDT 的交集在 RoboTwin 2.0 论文把 RDT 作为评测基线与真机骨干（见 §5），本精读按论文实况撰写。

## 7. 与本项目对照

- **扩散路线三代表对比**（生成式策略谱系，教程[第 08 章 §8.2](../08_策略训练与部署.md) 表格的展开）：

| 维度 | Diffusion Policy（RSS 2023） | RDT-1B（ICLR 2025） | π0（arXiv 2024） |
|---|---|---|---|
| 骨干 | 1D 时序 CNN（Transformer 为变体） | DiT：28 层 / hidden 2048 / 32 头 / 1.2B（Table 9） | VLM 主干 + 动作专家（流匹配头） |
| 生成式 | DDPM/DDIM 去噪，预测噪声 $\epsilon_\theta$ | DDPM 式 (1)(2)，**预测干净样本** $f_\theta$ | 条件流匹配（ODE 10 步欧拉积分，见[π0 精读 §4.1](./Pi0_arXiv2024.md)） |
| 条件注入 | FiLM/交叉注意力注入观测，无语言 | in-context（低维）+ 逐层交替交叉注意力（图文） | 分块因果注意力（图文先、动作后） |
| 动作空间 | 每任务定义 | **统一 128 维**跨本体对位填充 | 每本体定义、归一化 |
| 预训练 | 无（单任务从零训） | 46 数据集 / 100 万+ 轨迹 / 21TB | OXE + 多平台自采 |
| 双臂/语言 | 单臂为主、无语言 | 双臂 + 语言跟随（ACI） | 双臂 + 语言跟随 |

- **对 conv-based DP 的架构论证衔接**：Diffusion Policy 论文自证时序卷积偏好低频信号、对高频急变动作表现差（[Diffusion Policy 精读 §4.5](./DiffusionPolicy_RSS2023.md)）；RDT 把该问题升级为"机器人数据三特性"，用 Transformer 的表现力 + 稳定化改造给出规模化答案。
- **RoboTwin 场景的适配（实操）**：教程[第 08 章 §8.3](../08_策略训练与部署.md) 记录的 RoboTwin 官方配置——RDT 预训练 10 万步（8 卡 × batch 16）、单任务微调 1 万步（4 卡）；评测走 §8.6 的 Easy/Hard 各 100 次 rollout。选型经验（§8.8）：小数据任务优先预训练 VLA（RDT/Pi0），RDT 在 Hard 域偏移下显著抗跌于非预训练的 DP/ACT。
- **与仓库其他精读的关系**：RoboTwin 2.0 真机实验（sim-to-real）以 RDT 为策略骨干（[RoboTwin 2.0 精读 §5.4](./RoboTwin2.0_arXiv2025.md)）；RDT 的动作块/扩散目标可直接对照 ACT 的 CVAE 块预测（[ACT 精读](./ACT_ALOHA_RSS2023.md)）与 π0 的流匹配（互为"扩散 vs 流"教案）。

## 配套阅读

- 本目录：[Diffusion Policy 精读](./DiffusionPolicy_RSS2023.md)（扩散策略开山，式 (1)–(6) 对照）｜[π0 精读](./Pi0_arXiv2024.md)（流匹配对照路线）｜[RoboTwin 2.0 精读](./RoboTwin2.0_arXiv2025.md)（RDT 的基准成绩出处）｜[ACT 精读](./ACT_ALOHA_RSS2023.md)（同为动作块的 VAE 路线）
- 教程：[第 08 章｜策略训练与部署](../08_策略训练与部署.md)（RDT 训练/评测实操）｜[第 09 章｜进阶研究方向](../09_进阶研究方向.md)（VLA 清单与开放问题）
- 外部：项目页（论文 "project page" 链接）｜Ho et al. 2020《Denoising Diffusion Probabilistic Models》（式 (1)(2) 的背景）｜Peebles & Xie 2023《Scalable Diffusion Models with Transformers》（DiT 骨干）

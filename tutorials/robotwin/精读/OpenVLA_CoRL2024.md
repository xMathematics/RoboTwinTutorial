# 论文精读｜OpenVLA（CoRL 2024）

> **PDF**：[../../../papers/robotwin/frontier/arXiv-2406.09246_OpenVLA.pdf](../../../papers/robotwin/frontier/arXiv-2406.09246_OpenVLA.pdf) ｜ **教程**：[第 08 章](../08_策略训练与部署.md)、[第 09 章](../09_进阶研究方向.md) ｜ **代码**：openvla.github.io（PyTorch 代码库 + HuggingFace 权重 + 微调 notebook 全开源）
>
> 勘误提示（规格与论文不符处，均按 PDF 原文写）：
>
> - 论文只给出 **970k episodes**（Open X-Embodiment 子集），**未给出训练任务总数**——网传"1700 任务"不出自本 PDF（全文检索无此数）。
> - **speculative decoding 在本文中未被使用**：§3.5 明确 6 Hz 成绩"without compilation, speculative decoding, or other inference speed-up tricks"；§6 仅把 speculative decoding 与 action chunking 列为**未来候选**。论文亦未做与 RT-2-X 的速度对比，速度优势只体现在参数量（7B vs 55B）与自身吞吐数字上。
> - 评测覆盖的是 **4 个真机评测设置、3 种机械臂型号**（WidowX、Google Robot 移动平台、Franka-Tabletop、Franka-DROID——后两者同为 Franka Emika Panda，工作站不同）。
> - 本文与姊妹篇不同：**全文没有编号公式**（全文检索核实），第 4 节公式为本精读按论文文字整理的自编式（标号 OV1–OV3），每步注明依据，round/clip 等实现细节属整理而非原文。

## 1. 论文信息与一句话贡献

- **题目**：OpenVLA: An Open-Source Vision-Language-Action Model
- **作者/机构**：Moo Jin Kim、Karl Pertsch（共同一作），Siddharth Karamcheti、Ted Xiao、Suraj Nair、Rafael Rafailov、Chelsea Finn、Sergey Levine、Percy Liang、Dorsa Sadigh 等（Stanford / UC Berkeley / Toyota Research Institute / Google DeepMind / Physical Intelligence / MIT）
- **发表**：CoRL 2024（arXiv:2406.09246；本地 PDF 为 2024-09-05 的 v3，全文以该 PDF 为准）
- **一句话贡献**：训练并**完全开源**（数据 / 权重 / 代码三件套，Fig. 1）一个 **7B 参数** VLA——以 Prismatic VLM（DINOv2 + SigLIP 双视觉编码器 + Llama 2 7B）为骨干，把机器人动作离散成 token、用标准 next-token 交叉熵在 **970k episodes**（Open X-Embodiment 子集）上微调，以**少 7 倍的参数量**（7B vs 55B，论文口径 "7x fewer parameters"）在 29 个真机评测任务上取得对闭源 RT-2-X **16.5% 的绝对成功率优势**（§1），并系统给出 LoRA 微调与量化推理的低成本适配路线（单卡 A100 微调、7 GB 显存推理）。

## 2. 问题与动机

- **VLA 采纳的两大障碍**（§1 原文归纳）：
  - 现有 VLA 是**封闭的**——模型架构、训练数据、训练程序不可见（RT-2 即此类），社区无法检查、复用或改进；
  - 现有工作**不研究如何把 VLA 高效微调/部署**到新机器人、新环境、新任务——而"可微调"恰是开源语言模型生态繁荣的关键，机器人领域缺同款基础设施。
- **为什么复用视觉-语言基础模型**（§1）：
  - 机器人领域最大的操作数据集也只有 100k–1M episodes 量级，难以复现互联网级预训练；
  - 把 CLIP/SigLIP/Llama 2 这类基础模型当作策略的**核心构件**（core building block），有望把互联网规模预训练的泛化能力带进操作任务。
- **开源三要素的承诺**（§1、Fig. 1）：
  - Data：整理后可复现 OpenVLA 的 OpenX 训练子集；
  - Weights：HuggingFace checkpoint；
  - Code：PyTorch 训练/微调库，内置 LoRA 微调与量化推理。
  - 注：论文正文未提及 RLDS 一词（该数据格式出自官方代码库/数据卡，属代码库事实）。
- **设计决策的实验学**（§3.4，本文作为"开源基准"的方法论特色）：正式训练前在 BridgeData V2 上做小规模对照实验选型——VLM 骨干（Prismatic vs IDEFICS-1 vs LLaVA：LLaVA 比 IDEFICS-1 多物体语言接地高 35% 绝对值，Prismatic 又高约 10%）、分辨率、视觉编码器是否微调、训练轮数、学习率。本精读注：这套"先小扫后大训"的流程本身是其可复现性贡献的一部分。

## 3. 方法总览

OpenVLA = **Prismatic-7B VLM 骨干 + 动作 token 化 + next-token 微调**（§3，Fig. 2）。五个组件逐条：

1. **视觉编码器（vision encoder）**：Prismatic 的 *two-part* 设计——同一输入图像的 patch 分别过 **DINOv2**（低层空间细节）与 **SigLIP**（高层语义）两个预训练编码器，两组特征**逐通道拼接（channel-wise concat）**（§3.1）；融合动机：DINOv2 特征对机器人控制的空间推理有实证帮助（§3.1 及附录 D.2 消融：SigLIP-only 的更简版本在微调任务上仍可用，但融合版更优）。
2. **MLP projector**：小 2 层 MLP，把拼接后的视觉特征投影进语言嵌入空间（§3.1，Fig. 2 组件 2）。
3. **LLM 骨干**：**Llama 2 7B**。输入序列 = 视觉 token + 语言指令（模板 "What should the robot do? [task]"），输出侧自回归生成 **7 维末端执行器动作 token**：Δx, Δy, Δz, Δroll, Δpitch, Δyaw, 夹爪（Fig. 2 右侧 action de-tokenizer 还原为连续量）。
4. **动作表示**：每维动作独立离散成 **256 bins**，覆写 Llama 词表中**使用最少的 256 个 token（恰为词表末尾 256 个）**（§3.2，推导见 §4.1）。
5. **训练与推理配方**（§3.2–3.5）：
   - 损失：只对**动作 token 位**算交叉熵（推导见 §4.2）；
   - 数据：取 OpenX 中"至少有第三人称相机 + 单臂末端控制"的操作子集，按 Octo 配比加权（§3.3）；
   - DROID 的去留：以 10% 保守权重试混后，动作 token 准确率持续偏低，恐拖累最终模型，故从最后 1/3 训练中移除（§3.3）；
   - **微调视觉编码器**对好性能至关重要（§3.4；对应消融见 Table 1 的 frozen vision 行）；
   - 训练轮数：**27 epochs**，直至训练集动作 token 准确率超 95%——LLM 常规只训 1–2 epochs，此处刻意多轮（§3.4）；
   - 优化器设置：学习率固定 **2e-5**、无 warmup（§3.4）；
   - 分辨率：224×224（384×384 无成功率差异但训练慢 3 倍，§3.4）；
   - 基础设施：**64×A100、14 天、21,500 A100-hours、batch 2048**（§3.5）；
   - 推理：bfloat16 占 **15 GB** 显存，单张 RTX 4090 约 **6 Hz**（不用量化/编译/投机解码等任何技巧，§3.5）；
   - 部署形态：附远程推理服务器，允许无本地算力的机器人实时取动作流（§3.5、§4 代码库节）。

## 4. 关键公式推导（核心章节）

> **编号说明**：论文原文无编号公式。下式 OV1–OV3 为本精读自编编号；每步"依据"给到论文小节或外部文献。4.3 节为无公式的流水线串联小节。

### 4.1 动作离散化：256 bins × 7 DoF（与 RT-2 的异同）

设第 $d$ 维连续动作 $a_d$，训练集该维动作分布的 1% 与 99% 分位数为 $P^{1}_d, P^{99}_d$。离散到 bin 索引：

$$
b_d = \mathrm{clip}\!\left(\mathrm{round}\!\left(255 \cdot \frac{a_d - P^{1}_d}{P^{99}_d - P^{1}_d}\right),\ 0,\ 255\right)
\tag{OV1}
$$

逐依据拆解：

- **为什么逐维独立离散**：§3.2 "we discretize each dimension of the robot actions separately into one of 256 bins"——7 维各成一条 256 类的分类问题，输出即 7 个 token。
- **为什么用分位数边界**：§3.2 "we set the bin width to uniformly divide the interval between the 1st and 99th quantile of the actions in the training data"——以 $[P^{1}_d, P^{99}_d]$ 为离散区间（而非 min–max）；依据原文：分位数可丢弃离群动作，避免离散区间被撑大、有效粒度下降。该做法沿承 Brohan et al.（RT-2）。
- **OV1 各步的来源**：分母 $P^{99}_d - P^{1}_d$ 是区间长（等宽划分 → 线性位置）；乘 255 把 $[0,1]$ 映射到 $\{0,\dots,255\}$；round 取最近 bin；clip 处理落到区间外的端点。
  - **诚实标注**：round/clip 的具体写法是本精读对原文文字的公式化整理；论文原文只有文字描述"uniformly divide … into 256 bins"（与 [RT-1 精读 §4.1](./RT1_arXiv2022.md) 的情形相同：文字有、式子无）。
- **token 从哪来**：Llama tokenizer 只有 **100 个**预留新 token（§3.2 "reserved tokens"），不够 256 个 bin 使用；处理方式是"**覆写词表中使用最少的 256 个 token（恰为词表末尾 256 个）**"（§3.2 原文，following Brohan et al.）。
- **与 RT-2 的异同**（对照 [RT-2 精读](./RT2_arXiv2023.md) 与 [RT-2 论文 PDF](../../../papers/robotwin/frontier/arXiv-2307.15818_RT2.pdf)）：
  - 同：逐维 256 bins；分位数 bin 边界；覆写词表尾部 token；自回归预测离散动作 token；
  - 异：① OpenVLA 建立在**完全开源**的 Prismatic/Llama 2 上并开源数据、权重、代码；RT-2 基于 PaLI-X / PaLM-E 闭源模型、权重数据不公开；② OpenVLA 把"token 化微调"本身作为被研究对象（§3.4 消融视觉编码器微调、分辨率、epochs、学习率），RT-2 只报告最终策略；③ 骨干规模 7B vs 55B。

### 4.2 训练目标：仅动作 token 的交叉熵

记动作 token 位经覆写后的词表索引为 $t_i$，观测与指令 token 只作条件：

$$
\mathcal{L}(\theta) = - \sum_{i \in \text{动作位}} \log p_\theta\!\left(t_i \mid t_{<i},\ \text{图像},\ \text{指令}\right)
\tag{OV2}
$$

- **依据**：§3.2 原文 "trained with a standard next-token prediction objective, evaluating the cross-entropy loss on the predicted action tokens only"——**求和只覆盖动作位**（观察/指令 token 不计损失），即"把动作预测当视觉问答做"的损失形式化；OV2 是该文字的标准自回归交叉熵写法。
- **配套超参**（§3.4–3.5）：27 epochs、batch 2048、lr 2e-5（固定、无 warmup）、bfloat16 混合精度 + FlashAttention + FSDP（§4 代码库节）。

### 4.3 一条指令的完整 token 流水线（把 4.1/4.2 串起来）

以 Fig. 2 的例子 "Put eggplant in bowl" 为例，一次前向的序列构成：

| 段 | 内容 | token 数 | 来源 |
|---|---|---|---|
| 视觉段 | 224×224 图像 → DINOv2 ⊕ SigLIP 逐通道拼接 → MLP projector | 数百级 patch token（论文未给单值） | §3.1、Fig. 2 |
| 指令段 | "What should the robot do? Put eggplant in bowl" 经 Llama tokenizer | 文本长度可变 | §3.2、Fig. 2 |
| 动作段 | 7 维末端动作 → (OV1) 各维 256-bin → 覆写词表末尾 256 token | **恰 7 个** | §3.2、Fig. 2 |

- 训练时损失只落在动作段 7 个位置上（依据：§4.2 的 OV2）；推理时自回归生成这 7 个 token，经 action de-tokenizer 按 (OV1) 的逆映射还原为连续 Δx/Δθ/ΔGrip（Fig. 2 右侧）。

### 4.4 LoRA 参数高效微调：$\Delta W = BA$

LoRA（Low-Rank Adaptation，Hu et al. 2021，论文引文 [26]）把权重更新约束在低秩子空间：

$$
W = W_0 + \Delta W = W_0 + BA,\qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k)
\tag{OV3}
$$

- **前向形式**：$h = W_0 x + BAx$；训练时**冻结 $W_0$**，只学 $A, B$（依据：Hu et al. 2021 的低秩假设——任务适配所需的更新矩阵本就近似低秩）。
- **参数量账**：全量更新学 $d\times k$ 个数，低秩分解只学 $r(d+k)$ 个（依据：矩阵尺寸相加）。OpenVLA 对**所有线性层**施用、默认 $r=32$（§5.3 原文）。
- **论文数字核对**（Table 1：Franka-Tabletop 多任务、33 rollouts/法、batch 16、两卡 FSDP 分片）：

  | 策略 | 成功率 | 可训练参数（×10⁶） | 显存（batch 16） |
  |---|---|---|---|
  | Full FT（全微调） | 69.7±7.2% | 7,188.1 | 163.3 GB |
  | Last layer only | 30.3±6.1% | 465.1 | 51.4 GB |
  | Frozen vision | 47.0±6.9% | 6,760.4 | 156.2 GB |
  | Sandwich（冻 LLM 开视觉） | 62.1±7.9% | 914.2 | 64.0 GB |
  | **LoRA r=32** | **68.2±7.5%** | **97.6** | **59.7 GB** |
  | LoRA r=64 | 68.2±7.8% | 195.3 | 60.5 GB |
- **三条读数**：① 97.6 / 7,188.1 ≈ **1.4%** 参数达到与全微调统计相当的成功率（§5.3 原文口径）；② r=32 与 r=64 成功率完全相同 → **秩不敏感**，推荐默认 $r=32$；③ 冻结视觉编码器只剩 47.0% → **把预训练视觉特征适配到目标场景不可省**（与 §3.4 全量微调时的结论一致）。
- **算力收益**：LoRA 在**单张 A100 上 10–15 小时**完成新任务微调，比全微调节省 **8 倍**算力（§5.3 原文）。

### 4.5 量化推理（int8 / int4）

后训练量化把权重载入低精度以省显存（§5.4，Table 2：8 个 BridgeData V2 任务、80 rollouts/法）：

| 精度 | 成功率 | 显存 |
|---|---|---|
| bfloat16（默认） | 71.3±4.8% | 16.8 GB |
| int8 | 58.1±5.1% | 10.2 GB |
| **int4** | **71.9±4.7%** | **7.0 GB** |

- **反直觉读数**：int8 掉点而 int4 不掉。论文归因于**闭环频率而非精度**——int8 的量化算子开销拖慢推理，A5000 上只剩 **1.2 Hz**，与采集时 5 Hz 控制器的系统动力学显著失配；脚注 5 给出证据：int8/int4 的**离线 token 准确率与 bf16 相当**。int4 因省下的显存传输补偿量化开销反而更快，A5000 上 **3 Hz**，更接近采集动力学。
- 本条与 [Diffusion Policy 精读 §6](./DiffusionPolicy_RSS2023.md) 的"推理开销改变闭环动力学"是同一物理在两个论文中的体现。

## 5. 实验与结果解读

### 5.1 开箱即用 · BridgeData V2 / WidowX（§5.1，Fig. 3）

- 评测规模：**17 任务 × 10 次 = 170 rollouts**，覆盖 visual（未见背景/干扰物/颜色）、motion（未见位姿朝向）、physical（未见尺寸形状）、semantic（未见物体/指令/概念）、language（多物体语言接地）五条泛化轴。
- 平均成功率（Fig. 3 原图数字）：OpenVLA **70.9%** > RT-2-X **55.9%** > Octo 32.0% > RT-1-X 16.8%。
- 结论：除 semantic 轴外，OpenVLA 全面超过 RT-2-X（RT-2-X 语义强因其互联网数据与动作数据**共同微调**，OpenVLA 只在机器人数据上微调——论文对此差异的归因，§5.1）。

### 5.2 开箱即用 · Google Robot 移动操作平台（§5.1，Fig. 4）

- 评测规模：**12 任务 × 5 次 = 60 rollouts**；OpenVLA **76.2%** vs RT-2-X **72.6%**（Fig. 4 原图数字）。
- 汇总口径：两环境合计 **29 任务**上对 RT-2-X **+16.5% 绝对成功率、参数少约 7 倍**（§1/摘要口径："7x fewer parameters"）。
- 优势归因（§5.1）：数据量 970k vs RT-2-X 的 350k 轨迹；数据清洗更细（如过滤 Bridge 子集的全零动作，附录 C）；DINOv2+SigLIP 融合编码器的语义+空间双重特征。
- **可复现性陷阱**（附录 C，本精读认为最具启发性的一段）：对训练数据过滤"首帧转移"后 OpenVLA 无需技巧即可正常出招；RT-2-X 无法重训，只能沿用 OpenX 项目"总是取次优 token"的 workaround 规避其"冻结在原地"行为——**开源 VLA 可被诊断，闭源 VLA 只能打补丁**，这是开源路线的隐含论据。

### 5.3 微调到新本体 · Franka 双设置（§5.2，Fig. 5）

- 设置：Franka-Tabletop（5 Hz 控制器）与 Franka-DROID（15 Hz、可移动站桌）各 10–150 条演示，共 **7 任务**（从抓放到擦桌子），129 rollouts（99 + 30）。
- 对比对象：从零训练的 Diffusion Policy（及其输入匹配版 DP (matched)）、Octo 微调、OpenVLA (scratch)（不经 OpenX 预训练、直接微调 Prismatic）。
- 结论（§5.2 原文归纳）：
  - 微调后 OpenVLA **总平均最高**，且是唯一**在所有任务上 ≥50%** 的方法；
  - 对从零的 Diffusion Policy 领先 **20.4%**（§1 口径）；
  - 分工现象：DP 在窄单指令任务（Put Carrot in Bowl 等）更平滑精细；OpenVLA 在多物体、多指令、语言接地任务更强；
  - OpenVLA (scratch) 显著更差 → **大规模机器人数据预训练的收益**被单独消融证实；
  - 论文自承：DP 更平滑精确，给 OpenVLA 引入 action chunking 与 temporal smoothing 是可能补救方向。

### 5.4 效率实验小结

- 训练成本：64×A100 × 14 天 = 21,500 A100-hours（一次性，§3.5）；
- 适配成本：LoRA 单卡 10–15 小时（每次新任务，§5.3）；
- 推理成本：bfloat16 15 GB / ≈6 Hz（RTX 4090）；int4 7.0 GB / 3 Hz（A5000）——**消费级 GPU 可服务**（§5.4，Fig. 6 另给出 2080Ti/3080/A100/H100 的吞吐曲线，Ada Lovelace 架构卡受益最大）；
- **速度对照的正确表述**：论文未直接测 RT-2-X 的推理速度；可核实的对照是——OpenVLA 以少 7 倍的参数（论文口径）达到更高成功率，且 6 Hz（bf16/4090）与 3 Hz（int4/A5000）均为论文实测数字；speculative decoding 仅作为 §6 的未来候选，未参与任何实验。

## 6. 局限与后续影响

**作者自述（§6 Discussion and Limitations）**：

1. 只支持**单图像观测**：无本体感知（proprioception）输入、无观测历史，而真实机器人系统通常都有（扩展多图/本体输入是明确 future work）；
2. **推理吞吐**限制高频控制：ALOHA 类平台需要 50 Hz，OpenVLA 单步生成约 6 Hz——论文点名 **action chunking 与 speculative decoding 是候选补救（本文未实现）**；
3. 可靠性不足：多数任务 **<90%** 成功率；
4. 设计空间未探索：骨干规模、机器人数据与互联网数据共训、视觉特征选择等，因算力限制留白。

**后续影响**：

- 确立"开源 VLM + 动作 token + 数据/权重/代码全开源"的复现范式，成为大量下游工作的默认微调起点；
- 后续工作 OpenVLA-OFT 等把动作分块、并行解码与连续动作头引入该骨干，正面回应 §6 的吞吐与平滑性局限（论文外信息，供延伸）。

## 7. 与本项目对照

- **教程映射**：[第 08 章 §8.2](../08_策略训练与部署.md) 策略表未单列 OpenVLA（"预训练 VLA"位由 RDT/Pi0 占据），但 §8.3 训练细节、§8.4"预训练→微调"流程、§8.6 评测协议完全适用于它；[第 09 章 §9.2](../09_进阶研究方向.md) 的 VLA 清单收录 OpenVLA（"开源 VLA 基准"）；其微调属监督模仿学习，不涉及 [控制规划教程第 08 章 RL 式 (8.1)](../../control_planning/08_学习式控制强化学习.md) 的 RL 目标。
- **RoboTwin 场景微调 OpenVLA 的可行性**：
  - [RoboTwin 2.0 精读 §5.5](./RoboTwin2.0_arXiv2025.md) 的五策略基准不含 OpenVLA，但其"发布预训练权重 + 每任务 50 条演示微调"协议与 OpenVLA 的 LoRA 配方（单卡 10–15 h、59.7 GB、r=32）成本兼容；
  - RoboTwin 2.0 评测节奏与本体验证的 OpenVLA 低频单步控制（无分块）匹配良好；
  - 微调数据量可参照其 Franka 实验（10–150 条演示区间）起步，LoRA + 微调视觉编码器并保留（勿冻视觉，Table 1 反例）。
- **与 π0 流匹配路线的对照**：OpenVLA 的"离散 token + 单次前向"无动作分块、难以建模连续多模态动作分布——这正是 [π0 精读](./Pi0_arXiv2024.md) 选择"VLM + 动作专家 + 流匹配动作块"的对照面（π0 论文开箱实验中 OpenVLA 在高频分块任务上明显落后，详见 π0 精读 §5）。适用边界：低频、语言接地强需求 → OpenVLA 式；高频、灵巧连续控制 → π0 式。
- **数据视角**：OpenVLA 用 970k episodes（vs RT-2-X 350k）的预训练子集证明 OpenX 规模收益，训练混采口径见 [Open X-Embodiment 精读](./OpenXEmbodiment_ICRA2024.md)；呼应 [RoboTwin 2.0 精读 §5.6](./RoboTwin2.0_arXiv2025.md) 的"规模 × 分布宽度"双 scaling 轴。

## 配套阅读

- 教程内：[第 08 章 8.2/8.3/8.4/8.6](../08_策略训练与部署.md)、[第 09 章 9.2 阅读清单](../09_进阶研究方向.md)、[控制规划教程第 08 章 RL 式 (8.1)](../../control_planning/08_学习式控制强化学习.md)
- 姊妹精读：[RT-2（256-bin token 化直接来源）](./RT2_arXiv2023.md)、[RT-1（离散动作 token 谱系源头）](./RT1_arXiv2022.md)、[Open X-Embodiment（预训练数据池）](./OpenXEmbodiment_ICRA2024.md)、[π0 / 流匹配 VLA（对照路线）](./Pi0_arXiv2024.md)、[Diffusion Policy（连续动作路线）](./DiffusionPolicy_RSS2023.md)、[RoboTwin 2.0（微调协议所在基准）](./RoboTwin2.0_arXiv2025.md)
- 论文原文：[RT-2 论文 PDF](../../../papers/robotwin/frontier/arXiv-2307.15818_RT2.pdf)、Open X-Embodiment（arXiv:2310.08864）、Octo（arXiv:2405.12213）、DROID（arXiv:2403.12945）
- 教程外：Hu et al. 2021（LoRA，(OV3) 出处）、Karamcheti et al. 2024（Prismatic VLM）、Dettmers et al. 2023（QLoRA，量化微调背景）
- 项目页：openvla.github.io（权重、微调 notebook、远程推理服务器）

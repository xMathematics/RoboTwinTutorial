# 论文精读｜Diffusion Policy（RSS 2023）

> **PDF**：[../../../papers/robotwin/classics/arXiv-2303.04137_DiffusionPolicy.pdf](../../../papers/robotwin/classics/arXiv-2303.04137_DiffusionPolicy.pdf) ｜ **教程**：[第 08 章](../08_策略训练与部署.md)（策略训练） ｜ **代码**：官方代码与训练配置全开源（GitHub real-stanford/diffusion-policy 仓库，项目页 diffusion-policy.cs.columbia.edu）
>
> 注：本地 PDF 为 arXiv v5 扩展版（RSS 2023 会议版的扩展，含 §4.5 控制理论联系、§5.4 视觉编码器消融、§7 扩展真机实验）。式号 (1)–(6) 均按该 PDF 原文核对。

## 1. 论文信息与一句话贡献

- **题目**：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **作者/机构**：Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song（Columbia / MIT-PU / Toyota Research Institute / Harvard / Boston Dynamics AI Institute）
- **发表**：RSS 2023（本地 PDF 为 2024-03 的 arXiv v5 扩展版）
- **一句话贡献**：把机器人的视觉运动策略表示为**条件去噪扩散概率模型**（conditional denoising diffusion probabilistic model, DDPM）——在动作空间上从高斯噪声迭代去噪生成动作序列，借助随机 Langevin 动力学的采样机制天然表达**多峰动作分布**，在 4 个基准 15 个任务上平均超过此前 SOTA 模仿学习方法 **46.9%**（论文摘要数字）。

## 2. 问题与动机

模仿学习（[第 01 章](../01_具身智能入门.md) 1.3 路线二）的三个独特难点（论文 §1）：

1. **多峰分布**：同一个观测下，人类演示可以走多条等价路径（向左推或向右推都行）。显式策略（explicit policy，输出 $F_\theta(o)$ 直接回归）被迫在多个模式间取平均，执行出"哪条路都不是"的动作；
2. **序列相关与高精度**：动作维度高、精度要求高，多峰的采样（如 GMM 选峰）在时间维上不一致；
3. **训练稳定性**：隐式策略（implicit policy，如 Energy-Based Model 用 $\arg\min_a E_\theta(o,a)$ 求动作）依赖负采样估计难处理的归一化常数，训练不稳、对超参极敏感（IBC 的实践困难，论文 §4.4）。

**本文的回答**：用扩散模型统一三件事——(i) 从纯高斯初始化出发的**随机采样过程**天然携带多峰性；(ii) 在**动作序列**（而非单步动作）上扩散，保证时间一致性；(iii) 学习的是能量函数的**梯度场**而非能量值本身，绕开归一化常数，训练像普通回归一样稳定（论文 §1 四条性质小结）。

与 RL 路线的关系：本文是纯模仿学习（行为克隆目标），不需要回报信号；RL 目标函数 $\max_\theta \mathbb{E}[R(\tau)]$ 见[控制规划教程第 08 章](../../control_planning/08_学习式控制强化学习.md) 式 (8.1)——对比可理解：扩散策略解决的是"如何忠实复现专家分布"，RL 解决的是"如何超越专家"。

## 3. 方法总览

把 DDPM 的生成对象从图像换成**动作序列**、把生成过程**条件化到观测**上，即得到扩散策略。三个关键改造（论文 §1 贡献列表、§2.3）：

1. **闭环动作序列预测**：每个决策时刻取最新 $T_o$ 步观测，预测 $T_p$ 步动作，只执行其中 $T_a$ 步再重新预测（receding horizon，滚动时域）——长视野保平滑、短执行保反应（§4.1）；
2. **视觉条件化**：观测 $O_t$ 作为**条件**而非联合分布的一部分，视觉特征每次只编码一次、与去噪迭代解耦，使实时推理成为可能（§2.3 Visual observation conditioning）；
3. **时序扩散 Transformer**：以 minGPT 风格 Transformer 为去噪骨干的变体，缓解 CNN 时序卷积的低频偏置（§3.1）。

流水线：图像序列 → 视觉编码器（ResNet-18 变体，§4.6）→ 观测嵌入 $O_t$ → 以 $\epsilon_\theta(O_t, A_t^k, k)$ 为去噪网络的扩散过程 → 从 $A_t^K \sim \mathcal{N}(0, I)$ 经 $K$ 步去噪得 $A_t^0$ → 执行前 $T_a$ 步。

## 4. 关键公式推导

> 式号 (1)–(6) 与论文 PDF 逐字核对（§2.1–§4.4）。**注意**：论文使用 DDIM 风格的移参记号 $(\alpha, \gamma, \sigma)$（见 §2.1 末 "The choice of α, γ, σ as functions of iteration step k… can be interpreted as learning rate scheduling"），**没有**显式写出标准 DDPM 的前向闭式 $q(x^k \mid x^0)$；推导中引用它时明确标注为 Ho et al. (2020) 的标准背景结果。

### 符号表

| 符号 | 含义 | 备注（出处） |
|------|------|------|
| $A_t \in \mathbb{R}^{T_p \times d_a}$ | $t$ 时刻预测的动作序列（扩散变量 $x \equiv A_t$） | §2.3 |
| $O_t$ | 最近 $T_o$ 步观测的嵌入序列 | §2.3、Fig. 2 |
| $k \in \{1,\dots,K\}$ | 去噪迭代轮次（$K$ 为总轮数） | §2.1 |
| $A_t^k$ | 第 $k$ 轮噪声水平的动作序列（$k=K$ 为纯噪声） | §2.1 |
| $\epsilon_\theta(\cdot)$ | 噪声预测网络（策略本体） | 式 (1) |
| $\epsilon^k$ | 第 $k$ 轮加的 Gaussian 噪声 | §3.3 |
| $\alpha, \gamma, \sigma$ | 噪声调度（noise schedule）三函数，均随 $k$ 变化 | §3.3 |
| $T_o, T_p, T_a$ | 观测视野 / 动作预测视野 / 动作执行视野 | §2.3 |
| $E(x)$ | 能量函数（仅用于解读，$\nabla E$ 由 $\epsilon_\theta$ 隐式给出） | 式 (2) |

### 4.1 DDPM 基本式：反向去噪与训练目标

**背景（标准结果，Ho et al. 2020，非论文原文式）**：DDPM 前向过程逐维对数据加 Gaussian 噪声：

$$
q(x^k \mid x^{k-1}) = \mathcal{N}\big(x^k;\ \sqrt{1-\beta_k}\, x^{k-1},\ \beta_k I\big),\qquad
\bar\alpha_k = \prod_{j=1}^{k}(1-\beta_j)
\tag{D0}
$$

由 Gaussian 的可加性（依据：两个独立 Gaussian 之和仍为 Gaussian，方差相加；沿 (D0) 逐项代入合并，$k$ 步后方差线性累加），边缘分布有闭式：

$$
q(x^k \mid x^0) = \mathcal{N}\big(x^k;\ \sqrt{\bar\alpha_k}\, x^0,\ (1-\bar\alpha_k) I\big)
\tag{D0'}
$$

(D0′) 说明：任取一轮 $k$，都能**一步采样**出该噪声水平的样本 $x^k = \sqrt{\bar\alpha_k}\, x^0 + \sqrt{1-\bar\alpha_k}\,\epsilon^k$，其中 $\epsilon^k \sim \mathcal{N}(0,I)$——这正是训练时可以随机抽 $k$ 的原因（依据：训练随机抽轮次以覆盖所有噪声水平，见论文 §2.2 开头的流程描述）。标准 DDPM 的"简单目标"是把网络 $\epsilon_\theta$ 训练成预测所加噪声，其来源是对变分下界的化简：Ho et al. (2020) 证明，对 (D0) 定义的前向过程求 ELBO（evidence lower bound，证据下界），逐项分解后每项都是 Gaussian 之间的 KL 散度，把 KL 用均值差写出（依据：两个 Gaussian 的 KL 散度有闭式 $\frac{1}{2}\|\mu_1-\mu_2\|^2/\sigma^2 + \cdots$，均值差项占主导即得简单目标），并重参数化 $\mu_\theta = \frac{1}{\sqrt{1-\bar\alpha_k}}(x - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta)$ 后，负对数似然下界化简为 $\|\epsilon^k - \epsilon_\theta(x^k, k)\|^2$ 的期望（依据：Ho et al. 2020 定理 1 的化简链；论文 §2.2 也引此结论："minimizing the loss function in Eq 3 also minimizes the variational lower bound of the KL-divergence between the data distribution $p(x^0)$ and the distribution of samples drawn from the DDPM $q(x^0)$"）。

**论文式 (3)**（§2.2 DDPM Training，按 PDF 原文）：

$$
\mathscr{L} = \mathrm{MSE}\big(\epsilon^k,\ \epsilon_\theta(x^0 + \epsilon^k,\ k)\big)
\tag{3}
$$

与 (D0′) 的对应：论文把 $x^k$ 直接记作 $x^0 + \epsilon^k$——即 (D0′) 在论文的 $(\alpha,\gamma,\sigma)$ 移参记号下省略了 $\sqrt{\bar\alpha_k}$ 缩放系数（该缩放被吸收进噪声调度 $\alpha$ 与 $\gamma$ 的定义中；依据：论文 §2.1 末"Details about noise schedule will be discussed in Sec 3.3"，其中调度由 $\sigma,\alpha,\gamma$ 三函数联合定义）。训练流程（§2.2 原文）：从数据集抽干净的 $x^0$（此处即一条真值动作序列 $A_t^0$）→ 随机抽去噪轮次 $k$ → 按该轮方差抽噪声 $\epsilon^k$ → 最小化 (3)。

### 4.2 去噪即随机 Langevin 动力学：式 (1) 的逐项解读

**论文式 (1)**（§2.1，按 PDF 原文）：

$$
x^{k-1} = \alpha\big(x^k - \gamma\,\epsilon_\theta(x^k, k) + \mathcal{N}(0, \sigma^2 I)\big)
\tag{1}
$$

**论文式 (2)**（§2.1）：

$$
x' = x - \gamma \nabla E(x)
\tag{2}
$$

把 (1) 与 (2) 逐项对齐（依据：论文原句 "The above equation 1 may also be interpreted as a single noisy gradient descent step… the noise prediction network $\epsilon_\theta(x,k)$ effectively predicts the gradient field $\nabla E(x)$"）：

1. **确定性项** $x^k - \gamma\,\epsilon_\theta(x^k,k)$ 与 (2) 的梯度下降步 $x - \gamma\nabla E(x)$ 同构，对应关系为 $\epsilon_\theta \approx \nabla E$——网络学的是"把动作往哪个方向挪能降低能量/更接近数据分布"（依据：式 (2) 是无噪声纯梯度步的定义式，(1) 的对应项与其形式相同、$\gamma$ 的角色相同）；
2. **缩放因子 $\alpha$**：稍小于 1 的 $\alpha$ 相当于学习率调度（论文原文："An α slightly smaller than 1 has been shown to improve stability Ho et al. (2020)"），逐步衰减可防止去噪后期震荡；
3. **噪声项 $\mathcal{N}(0,\sigma^2 I)$**：每轮注入 Gaussian 噪声，使整个迭代成为**随机 Langevin 动力学**（Stochastic Langevin Dynamics，Welling & Teh 2011；论文 §2.1 首句即如此定性）。噪声的作用：让采样分布收敛到正确的平稳分布，同时赋予采样**随机性**——这是 §4.4 多峰性的第一来源。

调度 $(\alpha, \gamma, \sigma)$ 随 $k$ 的具体形状（论文 §3.3）：实验采用 iDDPM 的 Square Cosine Schedule；调度的形状决定扩散策略捕捉动作信号高/低频成分的程度。

### 4.3 条件化与滚动时域：式 (4)(5) 与 $T_o/T_p/T_a$

**观测条件化**（§2.3）：为逼近条件分布 $p(A_t \mid O_t)$（而非 Janner et al. 2022a 规划式的联合 $p(A_t, O_t)$），把 (1) 的 $\epsilon_\theta$ 参数中加进 $O_t$：

$$
A_t^{k-1} = \alpha\big(A_t^k - \gamma\,\epsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)\big)
\tag{4}
$$

训练损失相应地从 (3) 改为：

$$
\mathscr{L} = \mathrm{MSE}\big(\epsilon^k,\ \epsilon_\theta(O_t,\ A_t^0 + \epsilon^k,\ k)\big)
\tag{5}
$$

依据：论文原文 "To capture the conditional distribution $p(A_t|O_t)$, we modify Eq 1 to… The training loss is modified from Eq 3 to"。(4)(5) 与 (1)(3) 的唯一区别是条件项 $O_t$ 的加入——去噪网络在"看到观测"的前提下预测该减去的噪声方向。把 $O_t$ 排除在扩散变量之外（只扩散动作）有两重收益（§2.3 原文）：推理时视觉特征**只编码一次**（对 $K$ 轮迭代共享），大幅提速并使端到端训练视觉编码器可行。

**滚动时域执行**（§2.3 Closed-loop action-sequence prediction）：每个决策时刻 $t$：取最新 $T_o$ 步观测 $O_t$ → 预测 $T_p$ 步动作 → 只执行前 $T_a$ 步（$T_a \le T_p$）→ 下一时刻基于新观测重预测。设 $T_p$ 步开环执行与每 $T_a$ 步闭环重预测交替，观测信息进入控制的周期从 $T_p$ 缩短到 $T_a$——时间一致性与响应性由 $T_p/T_a$ 的比值显式调节（§4.3 消融：$T_a$ 过大趋近开环、过小丧失一致性与平滑性）。

**与 MPC 的同构性**（对照[控制规划教程第 05 章](../../control_planning/05_模型预测控制mpc.md) 式 (5.1)(5.2)）：MPC 每周期从实测 $x_t$ 解 OCP (5.1) 得未来序列、只执行首步 $u_0^*$（式 (5.2) 的策略 $\pi_{\mathrm{MPC}}(x) = u_0^*(x)$），时域前移循环；扩散策略每周期从当前 $O_t$ 扩散出 $T_p$ 步序列、只执行 $T_a$ 步、时域前移——**"重解 + 部分执行"的滚动结构完全同构**，差异只在"序列如何产生"：前者由模型 + 在线优化产生（有显式动力学与代价函数），后者由学习的去噪采样产生（无模型，分布即策略）。论文 §2.3 也直接援引 receding horizon control（Mayne & Michalska 1988）并指出可用上一轮预测**温启动**（warm-start）下一轮去噪以进一步平滑——与 MPC 教程 05.1 ③ 中"温启动省算力"同一条工程智慧。

### 4.4 多峰表达力：MSE 回归为何平均化、扩散采样为何保持多峰

这是本文最核心的论证（论文 §4.1 与 Fig. 3 的 Push-T 演示）。以下几何推导为标准论证的数学化（论文原文以直觉与实验呈现）。

**MSE 回归的模式平均化**。设给定观测 $o$ 下演示数据中动作以各 50% 出现于两个模式 $a = -1$ 与 $a = +1$。最小化均方误差 $\mathbb{E}_{(o,a)\sim\mathcal{D}}[(a - f_\theta(o))^2]$：对 $f$ 求导并令其为零（依据：凸二次函数极小的一阶条件）：

$$
\frac{\partial}{\partial f}\mathbb{E}\big[(a-f)^2\big] = -2\,\mathbb{E}\big[a - f\big] = 0
\;\Longrightarrow\;
f^*(o) = \mathbb{E}[a \mid o] = \tfrac{1}{2}(-1) + \tfrac{1}{2}(+1) = 0
\tag{D1}
$$

$f^* = 0$ 是**数据中从未出现过的动作**：机器人执行它就会卡在两峰之间的"沟壑"里（Push-T 场景中即把 T 块推歪）。(D1) 的机制与条件分布的形式无关——只要损失是二次的、输出是确定性的单点，条件期望就会把多峰平均成"既非此也非彼"的谷值；推广到序列输出 $A_t$ 与高斯混合/GMM 选峰方案，则还有时间维上峰选择不一致导致的抖动（论文 §1、§4.1）。

**扩散采样保持多峰的机制**（论文 §4.1 "multi-modality… arises from two sources"）：

1. **随机初始化**：每次 rollout 从 $A_t^K \sim \mathcal{N}(0,I)$ 起步（依据：式 (1) 的迭代从高斯噪声出发，§2.1），不同的初值落入 $\nabla E$ 梯度场的不同**收敛盆**（basin）；
2. **迭代中的随机扰动**：每轮注入 $\mathcal{N}(0,\sigma^2 I)$（依据：(1) 的噪声项），样本既能稳定收敛到盆底，也能在迭代早期跨越盆间势垒、在后期锁定单一模式。

因此**同一次 rollout 内**动作时间一致（锁定一个峰）、**跨 rollout 之间**模式覆盖数据分布（两峰都被采样到）——Fig. 3 的 Push-T 实验正是如此：扩散策略 40 步 rollout 要么整体向左推、要么整体向右推；LSTM-GMM 与 IBC 偏向单峰，BET 则因缺乏时间一致性在两峰间抖动。与 RT-1 的对照（[RT-1 精读](./RT1_arXiv2022.md) §4.1）：离散 256-bin 分类分布在**单维**上也能表达多峰，扩散方案则在**连续、多维、序列**输出上统一解决该问题。

**与 EBM 的关系**（论文 §4.4，按 PDF 原文）：

$$
p_\theta(a \mid o) = \frac{e^{-E_\theta(a,o)}}{Z(o,\theta)}
\tag{6}
$$

隐式策略（IBC）直接用 (6) 需要处理难算的归一化常数 $Z(o,\theta)$（依据：§4.4 原文 "an intractable normalization constant"）；扩散策略让 $\epsilon_\theta$ 学 $\nabla E$，$Z$ 在采样中被绕开（依据：Langevin 采样只需梯度场，(1) 中无 $Z$ 出现）——这是训练稳定性的来源。

### 4.5 两种去噪骨干：CNN（时序卷积）与 Transformer（时间 token）

（论文 §3.1；无编号公式，按原文定性 + 结构描述）

- **CNN-based**：采用 Janner et al. (2022b) 的 1D 时序卷积 + 三点改造：只建模条件分布 $p(A_t|O_t)$（FiLM 逐层逐通道注入 $O_t$，FiLM 形式同 [RT-1 精读](./RT1_arXiv2022.md) 式 (R2)）；只预测动作轨迹（不联合预测观测-动作序列）；去掉 inpainting 式目标条件（与滚动预测视野冲突）。归纳偏置：时序卷积偏好低频信号，对**高频、急变的动作**（如速度控制指令）表现差（论文引 Tancik et al. 2020 的频率分析）。
- **Time-series diffusion Transformer**：minGPT 风格 decoder。带噪动作 $A_t^k$ 作为输入 token，去噪轮次 $k$ 的正弦嵌入作为**前置 token**；观测 $O_t$ 经共享 MLP 变为嵌入，经每个 decoder 块的**交叉注意力**注入；每个输出 token 对应预测该位置动作的"梯度" $\epsilon_\theta(O_t, A_t^k, k)$；因果注意力掩码保证每步只看自己与前序动作。适合高频动作变化与速度控制空间，但对超参更敏感（论文 §3.1）。
- **官方建议**：新任务先试 CNN 版，不行再换 Transformer 版（§3.1 Recommendations）。

### 4.6 视觉编码器、观测空间与采样加速

- **编码器**（§3.2）：默认 **ResNet-18，不预训练**，两处修改——全局平均池化换 **spatial softmax 池化**（保空间信息）、BatchNorm 换 **GroupNorm**（与 DDPM 常用的指数移动平均 EMA 兼容，训练更稳）；不同机位用独立编码器，各时刻图像独立编码后拼接成 $O_t$，端到端与策略联合训练。注：论文正文只有 ResNet-18；扩展版 §5.4 消融了 ResNet-34（ImageNet-21k）与 ViT-B/16（CLIP 预训练），结论是"冻结预训练编码器很差、以 10 倍小学习率微调预训练编码器最佳（CLIP ViT 微调在 square(ph) 上达 98%）"。"ResNet/DINO" 等多编码器选项是官方代码库提供的，非本文正文内容。
- **观测空间**：图像（多机位）为默认；状态实验中另有拼接的关节信息；6D 旋转表示（Zhou et al. 2019）用于所有位置控制环境（扩展版附录 A.2）。
- **采样加速一句话**（§3.4）：DDIM（Denoising Diffusion Implicit Models, Song et al. 2021）把训练轮数与推理步数解耦——训练 100 步、推理只用 10 步，在 RTX 3080 上实现 **0.1s** 推理延迟，满足实时闭环。

## 5. 实验与结果解读

**评测范围**（论文 §1、§2.2 末）：**15 个任务、4 个基准**（Florence et al. 2021；Gupta et al. 2019 即 RoboTurk/Real-Robot Suite 系；Mandlekar et al. 2021 即 robomimic；Shafiullah et al. 2022 即 BashVision/BCQ 系），涵盖仿真与真机、2–6 自由度动作、单/多任务、全驱动/欠驱动（刚体与流体），演示者含单人与多人；统一在行为克隆设定下比较。**主要结论：所有基准上一致提升，平均改进 46.9%**（摘要与 §1；相对量，对比各自最强基线）。

**三个"为什么有效"的实验**：

1. **多峰行为**（§4.1, Fig. 3）：Push-T 中向左/向右两模式均被学到且单次 rollout 内不切换；对照组 LSTM-GMM/IBC 偏峰、BET 抖动；
2. **位置控制优于速度控制**（§4.2, Fig. 4）：与既往 BC 工作普遍采用速度控制相反，扩散策略在位置控制下更优。论文归因两点：位置控制下动作多峰性更显著，恰被扩散的表达力吸收；位置控制受复合误差影响更小、更适合序列预测；
3. **执行视野 $T_a$ 的权衡**（§4.3, Fig. 5 左）：$T_a$ 增大先提升（时间一致性）后下降（响应性变差），且位置控制下对系统延迟鲁棒（Fig. 5 右）——为"滚动时域"设计提供了定量依据。

**扩展真机实验**（扩展版 §7）：在双臂精细操作上增加三个长时域任务（Egg Beater 打蛋器、Mat Unrolling 席子展开、Shirt Folding 叠衣），均以扩散策略（Transformer 骨干 + 位置控制）完成。

**读表要点**：论文大量使用"相对最大值的成功率变化"呈现消融（Fig. 5），读原始表格时注意归一化口径；基线（BC-CNN/BC-RNN/IBC/BET/LSTM-GMM）的超参均经仔细调优（论文 §VI 消融说明）。

## 6. 局限与后续影响

**论文承认/可见的局限**：

1. **推理慢**：$K$ 轮迭代采样是结构性开销；DDIM 10 步推理缓解到 0.1s（RTX 3080），但相对单步前向策略（RT-1 的 3Hz、ACT 的 0.01s）仍重；
2. **无语言条件**（初版）：条件只有观测 $O_t$，不含指令嵌入——语言条件化要等后续工作（如把指令并入 $O_t$ 的多模态条件、RDT/π0 一类大规模实现）；
3. Transformer 骨干超参敏感；需逐任务调 $T_o/T_p/T_a$ 与调度。

**后续影响**：确立了"扩散 = 模仿学习默认策略表示"的地位；DP3（3D 点云扩散）、ScaleDP、RDT（1B 扩散 Transformer）、π0 的流匹配等一脉相承；RoboTwin 2.0 把 DP 与 DP3 列为官方非预训练基线（[第 08 章](../08_策略训练与部署.md) 8.2）。

## 7. 与本项目对照

- **RoboTwin 的核心 baseline**：DP/DP3 是 RoboTwin 基准最常用的非预训练基线（[第 08 章](../08_策略训练与部署.md) 8.2–8.3：DP 600 epoch、batch 128、horizon 8；DP3 用 1024 点点云）。[第 06 章](../06_实验结果与解读.md) 6.6.2 的结果极具启发性：DP 平均成功率 Easy 28.0% / Hard 0.6%，DP3 55.2% / 5.0%——本精读解释了其机制：域随机化把观测-动作映射变成强多峰/高方差分布，扩散的表达力优势在 Easy 下被体现，但非预训练小模型在 Hard 的分布偏移下整体崩溃，表达力救不了泛化。
- **CVAE vs 扩散对照表**（与[姊妹精读](./ACT_ALOHA_RSS2023.md)并读）：

| 维度 | ACT（CVAE） | Diffusion Policy（DDPM） |
|------|------|------|
| 生成式建模 | 条件 VAE，隐变量 $z$ 一次采样 | 迭代去噪 $K$ 轮，随机 Langevin |
| 训练目标 | 重构 L1 + $\beta\,$KL（ELBO） | 噪声 MSE（变分下界化简） |
| 多峰表达 | 经隐变量 $z$；$\beta$ 大时被压 | 原生（随机初始化 + 注噪） |
| 推理开销 | 单次前向（约 0.01s） | $K$ 轮迭代（DDIM 10 步约 0.1s） |
| 时序组织 | $k$ 帧 chunk + 时间集成 | $T_p$ 预测/$T_a$ 执行滚动时域 |
| 调参敏感点 | chunk size $k$、$\beta$、$m$ | 骨干选择、噪声调度、视野 |

- **动手建议**：在 RoboTwin 单任务上先复现 DP（[第 08 章](../08_策略训练与部署.md) 8.4 流程），再对比 ACT——用同一份演示数据体会"ELBO 重构"与"迭代去噪"在多峰任务上的行为差异。

## 配套阅读

- 教程内：[第 08 章 8.2/8.3 策略与训练细节](../08_策略训练与部署.md)、[第 06 章 6.6 分层成功率](../06_实验结果与解读.md)、[控制规划教程第 05 章 MPC 式 (5.1)(5.2)](../../control_planning/05_模型预测控制mpc.md)、[控制规划教程第 08 章 RL 式 (8.1)](../../control_planning/08_学习式控制强化学习.md)、[姊妹精读：ACT/ALOHA](./ACT_ALOHA_RSS2023.md)、[姊妹精读：RT-1](./RT1_arXiv2022.md)
- 教程外：Ho et al. 2020（DDPM 原始论文，(D0)/(D0′) 与变分下界化简）、Song et al. 2021（DDIM）、Janner et al. 2022（Diffuser，CNN 骨干来源）、3D Diffusion Policy（DP3）
- 项目页：diffusion-policy.cs.columbia.edu（代码、数据、超参全开源）

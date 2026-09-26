# 论文精读｜PPO：近端策略优化算法（arXiv 2017）

> **PDF**：[papers/control_planning/frontier/arXiv-1707.06347_PPO.pdf](../../../papers/control_planning/frontier/arXiv-1707.06347_PPO.pdf)（arXiv:1707.06347v2，共 12 页，公式编号以此版本为准） ｜ **教程**：[第 08 章](../08_学习式控制强化学习.md) §08.2（PPO：截断代理目标） ｜ **代码**：官方实现在 OpenAI Baselines（github.com/openai/baselines，ppo1/ppo2 两套；论文正文未附代码链接）

## 1. 论文信息与一句话贡献

- **题目**：Proximal Policy Optimization Algorithms
- **作者**：John Schulman、Filip Wolski、Prafulla Dhariwal、Alec Radford、Oleg Klimov（OpenAI）
- **发表**：arXiv:1707.06347v2（2017-08-28）。这是一份技术报告而非会议/期刊论文——后被引用时多记作 "arXiv preprint arXiv:1707.06347, 2017b"（本仓库 SAC 论文参考文献即此格式）。
- **编号约定（重要）**：论文**有编号公式 (1)–(12)**，本精读全部按原文式号引用，不自编式号；论文附录 A/B 为超参数表（Table 3–5）与 Atari 逐游戏结果（Table 6、Figure 6）。
- **一句话贡献**：用一阶方法实现信赖域（trust region）优化——以**截断概率比** (clipped probability ratio) 构造悲观下界式的代理目标 $L^{CLIP}$，使同一批采样数据可安全地做多轮 minibatch 梯度更新：取得 TRPO 的可靠性与样本效率，而实现只是一阶优化器里"几行代码的改动"（§7 结论原话），且兼容 dropout、策略/价值网络参数共享等架构（§1）。

## 2. 问题与动机

**要解决什么**（§1）：深度强化学习当时的三条主路线各有硬伤——

1. **DQN 系**（Q-learning + 函数逼近）：在许多简单问题上失效且理论理解差（§1 脚注 1：连续控制基准上从未被证明好用）；
2. **朴素策略梯度**（vanilla policy gradient）：数据效率与鲁棒性差——梯度估计只对"按当前策略采的数据"无偏，采一批只能走一步（**on-policy** (在线策略) 义务）；
3. **TRPO**（信赖域策略优化）：可靠但**相对复杂**（二阶近似 + 共轭梯度 + 线搜索），且与含噪声架构（如 dropout）和参数共享（策略/价值共用主干、辅助任务）**不兼容**——而参数共享正是大模型时代需要的（§1 原文列举）。

**动机核心**（§2.1 的警示 + 摘要）：朴素代理损失 $L^{PG}$ 若在同一批数据上做多轮优化（深度网络必须如此才划算），会带来 **"destructively large policy updates"**（破坏性大更新，§2.1 原话；§6.1 注明：其消融成绩与"无截断无惩罚"相近或更差）。想要的：**既有 TRPO 的单调改进气质、又只用一阶信息、还能多轮复用数据**的目标函数。

**论文的回答**：交替执行"采样 → 对代理目标做多轮随机梯度上升"两步，代理目标被设计成"策略偏离采样策略太远就失去梯度"——摘要称之为对新目标做随机梯度上升的**代理目标** (surrogate objective) 方法族，截断版表现最好。

## 3. 方法总览

**外层循环（Algorithm 1，§5）**：

```
for iteration = 1, 2, …:
    for actor = 1..N:                      # N 个并行采集器
        用 π_θold 与环境交互 T 步           # 固定长度轨迹段（MuJoCo: T=2048）
        计算优势估计 Â_1…Â_T                # GAE(λ)，式 (10)–(12)
    对代理损失 L（L^CLIP 或 L^KLPEN）做
      K 个 epoch 的 minibatch SGD/Adam      # MuJoCo: K=10, 批 64, Adam 3e-4
    θ_old ← θ                               # 数据作废，回到采样
```

**目标函数族**（§2–§5，式号按原文）：

- (2) $L^{PG}$：朴素代理——多轮优化不安全；
- (6) $L^{CPI}$：重要性加权代理——无约束则更新无界（§4.3 详推）；
- (5)/(8) $L^{KLPEN}$：KL 惩罚版——配自适应 $\beta$（§4）；
- (7) $L^{CLIP}$：**截断版（论文主推）**——分段形状自动刹车（§4.4 详推）；
- (9) $L^{CLIP+VF+S}$：截断项 + 价值函数回归 + 熵奖励的组合（参数共享/带熵奖励时用）。

**实现要点**（§5 + 附录 A Table 3–5）：优势用截断 GAE（$\lambda=0.95$、$\gamma=0.99$）；优化器 Adam；策略输出对角高斯的均值与可变标准差（tanh 非线性、两层 64 神经元 MLP，§6.1）；MuJoCo 基准不共享参数（$c_1$ 无关）且不用熵奖励，Atari 基准共享参数（$c_1=1$、熵系数 $c_2=0.01$，Table 5）。

## 4. 关键公式推导（式号全部按论文原文 (1)–(12)；分步依据逐条标注）

### 4.0 符号表

| 符号 | 含义（对应原文位置） |
|---|---|
| $\pi_\theta(a_t \mid s_t)$ | 随机策略（对角高斯，参数 $\theta$） |
| $\pi_{\theta_{old}}$ | 采数据时的旧策略；数据在其下采集，更新期间冻结 |
| $\hat{\mathbb{E}}_t$ | 有限批样本的经验平均（§2.1 定义） |
| $\hat{A}_t$ | 时刻 $t$ 优势函数的估计（GAE，式 (10)–(12)） |
| $r_t(\theta)$ | 概率比 $\pi_\theta(a_t \mid s_t)/\pi_{\theta_{old}}(a_t \mid s_t)$，$r_t(\theta_{old})=1$（§3） |
| $\epsilon$ | 截断半径超参（示例 $\epsilon=0.2$，§3） |
| $\beta,\ d_{targ}$ | KL 惩罚系数与其目标值（§4，$\beta$ 初值 1，Table 1 注） |
| $\gamma,\ \lambda$ | 折扣因子与 GAE 参数（Table 3：$0.99$、$0.95$） |
| $T,\ N,\ K,\ M$ | 轨迹段长 / 并行 actor 数 / epoch 数 / minibatch 大小（Algorithm 1，$M \le NT$） |
| $c_1,\ c_2,\ S$ | 价值损失系数、熵奖励系数、熵 bonus（式 (9)；Table 5：$1,\ 0.01$） |
| $V^{targ}_t$ | 优势估计隐含的价值目标（式 (9) 下文的 $L^{VF}_t = (V_\theta(s_t)-V^{targ}_t)^2$） |

### 4.1 背景：策略梯度、代理损失与"多轮更新为什么危险"（式 (1)(2)）

**式 (1)——最常用的梯度估计器**：

$$
\hat{g} = \hat{\mathbb{E}}_t\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\big].\tag{1}
$$

依据：策略梯度定理 + 回报只取 $t$ 之后（因果裁剪）+ 状态基线——即教程第 08 章的 (8.5)、(8.7)、(8.8) 链条（score 零均值保证 $k<t$ 项与基线项无偏性不变）。$\hat A_t$ 取优势估计而非原始回报是为了降方差：其正负号直接指示"该动作概率该升该降"。

**式 (2)——可微分的代理损失**：

$$
L^{PG}(\theta) = \hat{\mathbb{E}}_t\big[\log \pi_\theta(a_t \mid s_t)\,\hat{A}_t\big].\tag{2}
$$

依据：对 (2) 求 $\theta$ 导数恰好得到 (1)（$\nabla_\theta \log \pi_\theta \cdot \hat A_t$；$\hat A_t$ 对当前批视为常数）。自动微分框架需要的是"损失"而非"梯度"，(2) 就是 (1) 的原函数。

**危险所在**（§2.1）：$L^{PG}$ 只在一阶意义上等价于真目标——且仅当数据来自当前 $\pi_\theta$。用同一批数据做第 2、3、…、$K$ 个 epoch 的更新时，$\pi_\theta$ 已离开 $\pi_{\theta_{old}}$，(1) 的无偏性失效；论文实证：多轮优化 $L^{PG}$ 常导致破坏性大更新（§2.1、§6.1）。

### 4.2 信赖域与重要性加权代理（式 (3)(4)(5)(6)）

**式 (3)(4)——TRPO 的约束形式**：

$$
\max_\theta\ \hat{\mathbb{E}}_t\Big[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat{A}_t\Big],\qquad
\hat{\mathbb{E}}_t\big[\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot \mid s_t),\ \pi_\theta(\cdot \mid s_t)\big]\big] \le \delta.\tag{3,4}
$$

分子分母同乘 $\pi_{\theta_{old}}$ 把"新策略下的优势期望"改写成旧数据上的加权平均（依据：期望的恒等变形，前提 common support——凡 $\pi_\theta>0$ 处须 $\pi_{\theta_{old}}>0$；即教程 (8.9) 的重要性采样 (importance sampling)）。KL 约束保证"升代理 ≈ 升真目标"：TRPO 理论给出悲观下界 $J(\theta) \ge L^{CPI}(\theta) - C\cdot\max_s \mathrm{KL}(\pi_{\theta_{old}}\|\pi_\theta)$（§2.2 引 Schulman et al. 2015）。TRPO 用线性化目标 + 二次近似约束 + 共轭梯度近似求解——这就是它的"复杂"与"不兼容"（§1、§2.2）。

**式 (5)——理论建议的惩罚形式**：

$$
\max_\theta\ \hat{\mathbb{E}}_t\Big[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat{A}_t - \beta\,\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot \mid s_t),\ \pi_\theta(\cdot \mid s_t)\big]\Big].\tag{5}
$$

依据：约束优化的拉格朗日松弛。但论文明确：**固定单一 $\beta$ 不可行**——$\beta$ 要在不同问题、不同训练阶段之间都合适，"additional modifications are required"（§2.2 原文）。

**式 (6)——代理目标 $L^{CPI}$**（§3；上标 CPI = conservative policy iteration，Kakade & Langford 2002）：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)},\qquad
L^{CPI}(\theta) = \hat{\mathbb{E}}_t\big[r_t(\theta)\,\hat{A}_t\big].\tag{6}
$$

**一阶一致性**（多轮更新的"合法性底线"）：$\nabla_\theta r_t = r_t\,\nabla_\theta \log \pi_\theta$（依据：$\nabla_\theta \log r_t = \nabla_\theta \log \pi_\theta - \nabla_\theta \log \pi_{\theta_{old}}$，第二项与 $\theta$ 无关为零；$\nabla \log r = \nabla r / r$ 反用）。在 $\theta_{old}$ 处 $r_t = 1$，故 $\nabla_\theta L^{CPI}\big|_{\theta_{old}} = \hat{\mathbb{E}}_t[\nabla_\theta \log \pi_{\theta_{old}}\,\hat A_t] = \hat g$——正是式 (1)。问题在离开 $\theta_{old}$ 之后。

### 4.3 无约束比率的无界风险（论文 §3 的动机；本节为逐步论证）

**沿射线展开。** 取参数空间任一单位方向 $d$，令 $\theta(\eta) = \theta_{old} + \eta\, d$，记 $g_t = \nabla_\theta \log \pi_\theta(a_t \mid s_t)\big|_{\theta_{old}} \cdot d$。一阶 Taylor 展开（依据：$r_t(\theta_{old})=1$ 下的标准展开，$r_t$ 的导数即 $r_t \nabla_\theta \log\pi_\theta$，见 4.2）：

$$
r_t(\eta) = 1 + \eta\, g_t + O(\eta^2).
$$

**风险一：代理目标线性上升、无内点极大。** 对 $\hat A_t > 0$ 且 $g_t > 0$ 的样本项，$r_t(\eta)\hat A_t = \hat A_t + \eta\, g_t \hat A_t + O(\eta^2)$ 在一阶上严格递增——梯度上升会**持续**增大 $\eta$，只要 epoch 数够多；更新幅度由优化器日程（epoch 数、学习率）决定，而**不是**由目标函数的驻点决定。softmax 参数化下更直观：比率 $r_t = e^{z - z_{old}}$ 对 logit $z$ 严格递增，$\hat A_t>0$ 时 $z \to \infty$ 无任何目标函数阻力——概率质量塌向采到的"好动作"，策略退化为确定性。

**风险二：代理与真目标的脱钩按二阶增长。** 信赖域理论的误差界由 $\max_s \mathrm{KL}(\pi_{\theta_{old}}\|\pi_\theta)$ 控制；KL 对小 $\eta$ 是二阶量（$\tfrac{1}{2}\eta^2 d^\top F d$，$F$ 为 Fisher 信息阵——依据：KL 的二阶展开），故"多走一步"造成的失真按 $\eta^2$ 涨，而 (6) 驱动的位移按 epoch 数线性涨——**二阶项不消失**（$L^{CPI}$ 的 Hessian 一般非零），代理很快失去指示意义。

**风险三：common support 崩坏。** 策略塌缩后，下一批数据不再覆盖 $\pi_{\theta_{old}}$ 的支撑，式 (3)(6) 的重要性权重分母趋零、方差爆炸（依据：重要性权重方差对支撑重叠度敏感；(8.9)/§2.2 的前提被破坏）。

结论：必须给目标函数装"刹车"。两条路线——KL 惩罚（式 (8)）与截断（式 (7)，主推）。

### 4.4 截断目标（论文核心，式 (7)）：分段形状与"二阶更新自动消失"

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon)\,\hat{A}_t\big)\Big],\tag{7}
$$

其中 $\mathrm{clip}(r, 1-\epsilon, 1+\epsilon)$ 把 $r$ 限制在 $[1-\epsilon, 1+\epsilon]$（$\epsilon$ 如 $0.2$）。**min 取"截断项与未截断项的较小者"**：逐项看 $\min(\cdot) \le r_t \hat A_t$，故 $L^{CLIP}$ 处处是 $L^{CPI}$ 的**下界**（悲观界，§3 原文 "a lower bound (i.e., a pessimistic bound)"）。

**分段分析（对应论文 Figure 1 的 $\hat A>0$ 与 $\hat A<0$ 两幅图；$\hat A_t = 0$ 项梯度恒零，略）：**

| 情形 | 区间 | $\min$ 取哪项 | 目标值 | 梯度（对 $r_t$） |
|---|---|---|---|---|
| $\hat A_t > 0$ | $r_t \in (1-\epsilon,\ 1+\epsilon)$ | 相等（$r_t$ 未越界） | $r_t \hat A_t$ | $+\hat A_t$（推高） |
| $\hat A_t > 0$ | $r_t > 1+\epsilon$ | 截断项（$r_t\hat A_t$ 更大） | $(1{+}\epsilon)\hat A_t$ | **0**（平坦） |
| $\hat A_t > 0$ | $r_t < 1-\epsilon$ | 未截断项（$r_t\hat A_t$ 更小） | $r_t \hat A_t$ | $+\hat A_t$（拉回） |
| $\hat A_t < 0$ | $r_t \in (1-\epsilon,\ 1+\epsilon)$ | 相等 | $r_t \hat A_t$ | $\hat A_t$（压低） |
| $\hat A_t < 0$ | $r_t < 1-\epsilon$ | 截断项（$r_t\hat A_t$ 更大，因 $\hat A<0$ 不等号翻转） | $(1{-}\epsilon)\hat A_t$ | **0**（平坦） |
| $\hat A_t < 0$ | $r_t > 1+\epsilon$ | 未截断项 | $r_t \hat A_t$ | $\hat A_t$（拉回） |

逐格依据：$\min$ 逐点取小；$\hat A_t > 0$ 时 $r_t \hat A_t > (1{+}\epsilon)\hat A_t \iff r_t > 1{+}\epsilon$，$\hat A_t < 0$ 时乘负数翻转全部不等号。分段边界 $r_t = 1 \pm \epsilon$ 处两侧函数值相等（clip 连续），故上表处处成立。

**读表要点（论文 §3 的两句话的数学展开）**：

1. **梯度恰在"越界且继续走会继续改善代理"的方向上消失**：$\hat A_t>0,\ r_t>1{+}\epsilon$（利好别贪）与 $\hat A_t<0,\ r_t<1{-}\epsilon$（惩罚别过）两格目标值为常数——对 $r_t$（进而对 $\theta$）的导数严格为零。多轮 epoch 的第 2、3、… 次更新在这些方向上**没有任何推力**：这正是 4.3 的无界风险被从目标函数形状上拆掉——**平坦区域内 Hessian 也为零，二阶及更高的更新自动消失**；更新自然停在 $1 \pm \epsilon$ 附近，无需显式 KL 约束求解器。TRPO 用约束把步长锁在信赖域里，PPO 用目标形状达到同样效果。
2. **"变差"的方向保留纠错梯度**：$\hat A_t>0,\ r_t<1{-}\epsilon$ 与 $\hat A_t<0,\ r_t>1{+}\epsilon$ 两格，min 取未截断项、梯度把 $r_t$ 拉回 $1$——悲观化只豁免"改善"，不豁免"恶化"（§3 原话："we only ignore the change in probability ratio when it would make the objective improve"）。
3. **一阶等价保底**：$r_t = 1$ 处两支相等（clip 未激活），故 $\nabla L^{CLIP}\big|_{\theta_{old}} = \nabla L^{CPI}\big|_{\theta_{old}}$——论文原文注明 "$L^{CLIP}(\theta) = L^{CPI}(\theta)$ to first order around $\theta_{old}$"；起步阶段就是普通策略梯度，截断只在远离处生效。
4. **经验证据**（Figure 2）：沿"PPO 一次更新后的策略参数"与 $\theta_{old}$ 的线性插值方向画各目标：$L^{CLIP}$ 是 $L^{CPI}$ 的下界，且在更新点附近对过大的策略更新施加了惩罚（Hopper-v1 首次更新处 KL ≈ 0.02，图注数字）。

### 4.5 KL 惩罚变体与自适应 $\beta$（式 (8)；论文 §4）

**注意**：论文式 (8) 是**线性 KL 惩罚**目标（与式 (5) 同形、置于 §4 的操作化语境），并非"clip 作用在 KL 上"——"对 KL 截断"的变体本文没有；论文只提到尝试过"在 log 空间截断比率、效果不更好"（§6.1）。

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\Big[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}\hat{A}_t - \beta\,\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot \mid s_t),\ \pi_\theta(\cdot \mid s_t)\big]\Big].\tag{8}
$$

**每次策略更新的两步**（§4 原文流程；$\beta$ 初值 1）：

1. 若干 epoch 的 minibatch SGD 优化 (8)；
2. 实测 $d = \hat{\mathbb{E}}_t\big[\mathrm{KL}\big[\pi_{\theta_{old}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)\big]\big]$，调整 $\beta$：
   - 若 $d < d_{targ}/1.5$：$\beta \leftarrow \beta/2$；
   - 若 $d > d_{targ} \times 1.5$：$\beta \leftarrow \beta \times 2$。

（依据：把实测 KL 伺服到目标值 $d_{targ}$ 的比例控制器；$1.5$ 与 $2$ 是启发式，论文称算法对 $d_{targ}$ 初值与这些系数**不很敏感**。更新后的 $\beta$ 用于下一轮。）实验结论（§4 明说 + Table 1 定量）：KL 惩罚表现**不如**截断目标。

### 4.6 完整目标、GAE 与算法闭环（式 (9)(10)(11)(12)；Algorithm 1）

**式 (9)——策略/价值共享参数时的组合损失**：

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\big[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\big],\qquad L_t^{VF} = \big(V_\theta(s_t) - V^{targ}_t\big)^2,\tag{9}
$$

$c_1, c_2$ 为系数，$S$ 为熵 bonus（鼓励探索，沿 Williams 1992 / [Mni+16] 的做法）。MuJoCo 基准不共享参数、不用熵 bonus（§6.1：$c_1$ 无关、无熵项）；Atari 用 Table 5：$c_1 = 1$、$c_2 = 0.01$。

**式 (10)(12)——截断优势估计及其 TD 残差**：

$$
\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t-1} r_{T-1} + \gamma^{T-t} V(s_T),\qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).\tag{10,12}
$$

**式 (11)——截断 GAE**（论文：在长度 $T$ 段内不越界，段尾以 $V(s_T)$ 自举；$\lambda = 1$ 时应化归 (10)）：

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}.\tag{11}
$$

**式号校勘（重要）**：PDF 印刷版 (10)(11) 末项指数印作 $\gamma^{T-t+1}r_{T-1}$ 与 $(\gamma\lambda)^{T-t+1}\delta_{T-1}$（经 300 DPI 渲染核对）。正确指数应为 $T-t-1$，证据是论文自己的声明"式 (11) 在 $\lambda=1$ 时化归 (10)"——把 (11) 取 $\lambda=1$ 逐项展开：$\sum_{l=0}^{T-t-1}\gamma^l \delta_{t+l} = \sum_l \gamma^l r_{t+l} + \sum_l \gamma^{l+1}V(s_{t+l+1}) - \sum_l \gamma^l V(s_{t+l})$；后两个和式指标错位一格、中间项逐对相消（telescoping，依据：$\sum_{m=1}^{T-t}\gamma^m V(s_{t+m}) - \sum_{m=0}^{T-t-1}\gamma^m V(s_{t+m}) = \gamma^{T-t}V(s_T) - V(s_t)$），只剩 $-V(s_t) + \gamma^{T-t}V(s_T)$ 与 $\sum_{l=0}^{T-t-1}\gamma^l r_{t+l}$——末个回报 $r_{T-1}$（$l = T-t-1$）的系数是 $\gamma^{T-t-1}$。故本精读按 $T-t-1$ 引用并注明原文印刷如此（一般 $\lambda$ 的推导与偏差-方差折中见教程 (8.13)–(8.15)）。

**算法闭环**：采样（$N$ actors × $T$ 步）→ (12)(11) 算 $\hat A_t$ → $K$ 个 epoch 的 minibatch Adam 上升 (7)（或 (8)/(9)）→ $\theta_{old} \leftarrow \theta$。**超参数表（附录 A）**：MuJoCo（Table 3）：$T=2048$、Adam $3\times10^{-4}$、epoch 10、minibatch 64、$\gamma=0.99$、$\lambda=0.95$；Roboschool 人形（Table 4）：$T=512$、epoch 15、minibatch 4096、actor 32（跑动）/128（flagrun）、动作分布 log-std 从 $-0.7$ 线性退火到 $-1.6$、学习率按目标 KL 调整；Atari（Table 5）：$T=128$、8 actors、epoch 3、minibatch $32\times8$、$\epsilon = 0.1\times\alpha$（退火）。另注：论文正文**未提及 advantage 归一化**——每批 $\hat A$ 标准化是官方 Baselines 实现层面（ppo2）的技巧，引用时应注明实现级出处。

## 5. 实验与结果解读（定量处均出自论文原文）

**消融：四个变体同台（§6.1，Table 1）**。7 个 OpenAI Gym 连续控制任务（脚注 2：HalfCheetah、Hopper、InvertedDoublePendulum、InvertedPendulum、Reacher、Swimmer、Walker2d，均 ~v1，MuJoCo 引擎，各训 100 万步）；每任务 3 个随机种子、以最后 100 回合平均回报计分，分数归一化（随机策略 0 分、最优 1 分）后对 21 个 run 取平均。结果：**无截断无惩罚 −0.39**（负分：half-cheetah 上学崩、比初始策略还差）；**截断 $\epsilon=0.2$ 得 0.82（全场最优）**，$\epsilon=0.1/0.3$ 得 0.76/0.70；自适应 KL 三个 $d_{targ}$ 得 0.68/0.74/0.71；固定 KL 四个 $\beta$ 得 0.62–0.72。解读：截断不仅最好，且对 $\epsilon$ 相当不敏感（三档全部 0.7+）——超参鲁棒性是 PPO 的卖点之一；KL 惩罚系最好 0.74，验证 §4"KL 惩罚表现不如截断"。

**与已有算法对比（§6.2，Figure 3）**。PPO（clip、$\epsilon=0.2$）对比 TRPO、CEM（交叉熵方法）、自适应步长朴素策略梯度、A2C、A2C+trust region、ACER：在几乎所有连续控制环境上 PPO 占优（论文原话 "outperforms the previous methods on almost all the continuous control environments"）。

**机器人演示：人形跑与转向（§6.3，Figure 4/5）**。Roboschool 3D 人形三任务：(1) RoboschoolHumanoid 向前跑；(2) Flagrun——目标位置每 200 步随机更换；(3) FlagrunHarder——被方块砸还要起身。学习曲线显示回报随 1 亿步量级训练持续上升（定性），Figure 5 为学会的追逐-转向策略截图。并行工作 Heess et al.（2017）用 §4 的自适应 KL 变体学 3D 机器人运动。

**Atari（§6.4，Table 2）**。49 个游戏、与 A2C/ACER 同网络：以"整个训练期平均回报"计 PPO 赢 30/49（A2C 1、ACER 18）；以"最后 100 回合"计 ACER 赢 28、PPO 19——论文摘要的结论：PPO 在**样本复杂度**上显著更优、最终性能相近，实现更简单。

## 6. 局限与后续影响

1. **on-policy 样本效率是路线性上限**：每轮更新后数据作废（Algorithm 1 末行 $\theta_{old}\leftarrow\theta$）；真机上交互贵，PPO 的大样本消耗只能靠大规模并行仿真摊薄——这正是第 08 章对比表中"数据效率：低"的出处，也解释了 PETS 精读引的"PPO 样本比模型基多两个数量级"。
2. **无理论保证的"理论气质"**：截断目标的悲观下界论证是构造性直觉（min 的逐点不等式 + Figure 2 的经验下界），不是 TRPO 那样的性能改进定理；$\epsilon$ 与 $K$ 的组合实质上决定信任域半径，超出该区域的行为无刻画。
3. **仍有超参，只是变钝**：$\epsilon$、$K$、$T$、学习率退火等仍需调（Table 3–5 给出的是调好的值）；论文展示的是"对每档超参的性能差距小"（Table 1），不是"零调参"。
4. **实现细节敏感的后续讨论**：后续研究（如 Engstrom et al., ICLR 2020, *Implementation Matters in Deep RL*）系统指出 PPO 与 TRPO 的性能差距有相当部分来自实现细节（advantage 归一化、价值裁剪、初始化等）——印证"算法即实现"的工程属性。
5. **后续影响**：PPO 成为 on-policy 策略优化的**事实默认**——大规模并行仿真训练（Isaac Gym 系）、腿式运动（教程第 09 章 Rapid Locomotion 的教师策略）、直至 RLHF 对齐微调，均以 PPO 为训练器；其"截断代理"思路也被各类变体（滑动比率、KL 早停等）继承。

## 7. 与本项目对照

**教程第 08 章映射表**（编号已核实，§08.1–§08.2）：

| 本论文内容（原文位置） | 教程第 08 章（编号） |
|---|---|
| 策略梯度估计器 (1)、代理损失 (2) | (8.5)、(8.7)、(8.8) 的链式推导（§08.1 ⑤） |
| 重要性加权代理 (3)(6)、common support | 重要性采样 (8.9)、代理目标一阶一致性 (8.10)（§08.2 ⑤） |
| 无约束比率的无界风险（§3 动机） | (8.10) 下方"无约束比率的无界风险"段 |
| KL 惩罚 (5)/(8) + 自适应 $\beta$（§4） | $L^{KLPEN}$ (8.11) 及"单一 $\beta$ 难跨阶段"的选型理由 |
| **截断目标 (7)、分段形状分析（Figure 1）** | **$L^{CLIP}$ (8.12)、四情形梯度表（§08.2 ⑤）** |
| TD 残差 (12)、截断 GAE (11)（$\lambda=1$ 化归 (10)） | TD 残差 (8.13)、$k$ 步裂项 (8.14)、GAE (8.15) |
| 组合损失 (9)、Algorithm 1、超参 Table 3–5 | §08.2 ② 外层循环两步 + "辅以价值回归与熵奖励"一句 |

**为什么腿式 RL 都用 PPO 训练器**（回引[教程第 09 章 §09.2](../09_前沿学习式规划与腿式控制.md)：Rapid Locomotion 的特权教师即用 PPO 训练——其 RL 目标即第 08 章 PPO 的期望折扣回报）：GPU 大规模并行仿真使"采样"几乎免费（一次可并行数千环境、一批即百万级转移），on-policy 的最大短板在仿真里失效；而 PPO 的截断目标一阶、免共轭梯度、对超参钝感、便于与域随机化和策略/价值共享架构组合——工程可扩展性压倒样本效率。SAC 的 replay buffer / 目标网络 / 温度调参在大规模并行设定下收益有限、复杂度反增（对比表见教程 §08.3 ③ 的三维表）。

**与本项目其他精读的关系**：[PETS 精读](PETS_NeurIPS2018.md)以 PPO 为无模型对照（其 Fig 3 结论"渐近相当但样本多数量级"是 PPO 样本效率定位的第三方证据）；[CBF-Survey 精读](CBF-Survey_ECCV2019.md)对应"PPO 无安全保证、部署需 CBF-QP 兜底"（教程 (7.9)）。

## 配套阅读

- 教程主线：[第 08 章｜学习式控制：强化学习](../08_学习式控制强化学习.md)（§08.2 为本论文的教程化；(8.9)–(8.15) 是本精读 §4 的姊妹推导；§08.3 ③ 有 PPO/SAC/模型基三方对比表）。
- 同目录精读：[PETS 精读](PETS_NeurIPS2018.md)（模型基对照，PPO 为其无模型基线）｜ [CBF-Survey 精读](CBF-Survey_ECCV2019.md)（学习策略的部署安全兜底）。
- 姊妹精读：[SAC 精读](SAC_ICML2018.md)（off-policy 最大熵路线；其 §6.4 引本文为 [Schulman et al., 2017b] 基线）。
- 论文延伸：GAE（Schulman et al., arXiv:1506.02438——式 (10)(11) 的来源）；TRPO（Schulman et al., CoRR 1502.05477——式 (3)(4)(5) 的理论出处）；Kakade & Langford 2002（$L^{CPI}$ 上标出处）；Engstrom et al. 2020（实现细节研究）。
- 版本说明：本精读以 arXiv:1707.06347v2（2017-08-28，12 页）为准，公式 (1)–(12)、Algorithm 1、Table 1–6、Figure 1–6 均按该版本核对；式 (10)(11) 末项指数的印刷笔误已在 §4.6 校勘。

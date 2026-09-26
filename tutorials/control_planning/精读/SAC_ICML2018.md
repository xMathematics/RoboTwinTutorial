# 论文精读｜SAC：软演员–评论家（ICML 2018）

> **PDF**：[papers/control_planning/frontier/arXiv-1801.01290_SAC.pdf](../../../papers/control_planning/frontier/arXiv-1801.01290_SAC.pdf)（arXiv:1801.01290v2，共 14 页，ICML 2018 版式，公式编号 (1)–(21) 以此版本为准） ｜ **教程**：[第 08 章](../08_学习式控制强化学习.md) §08.3（SAC 与模型法对比） ｜ **代码**：官方实现 github.com/haarnoja/sac（论文 §5 脚注 1），配套视频 sites.google.com/view/soft-actor-critic

## 1. 论文信息与一句话贡献

- **题目**：Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
- **作者**：Tuomas Haarnoja、Aurick Zhou、Pieter Abbeel、Sergey Levine（UC Berkeley，BAIR）
- **发表**：ICML 2018（第 35 届国际机器学习会议）；本仓库为 arXiv:1801.01290v2（2018-08-08，即 ICML 版式）
- **版本说明（重要）**：本版含**独立价值网络 $V_\psi$**（式 (5)）；后续版本（arXiv:1812.05905 及通用实现）去掉 $V$ 网络、直接用双 Q 目标回归，并加入自动温度调节——本精读按本仓库 PDF 的 (1)–(21) 式写，差异处逐条注明。
- **一句话贡献**：把**最大熵** (maximum entropy) 强化学习做成**off-policy actor-critic**——策略在最大化期望回报的同时最大化自身熵，配合软策略迭代（软 Bellman 评估 + KL 投影改进，Lemma 1/2、Theorem 1 给出表格情形收敛证明）与重参数化随机策略、双 Q、目标网络、tanh 压缩的深度实现：在连续控制基准上样本效率超过 off-policy 的 DDPG、稳定性超过 on-policy 的 PPO，且随机种子间表现几乎一致。

## 2. 问题与动机

**要解决什么**（§1）：无模型深度 RL 有两大痼疾——(a) **样本复杂度** (sample complexity) 高：动辄数百万步交互，真机不可承受；(b) **超参敏感、易碎** (brittle convergence)：学习率、探索系数等必须逐任务仔细调。

**既有路线的缺口**（§1–§2）：

- **on-policy 策略梯度**（TRPO/PPO/A3C）：稳定但每步梯度都要新鲜样本，样本贵；以熵为正则的变体（如 soft 系先前工作）只把熵当正则项，"off-policy 训练下策略仍会过早收敛到确定性"（§2 对 SAC 前身 soft Q-learning 的批评：不是真 actor-critic、性能不超 DDPG）。
- **off-policy 价值方法**（DDPG）：样本效率高，但确定性 actor 与 Q 网络的耦合**极难稳定**（§1 原文 "extremely brittle and hyperparameter sensitive"，引 Duan 2016、Henderson 2017 的复现研究）；对高维任务（Humanoid）几乎不可用。
- 论文的处方：**off-policy 的样本效率 + 随机 actor 的稳定与探索**，从最大熵目标出发**推导**（而非启发式拼装）算法——§1 强调与先前最大熵工作的区别在于"证明收敛到策略类内最优、与策略参数化无关"。

**最大熵的三个概念性好处**（§3.2 原文列举）：(1) 策略被激励去**更广泛地探索**、同时不放弃明显不佳的动作；(2) 可**捕捉多个近似最优行为模式**——多动作同等吸引时概率质量均分；(3) 实证上**加速学习**并改善探索。

## 3. 方法总览

**理论层（§4.1，表格情形的软策略迭代）**：

```
评估 Soft Policy Evaluation：软 Bellman 算子 T^π（式 (2)(3)）迭代
        → Lemma 1：收敛到该策略的软 Q 值（γ-压缩）
改进 Soft Policy Improvement：向 exp(Q^π_old/α)/Z 做 KL 投影（式 (4)）
        → Lemma 2：Q^π_new ≥ Q^π_old 处处成立
交替至收敛 → Theorem 1：收敛到策略类 Π 内最优
```

**深度实现（§4.2，Algorithm 1）**——三个参数向量 $\psi, \theta, \phi$（价值、Q、策略；双 Q 则 $\theta_1,\theta_2$）：

- 每个环境步：$a_t \sim \pi_\phi$，存入 replay buffer $\mathcal{D}$；
- 每个梯度步：更新 $V_\psi$（回归"双 Q 取 min − log π"目标，式 (5)–(6)）；更新 $Q_{\theta_i}$（软 Bellman 残差，式 (7)–(9)，目标用目标价值网络 $\bar\psi$）；更新 $\pi_\phi$（KL 目标的重参数化形式，式 (10)–(13)）；目标网络软更新 $\bar\psi \leftarrow \tau\psi + (1-\tau)\bar\psi$。

**工程件**（附录 C/D/E）：tanh 压缩把高斯样本挤进有界动作区间（式 (20)(21)）；超参 Table 1（Adam 3e-4、$\gamma=0.99$、buffer $10^6$、两层 256、批 256、$\tau=0.005$）；奖励缩放 Table 2（充任温度，见 §4.7）；Trust-PCL 与 hard-update 消融（附录 E、Figure 4）。

## 4. 关键公式推导（式号全部按论文原文 (1)–(21)；分步依据逐条标注）

### 4.0 符号表

| 符号 | 含义（对应原文位置） |
|---|---|
| $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ | MDP 五元组；$p: \mathcal{S}\times\mathcal{S}\times\mathcal{A} \to [0,\infty)$ 转移密度；$r \in [r_{\min}, r_{\max}]$ 有界奖励（§3.1） |
| $\pi_\phi(a_t \mid s_t)$ | 随机策略（tanh-Gaussian 网络，$\phi$ 为参数） |
| $\rho_\pi(\cdot),\ \rho_\pi(\mathbf{s}_t, \mathbf{a}_t)$ | 策略诱导的状态/状态-动作访问分布（§3.1） |
| $\alpha$ | 温度 (temperature)：熵项相对奖励的权重（式 (1) 后文；正文按约定省略） |
| $\mathcal{H}(\pi(\cdot \mid s))$ | 策略在 $s$ 处的熵 $-\mathbb{E}_{a\sim\pi}[\log \pi(a \mid s)]$ |
| $\mathcal{T}^\pi$ | 软 Bellman 备份算子（式 (2)） |
| $Q^{soft},\ V^{soft}$ | 软状态-动作价值 / 软状态价值（式 (2)(3) 的不动点） |
| $Z^{\pi_{old}}_{soft}(s_t)$ | 配分函数 (partition function)：使 $\exp(Q^{soft})$ 归一化（式 (4) 下方） |
| $\Pi$ | 策略类（如参数化高斯族；Lemma/定理均在 $\lvert\mathcal{A}\rvert < \infty$ 假设下） |
| $\psi,\ \theta_i,\ \phi,\ \bar\psi$ | 价值网络 / 第 $i$ 个 Q 网络 / 策略网络 / 目标价值网络参数（§4.2） |
| $\mathcal{D}$ | replay buffer（式 (5)(7)(10)(12) 的采样源） |
| $\epsilon_t$ | 重参数化噪声（式 (11)：从固定分布如球面高斯采样） |
| $\tau$ | 目标网络滑动系数（Algorithm 1；Table 1 取 0.005） |

### 4.1 最大熵目标与温度的角色（式 (1)；折扣版式 (14)）

**式 (1)——有限时域最大熵目标**（§3.2）：

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \rho_\pi}\big[r(\mathbf{s}_t, \mathbf{a}_t) + \alpha\, \mathcal{H}\big(\pi(\cdot \mid \mathbf{s}_t)\big)\big].\tag{1}
$$

**符号读法**：熵项是**加**——每步在环境奖励之外再"发"一份熵奖励 $\alpha \mathcal{H} \ge 0$（熵非负；§2 原话 "augments the objective with an entropy maximization term"）。**$\alpha$ 的角色**（式 (1) 后原文）：决定熵项对奖励的相对重要性、从而控制最优策略的随机性；$\alpha \to 0$ 时退化回标准期望回报目标。**关键约定**：因为缩放奖励等价于缩放 $\alpha$，正文把温度显式省略（"subsumed into the reward by scaling it by $\alpha^{-1}$"）——所以式 (2)–(13) 里看不到 $\alpha$；本精读在 4.3 的 Boltzmann 推导处按教程 (8.19) 的方式补回。**无限时域折扣版**（附录 A，式 (14)）：对"从每个 $(\mathbf{s}_t,\mathbf{a}_t)$ 出发的未来折扣奖励与折扣熵之和"再按 $\rho_\pi$ 求和（折扣在策略梯度里只是方差削减手段，目标定义见 Thomas 2014 的讨论——附录 A 原话）；教程 (8.16) 即此形式。

### 4.2 软策略评估：软 Bellman 算子与收缩性（式 (2)(3)，Lemma 1，附录 B.1 式 (15)）

**式 (2)(3)——算子与软价值**（§4.1；温度按论文约定省略）：

$$
\mathcal{T}^\pi Q(\mathbf{s}_t, \mathbf{a}_t) \triangleq r(\mathbf{s}_t, \mathbf{a}_t) + \gamma\, \mathbb{E}_{\mathbf{s}_{t+1} \sim p}\big[V(\mathbf{s}_{t+1})\big],\qquad
V(\mathbf{s}_t) = \mathbb{E}_{\mathbf{a}_t \sim \pi}\big[Q(\mathbf{s}_t, \mathbf{a}_t) - \log \pi(\mathbf{a}_t \mid \mathbf{s}_t)\big].\tag{2,3}
$$

**熵项符号再强调**：$\log \pi \le 0$，故 $-\log\pi = \mathcal{H} \ge 0$——软价值 = "动作的 Q 期望**加**策略在该状态要挣的熵奖励"（教程 (8.17) 同）。$\mathcal{T}^\pi$ 的不动点记 $Q^{soft}$，得**软 Bellman 方程** $Q^{soft}(\mathbf{s},\mathbf{a}) = r(\mathbf{s},\mathbf{a}) + \gamma\,\mathbb{E}_{\mathbf{s}'}[V^{soft}(\mathbf{s}')]$（教程 (8.18)）。

**Lemma 1（Soft Policy Evaluation）**：反复施加 $\mathcal{T}^\pi$（$Q^{k+1} = \mathcal{T}^\pi Q^k$）收敛到 $\pi$ 的软 Q 值（$\lvert\mathcal{A}\rvert < \infty$）。

**论文证明的路线**（附录 B.1）：定义**熵增广奖励**（式 (15)）$r_\pi(\mathbf{s}_t,\mathbf{a}_t) \triangleq r(\mathbf{s}_t,\mathbf{a}_t) + \mathbb{E}_{\mathbf{s}_{t+1}\sim p}\big[\mathcal{H}(\pi(\cdot \mid \mathbf{s}_{t+1}))\big]$，把迭代改写成（式 (15)）$Q(\mathbf{s}_t,\mathbf{a}_t) \leftarrow r_\pi(\mathbf{s}_t,\mathbf{a}_t) + \gamma\,\mathbb{E}_{\mathbf{s}_{t+1}\sim p,\, \mathbf{a}_{t+1}\sim\pi}[Q(\mathbf{s}_{t+1},\mathbf{a}_{t+1})]$——形态与标准策略评估完全一致（依据：把 (3) 代入 (2) 后，下一状态的 $V$ 展开为 $Q - \log\pi$ 的期望，$\log\pi$ 期望即熵、归并进 $r_\pi$），故引用标准策略评估收敛结论（Sutton & Barto 1998）；$\lvert\mathcal{A}\rvert < \infty$ 保证 $r_\pi$ 有界（依据：熵的上界 $\mathcal{H} \le \log\lvert\mathcal{A}\rvert$，$r$ 本身有界）。

**压缩不等式的显式展开**（论文引标准结论，此处把其背后的论证写全）：先由 (3) 代入 (2) 得 $\mathcal{T}^\pi Q(\mathbf{s},\mathbf{a}) = r(\mathbf{s},\mathbf{a}) + \gamma\,\mathbb{E}_{\mathbf{s}'\sim p,\,\mathbf{a}'\sim\pi}\big[Q(\mathbf{s}',\mathbf{a}') - \log\pi(\mathbf{a}' \mid \mathbf{s}')\big]$（依据：期望的线性）。对任意 $Q_1, Q_2$：

$$
\big\lVert \mathcal{T}^\pi Q_1 - \mathcal{T}^\pi Q_2 \big\rVert_\infty
= \gamma \sup_{\mathbf{s},\mathbf{a}} \Big| \mathbb{E}_{\mathbf{s}',\mathbf{a}'}\big[Q_1(\mathbf{s}',\mathbf{a}') - Q_2(\mathbf{s}',\mathbf{a}')\big] \Big|
\le \gamma \sup_{\mathbf{s}',\mathbf{a}'} \big|Q_1 - Q_2\big| = \gamma \lVert Q_1 - Q_2 \rVert_\infty,
$$

第一步依据：奖励项与 $\log\pi$ 项都不含 $Q$，相减时恒等消去（算子对 $Q$ 是仿射的）；第二步依据：$|\mathbb{E}X| \le \mathbb{E}|X| \le \sup|X|$（期望的绝对值不等式 + 上确界放缩）。$\gamma < 1$ 故 $\mathcal{T}^\pi$ 是 $\gamma$-**压缩映射** (contraction mapping)：由 Banach 不动点定理，迭代从任意初始 $Q^0$ 收敛到唯一不动点，速率 $\gamma$。

### 4.3 软策略改进：最优策略是 Boltzmann 分布（式 (4)，Lemma 2，附录 B.2 式 (16)–(19)）

**式 (4)——把新策略向"旧 Q 的指数"投影**（§4.1）：

$$
\pi_{new} = \arg\min_{\pi' \in \Pi} D_{KL}\Big(\pi'(\cdot \mid \mathbf{s}_t)\ \Big\|\ \frac{\exp\big(Q^{soft_{old}}(\mathbf{s}_t, \cdot)\big)}{Z^{soft_{old}}(\mathbf{s}_t)}\Big).\tag{4}
$$

配分函数 $Z^{soft_{old}}(\mathbf{s}_t)$ 使指数分布归一（一般不可积；论文 §4.1 明说它对新策略的梯度无贡献、可忽略——它是 $\mathbf{s}_t$ 的函数，不含 $\phi$）。**为什么最小化 (4) 的策略是 Boltzmann 分布**——按教程 (8.19) 补回温度 $\alpha$（指数写作 $\exp(Q/\alpha)$，本节以下自洽），逐步：

**(i) KL 展开。** 记 $\pi_B(a \mid s) \triangleq \exp\big(Q(s,a)/\alpha\big) / Z_\alpha(s)$，$Z_\alpha(s) = \int \exp\big(Q(s,u')/\alpha\big)\,\mathrm{d}u'$。由定义 $D_{KL}(p \| q) = \mathbb{E}_p[\log p - \log q]$，且 $\log \pi_B = Q/\alpha - \log Z_\alpha$（依据：对指数取对数），

$$
D_{KL}\big(\pi' \,\big\|\, \pi_B\big) = \mathbb{E}_{a\sim\pi'}\big[\log \pi'(a \mid s)\big] - \frac{1}{\alpha}\,\mathbb{E}_{a\sim\pi'}\big[Q(s,a)\big] + \log Z_\alpha(s).
$$

（依据：期望的线性；$\log Z_\alpha(s)$ 与 $a$ 无关故移出期望。）

**(ii) 与 $\pi'$ 无关的项可弃。** $\log Z_\alpha(s)$ 只依赖 $s$——对 $\arg\min_{\pi'}$ 无影响；两侧同乘 $\alpha > 0$ 不改变最小元（依据：正数乘法保序）。于是 (4) 等价于最小化 **actor 损失** $\mathbb{E}_{a\sim\pi'}\big[\alpha \log \pi'(a \mid s) - Q(s,a)\big]$，也等价于最大化 $\mathbb{E}_{\pi'}[Q] + \alpha\,\mathcal{H}(\pi')$（依据：$\mathcal{H} = -\mathbb{E}[\log\pi']$，熵的定义）——**"奖励 + 熵"的直接梯度上升目标**正是从 KL 投影里自然长出来的。

**(iii) 吉布斯不等式（无约束 $\Pi$ 时闭式解）。** 断言：对任意密度 $p, q$，$D_{KL}(p\|q) \ge 0$ 且取等当且仅当 $p = q$。证：在 $p > 0$ 处令 $w(x) = q(x)/p(x) > 0$。对一切 $w>0$ 有 $\log w \le w - 1$（依据：$\log$ 是凹函数，其图像处处位于 $w=1$ 处的切线 $y = w-1$ 之下，等号 iff $w=1$）。两侧乘 $p(x) \ge 0$ 并积分：$\int_{p>0} p \log w\,\mathrm{d}x \le \int_{p>0} (q - p)\,\mathrm{d}x \le \int (q - p)\,\mathrm{d}x = 0$（第二不等式依据：在 $p=0$ 区域补上 $q - p = q \ge 0$ 只会增大右端；最后依据：两密度各自积分为 1）。左侧即 $-D_{KL}(p\|q)$，故 $D_{KL}(p\|q) \ge 0$；取等 iff $w = 1$ 几乎处处，即 $p = q$。

**(iv) 闭合。** 取 $p = \pi'$、$q = \pi_B$：$\pi_B$ 自身达到下界 0（$D_{KL}(\pi_B\|\pi_B) = 0$），且由 (iii) 是唯一最小元。故策略类不受限时式 (4) 的解为

$$
\pi^{*}(a \mid s) = \frac{\exp\big(Q^{soft}(s,a)/\alpha\big)}{Z_\alpha(s)}\quad\text{——Boltzmann 分布：概率随软 Q 指数增长、宽度由 } \alpha \text{ 调节},
$$

这与教程 (8.20) 一致。**副产品**：把 $\pi^*$ 代回 (3)：$\log\pi^* = Q/\alpha - \log Z_\alpha$，故 $V^{soft}(s) = \mathbb{E}_{\pi^*}[Q - \alpha\log\pi^*] = \mathbb{E}_{\pi^*}[Q - Q + \alpha \log Z_\alpha] = \alpha \log Z_\alpha(s)$——配分函数就是软价值（不可积，但不进梯度，见上）。

**Lemma 2（Soft Policy Improvement）**：$\pi_{new}$ 为式 (4) 的最优元，则 $Q^{\pi_{new}}(\mathbf{s}_t,\mathbf{a}_t) \ge Q^{\pi_{old}}(\mathbf{s}_t,\mathbf{a}_t)$ 对一切 $(\mathbf{s}_t,\mathbf{a}_t)$。**论文证明的链条**（附录 B.2）：

1. 式 (16)：把 (4) 的被最小化泛函记 $J_{\pi_{old}}(\pi' \mid \mathbf{s}_t) = \mathbb{E}_{a\sim\pi'}\big[\log\pi'(a \mid \mathbf{s}_t) - Q^{\pi_{old}}(\mathbf{s}_t,a) + \log Z^{\pi_{old}}(\mathbf{s}_t)\big]$（依据：KL 的展开，同 (i)）。
2. 因 $\pi_{old} \in \Pi$ 是可行点：$J(\pi_{new} \mid \mathbf{s}_t) \le J(\pi_{old} \mid \mathbf{s}_t)$（依据：最小值不超过任意可行值）——写开即式 (17)（两侧各按 $\pi_{new}, \pi_{old}$ 取期望的上述泛函不等式）。
3. 两侧的 $\log Z^{\pi_{old}}(\mathbf{s}_t)$ 只依赖状态，消去（依据：同 (ii)）；重排得式 (18)：
   $\mathbb{E}_{a\sim\pi_{new}}\big[Q^{\pi_{old}}(\mathbf{s}_t,a) - \log\pi_{new}(a \mid \mathbf{s}_t)\big] \ge V^{\pi_{old}}(\mathbf{s}_t)$（右端即 (3) 的定义）。
4. 把软 Bellman 方程反复施加于式 (18)：$Q^{\pi_{old}}(\mathbf{s}_t,a_t) = r + \gamma\mathbb{E}_{\mathbf{s}'}[V^{\pi_{old}}(\mathbf{s}')] \le r + \gamma\,\mathbb{E}_{\mathbf{s}'\sim p}\big[\mathbb{E}_{a'\sim\pi_{new}}[Q^{\pi_{old}}(\mathbf{s}',a') - \log\pi_{new}]\big]$（依据：(18) 对 $\mathbf{s}'$ 逐点成立、代入期望内）；对 RHS 中残留的 $Q^{\pi_{old}}$ 重复"软 Bellman 展开 + (18) 放缩"，$\gamma$ 折扣保证无穷级数收敛（依据：$r$ 与 $\log\pi$ 有界，几何级数求和），逐层放缩后恰得式 (19)：$Q^{\pi_{old}}(\mathbf{s}_t,\mathbf{a}_t) \le Q^{\pi_{new}}(\mathbf{s}_t,\mathbf{a}_t)$。收敛到 $Q^{\pi_{new}}$ 由 Lemma 1（对 $\pi_{new}$ 评估）给出。

**Theorem 1（Soft Policy Iteration）**：评估与改进交替收敛到 $\Pi$ 内最优。**证明骨架**（附录 B.3）：$\{Q^{\pi_i}\}$ 单调不减（Lemma 2）且有上界（$r$ 与熵有界、$\lvert\mathcal{A}\rvert<\infty$）⇒ 收敛到某 $\pi^*$（依据：单调有界数列收敛）；收敛时对任意 $\pi' \in \Pi, \pi' \ne \pi^*$ 必有 $J_{\pi^*}(\pi' \mid \mathbf{s}_t) < J_{\pi^*}(\pi^* \mid \mathbf{s}_t)$（否则还能再改进一步）；用 Lemma 2 证明同款迭代论证即得 $Q^{\pi^*}(\mathbf{s}_t,\mathbf{a}_t) > Q^{\pi'}(\mathbf{s}_t,\mathbf{a}_t)$ 对一切 $(\mathbf{s}_t,\mathbf{a}_t)$——$\Pi$ 内其余策略的软值均低于收敛策略，故 $\pi^*$ 即 $\Pi$ 内最优。

### 4.4 深度实现的三组损失（式 (5)–(10)；Algorithm 1）

**价值网络（式 (5)(6)）**——回归"双 Q 取 min 后的软目标"：

$$
J_V(\psi) = \mathbb{E}_{\mathbf{s}_t \sim \mathcal{D}}\Big[\tfrac{1}{2}\Big(V_\psi(\mathbf{s}_t) - \mathbb{E}_{\mathbf{a}_t\sim\pi}\big[Q_\theta(\mathbf{s}_t,\mathbf{a}_t) - \log \pi_\phi(\mathbf{a}_t \mid \mathbf{s}_t)\big]\Big)^2\Big],\tag{5}
$$

$$
\nabla_\psi J_V(\psi) = \nabla_\psi V_\psi(\mathbf{s}_t)\big(V_\psi(\mathbf{s}_t) - Q_\theta(\mathbf{s}_t,\mathbf{a}_t) + \log \pi_\phi(\mathbf{a}_t \mid \mathbf{s}_t)\big).\tag{6}
$$

（印刷版式 (5) 内是单个 $Q_\theta$；配合双 Q 时取 $\min_i Q_{\theta_i}$——min 的出处是 §4.2 末的文字说明（"use the minimum of the Q-functions for the value gradient in Equation 6 and policy gradient in Equation 13"）而非式 (5) 本身，本精读如实分层转述。内层对 $\mathbf{a}_t$ 的期望在实现中用当前策略采单样本近似且不引入偏差——论文 §4.2 原话。式 (6) 依据：对平方损失求导的链式法则，$a_t$ 按当前策略采。）

**Q 网络（式 (7)(8)(9)）**——最小化软 Bellman 残差：

$$
J_Q(\theta) = \mathbb{E}_{(\mathbf{s}_t,\mathbf{a}_t)\sim\mathcal{D}}\Big[\tfrac{1}{2}\big(Q_\theta(\mathbf{s}_t,\mathbf{a}_t) - \widehat{Q}(\mathbf{s}_t,\mathbf{a}_t)\big)^2\Big],\qquad
\widehat{Q}(\mathbf{s}_t,\mathbf{a}_t) = r(\mathbf{s}_t,\mathbf{a}_t) + \gamma\,\mathbb{E}_{\mathbf{s}_{t+1}\sim p}\big[V_{\bar\psi}(\mathbf{s}_{t+1})\big],\tag{7,8}
$$

$$
\nabla_\theta J_Q(\theta) = \nabla_\theta Q_\theta(\mathbf{a}_t,\mathbf{s}_t)\big(Q_\theta(\mathbf{s}_t,\mathbf{a}_t) - r(\mathbf{s}_t,\mathbf{a}_t) - \gamma V_{\bar\psi}(\mathbf{s}_{t+1})\big).\tag{9}
$$

依据：(7) 是对不动点方程 (2)(3)（$V$ 换成参数化 $V_{\bar\psi}$）的平方残差回归；(8) 的目标用**目标网络** $\bar\psi$——实验平滑系数 $\tau$ 的指数滑动平均，抑制"自举追自举"的发散（§4.2 原文；Algorithm 1 末行 $\bar\psi \leftarrow \tau\psi + (1-\tau)\bar\psi$；Table 1：$\tau = 0.005$；附录 E 另有每 1000 梯度步整权复制的 hard 变体）；(9) 依据：对 (7) 求导的链式法则（样本 $\mathbf{s}_{t+1}\sim p$）。

**策略网络（式 (10)）**——KL 投影的可操作化：

$$
J_\pi(\phi) = \mathbb{E}_{\mathbf{s}_t\sim\mathcal{D}}\Big[D_{KL}\Big(\pi_\phi(\cdot \mid \mathbf{s}_t)\ \Big\|\ \frac{\exp(Q_\theta(\mathbf{s}_t, \cdot))}{Z_\theta(\mathbf{s}_t)}\Big)\Big].\tag{10}
$$

依据：4.3 (i) 的展开——最小化 (10) 等价于最小化 $\mathbb{E}_{a\sim\pi_\phi}[\log\pi_\phi - Q_\theta]$（$Z_\theta$ 不含 $\phi$，弃之不影响 $\arg\min$）；每步把策略重新拉向当前 Q 的 Boltzmann 分布，即"改进步的函数逼近版"。

### 4.5 重参数化：路径导数为何低方差（式 (11)(12)(13)）

**式 (11)(12)——重参数化 (reparameterization)**：

$$
\mathbf{a}_t = f_\phi(\epsilon_t; \mathbf{s}_t),\qquad
J_\pi(\phi) = \mathbb{E}_{\mathcal{D},\,\epsilon\sim\mathcal{N}}\big[\log \pi_\phi\big(f_\phi(\epsilon_t; \mathbf{s}_t) \mid \mathbf{s}_t\big) - Q_\theta\big(\mathbf{s}_t,\, f_\phi(\epsilon_t; \mathbf{s}_t)\big)\big].\tag{11,12}
$$

（$\epsilon_t$ 从固定分布如球面高斯采样——式 (11) 下方原文；依据：把"从 $\pi_\phi$ 采样"的随机性全部搬进与 $\phi$ 无关的 $\epsilon$，期望的支撑不变、分布由 $f_\phi$ 的形状再现。）

**式 (13)——策略梯度**：

$$
\nabla_\phi J_\pi(\phi) = \nabla_\phi \log \pi_\phi(\mathbf{a}_t \mid \mathbf{s}_t) + \big(\nabla_{\mathbf{a}_t} \log \pi_\phi(\mathbf{a}_t \mid \mathbf{s}_t) - \nabla_{\mathbf{a}_t} Q(\mathbf{s}_t,\mathbf{a}_t)\big)\,\nabla_\phi f_\phi(\epsilon_t; \mathbf{s}_t),\tag{13}
$$

$\mathbf{a}_t$ 在 $f_\phi(\epsilon_t;\mathbf{s}_t)$ 处取值。**推导**（逐步）：记 $g(\phi, a) = \log\pi_\phi(a \mid \mathbf{s}_t) - Q_\theta(\mathbf{s}_t, a)$，则 $J_\pi = \mathbb{E}_\epsilon\big[g\big(\phi, f_\phi(\epsilon; \mathbf{s}_t)\big)\big]$。对 $\phi$ 求全导数（依据：多元链式法则——$g$ 同时通过显式参数与动作路径依赖 $\phi$）：$\nabla_\phi J = \mathbb{E}_\epsilon\big[\partial_\phi g(\phi,a)\big|_{a} + \nabla_a g(\phi,a)\big|_{a}\, \nabla_\phi f_\phi\big]$；代入 $\partial_\phi g = \nabla_\phi\log\pi_\phi$、$\nabla_a g = \nabla_a\log\pi_\phi - \nabla_a Q$ 即式 (13)。

**与 score-function 梯度的对照推导**（论文 §4.2 式 (11) 前的原文论证 + 标准结论）：

- **似然比率/score-function 估计**（Williams 1992；教程 (8.3)–(8.5) 同款）：$\nabla_\phi\,\mathbb{E}_{a\sim\pi_\phi}[Q(\mathbf{s},a)] = \mathbb{E}_{a\sim\pi_\phi}\big[Q(\mathbf{s},a)\,\nabla_\phi \log \pi_\phi(a \mid \mathbf{s})\big]$（依据：log-导数技巧）。它**不需要**对策略密度网络反向传播——这是它历史上的优点。
- **重参数化/路径导数估计**：$\nabla_\phi\,\mathbb{E}_{\epsilon\sim\mathcal{N}}\big[Q(\mathbf{s}, f_\phi(\epsilon;\mathbf{s}))\big] = \mathbb{E}_\epsilon\big[\nabla_a Q(\mathbf{s},a)\big|_{a=f_\phi}\,\nabla_\phi f_\phi(\epsilon;\mathbf{s})\big]$（依据：链式法则，同上）。
- **两者同为无偏**（依据：都精确等于同一目标的梯度），方差性质不同：score-function 把**整个收益值 $Q$** 作为乘法因子与 score 相乘——估计量的波动含 $Q$ 自身的波动，且需 $Q$ 与 score 的相关在期望意义下才"抵消"出来；路径导数只乘**局部敏感度** $\nabla_a Q$（一个通常有界的导数），不携带收益量级——一般方差显著更低（论文原话 "resulting in a lower variance estimator"；具体差多少与任务相关，此处定性）。
- **SAC 场景的决定性理由**（§4.2 原文）：这里的目标密度 $Q_\theta$ 本身就是**可微神经网络**——弃用似然比率、改走重参数化，把这个梯度用满；且策略是高斯时 $f_\phi$ 显式（$\mu_\phi(\mathbf{s}) + \sigma_\phi(\mathbf{s})\odot\epsilon$）。标量示例：$\pi_\phi = \mathcal{N}(\mu_\phi, 1)$ 时 score 估计为 $Q(a)(a-\mu_\phi)$、重参数化估计为 $Q'(\mu_\phi + \epsilon)$——前者要靠两因子的期望相关抵消，后者只是平均斜率。

**tanh 压缩与对数似然修正（附录 C，式 (20)(21)）**：无界高斯样本需压进有界动作区间。令 $\mathbf{u} \sim \mu(\mathbf{u}\mid\mathbf{s})$（未压缩高斯），$\mathbf{a} = \tanh(\mathbf{u})$（逐元素）。由变量替换公式（依据：概率密度的 Jacobian 变换），

$$
\pi(\mathbf{a} \mid \mathbf{s}) = \mu(\mathbf{u} \mid \mathbf{s})\,\Big|\det\frac{\partial \mathbf{a}}{\partial \mathbf{u}}\Big|^{-1},\qquad
\log \pi(\mathbf{a} \mid \mathbf{s}) = \log \mu(\mathbf{u} \mid \mathbf{s}) - \sum_{i=1}^{D} \log\big(1 - \tanh^2(u_i)\big),\tag{20,21}
$$

依据：$\partial\tanh(u_i)/\partial u_i = 1 - \tanh^2(u_i)$ 为对角 Jacobian，行列式取连乘后取对数。式 (12) 中的 $\log\pi_\phi$ 一律用 (21) 的修正值——否则熵项与 KL 都错。

### 4.6 双 Q 取 min、温度 $\alpha$ 的实况（含版本差异声明）

**双 Q 抑制正向偏差**（§4.2 末段原文；Algorithm 1 的 $i \in \{1,2\}$）：两个 Q 网络**独立**训练（各自 $J_{Q_i}(\theta_i)$），式 (5) 的价值目标与式 (13) 的策略梯度里取 $\min_i Q_{\theta_i}$。机理（论文引 Hasselt 2010、Fujimoto 2018；此处给标准论证）：对围绕真值波动的两个估计，$\mathbb{E}[\max(X_1,X_2)] \ge \max(\mathbb{E}X_1, \mathbb{E}X_2)$（依据：$\max$ 是凸函数，Jensen 不等式）——$Q$ 的正向偏差经软 Bellman 自举会复利放大，且 (4)/(10) 的**指数**会把高估的动作概率推得过尖（策略过早确定化）；取 min 是保守修正。论文实验结论：双 Q "显著加快训练、尤其难任务"（§4.2）。

**温度 $\alpha$ 的实况（重要声明）**：本文 $\alpha$ 是**固定超参**，且按 §3.2 的约定连显式都省略——实践中以**奖励缩放** (reward scale) 的面目出现：Table 2 逐环境调（Hopper/Walker2d/HalfCheetah/Ant 取 5、Humanoid-v1 取 20、Humanoid(rllab) 取 10）；§5.2 原话：奖励缩放是**唯一需要调的超参**，其自然解释是温度的倒数。**任务书所称"dual gradient descent 约束式自动调节 $\alpha$"不在本 PDF 中**（正文与附录 (1)–(21) 均无；经全文核对）——带目标熵约束的自动温度调节出自后续工作 *Soft Actor-Critic Algorithms and Applications*（Haarnoja et al., arXiv:1812.05905）：把 $\alpha$ 写成拉格朗日乘子、对约束优化做对偶梯度上升（教程 §08.3 ⑤ 末句"温度可手调或按目标熵自动调节（延伸）"指的即此延伸）。

## 5. 实验与结果解读（定量处均出自论文原文；图内具体数值定性转述）

**对比评估（§5.1，Figure 1）**。任务：OpenAI Gym 的 Hopper-v1、Walker2d-v1、HalfCheetah-v1、Ant-v1、Humanoid-v1 + rllab 的 Humanoid；基线：DDPG、PPO、SQL，另加同期工作 TD3（作者提供的 DDPG + 双 Q 实现）；每 1000 环境步做一次评估 rollout，实线为均值、阴影为各 trial 的最小-最大回报（种子数原文自述不一：§5.1 同时写 "three different random seeds" 与 "over five trials"，此处按原文并存转述、不取其一）。结论（§5.1 原文归纳）：简单任务上与基线**相当**，难任务（Ant、Humanoid 系）**大幅领先**；DDPG 在 Ant-v1 与两个 Humanoid 上**全无进展**；SAC 显著快于 PPO——论文归因于 PPO 需要大 batch 才能吃下高维复杂任务（on-policy 义务）；SQL 能学所有任务但更慢、渐近性能更差。作者指出成绩超过 Duan 2016 / Gu 2016 报告的先前最优。

**随机 vs 确定性策略（§5.2，Figure 2）**。把 SAC 改成"价值更新确定性 + 固定高斯探索噪声"的 DDPG 近似变体，在 Humanoid(rllab) 上逐种子对比：随机策略的种子间波动**远小于**确定性变体——熵最大化在难调超参的任务上就是稳定器（§5.2 原文 "stabilize training"）。

**三组敏感性消融（§5.2，Figure 3）**：(a) **评估方式**——训练随机、评估取均值动作常常回报更高（随机是为探索，评估时后验取模合理；Figure 3a）；(b) **奖励缩放**——取值过小则策略近乎均匀、无法利用奖励；过大则学得快但迅速确定化、陷局部最优（Figure 3b 五档对照）——即温度敏感性本身；(c) **目标平滑系数 $\tau$**——过大（快跟随）不稳、过小（慢跟随）拖学习，全任务统一用 0.005（Figure 3c）。

**附录 E（Figure 4）**：Trust-PCL 在给定交互步数内多数任务学不出来；SAC 的 hard-update 变体与标准版相当，唯 Humanoid(rllab) 标准版最快。

**关于"多峰采样演示"的核对声明**：任务书提到的"蚂蚁迷宫多峰行为演示**不在本 PDF 中**"——本文对多模态的支撑是 §3.2 的论断（多动作同等吸引时概率质量均分）与 §4.3 的 Boltzmann 推导；蚂蚁迷宫双出口的采样可视化出自后续版本/应用报告（arXiv:1812.05905）。本精读按论文实况撰写。

## 6. 局限与后续影响

1. **温度/奖励缩放是最大痛点**：Figure 3b 显示性能对 reward scale 高度敏感，而本文需要**逐任务手调**（Table 2）——自动温度调节（目标熵约束的对偶梯度上升）是后续版本才补上的（arXiv:1812.05905），此后 SAC 的"只需要一个目标熵超参"才成立。
2. **理论保证限于表格/精确设定**：Lemma 1/2、Theorem 1 假设 $\lvert\mathcal{A}\rvert<\infty$ 与精确值函数；函数逼近下的收敛性无定理（§4.2 只给出实践近似——每步交替而非跑到收敛）。
3. **结构相对重**：本版含独立 $V$ 网络 + 双 Q + 目标网络 + 策略网络（后续版本砍掉 $V$）；部署时随机策略还需决定"采样还是取均值"（Figure 3a：取均值常更好——多一步评估策略）。
4. **样本效率仍逊于模型基**：与第 05 章 PETS 精读引的量化对照（模型基以远少的数据逼近无模型渐近性能）相比，SAC 的 off-policy 复用只是缓解而非消除交互需求。
5. **后续影响**：(a) 自动温度版成为机器人无模型 RL 的事实默认（后续作者将其用于真实机器人操作与行走，arXiv:1812.05905）；(b) 像素输入、多任务、离线设定等方向上大量后续工作以 SAC 为骨架（离线需另加分布偏移约束，如 BCQ/CQL 一脉对 off-policy 值学习加保守正则——属后续领域，本文未涉及）；(c) 最大熵框架与能量模型的联系（§2 引的 soft Q-learning 一脉）由本文的收敛证明补齐了 off-policy actor-critic 缺环。

## 7. 与本项目对照

**教程第 08 章映射表**（编号已核实，§08.3）：

| 本论文内容（原文位置） | 教程第 08 章（编号） |
|---|---|
| 最大熵目标 (1)、温度角色、折扣版 (14) | (8.16)（含"$\alpha$ 是温度、与第 07 章 class-K 记号撞车"的提醒） |
| 软 Bellman 算子与不动点 (2)(3)、Lemma 1 | 软价值/软 Bellman (8.17)、软 Bellman 方程 (8.18) |
| 软策略改进 (4)、Boltzmann 闭式解（4.3 推导） | KL 投影 (8.19)、Boltzmann 分布 (8.20)、$V = \alpha\log Z$ 副产品 |
| Lemma 2（附录 B.2，式 (16)–(19)）、Theorem 1 | §08.3 ⑤"评估 + 改进交替，收敛到 $\Pi$ 内最优"一句 |
| KL 目标 (10)、重参数化 (11)–(13) | "actor 损失 $\mathbb{E}[\alpha\log\pi - Q]$，经重参数化采样可求梯度" |
| 双 Q 取 min、目标网络、$\tau = 0.005$ | §08.3 ⑤ 深度实现三件套 (a)(b)(c) |
| 温度固定、奖励缩放 Table 2；自动调节属后续 | "温度 $\alpha$ 可手调或按目标熵自动调节（延伸）" |

**与 PPO 的选型对照**（对比表见教程 §08.3 ③：数据效率 SAC > PPO、部署复杂度反序、PPO 相对鲁棒而 SAC 对 $\alpha/\tau$ 敏感）：SAC 的优势区间是**交互昂贵**——真机在线适配、replay 复用、训练后保持随机探索；PPO 的优势区间是**仿真便宜**——大规模并行采样摊薄 on-policy 成本。

**为什么腿式控制选 PPO 而非 SAC**（回引[教程第 09 章 §09.2](../09_前沿学习式规划与腿式控制.md)：Rapid Locomotion 的特权教师用 PPO 训练、本文未出现于第 09 章）：(a) GPU 大规模并行仿真下 on-policy 的样本短板失效，PPO 一次批量即百万级转移，SAC 的 replay/目标网络/温度调参三件套反而增加跨大规模并行的工程复杂度；(b) sim-to-real 训练范式主打域随机化 + 单策略前向部署，PPO 结构简单、超参钝感（其 Table 1 的 $\epsilon$ 三档成绩接近），便于在数千环境间同步副本；(c) 腿式任务奖励工程已相当成熟，SAC 的"熵保探索"收益小于其调参成本——故第 09 章一脉（含第 08 章三方对比表的"仿真便宜 → PPO"选型逻辑）默认 PPO。

**跨主题联系**：重参数化 $a_\phi = f_\phi(\epsilon; s)$ 与生成式策略同族——[π0 精读](../../robotwin/精读/Pi0_arXiv2024.md)的流匹配同样以"固定噪声 $\epsilon \to$ 网络变换 $\to$ 输出"产生动作/Token，路径导数（低方差）替代 score-function 的论证在两类模型中同构。

## 配套阅读

- 教程主线：[第 08 章｜学习式控制：强化学习](../08_学习式控制强化学习.md)（§08.3 为本论文的教程化；(8.16)–(8.20) 是本精读 §4.1–4.3 的姊妹推导；§08.3 ③ 对比表是 §7 选型论证的出处）。
- 姊妹精读：[PPO 精读](PPO_arXiv2017.md)（on-policy 对照；本文 §1/§5 的 PPO 基线即 Schulman et al., 2017b）｜ [PETS 精读](PETS_NeurIPS2018.md)（模型基对照：以少 8 倍样本逼近无模型渐近性能，SAC 为其无模型基线之一）。
- 跨主题精读：[π0 精读](../../robotwin/精读/Pi0_arXiv2024.md)（重参数化生成式策略的流匹配版）；部署安全兜底见 [CBF-Survey 精读](CBF-Survey_ECCV2019.md)与教程 (7.9)。
- 本仓库相关论文 PDF：[PPO](../../../papers/control_planning/frontier/arXiv-1707.06347_PPO.pdf)、[PETS](../../../papers/control_planning/frontier/arXiv-1805.12114_PETS.pdf)。
- 论文延伸：soft Q-learning（Haarnoja et al., ICML 2017——最大熵前身）；TD3（Fujimoto et al., 2018——双 Q min 与目标技巧的出处，本文同期工作）；后续版本 *Soft Actor-Critic Algorithms and Applications*（arXiv:1812.05905——自动温度调节、去掉 $V$ 网络、机器人应用与蚂蚁迷宫多峰演示）；官方代码 github.com/haarnoja/sac。
- 版本说明：本精读以 arXiv:1801.01290v2（2018-08-08，14 页）为准——含独立价值网络 (5)、温度固定（正文省略 $\alpha$、以 reward scale 代替）；(1)–(21) 式、Lemma 1/2、Theorem 1、Algorithm 1、Figure 1–4、Table 1–2 均按该版本核对。

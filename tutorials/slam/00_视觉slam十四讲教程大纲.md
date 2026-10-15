# 《视觉 SLAM 十四讲》教程大纲（book 逐讲设计）

> **版本依据**：[book/视觉slam十四讲.pdf](../../book/视觉slam十四讲.pdf)（第一版，2017-3-31 目录，
> 第 9 讲为实践章"设计前端"，后端/回环/建图依次为第 10–13 讲）。与第 2 版讲次对照见 §0.2。
> **写作规范**：每讲正文按 [CONSTRAINTS.md §3.4](../../CONSTRAINTS.md) 五步法展开
> （问题场景 → 解决方法 → 选型理由 → 理论依据 → 完整推导）；本大纲给出每讲的
> 目标、场景锚点、知识点清单、**核心公式的逐步推导链**（标注每步依据）、实践任务、
> 易错点与自检题，作为后续写作的"合同"。
> **编号约定**：公式编号 (讲号.序号)，如 (4.3) = 第 4 讲第 3 式；推导链编号 D<讲号>-<序号>。
> 后续章节与代码以这些编号回引。

---

## 0. 总体设计

### 0.1 定位与读者

- **定位**：以《视觉 SLAM 十四讲》为主线的教材型教程，从零基础到能读懂 ORB-SLAM3 / VINS-Mono 级别的系统论文；与既有 [10 章主题教程](README.md)（概率机器人视角）、[18 篇论文精读](精读/README.md)、[projects/slam 纯 numpy 代码](../../projects/slam/README.md) 互为引用，不重复展开（映射总表见 §0.6）。
- **读者**：有线性代数与概率论基础、无 SLAM 经验的研究生 / 工程师。
- **主线一句话**：机器人无 GPS 时边走边建图、边建图边定位——SLAM 把它写成"带噪声的运动方程 + 观测方程"下的状态估计，全书十四讲就是把这个估计问题**表示**（讲 3–5）、**求解**（讲 6、10–11）、**喂料**（讲 7–8 前端）、**纠偏**（讲 12 回环）、**成图**（讲 13）直到**成系统**（讲 9、14）。

### 0.2 版本说明：第一版 ↔ 第二版讲次对照

本仓库 `book/` 内 PDF 为第一版；既有 [OVERVIEW.md](OVERVIEW.md) 引用的是第 2 版讲次，写作与互链时按下表换算。

| 内容 | 第一版（本 PDF） | 第 2 版（slambook2） |
|------|-----------------|---------------------|
| 前言 / 预备知识 | 第 1 讲 前言 | 第 1 讲 预备知识 |
| 初识 SLAM | 第 2 讲 | 第 2 讲 |
| 三维空间刚体运动 | 第 3 讲 | 第 3 讲 |
| 李群与李代数 | 第 4 讲 | 第 4 讲 |
| 相机与图像 | 第 5 讲 | 第 5 讲 |
| 非线性优化 | 第 6 讲 | 第 6 讲 |
| 视觉里程计 1（特征点法） | 第 7 讲 | 第 7 讲 |
| 视觉里程计 2（直接法） | 第 8 讲 | 第 8 讲 |
| 实践：设计前端 | **第 9 讲**（独立实践章） | 并入第 8 讲末与第 13 讲 |
| 后端 1（KF/EKF + BA 稀疏求解） | **第 10 讲** | 第 9 讲 |
| 后端 2（位姿图 + 因子图） | **第 11 讲** | 第 10 讲 |
| 回环检测 | **第 12 讲** | 第 11 讲 |
| 建图 | **第 13 讲** | 第 12 讲 |
| SLAM：现在与未来 | 第 14 讲 | 第 14 讲 |

### 0.3 四阶段路线与依赖图

```
第一阶段 数学地基（讲1–6）        第二阶段 前端（讲7–9）
┌─────────────────────────┐      ┌──────────────────────────┐
│ 讲1 前言                 │      │ 讲7 VO1 特征点法          │
│ 讲2 初识SLAM（数学表述）  │ ───→ │ 讲8 VO2 直接法            │
│ 讲3 刚体运动 ─→ 讲4 李群  │      │ 讲9 实践：设计前端(工程章) │
│ 讲5 相机与图像           │      └───────────┬──────────────┘
│         讲6 非线性优化 ←─┼── 讲4 的雅可比 + 讲2 的概率表述   │
└─────────────────────────┘                  │
                 ┌───────────────────────────┘
                 ▼
第三阶段 后端与回环（讲10–12）            第四阶段 系统与前沿（讲13–14）
┌──────────────────────────┐            ┌──────────────────────────┐
│ 讲10 后端1 KF/EKF + BA    │            │ 讲13 建图(稠密/八叉树/TSDF)│
│      （Schur 补/鲁棒核）   │ ─────────→ │ 讲14 现在与未来(VIO/语义)  │
│ 讲11 后端2 位姿图/因子图   │── 讲12 回环└──────────────────────────┘
│ 讲12 回环检测 词袋+位姿图  │   修正回接到 讲11
└──────────────────────────┘
```

- **硬依赖**：讲 4 ← 讲 3；讲 6 ← 讲 2+4；讲 7、8 ← 讲 4+5+6；讲 9 ← 讲 7；讲 10 ← 讲 6；讲 11 ← 讲 10+4；讲 12 ← 讲 11；讲 13 ← 讲 7；讲 14 ← 全部。
- **自学最短路径**：讲 2 → 3 → 4 → 5 → 6 → 7（可先跳过 8–9 到 10，再回头）。

### 0.4 课时与验收建议

| 阶段 | 讲次 | 建议课时 | 验收形式 |
|------|------|---------|---------|
| 数学地基 | 1–6 | 6 × 2 学时 | 每讲自检题全对 + 手推讲 4 全部推导链 |
| 前端 | 7–9 | 3 × 2.5 学时 | 跑通 `projects/slam/epipolar`、`direct`；讲 9 阅读代码画数据流图 |
| 后端与回环 | 10–12 | 3 × 2.5 学时 | 手推 Schur 补与位姿图雅可比；跑通 `photoba`、`bowloop` |
| 系统与前沿 | 13–14 | 2 × 1.5 学时 + 大作业 | 大作业：把讲 9 前端 + 讲 10 后端 + 讲 12 回环拼成迷你 VO |

### 0.5 每讲写作模板

每讲正文按五步法组织（见 [CONSTRAINTS.md §3.4](../../CONSTRAINTS.md)）；大纲中每个
"推导 DN-k"即该讲必须完成的第⑤步产出清单。写作硬要求：

1. 推导从已知出发逐步推进，**每步注明依据**（定义 / 公理 / 前一步结论 / 引用定理）；
2. 禁止"易得、显然、读者可自行验证"；
3. 公式编号沿用本大纲 (N.m)，供代码与后续讲回引；
4. 纯操作内容（安装、命令、库用法）注明"操作步骤，不含理论"予以豁免。

### 0.6 讲次 × 本仓库资产映射总表

| 讲 | 主题 | 10 章主题教程 | projects/slam 代码 | 精读论文 |
|----|------|--------------|-------------------|----------|
| 1 | 前言 | ——（导览） | [TUTORIAL.md](../../projects/slam/TUTORIAL.md) 环境导读 | —— |
| 2 | 初识 SLAM | [01](01_SLAM问题定义与全景.md) | —— | SLAM-Survey、FastSLAM |
| 3 | 刚体运动 | [02](02_三维刚体运动旋转与位姿.md) | `core/lie.py` | —— |
| 4 | 李群李代数 | [02](02_三维刚体运动旋转与位姿.md)（雅可比部分 [08](08_后端-ii图优化与-ba.md)） | `core/lie.py`、`vins`、`preint` | —— |
| 5 | 相机与图像 | [04](04_相机模型与特征提取.md) | `core/camera.py` | —— |
| 6 | 非线性优化 | [03](03_概率状态估计基础.md)+[08](08_后端-ii图优化与-ba.md) | `core/solver.py` | —— |
| 7 | VO1 特征点法 | [05](05_视觉里程计-i特征点法.md) | `epipolar/`、`ptam/` | PTAM、ORB-SLAM |
| 8 | VO2 直接法 | [06](06_视觉里程计-ii直接法.md) | `direct/`、`photoba/` | LSD-SLAM、DSO |
| 9 | 设计前端 | [05](05_视觉里程计-i特征点法.md)+[10](10_建图与系统实战.md) | `ptam/`（双线程架构教学代理） | PTAM、ORB-SLAM |
| 10 | 后端 1 | [03](03_概率状态估计基础.md)+[07](07_后端-i滤波与增量估计.md)+[08](08_后端-ii图优化与-ba.md) | `photoba/`（BA）、`fastslam/` | FastSLAM、ORB-SLAM2 |
| 11 | 后端 2 | [08](08_后端-ii图优化与-ba.md) | `vins/`（位姿图）、`bowloop/`（回环校正） | ORB-SLAM3 |
| 12 | 回环检测 | [09](09_回环检测.md) | `bowloop/` | ORB-SLAM（词袋部分） |
| 13 | 建图 | [10](10_建图与系统实战.md)、[3D 重建 04 章](../3d_reconstruction/04_RGB-D融合与TSDF.md) | `droidlite/`（延伸）；深度滤波器暂无模块（列为扩展任务） | LSD-SLAM（深度滤波）、NeRF-SLAM |
| 14 | 现在与未来 | [10 章](10_建图与系统实战.md) 末扩展节 | `preint/`、`vins/`、`loam2d/`、`droidlite/` | IMU-预积分、VINS-Mono、DROID-SLAM、GS 系列 |

---

# 第一阶段｜数学地基（讲 1–6）

## 第 1 讲｜前言

**性质**：导览 + 环境准备，**操作步骤，不含理论**（按 §0.5 第 4 条豁免推导）。

- **知识点清单**（书 1.1–1.4）：本书结构与用法、目标读者、风格约定。
- **大纲要点**：
  - 给出全书地图（本大纲 §0.3 依赖图的文字版），说明"理论章 / 实践章 / 综述章"的区分与读法；
  - 环境清单：Ubuntu + CMake + Eigen/OpenCV/Sophus/Ceres/g2o 安装要点（版本坑列表：g2o 与 Ceres 依赖冲突、OpenCV 版本 API 差异）；
  - 与本仓库代码的关系：书代码库 slambook（书内）+ [slambook2](https://github.com/gaoxiang12/slambook2)（第 2 版）+ 本仓库 `projects/slam/` 纯 numpy 教学实现（不依赖编译环境即可复现算法核心）。
- **自检**：能编译运行 hello_SLAM 与 useEigen；知道每个库在全书哪一讲出场。

## 第 2 讲｜初识 SLAM

- **学习目标**：① 说出经典视觉 SLAM 框架四模块的职责与数据流；② 写出 SLAM 的运动方程与观测方程；③ 说清"定位"与"建图"为何互为前提。
- **问题场景锚点（①）**：小萝卜在无 GPS 的办公楼里徘徊——手头：带噪声的编码器/惯性输入 $u$ 与相机观测 $z$；缺：任何地图；想要：每个时刻自己的位置 + 环境中路标的位置。
- **知识点清单**（书 2.1–2.4）：引子；四模块（视觉里程计 / 后端优化 / 回环检测 / 建图）；数学表述；编程基础（豁免）。

### 核心公式与逐步推导

**推导 D2-1：SLAM 问题的数学表述**（书 2.3，回引 [01 章](01_SLAM问题定义与全景.md) §1）

前提：时刻 $k=1..N$，机器人位姿记 $x_k$（平面 $[x,y,\theta]^\top$，空间用变换矩阵 $T_k$）；环境路标 $\mathbf{y}_1..\mathbf{y}_M$。

1. 状态定义：把"定位 + 建图"合并成**联合状态** $\{\,x_{1:N},\, \mathbf{y}_{1:M}\,\}$——两者都不知道，都要估（依据：问题定义，鸡生蛋须联立）。
2. 运动方程：
$$x_k = f(x_{k-1}, u_k) + w_k \tag{2.1}$$
   （依据：**马尔可夫假设**——给定上一状态与输入，当前状态与更早历史独立；$u_k$ 为运动输入读数，$w_k \sim \mathcal{N}(0, R_k)$ 为运动噪声。）
3. 观测方程：机器人在 $x_k$ 处看到第 $j$ 个路标：
$$z_{k,j} = h(\mathbf{y}_j, x_k) + \mathbf{v}_{k,j} \tag{2.2}$$
   （依据：观测只取决于"谁在看、看什么"两个量；$\mathbf{v}_{k,j} \sim \mathcal{N}(0, Q_{k,j})$ 为观测噪声。）
4. SLAM = 已知带噪数据 $\{u\}, \{z\}$，求后验 $P(x_{1:N}, \mathbf{y}_{1:M} \mid u, z)$ 的最优估计（依据： $x, y$ 均为随机变量，噪声存在 ⇒ 只能谈概率分布）。
5. 留钩子：直接求后验难以解析 → 第 6 讲在**高斯假设 + 最大后验（MAP）**下把它变成最小二乘 (6.2)；无运动方程的纯视觉特例（$u$ 不可得）就是视觉里程计，由帧间观测差分代替 $f$。

### 实践 / 易错点 / 自检

- **实践**：Linux + CMake + hello_SLAM + useEigen（操作豁免）。
- **易错点**：把 $w_k$ 当"可以忽略的扰动"——它是后端存在的根本原因；混淆"里程计（局部、漂移）"与"SLAM（全局、回环纠偏）"。
- **自检**：① 四模块各解决 (2.1)(2.2) 中的哪部分、数据流向是什么？② 为什么视觉 SLAM 常缺 (2.1)？后果是什么？
- **配套**：[01 章](01_SLAM问题定义与全景.md)；精读 [SLAM-Survey](精读/SLAM-Survey_TRO2016.md)。

## 第 3 讲｜三维空间刚体运动

- **学习目标**：① 推导旋转矩阵的来源与 SO(3) 定义；② 用齐次坐标统一旋转平移；③ 掌握旋转向量 / 欧拉角 / 四元数三种表示及互转公式；④ 解释万向锁与旋转不可交换的结构性原因。
- **问题场景锚点（①）**：相机"先绕 x 转 90° 再绕 y 转 90°"与"先 y 后 x"结果不同——旋转**不可交换**；欧拉角在某些姿态丢一个自由度。手头：需要无歧义、可计算的旋转描述；缺：数学工具。
- **知识点清单**（书 3.1–3.7）：旋转矩阵；Eigen 实践（豁免）；旋转向量与欧拉角；四元数；*相似/仿射/射影变换；Eigen 几何模块（豁免）；可视化（豁免）。

### 核心公式与逐步推导

**推导 D3-1：旋转矩阵与 SO(3)**（书 3.1.1–3.1.2，回引 [02 章](02_三维刚体运动旋转与位姿.md) §02.2）

1. 刚体变换定义：保持任意两点间距离与内积的空间变换（依据：刚体定义）。
2. 其线性部分 $R$ 保内积：$(R\mathbf{a})\cdot(R\mathbf{b}) = \mathbf{a}\cdot\mathbf{b}\ \forall \mathbf{a},\mathbf{b} \Rightarrow \mathbf{a}^\top(R^\top R - I)\mathbf{b} = 0\ \forall\mathbf{a},\mathbf{b} \Rightarrow R^\top R = I$（依据：二次型恒为零则矩阵为零）。
3. $\det(R^\top R) = \det^2 R = 1 \Rightarrow \det R = \pm 1$；物理旋转可从恒等变换连续变化而来且 $\det I = 1$，而 $\det = -1$ 的反射分支与 $I$ 不连通（依据：正交群 $O(3)$ 有两个连通分支）⇒ $\det R = +1$。
4. 特殊正交群：
$$SO(3) = \{\, R \in \mathbb{R}^{3\times3} \mid R^\top R = I,\ \det R = 1 \,\} \tag{3.1}$$
5. 点变换：$\mathbf{a}' = R\mathbf{a} + \mathbf{t}$（依据：刚体变换 = 线性部分 + 平移；**次序固定**——先旋转后平移）。

**推导 D3-2：齐次坐标与 SE(3)**（书 3.1.3）

1. 动机：$\mathbf{a}' = R\mathbf{a}+\mathbf{t}$ 复合两次即膨胀为 $R_2(R_1\mathbf{a}+\mathbf{t}_1)+\mathbf{t}_2$，不便统一。
2. 提升到齐次坐标 $[\mathbf{a}^\top, 1]^\top$，则：
$$T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}, \qquad \begin{bmatrix}\mathbf{a}'\\1\end{bmatrix} = T\begin{bmatrix}\mathbf{a}\\1\end{bmatrix} \tag{3.2}$$
   （依据：分块乘法逐项展开验证：上行 = $R\mathbf{a} + \mathbf{t}\cdot 1$，下行 = 1。）
3. 复合闭合：$T_1 T_2$ 仍具同构形式，且逆 $T^{-1}$ 亦然 ⇒ $SE(3)$ 对乘法成群（依据：分块乘法逐条验证群四公理）。

**推导 D3-3：旋转向量（轴角）与 Rodrigues 公式**（书 3.3.1）

1. 欧拉旋转定理：任一三维旋转存在不动轴 $\mathbf{n}$（单位向量）：$R\mathbf{n} = \mathbf{n}$（依据：实旋转的特征值集合 $\{1, e^{\pm i\theta}\}$ 中必有实特征值 1）。
2. 转角与轴二参即可表示旋转：$(\theta, \mathbf{n})$，向量形式 $\boldsymbol{\phi} = \theta\mathbf{n}$（旋转向量）。
3. 正向公式（Rodrigues，讲 4 用幂级数重推，此处直接引用 + 验证）：
$$R = \cos\theta\, I + (1-\cos\theta)\,\mathbf{n}\mathbf{n}^\top + \sin\theta\,\mathbf{n}^{\wedge} \tag{3.3}$$
   验证不动轴：$R\mathbf{n} = \cos\theta\,\mathbf{n} + (1-\cos\theta)\mathbf{n}(\mathbf{n}^\top\mathbf{n}) + \sin\theta\,(\mathbf{n}\times\mathbf{n}) = \mathbf{n}$（依据：$\mathbf{n}^\wedge\mathbf{n} = \mathbf{n}\times\mathbf{n} = \mathbf{0}$，叉积矩阵记号 $\mathbf{n}^\wedge$ 见 (3.5) 步）。
4. 反向公式：对 (3.3) 取迹（$\operatorname{tr}(\mathbf{n}\mathbf{n}^\top) = \mathbf{n}^\top\mathbf{n} = 1$，$\operatorname{tr}(\mathbf{n}^\wedge) = 0$，依据：迹的循环性与反对称矩阵迹为零）：
$$\operatorname{tr}(R) = 1 + 2\cos\theta \ \Rightarrow\ \theta = \arccos\!\Big(\tfrac{\operatorname{tr}(R)-1}{2}\Big) \tag{3.4}$$
   轴由反对称部分提取（依据：$R - R^\top = 2\sin\theta\,\mathbf{n}^\wedge$，由 (3.3) 对称/反对称分项直接相减）：
$$\mathbf{n} = \tfrac{1}{2\sin\theta}\,(R - R^\top)^{\vee} \tag{3.5}$$

**推导 D3-4：欧拉角与万向锁**（书 3.3.2）

1. 约定 Z-Y-X 顺序（yaw–pitch–roll）：$R = R_z(\psi)\,R_y(\theta)\,R_x(\phi)$（依据：约定，务必在教程中显式声明）。
2. 万向锁推导：取 $\theta = +90°$（$\cos\theta = 0,\ \sin\theta = 1$），逐矩阵代入展开（依据：分块矩阵乘法 + 三角恒等式 $\sin\phi\cos\psi \pm \cos\phi\sin\psi = \sin(\phi\pm\psi)$）：
$$R = R_z(\psi)\,R_y(90°)\,R_x(\phi) = \begin{bmatrix} 0 & \sin(\phi-\psi) & \cos(\phi-\psi) \\ 0 & \cos(\phi-\psi) & -\sin(\phi-\psi) \\ -1 & 0 & 0 \end{bmatrix}$$
   所有元素只含 $\phi - \psi$：$\phi$ 与 $\psi$ 中有一个是冗余的——**三个参数只控制了两个自由度**，这就是丢失自由度的代数本质。
3. 结论：任何 3 参数旋转表示必有奇点（后续第 4 讲给出"全局光滑无奇异的 3 参数化不存在"的理论说明）。

**推导 D3-5：四元数**（书 3.4）

1. 定义（Hamilton）：$\mathbf{q} = q_0 + q_1 i + q_2 j + q_3 k = [s, \mathbf{v}]$，乘法表 $i^2 = j^2 = k^2 = -1$，$ij = k,\ jk = i,\ ki = j$。
2. 乘法公式（依据：按乘法表逐项展开并按实部/虚部归组）：
$$\mathbf{q}\otimes\mathbf{p} = \big[\, s_q s_p - \mathbf{v}_q\cdot\mathbf{v}_p,\ \ s_q\mathbf{v}_p + s_p\mathbf{v}_q + \mathbf{v}_q\times\mathbf{v}_p \,\big] \tag{3.6}$$
3. 旋转作用：单位四元数 $\|\mathbf{q}\| = 1$ 对点 $\mathbf{p}$（视为纯虚四元数 $[0,\mathbf{p}]$）：
$$\mathbf{p}' = \mathbf{q}\,\mathbf{p}\,\mathbf{q}^{-1} \tag{3.7}$$
   合法性检查：$\|\mathbf{p}'\| = \|\mathbf{q}\|\,\|\mathbf{p}\|\,\|\mathbf{q}^{-1}\| = \|\mathbf{p}\|$（依据：范数乘性）——模长不变，是旋转（不是缩放）。
4. 四元数 → 旋转矩阵：把 (3.7) 按分量展开（依据：用 (3.6) 两次结合律展开 $\mathbf{q}[0,\mathbf{p}]\mathbf{q}^*$，机械但无跳步——写作时给出 4×4 网格的完整展开表）：
$$R(\mathbf{q}) = (s^2 - \mathbf{v}^\top\mathbf{v})I + 2\,\mathbf{v}\mathbf{v}^\top + 2s\,\mathbf{v}^{\wedge} \tag{3.8}$$
5. 旋转向量 ↔ 四元数：对比 (3.3) 与 (3.8) 得 $\mathbf{q} = [\cos\tfrac{\theta}{2},\ \mathbf{n}\sin\tfrac{\theta}{2}]$（依据：代入验证 $R(\mathbf{q}) = (3.3)$）。**半角** $\theta/2$ ⇒ $\mathbf{q}$ 与 $-\mathbf{q}$ 表示同一旋转（双倍覆盖）——写作时用 $\theta = 360°$ 特例说明。
6. 三种表示对照表（写作产出）：旋转矩阵（9 参冗余、无奇点、不直观）/ 旋转向量（3 参紧凑、有 ±π 奇点）/ 欧拉角（直观、万向锁）/ 四元数（4 参紧凑、插值好、双倍覆盖）。

***3.5 相似/仿射/射影变换**：表格呈现（名称 / 矩阵形式 / 自由度 / 典型用途：相似变换—SLAM 尺度漂移、仿射—单应退化、射影—相机投影本质），无推导。

### 实践 / 易错点 / 自检

- **实践**：Eigen `AngleAxis/Quaternion/Matrix4d` 几何模块练习（操作豁免）；可视化轨迹（豁免）。对应 `core/lie.py` 的旋转工具层。
- **易错点**：四元数左右乘次序（$\mathbf{q}\otimes\mathbf{p}\otimes\mathbf{q}^{-1}$ 不可写成对称）；$-\mathbf{q}$ 与 $\mathbf{q}$ 混用导致插值绕远路；Eigen 中 `rotate` 与 `translate` 的复合次序。
- **自检**：① 手推 (3.4)(3.5)；② 用 (3.6) 验证四元数乘法结合律；③ 举例说明"先旋转后平移"与"先平移后旋转"结果不同。
- **配套**：[02 章](02_三维刚体运动旋转与位姿.md)（更深入的连通性论证）；`core/lie.py`。

## 第 4 讲｜李群与李代数（全书数学核心）

- **学习目标**：① 说清"为什么优化旋转必须进李代数"；② 独立推导 SO(3)/SE(3) 指数映射；③ 推导 BCH 一阶近似与左/右雅可比；④ 掌握扰动模型雅可比 (4.9)(4.10)——**后续第 7/8/10/11 讲所有雅可比的源头**。
- **问题场景锚点（①）**：优化的对象是 $R$，但 $R$ 不构成向量空间——$R + \Delta$ 一般不再是旋转（$R^\top R = I$ 是非线性约束）。手头：要对着旋转做梯度下降；缺：把"旋转上的增量"变成"向量上的增量"的通道。
- **知识点清单**（书 4.1–4.6）：李群李代数基础；指数与对数映射；BCH 与扰动模型；Sophus 实践（豁免）；*相似变换群。

### 核心公式与逐步推导

**推导 D4-1：群、李群、李代数与 so(3)/se(3)**（书 4.1）

1. 群公理：封闭 + 结合 + 幺元 + 逆；$SO(3)$、$SE(3)$ 对矩阵乘法成群（依据：逐条验证；连同比较加法不封闭——$R_1 + R_2$ 不正交）。
2. 李群 = 具光滑流形结构的群；其李代数 = **单位元处的切空间** + 李括号——切空间是线性空间，这就是"进李代数"的合法性来源（依据：李群–李代数对应定理，引用 Varadarajan / 书 4.1.2）。
3. 李代数公理：双线性、自反性 $[X,X]=0$、Jacobi 恒等式 $[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0$。
4. so(3)：三维向量 $\boldsymbol{\phi}$ 经 $\wedge$ 映射到反对称矩阵；括号的叉积本质：
$$\boldsymbol{\phi}^{\wedge}\boldsymbol{\psi}^{\wedge} - \boldsymbol{\psi}^{\wedge}\boldsymbol{\phi}^{\wedge} = (\boldsymbol{\phi}\times\boldsymbol{\psi})^{\wedge} \tag{4.1}$$
   （推导：两边同作用在任意 $\mathbf{c}$ 上——左边 = $\boldsymbol{\phi}\times(\boldsymbol{\psi}\times\mathbf{c}) - \boldsymbol{\psi}\times(\boldsymbol{\phi}\times\mathbf{c}) = (\boldsymbol{\phi}\times\boldsymbol{\psi})\times\mathbf{c}$，依据：**二重叉积展开公式** $\mathbf{a}\times(\mathbf{b}\times\mathbf{c}) = \mathbf{b}(\mathbf{a}^\top\mathbf{c}) - \mathbf{c}(\mathbf{a}^\top\mathbf{b})$；由 $\mathbf{c}$ 任意 ⇒ 两矩阵相等。）
5. se(3)：$\boldsymbol{\xi} = [\boldsymbol{\rho}^\top, \boldsymbol{\phi}^\top]^\top \in \mathbb{R}^6$，$\boldsymbol{\xi}^{\wedge} = \begin{bmatrix}\boldsymbol{\phi}^\wedge & \boldsymbol{\rho}\\ \mathbf{0}^\top & 0\end{bmatrix} \in \mathbb{R}^{4\times4}$（依据：与 (3.2) 同构、保证 $\wedge$ 后仍在 $se(3)$ 内——平移部分放右上块）。

**推导 D4-2：SO(3) 上的指数映射（重推 Rodrigues）**（书 4.2.1）

1. $R = \exp(\boldsymbol{\phi}^{\wedge}) = \sum_{k=0}^{\infty} \frac{(\boldsymbol{\phi}^{\wedge})^k}{k!}$（依据：矩阵指数定义）。
2. 归一化 $\boldsymbol{\phi} = \theta\mathbf{n},\ \|\mathbf{n}\| = 1$。幂关系：
$$\mathbf{n}^{\wedge 2} = \mathbf{n}\mathbf{n}^\top - I, \qquad \mathbf{n}^{\wedge 3} = -\mathbf{n}^{\wedge} \tag{4.2}$$
   （推导：$\mathbf{n}^{\wedge 2}$ 的 $(i,j)$ 元 = $\mathbf{n}_i\mathbf{n}_j - \delta_{ij}$，即 $\mathbf{n}\mathbf{n}^\top - I$；$\mathbf{n}^{\wedge 3} = \mathbf{n}^{\wedge}(\mathbf{n}\mathbf{n}^\top - I) = (\mathbf{n}^{\wedge}\mathbf{n})\mathbf{n}^\top - \mathbf{n}^{\wedge} = -\mathbf{n}^{\wedge}$，依据：$\mathbf{n}^{\wedge}\mathbf{n} = \mathbf{n}\times\mathbf{n} = \mathbf{0}$。）
3. 由 (4.2) 归纳：偶次幂 $\mathbf{n}^{\wedge 2m} = (-1)^m(I - \mathbf{n}\mathbf{n}^\top)$，奇次幂 $\mathbf{n}^{\wedge(2m+1)} = (-1)^m\mathbf{n}^{\wedge}$（依据：对 $m$ 归纳，每步乘 $\mathbf{n}^{\wedge 2}$）。
4. 级数按奇偶重组，对号入座 $\cos\theta$、$\sin\theta$ 的 Taylor 级数：
$$R = \cos\theta\, I + (1-\cos\theta)\,\mathbf{n}\mathbf{n}^\top + \sin\theta\,\mathbf{n}^{\wedge} \tag{4.3}$$
   ——与 (3.3) 一致，且此式在 $SO(3)$ 内（$R^\top R = I$ 可由 (4.3) 两行验证，写作时补）。**指数映射把李代数（向量）送回李群（旋转）**。
5. 对数映射（反向）：直接用 (3.4)(3.5)（依据：同一公式的反解）。

**推导 D4-3：SE(3) 上的指数映射**（书 4.2.2）

1. 归纳求幂（依据：对 $k$ 归纳 + 分块乘法）：
$$\big(\boldsymbol{\xi}^{\wedge}\big)^k = \begin{bmatrix} (\boldsymbol{\phi}^{\wedge})^k & (\boldsymbol{\phi}^{\wedge})^{k-1}\boldsymbol{\rho} \\ \mathbf{0}^\top & 0 \end{bmatrix}, \quad k \ge 1 \tag{4.4}$$
2. 对 $k$ 求和：左上块 = $\exp(\boldsymbol{\phi}^{\wedge})$（即 D4-2）；右上块 = $\sum_k \frac{(\boldsymbol{\phi}^{\wedge})^k}{(k+1)!}\boldsymbol{\rho} = J(\boldsymbol{\phi})\,\boldsymbol{\rho}$，其中再次奇偶分离（依据：D4-2 第 3 步同法）：
$$J(\boldsymbol{\phi}) = \frac{\sin\theta}{\theta} I + \Big(1 - \frac{\sin\theta}{\theta}\Big)\mathbf{n}\mathbf{n}^\top + \frac{1-\cos\theta}{\theta}\,\mathbf{n}^{\wedge} \tag{4.5}$$
3. 结论：$\exp(\boldsymbol{\xi}^{\wedge}) = \begin{bmatrix}\exp(\boldsymbol{\phi}^{\wedge}) & J\boldsymbol{\rho}\\ \mathbf{0}^\top & 1\end{bmatrix} = T$；**平移必须经过 $J$ 加权**——写作时强调这是新手常漏的因子。

**推导 D4-4：BCH 近似与左/右雅可比**（书 4.3.1）

1. 问题：两个旋转相乘，李代数**不能直接相加**（旋转不可交换）；$\ln\big(\exp(A)\exp(B)\big)$ 的展开叫 BCH 公式。
2. 一阶近似（对 $\Delta\boldsymbol{\phi}$ 保留一阶；依据：书 4.3.1 微分方程法——设 $F(t) = \ln(\exp(t\boldsymbol{\Delta}^{\wedge})\exp(t\boldsymbol{\phi}^{\wedge}))$，求 $F'(t) = \boldsymbol{\Delta}^{\wedge} + \mathrm{ad}_{F}(\boldsymbol{\phi}^{\wedge}) + \tfrac12\mathrm{ad}_{F}^2(\boldsymbol{\phi}^{\wedge}) + \cdots$ 后取一阶解；写作时把 $\mathrm{ad}$ 逐步代入）：
$$\exp(\Delta\boldsymbol{\phi}^{\wedge})\,\exp(\boldsymbol{\phi}^{\wedge}) \approx \exp\Big(\big(\boldsymbol{\phi} + J_l(\boldsymbol{\phi})^{-1}\Delta\boldsymbol{\phi}\big)^{\wedge}\Big) \tag{4.6}$$
3. 左雅可比（与 (4.5) 同形——这非巧合，$J$ 即 $J_l$，写作时点明同源性）：
$$J_l(\boldsymbol{\phi}) = \frac{\sin\theta}{\theta} I + \Big(1 - \frac{\sin\theta}{\theta}\Big)\mathbf{n}\mathbf{n}^\top + \frac{1-\cos\theta}{\theta}\,\mathbf{n}^{\wedge}, \qquad J_r(\boldsymbol{\phi}) = J_l(-\boldsymbol{\phi}) \tag{4.7}$$
4. 小角度极限：$\frac{\sin\theta}{\theta} = 1 - \frac{\theta^2}{6} + \cdots$，$1 - \frac{\sin\theta}{\theta} = O(\theta^2)$，$\frac{1-\cos\theta}{\theta^2} = \frac12 - \frac{\theta^2}{24} + \cdots$（依据：Taylor 展开）⇒
$$J_l \approx I + \tfrac12\boldsymbol{\phi}^{\wedge} \tag{4.8}$$
   ——$\boldsymbol{\phi}$ 较小时"加扰动 ≈ 李代数上直接加一半修正"。

**推导 D4-5：李代数求导 → 扰动模型（本讲落脚点）**（书 4.3.2–4.3.4）

1. 目标：$\dfrac{\partial (R\mathbf{p})}{\partial \boldsymbol{\phi}}$。障碍：$\boldsymbol{\phi}$ 在流形上，无全局线性坐标。
2. **求导模型**：给 $\boldsymbol{\phi}$ 加 $\Delta\boldsymbol{\phi}$。用 (4.6) 换序（令右端为 $\boldsymbol{\phi} + \Delta\boldsymbol{\phi}$，反解出左因子 = $\exp\big((J_l\Delta\boldsymbol{\phi})^{\wedge}\big)$，依据：(4.6) 移项）：
$$\exp\big((\boldsymbol{\phi}+\Delta\boldsymbol{\phi})^{\wedge}\big)\mathbf{p} = \exp\big((J_l\Delta\boldsymbol{\phi})^{\wedge}\big)\,R\mathbf{p} \approx R\mathbf{p} + (J_l\Delta\boldsymbol{\phi})\times(R\mathbf{p}) = R\mathbf{p} - (R\mathbf{p})^{\wedge}\,J_l\,\Delta\boldsymbol{\phi} \tag{4.9}$$
   （依据：一阶 $\exp(\boldsymbol{\varepsilon}^{\wedge}) \approx I + \boldsymbol{\varepsilon}^{\wedge}$；$\mathbf{a}\times\mathbf{b} = -\mathbf{b}^{\wedge}\mathbf{a}$。）⇒ $\partial(R\mathbf{p})/\partial\boldsymbol{\phi} = -(R\mathbf{p})^{\wedge} J_l(\boldsymbol{\phi})$——多出 $J_l$ 因子，形式不便。
3. **扰动模型（左乘）**——换个思路，把扰动 $\delta\boldsymbol{\phi}$ 定义在李群左侧：
$$R' = \exp(\delta\boldsymbol{\phi}^{\wedge})\,R \ \Rightarrow\ R'\mathbf{p} = (I + \delta\boldsymbol{\phi}^{\wedge})R\mathbf{p} = R\mathbf{p} + \delta\boldsymbol{\phi}\times(R\mathbf{p}) = R\mathbf{p} - (R\mathbf{p})^{\wedge}\,\delta\boldsymbol{\phi} \tag{4.10}$$
   $$\boxed{\ \frac{\partial\,(R\mathbf{p})}{\partial\,\delta\boldsymbol{\phi}} = -(R\mathbf{p})^{\wedge}\ }$$
   （依据：同上一步；**无 $J_l$ 因子**——这是优化中统一采用的模型；对"点坐标"的导数左右形式相同，差异出现在对位姿自身的增量求导，需伴随矩阵，见 [08 章](08_后端-ii图优化与-ba.md)。）
4. **SE(3) 扰动模型**：$T' = \exp(\delta\boldsymbol{\xi}^{\wedge})\,T$，$\delta\boldsymbol{\xi} = [\delta\boldsymbol{\rho}^\top, \delta\boldsymbol{\phi}^\top]^\top$，作用在点 $\mathbf{q} = R\mathbf{p} + \mathbf{t}$：
$$T'\mathbf{p} = \mathbf{q} + \delta\boldsymbol{\rho} + \delta\boldsymbol{\phi}\times\mathbf{q} = T\mathbf{p} + \delta\boldsymbol{\rho} - \mathbf{q}^{\wedge}\,\delta\boldsymbol{\phi} \ \Rightarrow\ \boxed{\ \frac{\partial\,(T\mathbf{p})}{\partial\,\delta\boldsymbol{\xi}} = \Big[\, I \ \Big|\ \ -\,(T\mathbf{p})^{\wedge}\ \Big]\ } \tag{4.11}$$
   （依据：分块展开——$\delta\boldsymbol{\rho}$ 平移点、$\delta\boldsymbol{\phi}$ 绕原点旋转点，各得一个块列。）
5. **用途锚点（写作必须回引表）**：讲 7 重投影误差雅可比、讲 8 直接法雅可比、讲 10 BA 雅可比、讲 11 位姿图雅可比，全部由 (4.10)(4.11) 出发。

### 实践 / 易错点 / 自检

- **实践**：Sophus 做 rotation/update/扰动数值验证（操作豁免）；对应 `core/lie.py`、`vins` 中 SE(3) 工具。
- **易错点**：$J_l$ 与 $J_r$ 用混（左扰动配 $J_l$）；忘记 SE(3) 平移的 $J$ 因子；把 (4.9) 的 $J_l$ 也带进扰动模型。
- **自检**：① 从 (4.2) 推到 (4.3) 不看书写全；② 推 (4.6) 的小角度版 $\boldsymbol{\phi}\oplus\Delta\boldsymbol{\phi} \approx \boldsymbol{\phi} + J_l^{-1}\Delta\boldsymbol{\phi}$；③ 说明为何 (4.10) 比 (4.9) 更适合做优化变量。
- **配套**：[02 章](02_三维刚体运动旋转与位姿.md)（指数映射/扰动模型的另一条叙述线）、[08 章](08_后端-ii图优化与-ba.md)（伴随与完整雅可比表）。

## 第 5 讲｜相机与图像

- **学习目标**：① 逐步推出针孔模型与内参矩阵；② 写出畸变模型并说明"去畸变发生在归一化平面"；③ 推导双目视差–深度关系；④ 理解 RGB-D 测距原理。
- **问题场景锚点（①）**：3D 世界点如何变成图像里的一个像素？（讲 2 观测方程 (2.2) 的 $h$ 就是它）；反过来只有像素，深度丢了——单目尺度的根源。
- **知识点清单**（书 5.1–5.4）：针孔相机模型；畸变；双目相机；RGB-D；图像表示；OpenCV 实践与拼接点云（操作豁免）。

### 核心公式与逐步推导

**推导 D5-1：针孔模型与内参矩阵**（书 5.1.1，回引 [04 章](04_相机模型与特征提取.md)）

1. 世界 → 相机系（外参）：$\mathbf{P}_c = R\,\mathbf{P}_w + \mathbf{t}$（依据：D3 刚体变换；$R, \mathbf{t}$ 为相机外参）。
2. 相机系 → 归一化平面（依据：过光心连线与 $Z=1$ 平面的**相似三角形**）：
$$\mathbf{x}_{nor} = \Big[\tfrac{X}{Z},\ \tfrac{Y}{Z}\Big]^\top \tag{5.1}$$
3. 归一化平面 → 像素平面（依据：焦距 $f$ 等比缩放 + 主点 $c_x, c_y$ 平移；$f$ 单位为像素/米）：
$$u = f_x \tfrac{X}{Z} + c_x, \qquad v = f_y \tfrac{Y}{Z} + c_y \tag{5.2}$$
4. 齐次合并（依据：用齐次坐标吸收 $1/Z$）：
$$\mathbf{P}_{uv} = \frac{1}{Z}\, K\, \mathbf{P}_c, \qquad K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \tag{5.3}$$
5. 组合全链：$\mathbf{P}_{uv} = \tfrac{1}{Z}\,K\,(R\mathbf{P}_w + \mathbf{t})$——**这就是 (2.2) 中 $h(\cdot)$ 的显式形式**（$Z$ 取 $\mathbf{P}_c$ 第三个分量）。讨论：$f_x = f\,s_x$（$s_x$ 为像素密度）；主点不必在图像中心。

**推导 D5-2：畸变模型**（书 5.1.2）

1. 径向畸变（透镜形状所致，关于光轴对称）——建模在**归一化平面**上，记 $r^2 = x^2 + y^2$：
$$x_{d} = x\,(1 + k_1 r^2 + k_2 r^4), \qquad y_{d} = y\,(1 + k_1 r^2 + k_2 r^4) \tag{5.4}$$
   （依据：对称性 ⇒ 只含 $r$ 的偶次项；多项式取前几阶近似。）
2. 切向畸变（透镜装配不平行所致，薄棱镜模型）：
$$x_{d} = x + 2p_1 xy + p_2(r^2 + 2x^2), \qquad y_{d} = y + p_1(r^2 + 2y^2) + 2p_2 xy \tag{5.5}$$
3. 去畸变流程：像素 → (5.3) 逆变换回归一化平面 → 解 (5.4)(5.5) 的反问题（迭代，纯径向时有闭式）→ 再投影像素。强调：**畸变与去畸变都在归一化平面进行**，之后再乘 $K$。
4. 讨论：鱼眼相机超出多项式适用范围 → 等距/等立体角模型（点到为止，引 OpenCV fisheye）。

**推导 D5-3：双目相机模型**（书 5.1.3）

1. 配置：两相机水平基线 $b$，光轴平行，右相机原点在左相机 $(x,-,-)$ 方向偏 $b$。
2. 左右像素横坐标：$u_L = f\tfrac{X}{Z} + c_x$，$u_R = f\tfrac{X - b}{Z} + c_x$（依据：D5-1 第 2 步，右相机系下 $X$ 坐标为 $X - b$）。
3. 视差与深度（两式相减，依据：代数）：
$$d = u_L - u_R = \frac{f\,b}{Z} \quad\Longrightarrow\quad Z = \frac{f\,b}{d} \tag{5.6}$$
4. 结论：深度与视差成反比——**近处视差大精度高，远处视差趋零深度不可测**（写作时画 $Z$–$d$ 曲线）；这解释了讲 13 单目稠密重建在远处为何崩溃。

**RGB-D（5.1.4）**：结构光（主动投射图案，依据三角测距）与 ToF（光飞行时间）原理，概念介绍，无公式推导（物理测距器件）。

### 实践 / 易错点 / 自检

- **实践**：OpenCV 读写图像、遍历像素、图像去畸变；拼接 RGB-D 点云为 PCD（操作豁免）。对应 `core/camera.py`。
- **易错点**：$K$ 与外参顺序（先外参后内参）；畸变参数作用于归一化坐标而非像素；$f_x \neq f_y$ 时把 $f$ 当标量。
- **自检**：① 从 (5.1)(5.2) 合成 (5.3) 并说明齐次坐标的作用；② 双目 $b$ 加倍、$f$ 加倍对深度精度的影响是否相同？③ 为什么去畸变不能在像素平面直接做？
- **配套**：[04 章](04_相机模型与特征提取.md)；[3D 重建 03 章](../3d_reconstruction/03_多视图立体重建MVS.md)（双目/立体匹配延伸）。

## 第 6 讲｜非线性优化（理论核心枢纽）

- **学习目标**：① 从概率表述一步步推出最小二乘；② 推导 GD/Newton/GN/LM 四种迭代法的更新公式并比较；③ 理解图优化的顶点–边–信息矩阵结构（为讲 10/11 铺路）。
- **问题场景锚点（①）**：讲 2 留下的钩子——$P(x,y\mid u,z)$ 无解析解。手头：带噪声的方程组；缺：把"概率最优"变成"可数值求解"的桥。
- **知识点清单**（书 6.1–6.5）：状态估计问题（MAP→最小二乘）；非线性最小二乘（一阶/二阶梯度、GN、LM）；Ceres 实践；g2o 与图优化理论。

### 核心公式与逐步推导

**推导 D6-1：从 MAP 到最小二乘**（书 6.1，回引 [03 章](03_概率状态估计基础.md)）

1. 后验（依据：贝叶斯定理）：
$$P(x, \mathbf{y} \mid z, u) \propto P(x, \mathbf{y})\,P(z, u \mid x, \mathbf{y}) \tag{6.1}$$
2. 马尔可夫假设下条件独立分解（依据：链式法则 + 各时刻只依赖相邻状态/被观测路标）：
$$P(x,\mathbf{y}\mid z,u) \propto \prod_{k} P(u_k \mid x_{k-1}) \prod_{k}\prod_{j} P(z_{k,j} \mid x_k, \mathbf{y}_j)$$
3. 高斯噪声代入（依据：第 2 讲噪声假设）：$P(u_k\mid x_{k-1}) = \mathcal{N}\big(f(x_{k-1},u_k),\, R_k\big)$，$P(z_{k,j}\mid x_k,\mathbf{y}_j) = \mathcal{N}\big(h(\mathbf{y}_j, x_k),\, Q_{k,j}\big)$。
4. MAP = 最大化上式 ⇔ 最小化负对数（依据：$\log$ 单调；高斯密度指数展开 $\exp\big(-\tfrac12\|\mathbf{e}\|^2_{\Sigma^{-1}}\big)$；与 $x,y$ 无关的常数项丢弃；马氏范数 $\|\mathbf{e}\|^2_{\Sigma^{-1}} := \mathbf{e}^\top\Sigma^{-1}\mathbf{e}$）：
$$\min_{x,\mathbf{y}} \sum_k \big\| x_k - f(x_{k-1}, u_k) \big\|^2_{R_k^{-1}} + \sum_{k,j} \big\| z_{k,j} - h(\mathbf{y}_j, x_k) \big\|^2_{Q_{k,j}^{-1}} \tag{6.2}$$
5. 解读（写作必须给）：每项 = **一条边**的误差（运动边 / 观测边），总目标 = 所有边加权平方和——(6.2) 与图优化一一对应。特例：无运动项 → 纯 BA；线性 + 高斯 → 卡尔曼滤波（讲 10 证明等价）。

**推导 D6-2：一阶与二阶梯度法**（书 6.2.1）

1. 记目标 $F(\mathbf{x})$，二阶 Taylor（依据：多元泰勒展开）：
$$F(\mathbf{x} + \Delta\mathbf{x}) \approx F + \mathbf{J}\Delta\mathbf{x} + \tfrac12\,\Delta\mathbf{x}^\top H\,\Delta\mathbf{x} \tag{6.3}$$
2. 梯度下降：取 $\Delta\mathbf{x} = -\lambda\,\mathbf{J}^\top$（依据：负梯度方向是 (6.3) 中一阶项最速下降方向）。
3. 牛顿法：对 (6.3) 右端关于 $\Delta\mathbf{x}$ 求导置零（依据：驻点条件）：
$$H\,\Delta\mathbf{x} = -\mathbf{J}^\top \tag{6.4}$$
   问题：$H$ 计算昂贵且可能病态/不正定。

**推导 D6-3：Gauss–Newton**（书 6.2.2）

1. 设 $F(\mathbf{x}) = \tfrac12\|\mathbf{f}(\mathbf{x})\|^2$。展开 $\mathbf{f}(\mathbf{x}+\Delta\mathbf{x}) \approx \mathbf{f} + \mathbf{J}\Delta\mathbf{x}$（一阶），平方后：
$$F(\mathbf{x}+\Delta\mathbf{x}) \approx \tfrac12\|\mathbf{f}\|^2 + \mathbf{f}^\top\mathbf{J}\Delta\mathbf{x} + \tfrac12\,\Delta\mathbf{x}^\top \mathbf{J}^\top\mathbf{J}\,\Delta\mathbf{x} \tag{6.5}$$
   （**GN 核心近似**：$\Delta\mathbf{x}$ 的二阶项用 $\mathbf{J}^\top\mathbf{J}$ 代替真 Hessian $H = \mathbf{J}^\top\mathbf{J} + \sum_i f_i H_i$，即丢弃 $\mathbf{f}$ 的二阶导数项——依据：残差较小时该项贡献小。）
2. 驻点条件（依据：对 (6.5) 求导置零）：
$$\big(\mathbf{J}^\top\mathbf{J}\big)\,\Delta\mathbf{x} = -\mathbf{J}^\top\mathbf{f} \tag{6.6}$$
   ——法方程；$\mathbf{J}^\top\mathbf{J}$ 半正定但可能奇异/病态（讨论：接近奇异时 $\Delta\mathbf{x}$ 爆炸）→ 需要置信域。

**推导 D6-4：Levenberg–Marquardt**（书 6.2.3）

1. 思想：线性近似可信时大胆走、不可信时收缩——把"可信度"编码进阻尼：
$$\big(H + \lambda\,D^\top D\big)\,\Delta\mathbf{x} = -\mathbf{J}^\top\mathbf{f}, \qquad D \succeq 0 \tag{6.7}$$
   （依据：加正定阻尼保证矩阵正定；$\lambda\to 0$ 退化为 GN，$\lambda\to\infty$ 退化为小步梯度下降。）
2. 自适应规则：定义实际下降/预测下降之比 $\rho = \dfrac{F(\mathbf{x}) - F(\mathbf{x}+\Delta\mathbf{x})}{\tfrac12\Delta\mathbf{x}^\top(\lambda D^\top D\,\Delta\mathbf{x}) - \mathbf{f}^\top\mathbf{J}\Delta\mathbf{x}}$；$\rho$ 大 ⇒ 模型准 ⇒ 减小 $\lambda$；$\rho$ 小/负 ⇒ 增大 $\lambda$ 并重解（依据：置信域方法标准策略）。
3. 四法对照表（写作产出）：迭代式 / 每步代价 / 收敛速度 / 鲁棒性 / 适用场景。

**推导 D6-5：图优化**（书 6.4.1）

1. 结构化视角：顶点 = 待估状态（位姿/路标），边 = 一项误差 $\mathbf{e}(\text{顶点集}, z)$ 带信息矩阵 $\Omega$；总目标 $= \sum_i \mathbf{e}_i^\top \Omega_i \mathbf{e}_i$——即 (6.2) 的图表示。
2. 线性化与 Hessian 累加（依据：对每条边独立线性化后求和）：
$$H = \sum_i \mathbf{J}_i^\top\,\Omega_i\,\mathbf{J}_i, \qquad \mathbf{g} = \sum_i \mathbf{J}_i^\top \Omega_i\, \mathbf{e}_i \tag{6.8}$$
3. **结构预告**：路标类边只连少数顶点 ⇒ $H$ 高度稀疏——讲 10 用 Schur 补把它算快；位姿类边只在相邻/回环帧间 ⇒ 讲 11 位姿图。
4. 实践框架对比：Ceres（自动求导、工程通用）vs g2o（顶点/边抽象、SLAM 传统）（操作豁免 + 选型理由表）。

### 实践 / 易错点 / 自检

- **实践**：Ceres / g2o 拟合曲线（书例）；对应 `core/solver.py`（纯 numpy GN/LM，含数值验证测试）。
- **易错点**：把 GN 当牛顿（混淆 $J^\top J$ 与 $H$）；信息矩阵与协方差弄反；LM 中 $\lambda$ 忘记重置。
- **自检**：① 不看书写出 (6.2) 并逐项对应图优化的边；② 推 (6.6) 并指出近似发生在哪一步；③ 说明 $H + \lambda I$ 为什么保证下降。
- **配套**：[03 章](03_概率状态估计基础.md)（MAP 叙述线）、[08 章](08_后端-ii图优化与-ba.md)（稀疏求解主线）。

---

# 第二阶段｜前端（讲 7–9）

## 第 7 讲｜视觉里程计 1：特征点法

- **学习目标**：① 掌握 ORB 特征的构成与不变性来源；② 完整推导对极约束、E/F/H 矩阵及八点法；③ 掌握三角测量、PnP（DLT/P3P/BA）、ICP（SVD）的推导；④ 说清 2D-2D / 3D-2D / 3D-3D 三种问题各自解什么。
- **问题场景锚点（①）**：两帧图像 + 匹配点对（2D-2D），或匹配点 + 已有 3D 地图（3D-2D），或两组 3D 点（3D-3D）——手头：对应关系；想要：相机运动 $R,t$（进而地图点深度）。
- **知识点清单**（书 7.1–7.11）：特征点与 ORB；特征匹配；对极几何（对极约束/本质矩阵/单应矩阵）；求解相机运动；三角测量；PnP（DLT/P3P/BA）；ICP（SVD/非线性）。

### 核心公式与逐步推导

**特征部分（书 7.1–7.2，原理级）**：FAST 角点判据（圆周 16 像素中连续 $N\ge 9$ 个显著亮/暗于中心）；BRIEF 描述子（随机点对灰度比较 → 二进制串，Hamming 距离匹配）；ORB = FAST + 图像金字塔（尺度不变）+ 灰度质心法（旋转不变）：$\theta = \operatorname{atan2}(m_{01}, m_{10})$，$m_{pq} = \sum_{R} x^p y^q I(x,y)$（依据：图像矩定义，主方向为质心与形心的连线角）。特征匹配 = 描述子最近邻 + 比值检验（比例阈值剔除歧义匹配，论证）。

**推导 D7-1：对极约束、本质矩阵与基础矩阵**（书 7.3，回引 [05 章](05_视觉里程计-i特征点法.md)）

1. 归一化坐标：$\mathbf{x}_1 = K^{-1}\mathbf{p}_1,\ \mathbf{x}_2 = K^{-1}\mathbf{p}_2$（依据：D5-1 第 2 步的逆——去内参）。
2. 同一 3D 点 $P$ 在两相机系下的坐标关系（依据：$P$ 在系 2 的坐标 = 系 1 坐标经刚体变换；$s_1, s_2$ 为深度）：
$$s_2\,\mathbf{x}_2 = s_1\,R\,\mathbf{x}_1 + \mathbf{t} \tag{7.1}$$
3. 左乘 $\mathbf{t}^{\wedge}$（依据：$\mathbf{t}^{\wedge}\mathbf{t} = \mathbf{t}\times\mathbf{t} = \mathbf{0}$，消去平移项）：
$$s_2\,\mathbf{t}^{\wedge}\mathbf{x}_2 = s_1\,\mathbf{t}^{\wedge}R\,\mathbf{x}_1 \tag{7.2}$$
4. 左乘 $\mathbf{x}_2^\top$（依据：$\mathbf{x}_2^\top\mathbf{t}^{\wedge}\mathbf{x}_2 = \mathbf{x}_2\cdot(\mathbf{t}\times\mathbf{x}_2) = 0$——三向量混积有公共向量必为零；$s_1 \ne 0$ 约去）：
$$\mathbf{x}_2^\top\,\mathbf{t}^{\wedge}R\,\mathbf{x}_1 = 0 \tag{7.3}$$
5. 命名与像素版：$E = \mathbf{t}^{\wedge}R$（**本质矩阵**）；代回归一化关系（依据：$\mathbf{x} = K^{-1}\mathbf{p}$）：
$$\mathbf{p}_2^\top \underbrace{K^{-\top} E K^{-1}}_{F}\ \mathbf{p}_1 = 0 \tag{7.4}$$
   $F$（**基础矩阵**）。几何解读：$\mathbf{p}_2$ 必在 $\mathbf{p}_1$ 对应的**极线**上——匹配搜索从 2D 降为 1D（讲 13 极线搜索直接复用）。
6. $E$ 的性质（逐条推导）：3×3 共 9 参 − 尺度等价 1 − 秩 2 约束（$\det E = \det \mathbf{t}^{\wedge}\cdot\det R = 0 \cdot 1 = 0$，依据：含零特征值）⇒ 5 自由度（五点法理论下限）；八点法用秩 3 近似 + 后投影修正。

**推导 D7-2：八点法求解 E 与 R, t 分解**（书 7.4）

1. 线性化：$\mathbf{p}_2^\top F \mathbf{p}_1 = 0$ 展开为 $[u_2u_1,\ u_2v_1,\ u_2,\ v_2u_1,\ \cdots,\ 1]\,\mathbf{f} = \mathbf{0}$，$\mathbf{f} \in \mathbb{R}^9$（依据：逐项展开）；≥8 对点堆叠 $A\mathbf{f} = \mathbf{0}$。
2. 求解：$\mathbf{f}$ = $A$ 零空间的单位向量 = 最小奇异值对应的右奇异向量（依据：齐次线性方程的最小二乘解 / Rayleigh 商），SVD 后把 $F$ 强制为秩 2（零化最小奇异值）。
3. 数值稳定：先归一化（平移缩放坐标至 $[-1,1]^2$，再回代补偿）（依据：条件数分析——直接像素值量级 $10^3$ 使 $A$ 病态）。
4. **E → R, t 分解**：$E = U\Sigma V^\top$，令 $W = \begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$（绕 z 转 +90°，满足 $W\mathbf{z}^\wedge = (W\mathbf{z})^\wedge W$ 型对易关系——写作时验证）。四个候选：
$$R \in \{UWV^\top,\ UW^\top V^\top\}, \qquad \mathbf{t}^{\wedge} \in \{\,UW\Sigma U^\top,\ UW^\top\Sigma U^\top\,\}\ \text{（按 } R \text{ 配对）} \tag{7.5}$$
   （依据：$W$ 正交且 $\det W = +1$ ⇒ 候选 $R$ 保持 $\det = +1$ 且 $U\,R_z(\pm 90°)\,\Sigma U^\top$ 保持反对称。）
5. 筛选：先剔 $\det R = -1$ 者；再对每个候选做三角化，取**所有匹配深度为正**的唯一解（依据：物理可行性 cheirality 检查）。讨论：单目尺度不确定性——$E$ 只定出方向与单位化平移；纯旋转场景 $E = 0$ 失效 → 单应矩阵。
6. **单应矩阵 H**（书 7.3.3）：3D 点落在平面 $\mathbf{n}^\top P = 1$ 上时（依据：把平面约束代入 $s_2\mathbf{x}_2 = s_1 R\mathbf{x}_1 + \mathbf{t}$）：
$$s_2\mathbf{x}_2 = s_1 R\mathbf{x}_1 + \mathbf{t}\,\mathbf{n}^\top\mathbf{x}_1 \ \Rightarrow\ \mathbf{x}_2 \propto (R + \mathbf{t}\mathbf{n}^\top)\,\mathbf{x}_1, \quad H := R + \mathbf{t}\mathbf{n}^\top \tag{7.6}$$
   自由度：9 − 1（尺度）= 8 ⇒ 4 对点线性求解（每对贡献 2 独立方程）；纯旋转 $\mathbf{t}=0$ 时 $H = R$ 自然覆盖。E vs H 选型（表）：场景含平面/纯旋转 → H；一般场景 → E；实践中两者都算、按内点数择优。

**推导 D7-3：三角测量**（书 7.5）

1. 已知 $R,\mathbf{t}$ 与匹配 $\mathbf{x}_1,\mathbf{x}_2$，求深度 $s_1, s_2$：出发同 (7.1)。
2. 左乘 $\mathbf{x}_2^{\wedge}$（依据：$\mathbf{x}_2^{\wedge}\mathbf{x}_2 = 0$ 消去左边）：
$$s_1\,\mathbf{x}_2^{\wedge}R\,\mathbf{x}_1 + \mathbf{x}_2^{\wedge}\mathbf{t} = \mathbf{0} \tag{7.7}$$
3. 这是关于 $s_1$ 的线性方程组（秩 2），最小二乘解（依据：反对称左乘把 3 维方程压到与 $\mathbf{x}_2$ 正交的平面内）：
$$s_1 = \big(\mathbf{x}_2^{\wedge}R\,\mathbf{x}_1\big) \oslash \big(\mathbf{x}_2^{\wedge}\mathbf{t}\big)\ \text{（按分量相除取有效分量）} \tag{7.8}$$
4. 有效性：噪声下两视线不严格相交 ⇒ 深度有不确定度；误差沿深度方向放大（写作：以视锥张角示意图说明远处三角化病态）——引出讲 13 深度滤波器的"多次观测融合"。

**推导 D7-4：PnP 三条路线**（书 7.7–7.8）

1. **DLT**：$s\,[\mathbf{u};1] = K[R\,|\,\mathbf{t}]\,[\mathbf{P};1]$，记 $T \in \mathbb{R}^{3\times4}$（12 元 − 尺度 = 11 自由度）。消尺度（依据：两侧叉乘 $(\mathbf{u}^{\wedge})\,T\,\tilde{\mathbf{P}} = \mathbf{0}$ 给出每点 2 个独立线性方程——3 行秩 2）⇒ ≥6 对点堆叠成线性方程，SVD 取最小奇异向量；$R$ 部分 SVD 投影回 $SO(3)$（依据：Frobenius 范数下最近正交阵 = $UV^\top$）。
2. **P3P**：3 点 + 1 验证点。记相机中心 $P$ 到三点距离 $a=|PA|, b=|PB|, c=|PC|$，投影三角形夹角 $\langle\mathbf{a},\mathbf{b}\rangle$ 等由图像可测（依据：投影保持夹角——中心投影的性质）。余弦定理三式：
$$a^2 + b^2 - 2ab\cos\langle\mathbf{a},\mathbf{b}\rangle = |AB|^2, \quad (\text{对 } BC, CA \text{ 同理}) \tag{7.9}$$
   比例消元（$v = c/a$, $u = b/a$）化为关于 $u$ 的四次方程（依据：书 7.7.2 推导；解析给 2–4 组解）→ 第 4 点验证剔除歧义。特点：闭式、适合初始化。
3. **BA 形式**：$\min_{T} \sum_i \|\mathbf{u}_i - \pi(K, T, \mathbf{P}_i)\|^2$，用讲 6 GN/LM 迭代；雅可比 $\partial\mathbf{e}/\partial\delta\boldsymbol{\xi}$ 由 (4.11) + 投影求导拼出（完整展开放讲 10 D10-4，此处引用编号）。
4. 工程流程（表）：P3P/EPnP 给初值 → g2o/Ceres 投影误差 BA 精化。

**推导 D7-5：ICP 的 SVD 解**（书 7.9）

1. 已知两组 3D 点配对 $\{p_i\} \leftrightarrow \{q_i\}$，目标：
$$\min_{R,\mathbf{t}} \sum_i \|R\,p_i + \mathbf{t} - q_i\|^2 \tag{7.10}$$
2. 最优平移：对 $\mathbf{t}$ 求偏导置零：$2\sum_i(Rp_i + \mathbf{t} - q_i) = \mathbf{0} \Rightarrow \mathbf{t}^* = \bar{q} - R\bar{p}$（依据：$\bar{p}, \bar{q}$ 为质心）。
3. 代入并去质心（$p_i' = p_i - \bar{p},\ q_i' = q_i - \bar{q}$，依据：代入整理，$\mathbf{t}$ 消失）：
$$\min_R \sum_i \|R\,p_i' - q_i'\|^2 = \min_R\Big[\underbrace{\sum_i\big(\|p_i'\|^2 + \|q_i'\|^2\big)}_{R\text{ 无关}} - 2\sum_i q_i'^\top R\,p_i'\Big] \Rightarrow \max_R\ \operatorname{tr}\!\Big(R\,\underbrace{\textstyle\sum_i p_i' q_i'^\top}_{H}\Big) \tag{7.11}$$
   （依据：$\|Rp'\| = \|p'\|$ 保范数 + 迹的循环性质 $\sum q'^\top Rp' = \operatorname{tr}(R\sum p'q'^\top)$。）
4. SVD 求解：$H = U\Lambda V^\top$，则 $\operatorname{tr}(RH) = \operatorname{tr}\big((V^\top R U)\,\Lambda\big)$；令 $Q = V^\top R U$ 为正交阵 ⇒ 对角元 $|Q_{ii}| \le 1$ ⇒ $\operatorname{tr}(Q\Lambda) \le \sum\lambda_i$，等号当 $Q = I$（依据：正交矩阵元素有界）：
$$R^* = V U^\top \tag{7.12}$$
   $\det = -1$ 情形为反射解，须丢弃（依据：物理旋转 $\det = +1$；或翻转最小奇异方向修正）。
5. 非线性 ICP：配对未知时"配对 ↔ 求解"交替；位姿用李代数参数化 + 讲 6 优化（回引 (4.10)(6.6)）。

### 实践 / 易错点 / 自检

- **实践**：书 7.2/7.4/7.6/7.8/7.10 五个实验（ORB 匹配、E 求解、三角化、EPnP+BA、SVD-ICP）；对应 `projects/slam/epipolar/`（对极/三角化/PnP/ICP 全链）与 `ptam/`（配准进关键帧地图）。
- **易错点**：$E$ 的尺度等价与秩 2 处理次序；四解筛选忘了深度检查；ICP 忘记去质心导致 $R$ 求错；PnP 的 $s$ 是相机系深度不是图像量。
- **自检**：① 独立复现 D7-1 全链；② 说明 H 与 E 各自的适用场景并推导 (7.6)；③ 手推 (7.11)(7.12)。
- **配套**：[05 章](05_视觉里程计-i特征点法.md)；精读 [PTAM](精读/PTAM_ISMAR2007.md)、[ORB-SLAM](精读/ORB-SLAM_TRO2015.md)。

## 第 8 讲｜视觉里程计 2：直接法

- **学习目标**：① 推导 LK 光流并说明"孔径问题"的代数形态；② 完整推导直接法的光度残差与雅可比；③ 比较稀疏/半稠密/稠密三种直接法。
- **问题场景锚点（①）**：弱纹理白墙场景提不出可靠特征 → 特征法失效。手头：图像灰度本身（信息还在，只是没有"点"）；想要：继续估计运动。
- **知识点清单**（书 8.1–8.5）：直接法引出；LK 光流；金字塔 LK；直接法推导与讨论；RGB-D 直接法实践。

### 核心公式与逐步推导

**推导 D8-1：LK 光流**（书 8.2.1，回引 [06 章](06_视觉里程计-ii直接法.md)）

1. 灰度不变假设（依据：短时间间隔内同一物点灰度恒定——光照恒定假设）：
$$I(x + dx,\ y + dy,\ t + dt) = I(x, y, t) \tag{8.1}$$
2. 一阶 Taylor 展开（依据：多元泰勒，除以 $dt$）：
$$I_x u + I_y v + I_t = 0, \qquad (u, v) = \Big(\frac{dx}{dt}, \frac{dy}{dt}\Big) \tag{8.2}$$
   ——1 个方程 2 个未知数（**孔径问题**：单像素无法定光流）。
3. LK 补充假设：窗口（如 15×15）内像素共享同一 $(u,v)$（依据：小窗口运动一致性）。窗口内最小二乘 $\min \sum_W (I_x u + I_y v + I_t)^2$，法方程（依据：D6-3 同型，$A = [\,I_x\ \ I_y\,]$ 按像素堆叠）：
$$\big(A^\top A\big)\begin{bmatrix}u\\v\end{bmatrix} = -A^\top \mathbf{b}, \qquad A^\top A = \begin{bmatrix}\sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2\end{bmatrix} \tag{8.3}$$
4. 可解性：$A^\top A$ 两特征值都大 ⇔ 窗口内纹理沿两个方向都有变化（依据：特征值 = 二阶结构张量的主方差——与角点判据同源，呼应 FAST）；单方向纹理（如直边）⇒ 奇异 ⇒ 沿边缘方向速度不可定。
5. 金字塔 LK：粗层大位移 → 逐层上采样为初值再精化（依据：位移随分辨率降低而缩小，使线性化有效）；前向–后向一致性校验剔除误跟踪（工程，豁免推导）。

**推导 D8-2：直接法**（书 8.4.1）

1. 假设延续 (8.1) 灰度不变；已知 3D 点 $P$（RGB-D 深度或已三角化）。目标（依据：把"灰度一致"写成优化目标；位姿用李代数增量 $\exp(\boldsymbol{\xi}^{\wedge})$ 参数化——回引 (4.3)）：
$$\min_{\boldsymbol{\xi}}\ \sum_{i}\ \big\|\, I_1(\mathbf{p}_{1,i}) - I_2\big(\boldsymbol{\pi}(K,\ \exp(\boldsymbol{\xi}^{\wedge})\,P_i)\big)\,\big\|^2 \tag{8.4}$$
2. 残差 $\mathbf{e}(\boldsymbol{\xi}) = I_2(\mathbf{p}_2(\boldsymbol{\xi})) - I_1(\mathbf{p}_1)$，链式求导（依据：复合函数求导；图像梯度 $I_{2x}, I_{2y}$ 数值可预计算）：
$$\frac{\partial \mathbf{e}}{\partial \delta\boldsymbol{\xi}} = -\,\big[\,I_{2x}\ \ I_{2y}\,\big]\ \frac{\partial \mathbf{p}_2}{\partial \delta\boldsymbol{\xi}} \tag{8.5}$$
3. 投影对扰动的导数（两步拼装——依据：D5-1 投影求导 × **D4-5 (4.11) 的 SE(3) 扰动雅可比**，$\mathbf{q} = \exp(\boldsymbol{\xi}^{\wedge})P = [X', Y', Z']^\top$）：
$$\frac{\partial \mathbf{p}_2}{\partial \delta\boldsymbol{\xi}} = \begin{bmatrix} f_x/Z' & 0 & -f_x X'/Z'^2 \\ 0 & f_y/Z' & -f_y Y'/Z'^2 \end{bmatrix}\Big[\ I \ \Big|\ -\mathbf{q}^{\wedge}\ \Big] \tag{8.6}$$
4. GN 迭代（回引 (6.6)）；三种选点策略（表）：稀疏（关键点/角点）/ 半稠密（梯度非零像素）/ 稠密（全部像素）——由目标函数求和范围决定，数学完全一致。
5. 讨论（书 8.4.2/8.5.5）：灰度不变在光照变化/曝光下失效 → 鲁棒核（讲 10 D10-5）与仿射亮度补偿（DSO，精读延伸）；相机基线大时线性化差 → 金字塔/由粗到细同样适用。

### 实践 / 易错点 / 自检

- **实践**：LK 光流 + RGB-D 稀疏/半稠密直接法（书例）；对应 `projects/slam/direct/`、`photoba/`。
- **易错点**：把光流当"特征匹配"（光流是增量假设，大位移要金字塔）；直接法雅可比漏掉图像梯度项；$\mathbf{q}$ 忘记用**扰动后**坐标。
- **自检**：① 从 (8.1) 推 (8.3) 并解释孔径问题；② 不看书写出 (8.6) 并指认每个因子来自哪一讲哪个公式；③ 列出特征法 vs 直接法的假设差异表。
- **配套**：[06 章](06_视觉里程计-ii直接法.md)；精读 [LSD-SLAM](精读/LSD-SLAM_ECCV2014.md)、[DSO](精读/DSO_PAMI2018.md)。

## 第 9 讲｜实践章：设计前端

**性质**：工程集成章，理论豁免为主（按 §0.5 第 4 条），但须注明理论回引。

- **学习目标**：① 读懂 VO 的类结构与数据流；② 独立复现"两两帧 VO → PnP 优化 → 局部地图"三级演进；③ 能把每一环节对应回前八讲的公式。
- **问题场景锚点（①）**：前八讲的算法都是"单次求解"；真实系统要**持续**运行——帧来了放哪、地图存哪、何时优化、何时丢数据。
- **知识点与理论回引表**（写作产出，每行注明回引公式）：

| 组件 | 职责 | 理论回引 |
|------|------|---------|
| Camera / Frame / MapPoint / Map / Config | 相机参数、帧与特征、路标（位置+观测）、地图容器、配置读取 | (5.3) 内参；D7-4 路标即三角化产物 |
| 两两帧 VO（初始化/跟踪/丢失判定） | 提特征 → 匹配 → 求运动 → 入图 | D7-1（对极）、D7-4（PnP） |
| 改进 1：g2o 优化 PnP | 投影误差最小化替代 EPnP 直接解 | (6.8) 图优化 + (4.11) 雅可比 |
| 改进 2：局部地图 | MapPoint 筛选（观测次数/替换/外点剔除）、关键帧判断 | 讲 10 局部 BA 的工程雏形 |

- **架构讨论**（理论点，需论证而非豁免）：参考帧/关键帧的选择依据（运动量 vs 匹配质量）——PTAM"跟踪/建图分离"的教学源头（精读 [PTAM](精读/PTAM_ISMAR2007.md)；对应 `projects/slam/ptam/` 双线程实现）。
- **自检**：画出完整数据流图并标注"此步用了第几讲哪个公式"；解释丢失后重定位为什么需要回环/重定位机制（引出讲 12）。

---

# 第三阶段｜后端与回环（讲 10–12）

## 第 10 讲｜后端 1：滤波与 BA

- **学习目标**：① 完整推导 KF 五公式与 EKF；② 推导 BA 代价函数与两块雅可比；③ 完整推导 Schur 补稀疏求解及其概率意义；④ 掌握鲁棒核。
- **问题场景锚点（①）**：前端逐帧估计有噪声、有漂移——单次求解不给"这个估计多可信"；且规模大后要全局一致。手头：整条轨迹的观测；想要：全局最优 + 不确定度 + 可扩展求解。
- **知识点清单**（书 10.1–10.5）：状态估计的概率解释；线性系统 KF；非线性系统 EKF 与讨论；BA 投影模型与代价函数；BA 求解（稀疏性与边缘化/Schur 补）；鲁棒核；g2o/Ceres 实践。

### 核心公式与逐步推导

**推导 D10-1：状态估计的概率解释**（书 10.1.1）：即 D6-1 的滤波视角重述——把"批量 MAP"与"递归滤波"的关系说清（线性高斯下两者后验一致，结论级，依据：高斯分布的共轭性质，详见 [03 章](03_概率状态估计基础.md) 完整推导）。

**推导 D10-2：卡尔曼滤波五公式**（书 10.1.2；[03 章](03_概率状态估计基础.md) §03 有同源完整推导，此处给推导链骨架，写作时展开为无跳步版本）

1. 线性高斯模型：
$$x_k = A x_{k-1} + u + w,\ \ w \sim \mathcal{N}(0,R); \qquad z_k = C x_k + v,\ \ v \sim \mathcal{N}(0,Q) \tag{10.1}$$
2. **预测步**（依据：高斯线性变换——附录 A.2.1：$y = Ax + b \Rightarrow y \sim \mathcal{N}(A\mu + b,\ A\Sigma A^\top)$；独立噪声方差相加）：
$$\hat{x}_k = A\bar{x}_{k-1} + u, \qquad \hat{P}_k = A\,\bar{P}_{k-1}A^\top + R \tag{10.2}$$
3. **更新步**：后验 $\propto \mathcal{N}(z;\, Cx, Q)\,\mathcal{N}(x;\, \hat{x}, \hat{P})$，负对数对 $x$ 求导置零（依据：完成平方 / 梯度为零）：
$$\big(\hat{P}^{-1} + C^\top Q^{-1} C\big)\,\bar{x} = \hat{P}^{-1}\hat{x} + C^\top Q^{-1} z \tag{10.3}$$
4. 定义卡尔曼增益（**push-through 恒等式**：$\big(A^{-1}+C^\top Q^{-1}C\big)^{-1}C^\top Q^{-1} = AC^\top(CAC^\top + Q)^{-1}$——两行证明：两边左乘左式括号的逆、右乘右式括号验证相等）：
$$K = \hat{P}C^\top\big(C\hat{P}C^\top + Q\big)^{-1} \tag{10.4}$$
5. 均值与协方差更新（依据：(10.3) 改写 + Woodbury 恒等式）：
$$\bar{x} = \hat{x} + K\big(z - C\hat{x}\big), \qquad \bar{P} = (I - KC)\,\hat{P} \tag{10.5}$$
6. 直觉核验：标量情形 $K = \hat{\sigma}^2/(\hat{\sigma}^2 + \sigma^2)$——谁的方差小听谁的（依据：(10.4) 的 $1\times1$ 特例）；$K$ 同时给出新息 $z - C\hat{x}$ 的权重与不确定度收缩比例。
7. 五公式汇总表（写作产出）。

**推导 D10-3：EKF 与讨论**（书 10.1.3–10.1.4）

1. 非线性 $f, h$ 在工作点一阶 Taylor（依据：多元泰勒）：$f(x) \approx f(\bar{x}) + F(x - \bar{x})$，$h(x) \approx h(\hat{x}) + H(x - \hat{x})$。
2. 五公式中 $A \to F$、$C \to H$、线性量换非线性量：
$$\hat{x}_k = f(\bar{x}_{k-1}, u),\ \ \hat{P}_k = F\bar{P}F^\top + R;\qquad K = \hat{P}H^\top(H\hat{P}H^\top + Q)^{-1};\ \ \bar{x} = \hat{x} + K(z - h(\hat{x})),\ \ \bar{P} = (I - KH)\hat{P} \tag{10.6}$$
3. 讨论（逐条给依据）：线性化误差（只展开一阶、误差大时发散）；单峰高斯假设（多峰场景失效，对比粒子滤波）；一致性（$P$ 常低估）；大场景存储 $O(n^2)$——四条共同指向优化后端。

**推导 D10-4：BA 的代价函数与雅可比**（书 10.2.1–10.2.2，回引 [08 章](08_后端-ii图优化与-ba.md)）

1. 重投影误差（依据：D5-1 投影 + D4-2 李代数位姿）：
$$\mathbf{e} = \mathbf{z} - \frac{1}{s_3}\,K\,\exp(\boldsymbol{\xi}^{\wedge})\,\mathbf{P}, \qquad \mathbf{e} \in \mathbb{R}^2 \tag{10.7}$$
2. 对位姿扰动的雅可比（依据：投影求导 × (4.11)；记 $\mathbf{q} = \exp(\boldsymbol{\xi}^{\wedge})\mathbf{P} = [X', Y', Z']^\top$）：
$$\frac{\partial \mathbf{e}}{\partial \delta\boldsymbol{\xi}} = -\begin{bmatrix} f_x/Z' & 0 & -f_x X'/Z'^2 \\ 0 & f_y/Z' & -f_y Y'/Z'^2 \end{bmatrix}\Big[\ I \ \Big|\ -\mathbf{q}^{\wedge}\ \Big] \tag{10.8}$$
3. 对路标的雅可比（依据：链式 $\partial\mathbf{q}/\partial\mathbf{P} = R$）：
$$\frac{\partial \mathbf{e}}{\partial \mathbf{P}} = -\begin{bmatrix} f_x/Z' & 0 & -f_x X'/Z'^2 \\ 0 & f_y/Z' & -f_y Y'/Z'^2 \end{bmatrix} R \tag{10.9}$$
4. 总目标 $\min \sum \mathbf{e}^\top \Omega\, \mathbf{e}$，$H = \sum \mathbf{J}^\top\Omega\mathbf{J}$ 累加（回引 (6.8)）。

**推导 D10-5：稀疏性与 Schur 补（边缘化）**（书 10.2.3，全书后端最重要推导之一）

1. $H$ 的结构：按 [相机; 路标] 分块 $H = \begin{bmatrix} B & E \\ E^\top & C \end{bmatrix}$。**C 是块对角**：路标 $\mathbf{P}_j$ 的雅可比只出现在它被观测的边里，不同路标间无耦合（依据：(6.8) 逐块分析）；$B$ 中相机间耦合来自共享路标。
2. 线性方程 $H\,\Delta\mathbf{x} = \mathbf{g}$，$\Delta\mathbf{x} = [\Delta\mathbf{x}_c; \Delta\mathbf{x}_p]$，$\mathbf{g} = [\mathbf{v}; \mathbf{w}]$。左乘消元矩阵（依据：分块乘法逐项验证；$C$ 每个对角块 = 路标的信息阵 $\ge$ 单观测贡献，可逆）：
$$\begin{bmatrix} I & -EC^{-1} \\ 0 & I \end{bmatrix}\begin{bmatrix} B & E \\ E^\top & C \end{bmatrix} = \begin{bmatrix} B - EC^{-1}E^\top & \mathbf{0} \\ E^\top & C \end{bmatrix} \tag{10.10}$$
3. 两步求解（依据：上式右端第一行已与 $\Delta\mathbf{x}_p$ 解耦）：
$$\big(B - EC^{-1}E^\top\big)\,\Delta\mathbf{x}_c = \mathbf{v} - EC^{-1}\mathbf{w}; \qquad \Delta\mathbf{x}_p = C^{-1}\big(\mathbf{w} - E^\top\Delta\mathbf{x}_c\big) \tag{10.11}$$
   $S = B - EC^{-1}E^\top$ 即 **Schur 补**：规模只含相机（远小于路标总数），求逆代价从整体维数降为相机维数（写作：给稠密 $O((a+b)^3)$ vs 稀疏 $O(a^3 + b)$ 的量级对比）。
4. **概率意义**：消去 $\Delta\mathbf{x}_p$ = 对路标边缘化 ⇒ $S$ 是"路标已知条件下相机状态"的条件分布的信息阵的 Schur 表示——与高斯条件分布 $P(x_c \mid x_p)$ 的信息矩阵公式同构（依据：附录 A.2.2 高斯乘积/条件分布；详论放 [08 章](08_后端-ii图优化与-ba.md)，含滑窗边缘化先验的先验信息守恒）。
5. 鲁棒核（书 10.2.4）：Huber 函数（$s = \mathbf{e}^\top\Omega\mathbf{e}$）：
$$\rho(s) = \begin{cases} s, & s \le 1 \\ 2\sqrt{s} - 1, & s > 1 \end{cases} \quad\Leftrightarrow\quad \text{等价于把该边信息矩阵乘 } w(s) = \rho'(s) \tag{10.12}$$
   （依据：$\sum\rho(s_i)$ 对状态求导 = $\sum \rho'(s_i)\,\partial s_i/\partial x$，与"信息矩阵加权 $\rho'(s)$ 的最小二乘"梯度同形——IRLS 一行推导。）

### 实践 / 易错点 / 自检

- **实践**：g2o / Ceres 求解 BAL 类 BA 数据集（书例）；对应 `projects/slam/photoba/`（分块 Schur 求解的纯 numpy 实现）与 `fastslam/`（滤波路线对照）。
- **易错点**：把 $E$（分块耦合阵）与本质矩阵 $E$ 撞名（写作时改用其他记号或显式区分）；忘记 $C$ 的块对角性是 Schur 补高效的前提；鲁棒核改变了目标函数但仍在旧线性化点迭代。
- **自检**：① 不看书写出 KF 五公式并推导 push-through 恒等式；② 从 (6.8) 出发解释 $C$ 为何块对角；③ 手推 (10.10)(10.11) 并解释"边缘化"两个字的概率含义。
- **配套**：[03](03_概率状态估计基础.md)、[07](07_后端-i滤波与增量估计.md)、[08 章](08_后端-ii图优化与-ba.md)；精读 [ORB-SLAM2](精读/ORB-SLAM2_TRO2016.md)。

## 第 11 讲｜后端 2：位姿图与因子图

- **学习目标**：① 说清位姿图相对 BA 的取舍；② 完整推导位姿图误差与雅可比（含伴随抵消）；③ 了解因子图与贝叶斯网络的关系。
- **问题场景锚点（①）**：BA 里路标主导规模；且回环完成后地图点已经成熟——只想**全局校正位姿**。手头：关键帧位姿 + 帧间/回环相对运动观测；想要：只含位姿的快速全局优化。
- **知识点清单**（书 11.1–11.3）：位姿图的意义与优化；g2o 原生 vs 李代数位姿图；*因子图优化初步。

### 核心公式与逐步推导

**推导 D11-1：位姿图的定义**（书 11.1）

1. 顶点 = 关键帧位姿 $T_i \in SE(3)$；边 = ①相邻帧 VO 的相对运动 $\Delta T_{ij}$ ②回环检测的相对运动（带信息阵 $\Omega_{ij}$）。"等一下再优化"的历史位姿不再带路标——规模从 $O(帧+路标)$ 降到 $O(帧)$。
2. 误差（依据：李群上无减法，把"预测相对运动 $\ne$ 观测相对运动"搬到李代数量度）：
$$\mathbf{e}_{ij} = \ln\big(\Delta T_{ij}^{-1}\,T_i^{-1}T_j\big)^{\vee} \in \mathbb{R}^6 \tag{11.1}$$
3. 总目标 $\min \sum_{(i,j)} \mathbf{e}_{ij}^\top \Omega_{ij}\, \mathbf{e}_{ij}$。

**推导 D11-2：位姿图雅可比（含伴随抵消，全书最烧脑推导）**（书 11.1.2/11.2.2）

1. 扰动约定：$T_i \leftarrow \exp(\delta\boldsymbol{\xi}_i^{\wedge})\,T_i$（**左扰动**），$T_j \leftarrow T_j\exp(\delta\boldsymbol{\xi}_j^{\wedge})$（**右扰动**）——即不确定度都定义在"边"两端的切口处（依据：约定，写作时画扰动插入位置示意图）。
2. 李代数"加法"算子（回引 D4-4）：$\mathbf{a} \boxplus \boldsymbol{\delta} = \ln\big(\exp(\mathbf{a}^{\wedge})\exp(\boldsymbol{\delta}^{\wedge})\big)^{\vee} \approx \mathbf{a} + J_r(\mathbf{a})^{-1}\boldsymbol{\delta}$（依据：BCH 级数对小量 $\boldsymbol{\delta}$ 逐项展开 $\mathbf{a} + \boldsymbol{\delta} + \tfrac12[\mathbf{a},\boldsymbol{\delta}] + \tfrac{1}{12}[\mathbf{a},[\mathbf{a},\boldsymbol{\delta}]]$，与 $J_r^{-1} = I + \tfrac12\boldsymbol{\phi}^{\wedge} + \tfrac{1}{12}(\boldsymbol{\phi}^{\wedge})^2$ 的级数逐项对照；小量在**左**时才是 $J_l^{-1}$，见 (4.6)）。
3. 逐步推导（每步给依据）：
   - 求逆翻转：$(\exp(\delta\boldsymbol{\xi}_i^{\wedge})T_i)^{-1} = T_i^{-1}\exp(-\delta\boldsymbol{\xi}_i^{\wedge})$（依据：逆的性质）：
$$\mathbf{e}_{ij}(\boldsymbol{\delta}) = \ln\Big(\Delta T_{ij}^{-1}\,T_i^{-1}\exp(-\delta\boldsymbol{\xi}_i^{\wedge})\,T_j\exp(\delta\boldsymbol{\xi}_j^{\wedge})\Big)^{\vee}$$
   - 扰动左移：$T_i^{-1}\exp(-\delta_i^{\wedge}) = \exp\big(-(\mathrm{Ad}_{T_i^{-1}}\,\delta_i)^{\wedge}\big)T_i^{-1}$（依据：伴随共轭恒等式 $T^{-1}\exp(\boldsymbol{\delta}^{\wedge})\,T = \exp\big((\mathrm{Ad}_{T^{-1}}\boldsymbol{\delta})^{\wedge}\big)$，来源 $g\exp(X)g^{-1} = \exp(\mathrm{Ad}_g X)$）：
$$= \exp\big(-(\mathrm{Ad}_{\Delta T^{-1}T_i^{-1}}\,\delta_i)^{\wedge}\big)\,\underbrace{\Delta T^{-1}T_i^{-1}T_j}_{\exp(\mathbf{e}_{ij}^{\wedge})}\,\exp(\delta_j^{\wedge})$$
   - 把 $\exp(\mathbf{e}^{\wedge})$ 提到最左（依据：同一恒等式反向使用，乘积 $g = \exp(-\mathbf{e}^{\wedge})$）：
$$= \exp(\mathbf{e}_{ij}^{\wedge})\,\exp\Big(\big(-\,\mathrm{Ad}_{\exp(-\mathbf{e}^{\wedge})}\,\mathrm{Ad}_{\Delta T^{-1}T_i^{-1}}\,\delta_i\big)^{\wedge}\Big)\exp(\delta_j^{\wedge})$$
   - **伴随抵消**：$\mathrm{Ad}_{\exp(-\mathbf{e}^{\wedge})}\,\mathrm{Ad}_{\Delta T^{-1}T_i^{-1}} = \mathrm{Ad}_{\exp(-\mathbf{e}^{\wedge})\Delta T^{-1}T_i^{-1}}$，而 $\exp(-\mathbf{e}^{\wedge}) = T_j^{-1}T_i\Delta T$ ⇒ $\exp(-\mathbf{e}^{\wedge})\Delta T^{-1}T_i^{-1} = T_j^{-1}T_i\Delta T\,\Delta T^{-1}T_i^{-1} = I$（依据：伴随的乘法性 + 代入化简）——**扰动系数恰好还原为 $-\delta_i$ 本身**：
$$= \exp(\mathbf{e}_{ij}^{\wedge})\,\exp\big((-\delta_i)^{\wedge}\big)\exp(\delta_j^{\wedge})$$
   - 合并：$\exp\big((-\delta_i)^{\wedge}\big)\exp(\delta_j^{\wedge}) \approx \exp\big((\delta_j - \delta_i)^{\wedge}\big)$（依据：BCH 零阶项），再过一次 $\boxplus$（依据：本节第 2 步，小量在右配 $J_r^{-1}$）：
$$\mathbf{e}_{ij}(\boldsymbol{\delta}) \approx \mathbf{e}_{ij} + J_r(\mathbf{e}_{ij})^{-1}\big(\delta\boldsymbol{\xi}_j - \delta\boldsymbol{\xi}_i\big)$$
4. 结论：
$$\boxed{\ \frac{\partial \mathbf{e}_{ij}}{\partial \delta\boldsymbol{\xi}_i} = -J_r(\mathbf{e}_{ij})^{-1}, \qquad \frac{\partial \mathbf{e}_{ij}}{\partial \delta\boldsymbol{\xi}_j} = +J_r(\mathbf{e}_{ij})^{-1}\ } \tag{11.2}$$
   （依据：上式读出。约定注记：$J_l^{-1}$ 还是 $J_r^{-1}$ 随扰动约定而变——若把 $T_j$ 改为左扰动并通过伴随重定义 $\boldsymbol{\delta}_j$，会出现 $J_r^{-1}\mathrm{Ad}_{T_j^{-1}}$；落地时以实现残差的数值差分为准。）写作提示：伴随抵消一步是难点，建议先在 $SE(2)$ / 阿贝尔群上验证符号（$\mathbf{e} = -\ln B - \ln A + \ln C$，扰动后 $+\delta_j - \delta_i$，与 (11.2) 在 $J_l = I$ 时一致）。
5. 求解：GN（回引 (6.6)）；$H$ 只在相邻/回环帧间有块——稀疏位姿图；g2o 原生位姿边（四元数+平移参数化）vs 李代数自定义边（(11.2) 雅可比）实现对照（实践，豁免推导）。

**推导 D11-3：*因子图初步**（书 11.3）

1. 贝叶斯网络：联合分布按条件链分解 $P(x_1, ..., x_N) = \prod_i P(x_i \mid \mathrm{pa}(x_i))$（依据：链式法则 + 条件独立性）。
2. 因子图：把"变量"与"因子（条件概率项）"分两类节点的二部图——图优化的一条边一个因子，信息矩阵 = 因子的参数（依据：同一目标函数的两种图表示，等价性结论级）。
3. 在 SLAM 中的价值（点到为止）：统一表达观测因子/运动因子/回环因子/先验因子——为 VIO（精读 [IMU 预积分](精读/IMU-Preintegration_TRO2017.md)）与 GTSAM 生态铺路。

### 实践 / 易错点 / 自检

- **实践**：位姿图优化实验（书例：球/网格位姿图 + 人工回环）；对应 `projects/slam/vins/`（位姿图校正）与 `bowloop/`（回环触发的位姿图）。
- **易错点**：左右扰动约定混用导致雅可比符号错；把 $\Delta T_{ij}$ 当绝对位姿；回环边的信息矩阵随意给（应与前端不确定度一致）。
- **自检**：① 复现 D11-2 全链（含 $SE(2)$ 验证）；② 解释为什么误差要经 $\ln(\cdot)^{\vee}$；③ 画出位姿图的 $H$ 稀疏模式。
- **配套**：[08 章](08_后端-ii图优化与-ba.md)；精读 [ORB-SLAM3](精读/ORB-SLAM3_TRO2021.md)（多地图位姿图合并）。

## 第 12 讲｜回环检测

- **学习目标**：① 掌握词袋模型的建模动机与字典结构；② 会计算 TF-IDF 权重与相似度并解释公式构成；③ 会用准确率/召回率评价回环算法并说明检测后验证的必要性。
- **问题场景锚点（①）**：机器人走回走过的地方，前端/后端都不知道——漂移持续累积。手头：历史关键帧库 + 当前帧；想要：快速"认出旧地"（毫秒级、免几何匹配）。
- **知识点清单**（书 12.1–12.5）：回环概述（意义/方法/P-R）；词袋模型；字典（结构/创建）；相似度计算；实验分析与评述。

### 核心公式与逐步推导

**推导 D12-1：回环的意义与评价**（书 12.1）

1. 漂移累积模型：$\hat{x}_k = x_k + \epsilon_k$，$\epsilon_k$ 随时间随机游走（依据：前端误差叠加）——只有回环边提供全局约束（连接到讲 11 位姿图：回环边即 (11.1) 中信息量最大的边）。
2. 评价（依据：混淆矩阵定义）：
$$\text{准确率 } P = \frac{TP}{TP + FP}, \qquad \text{召回率 } R = \frac{TP}{TP + FN} \tag{12.1}$$
   SLAM 语境解读：FP（把新地方认成旧的）会**破坏**地图——比 FN（漏检）更致命 ⇒ 阈值偏向高准确率（P-R 曲线取舍，写作给曲线示例）。
3. 方法谱系：几何法（对比特征点/投影）慢且依赖初值；**外观法**（词袋）O(log 字典) 查询——主流选择（选型论证）。

**推导 D12-2：词袋与相似度**（书 12.2–12.4）

1. 建模：图像 = "单词的无序集合" → 稀疏向量 $\mathbf{v}$（依据：把连续图像匹配问题离散化为向量比较；顺序与位置信息被有意丢弃——换取旋转/视角鲁棒性，代价是几何验证必须后置）。
2. 字典结构（书 12.3）：对描述子做 $k$ 叉树聚类，深度 $d$，叶子 = 单词，$W = k^d$；查询沿树 $O(\log W)$（依据：树查找复杂度）。
3. TF-IDF 权重（书 12.4.1）：
$$\eta_i = n_{ij}\cdot\log\Big(\frac{N}{f_j}\Big) \tag{12.2}$$
   （依据：TF 项 $n_{ij}$——单词 $i$ 在图像 $j$ 出现越多越能代表该图像；IDF 项 $\log(N/f_j)$——单词在 $N$ 帧语料中仅 $f_j$ 帧出现时区分度大、 ubiquitous 词权重低。直觉论证，书亦如此。）
4. 相似度（书 12.4.1，$\mathbf{v}$ 经 L1 归一化 $\|\mathbf{v}\|_1 = 1$）：
$$s(A, B) = 1 - \tfrac12\,\|\mathbf{v}_A - \mathbf{v}_B\|_1 \in [0, 1] \tag{12.3}$$
   （推导：同一图 → 距离 0 → $s=1$；支撑集完全不相交 → $\|\cdot\|_1 = \|\mathbf{v}_A\|_1 + \|\mathbf{v}_B\|_1 = 2$ → $s=0$；中间单调。）
5. 实用化四件套（书 12.5，每条给依据）：字典规模试验（大到泛化小到区分，权衡曲线）；相似性评分的**组归一化**（与历史均值比较，抗图像自身质量波动）；关键帧筛选（相近帧不重复入库）；检测后验证（时间一致性：连续命中才算 + 几何验证：对匹配特征做 RANSAC 求解 PnP/F 检验内点数——把 FP 压下去，呼应 D12-1 第 2 步）。

### 实践 / 易错点 / 自检

- **实践**：DBoW3 建字典 + 相似度矩阵可视化（书例）；对应 `projects/slam/bowloop/`（词袋召回 + 几何验证 + 位姿图校正全链）。
- **易错点**：忘做组归一化导致阈值难定；只看词袋分数不做几何验证；字典与特征描述子不配套（ORB 字典配 BRISK 特征）。
- **自检**：① 从词袋建模动机推出"为何必须后置几何验证"；② 手算两幅玩具图像的 (12.2)(12.3)；③ P=0.99/R=0.3 与 P=0.6/R=0.95 两种检测器该选哪个？为什么？
- **配套**：[09 章](09_回环检测.md)；精读 [ORB-SLAM](精读/ORB-SLAM_TRO2015.md)（词袋三线程协同）。

---

# 第四阶段｜系统与前沿（讲 13–14）

## 第 13 讲｜建图

- **学习目标**：① 理解从稀疏到稠密深度的两条路径（立体匹配 / 深度滤波）；② 推导高斯–均匀深度滤波器的更新；③ 会解释逆深度参数化与像素梯度问题；④ 了解点云/八叉树/TSDF 三类地图。
- **问题场景锚点（①）**：VO 只产出稀疏特征点地图——导航/碰撞规避需要稠密或半稠密结构。手头：带位姿的图像序列；想要：每像素（或每块）深度及其可信度。
- **知识点清单**（书 13.1–13.6）：单目稠密重建（立体视觉/极线搜索与块匹配/深度滤波器）；实践与实验分析（像素梯度问题/逆深度）；点云与八叉树地图；*TSDF 与 Fusion 系列。

### 核心公式与逐步推导

**推导 D13-1：极线搜索与块匹配**（书 13.2.1–13.2.2）

1. 匹配约束：给定左图 $\mathbf{p}_L$ 与已知 $R,\mathbf{t}$，右图对应点必在**极线**上（依据：D7-1 第 5 步对极约束的几何形式）——匹配从 2D 搜索降为 1D 搜索。
2. 1D 上做块匹配：取左图小块沿极线滑动，归一化互相关（依据：NCC 对线性光照变化 $I \to aI + b$ 不变——分母中心化+模长归一）：
$$\mathrm{NCC}(a, b) = \frac{\sum (a_i - \bar{a})(b_i - \bar{b})}{\sqrt{\sum (a_i - \bar{a})^2 \sum (b_i - \bar{b})^2}} \in [-1, 1] \tag{13.1}$$
3. 取峰值处视差 → 深度初值 + 峰宽 → 不确定度（衔接 D5-3：$\delta Z/Z \propto \delta d/d$，远处不确定度爆炸）。

**推导 D13-2：深度滤波器**（书 13.2.3；[03 章](03_概率状态估计基础.md) 滤波思想的一维应用）

1. 深度先验（高斯–均匀混合，依据：已有估计用高斯建模、未初始化像素用均匀表达"完全未知"）：
$$p(d) = \xi\,\mathcal{N}(d;\, \mu, \sigma^2) + \gamma\, \mathcal{U}(d_{\min}, d_{\max}), \qquad \xi + \gamma = 1 \tag{13.2}$$
2. 每来一次新观测 $d_{obs} \sim \mathcal{N}(\nu, \tau^2)$（极线搜索+NCC 产出），按贝叶斯乘积融合；高斯部分两高斯相乘（依据：附录 A.2.2，配方展开 $-\tfrac{(d-\mu)^2}{2\sigma^2} - \tfrac{(d-\nu)^2}{2\tau^2}$ 按平方项系数/一次项系数对照读出）：
$$\mu_{fuse} = \frac{\sigma^2\,\nu + \tau^2\,\mu}{\sigma^2 + \tau^2}, \qquad \sigma^2_{fuse} = \frac{\sigma^2\,\tau^2}{\sigma^2 + \tau^2} \tag{13.3}$$
3. 收敛判据：$\sigma < \sigma_{max}$（依据：观测融合后方差单调不增——两高斯乘积方差恒小于二者）⇒ 该像素深度"定点"，输出高斯部分；否则继续观测。
4. 讨论（书 13.4.1–13.4.2，论证题）：**像素梯度问题**——弱梯度区域 NCC 峰平坦（似然曲线宽）→ $\tau$ 大 → 长期不收敛（与讲 8 半稠密"选梯度明显像素"同根同源）；**逆深度参数化** $\rho = 1/d$——远场景 $d \to \infty$ 时 $\rho \to 0$ 有限可表示、且投影误差对 $\rho$ 近似线性（论证 + 对比示意图）。

**推导 D13-3：三类地图模型**（书 13.5–13.6，概念 + 关键公式）

1. 点云地图：无拓扑、无概率（存疑度不表达）→ 简单但重叠难消。
2. 八叉树地图（OctoMap）：每个叶节点存占用概率，对数域更新（依据：概率乘法在对数域化为加法，避免数值下溢；教程补充公式，书为概念介绍）：
$$l \leftarrow l + \log\frac{p_{occ}}{1 - p_{occ}} - l_{prior} \tag{13.4}$$
   稀疏性：未观测空间不存储（与 TSDF 的差异点）。
3. *TSDF 与 KinectFusion：截断符号距离场——每体素存到表面的带符号距离（截断在 $\pm\delta$ 内），多帧加权平均融合；网格化 = 提取零等值面（Marching Cubes）。概念级，公式展开放 [3D 重建 04 章](../3d_reconstruction/04_RGB-D融合与TSDF.md)（互链，不重复推导）。

### 实践 / 易错点 / 自检

- **实践**：单目稠密重建（书例：极线搜索 + 深度滤波逐像素融合）；八叉树/点云地图。对应：`projects/slam/droidlite/`（稠密深度迭代更新的延伸对照）；**深度滤波器暂无对应模块——列为扩展任务**（纯 numpy 一维滤波即可实现，公式即 (13.2)(13.3)）。
- **易错点**：极线方向算反（内参/外参次序）；深度滤波用正态而忘掉均匀分量导致远处假收敛；逆深度下均值方差公式直接套 (13.3) 而不换元。
- **自检**：① 推 (13.3) 并说明与 KF 更新 (10.5) 的关系（同为两高斯融合）；② 为什么逆深度在纯旋转/远平面更稳？③ 对比点云/八叉树/TSDF 的存储、概率表达、更新方式。
- **配套**：[10 章](10_建图与系统实战.md)、[3D 重建 04 章](../3d_reconstruction/04_RGB-D融合与TSDF.md)；精读 [LSD-SLAM](精读/LSD-SLAM_ECCV2014.md)（半稠密深度滤波的系统能形态）、[NeRF-SLAM](精读/NeRF-SLAM_ICRA2023.md)（神经隐式建图方向）。

## 第 14 讲｜SLAM：现在与未来

**性质**：综述章，公式豁免；每方向给"入门最小问题集"。

- **知识点清单**（书 14.1–14.2）：六大开源方案对比；视觉 + 惯导；语义 SLAM；展望。
- **大纲要点**：
  1. **方案对比表**（写作产出）：MonoSLAM（EKF/稀疏）→ PTAM（双线程/关键帧/BA）→ ORB-SLAM（三线程/词袋/单目双目 RGB-D）→ LSD-SLAM（半稠密直接法）→ SVO（半直接：粗光流 + 深度滤波）→ RTAB-MAP（RGB-D/记忆管理）。每个方案一行：特征/直接/半直接、滤波/优化、稀疏度、回环、建图形态——全部对应到前面各讲的公式（回引编号）。
  2. **VIO（视觉 + IMU）**：互补性论证（相机：纹理丰富但慢、尺度不明；IMU：快、有尺度，但漂移）；耦合谱系：松耦合（结果级融合）/紧耦合（滤波 MSCKF vs 优化——**预积分**：把两 IMU 时刻间多项测量折叠成一个与线性化点解耦的运动增量，公式展开放 [10 章](10_建图与系统实战.md)，配精读 [IMU 预积分](精读/IMU-Preintegration_TRO2017.md)、[VINS-Mono](精读/VINS-Mono_RAM2018.md) 与代码 `preint/`、`vins/`）。
  3. **语义 SLAM**：CNN 提供物体级标签 → 地图对象化 / 动态物剔除 / 人机交互接口（论证方向 + 开放问题：语义不确定度如何进后端）。
  4. **前沿延伸**（本仓库精读已有）：端到端可微 SLAM（[DROID-SLAM](精读/DROID-SLAM_NeurIPS2021.md)、`droidlite/`）；3D 高斯（[GS-SLAM](精读/GS-SLAM_CVPR2024.md)、[SplaTAM](精读/SplaTAM_CVPR2024.md)、[MonoGS](精读/MonoGS_CVPR2024.md)）；即时重建（[MASt3R-SLAM](精读/MASt3R-SLAM_CVPR2025.md)、[VGGT-GS](精读/VGGT-GS-SLAM_arXiv2026.md)）；激光（[LOAM](精读/LOAM_RSS2014.md)、`loam2d/`）。
- **自检**：给任一开源方案画"四模块 + 数据流"图并标注对应讲次；说出紧耦合优于松耦合的一个可量化理由。

---

## 附录处理

### 附录 A｜高斯分布的性质（公共引理库）

书 A.1–A.3 的三组性质**不是独立章节，而是全书的公共引理**，写作时统一编号供回引：

| 引理 | 内容 | 被引用处 |
|------|------|---------|
| A.2.1 线性变换 | $y = Ax + b \Rightarrow y \sim \mathcal{N}(A\mu + b,\ A\Sigma A^\top)$ | 讲 10 (10.2) 预测步 |
| A.2.2 乘积 | 两高斯乘积 = 高斯（配方展开） | 讲 10 (10.3)–(10.5)、讲 13 (13.3)、讲 3 滤波推导 |
| A.2.3 条件/边缘 | 条件分布均值 = $\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)$ | 讲 10 Schur 补概率意义、滑窗边缘化 |
| A.2.3 复合 | 两高斯之差仍是高斯（推导给全） | 讲 3、讲 10 讨论题 |

### 附录 B｜ROS 入门

操作豁免；大纲仅要求：给出 ROS 话题/服务与 SLAM 四模块的对应表，标注哪些书例有 ROS 版、哪些须自装。

---

## 写作验收清单（每讲交稿前自查）

1. [ ] 五步法齐全（[CONSTRAINTS.md §3.4](../../CONSTRAINTS.md)）；纯操作内容有豁免声明。
2. [ ] 本大纲列出的每条推导链（DN-k）在正文中**无跳步**完成，每步有依据。
3. [ ] 公式编号与本大纲 (N.m) 一致，可被代码注释与后续章节回引。
4. [ ] 与 [README.md](README.md) 10 章教程的交叉引用已写（何处展开、何处引用，不重复推导）。
5. [ ] 每个代码示例注明对应推导步（不丢裸公式编号）。
6. [ ] 自检题全部可由正文内容独立推出答案。

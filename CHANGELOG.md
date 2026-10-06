# 变更日志

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增（论文级代码映射全覆盖，2026-10-07）

- **全库 52 篇论文建立「论文 × 代码状态」映射**：各主题 `精读/README.md` 新增映射表，
  每篇标注 ✅实现（链接模块）/ 🔶教学代理（说明简化项）/ ❌未实现（一句话原因）——
  现有 27 篇有代码对应，25 篇（VLA 基础模型、需 GPU 训练的神经隐式方法、完整系统）
  给出明确不做的原因
- **projects/3d_reconstruction/ 新增三模块**（39→55 项测试）：poisson.py（FFT 均匀网格
  泊松重建，Kazhdan 2006 教学代理）、plane_sweep.py（平面扫描立体 + NCC 置信度，
  COLMAP-MVS/MVSNet 共享内核）、splatting.py（3DGS 前向泼溅渲染：EWA 投影 + 深度排序
  + alpha 合成，优化侧不做）；scene.py 增程序化纹理与彩色渲染
- **projects/slam/ 新增 ptam/**（80→87 项测试）：PTAM 教学代理——跟踪/建图两任务分离、
  关键帧判据、motion-only G-N 跟踪、X/Y/Z 三分局部 BA（复用 core/epipolar）
- **projects/robotwin/ 新增两模块**（35→52 项测试）：diffusion_lite.py（2D 轨迹 DDPM
  生成式策略，T=50 手写 MLP 反传）、grasp_2d.py（2D 平行夹爪力闭合 + Ferrari-Canny L1
  质量，双口径交叉验证）
- **projects/control_planning/ 新增 mpnet_lite.py**（46→53 项测试）：学习式采样偏置
  规划（SDF 编码 + 手写 MLP + 偏置 RRT，迭代数降至均匀版 ~0.79）；并勘正 MPNet 署名
  为 Qureshi et al., ICRA 2019
- 全部新测试带 `__main__` 子串过滤、双环境全绿；五项目合并收集 **262 passed**

### 新增（三大主题项目实现，2026-10-07）

- **projects/3d_reconstruction/**：3D 重建教学实现（纯 NumPy）——scene.py（SDF 基元 +
  球追踪深度渲染）、tsdf.py（Curless & Levoy 1996 加权积分）、marching.py（Marching
  Tetrahedra 面提取，16 case 表含穷举自检）、metrics.py（Chamfer / 精度完成度 DTU 口径 /
  F-score@τ / 网格面积加权采样）+ demo.py 一键管线（24 视角 → TSDF → 网格 → 评测，
  F@20mm≈0.96）；39 项测试双环境全绿
- **projects/control_planning/**：控制与规划教学实现（纯 NumPy）——rrt.py（RRT/RRT* 重布线）、
  ilqr.py（含与 Riccati 代数解交叉验证）、mpc_cem.py（CEM-MPC，PETS 简化档）、cbf.py
  （QP 安全滤波，闭式投影 + POCS 多约束）、osc_arm.py（2R 臂 FK/雅可比/DLS-IK/阻抗控制）、
  ppo_lite.py（微型 PPO：手写前向反向 + GAE + clip）+ metrics.py（规划成功率/路径长度/
  轨迹代价，§4.7 点名基线补齐）+ demo.py 四段冒烟；46 项测试双环境全绿
- **projects/robotwin/**：RoboTwin 方法论迷你基准（纯 NumPy）——把论文两条主线做成 2D
  双臂玩具世界：dr.py（none/mild/strong 三档 × 五维域随机化采样）、benchmark.py（评测协议：
  每任务成功率 / 宏平均 + 最差任务 / seen-unseen 泛化差距，种子规则写死全网格逐位确定）、
  arm.py/tasks.py（FK/IK + reach/push/pick_place 回合环境）、policies.py（标定型 vs
  自标定型对照，复现"DR 剂量 vs 成功率"曲线）+ metrics.py；35 项测试双环境全绿，
  demo.py 约 2 秒复现教程 §4.2.3/§6.4 论断
- **三项目四件套文档**（README/TUTORIAL/DEBUG/METRICS）按 §4.4–§4.7 交付，测试全部支持
  `python tests/test_X.py <子串>` 单点过滤；DEBUG.md 变量表（24/27/25 条）健康值均为实测
- **VS Code 注册扩容**（§4.8 五处同步）：settings.pytestArgs/extraPaths/PYTHONPATH 覆盖
  五项目；launch.json 21 条（每新项目核心入口/全局测试/单点测试三条）；tasks.json 18 条；
  SETUP.md 同步
- **索引联动**：projects/README、根 README 目录树、三个主题 OVERVIEW 代码状态、
  CONSTRAINTS §4.7 指标基线（3d_reconstruction / control_planning / robotwin 补齐）

### 修复（全库查缺补漏）

- 根 README / CHANGELOG 计数与状态勘误：精读总数 38→**52**（18+13+11+9+1）、
  projects/slam 测试数 95→**80**（95 为双项目之和，nerf 为 15）、SLAM 经典论文 10→**11 篇**
  （补计 Cadena 综述）；`tutorials/3d_reconstruction` 与 `control_planning` 的"写作中"状态更新为十章完结
- 修复 GitHub Actions 自动提交工作流**永不提交**的缺陷（检查步骤未写 `$GITHUB_OUTPUT`，
  下游条件恒假）：改为 bash 直检 `git status --porcelain` 并正确声明步骤输出；
  补 `permissions: contents: write`；移除未使用的 Python / gitpython 依赖与 status.json 中转文件
- 修复 `scripts/auto_commit.sh` 提交信息生成缺陷（`echo | while read` 管道进子 shell 导致
  文件清单丢失、双引号内 `\n` 为字面量）：改用临时文件 + `printf` 逐行拼接，
  产出与 scripts/README.md 文档一致的"新增文件/修改文件"格式
- `.vscode/` 对齐 §4.5/§4.8：launch.json 补齐 **NeRF 全局/单点测试**与 **SLAM 核心入口**
  调试配置（新增 `projects/slam/demo.py` 冒烟演示）；tasks.json 补 NeRF 单点任务并将
  "运行核心测试"升级为全局测试（pytest tests/）；NeRF 冒烟配置改用开箱即得的
  `data/demo_scene`（全量训练仍用 `data/lego`，获取方式见 DEBUG.md）
- 索引与导航补全：projects/README.md 补 **slam 条目与快速开始**；tutorials/README.md 表格
  补精读列；各主题 README 补 OVERVIEW / 精读入口；新建 `tutorials/robotwin/OVERVIEW.md`
  （此前唯一缺主题总览的主题）与 `papers/nerf/README.md`
- 修复目录重构遗留的失效相对路径：`tutorials/nerf/README.md`、`nerf/08`、`nerf/09` 的
  `../paper`、`../code/`；`projects/nerf/README.md` 的 `../tutorial/`；
  `projects/nerf/METRICS.md` 的 `../tutorials/...`（7 处，均应为 `../../` 两级）
- 测试文档与实际输出对齐（实测核验）：slam DEBUG/TUTORIAL 的 `gate` 单点示例 3/3→**1/1**、
  无匹配示例由 `gat`（实匹配 2 个）改为 `zzz`、launch.json 单点示例 `nn`（实无匹配）→`gat`、
  METRICS.md `scale` 示例 2/2→**4/4**、slam README 变量表计数 40→**36 条**
- 纯操作章节（nerf 08/09、robotwin 07/08）开头补注"操作步骤，不含理论"（§3.4 豁免声明）
- environment.yml 补录 NeRF 额外依赖（Pillow / imageio / scipy / tqdm，与
  projects/nerf/requirements.txt 同步），torch 下界对齐为 `>=2.5`
- cron 时区描述更正（README / scripts/README / workflow 注释）：`0 9 * * 1` 为
  **UTC 周一 09:00 = 北京时间周一 17:00**
- 勘正本文件原"误推送的论文提交已强制覆盖移除"的失实表述：论文 PDF 当时以**普通提交**
  从工作树移除（8f1910a，非历史重写），早期提交中的论文文件一直留在 git 历史中。
  **已于 2026-10-06 处置**：`git filter-repo` 重写全部历史剔除 `papers/` 并强制推送，
  远程不再含论文文件；本地 `papers/` 目录与推送前的全量备份（bundle）保留

### 变更（内容规范对齐，2026-10-06）

- **nerf / robotwin 二十章按 §3.4 五步法重构**：共 77 个五步块（数学知识点 = 逐步推导并
  逐项注明依据；概念/设计决策知识点 = 完整推理链）；操作章（nerf 08/09、robotwin 07/08）
  豁免声明核对补全，两主题第 10 章术语表加"参考资料性质"豁免；新增章内公式编号
  nerf (2.1)–(5.6)、robotwin (2.1)(2.2)(4.1)–(4.4)，既有标题、式号回引、链接零破坏
- **slam 小节编号统一为双数字**（01/02/09 章 `1.1→01.1` 等，65 处，含 精读 与
  3d_reconstruction 的跨主题回引同步；公式 tag (1.1) 等保持不变）
- 清理：.gitignore 死规则 `docs/`、settings.json 弃用键
  `python.terminal.activateEnvInCurrentTerminal`；README 补 `papers/` 本地化说明

## [2.0.0] - 2026-09-27

按资源类型重构目录结构（不兼容变更，版本号依 §8.1 升 MAJOR）；新增三大主题教程、
52 篇论文精读、SLAM 教学实现与双项目测评体系。

### 新增

- 全局约束文档 (CONSTRAINTS.md)
- 变更日志 (CHANGELOG.md)
- 文档规范新增 §3.4「论述结构规范（问题导向五步法）」：核心知识点必须按"问题场景 → 解决方法 → 选型理由 → 理论依据 → 完整推导"展开，推导禁止跳步；附 NeRF 体积渲染完整示范与检查清单项
- SLAM 教程架构文档：10 章规划，每章预埋问题场景锚点与推导产出清单，含学习路径依赖图与资源清单
- SLAM 论文库 `papers/slam/`：经典 11 篇（FastSLAM、PTAM、ORB-SLAM 三部曲、LSD-SLAM、DSO、LOAM、IMU 预积分、VINS-Mono、SLAM 权威综述）+ 前沿 7 篇（DROID-SLAM、NeRF-SLAM、GS-SLAM、SplaTAM、MonoGS、MASt3R-SLAM、VGGT-GS SLAM），全部校验 PDF 完整性；分类索引含入选理由与教程章节映射
- **教程三主题十章完结**：SLAM 10 章（滤波/图优化/回环/VIO 预积分全覆盖，五步法）、3D 重建架构 + 10 章（SfM/MVS/TSDF/Poisson/DeepSDF/NeuS/3DGS/前馈重建/机器人选型）、控制与规划新主题架构 + 10 章（RRT*/iLQR/MPC/阻抗控制/CBF/PPO/SAC/学习式前沿/全栈衔接）
- **新增第五主题 control_planning**：本地论文 9 篇（RRT、RRT*、CBF 综述、Crocoddyl、PETS、MPNet、PPO、SAC、Rapid Locomotion，git 排除）+ 教程十章
- **论文库扩容（仅本地，`papers/` 已加入 .gitignore 不上传）**：`papers/3d_reconstruction/` 12 篇 + MapAnything 源码包；`papers/robotwin/` 11 篇；全部 arXiv 论文 LaTeX 源码包 43 套；所有 PDF 逐篇标题验证
- **projects/slam/**：SLAM 论文教学实现——core（李群/相机/GN-LM）+ 9 个论文模块（fastslam/epipolar/bowloop/direct/photoba/loam2d/preint/vins/droidlite）+ metrics.py（Umeyama ATE/旋转误差/尺度比/RPE），80 项测试双环境全绿（另 projects/nerf 15 项）；纯注释改动经 AST 级核验
- **双项目统一测评**：metrics.py + METRICS.md（§4.7：作用/如何计算/健康值范围/如何运行）
- **双项目零基础导读**：TUTORIAL.md（§4.4 七段结构，代码片段逐段实跑）与 DEBUG.md（§4.5：单点/全局测试、重点观察变量表、实战调试案例）
- **全局约束扩展**：§4.3 代码中文注释（变量含义/作用/函数流水线）、§4.4 TUTORIAL.md、§4.5 DEBUG.md + launch.json、§4.6 Anaconda 环境清单、§4.7 测评指标、§4.8 VS Code 统一配置
- **VS Code 统一**：settings.json 写入 conda llm_env 解释器并覆盖全项目；launch.json 调试配置（含单点测试交互过滤）；tasks.json 测试任务；`.vscode/SETUP.md` 配置教程
- **environment.yml**：Anaconda 环境配置清单（llm_env：python 3.10 / torch 2.11 / numpy 2.2 / pytest / pypdf）
- 全部测试文件主入口支持单点过滤（`python tests/test_X.py <子串>`）
- **论文精读系列（52 篇）**：
  - **SLAM 18 篇**：`tutorials/slam/精读/`——经典 11 + 前沿 7，逐篇含关键公式推导（实读 PDF 核对式号；含 DSO 式(15)/DROID 式(7)(8)(14)/VINS 式(5) 等原论文排印勘误注）
  - **3D 重建 13 篇**：`tutorials/3d_reconstruction/精读/`——COLMAP×2/MVSNet/Poisson/ONet/DeepSDF/NeuS/Neuralangelo/Instant-NGP/3DGS/DUSt3R/MASt3R/MapAnything（MVSNet/NeuS 与 LaTeX 源码双重核对）
  - **RoboTwin 11 篇**：`tutorials/robotwin/精读/`——RoboTwin 双基准 + RT-1/Diffusion Policy/ACT/RT-2/OXE/OpenVLA/π0/RDT/DexGraspNet
  - **控制规划 9 篇**：`tutorials/control_planning/精读/`——RRT/RRT*（Theorem 15–71 全核）/CBF 综述/PETS/Crocoddyl/MPNet/PPO/SAC/RapidLocomotion
  - **NeRF 1 篇**：`tutorials/nerf/精读/`——基于 LaTeX 源码逐式核对（论文仅 6 个编号式）+ 论文↔教程↔代码三方映射表
  - 各主题 `精读/README.md` 总索引（按管线/谱系分组 + 阅读顺序 + 跨主题衔接）

### 变更

- `papers/` 加入 `.gitignore`：论文仅本地研读，不上传 GitHub（论文文件随后以普通提交移出工作树；早期提交中的论文文件仍保留在 git 历史中，彻底清除需重写历史，见 Unreleased 勘误）
- **项目重构**：按资源类型重新划分目录——`papers/`（论文库）、`tutorials/`（教程文档）、`projects/`（代码项目），原 `nerf/`、`robotwin/`、`slam/`、`3d_reconstruction/` 按主题拆分归入三类
- 同步更新根 README、各索引 README、`.vscode/` 配置、`.gitignore` 及 CONSTRAINTS.md
- 全部项目代码注释中文化（模块流水线/变量含义/推导依据，AST 级核验零逻辑改动）

### 修复

- 6 个张冠李戴的 arXiv PDF（NeuS、LSD-SLAM、MVSNet、MASt3R、DexGraspNet、RoboTwin 1.0），逐篇以 PDF 首页标题核对并改为正确编号
- SLAM 教程第 09 章 BoW 评分公式端点（s∈[1/2,1]，与 DBoW2 及代码实现一致）
- test_droidlite 的 numpy 2.x 容差（Schur 交叉验证 1e-5，双环境通过）
- photoba LM λ 塌缩 bug（Nielsen 增益比 + 逆深度物理箱）
- projects/nerf 内联 PSNR/SSIM 抽取至 nerf/metrics.py（语义逐位一致）

## [1.0.0] - 2026-09-25

### 新增

- NeRF 模块基础教程
- RoboTwin2.0 模块教程
- SLAM 模块教程
- 3D 重建模块教程
- 项目全局文档结构

### 文档

- 各模块 README
- 各模块教程文档
- 全局约束规范

---

## 版本说明

### [Unreleased] - 开发中
当前开发版本，包含未发布的新功能和变更。

### [2.0.0] - 目录重构 + 三主题教程 + 52 篇精读
按 papers/tutorials/projects 三类重构目录（不兼容）；新增 3D 重建、控制与规划两主题教程、
SLAM 教学实现（80 项测试）、双项目 TUTORIAL/DEBUG/METRICS 体系与 VS Code 统一配置。

### [1.0.0] - 初始版本
项目初始发布版本，包含所有基础模块和文档。

---

## 变更类型

- **新增** (Added): 新功能
- **变更** (Changed): 现有功能的变更
- **弃用** (Deprecated): 即将移除的功能
- **移除** (Removed): 已移除的功能
- **修复** (Fixed): Bug 修复
- **安全** (Security): 安全性相关修复

## 变更追踪

- **问题**: [链接到 GitHub Issue]
- **PR**: [链接到 Pull Request]

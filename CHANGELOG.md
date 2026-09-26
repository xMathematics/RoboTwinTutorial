# 变更日志

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增
- 全局约束文档 (CONSTRAINTS.md)
- 变更日志 (CHANGELOG.md)
- 文档规范新增 §3.4「论述结构规范（问题导向五步法）」：核心知识点必须按"问题场景 → 解决方法 → 选型理由 → 理论依据 → 完整推导"展开，推导禁止跳步；附 NeRF 体积渲染完整示范与检查清单项
- SLAM 教程架构文档：10 章规划，每章预埋问题场景锚点与推导产出清单，含学习路径依赖图与资源清单
- SLAM 论文库 `papers/slam/`：经典 10 篇（FastSLAM、PTAM、ORB-SLAM 三部曲、LSD-SLAM、DSO、LOAM、IMU 预积分、VINS-Mono）+ 前沿 7 篇（DROID-SLAM、NeRF-SLAM、GS-SLAM、SplaTAM、MonoGS、MASt3R-SLAM、VGGT-GS SLAM），全部校验 PDF 完整性；分类索引含入选理由与教程章节映射
- **教程三主题十章完结**：SLAM 10 章（滤波/图优化/回环/VIO 预积分全覆盖，五步法）、3D 重建架构 + 10 章（SfM/MVS/TSDF/Poisson/DeepSDF/NeuS/3DGS/前馈重建/机器人选型）、控制与规划新主题架构 + 10 章（RRT*/iLQR/MPC/阻抗控制/CBF/PPO/SAC/学习式前沿/全栈衔接）
- **新增第五主题 control_planning**：本地论文 9 篇（RRT、RRT*、CBF 综述、Crocoddyl、PETS、MPNet、PPO、SAC、Rapid Locomotion，git 排除）+ 教程十章
- **论文库扩容（仅本地，`papers/` 已加入 .gitignore 不上传）**：`papers/3d_reconstruction/` 12 篇 + MapAnything 源码包；`papers/robotwin/` 11 篇；全部 arXiv 论文 LaTeX 源码包 43 套；所有 PDF 逐篇标题验证
- **projects/slam/**：SLAM 论文教学实现——core（李群/相机/GN-LM）+ 9 个论文模块（fastslam/epipolar/bowloop/direct/photoba/loam2d/preint/vins/droidlite）+ metrics.py（Umeyama ATE/旋转误差/尺度比/RPE），95 项测试双环境全绿；纯注释改动经 AST 级核验
- **双项目统一测评**：metrics.py + METRICS.md（§4.7：作用/如何计算/健康值范围/如何运行）
- **双项目零基础导读**：TUTORIAL.md（§4.4 七段结构，代码片段逐段实跑）与 DEBUG.md（§4.5：单点/全局测试、重点观察变量表、实战调试案例）
- **全局约束扩展**：§4.3 代码中文注释（变量含义/作用/函数流水线）、§4.4 TUTORIAL.md、§4.5 DEBUG.md + launch.json、§4.6 Anaconda 环境清单、§4.7 测评指标、§4.8 VS Code 统一配置
- **VS Code 统一**：settings.json 写入 conda llm_env 解释器并覆盖全项目；launch.json 9 条调试配置（含单点测试交互过滤）；tasks.json 7 条任务；`.vscode/SETUP.md` 配置教程
- **environment.yml**：Anaconda 环境配置清单（llm_env：python 3.10 / torch 2.11 / numpy 2.2 / pytest / pypdf）
- 全部测试文件主入口支持单点过滤（`python tests/test_X.py <子串>`）

### 变更
- `papers/` 加入 `.gitignore`：论文仅本地研读，不上传 GitHub（误推送的一个论文提交已强制覆盖移除）

### 修复
- 6 个张冠李戴的 arXiv PDF（NeuS、LSD-SLAM、MVSNet、MASt3R、DexGraspNet、RoboTwin 1.0），逐篇以 PDF 首页标题核对并改为正确编号
- SLAM 教程第 09 章 BoW 评分公式端点（s∈[1/2,1]，与 DBoW2 及代码实现一致）
- test_droidlite 的 numpy 2.x 容差（Schur 交叉验证 1e-5，双环境通过）
- photoba LM λ 塌缩 bug（Nielsen 增益比 + 逆深度物理箱）
- projects/nerf 内联 PSNR/SSIM 抽取至 nerf/metrics.py（语义逐位一致）

### 变更
- **项目重构**：按资源类型重新划分目录——`papers/`（论文库）、`tutorials/`（教程文档）、`projects/`（代码项目），原 `nerf/`、`robotwin/`、`slam/`、`3d_reconstruction/` 按主题拆分归入三类
- 同步更新根 README、各索引 README、`.vscode/` 配置、`.gitignore` 及 CONSTRAINTS.md
- 全部项目代码注释中文化（模块流水线/变量含义/推导依据，AST 级核验零逻辑改动）

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
